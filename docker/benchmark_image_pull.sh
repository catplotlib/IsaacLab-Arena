#!/bin/bash
# Compare registry pulls without reusing or clearing the host's Docker cache.
set -euo pipefail

if [ "$#" -ne 3 ] || [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <baseline-image> <candidate-image> <output-directory>" >&2
    exit 2
fi
IMAGES=("$1" "$2")
OUTPUT=$3
mkdir -p "$OUTPUT"
OUTPUT=$(realpath "$OUTPUT")
DAEMON=
LABEL=
cleanup() {
    if [ -n "$DAEMON" ]; then
        docker logs "$DAEMON" > "$OUTPUT/$LABEL-daemon.log" 2>&1 || true
        # Only this probe's container and anonymous volumes are removed.
        docker rm --force --volumes "$DAEMON" >/dev/null || true
        DAEMON=
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Match the host's Engine version and image store for comparable pull/unpack behavior.
ENGINE_VERSION=$(docker version --format '{{.Server.Version}}')
DRIVER=$(docker info --format '{{.Driver}}')
case "$DRIVER" in
    overlay2) CONTAINERD=false ;;
    overlayfs) CONTAINERD=true ;;
    *) echo "Unsupported benchmark storage driver: $DRIVER" >&2; exit 2 ;;
esac
DIND_IMAGE="docker:${ENGINE_VERSION}-dind"
docker version > "$OUTPUT/host-version.txt"
docker info > "$OUTPUT/host-info.txt"
docker pull "$DIND_IMAGE"

printf '| Image | Digest | Cold pull + unpack | Container create/start | Total |\n' > "$OUTPUT/summary.md"
printf '| --- | --- | ---: | ---: | ---: |\n' >> "$OUTPUT/summary.md"
for INDEX in 0 1; do
    LABEL=baseline
    if [ "$INDEX" -eq 1 ]; then LABEL=candidate; fi
    IMAGE=${IMAGES[$INDEX]}
    # Freeze mutable tags before measuring; record layer digests and sizes.
    DIGEST=$(docker buildx imagetools inspect "$IMAGE" --format '{{.Manifest.Digest}}')
    REPOSITORY=${IMAGE%@*}
    # Strip a tag only from the final path component, preserving registry ports.
    LAST_COMPONENT=${REPOSITORY##*/}
    if [[ "$LAST_COMPONENT" == *:* ]]; then REPOSITORY=${REPOSITORY%:*}; fi
    PINNED="$REPOSITORY@$DIGEST"
    docker buildx imagetools inspect --raw "$PINNED" > "$OUTPUT/$LABEL-manifest.json"
    echo "$PINNED" > "$OUTPUT/$LABEL-image.txt"

    DAEMON="arena-pull-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}-$LABEL-$$"
    # No host socket/data mounts or exposed TCP listener. Each daemon gets empty storage.
    docker run --detach --privileged --name "$DAEMON" \
        --volume /var/lib/docker --volume /var/lib/containerd \
        --env DOCKER_TLS_CERTDIR= "$DIND_IMAGE" \
        dockerd --host=unix:///var/run/docker.sock \
        --feature "containerd-snapshotter=$CONTAINERD" --storage-driver "$DRIVER" >/dev/null
    for ((ATTEMPT=0; ATTEMPT<60; ATTEMPT++)); do
        if docker exec "$DAEMON" docker info >/dev/null 2>&1; then break; fi
        sleep 1
    done
    docker exec "$DAEMON" docker info > "$OUTPUT/$LABEL-info-before.txt"
    test "$(docker exec "$DAEMON" docker image ls --quiet | wc -l)" -eq 0
    test "$(docker exec "$DAEMON" docker ps --all --quiet | wc -l)" -eq 0
    docker exec "$DAEMON" docker system df | tee "$OUTPUT/$LABEL-storage-before.txt"
    if [ -n "${NGC_API_KEY:-}" ]; then
        printf '%s' "$NGC_API_KEY" | docker exec --interactive "$DAEMON" \
            docker login nvcr.io --username '$oauthtoken' --password-stdin >/dev/null
    fi

    echo "Measuring $LABEL: $PINNED with empty storage"
    START=$SECONDS
    docker exec "$DAEMON" docker pull --platform linux/amd64 "$PINNED" 2>&1 | tee "$OUTPUT/$LABEL-pull.log"
    PULLED=$SECONDS
    # Measure basic container preparation separately; this does not start Isaac Sim.
    docker exec "$DAEMON" docker run --rm --network none --entrypoint /bin/sh "$PINNED" -c true
    READY=$SECONDS
    printf '| %s | `%s` | %ss | %ss | %ss |\n' \
        "$LABEL" "$DIGEST" "$((PULLED-START))" "$((READY-PULLED))" "$((READY-START))" \
        | tee -a "$OUTPUT/summary.md"
    cleanup
done
printf '\nBoth pulls used fresh Docker storage. Daemon setup and registry login are excluded. Container startup runs only /bin/sh, without Isaac Sim or GPU initialization.\n' >> "$OUTPUT/summary.md"
cat "$OUTPUT/summary.md"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then cat "$OUTPUT/summary.md" >> "$GITHUB_STEP_SUMMARY"; fi
