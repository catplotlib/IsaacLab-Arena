#!/bin/bash
# Compare a cold registry-cache build with pulling the published image.
set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <baseline-image> <cache-from-spec> <source-directory> <output-directory>" >&2
    exit 2
fi
BASELINE=$1
CACHE_FROM=$2
SOURCE=$(realpath "$3")
OUTPUT=$4
TARGET=${BENCHMARK_TARGET:-dev}
mkdir -p "$OUTPUT"
OUTPUT=$(realpath "$OUTPUT")
test -f "$SOURCE/docker/Dockerfile.isaaclab_arena"
DAEMON=
LABEL=
cleanup() {
    if [ -n "$DAEMON" ]; then
        docker logs "$DAEMON" > "$OUTPUT/$LABEL-daemon.log" 2>&1 || true
        docker rm --force --volumes "$DAEMON" >/dev/null || true
        DAEMON=
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ENGINE_VERSION=$(docker version --format '{{.Server.Version}}')
DIND_IMAGE="docker:${ENGINE_VERSION}-dind"
docker version > "$OUTPUT/host-version.txt"
docker info > "$OUTPUT/host-info.txt"
docker pull "$DIND_IMAGE"
printf 'Baseline: %s\nCache: %s\nTarget: %s\n' "$BASELINE" "$CACHE_FROM" "$TARGET" > "$OUTPUT/inputs.txt"
printf '| Path | Image preparation | Container create/start | Total |\n| --- | ---: | ---: | ---: |\n' > "$OUTPUT/summary.md"

for LABEL in pull cache-build; do
    DAEMON="arena-cache-compare-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}-$LABEL-$$"
    # Independent anonymous volumes; no host socket or data mounted in either daemon.
    # CI uses the containerd image store, required for registry cache in the default builder.
    docker run --detach --privileged --name "$DAEMON" \
        --volume /var/lib/docker --volume /var/lib/containerd \
        --env DOCKER_TLS_CERTDIR= "$DIND_IMAGE" \
        dockerd --host=unix:///var/run/docker.sock \
        --feature containerd-snapshotter=true --storage-driver overlayfs >/dev/null
    for ((ATTEMPT=0; ATTEMPT<60; ATTEMPT++)); do
        if docker exec "$DAEMON" docker info >/dev/null 2>&1; then break; fi
        sleep 1
    done
    docker exec "$DAEMON" docker info > "$OUTPUT/$LABEL-info-before.txt"
    test "$(docker exec "$DAEMON" docker image ls --quiet | wc -l)" -eq 0
    test "$(docker exec "$DAEMON" docker ps --all --quiet | wc -l)" -eq 0
    docker exec "$DAEMON" docker system df | tee "$OUTPUT/$LABEL-storage-before.txt"
    docker exec "$DAEMON" docker buildx du --builder default | tee "$OUTPUT/$LABEL-build-cache-before.txt"
    if [ -n "${NGC_API_KEY:-}" ]; then
        printf '%s' "$NGC_API_KEY" | docker exec --interactive "$DAEMON" \
            docker login nvcr.io --username '$oauthtoken' --password-stdin >/dev/null
    fi
    IMAGE=$BASELINE
    if [ "$LABEL" = cache-build ]; then
        IMAGE=arena-cache-benchmark:result
        # Harness staging is excluded; BuildKit's context transfer remains timed.
        docker cp "$SOURCE/." "$DAEMON:/context"
        docker exec "$DAEMON" docker buildx inspect default > "$OUTPUT/builder.txt"
    fi

    echo "Measuring $LABEL with empty Docker storage"
    START=$SECONDS
    if [ "$LABEL" = pull ]; then
        docker exec "$DAEMON" docker pull --platform linux/amd64 "$IMAGE" 2>&1 | tee "$OUTPUT/pull.log"
    else
        docker exec "$DAEMON" docker buildx build --builder default --progress=rawjson --pull --load \
            --platform linux/amd64 --target "$TARGET" \
            --build-arg WORKDIR=/workspaces/isaaclab_arena \
            --cache-from "$CACHE_FROM" --tag "$IMAGE" \
            --file /context/docker/Dockerfile.isaaclab_arena /context 2>&1 | tee "$OUTPUT/build-cache.jsonl"
    fi
    PREPARED=$SECONDS
    docker exec "$DAEMON" docker create --name ready --network none --entrypoint /bin/sh "$IMAGE" -c 'exec sleep infinity'
    docker exec "$DAEMON" docker start ready
    docker exec "$DAEMON" docker exec ready /bin/sh -c true
    READY=$SECONDS
    printf '| %s | %ss | %ss | %ss |\n' "$LABEL" "$((PREPARED-START))" "$((READY-PREPARED))" "$((READY-START))" \
        | tee -a "$OUTPUT/summary.md"
    docker exec "$DAEMON" docker image inspect "$IMAGE" > "$OUTPUT/$LABEL-image.json"
    if [ "$LABEL" = cache-build ]; then
        python3 "$(dirname -- "${BASH_SOURCE[0]}")/report_build_cache.py" < "$OUTPUT/build-cache.jsonl" \
            | tee "$OUTPUT/cache-summary.md"
    fi
    cleanup
done
printf '\nBoth paths use separate empty Docker stores. Preparation includes network transfer, unpacking, and a locally runnable image. No image is pushed. Daemon setup, login, checkout, harness staging, and cleanup are excluded. Container readiness uses /bin/sh, without Isaac Sim or GPU initialization.\n' >> "$OUTPUT/summary.md"
cat "$OUTPUT/summary.md"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    cat "$OUTPUT/inputs.txt" "$OUTPUT/summary.md" "$OUTPUT/cache-summary.md" >> "$GITHUB_STEP_SUMMARY"
fi
