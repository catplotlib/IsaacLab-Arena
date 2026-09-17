#!/bin/bash
# Seed CI's registry caches from a clean checkout, without changing the worktree.
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <registry/image> [git-ref=HEAD] [builder=arena-ci-cache]" >&2
    echo "Replaces <registry/image>:buildcache-main-{dev,dev-curobo}; requires docker login." >&2
    exit 2
fi
IMAGE=$1
REVISION=${2:-HEAD}
BUILDER=${3:-arena-ci-cache}
ROOT=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
COMMIT=$(git -C "$ROOT" rev-parse "${REVISION}^{commit}")
LOG_DIR=$(mktemp -d /tmp/arena-ci-cache.XXXXXX)
CONTEXT="$LOG_DIR/context"
trap 'rm -rf -- "$CONTEXT"' EXIT

# BuildKit hashes file modes for COPY and RUN bind mounts. A developer's umask
# (often 002) must not produce 664/775 files when CI checks out 644/755 files.
# Only committed files are used. Match CI: submodule LFS stays as pointers;
# root LFS objects are fetched explicitly after checkout.
umask 022
export GIT_LFS_SKIP_SMUDGE=1
git clone --shared --no-checkout "$ROOT" "$CONTEXT"
git -C "$CONTEXT" remote set-url origin "$(git -C "$ROOT" remote get-url origin)"
git -C "$CONTEXT" checkout --detach "$COMMIT"
git -C "$CONTEXT" -c url.https://github.com/.insteadOf=git@github.com: submodule update --init
git -C "$CONTEXT" lfs pull

if ! docker buildx inspect "$BUILDER" >/dev/null 2>&1; then
    docker buildx create --name "$BUILDER" --driver docker-container
fi
# Retain this dedicated builder so subsequent seeds reuse its local cache.
docker buildx inspect "$BUILDER" --bootstrap
printf 'Seeding commit %s; logs: %s\n' "$COMMIT" "$LOG_DIR"
stat -c '%a %n' "$CONTEXT/docker/setup/install-system-deps.sh" "$CONTEXT/pyproject.toml"

# Build the superset first, then export dev from the same builder.
for TARGET in dev-curobo dev; do
    CACHE="${IMAGE}:buildcache-main-${TARGET}"
    started=$SECONDS
    docker buildx build --builder "$BUILDER" --progress=plain --pull \
        --platform linux/amd64 --target "$TARGET" \
        --build-arg WORKDIR=/workspaces/isaaclab_arena \
        --file "$CONTEXT/docker/Dockerfile.isaaclab_arena" \
        --output type=cacheonly \
        --cache-from "type=registry,ref=$CACHE" \
        --cache-to "type=registry,ref=$CACHE,mode=max,oci-mediatypes=true,image-manifest=true" \
        "$CONTEXT" 2>&1 | tee "$LOG_DIR/$TARGET.log"
    echo "Seed target=$TARGET elapsed=$((SECONDS - started))s cache=$CACHE" | tee -a "$LOG_DIR/timings.txt"
    docker buildx imagetools inspect --raw "$CACHE" > "$LOG_DIR/$TARGET-manifest.json"
done
