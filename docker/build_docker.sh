#!/bin/bash
# Build an Arena image without starting or attaching to a container.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TARGET=dev
IMAGE=""
LOG_FILE=""
BUILD_ARGS=()

usage() {
    echo "Usage: $0 [-t runtime|dev|runtime-curobo|dev-curobo] [-n image:tag] [-R] [-l log-file]"
    echo "  -t  Docker target (default: dev)"
    echo "  -n  Output image reference (default: isaaclab_arena:<target compatibility tag>)"
    echo "  -R  Build without cache; base image metadata is always refreshed"
    echo "  -l  Save plain BuildKit output and elapsed time to a log file"
}

while getopts ':t:n:l:Rh' option; do
    case "$option" in
        t) TARGET=$OPTARG ;;
        n) IMAGE=$OPTARG ;;
        l) LOG_FILE=$OPTARG ;;
        R) BUILD_ARGS+=(--no-cache) ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
if [ "$#" -ne 0 ]; then
    usage >&2
    exit 2
fi
case "$TARGET" in
    dev) DEFAULT_TAG=latest ;;
    dev-curobo) DEFAULT_TAG=curobo ;;
    runtime|runtime-curobo) DEFAULT_TAG=$TARGET ;;
    *) echo "Unsupported Arena target: $TARGET" >&2; exit 2 ;;
esac
IMAGE=${IMAGE:-isaaclab_arena:$DEFAULT_TAG}

build() {
    local started=$SECONDS status=0
    docker build --pull --progress=plain \
        --target "$TARGET" \
        --build-arg WORKDIR=/workspaces/isaaclab_arena \
        "${BUILD_ARGS[@]}" \
        --tag "$IMAGE" --file "$SCRIPT_DIR/Dockerfile.isaaclab_arena" "$SCRIPT_DIR/.." || status=$?
    echo "Arena build target=$TARGET image=$IMAGE exit=$status elapsed=$((SECONDS - started))s"
    return "$status"
}

if [ -n "$LOG_FILE" ]; then
    mkdir -p -- "$(dirname -- "$LOG_FILE")"
    build 2>&1 | tee -- "$LOG_FILE"
else
    build
fi
