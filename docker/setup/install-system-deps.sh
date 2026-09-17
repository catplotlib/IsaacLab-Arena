#!/bin/bash
set -euo pipefail

# Common simulation/runtime dependencies; preserve the existing feature set.
if [ -e /usr/share/vulkan ] && [ -e /etc/vulkan ]; then
    mv /usr/share/vulkan /usr/share/vulkan_hidden
fi
apt-get update
apt-get install -y git git-lfs cmake ffmpeg sudo jq python3-pip
apt-get install -y --no-install-recommends pqiv
