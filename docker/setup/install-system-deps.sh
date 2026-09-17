#!/bin/bash
# Common simulation/runtime dependencies

set -euo pipefail

# Hide conflicting Vulkan files, if needed.
if [ -e /usr/share/vulkan ] && [ -e /etc/vulkan ]; then
    mv /usr/share/vulkan /usr/share/vulkan_hidden
fi

# Preserve Kit's existing writable installation and host-user access.
chmod 777 -R /isaac-sim/kit/
chmod a+x /isaac-sim

apt-get update
apt-get install -y git git-lfs cmake ffmpeg sudo jq python3-pip
apt-get install -y --no-install-recommends pqiv
