#!/bin/bash
# Install isaac lab

set -euo pipefail

ln -s /isaac-sim/ "${ISAACLAB_PATH}/_isaac_sim"
"${ISAACLAB_PATH}/isaaclab.sh" -i
