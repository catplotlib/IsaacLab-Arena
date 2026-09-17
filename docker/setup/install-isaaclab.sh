#!/bin/bash
# Install Isaac Lab and its dependencies.

set -euo pipefail

ln -s /isaac-sim/ "${ISAACLAB_PATH}/_isaac_sim"
"${ISAACLAB_PATH}/isaaclab.sh" -i
