#!/bin/bash
set -euo pipefail

ln -s /isaac-sim/ "${ISAACLAB_PATH}/_isaac_sim"
# Preserve Kit's existing writable installation and host-user access.
chmod 777 -R /isaac-sim/kit/
chmod a+x /isaac-sim
"${ISAACLAB_PATH}/isaaclab.sh" -i
