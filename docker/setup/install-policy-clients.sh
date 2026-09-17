#!/bin/bash
# Install policy clients and deps

set -euo pipefail

/isaac-sim/python.sh -m pip install msgpack==1.1.0 msgpack-numpy==0.4.8 pyzmq==27.0.1
/isaac-sim/python.sh -m pip install --no-deps --ignore-requires-python -e "${WORKDIR}/submodules/Isaac-GR00T/"
/isaac-sim/python.sh -m pip install av
OPENPI_COMMIT=$(tr -d '[:space:]' < /tmp/openpi_commit)
/isaac-sim/python.sh -m pip install --no-cache-dir \
    "openpi-client @ git+https://github.com/Physical-Intelligence/openpi@${OPENPI_COMMIT}#subdirectory=packages/openpi-client"
