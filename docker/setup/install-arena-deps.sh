#!/bin/bash
set -euo pipefail

/isaac-sim/python.sh /tmp/export_requirements.py /tmp/arena-pyproject.toml runtime > /tmp/arena-requirements.txt
/isaac-sim/python.sh -m pip install -r /tmp/arena-requirements.txt
# simready-search declares an AWS stack conflicting with Isaac Sim's bundle.
/isaac-sim/python.sh -m pip install --force-reinstall --no-deps \
    boto3==1.40.61 botocore==1.40.61 s3transfer==0.14.0 requests==2.32.3
rm /tmp/arena-requirements.txt

# User-facing model/dataset tooling has its own interpreter and dependencies.
apt-get update
apt-get install -y pipx
PIPX_HOME=/opt/pipx PIPX_BIN_DIR=/usr/local/bin pipx install "huggingface-hub[cli]"
