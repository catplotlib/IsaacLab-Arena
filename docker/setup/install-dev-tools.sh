#!/bin/bash
set -euo pipefail

/isaac-sim/python.sh /tmp/export_requirements.py /tmp/arena-pyproject.toml dev > /tmp/arena-dev-requirements.txt
/isaac-sim/python.sh -m pip install -r /tmp/arena-dev-requirements.txt
rm /tmp/arena-dev-requirements.txt
PIPX_HOME=/opt/pipx PIPX_BIN_DIR=/usr/local/bin pipx install pre-commit

if ! command -v wget >/dev/null; then
    apt-get update
    apt-get install -y wget
fi
mkdir -p -m 755 /etc/apt/keyrings
key_file=$(mktemp)
wget -nv -O "$key_file" https://cli.github.com/packages/githubcli-archive-keyring.gpg
cat "$key_file" > /etc/apt/keyrings/githubcli-archive-keyring.gpg
rm "$key_file"
chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
mkdir -p -m 755 /etc/apt/sources.list.d
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list
apt-get update
apt-get install -y gh

cat >> /etc/bash.bashrc <<'SHELL'
alias debugpy='python -Xfrozen_modules=off -m debugpy --listen localhost:5678 --wait-for-client'
SHELL
cp /etc/bash.bashrc /root/.bashrc
