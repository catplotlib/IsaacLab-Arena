#!/bin/bash
set -euo pipefail

cat >> /etc/bash.bashrc <<'SHELL'
alias python='/isaac-sim/python.sh'
alias pip3='/isaac-sim/python.sh -m pip'
alias pytest='/isaac-sim/python.sh -m pytest'
PS1='[IsaacLab Arena] \[\e[0;32m\]~\u \[\e[0;34m\]\w\[\e[0m\] \$ '
alias ll='ls -alF --color=auto'
alias ..='cd ..'
SHELL
cp /etc/bash.bashrc /root/.bashrc
