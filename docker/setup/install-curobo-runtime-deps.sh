#!/bin/bash
set -euo pipefail

# Resolve only the wheel's declared runtime dependencies. Protect the existing
# simulation ABI and compatibility repairs while installing missing packages.
/isaac-sim/python.sh - <<'PYTHON' > /tmp/curobo-constraints.txt
from importlib.metadata import PackageNotFoundError, version
for name in ('torch', 'torchvision', 'torchaudio', 'numpy', 'warp-lang',
             'boto3', 'botocore', 's3transfer', 'requests'):
    try:
        installed_version = version(name)
    except PackageNotFoundError:
        continue
    print(f'{name}=={installed_version}')
PYTHON
/isaac-sim/python.sh -m pip install -r /wheels/runtime-requirements.txt -c /tmp/curobo-constraints.txt
rm /tmp/curobo-constraints.txt
# The precompiled extensions use the inherited Torch/CUDA libraries. The local
# cold-cache IK check verifies that no runtime nvcc/toolkit installation is needed.
