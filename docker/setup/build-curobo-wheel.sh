#!/bin/bash
# Install cuda and build curobo python wheel
set -euo pipefail

bash /tmp/install_cuda.sh
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
/isaac-sim/python.sh -m pip install setuptools_scm wheel ninja
mkdir -p /wheels
/isaac-sim/python.sh -m pip wheel --no-deps --no-build-isolation --wheel-dir /wheels \
    "nvidia-curobo @ git+https://github.com/NVlabs/curobo.git@${CUROBO_COMMIT}"
/isaac-sim/python.sh - <<'PYTHON'
import email
import hashlib
import json
import os
from pathlib import Path
import zipfile

import torch

wheels = list(Path('/wheels').glob('*.whl'))
assert len(wheels) == 1, wheels
wheel = wheels[0]
with zipfile.ZipFile(wheel) as archive:
    metadata_path = next(name for name in archive.namelist() if name.endswith('.dist-info/METADATA'))
    metadata = email.message_from_bytes(archive.read(metadata_path))
    requirements = metadata.get_all('Requires-Dist', [])
Path('/wheels/runtime-requirements.txt').write_text('\n'.join(requirements) + '\n')
Path('/wheels/build.json').write_text(json.dumps({
    'commit': os.environ['CUROBO_COMMIT'],
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'architectures': os.environ['TORCH_CUDA_ARCH_LIST'],
    'wheel': wheel.name,
    'sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
}, indent=2) + '\n')
PYTHON
