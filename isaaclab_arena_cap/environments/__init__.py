# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Importing this package fires each environment module's ``@register_environment``."""

import importlib
import pkgutil

for _importer, _modname, _ispkg in pkgutil.iter_modules(__path__):
    if not _ispkg:
        importlib.import_module(f"{__name__}.{_modname}")
