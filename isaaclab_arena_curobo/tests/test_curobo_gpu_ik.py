# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise packaged cuRobo extensions and robot assets on a real GPU."""

import importlib

import pytest

pytestmark = pytest.mark.curobo_deps


def test_packaged_curobo_solves_gpu_ik():
    """Solve a reachable pose using the wheel's robot data and compiled extensions."""
    import torch

    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    assert torch.cuda.is_available(), "cuRobo image validation requires a CUDA GPU"
    # Import the wheel's binaries directly: a successful JIT fallback in the
    # solver wrappers must not conceal a missing compiled extension.
    for name in ("lbfgs_step_cu", "kinematics_fused_cu", "line_search_cu", "tensor_step_cu", "geom_cu"):
        importlib.import_module(f"curobo.curobolib.{name}")
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    solver = IKSolver(
        IKSolverConfig.load_from_robot_config(
            robot_cfg,
            None,
            tensor_args=tensor_args,
            num_seeds=12,
            position_threshold=0.005,
            rotation_threshold=0.05,
            self_collision_check=False,
            self_collision_opt=False,
            use_cuda_graph=False,
        )
    )
    state = solver.fk(solver.get_retract_config().reshape(1, -1))
    result = solver.solve_single(Pose(state.ee_position, state.ee_quaternion))
    torch.cuda.synchronize()
    assert bool(result.success.any()), "cuRobo failed to recover a reachable end-effector pose"
    assert torch.isfinite(result.solution).all(), "cuRobo returned non-finite joint positions"
