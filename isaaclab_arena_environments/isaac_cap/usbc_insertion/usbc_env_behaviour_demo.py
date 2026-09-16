# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Teleport or keyboard-control a USB-C connector, then verify task success and reset."""

from __future__ import annotations

import argparse
import math

from isaaclab_arena_environments.isaac_cap.tools import EnvBehaviourDemo

_MATING_ROTATIONS_XYZW = {
    "easy": (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0),
    "medium": (1.0, 0.0, 0.0, 0.0),
}
"""Plug-to-receiver rotations aligning both the insertion and wide cross-section axes."""


def _build_demo_environment(variant: str):
    from isaaclab_arena_environments.isaac_cap.usbc_insertion.environment import (
        UsbcInsertionEasyEnvironment,
        UsbcInsertionEasyEnvironmentCfg,
        UsbcInsertionMediumEnvironment,
        UsbcInsertionMediumEnvironmentCfg,
    )

    assert variant in ("easy", "medium"), f"Unsupported USB-C variant: {variant!r}."
    if variant == "easy":
        return UsbcInsertionEasyEnvironment().build(UsbcInsertionEasyEnvironmentCfg())
    return UsbcInsertionMediumEnvironment().build(UsbcInsertionMediumEnvironmentCfg())


def _plug_pose_at_depth(T_W_R, q_R_P, mating_params, depth: float):
    """Return the plug pose at a specified insertion depth, centered on the receiver.

    Args:
        T_W_R: Batched receiver-to-world poses in XYZ/XYZW order.
        q_R_P: Batched plug-to-receiver mating orientations in XYZW order.
        mating_params: Task predicate parameters defining the tip, mouth, and axis.
        depth: Signed tip depth along the receiver's inward axis, in meters.

    Returns:
        Batched plug-to-world poses satisfying the requested depth and zero lateral error.
    """
    import torch

    from isaaclab.utils.math import quat_apply, quat_mul

    axis_R = T_W_R.new_tensor(mating_params["receiver_axis"])
    axis_R = axis_R / torch.linalg.vector_norm(axis_R)
    mouth_R = T_W_R.new_tensor(mating_params["target_offset_xyz"])
    tip_P = T_W_R.new_tensor(mating_params["subject_offset_xyz"]).expand(T_W_R.shape[0], -1)
    q_W_P = quat_mul(T_W_R[:, 3:], q_R_P)
    tip_W = T_W_R[:, :3] + quat_apply(T_W_R[:, 3:], (mouth_R + depth * axis_R).expand(T_W_R.shape[0], -1))
    return torch.cat((tip_W - quat_apply(q_W_P, tip_P), q_W_P), dim=-1)


class UsbcEnvBehaviourDemo(EnvBehaviourDemo):
    """Inspect mating geometry without physics, then release it into the real task."""

    label = "usbc-validation"

    def __init__(
        self, *args, variant: str, control: str = "plug", teleop: bool = False, pause_steps: int = 60, **kwargs
    ) -> None:
        assert control in ("plug", "receiver"), f"Unsupported controlled connector: {control!r}."
        assert pause_steps > 0, "pause_steps must be positive."
        super().__init__(*args, **kwargs)
        self.variant = variant
        self.control = control
        self.teleop = teleop
        self.pause_steps = pause_steps
        self.keyboard = None
        self._request = None

    def setup_demo(self) -> None:
        """Resolve the unmodified success predicates and optional keyboard controls."""
        import torch

        from isaaclab_arena.tasks.predicates.spatial import depth_in_range

        self.torch = torch
        self.task = self.arena_environment.task
        self.predicates = self.base_env.termination_manager.get_term_cfg("success").params["predicates"]
        self.mating = next(predicate.params for predicate in self.predicates if predicate.func is depth_in_range)
        depth_min = self.mating["depth_min"]
        depth_max = self.mating["depth_max"]
        self.target_depth = (depth_min + depth_max) / 2 if depth_max is not None else depth_min + 0.001
        self.env_ids = torch.arange(self.base_env.num_envs, device=self.base_env.device, dtype=torch.int32)
        self.zero_velocity = torch.zeros((self.base_env.num_envs, 6), device=self.base_env.device)
        self.q_R_P = self.zero_velocity.new_tensor(_MATING_ROTATIONS_XYZW[self.variant]).expand(
            self.base_env.num_envs, -1
        )
        if self.teleop:
            from isaaclab.devices.keyboard import Se3Keyboard, Se3KeyboardCfg

            self.keyboard = Se3Keyboard(
                Se3KeyboardCfg(
                    pos_sensitivity=0.00025,
                    rot_sensitivity=0.005,
                    gripper_term=False,
                    sim_device=self.base_env.device,
                )
            )
            for key, request in (("R", "near"), ("F", "success"), ("SPACE", "release")):
                self.keyboard.add_callback(key, lambda request=request: setattr(self, "_request", request))
            print(
                f"[{self.label}] controlling {self.control}: W/S X, A/D Y, Q/E Z; "
                "Z/X roll, T/G pitch, C/V yaw; R near target, F exact target, SPACE release/check. "
                "Focus the Kit viewport. Pose editing is paused physics, not robot teleoperation.",
                flush=True,
            )

    def _write_pose(self, name: str, T_W_O) -> None:
        asset = self.base_env.scene[name]
        asset.write_root_pose_to_sim_index(root_pose=T_W_O, env_ids=self.env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=self.zero_velocity, env_ids=self.env_ids)

    def _place(self, depth: float) -> None:
        """Move the selected connector along its mating axis while keeping the other fixed."""
        from isaaclab.utils.math import quat_apply

        if self.control == "plug":
            T_W_P = _plug_pose_at_depth(self.receiver_pose, self.q_R_P, self.mating, depth)
            self._write_pose(self.task.plug.name, T_W_P)
        else:
            T_W_R = self.receiver_pose.clone()
            axis_R = T_W_R.new_tensor(self.mating["receiver_axis"])
            axis_R /= self.torch.linalg.vector_norm(axis_R)
            T_W_R[:, :3] += quat_apply(T_W_R[:, 3:], axis_R.expand(self.base_env.num_envs, -1)) * (
                self.target_depth - depth
            )
            self._write_pose(self.task.receiver.name, T_W_R)

    def _diagnostics(self) -> dict:
        """Evaluate child predicates without advancing the consecutive-success counter."""
        from isaaclab_arena.tasks.predicates.spatial import _relative_axial_distances

        depth, lateral = _relative_axial_distances(
            self.base_env,
            **{key: value for key, value in self.mating.items() if key not in ("depth_min", "depth_max")},
        )
        return {
            "depth_mm": (1000 * depth).tolist(),
            "lateral_mm": (1000 * lateral).tolist(),
            **{
                predicate.func.__name__: predicate.func(self.base_env, **predicate.params).tolist()
                for predicate in self.predicates
            },
        }

    def _all_predicates_pass(self) -> bool:
        return all(
            bool(predicate.func(self.base_env, **predicate.params).all().item()) for predicate in self.predicates
        )

    def _hold_action(self):
        action = self.torch.zeros(self.env.action_space.shape, device=self.base_env.device)
        offset = 0
        for name in self.base_env.action_manager.active_terms:
            term = self.base_env.action_manager.get_term(name)
            if name.endswith("arm_action"):
                action[:, offset : offset + term.action_dim] = term._asset.data.joint_pos.torch[:, term._joint_ids]
            offset += term.action_dim
        return action

    def _release_and_check(self) -> None:
        """Require success through ordinary physics stepping and automatic environment reset."""
        succeeded = self.torch.zeros(self.base_env.num_envs, dtype=self.torch.bool, device=self.base_env.device)
        episode_indices = [self.base_env.get_episode_index(env_id) for env_id in range(self.base_env.num_envs)]
        for _ in range(max(self.pause_steps, 120)):
            _, _, terminated, truncated, _ = self.step(self._hold_action())
            success = self.base_env.termination_manager.get_term("success")
            unexpected = (terminated | truncated) & ~success & ~succeeded
            assert not bool(unexpected.any().item()), "Episode ended without USB-C success."
            succeeded |= terminated & success
            if bool(succeeded.all().item()):
                assert all(
                    self.base_env.get_episode_index(env_id) > episode_indices[env_id]
                    for env_id in range(self.base_env.num_envs)
                ), "Success did not trigger the normal environment reset."
                print(f"[{self.label}] success reset observed in all {self.base_env.num_envs} environments", flush=True)
                return
        raise RuntimeError(f"Placement did not produce a physical success/reset: {self._diagnostics()}")

    def run_cycle(self, cycle: int) -> None:
        """Show the failed near-mouth pose, adjust to success, and verify the reset path."""
        self.receiver_pose = self.base_env.arena_world.get_pose_w(self.task.receiver.name).clone()
        if self.control == "receiver":
            self._write_pose(
                self.task.plug.name,
                _plug_pose_at_depth(self.receiver_pose, self.q_R_P, self.mating, self.target_depth),
            )
        self._place(-0.004)
        assert not self._all_predicates_pass(), "Pre-insertion placement must not pass success."
        print(f"[{self.label}] cycle {cycle}: near mouth; {self._diagnostics()}", flush=True)
        if self.keyboard is None:
            for _ in range(self.pause_steps):
                self.render()
            self._place(self.target_depth)
            assert self._all_predicates_pass(), f"Generated target is invalid: {self._diagnostics()}"
            print(f"[{self.label}] target; {self._diagnostics()}", flush=True)
            for _ in range(self.pause_steps):
                self.render()
        else:
            from isaaclab.utils.math import apply_delta_pose

            self.keyboard.reset()
            self._request = None
            frames = 0
            while self.is_running():
                request, self._request = self._request, None
                if request in ("near", "success"):
                    self._place(-0.004 if request == "near" else self.target_depth)
                elif request == "release":
                    if self._all_predicates_pass():
                        break
                    print(f"[{self.label}] not yet successful: {self._diagnostics()}", flush=True)
                delta = self.keyboard.advance().expand(self.base_env.num_envs, -1)
                if bool(delta.any().item()):
                    name = self.task.plug.name if self.control == "plug" else self.task.receiver.name
                    T_W_O = self.base_env.arena_world.get_pose_w(name).clone()
                    position, rotation = apply_delta_pose(T_W_O[:, :3], T_W_O[:, 3:], delta)
                    self._write_pose(name, self.torch.cat((position, rotation), dim=-1))
                if frames % 60 == 0:
                    print(f"[{self.label}] {self._diagnostics()}", flush=True)
                frames += 1
                self.render()
        self._release_and_check()


def run_demo(
    simulation_app,
    *,
    variant: str = "medium",
    cycles: int = 0,
    pause_steps: int = 60,
    real_time: bool = True,
    teleop: bool = False,
    control: str = "plug",
    device: str = "cuda:0",
    visualizer_cfg=None,
) -> None:
    """Run connector placement validation using the unchanged USB-C task configuration."""
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    demo = UsbcEnvBehaviourDemo(
        simulation_app,
        _build_demo_environment(variant),
        ArenaEnvBuilderCfg(num_envs=1, device=device),
        variant=variant,
        control=control,
        teleop=teleop,
        pause_steps=pause_steps,
        real_time=real_time,
        visualizer_cfg=visualizer_cfg,
    )
    demo.run_demo(cycles)


def main() -> None:
    """Launch the scripted demo or interactive connector-pose editor."""
    from isaaclab.app import AppLauncher

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", nargs="?", choices=("easy", "medium"), default="medium")
    parser.add_argument("--cycles", type=int, default=0, help="Zero repeats until the simulation closes.")
    parser.add_argument("--pause-steps", type=int, default=60)
    parser.add_argument("--teleop", action="store_true", help="Edit connector poses with the keyboard before release.")
    parser.add_argument(
        "--control", choices=("plug", "receiver"), default="plug", help="Move the plug or port/bulkhead."
    )
    parser.add_argument("--no-real-time", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=["kit"])
    args = parser.parse_args()
    use_kit = "kit" in (args.visualizer or ())
    if args.teleop and not use_kit:
        parser.error("--teleop requires the Kit GUI (--viz kit).")
    if args.cycles < 0 or args.pause_steps < 1:
        parser.error("cycles must be non-negative and pause-steps must be positive.")
    args.limit_cpu_threads = 1
    with SimulationAppContext(args) as simulation_app:
        if args.teleop:
            assert simulation_app.app_launcher.has_window, "--teleop requires an interactive Kit window."
        visualizer_cfg = None
        if use_kit:
            from isaaclab_visualizers.kit import KitVisualizerCfg

            center = (0.44, 0.0, 0.82) if args.variant == "easy" else (-0.05, -0.1, 0.81)
            visualizer_cfg = KitVisualizerCfg(
                eye=(center[0] + 0.45, center[1] - 0.6, center[2] + 0.4), lookat=center, origin_type="world"
            )
        run_demo(
            simulation_app,
            variant=args.variant,
            cycles=args.cycles,
            pause_steps=args.pause_steps,
            real_time=not args.no_real_time,
            teleop=args.teleop,
            control=args.control,
            device=args.device,
            visualizer_cfg=visualizer_cfg,
        )


if __name__ == "__main__":
    main()
