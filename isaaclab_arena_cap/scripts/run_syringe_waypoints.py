# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Roll out the CAP open-loop syringe waypoint plan against the Arena Newton env.

Exploratory runner: builds ``vabar_tool_sort__syringe_easy_newton``, settles the
scene, then follows the vendored Cartesian waypoint plan with the CAP policy,
logging poses and the terminal success/timeout signal.

    /isaac-sim/python.sh -m isaaclab_arena_cap.scripts.run_syringe_waypoints
"""

import pathlib

# Import nothing numpy/torch-bearing at module scope: those imports must happen after
# SimulationApp launches, or the Kit launcher's fork crashes on OpenBLAS threads.
PLAN = str(pathlib.Path(__file__).resolve().parents[1] / "policy" / "plan" / "vabar_tool_sort_syringe.yaml")
ENV_NAME = "vabar_tool_sort__syringe_easy_newton"


def _run(_sim_app) -> bool:
    import torch

    import isaaclab_arena_cap.environments  # noqa: F401  # registers the syringe env factory
    import isaaclab_arena_cap.policy.cartesian_waypoint_policy as cwp
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    registry = EnvironmentRegistry()
    factory = registry.get_component_by_name(ENV_NAME)
    cfg_type = registry.get_environment_cfg_type(factory)
    arena_env = factory().build(cfg_type())
    args = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args)).make_registered()

    policy = cwp.CartesianWaypointPolicy(
        cwp.CartesianWaypointPolicyArgs(policy_device="cuda", plan_path=PLAN, terminate_on_exhaustion=True)
    )

    def _t(v):
        return v.torch if hasattr(v, "torch") else v

    def _p(name, asset):
        pos = _t(asset.data.root_pos_w)[0]
        return f"{name}=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})"

    try:
        obs, _ = env.reset()
        policy.reset()
        u = env.unwrapped
        syr = u.scene["syringe_0"]
        cont = u.scene["sharps_container"]
        robot = u.scene["robot"]
        ee_idx = robot.find_bodies("robotiq_base")[0][0]

        # Let the scene settle so the policy snapshots the syringe at its resting pose.
        hold_arm = _t(robot.data.joint_pos)[:, :7]
        hold = torch.cat((hold_arm, torch.zeros((u.num_envs, 1), device=u.device)), dim=-1)
        for _ in range(90):
            obs, _, _, _, _ = env.step(hold)

        print("INIT", _p("syringe", syr), _p("container", cont), flush=True)
        print("PLAN_WAYPOINTS", [w.name for w in policy._plan.waypoints], flush=True)

        last_wp = -1
        fired = False
        for step in range(2200):
            action = policy.get_action(env, obs)
            obs, _, term, trunc, _ = env.step(action)
            if not fired and (bool(term.any()) or bool(trunc.any())):
                fired = True
                kind = "SUCCESS(terminated)" if bool(term.any()) else "TIMEOUT(truncated)"
                print(f"*** TERMINATION {kind} at step={step} {_p('syringe', syr)}", flush=True)
            wp = policy._waypoint_index
            if wp != last_wp:
                last_wp = wp
                name = policy._plan.waypoints[wp].name if wp < len(policy._plan.waypoints) else "DONE"
                eepos = _t(robot.data.body_pos_w)[0, ee_idx]
                grip = float(action[0, -1])
                print(
                    f"step={step:4d} -> waypoint[{wp}]={name} "
                    f"ee=({eepos[0]:.3f},{eepos[1]:.3f},{eepos[2]:.3f}) grip={grip:.2f} "
                    f"{_p('syringe', syr)}",
                    flush=True,
                )
            mask = policy.get_episode_termination_mask()
            if mask is not None and bool(mask.all()):
                print(f"plan exhausted at step={step}", flush=True)
                break

        print("FINAL", _p("syringe", syr), _p("container", cont), flush=True)
    finally:
        env.close()
    return True


if __name__ == "__main__":
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    print("RUN_OK", run_function_with_persistent_simulation_app(_run), flush=True)
