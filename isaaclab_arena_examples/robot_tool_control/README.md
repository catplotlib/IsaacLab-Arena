# Control a RoboLab experiment from an Astra session

Run the same DROID experiment with OpenPI or an existing Astra session. The
experiment runner owns simulation, episode limits, success checks, and recording.
`DroidSessionPolicy` exchanges observations and joint-action chunks through files;
it does not require an inference server or start a model session.

## Run the experiment

Use an installed Arena checkout and its running Docker container. Run these commands
from `/workspaces/isaaclab_arena` inside the container as the host user. Outside an
interactive shell, use `/isaac-sim/python.sh` instead of the `python` alias.

The example contains one Rubik's-cube task and one episode. With an OpenPI `pi05`
server ready on `127.0.0.1:8003`, run:

```bash
python isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_examples/robot_tool_control/experiment_configs/robolab_rubiks_cube.yaml \
  --viz none --record_camera_video \
  --output_base_dir outputs/robolab_comparison
```

Select the session policy by adding one option:

```bash
python isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_examples/robot_tool_control/experiment_configs/robolab_rubiks_cube.yaml \
  --viz none --record_camera_video \
  --output_base_dir outputs/robolab_comparison \
  --policy_config isaaclab_arena_examples/robot_tool_control/policy_configs/astra_droid.yaml
```

The same option works with the full `robolab_openpi_jobs_config.yaml`. It replaces
the complete policy for **every Run**, including any per-Run policy. Environments,
task instructions, seeds, episode limits, and recording settings remain configured
by the experiment. The option supports typed YAML experiments, not legacy JSON.
Experiment settings must not depend on fields removed with the original policy.

Configuration paths resolve from the runtime working directory. Copy downloaded
YAML files into the mounted checkout or another existing mount before running them.
For Docker, a host Downloads directory is not automatically available in the container.

## Attach a controller

The runner prints the policy's unique session directory under the experiment outputs.
It waits for the controller without advancing physics, with a 600-second timeout
per request by default. Keep the runner alive in another terminal.

Give a **fresh Astra conversation** the printed session directory and this instruction:

> Read `controller_prompt.md` in this session directory and control its first
> episode. Use only that prompt and the current episode's requests and observations.
> Stop at the episode boundary. Session directory: `<path>`.

If the conversation runs on the host, translate the printed
`/workspaces/isaaclab_arena/...` prefix to the local checkout path. All image and
response paths in requests are relative to the session directory, so they work on
either side of the mount. The client uses only Python's standard library and can
run on the host from the checkout root:

```bash
python3 -m isaaclab_arena_examples.robot_tool_control.session_client \
  inspect --session /path/to/session
```

`inspect` prints the active request and its absolute path. The controller reads the
request, views both images, and writes a JSON file containing only the requested
`H × 8` action array. Submit that file using the request path returned by `inspect`:

```bash
python3 -m isaaclab_arena_examples.robot_tool_control.session_client \
  submit --request /path/to/session/episode/request/request.json \
  --actions /path/to/actions.json
```

The client copies the request identity into the response and publishes it atomically.
Read the next request after the chunk executes. A request must be answered once only.

## Controller contract

Each request contains:

- The Run's task instruction.
- One external RGB image and one wrist RGB image, resized and padded to 224 × 224
  using the same `Pi0DroidAdapter` as OpenPI.
- Seven measured joint angles and the normalized gripper position.
- Joint ordering and limits, chunk length, control-step duration, and request identity.

Each output row is seven **absolute joint targets in radians**, then gripper `0`
(open) or `1` (closed). The default chunk has 15 rows. `ActionChunkScheduler` returns
one row per simulation step; the next inference sees observations after that chunk.
No IK or motion planner is inserted. Joint-chunk performance is separate from the
earlier pose-command demo's results.

The controller prompt is in [astra_droid_joint_chunks.md](prompts/astra_droid_joint_chunks.md).
Task instructions stay in `environment_builder.language_instruction`, so changing
policy does not change the task. The policy saves the exact controller prompt,
effective policy configuration, requests, images, responses, and errors with each
session. Arena's normal episode records determine success; an `episode_ended`
exchange event reports only the boundary. The `inspect` command exposes the latest
boundary under `session.last_event` and closure under `session.status`.

## Episode isolation and scope

One controller conversation serves one episode. It can use feedback from earlier
chunks in that episode, but must not reuse targets or observations from prior
episodes. Resetting an Arena policy does not erase the model's conversation.
For multiple episodes, an external driver must start a fresh conversation at each
boundary. This example supplies the exchange protocol and client, not that driver.

The initial interface supports one DROID environment with `droid_abs_joint_pos`.
Invalid, stale, or timed-out responses stop the Run and retain diagnostic artifacts;
they do not trigger another attempt. Keep unsuccessful episodes in evaluation results.
The experiment runner is the supported entry point; the standalone policy runner,
pose commands, and managed OSMO execution are outside this example's scope.
