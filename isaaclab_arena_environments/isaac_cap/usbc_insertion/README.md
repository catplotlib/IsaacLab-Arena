# USB-C insertion behavior demo

Like the gear behavior demo, this is a task-validation tool, not a robot policy.
It places a connector 4 mm outside the receiver mouth, shows the failing depth
predicate, then moves it to an aligned insertion pose derived from the task's
actual geometry and thresholds. It reports each predicate, releases the pose
into normal physics, and requires **success termination and automatic reset**.
A timeout or another termination is not accepted as success. Both `easy` and
`medium` variants are supported with the registered task's unchanged thresholds.
The medium plug uses half the shared USD's scale, matching its 6.65 mm tip offset
and the precision port bore; the easy plug retains its original scale.

Run inside this checkout's Arena Docker container, as the host user, from
`/workspaces/isaaclab_arena`:

```bash
/isaac-sim/python.sh isaaclab_arena_environments/isaac_cap/usbc_insertion/usbc_env_behaviour_demo.py medium --cycles 2
```

To keyboard-control the **port** (or the bulkhead in `easy`) around the mating pose:

```bash
/isaac-sim/python.sh isaaclab_arena_environments/isaac_cap/usbc_insertion/usbc_env_behaviour_demo.py medium --teleop --control receiver
```

Use `--control plug` (the default) to move the plug instead. Focus the Kit viewport:

| Keys | Action |
| --- | --- |
| W / S, A / D, Q / E | Translate along world X, Y, Z (0.25 mm per frame) |
| Z / X, T / G, C / V | Rotate around world X, Y, Z (0.005 rad per frame) |
| R | Restore the aligned near-mouth pose |
| F | Snap to the exact successful pose |
| SPACE | Release into physics and verify success/reset, if all predicates pass |

Pose editing and display pauses render without advancing physics, so gravity
does not fight the keyboard. Velocities are zeroed when teleporting. Passing the
preview predicates alone is **not** proof of a stable physical insertion: the
subsequent physics steps must satisfy the unchanged task too. The robot arms
hold their current joint positions with open grippers during that check.

For a finite headless check, use `--cycles 1 --pause-steps 1 --no-real-time --viz none`.
The demo uses one environment for close-up inspection.
Without `--cycles`, the demo repeats until the application closes.

## Registered assets

The connectors, fixtures, tables, and shadow receiver use `@register_asset` and
are available through `AssetRegistry`. Both environment graphs use those registered
classes; poses and per-scene instance names live in YAML.

```python
from isaaclab_arena.assets.registries import AssetRegistry

plug = AssetRegistry().get_asset_by_name("usbc_insertion_precision_plug")(
    instance_name="usbc_plug",
    prim_path="{ENV_REGEX_NS}/UsbcPlug",
)
```

Registry names share the `usbc_insertion_` prefix: `plug`, `precision_plug`,
`bulkhead`, `port`, `bench`, `cradle_front`, `cradle_rear`, `fr3_table`, `yam_table`,
`hdr_shadow_receiver`, `dome_light`, and `connector_cable`. The precision plug retains its half scale, 4 g mass,
and tuned contact material. Each instance gets independent mutable simulator
configuration.

Both workcell tables are `Background` assets, not rigid task objects. Their roots
remain static, while any discovered nested physics uses Arena's background reset.
The port, bench, and cradles remain kinematic objects with ordinary initial-pose
reset events, so explicitly moving a fixture does not persist across episodes.

## Environment graphs

Success uses the shared spatial predicates `depth_in_range`, `xy_in_proximity`,
`tilt_axis_aligned`, and `velocity_below_threshold`. The task supplies plug-tip and
receiver-mouth offsets plus the receiver-local insertion axis; easy additionally
allows antiparallel mating axes. No USB-C-specific predicate implementation is needed.

`usbc_easy.yaml` and `usbc_medium.yaml` use the same `ArenaEnvGraphSpec` schema as
the Robolab environments. They define the embodiment, registered asset instances,
fixed poses, plug reset ranges, HDR lighting, and task parameters. An asset's
`initial_pose` accepts either `position_xyz` / `rotation_xyzw` or a reset range
with `position_xyz_min`, `position_xyz_max`, `rpy_min`, and `rpy_max`.

The registered factories in `environment.py` load these graphs, apply runtime
camera/mesh options, and attach the appropriate Newton callback. `assets.py`
contains reusable asset definitions only; the task predicates, physics tuning,
and procedural cable generation remain in Python. Launch through the registered
factory or the demo to retain those physics and cable callbacks.

The easy graph declares `plug_cable` and `bulkhead_cable` as separate registered
assets. Each installs its own deferred Newton builder hook and contributes a reset
event through the normal scene asset interface. Reset restores that cable's zero
bend coordinates and velocities in both Newton state buffers, only for the selected
environments. Neither the task nor the physics callback adds cable reset events.
These attached MJWarp hinge chains intentionally do not use Arena's `Cable` class:
its current backend expects standalone, unwelded VBD cable articulations rather
than revolute-joint chains attached to rigid connectors.
