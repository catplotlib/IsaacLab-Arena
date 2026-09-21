Offline Placement Recording
===========================

A geometrically valid layout may move when physics starts. This recorder applies
solved layouts, settles the complete scene, filters unsuitable results, and saves
the accepted final poses for replay. Ordinary relations and ``ClutterOn`` can
coexist on a fixed clutter support.

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/scripts/record_placement_layouts.py \
       env_spec=scene.yaml output=placements.jsonl \
       num_envs=4 layouts_per_env=10 presets=newton --viz none

Each environment supplies its own candidate queue. Recording preserves the pool
and restores scene roots, joints and actuator targets after success or failure.
It adds no settling to runtime resets. Use this recorder for ordinary or mixed
relation scenes. ``generate_clutter_scene.py`` remains the dedicated clutter
sampler with release resampling; both write the same replay format.

Acceptance checks
-----------------

Every accepted layout must pass the source solver's required checks and remain
below the velocity limits throughout a quiet window. Final checks use the
measured poses:

* Ordinary objects must remain within the configured displacement and rotation
  limits and pass their required relation checks, including ``On``.
* ``ClutterOn`` members may drop and rotate freely. They must remain above and
  inside their support's full footprint, rather than the smaller release region.
* Collision checks use full rotations and collision meshes, with conservative
  bounding-box proxies when meshes are unavailable. Resting contact is permitted
  within ``penetration_tolerance_m``; intersecting release layouts are not accepted.
* Articulation roots are recorded. Links must stay near their initial poses
  relative to their root, because joint states are not included in placement files.

For example, an ordinary object that falls to the floor fails its final ``On``
check even if it stops moving. A clutter member that leaves its support is also
rejected. Enabled IK reachability checks use measured object orientations and the
recorded robot root pose. Enabled and required checks are retained for final
validation, even when the source pool permits optional-check failures.

The command prints accepted/attempted counts. The Python result also contains
accepted source indices and rejection reasons, including source solver failures.
The output can contain fewer layouts than the pool. Set ``settle.min_layouts``
to require a minimum yield; recording fails without writing when it is not met.
Existing output files are not overwritten.

Settings
--------

Settings are Hydra overrides:

* ``settle.settle_time_s=4`` and ``settle.quiet_time_s=1``: simulated settling and
  quiet-window durations, independent of control decimation.
* ``settle.lin_vel_thresh=0.1`` and ``settle.ang_vel_thresh=0.1``: maximum speeds
  in metres/second and radians/second throughout the quiet window.
* ``settle.max_translation_m=0.1`` and ``settle.max_rotation_deg=15``: maximum
  ordinary-object motion from solved to recorded pose. These do not limit clutter
  drops and do not replace final geometry checks.
* ``settle.max_facing_error_deg=2``: maximum final ``FaceTo`` heading error.
* ``settle.max_link_translation_m=0.002`` and ``settle.max_link_rotation_deg=2``:
  permitted unrecorded articulation-link motion relative to the root. An arm that
  sags beyond these limits is rejected: replaying its root cannot reproduce its
  settled joint configuration. These limits are not backend tuning parameters.
* ``settle.penetration_tolerance_m=0.002``: tolerated final contact penetration.
* ``settle.min_layouts=1``: minimum accepted output layouts.
* ``render=true --viz kit``: show settling in the visualizer.

Replay and identity
-------------------

The output uses the existing JSONL ``scene.relation_placement`` envelope.
Positions are environment-local in metres and rotations are xyzw quaternions.
Each record contains a complete settled layout, including ordinary objects,
clutter members and relation-placed articulation roots.

.. code-block:: bash

   /isaac-sim/python.sh isaaclab_arena/scripts/environment_runner.py \
       --env_spec scene.yaml --placement_layouts placements.jsonl

The graph CLI writes YAML node IDs. Python environments call
``collect_settled_pool_layouts(env, pool, params)`` from
``isaaclab_arena.relations.settled_placement`` and receive a
``PlacementRecordingResult``. Its ``layouts`` use runtime scene keys, matching
Python cached replay. Pass ``scene_assets=arena_env.get_placement_assets()`` when
articulations are not relation-placed, so their geometry is also validated.
Write them with ``result.layouts.write_episode_jsonl(path)``.
To use that file with a YAML graph, map scene keys to graph node IDs first; sharing
the JSONL envelope does not make the two identity spaces interchangeable.

Replay restores poses and zeroes root velocities without solving or dropping
again. Use the same assets, physics settings and compatible robot joint reset
configuration. These are placement records, not complete evaluation conditions:
variation values such as mass, friction and camera settings are not captured.

Supported scenes
----------------

Movable assets must expose rigid or articulation roots. Other dynamic rigid
objects must participate in placement. Object sets must be resolved before
recording, and ``RandomAroundSolution`` must be removed for cached replay.
``ClutterOn`` supports remain static or kinematic anchors. Unrecorded joint
randomization must be held fixed when reproducible full-scene conditions are
required.
