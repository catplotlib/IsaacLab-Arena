Predicates and Subtask Progress Tracking
========================================

Arena defines task success through ``ProgressObjective`` objects. These objectives organize
Boolean predicates into required milestones, such as settling, lifting, and placing an object.
Their scores also describe partial progress when an episode ends before the task is complete.

Every task returns a ``TaskTerminationCfg`` from ``get_termination_cfg()``. This configuration
declares its ``success`` objectives, named ``failures``, and ``timeout_s`` in one place. The
environment builder creates one success termination that advances the objectives and reports
success when all required objectives are complete.


Predicates
----------

A predicate represents a boolean condition in a task, such as an object settling,
being lifted, or reaching its destination. In Arena, a predicate is a callable that receives
the manager-based environment (and optionally additional configuration arguments) and returns one Boolean per parallel environment.

Included predicates
~~~~~~~~~~~~~~~~~~~

Arena comes with an existing collection of predicates under ``isaaclab_arena.tasks.predicates``, including:

* ``objects_settled`` — all selected objects are below linear and angular velocity thresholds.
* ``object_is_above_height`` — an object is above a fixed reference height.
* ``ObjectSettledWithReference`` — captures a reference height after consecutive low-velocity steps.
* ``object_lifted`` — an object has risen above that captured reference height.
* ``object_moving`` — an object exceeds a linear velocity threshold.
* ``objects_in_proximity`` — two objects are within configured axis-aligned distances.
* ``object_on_destination`` — destination-footprint, upward-support, and velocity checks for a placement goal.

.. note::

    ``ObjectSettledWithReference`` owns its settling counter and reference height per environment.
    It starts observing when its objective becomes active, including in sequential subtasks.
    Pick-and-place defaults to five consecutive low-velocity control steps, configurable through
    ``settling_steps``. It records the height at the end of that window and keeps it until reset.
    Settling earns no progress. This is a velocity-based reference, not a contact/support check.
    The progress tracker forwards episode resets to the predicate.

    The shared ``ObjectInitialRestPoseRecorder`` and ``use_settled_state`` argument have been removed.
    Use ``ObjectSettledWithReference`` as a prerequisite and ``object_lifted`` for a measured reference;
    use ``object_is_above_height`` with ``surface_height`` for a fixed reference.
    The ordinary settling predicates only check stability.


Defining a custom predicate
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a custom predicate when Arena's included predicates do not express the condition you need.
A custom predicate must:

* Accept ``env`` as its first argument.
* Evaluate all parallel environments in one call.
* Return a Boolean tensor with shape ``(env.num_envs,)``.

A predicate may accept any task-specific arguments it needs after ``env``. For example:

.. code-block:: python

   import torch

   def object_inside_x_bounds(env, object_name: str, min_x: float, max_x: float) -> torch.Tensor:
       object_x_e = env.arena_world.get_pose_e(object_name)[:, 0]
       return (object_x_e >= min_x) & (object_x_e <= max_x)

The arguments after ``env`` are configured when the predicate is added to a progress objective
(shown in the next section).


Defining a progress objective
-----------------------------

Use ``prerequisites`` for preparation conditions that must hold together before an objective's
predicate sequences begin:

.. code-block:: python

   ProgressObjective(
       name="pick_and_place",
       prerequisites=[settled],
       predicate_sequence=[lifted, placed],
   )

Prerequisites earn no score or completion events. Readiness is remembered separately per environment
until reset; the first sequence predicate can run in the same update that establishes readiness.
The runner's state exposes ``prerequisites_met`` without evaluating the conditions again.
For sequential subtasks, prerequisites begin when the subtask becomes active.
Policy actions and the episode clock continue while prerequisites are pending.
Ordinary callable prerequisites evaluate the full batch and should be free of state updates;
managed consecutive predicates receive the active-environment mask and reset through the tracker.

Add ``ProgressObjective`` entries to ``TaskTerminationCfg.success``. Provide exactly one of
``predicate_sequence`` for a list of predicates or ``predicate_sequences`` for a dictionary of named lists.

``PickAndPlaceTask`` requires the object to be lifted and then placed. A prerequisite captures
an initial resting reference, without treating settling as a success milestone:

.. code-block:: python

   from functools import partial

   from isaaclab.envs import mdp
   from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

   from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
   from isaaclab_arena.tasks.predicates.object_lifted import ObjectSettledWithReference, object_lifted
   from isaaclab_arena.tasks.predicates.spatial import object_on_destination
   from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

   def get_termination_cfg(self) -> TaskTerminationCfg:
       settled = TerminationTermCfg(
           func=ObjectSettledWithReference,
           params={"object_name": self.pick_up_object.name, "consecutive_steps": self.settling_steps},
       )
       lifted = TerminationTermCfg(func=object_lifted, params={"settled_reference": settled})
       return TaskTerminationCfg(
           success=[
               ProgressObjective(
                   name="pick_and_place",
                   prerequisites=[settled],
                   predicate_sequence=[
                       lifted,
                       partial(
                           object_on_destination,
                           object_cfg=SceneEntityCfg(self.pick_up_object.name),
                           destination_cfg=SceneEntityCfg(self.destination_location.name),
                           contact_sensor_cfg=SceneEntityCfg(self.contact_sensor_name),
                           force_threshold=self.force_threshold,
                           velocity_threshold=self.velocity_threshold,
                           support_cone_half_angle_rad=self.support_cone_half_angle_rad,
                       ),
                   ],
               ),
           ],
           failures={
               "object_dropped": TerminationTermCfg(
                   func=mdp.root_height_below_minimum,
                   params={
                       "minimum_height": self.background_scene.object_min_z,
                       "asset_cfg": SceneEntityCfg(self.pick_up_object.name),
                   },
               ),
           },
           timeout_s=self.episode_length_s,
       )

Use ``functools.partial`` to pass task-specific arguments to a predicate.

Use a dictionary to track several sequences independently. This example requires any two
objects to be lifted and placed:

.. code-block:: python

   objective = ProgressObjective(
       name="pack_objects",
       predicate_sequences={
           "can": [can_lifted, can_placed],
           "bottle": [bottle_lifted, bottle_placed],
           "box": [box_lifted, box_placed],
       },
       logical="choose",
       K=2,
   )

``logical`` and ``K`` control how completed predicate sequences make the objective complete:

* ``all`` — every sequence must complete. This is the default.
* ``any`` — one sequence must complete.
* ``choose`` — at least ``K`` sequences must complete.

Completed stages are remembered until the environment resets. Separate chains therefore describe
milestones that may complete at different times. If several conditions must hold simultaneously,
combine them into one predicate. For example, checking that all gears are seated together requires
one combined condition; remembering each gear's earlier placement would allow a gear to be removed
before the task completes.


Subtask progress tracking in composite and sequential tasks
-----------------------------------------------------------

``CompositeTaskBase`` collects subtask objectives in a flat ``TaskTerminationCfg.success`` list.
It prefixes their names with ``subtask_<index>/`` and sets ``parent_subtask_idx`` to identify
which subtask each objective belongs to. Standalone tasks retain their original objective names,
such as ``pick_and_place``. Nested composite or sequential tasks are not supported.

For an order-independent composite task, every subtask's progress objectives are active.
With ``CompositeTaskBase(..., subtasks_are_sequential=True)``, ``ProgressTracker``
activates each subtask only after all objectives of the preceding subtask complete in that
environment. The next subtask starts on the following environment step. A later subtask's predicates
cannot advance before that subtask becomes active, even if their physical conditions already happen
to be true.

``ProgressTracker`` determines task success and reports the same objective completion history.
Completed milestones remain recorded. ``TaskTerminationCfg.desired_subtask_success_state``
preserves the composition's optional final-condition checks. Reports contain the flat objectives
and their weighted overall progress; subtask metrics read ``ProgressTracker.get_subtask_completion()``.
There are no additional parent-objective reports. See
:doc:`concept_composite_tasks_design` for composition and success semantics.

.. figure:: ../../../images/composite_vs_sequential_progress_tracking.png
   :width: 100%
   :alt: Comparison of predicate tracking activation in composite and sequential tasks
   :align: center

   Composite tasks activate tracking on all subtasks' predicates together, while sequential tasks activate
   tracking on each subtask's predicates only after the preceding subtask succeeds.


Reading subtask progress tracking at runtime
--------------------------------------------

``ProgressTrackingRecorder`` puts progress results in the environment's ``extras`` dictionary.
Read each environment's state and completed-predicate events as follows:

.. code-block:: python

   progress = env.unwrapped.extras["progress_tracking"]

   state = progress["states"][env_id]
   print(state.overall_score, state.all_complete)

   objective = state.progress_objectives["pick_and_place"]
   print(objective.score, objective.is_complete)
   print(objective.active_predicates)

   for event in progress["events"][env_id]:
       print(event.step, event.progress_objective, event.group, event.predicate_name)

After an automatic reset, ``env.extras["progress_tracking"]`` still shows the finished episode
until the next step.

Arena's episode recorder also serializes the final progress state and predicate events into the
episode's JSONL record when an output path is configured. Tasks without progress objectives have
no success termination or progress-tracking configuration and produce no progress fields.

For example, one entry of the JSONL record may look like:

.. code-block:: json

   {
     "progress": {
       "overall_score": 0.5,
       "all_complete": false,
       "objectives": {
         "pick_and_place": {
           "score": 0.5,
           "is_complete": false,
           "completed_groups": 0,
           "total_groups": 1,
           "active_predicates": {
             "default_group": "object_on_destination"
           }
         }
       },
       "events": [
         {
           "step": 18,
           "objective": "pick_and_place",
           "group": "default_group",
           "predicate_index": 0,
           "predicate_name": "object_lifted(...)",
           "score_delta": 0.5
         }
       ]
     }
   }

The object has been lifted: one of two predicates is complete, giving a score of ``0.5``.
Placement is still required. The event records when lifting completed; settling earns no progress.
