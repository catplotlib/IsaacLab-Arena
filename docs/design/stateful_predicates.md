# Stateful predicates using shared task progress

Implementation draft — 2026-09-18. The local implementation adds
`TrueForConsecutiveStepsCfg` and uses it in `PickAndPlaceTask`.
Existing managed predicates remain until their callers migrate.

Builds on [Task success and progress from one definition](https://docs.google.com/document/d/1f8_TUCJA47ArpfL1X9xA1U9uwajGAKSEEFfvvBSdlZI/edit)
and the shared task-termination API merged in [#1255](https://github.com/isaac-sim/IsaacLab-Arena/pull/1255).

## Bigger picture

Some tasks need more than a condition being true for one step. For the Berkeley insertion task,
for example, briefly reaching the correct position is not enough. The required insertion
conditions must remain true for several steps before the task succeeds.

Stateful predicates already exist on `main` through the current workaround. `ConsecutivePredicate`
owns a counter for each parallel environment and inherits from `ManagerTermBase`.
`CompositePredicate` combines child predicates, can count consecutive results itself, and forwards
resets to its children.

This introduces a few complications:

- **State lives in several places.** `ProgressTracker` remembers sequence progress, while
  individual predicates hold their counters.
- **`CompositePredicate` does several jobs.** It combines conditions, counts steps, stores
  diagnostics, and forwards resets.
- **Resets must reach every nested predicate.** Otherwise, a counter could carry state into
  the next episode.
- **Checking can change state.** Evaluating a predicate twice can increment its counter twice.
- **Counting must start at the right time.** A "rest after lifting" requirement must not count
  resting steps before lifting.

The refactor described in [Task success and progress from one definition](https://docs.google.com/document/d/1f8_TUCJA47ArpfL1X9xA1U9uwajGAKSEEFfvvBSdlZI/edit)
established one update and reset lifecycle for task progress and success. The counters themselves,
however, still live inside the manager-based predicates.

This change builds on that groundwork. Wrapping a configured predicate in
`TrueForConsecutiveStepsCfg` says "this condition must hold for 10 consecutive steps."
`ProgressObjectiveRunner` creates and manages a matching `_TrueForConsecutiveSteps` runtime
instance. Task authors do not create another manager term or configure another reset callback.

The separation is simple: **the predicate describes what must be true now; the consecutive-step
requirement describes how long it must remain true.**

## From a manager configuration to a consecutive-step requirement

These two definitions express the same requirement. `placed_and_stable` represents
an already-configured predicate callable returning one Boolean per parallel environment.
Its scene queries and arguments do not change between the examples.

### Before: a managed predicate configuration

```python
placement_held = TerminationTermCfg(
    func=CompositePredicate,
    params={
        "predicates": [TerminationTermCfg(func=placed_and_stable)],
        "consecutive_steps": 10,
    },
)
```

### After: a consecutive-step requirement

```python
placement_held = TrueForConsecutiveStepsCfg(
    predicate=placed_and_stable,
    required_steps=10,
)
```

`TrueForConsecutiveStepsCfg` stores the predicate and the required count, not runtime state.
`ProgressObjectiveRunner` creates a `_TrueForConsecutiveSteps` from that configuration for each occurrence.
This internal runtime class stores the current count for each environment; the runner evaluates the predicate.
Neither class inherits from `ManagerTermBase`.

## How a task declares it

Using the `placement_held` definition above, the task API stays the same:

```python
def get_termination_cfg(self) -> TaskTerminationCfg:
    return TaskTerminationCfg(
        timeout_s=self.episode_length_s,
        success=[
            ProgressObjective(
                name="pick_and_place",
                predicate_sequence=[settled, lifted, placement_held],
            ),
        ],
        failures={"object_dropped": object_dropped},
    )
```

`settled` and `lifted` are shorthand for configured callables, such as `functools.partial` instances;
`object_dropped` is the existing configured failure term. Unchanged argument setup and imports
are omitted. These are illustrative names, not new task attributes.

After settling and lifting, the object must be correctly placed and stable for 10 consecutive
steps. If it moves out of place or stops being stable, the count starts again from zero.
Counting only stability would be a different requirement: it could count an object resting
outside its destination.

`PickAndPlaceTask` exposes this as `placement_consecutive_steps`. Its default is `1`, preserving
the existing placement behavior. Setting it to `10` holds the complete placement condition for
10 consecutive control steps; the settling and lifting predicates are unchanged.

The requirement can also appear in the middle of a sequence:

```python
predicate_sequence=[
    settled,
    lifted,
    TrueForConsecutiveStepsCfg(
        predicate=object_is_stable,
        required_steps=10,
    ),
    placed,
]
```

`object_is_stable` and `placed` are configured callables.
Counting begins after lifting. Placement is checked only after the stability requirement
completes. As with existing sequences, the next predicate starts on the following control step.

## When two conditions need overlapping steps

Suppose a task requires lifting, then placing, then both of these conditions for the same N steps:

- Object A is resting.
- Object B is touching its required target.

Putting two consecutive-step requirements into a sequence would check them one after the other.
Putting them into independent sequences would remember each completion, but would not require
their successful periods to overlap.

Instead, combine the instantaneous conditions and count their shared result:

```python
def both_conditions_hold(env):
    return object_a_is_resting(env) & object_b_is_touching(env)


objective = ProgressObjective(
    name="place_and_hold",
    predicate_sequence=[
        lifted,
        placed,
        TrueForConsecutiveStepsCfg(
            predicate=both_conditions_hold,
            required_steps=n,
        ),
    ],
)
```

This uses one counter per environment. Both conditions must be true on every counted step.
If either becomes false, the counter returns to zero.

They do not need to become true at the same time. A may already be resting when B starts touching;
the first step when both are true is the first counted step. Samples before `placed` completes
do not count.

The same rule applies to insertion: combine the required alignment, depth, contact, and motion
checks first, then apply the consecutive-step requirement to that whole condition.

## Who owns what

| Component | Responsibility |
| --- | --- |
| `TrueForConsecutiveStepsCfg` — new | Declare the predicate and a positive integer `required_steps`. Hold no runtime state. |
| `_TrueForConsecutiveSteps` — new, internal | Store per-environment counts. Increment on true, clear on false, and report whether the configured count is reached. |
| `ProgressObjectiveRunner` — extended | Create a runtime instance per configured occurrence. Evaluate predicates, control activation, record completion, and reset its runtime instances. |
| `ProgressTracker` — existing | Coordinate objective runners and subtask order, and combine their results into task success. |
| `TaskSuccessTerm` — existing | Connect tracker updates and resets to `TerminationManager`. |
| `ProgressTrackingRecorder` — existing | Read progress without changing counters or reevaluating predicates. |

Reusing a predicate or requirement definition in another sequence does not share its counter.
`ProgressObjectiveRunner` creates a separate `_TrueForConsecutiveSteps` from the configuration for each occurrence. The instantaneous
predicate result can still be shared within an update.

`TaskSuccessTerm` remains a `ManagerTermBase` because Isaac Lab calls it.
`ProgressObjectiveRunner` calls `_TrueForConsecutiveSteps.update(predicate_results, active_envs)`
and `reset(env_ids)`. The runtime class does not evaluate predicates or register manager callbacks.
Task authors wrap configured callables with `TrueForConsecutiveStepsCfg` when a condition must
remain true; they do not create or reset runtime instances.

The existing reset path extends to the new counters:

```text
TerminationManager.reset(env_ids)
  → TaskSuccessTerm.reset(env_ids)
    → ProgressTracker.reset(env_ids)
      → ProgressObjectiveRunner.reset(env_ids)
        → _TrueForConsecutiveSteps.reset(env_ids)
```

`ProgressObjectiveRunner.reset()` also clears recorded progress for those environments.
Reset forwarding remains explicit. We do not add a `PredicateGroup` manager wrapper or another
reset event. Other parallel environments keep their progress and counters.

## What counts as a consecutive step

- One step means one environment control step, not a physics substep or a function call.
- Counting starts when the requirement becomes the current predicate in its sequence.
- A true result adds one, up to `required_steps`. A false result clears the streak.
- Environments that have not reached the requirement do not accumulate counts.
- A duplicate update for the same control step must reuse the result, not count again or advance
  another predicate. Reset also clears this step bookkeeping for the restarting environments.
- Skipped control steps break the streak: an unobserved condition cannot count as true.
- Reading progress, diagnostics, or the cached success result never advances a counter.

`TaskSuccessTerm` already passes the per-environment episode step indices to `ProgressTracker.step()`.
The tracker now remembers the last processed index for each environment. Calling it directly with
consecutive-step requirements also requires `step_index`; repeated calls with the same index do
not update progress again. Predicate results remain cached within each update so final-condition
checks do not count twice either.

On the Nth qualifying step, the requirement completes. If it is the last required predicate,
`TaskSuccessTerm` returns success on that same step, without another policy action.

Step counts depend on the configured control rate. They describe consecutive observations,
not proof that a condition remained true between observations. A seconds-based API is outside
the first change.

### Remembered completion is different from a current condition

Once a middle-of-sequence requirement completes, `ProgressObjectiveRunner` remembers it and
moves on. It does not require that condition to remain true through later predicates.

Some composite tasks also use `desired_subtask_success_state` to check final conditions at
overall completion. When `TrueForConsecutiveStepsCfg` is the final predicate of a subtask sequence,
it must continue receiving one update per control step while that final-condition check is
required. If its underlying condition becomes false, its current result becomes false and its
streak clears, without erasing the recorded subtask completion. Becoming true again requires a
new streak.

Sequence advancement and final-condition checks must share that step's result; checking the
final condition must not increment the counter a second time.

## First implementation and follow-up cleanup

The first change adds `TrueForConsecutiveStepsCfg`, runner-owned
`_TrueForConsecutiveSteps` instances, and one real task use. Existing callables, `partial`
definitions, and managed `TerminationTermCfg` predicates remain supported.
`TaskTerminationCfg`, failure checks, and timeouts stay unchanged. Nested duration requirements
and a general interface for arbitrary stateful predicates are outside this first change.
Unifying ordinary and consecutive-step declarations, or adding convenience factories, is follow-up work.

Tests cover interrupted streaks, the Nth-step boundary, middle-of-sequence activation,
overlapping conditions, independent parallel environments, partial resets, duplicate updates,
passive reads, and composite final-condition checks.

Existing implementations need deliberate migration before their adapters can be removed:

- `ObjectsSettledForConsecutiveSteps` records initial resting positions only after the hold
  completes. Wrapping `objects_settled()` directly would record them too early.
- `GearIsSupported` needs environment-aware geometry initialization. That requirement is separate
  from counting steps.
- `GearInsertionFractionRecorder` and `GearEnvBehaviourDemo` read cached `CompositePredicate`
  diagnostics. Their data must remain available without reevaluating stateful predicates.

The goal is one reusable consecutive-step requirement using the tracking and reset ownership
already established by #1255, not another system for deciding when a task is complete.
