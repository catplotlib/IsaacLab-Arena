# Proposal: policy-requested episode termination

Add `PolicyBase.get_episode_stop_mask() -> torch.Tensor | None`: a boolean mask
for environments where the policy has finished, or `None` when none have.
`reset(env_ids)` clears only the corresponding requests. This is a proposed API,
not implemented in this PR.

Add `ArenaEnv.request_episode_stop(mask)` to store requests in environment-owned
state. After `get_action()` and before `env.step()`, the rollout loop forwards
any policy stop mask to this method. Every Arena environment config includes a
framework-owned termination term that reads that state. This keeps policies out
of task definitions and does not bind a policy object to an environment.

A request ends the episode without declaring success; the task's success term
remains authoritative. Isaac Lab then records termination, computes rewards,
and performs its normal per-environment reset. The environment clears requests
for reset IDs, and Arena's existing rollout loop counts the episode and resets
the policy. Infrastructure errors still raise and become run errors, rather
than task failures.

In the pinned Isaac Lab checkout, `ManagerBasedRLEnv.step()` accepts only actions
and computes resets through `TerminationManager.compute()`; there is no policy
stop argument or policy callback. `TerminationManager.set_term_cfg()` can update
an existing term but cannot add one. Arena's `PolicyBase` likewise has no stop
request. Updating reset buffers after `step()` would bypass normal termination
and recording, so the request must participate in the manager's computation.

The current `cap_episode_finished` hook is used only by syringe sort. Keep it
until the shared API lands, then let CAP set its stop mask after the settling
period and remove the CAP-specific term from the task. The socket protocol does
not distinguish intentional graph completion from a server crash that closes the
connection; an explicit completion message would remove that ambiguity.
