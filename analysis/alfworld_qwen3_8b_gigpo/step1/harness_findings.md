# Harness-oriented findings from real ALFWorld Qwen3-8B GiGPO data — optimizer step 1

This note converts the raw/context-matched analysis into research hypotheses for our **dynamic heterogeneous Harness** project.

## 0. What the public dump actually contains

`1.jsonl` has **5,120 rows** and exactly four fields per row:

- `input`: agent-visible prompt for one ALFWorld decision turn, including the task, recent observation/action history, the current observation, and admissible actions;
- `output`: one model response, usually `<think>...</think>` + `<action>...</action>`;
- `score`: rollout/training score associated with that sample;
- `step`: optimizer/global training step (`1` for every row in this file), **not the ALFWorld environment-step index**.

The environment step must be parsed from text such as `You are now at step 39` inside `input`.

The public generation dump does **not** preserve the trainer's `traj_uid`, `uid`, or `anchor_obs`. Therefore exact trajectory identity cannot be reconstructed losslessly from the JSONL alone. Our reliable analysis unit is instead an **exact visible context group**:

`task + ALFWorld env step + recent observation/action history + current observation`.

Within such a group the model is facing the same visible decision situation; different sampled actions form local branches.

## 1. Step-1 batch statistics

From `local_branch_summary.json`:

- raw decision-turn samples: **5,120**
- task templates/scenes represented: **16 extracted task strings**
- exact visible-context groups: **4,293**
- repeated context groups (`n >= 2`): **401**
- repeated contexts with >1 sampled action: **138**
- repeated contexts containing both high-score and low-score samples: **94**
- matched contexts with both mixed outcomes **and** different actions: **41 candidate local branch points**
- same-context + same-action but mixed outcome cases: **71 negative controls**
- outputs from which an action was parsed: **4,258 / 5,120 (~83.2%)**
- outputs with a closed `</action>` tag: **4,248 / 5,120 (~83.0%)**
- parsed actions not present in the prompt's admissible-action list: **89 / 4,258 (~2.1%)**

Observed score values are only `10.0`, `9.9`, `0.0`, and `-0.1`. This is consistent with an outcome score plus a small invalid-action/format penalty, but the JSONL itself does not expose the reward decomposition, so we should not treat that interpretation as proven without checking the exact training reward code/config.

### Immediate methodological warning

The **71 same-context + same-action + different-score** controls are important. They prove that:

`score difference != causal credit of the current action`.

Later decisions remain a confounder. Therefore a Harness trigger should not be justified solely by correlating one local action with final trajectory reward. The right follow-up is **replay/branch intervention from the matched state**.

---

## 2. Case A — a clean Action/Plan candidate

Task: `heat some apple and put it in garbagecan.`  
ALFWorld step: **39**

Matched recent history:

- step 37: microwave closed -> `open microwave 1`
- step 38: microwave open, apple 2 visible -> `take apple 2 from microwave 1`
- current observation: `You pick up the apple 2 from the microwave 1.`

Two sampled branches under the same visible context:

- HIGH-score branch: `heat apple 2 with microwave 1` -> score `10.0`
- LOW-score branch: `move apple 2 to microwave 1` -> score `0.0`

Both actions are admissible.

### Why this is useful

The low branch is **not an invalid-action failure**. It selects a legal action, but the prompt already exposes the task-relevant higher-level operator `heat apple 2 with microwave 1`.

This is exactly the kind of failure that an Action/Plan Harness should target:

1. parse the current goal/subgoal (`heat apple`);
2. match it against currently available affordances/operators;
3. prioritize the task-progressing operator over merely legal state manipulation.

A useful replay experiment is:

`same state -> vanilla continuations vs affordance-ranked Action Harness continuations`.

The public association is only a candidate signal; replay is needed for causal evidence.

---

## 3. Case B — oscillation / Recovery Harness candidate

Task: `heat some apple and put it in garbagecan.`  
ALFWorld step: **36**

Matched recent history:

- step 34: microwave open with apple -> `close microwave 1`
- step 35: microwave closed -> `open microwave 1`
- current observation: microwave open with apple inside

Branches:

- HIGH-score branch: `take apple 2 from microwave 1` -> score `10.0`
- LOW-score branch: `close microwave 1` -> score `0.0`

The low action is legal, but it **repeats the recent open/close cycle**.

### Harness implication

This suggests a separate **Recovery/Stuck Harness**, not just an Action Harness.

A trigger can use structural trajectory signals:

- repeated inverse action pairs (`open -> close -> open -> close`),
- repeated state signatures,
- no progress toward task predicates,
- repeated tool/action cycles.

The intervention need not tell the model the exact next action. It can inject a compact recovery state such as:

`You are revisiting the same microwave state. Identify the unresolved goal predicate and choose an action that changes task progress.`

This kind of component may remain useful even after basic action syntax has been internalized.

---

## 4. Case C — early search is noisy, so do not over-credit one action

Task: `clean some potato and put it in garbagecan.`  
ALFWorld step: **3**

Shared prefix:

- step 1: `go to fridge 1`
- step 2: `open fridge 1`
- current observation: fridge contains bowls/egg/mug/tomato, **no potato**

Nine matched samples reach this same context:

- `go to sinkbasin 1`: 3 samples, all high-score
- `look`: 1 sample, high-score
- `go to countertop 1`: 5 samples, 2 high-score + 3 low-score

This is not enough to conclude that `sinkbasin` is the uniquely correct action. Search choices have downstream stochastic/behavioral consequences, and one action itself appears with both outcomes.

### Harness implication

For exploration/search states, a controller should use **uncertainty over branch value**, not a brittle single-action rule.

Candidate design:

- if several legal exploratory branches remain plausible, retain diversity / branch;
- reserve strong Action Harness intervention for states with clearer task-affordance mismatch or repeated failure structure.

This gives us a possible distinction between:

`exploration ambiguity` vs `recoverable policy error`.

---

## 5. Case D — malformed/truncated action generation is common enough to measure explicitly

Only about **83%** of rows contain a cleanly parsed action under the current parser, while 89 parsed actions are not in the supplied admissible list.

At least some low-score examples visibly end with malformed/truncated outputs such as an unfinished `<action>...` rather than a complete admissible command.

We should separate two subtypes:

1. **format generation failure** — no complete action tag / truncated action;
2. **semantic invalid action** — complete action string but not in the current admissible set.

### Harness implication

This is the simplest Action Harness baseline:

`generation -> syntax/admissibility validator -> repair/resample if invalid`.

But it is probably **not** where our final novelty should live: syntax/admissibility failures are likely to shrink quickly during RL and may be easy for the policy to internalize.

That makes them useful as a test case for **Harness retirement**:

- early training: validator has high causal utility;
- later training: invalid-format rate falls;
- controller should reduce/retire this component.

---

## 6. Information Harness hypothesis: the prompt itself creates partial observability

The raw inputs usually expose only a short recent history (often the most recent two observation/action pairs). At late environment steps, task-relevant events from much earlier in the episode may have disappeared from the visible context.

This creates a concrete Information Harness opportunity:

`long raw history -> compact persistent task-state summary`.

The summary should track goal predicates, e.g.:

- target object identity/location;
- whether it has been picked up;
- whether `heated/cleaned/cooled/sliced` has already happened;
- destination state;
- recently exhausted search locations;
- detected cycles.

Crucially, we should not assume memory is useful everywhere. We can test it only on states where the same visible local context is compatible with different hidden task-progress histories.

---

## 7. A data-derived Harness decomposition

The real step-1 data suggests a cleaner component set than inventing arbitrary modules upfront:

### A. Action Validity Harness
- trigger: malformed output or action not in admissible set
- intervention: repair/resample/constrain
- expected behavior over training: likely rapidly internalized -> retire early

### B. Goal-Affordance / Action Harness
- trigger: legal action selected despite a clearly available goal-progressing operator
- intervention: rank/filter actions by current subgoal and affordance
- example: `heat apple ...` vs merely `move apple ...`

### C. Recovery Harness
- trigger: state/action cycle, repeated inverse actions, stalled progress
- intervention: summarize loop + unresolved predicate + force a progress-changing decision
- example: microwave open/close oscillation

### D. Information/State Harness
- trigger: relevant task predicate/history no longer visible or state aliasing suspected
- intervention: persistent compact state summary / memory retrieval

### E. Exploration Harness (tentative)
- trigger: genuinely ambiguous search state with several plausible legal actions
- intervention: maintain branch diversity or allocate extra rollout budget rather than dictate one action
- caution: current-step outcome association is especially weak here

This decomposition is more evidence-grounded than starting from generic `Information / Plan / Action / Recovery` names alone. In particular, **Action Validity** and **Goal-Affordance Action** should be kept separate because one is syntax/interface competence while the other is task-level decision competence.

---

## 8. What should be measured across training steps

Run exactly the same analyzer on `1 / 20 / 80 / 160 / 300 / 400.jsonl` and track:

1. malformed-action rate;
2. non-admissible action rate;
3. repeated-state/action-cycle rate;
4. number/fraction of exact-context multi-action groups;
5. mixed-outcome branch-value dispersion;
6. frequency of goal-affordance mismatches;
7. frequency of state/history aliasing candidates;
8. component-specific replay gain after applying each Harness.

Then define component utility as a function of policy age:

`U_h(k) = E[R | intervene with h at step k] - E[R | no intervention at step k]`.

A component becomes a retirement/internalization candidate when its **causal intervention gain** falls persistently toward zero, not merely when its usage frequency falls.

---

## 9. Strongest next experiment

Pick ~20 candidate local states from the real dump, stratified into:

- 5 malformed/invalid-action states;
- 5 goal-affordance mismatch states;
- 5 loop/stuck states;
- 5 history/information states.

Replay each state with one Qwen3/3.5 4B–8B policy using:

- vanilla continuation × N;
- matched Harness component × N;
- irrelevant Harness component × N (negative control).

Measure:

`Delta_h(s) = P(success | s, h) - P(success | s, no h)`.

This directly answers **where a Harness helps, what kind helps, and which components later become unnecessary**.

## Files

- `summary.json` — raw row/schema summary
- `local_branch_summary.json` — exact-context group statistics
- `local_branch_candidates.jsonl` — all 41 mixed-score/different-action matched contexts
- `local_branch_report.md` — readable candidate cases
- `same_action_mixed_controls.jsonl` — negative controls showing local score is not causal credit
- `representative_rows.jsonl` — raw examples
