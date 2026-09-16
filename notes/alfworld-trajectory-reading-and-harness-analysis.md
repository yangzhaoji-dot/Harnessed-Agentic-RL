# ALFWorld Trajectory Reading and Harness Analysis Workflow

This note records a practical workflow for reading **real ALFWorld Agentic RL trajectories** and turning them into evidence for Harness design.

The immediate target is the public **Qwen3-8B + GiGPO** rollout artifact:

- Dataset: `wckwan/PRM-agent-rl-artifacts`
- Folder: `rollouts/alfworld_qwen3_8b_gigpo/`
- One `*.jsonl` file corresponds to one optimizer step.
- Each JSONL row is one sampled trajectory with decoded prompt / response / reward information.

The goal is **not** to read thousands of trajectories linearly. The goal is to identify matched success/failure behavior, locate divergence points, classify residual failure modes, and study how those modes change over training.

---

## 1. Recommended optimizer-step snapshots

Do not start by reading all 400 steps.

Use a sparse training-time view first:

```text
1.jsonl      # almost-untrained / very early
20.jsonl     # early
80.jsonl     # learning becomes clearer
160.jsonl    # middle
300.jsonl    # late
400.jsonl    # final
```

The exact set can change later. The first purpose is to observe **behavioral evolution**, not to reconstruct every update.

---

## 2. First task: understand the row schema

Before building any automated analysis, inspect the keys of several JSON objects.

Minimal inspection script:

```python
import json

path = "1.jsonl"

with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        x = json.loads(line)

        print("=" * 100)
        print("INDEX:", i)
        print("KEYS:", x.keys())

        for k, v in x.items():
            if isinstance(v, str):
                print(k, ":", v[:300].replace("\n", "\\n"))
            else:
                print(k, ":", v)

        if i >= 2:
            break
```

Do **not** infer field semantics from names alone. Confirm what fields such as `input`, `response`, `reward`, task identifiers, group identifiers, and any step metadata actually contain.

---

## 3. Build a trajectory index before reading full episodes

Instead of opening the raw JSONL directly in VS Code, build a compact index containing at least:

```text
row_index
optimizer_step
task / task-prefix
reward
success/failure if derivable
response_length
trajectory_length if derivable
```

Desired view:

```text
idx   task                          reward   length
---------------------------------------------------
0     put apple in fridge          1.0      2310
1     put apple in fridge          0.0      3412
2     heat potato then place       1.0      1944
3     heat potato then place       0.0      4290
```

The main target is a **same-task success/failure pair**.

---

## 4. Reconstruct each episode as turns

A raw JSON string is not the right unit of analysis.

Convert each trajectory to:

```text
Task
│
├─ Observation 0
│    Action 0
│
├─ Observation 1
│    Action 1
│
├─ Observation 2
│    Action 2
│
├─ ...
│
└─ Final reward / termination
```

Depending on the artifact format, `response` may already contain the full interaction trace or may need to be combined with prompt/environment text.

The important unit is the **environment turn**, not the token sequence.

---

## 5. Success–failure pair analysis

For one task, compare trajectories side-by-side:

```text
                    Success             Failure
----------------------------------------------------
t0 observation      same                same
t0 action           go fridge           go fridge

t1 observation      same                same
t1 action           open fridge         open fridge

t2 observation      same                same
t2 action           take potato         look
                                           ^
                                   first divergence
```

Define the first behavioral divergence:

\[
t^* = \min \{t : a_t^{succ} \neq a_t^{fail}\}.
\]

This is a **candidate intervention point**, not automatically the causal failure point.

A divergence can be harmless; later actions may recover. Therefore inspect the downstream environment consequences before labeling it pivotal.

---

## 6. Harness-oriented failure taxonomy

When a failed trajectory diverges, do not stop at “the action was wrong.”

Ask **what capability was missing**.

### 6.1 Information Harness

Typical pattern:

```text
The trajectory previously observed that the potato is in the fridge.
Several turns later the agent starts searching for the potato again.
```

Possible diagnosis:

- lost object-location information
- forgotten inventory state
- missed environment-state update
- long-history compression failure

Candidate component:

```text
Information / State / Memory Harness
```

---

### 6.2 Plan Harness

Typical pattern:

```text
Goal: heat potato and place it in cabinet
Agent knows where potato and microwave are
Agent puts potato directly in cabinet
```

Possible diagnosis:

- missing subgoal
- wrong subgoal order
- global-plan failure despite locally legal actions

Candidate component:

```text
Plan / Subgoal Harness
```

---

### 6.3 Action Harness

Typical pattern:

```text
Agent knows it must heat the potato.
Agent knows the microwave location.
Agent emits an incorrect or inadmissible concrete command.
```

Possible diagnosis:

- action grounding
- tool syntax
- object/receptacle mismatch
- environment affordance mismatch

Candidate component:

```text
Action / Grounding Harness
```

---

### 6.4 Recovery Harness

Typical pattern:

```text
Agent makes one wrong move.
The environment returns evidence that the action failed.
The agent repeats, oscillates, or begins unrelated exploration.
```

Possible diagnosis:

- failure detection
- loop detection
- replanning failure
- inability to recover after an irreversible/local error

Candidate component:

```text
Recovery / Reflection Harness
```

---

## 7. Failed trajectories still contain useful steps

A failed trajectory can contain a long prefix of correct behavior:

```text
t0  ✓ find potato
t1  ✓ take potato
t2  ✓ find microwave
t3  ✓ put potato in microwave
t4  ✗ fail to activate / complete heating
t5  ✗ wander
...
final reward = 0
```

This matters because trajectory-level outcome reward can make all steps share the same poor signal even though early actions are useful.

For our project, this creates two distinct questions:

1. **Credit assignment:** which useful local behaviors should still receive positive learning signal?
2. **Harness placement:** should assistance be injected only near the first unrecovered failure instead of across the full rollout?

This is one of the main reasons real trajectory inspection is necessary before choosing a Harness architecture.

---

## 8. Study failure modes across training time

The most important analysis is not only success vs failure at one checkpoint.

Track failure types over optimizer steps:

```text
                 step 1   step 80   step 160   step 300   step 400
------------------------------------------------------------------
invalid action      high      low        low        low        low
state forgetting    high     medium     medium     medium     medium
plan omission       high     high       medium     medium     medium
recovery failure    high     high       high       medium     medium
```

Interpretation:

- Failure types that disappear naturally under RL may not justify a permanent Harness.
- Failure types that persist after substantial training are **persistent residual weaknesses**.
- Persistent residual weaknesses are stronger candidates for long-lived external Harness components.
- A component whose usefulness decreases over training may be a candidate for **internalization / retirement**.

This gives a concrete empirical interpretation of Harness handoff:

\[
\text{Harness value}_k(t) \downarrow
\quad \Rightarrow \quad
\text{candidate internalization or retirement}.
\]

---

## 9. Candidate causal test for a Harness component

A behavioral divergence is only a hypothesis.

For a selected state `s_t`, replay / branch from the same state:

```text
same state s_t
│
├─ no Harness             × N continuations
├─ Information Harness    × N continuations
├─ Plan Harness           × N continuations
├─ Action Harness         × N continuations
└─ Recovery Harness       × N continuations
```

Estimate the component effect:

\[
\Delta_H(s_t)
=
P(\text{success} \mid s_t, H)
-
P(\text{success} \mid s_t, \varnothing).
\]

This is much stronger than saying “entropy was high here” or “an LLM judge thinks this step is important.”

A useful future Harness controller should eventually answer two separate questions:

```text
1. Does this state need intervention?
2. Which typed Harness component has positive marginal value here?
```

---

## 10. First empirical study protocol

A small first pass is enough.

### Phase A — schema and parsing

- inspect `1.jsonl`
- confirm row schema
- reconstruct task / turn / reward
- build a compact trajectory index

### Phase B — within-step behavioral pairs

For optimizer step 1:

- choose ~10 tasks
- select one successful and one failed trajectory where possible
- identify first divergence
- classify the failure
- record whether recovery occurred

### Phase C — training evolution

Repeat the same analysis at:

```text
20 / 80 / 160 / 300 / 400
```

Prefer recurring task families so failure modes can be compared across training stages.

### Phase D — candidate Harness intervention study

Select a small set of representative states and perform controlled branching with:

```text
none / information / plan / action / recovery
```

Estimate marginal success improvement and extra token / rollout cost.

---

## 11. What we should record for every analyzed case

A compact case record should contain:

```text
optimizer_step
task_id / task_family
trajectory_id
success / reward
trajectory_length
first_divergence_turn
candidate_pivotal_turn
failure_type
recoverable? yes/no
information_missing? yes/no
plan_missing? yes/no
action_grounding_error? yes/no
recovery_failure? yes/no
candidate_harness_component
notes
```

Later, after controlled branching:

```text
plain_success_rate
harness_success_rate
marginal_effect
extra_tokens
extra_model_calls
```

---

## 12. Research objective

The purpose of this trajectory study is not descriptive logging.

It should answer four design questions for Harnessed Agentic RL:

1. **WHERE** should intervention occur?
2. **WHAT** type of Harness is needed at that state?
3. **WHICH failures are naturally internalized by RL?**
4. **WHEN can a Harness component be reduced or retired?**

A useful working abstraction is:

\[
(s_t, \tau_{<t})
\rightarrow
\text{capability diagnosis}
\rightarrow
H_{info/plan/action/recovery}
\rightarrow
\text{counterfactual branch}
\rightarrow
\Delta_H(s_t).
\]

This provides a data-driven route from real ALFWorld behavior to the eventual dynamic heterogeneous Harness design.
