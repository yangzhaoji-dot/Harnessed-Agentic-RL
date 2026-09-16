# ALFWorld Public Trajectory Corpora — Cross-Model / Cross-Training Comparison Index

> Snapshot: **2026-09-16**.  
> Goal: maintain a compact index of public ALFWorld trajectory / transition resources that are useful for **behavior-level comparison across model scales, training stages, retrieval/memory settings, and RL recipes**.
>
> This file is intentionally separate from the broader literature map. It is an empirical data inventory for future analysis.

## Why this matters for our Harness project

Aggregate success rate is not enough. We want to compare how different agents actually fail:

- invalid / no-effect actions
- search inefficiency and loops
- state / history failures
- wrong subgoal or planning failures
- legal-but-nonprogressing actions
- long-horizon timeout behavior
- sensitivity to memory / retrieval / full-history support

The central question is:

> **Which failure modes disappear with model scale or RL, and which remain persistent enough to justify an external Harness?**

---

## Corpus matrix

| Resource | Data type | Scale / models | ALFWorld size | Full episode identity? | Main fields / strengths | Best use for us |
|---|---|---|---:|---|---|---|
| [`wckwan/PRM-agent-rl-artifacts`](https://huggingface.co/datasets/wckwan/PRM-agent-rl-artifacts) | **True optimizer-step RL rollout dumps** | Qwen3-8B GiGPO (plus other agent runs in the artifact repo) | ALFWorld dumps for optimizer steps through training; whole repo ~84.7 GB | **No, not in the public ALFWorld JSONL we audited** | `input`, `output`, `score`, optimizer `step`; our repo already contains score, token-length, failure-signal and matched-context analyses | **Training dynamics**: how failure signals and behavior change during RL |
| [`talesuite/tale_suite_trajectories`](https://huggingface.co/datasets/talesuite/tale_suite_trajectories) | Evaluation trajectories | Many LLMs; multi-model zero-shot text-game evaluation | **~3.42k ALFWorld trajectories** | **Yes** | `transcript_id`, model, game, total steps, score, full `turns`, score progression, thinking/token metadata | **Cross-model behavior comparison** and complete success/failure trajectory reading |
| [`toeunkim/matm-trajectories`](https://huggingface.co/datasets/toeunkim/matm-trajectories) | Evaluation + retrieval/memory-condition trajectories | 34-model ALFWorld consumer population; additional cohorts | **2,130 ALFWorld population-run episodes** + **7,147 ALFWorld prepopulation successful trajectories** | **Yes** | action, observation, reasoning, success, final score, retrieval strategy, trajectory ID | **Same model with/without retrieval-style assistance**; Harness-like comparison |
| [`XinnanZhang/alfworld-transition-pairs-1.5b`](https://huggingface.co/datasets/XinnanZhang/alfworld-transition-pairs-1.5b) | Replayable transition pairs from real rollouts | **1.5B agent** | **420 episodes → 7,954 raw transitions → 7,552 distinct pairs** | **Provenance retained** via `sources.traj_uid` | state, action, next state, validity, success/failure counts, action prefix, cost-to-go, replay provenance | **Credit / branch analysis**; small-model failure behavior; replay experiments |
| [`uyffg/auto-dreamer`](https://huggingface.co/datasets/uyffg/auto-dreamer) | Rollout trajectories + memory-writer outputs | auto-dreamer memory-RL pipeline; memory-writer outputs include mixed writers in train split | **3,553 ALFWorld train episodes + 274 test episodes** | **Yes** | `episodes.jsonl`, `steps.jsonl`, `trace_index.jsonl`; step-level observation/action/reward plus memory-writer diagnoses / inserts | **Memory / Information Harness analysis**, failure taxonomy, trace-level diagnosis |

---

## 1. Qwen3-8B GiGPO training rollouts — `PRM-agent-rl-artifacts`

Source: https://huggingface.co/datasets/wckwan/PRM-agent-rl-artifacts

Relevant path:

```text
rollouts/alfworld_qwen3_8b_gigpo/
```

This is currently our strongest source for **within-training temporal analysis**.

### Important schema correction from our direct audit

For the ALFWorld dump we inspected, a JSONL row is a **model decision/action turn**, not a losslessly reconstructable full episode. The public fields are:

```json
{
  "input": "...",
  "output": "...",
  "score": 10.0,
  "step": 1
}
```

The public dump does **not** preserve the internal `traj_uid` / `uid` / `anchor_obs` fields needed to reconstruct exact trajectory identity. Therefore:

- exact full-trajectory reconstruction is unsafe;
- exact visible-context grouping is safe;
- final trajectory score should not be naively interpreted as causal current-step credit.

Existing analyses in this repository:

- `analysis/alfworld_qwen3_8b_gigpo/score_curve/`
- `analysis/alfworld_qwen3_8b_gigpo/failure_signals/`
- `analysis/alfworld_qwen3_8b_gigpo/token_lengths/`
- `analysis/alfworld_qwen3_8b_gigpo/step1/`

### Best future use

Track failure signals at matched optimizer steps:

```text
protocol error
semantic non-admissibility
no-effect / "Nothing happens"
short loops
long-horizon failure
reasoning/output length
matched-context action disagreement
```

---

## 2. TALE Suite — complete multi-model trajectories

Source: https://huggingface.co/datasets/talesuite/tale_suite_trajectories

ALFWorld subset contains roughly **3.42k full trajectories**. Each row is one episode and exposes fields such as:

```text
transcript_id
model
game
agent_type
total_steps
score / normalized_score
turns
score_progression
thinking_available
episode_token_usage
model_family
seed
```

### Why it is high value

This lets us compare complete behavior across model scales without the episode-identity problem of the 8B training dump.

Candidate analyses:

- same ALFWorld game, different model;
- successful short trajectory vs failed max-horizon trajectory;
- model-scale effect on loops, exploration, invalid/no-effect actions, and goal-progress efficiency;
- trajectory/token efficiency versus success.

### Caveat

These are primarily **evaluation trajectories**, not optimizer-step RL training dumps. Do not interpret differences as training dynamics unless the model/training provenance supports that comparison.

---

## 3. MATM — retrieval / memory intervention comparison

Source: https://huggingface.co/datasets/toeunkim/matm-trajectories

Verified ALFWorld partitions include:

- `alfworld/population_runs`: **2,130** evaluation trajectories from **34 consumer models**, with successes and failures;
- `alfworld/prepopulation`: **7,147** publicly available successful trajectories used as index sources.

The corpus records retrieval conditions including:

```text
no_retrieval
single_stage
rerank_N
```

and trajectory-step fields such as:

```text
action
observation
reasoning
inventory
reward / score
```

### Why it is unusually relevant to Harness

It supports a quasi-Harness comparison:

```text
same / similar model and task
      ├── no retrieval
      └── retrieval / reranking support
```

This can help us study whether external information assistance changes:

- success;
- trajectory length;
- exploration pattern;
- repeated search;
- planning / action-selection errors.

### Caveat

Model identity, retrieval strategy, index composition and other agent settings can be confounded. Use matched subsets whenever possible.

---

## 4. 1.5B transition-pairs — replay and branch analysis

Source: https://huggingface.co/datasets/XinnanZhang/alfworld-transition-pairs-1.5b

Verified summary:

```text
3 runs × 140 valid_seen tasks = 420 episodes
7,954 raw transitions
7,552 deduplicated (state, action, next_state) pairs
```

The dataset retains:

```text
state
action
next_state
reward
done
is_action_valid
action_prefix
n_success / n_failure
min/max steps_to_success
sources: source file + traj_uid + step + run provenance
```

A particularly valuable property is that **185 states are reached with more than one action across the repeated runs**. These are natural branch candidates.

### Why it matters for our credit problem

We can recover / replay a state by resetting the corresponding ALFWorld game and applying `action_prefix`, then test:

```text
same state s_t
   ├── original action
   ├── alternate policy-supported action
   └── Harness-guided action
```

and compare executed outcomes / cost-to-go.

### Important weighting caveat

Failed episodes run to the 50-step horizon more often, so failures contribute more transition rows. Pair-level percentages are **not** episode-level percentages; reweight by episode/task before comparing task classes or model behavior.

---

## 5. auto-dreamer — memory-RL traces

Source: https://huggingface.co/datasets/uyffg/auto-dreamer

Verified ALFWorld layout:

```text
alfworld/
  train/{episodes,steps,add_groups_mixed,trace_index}.jsonl   # 3553 episodes
  test/ {episodes,steps,add_groups,trace_index}.jsonl         # 274 episodes
```

File roles:

- `episodes.jsonl`: one record per trajectory, including task, success, score and step count;
- `steps.jsonl`: one record per `(episode, step)` with observation / action / reward;
- `trace_index.jsonl`: task description + action sequence per episode;
- `add_groups*.jsonl`: memory-writer outputs, including failure diagnoses and procedural / semantic memory inserts.

### Why this is valuable for our Information / Memory Harness

Unlike plain evaluation traces, this corpus also exposes **what memory content was proposed from a trajectory**. It is therefore useful for asking:

- which failures produce useful semantic memory versus procedural memory;
- whether loop / goal-misinterpretation failures are repeatedly diagnosed;
- what information should remain external rather than be internalized;
- how memory/harness content should differ across task types.

### Caveat

Do not confuse the **memory-writer model** with the acting policy. The train memory-writer file mixes multiple writers; acting-policy provenance needs to be checked from the corresponding pipeline/code before making model-scale claims.

---

# Cross-corpus comparison protocol

For future experiments, standardize all corpora into an episode / step schema:

```text
episode_id
source
model
model_scale
training_or_eval
algorithm
history_regime
retrieval_or_harness_condition
task_id
task_type
split
success
raw_reward
reward_definition
num_steps
step_index
observation
action
is_action_valid
next_observation
reasoning_available
prompt_tokens
completion_tokens
trajectory_provenance
```

Then compare **normalized behavioral metrics**, not raw rewards:

1. episode success rate;
2. episode length / timeout rate;
3. invalid or non-admissible action rate;
4. no-effect / `Nothing happens` rate;
5. repeated-state / loop rate;
6. legal-but-nonprogressing action rate;
7. search / exploration efficiency;
8. task/subgoal completion pattern;
9. token cost per successful episode;
10. failure taxonomy conditioned on task type and model scale.

## Never directly compare raw reward values across corpora

Reward scales differ. For example, the Qwen3-8B GiGPO run uses a much larger success reward scale than the Qwen3.5-4B recipe, where success is `1` and each invalid action costs `0.1`. Cross-source comparison should therefore separate:

```text
task success
invalid-action penalty
other shaping terms
```

before drawing conclusions.

---

# Comparison plan for our project

### Phase A — behavior-scale comparison

Use:

```text
1.5B transition pairs
TALE Suite multi-model trajectories
Qwen3-8B GiGPO training dump
```

Question:

> Which failure modes disappear as model capability increases, and which persist?

### Phase B — external-support comparison

Use:

```text
MATM retrieval arms
auto-dreamer memory traces
```

Question:

> Which persistent failures are actually improved by external information / memory assistance?

### Phase C — causal Harness audit

Replay selected states from the 1.5B / our own ALFWorld environment and compare:

```text
plain continuation
vs
Information Harness
vs
Action Harness
vs
Plan Harness
vs
Recovery Harness
```

Estimate:

```text
Δ_H(s_t) = P(success | s_t, H) - P(success | s_t, no-H)
```

This should become the empirical bridge between **observed failure mode** and **actual Harness utility**.

---

## Related local files

- `papers/alfworld-empirical-design-map-2026-09-16.md`
- `notes/alfworld-trajectory-reading-and-harness-analysis.md`
- `analysis/alfworld_qwen3_8b_gigpo/`

## Source policy

This repository stores the **index, analysis and derived statistics**, not large third-party datasets themselves. Pull raw data from the original public source so provenance, licensing and updates remain explicit.
