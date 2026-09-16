# ALFWorld Empirical Design Map — Real Trajectories, Training Dynamics, and Harness-Relevant Methods

> Search snapshot: **2026-09-16**.  
> Scope: public ALFWorld Agentic-RL training traces / trajectory datasets / reproducible training logs, plus methods that can materially affect our Harness design: **where to intervene, what to inject, how to assign credit, and how/when to adapt or retire assistance**.
>
> This file is deliberately narrower than the general literature map. The goal is not to collect every ALFWorld paper; it is to collect resources from which we can learn **how agents actually fail and improve during RL**, then turn those observations into testable Harness mechanisms.

---

## 0. Executive takeaway for our project

The public ecosystem is now rich enough that we should not design Harness components only from intuition.

We can study three complementary evidence types:

1. **Optimizer-step on-policy rollouts** — observe how behavior changes during RL.
2. **Paired / replayable decisions** — estimate where alternative actions actually change outcomes.
3. **Harness / memory / skill ablation arms** — estimate what external assistance still provides after training.

A useful empirical pipeline is therefore:

```text
training-step rollouts
      ↓
behavior / failure taxonomy
      ↓
first-divergence + replayable decision points
      ↓
causal or quasi-causal intervention value
      ↓
Harness component + trigger signal
      ↓
paired with-H / without-H rollouts
      ↓
internalization / retirement test
```

The strongest novelty warning from this search is also clear:

- **T²PO** already does uncertainty-triggered token/turn interventions.
- **Guided-OPD / TCOD / TurnOPD** already adapt assistance or rollout depth over training.
- **SKILL0 / SKILLC / SIRI / Skill0.5** already study skill assistance, internalization and withdrawal.
- **GiGPO / SALT / HiPER / HCAPO / SHADOW / PGPO** already attack step-level or hierarchical credit.

So a contribution framed only as **“dynamic Harness”**, **“intervene at uncertain steps”**, **“withdraw assistance as the model improves”**, or **“paired assisted/unassisted rollouts”** is no longer enough.

The more defensible target is:

> **heterogeneous Harness components (Information / Action / Plan / Recovery), whose intervention and retirement are determined by empirically audited contribution rather than a single heuristic uncertainty score.**

---

# 1. Public ALFWorld data that is actually useful for method design

## 1.1 `wckwan/PRM-agent-rl-artifacts` — highest-priority true training rollouts

- Dataset: https://huggingface.co/datasets/wckwan/PRM-agent-rl-artifacts
- ALFWorld path: `rollouts/alfworld_qwen3_8b_gigpo`
- Type: **true on-policy training rollouts**.
- Format: one JSONL per optimizer step; each row is one sampled trajectory with decoded prompt, response and reward.
- Also contains trainer-free checkpoint evaluation outputs.
- Whole artifact repository is large (~84.7 GB), so we should selectively pull ALFWorld steps rather than clone everything.

### Why this is our most important source

It lets us directly estimate:

```text
P(failure_type | optimizer_step)
P(trigger_signal | success/failure, optimizer_step)
P(Harness_need | optimizer_step)
```

Suggested sample checkpoints:

`early → early-middle → middle → late-middle → late`, e.g. 5–8 evenly spaced optimizer steps.

### Questions to ask

- Which failure modes disappear naturally under RL?
- Which remain persistent late in training?
- At what turn do successful and failed members of the same rollout group first meaningfully diverge?
- Do entropy / repetition / invalid-action / group-disagreement signals rise **before** that divergence or merely after it?
- Are late-training failures dominated by planning/state errors rather than syntax/action-format errors?

**Priority: P0.**

---

## 1.2 `boerz-coding/alfworld-vanilla-grpo` — unusually clean training dynamics

- Repo: https://github.com/boerz-coding/alfworld-vanilla-grpo
- Model: Qwen2.5-7B-Instruct, full-parameter vanilla GRPO.
- ALFWorld version: json_2.1.1.
- Three training runs with machine-readable checkpoint curves.
- `reference/reference_curve.csv`: fixed 140-game valid_seen evaluation curve.
- `reference/telemetry/run_{a,b,c}_telemetry.csv`: per-step training telemetry.
- Telemetry fields include:
  - train / validation success
  - critic score mean
  - KL
  - entropy
  - policy-gradient loss
  - gradient norm
  - **valid-action ratio**
  - episode length
  - response length

### Important empirical clue already visible

In run A, `episode/valid_action_ratio` rises from roughly **0.40 at step 1** to essentially **1.0 by about step 8**.

That is exactly the sort of signal we need: some weaknesses are learned extremely quickly and should probably **not** justify a permanent Harness component.

Candidate interpretation to verify across runs:

> action-format / admissibility support may be mostly an early-training scaffold, while later residual failures are likely to shift toward state tracking, subgoal choice, exploration, or planning.

The repo also reports substantial late-run seed variance, so this resource is useful for studying not only mean learning but **training regime transitions / instability**.

**Priority: P0.**

---

## 1.3 `KMnO4-zx/agentic-rl-lab/08-alfworld` — transparent GRPO-style implementation

- Repo: https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/08-alfworld
- Model: Qwen3.5-4B.
- Setup: 80 updates, 8 games/update, 8 trajectories/game = **5,120 trajectories**.
- Group unit: same ALFWorld game / same initial state.
- Reward: terminal win/loss with invalid-action penalty.
- It explicitly stores per-step rollout internals in the rollout object:
  - prompt tokens
  - assistant completion tokens
  - old logprobs
  - assistant text / parsed action
  - tool observation
  - admissible / done / won
- Evaluation can dump full JSONL trajectories for checkpoints.

### Why useful

This is much smaller and easier to understand than a production veRL stack. It is a good **instrumentation prototype** for our own experiments: if we want to add trajectory dumps, divergence statistics or Harness-event logging, the data model is easy to inspect.

**Priority: P0 for implementation reference.**

---

## 1.4 `XinnanZhang/alfworld-transition-pairs-1.5b` — replay / branching gold mine

- Dataset: https://huggingface.co/datasets/XinnanZhang/alfworld-transition-pairs-1.5b
- Downloadable JSONL: ~36.9 MB.
- 420 episodes from 3 runs on 140 valid_seen tasks.
- 7,954 raw transitions → **7,552 distinct `(state, action, next_state)` pairs**.
- Key fields:
  - `state`, `action`, `next_state`
  - reward / done / action validity
  - action prefix / prefix length
  - `n_success`, `n_failure`, `any_success`
  - min/max `steps_to_success`
  - min/max steps to episode end
  - task type and provenance
- **185 states have multiple distinct actions** across runs: natural branch points.
- Dataset card explicitly warns that failed episodes contribute many more transitions, so analyses must reweight by episode/task rather than naïvely count pairs.

### Why this is unusually valuable for us

This directly supports a **branch-value study**:

```text
same / replayable state s_t
       ├── action a₁ → next state → eventual success
       └── action a₂ → next state → failure / longer cost-to-go
```

We can use these points to test candidate trigger signals and classify what sort of Harness would have helped.

Important caveat: these are repeated-run branch observations, not automatically exact same-policy randomized causal interventions. For a causal claim we should execute replay ourselves.

**Priority: P0.**

---

## 1.5 `inference-sh/skillos-alfworld-eval-arms` — paired assistance / memory ablations

- Dataset: https://huggingface.co/datasets/inference-sh/skillos-alfworld-eval-arms
- Code / report: https://github.com/belt-sh/skillos
- Independent SkillOS reproduction.
- Same 140 valid_seen games are evaluated across many arms/checkpoints.
- Arms are paired by gamefile, enabling paired significance tests such as McNemar.
- Includes no-memory baselines and multiple curator / checkpoint conditions.

### Why useful

This is close to the experimental structure we eventually want:

```text
same task
  ├── without Harness
  └── with Harness / memory / curator
```

It also contains a valuable **negative reproduction result**, reminding us not to infer causal utility from one reported aggregate gain.

**Priority: P0 for paired Harness-effect methodology.**

---

## 1.6 `erv1n/e2l-rwml-alfworld-data` — next-state prediction / world-model signal

- Dataset: https://huggingface.co/datasets/erv1n/e2l-rwml-alfworld-data
- Paper: https://arxiv.org/abs/2602.05842
- Contains RWML ALFWorld prompts and validation transitions collected from Qwen2.5-7B rollouts.
- Includes next-observation targets and calibrated embedding-distance threshold artifacts.

### Harness relevance

This gives us a concrete candidate for **Information Harness need**:

> If the policy badly predicts the consequence of its candidate action, an information/state harness may be useful.

Unlike plain token entropy, this is tied to **environment dynamics**.

**Priority: P1.**

---

## 1.7 `explcre/tcod-v1-alfworld-data` — teacher trajectories and turn-level pairs

- Dataset: https://huggingface.co/datasets/explcre/tcod-v1-alfworld-data
- TCOD paper: https://arxiv.org/abs/2604.24005
- Code: https://github.com/kokolerk/TCOD
- Contains successful teacher trajectories and reconstructed per-turn prompt/response pairs used for temporal-curriculum OPD.

### Use for us

Useful for comparing:

- early vs late turn difficulty,
- teacher support decay with trajectory depth,
- how assistance should expand/withdraw over a horizon.

**Priority: P1.**

---

## 1.8 Large behavior corpora: useful, but not true RL-training dynamics

### TALE Suite trajectories
- https://huggingface.co/datasets/talesuite/tale_suite_trajectories
- ALFWorld subset: ~3.4k trajectories.
- Multiple models; full turns, action/observation history, score progression.

### MATM trajectories
- https://huggingface.co/datasets/toeunkim/matm-trajectories
- ALFWorld population runs: 2,130 trajectories from many consumer models, success + failure.
- ALFWorld prepopulation: 7,147 successful trajectories.

These are useful for building a robust **failure taxonomy** across models, but they should not be confused with optimizer-step on-policy RL data.

**Priority: P2.**

---

# 2. Method map by the four questions our Harness must answer

# 2.1 WHERE should Harness intervene?

## T²PO — uncertainty-triggered token and turn intervention

- Paper: https://arxiv.org/abs/2605.02178
- Code: https://github.com/WillDreamer/T2PO
- ICML 2026 Spotlight.
- Token level: track marginal uncertainty dynamics; trigger a thinking intervention when exploration progress stalls.
- Turn level: dynamically resample interactions with negligible exploration progress.

### Direct collision with us

A generic claim such as **“detect uncertainty and dynamically trigger Harness”** is already occupied.

### What remains open

T²PO primarily answers **whether exploration appears stalled**, not **which external capability is missing**. Our potential distinction is a typed intervention decision:

```text
triggered state
   → information deficit ?
   → action-space deficit ?
   → plan/subgoal deficit ?
   → recovery / reflection deficit ?
```

---

## OCSD — observation-calibrated local support

- Paper: https://arxiv.org/abs/2608.04788
- Code: https://github.com/yiy1x/OCSD
- Compares structurally matched `Full` vs `Observation-Ablated` replay views.
- Uses their score residual to isolate support attributable to future environment observation rather than to replay scaffolding itself.
- Applies the residual at high-uncertainty steps.

### Design lesson

If we add privileged observation / state information, we should not compare a richly reformatted replay prompt against the plain policy and call the whole logprob gap “information value”. We need a **matched control view** that differs only in the information being tested.

This is directly relevant to our earlier concern that an Observation Harness may live outside the action loss.

---

## AgentOPSD — history-dependent pivotal-turn signal

- Paper: https://arxiv.org/abs/2608.05987
- Placeholder code repo: https://github.com/ZethWang/AgentOPSD (README public; full code was still marked forthcoming at search time).
- Aggregates teacher–student token logprob gaps into turn evidence.
- Recursively updates a success-belief state; marginal belief revision becomes turn-level credit.

### Design lesson

A trigger may need to depend on **how the trajectory belief changes over time**, not only a local uncertainty scalar at the current turn.

---

## HCAPO — hindsight post-hoc critic

- Paper: https://arxiv.org/abs/2603.08754
- Uses the LLM itself as a post-hoc hindsight critic to refine step Q estimates, then combines multiple credit scales.

### Harness implication

A separate post-hoc analysis pass can discover candidate critical turns offline. That may be a good way to **bootstrap our trigger dataset without training a Harness controller**.

---

## Critical caution: executed-replay audit

- Paper: https://arxiv.org/abs/2608.19760
- **Credit Without Ground Truth: Auditing Step-Level Credit Assignment in LLM Agents Against Executed Replay**.
- In ALFWorld, the audited LLM-judge, outcome-conditioned logprob-ratio and policy-confidence signals did not reliably identify causal pivotal steps better than matched controls/chance under executed replay.
- Causal contribution was sparse, and whether a usable policy-supported counterfactual even existed depended on the model.

### This should constrain our method

Do **not** select a Harness trigger merely because a signal correlates with failures.

For at least a sampled subset, audit:

```text
candidate trigger score at t
        ↓
resample / replace action at t
        ↓
execute environment forward
        ↓
Δ success probability / Δ cost-to-go
```

This can become a core methodological strength rather than an afterthought.

---

# 2.2 WHAT should Harness inject?

## Plan Harness → HiPER

- Paper: https://arxiv.org/abs/2602.16165
- Code: https://github.com/JonP07/HiPER-agent
- ICML 2026.
- Explicit Plan–Execute hierarchy: high-level subgoal selection + low-level action execution.
- Hierarchical Advantage Estimation assigns credit at both levels.

### What we should steal conceptually

Our `Plan Harness` should be represented as an explicit **subgoal / phase variable**, not a vague “think harder” prompt.

Potential interfaces:

```text
current goal
current subgoal
completion condition
switch / continue decision
```

Then we can measure whether a Harness fixes **subgoal choice** versus **execution under a correct subgoal**.

---

## Information / State Harness → RWML + MemRL

### RWML
- Paper: https://arxiv.org/abs/2602.05842
- Learns action-conditioned next-state/world-model predictions from actual transitions.
- Gives a dynamics-grounded signal for whether the policy understands consequences.

### MemRL
- Paper: https://arxiv.org/abs/2601.03192
- Code: https://github.com/yanan1116/MemRL
- Frozen model + plastic episodic memory; semantic retrieval followed by learned utility/Q-value selection.

### Our distinction

An Information Harness does not have to mean “append more history”. It can provide one of three objects:

1. **state reconstruction** — what is true now;
2. **relevant episodic memory** — what prior experience is useful;
3. **transition forecast** — what will likely happen after candidate action.

These should be separately ablated.

---

## Action Harness → constrained / augmented action support

ALFWorld itself exposes admissible commands, and vanilla-GRPO telemetry suggests invalid-action behavior may be learned rapidly. This implies a useful distinction:

- **format/admissibility filtering** may be an early scaffold;
- **semantic action proposal** (which admissible action is strategically useful) may remain hard.

We should not merge these into one “Action Harness”.

Candidate components:

```text
Action-Validity Filter
Action Candidate Proposer
Action Consequence Preview
Recovery Action Proposer
```

---

## Skill Harness → SKILL0 / Skill0.5 / SIRI / SKILLC / ReSkill / D2Skill

### SKILL0
- Paper: https://arxiv.org/abs/2604.02268
- Code: https://github.com/ZJU-REAL/SkillZero
- Full skill context early, progressively withdrawn via a dynamic curriculum.

### Skill0.5
- Paper: https://arxiv.org/abs/2605.28424
- Code: https://github.com/JasonZhujp/Skill0_5
- Splits skills into general/internalizable vs task-specific/externalizable; uses a difficulty-aware router.

### SIRI
- Paper: https://arxiv.org/abs/2606.02355
- Code: https://github.com/kirito618/SIRI
- Self-mines skills from successful plain rollouts, validates with paired skill/no-skill rollouts, distills beneficial action tokens, then deploys without skill bank.

### SKILLC
- Paper: https://arxiv.org/abs/2605.27899
- Paired skill-injected and skill-free rollouts inside an update; contrast enters credit assignment; active skills are progressively pruned.

### Existing items already in this repository
- ReSkill
- D2Skill

### Novelty consequence

We should treat “skill” as only **one Harness type**. The project becomes more distinct if it asks whether **different component classes have different internalizability**:

```text
information retrieval      → often irreducibly external?
state summarization        → partly internalizable?
planning template          → likely internalizable?
action-format constraints  → quickly internalizable?
external tool capability   → cannot be internalized as factual access?
```

---

# 2.3 HOW should credit be assigned?

## GiGPO — state-group micro advantage

- Paper: https://arxiv.org/abs/2505.10978
- Code: https://github.com/langfengQ/verl-agent
- Episode-level macro relative advantage + step-level micro relative advantage among actions from anchor/repeated states.

**Use in our project:** baseline for step-level credit in ALFWorld.

---

## SALT — trajectory graph credit

- Paper: https://aclanthology.org/2026.findings-eacl.247/
- Builds a graph from same-prompt trajectories and derives step-quality / advantage from outcome-only data.
- Plug-in to group-based RL without changing rollout procedure.

**Use:** relevant if our with/without-Harness branches create a trajectory graph rather than flat independent rollouts.

---

## PGPO — state potential differences, especially inside failures

- Paper: https://arxiv.org/abs/2609.02236
- Estimates empirical state potentials from anchor-state-group return statistics.
- Action credit comes from potential differences between adjacent states.
- Explicitly targets the problem that useful actions inside failed trajectories otherwise inherit negative terminal credit.

**Use:** strong baseline for our belief that a failed Harness branch may still contain locally valuable decisions.

---

## HiPER — hierarchical planning/execution credit

See §2.2. If we make Plan Harness an explicit hierarchy, the credit estimator should respect the same hierarchy.

---

## SHADOW — transition-dynamics-aware credit

- AAAI 2026: https://ojs.aaai.org/index.php/AAAI/article/view/39570
- Dynamics-aware state grouping + local dynamic advantage estimation.

**Use:** reinforces the idea that state similarity should be defined by **environment-transition compatibility**, not superficial text similarity.

---

## HCAPO — hindsight Q refinement

See §2.1. Valuable as an offline critical-step labeler even if we do not use it inside the final optimizer.

---

## Q-Evolve — critic + process reward co-evolution

- Paper: https://arxiv.org/abs/2606.07367
- Code: https://github.com/QEvolve/q-evolve
- Learns an in-distribution critic from expert + agent-generated trajectories.
- Derives step-wise process rewards / advantages and then performs behavior-proximal policy optimization.

**Use:** useful contrast to our likely critic-free first version; also demonstrates that process supervision and the policy can co-evolve.

---

## AgentPRM — Monte-Carlo process reward model

- Paper: https://arxiv.org/abs/2502.10325
- Code: https://github.com/sanjibanc/agent_prm
- Monte-Carlo rollout targets for an agent process reward model.

**Use:** canonical “train a separate PRM” baseline, but computationally and operationally heavier than our preferred first MVP.

---

## GRSD — verified-group reflection as turn credit

- Paper: https://arxiv.org/abs/2607.28076
- Code: https://github.com/BinbZheng1/GRSD
- Policy reflects on its own verified success/failure rollouts, constructs group-level `DO/AVOID` privileged guidance, then uses it to modulate turn-level credit.

**Use:** very relevant if we use successful-vs-failed rollout contrast to construct a temporary Harness.

---

# 2.4 WHEN should assistance change or disappear?

## Guided-OPD — intervention probability decays to zero

- Paper: https://arxiv.org/abs/2606.15912
- Mixes teacher and student turns in the same rollout.
- Teacher-intervention probability follows a curriculum and decays to zero.

### Direct lesson

“Strong assistance early, withdraw later” is already an explicit algorithmic pattern. Our retirement rule therefore needs to be **component-specific and evidence-based**, not merely a global time schedule.

---

## TCOD — progressively expose longer trajectory depth

- Paper: https://arxiv.org/abs/2604.24005
- Code: https://github.com/kokolerk/TCOD
- Diagnoses trajectory-level KL instability caused by compounding errors and progressively expands student exposure from shorter to longer horizons.

### Lesson

Harness scheduling may need a **trajectory-depth curriculum**, not only a training-step curriculum.

---

## TurnOPD — adaptive rollout-depth and turn-normalized loss budgets

- Paper: https://arxiv.org/abs/2607.05804
- Learns / probes how deep rollouts need to go and reallocates supervision across turns.

### Lesson

The “intervention budget” and “loss budget” can be separate controls.

---

## ATOD — anneal distillation toward RL + turn disagreement/uncertainty weighting

- Paper: https://arxiv.org/abs/2606.27814
- Code: https://github.com/TanQitai/ATOD
- Smooth OPD→GRPO annealing with turn-aware disagreement/uncertainty reweighting.

### Lesson

A useful control variable is not only **whether Harness exists**, but **how much of the policy update is allowed to come from Harness-induced guidance**.

---

## Selective Rollout — stop zero-information groups mid-trajectory

- Paper: https://arxiv.org/abs/2605.05802
- Code: https://github.com/zhiyuanZhai20/selective-rollout
- Uses partial action-prefix divergence to predict groups likely to end with zero reward variance and stops them early.
- On ALFWorld Qwen2.5-7B, reports lower wall-clock training and reduced zero-advantage gradient dilution.

### Lesson

Our dynamic Harness should be **budget aware**. If a rollout group already contains no useful counterfactual diversity, spending extra Harness calls on it may be wasteful.

---

## SDAR / OPID / SEED family — privileged guidance without inference-time dependence

### SDAR
- Paper: https://arxiv.org/abs/2605.15155
- Code: https://github.com/ZJU-REAL/SDAR
- Uses self-distillation as a gated auxiliary objective while keeping RL as the backbone.

### OPID
- Paper: https://arxiv.org/abs/2606.26790
- Hindsight skills re-score the same sampled actions; logprob shifts supply dense self-distillation signal.

### SEED
- Project: https://jinyangwu.github.io/seed/
- Code: https://github.com/jinyangwu/SEED
- Same policy acts, analyzes completed trajectories into hindsight skills, and distills the skill-induced change back into the ordinary policy.

### Lesson

For Harness components that *can* be internalized, a clean pattern is:

```text
ordinary on-policy trajectory
       ↓
privileged / Harness replay view
       ↓
measure policy change under Harness
       ↓
use change as auxiliary training signal
       ↓
deploy plain policy
```

But our causal audit must distinguish “Harness changed logprobs” from “Harness actually changes environment outcome”.

---

# 3. A concrete empirical study we can run before finalizing the method

## Study A — failure modes over training

Data:
- `PRM-agent-rl-artifacts`
- vanilla-GRPO telemetry/checkpoints

Label each failed trajectory with a compact taxonomy:

```text
A. action-format / invalid action
B. object/state tracking failure
C. exploration/search failure
D. wrong subgoal / ordering
E. execution failure under correct plan
F. repetition / loop / stuck
G. premature termination / timeout
H. other / ambiguous
```

Estimate per checkpoint:

```text
failure-rate(type)
avg first-error turn
avg episode length
recovery probability after first error
```

**Goal:** identify which failures RL naturally internalizes and which persist.

---

## Study B — first-divergence analysis within rollout groups

For same-task rollout groups:

1. align histories until the first semantically different action;
2. classify the divergence;
3. compare eventual success and cost-to-go;
4. record local signals at the pre-divergence state.

Candidate signals:

- token / action entropy
- top-1 vs top-2 action margin
- group action disagreement
- repeated-action / loop score
- action validity
- next-state prediction error
- progress / state-potential estimate
- plan/subgoal consistency

**Goal:** determine whether any signal predicts *pivotal* turns, not merely failed turns.

---

## Study C — executed intervention audit

Use replayable ALFWorld states from `alfworld-transition-pairs-1.5b` plus our own environment replay.

At sampled state `s_t`:

```text
plain policy action(s)
Harness-assisted action(s)
random / matched-control action(s)
```

roll each forward and estimate:

```text
ΔP(success)
Δsteps-to-success
Δinvalid-action count
Δloop probability
```

This gives an empirical Harness contribution target:

`C_H(s_t, H_type)`.

This can serve as evaluation even if we do **not** train a Harness controller.

---

## Study D — component-specific internalization curves

For each Harness type `H ∈ {Information, Action, Plan, Recovery}` and checkpoint `k`:

```text
U_H(k) = E[R(with H) - R(without H)]
```

Also track cost:

```text
Cost_H = extra tokens + extra calls + rollout latency
```

A component becomes a retirement candidate when its marginal utility becomes small **and stays small on held-out tasks**.

Crucially, retirement should not be globally monotonic by assumption: a component may become unnecessary on easy task families while remaining useful on harder / OOD families.

---

# 4. What this changes in our current Harness design

Our initial `Information / Action / Plan` split is still useful, but it should become more operational.

## Information Harness

Possible units:
- state reconstruction
- episodic-memory retrieval
- missing-object/location summary
- action-consequence / next-state forecast

Trigger candidates:
- state contradiction
- world-model prediction error
- repeated exploration with no new information

## Action Harness

Split into:
- validity / syntax filter
- candidate action proposer
- semantic ranking / consequence-aware selector
- recovery action proposer

Do **not** assume validity filtering deserves a permanent role; public GRPO telemetry suggests admissibility can be learned very early.

## Plan Harness

Represent explicitly:
- current subgoal
- subgoal completion condition
- continue/switch decision
- dependency / ordering constraints

This lets us distinguish plan errors from executor errors and makes hierarchical credit possible.

## Recovery Harness — add as a fourth component

The trajectory datasets repeatedly motivate a separate recovery capability:

- detect loops / no progress
- summarize what has already been tried
- propose a state-resetting or exploration-changing action
- optionally re-plan

This should not be conflated with ordinary Plan Harness because it is conditional on **trajectory degeneration**.

---

# 5. Strongest current candidate thesis

A more defensible method direction after this search is:

> **Causally Audited Adaptive Harnessing for Agentic RL**: use real on-policy group trajectories to identify candidate intervention points, classify the missing capability into heterogeneous Harness types, audit trigger/component utility through executed counterfactual replay, and use assisted–unassisted contrast for policy learning and component-specific retirement.

Possible decomposition:

```text
1. Candidate event detector
   cheap signals only; no trained controller initially

2. Harness router
   rule-based typed diagnosis for MVP:
   Information / Action / Plan / Recovery

3. Paired branch
   plain continuation vs Harness continuation

4. Executed utility
   environment outcome / progress difference

5. Credit
   trajectory advantage + local branch contribution

6. Internalization test
   periodically rerun matched tasks without Harness

7. Retirement
   per-component, per-task-family utility threshold
```

### What is *not* enough anymore

- entropy threshold alone;
- fixed mid-trajectory branch alone;
- one global “Harness on/off” variable;
- linear decay schedule alone;
- LLM judge saying a step is important;
- skill-only internalization;
- paired rollouts without an executed contribution audit.

---

# 6. Immediate source-code shortlist

If we only inspect code that can materially affect our MVP, use this order:

1. **verl-agent / GiGPO** — base ALFWorld group rollout + step grouping  
   https://github.com/langfengQ/verl-agent
2. **T²PO** — dynamic intervention / resampling signal  
   https://github.com/WillDreamer/T2PO
3. **HiPER** — explicit plan/execution hierarchy  
   https://github.com/JonP07/HiPER-agent
4. **OCSD** — matched privileged-context controls  
   https://github.com/yiy1x/OCSD
5. **SKILL0 / SDAR** — withdrawal + internalization + privileged self-distillation  
   https://github.com/ZJU-REAL/SkillZero  
   https://github.com/ZJU-REAL/SDAR
6. **SIRI** — self-mined skills + paired validation + action-level distillation  
   https://github.com/kirito618/SIRI
7. **GRSD** — same-group success/failure reflection → turn credit  
   https://github.com/BinbZheng1/GRSD
8. **Selective Rollout** — dynamic compute gating in grouped rollout  
   https://github.com/zhiyuanZhai20/selective-rollout
9. **Q-Evolve** — critic/process-reward alternative  
   https://github.com/QEvolve/q-evolve
10. **vanilla GRPO reproducibility repo** — clean instrumentation + telemetry  
    https://github.com/boerz-coding/alfworld-vanilla-grpo

---

# 7. Data shortlist for the first trajectory study

Do **not** download everything initially.

### Minimal useful bundle

1. `XinnanZhang/alfworld-transition-pairs-1.5b` — only ~36.9 MB, start here.
2. `boerz-coding/alfworld-vanilla-grpo/reference/` — tiny curves + telemetry.
3. A sparse subset of `wckwan/PRM-agent-rl-artifacts/rollouts/alfworld_qwen3_8b_gigpo` at selected optimizer steps.
4. Two or three paired arms from `skillos-alfworld-eval-arms`.
5. RWML validation transitions if we test an Information/world-model signal.

This is enough to answer several method questions before spending our own GPU budget.

---

# 8. Evidence-strength labels used in this file

- **True training rollout**: sampled during policy optimization.
- **Checkpoint eval trajectory**: generated by a trained checkpoint, but not necessarily used for that checkpoint's update.
- **Transition/replay dataset**: state-action-transition records useful for counterfactual or dynamics analysis.
- **Teacher/offline trajectory**: useful for behavior analysis or distillation but not evidence of on-policy training dynamics.
- **Paper aggregate only**: method/result source; cannot by itself support detailed behavioral claims.

When we later write the paper motivation, these categories should remain separate.

---

# 9. Next repository task

The next useful step is not another giant literature list. It is to create an analysis script that normalizes trajectories into a common schema:

```python
{
  "run_id": ...,            # source / training run
  "checkpoint": ...,        # optimizer step if available
  "task_id": ...,
  "task_type": ...,
  "group_id": ...,
  "trajectory_id": ...,
  "step": ...,
  "observation": ...,
  "action": ...,
  "next_observation": ...,
  "valid": ...,
  "reward": ...,
  "done": ...,
  "success": ...,
  "logprob": ...,           # if available
  "entropy": ...,           # if available
  "harness_type": None,
  "harness_payload": None,
  "source": ...
}
```

Once the data is normalized, our first plots should be:

1. failure type vs training step;
2. valid-action ratio vs training step;
3. episode length vs success across checkpoints;
4. first-divergence turn distributions;
5. candidate trigger signal vs executed branch contribution;
6. with/without-Harness utility by component and checkpoint.

That empirical layer should constrain the algorithm, rather than fitting evidence to a pre-selected Harness design.
