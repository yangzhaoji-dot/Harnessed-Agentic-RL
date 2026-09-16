# ALFWorld Search Addendum — Closest Novelty Threats Found on 2026-09-16

This addendum records several late-stage hits from the broad web search that are especially close to the current **dynamic heterogeneous Harness** idea. They should be read before we freeze the method.

The main empirical map is `papers/alfworld-empirical-design-map-2026-09-16.md`.

---

## 1. EDGE — Experience-Distillation for Guided Exploration in Agentic Reinforcement Learning

- Paper: https://arxiv.org/abs/2608.21946
- Code: https://github.com/xvolcano02/EDGE
- Venue: EMNLP 2026 Main.
- Benchmarks: ALFWorld, WebShop, Search.

### Core mechanism

EDGE treats retrieved experience as a **temporary training-time scaffold** rather than a permanent inference dependency.

For each rollout group it partitions trajectories into:

```text
experience-conditioned trajectories
experience-free trajectories
```

and uses their matched contrast to admit experiences with positive marginal gain. It then distills the experience-induced behavior into the ordinary policy with a reverse-KL objective. Its experience bank also evolves with the policy: new guidance is synthesized from current failure modes and obsolete entries are pruned.

### Why this is a serious novelty threat

This overlaps several pieces we had independently considered:

- assisted vs unassisted rollouts in one group;
- marginal utility of an external scaffold;
- training-time external support but plain-policy inference;
- dynamic scaffold bank;
- retirement / pruning as the policy improves.

### What EDGE still does not fully answer

The opportunity is to move from a single broad **experience scaffold** to typed Harness capabilities with different causal roles and internalizability:

```text
Information
Action
Plan
Recovery
```

and to estimate **where and why** a component is useful through executed branch interventions rather than only group-level marginal return.

**Priority: P0 / closest novelty threat.**

---

## 2. Agent-G² — Gaussian Guidance for Agentic Reinforcement Learning

- Paper: https://arxiv.org/abs/2608.23318
- Code: https://github.com/ZJU-REAL/Agent-G2
- Project: https://zjureal.com/Agent-G2/
- Venue: EMNLP 2026 Main.
- Benchmarks: ALFWorld, WebShop.

### Core mechanism

Hint-based RL starts a rollout after an expert-trajectory prefix. Agent-G² argues that useful guidance depth is not one deterministic optimum but a **band**. It models task-specific guidance depth with an adaptive Gaussian distribution whose center/spread are updated from rollouts already generated for RL.

The schedule generally moves from deeper guidance toward shallower guidance as the policy becomes stronger.

### Collision with our idea

Agent-G² already demonstrates:

- task-dependent intervention depth;
- online adaptation from rollout statistics;
- assistance reduction as competence rises;
- no separate learned guidance-depth controller required.

Therefore “dynamically decide at which depth to add Harness” is not enough as a standalone contribution.

### Remaining gap

Agent-G² selects **how much expert prefix guidance** to provide. It does not solve typed component diagnosis or state-local causal routing among different Harness capabilities.

**Priority: P0.**

---

## 3. BCSD — Bidirectional Context Self-Distillation for Skill-Based Agents

- Paper: https://arxiv.org/abs/2608.09555
- Code: paper states code will be released; no verified full official implementation at this search snapshot.
- Benchmarks: ALFWorld, WebShop.

### Core mechanism

BCSD scores the same trajectory through two complementary skill-context modifications:

1. an **augmented** view with higher-level Meta-Skill guidance;
2. a **reduced** view that removes general guidance to expose task-specific skill reliance.

The two token-level signals are combined to rescale RL advantage.

### Why useful to us

This is direct evidence that a single “with Harness vs without Harness” contrast may be too coarse. For heterogeneous Harnesses we may need **matched component ablations**:

```text
Full Harness
- Information
- Action
- Plan
- Recovery
```

or positive/negative context views that isolate a component's contribution.

**Priority: P1.**

---

## 4. Agent-R1 — step-native infrastructure relevant to Harness injection

- Code: https://github.com/AgentR1/Agent-R1
- ALFWorld recipe: `examples/alfworld/run_grpo.sh`
- Design doc: `docs/core-concepts/step-level-mdp.md`

Agent-R1 models each environment turn as a first-class MDP step rather than treating the full interaction as one ever-growing token stream.

This matters for us because the environment owns `s_{t+1}` and can legally:

- summarize history;
- replace context;
- inject structured state;
- alter observation representation;
- attach per-step reward.

In other words, Agent-R1 gives a clean engineering abstraction for **Observation / Information Harness injection without pretending that environment text itself is a policy action token**.

**Priority: P1 implementation reference.**

---

## 5. Selective Rollout + T²PO form an important efficiency / trigger baseline pair

### Selective Rollout
- Paper: https://arxiv.org/abs/2605.05802
- Code: https://github.com/zhiyuanZhai20/selective-rollout
- Predicts zero-variance rollout groups from partial action-prefix diversity and stops uninformative groups early.

### T²PO
- Paper: https://arxiv.org/abs/2605.02178
- Code: https://github.com/WillDreamer/T2PO
- Uses uncertainty dynamics for token-level thinking intervention and turn-level resampling.

Together they imply that a Harness trigger should be judged along **two axes**:

```text
expected behavioral benefit
expected information / gradient value
```

A Harness call can improve one trajectory but still be a poor training expenditure if the group provides no useful contrast.

---

# Updated novelty boundary after the broad search

A method consisting only of the following pieces is now heavily occupied:

```text
uncertainty-triggered intervention          → T²PO
adaptive guidance depth                     → Agent-G²
paired assisted/unassisted group rollouts   → EDGE / SKILLC / SIRI family
guidance withdrawal / internalization       → SKILL0 / Guided-OPD / SDAR family
dynamic external experience bank            → EDGE / ReSkill / D2Skill-related work
step-level credit                            → GiGPO / SALT / HCAPO / SHADOW / PGPO
hierarchical plan credit                     → HiPER
future-observation privileged replay         → OCSD
```

The most promising remaining combination is narrower and more empirical:

> **At a candidate state, diagnose which capability is missing; route to a typed Harness; estimate the component's executed counterfactual contribution; learn from assisted/unassisted branches; and retire components independently when their causal marginal value disappears.**

The key novelty should therefore live in **typed causal routing + component-specific internalization/retirement**, not in dynamic triggering alone.
