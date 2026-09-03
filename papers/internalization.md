# Harness / Skill Internalization and Capability Handoff

This category tracks work asking whether an external scaffold can become an **intrinsic model capability** after post-training.

## OPHSD — strongest direct harness-level internalization evidence

**Training with Harnesses: On-Policy Harness Self-Distillation for Complex Reasoning**
- arXiv: https://arxiv.org/abs/2605.08741
- Code: https://github.com/zzy1127/OPHSD-On-Policy-Harness-Self-Distillation
- Teacher: current model running inside a deterministic stateful harness.
- Student: direct model rollout.
- Training: on-policy self-distillation using the harness-produced terminal context as stronger supervision.
- Evidence: standalone model performance improves; the paper reports that reattaching the harness can provide little/no additional gain or hurt, supporting real capability transfer.

## SKILLC — critical direct prior for RL internalization

**SKILLC: Learning Autonomous Skill Internalization in LLM Agents via Contrastive Credit Assignment**
- arXiv: https://arxiv.org/abs/2605.27899
- Code: no official repository confirmed yet
- Benchmarks: ALFWorld, WebShop
- Core setup: within the **same policy update**, sample paired skill-injected and skill-free rollouts.
- Core method: Contrastive Skill Credit Assignment (CSCA) converts assisted-vs-autonomous performance contrast into a direct learning signal through a dual-stream advantage estimator.
- Curriculum: a validation-level signal adapts credit strength, rollout allocation and monotonic active-skill pruning.
- Evaluation: **runtime skill access is removed**, so autonomous performance is directly measured.
- Reported result: +5.5% on ALFWorld and +4.4% on WebShop over the strongest prior skill-internalization RL baseline.
- Consequence for our novelty: a simple idea of "paired with-component vs without-component rollouts + retire the component when the gap closes" is no longer enough by itself.

## D2Skill — paired utility signal and dynamic pruning

**Dynamic Dual-Granularity Skill Bank for Agentic RL**
- arXiv: https://arxiv.org/abs/2603.28716
- Venue: EMNLP 2026 Main
- Code: https://github.com/TU2021/D2Skill-AgenticRL
- Models: Qwen2.5-7B-Instruct and Qwen3-4B-Instruct-2507
- Benchmarks: ALFWorld, WebShop
- External state: task-level + step-level skill bank.
- Key methodology: paired baseline and skill-injected rollouts under the same policy; performance gap yields a hindsight utility signal used for skill-bank maintenance and policy optimization.
- Lifecycle: skills are expanded, retrieved, updated and pruned.
- Important distinction: **skill utility / pruning is not automatically equivalent to proof of model-side capability transfer**. A skill may become useless because the policy switches strategy or because task distribution changes.

## EvoHarness-RL — suggestive but not causal evidence

- arXiv: https://arxiv.org/abs/2608.05446
- Observation: harness calls decrease during cost-aware GRPO ("Harness Annealing").
- Problem: usage decay can also be caused by action cost, reward shaping, curriculum design, or strategy substitution.
- Missing key test: evaluate trained checkpoints under paired **with-component vs without-component** conditions and/or fully harness-free deployment.

## ReSkill — skill pruning during policy co-evolution

- arXiv: https://arxiv.org/abs/2606.01619
- Code: https://github.com/amazon-science/reskill
- GRPO loop jointly evolves the policy and an external skill library.
- Skills are created, tested, refined, versioned and pruned based on in-loop comparisons.
- Relevance: a component can disappear during RL, but ReSkill is primarily about skill-policy compatibility and utility, not a formal proof that a capability has moved into model weights.

## Co-Harness

- arXiv: https://arxiv.org/abs/2607.22688
- Improved harness trajectories are fine-tuned back into the model, so scaffolding is intentionally distilled into weights.
- However, the paper is primarily about alternating harness/model optimization rather than detecting the exact moment when a component has become unnecessary.

## TaoLive HAT

- arXiv: https://arxiv.org/abs/2608.15763
- Training: Harness-State-Augmented SFT -> General On-Policy Distillation -> Harness-State-Augmented RL.
- Importance: demonstrates a practical reason to train the model under a distribution of changing harness states rather than overfit one fixed harness.
- Not the same as internalization: the goal is robustness/adaptation to harness evolution, not causal retirement of a specific harness capability.

## What remains potentially open

The initial candidate metric

`D_i(t) = R(M_t, H) - R(M_t, H \ h_i)`

is **not sufficient novelty on its own** because D2Skill and especially SKILLC already use paired assisted/unassisted rollouts in closely related ways.

A stronger contribution must answer something these works do not, for example:

1. **General harness components, not only skill prompts** — planner, verifier, memory, routing, sub-agent topology, tool policy, etc.
2. **Causal identification vs local utility** — distinguish true transfer into parameters from cost avoidance, strategy substitution, distribution shift, or redundancy between components.
3. **Temporal / sequential handoff** — identify *when* a capability has transferred with confidence rather than simply using current rollout contrast as a curriculum signal.
4. **Boundary optimization** — optimize which capability should live externally vs internally under performance, training cost and inference cost constraints.
5. **Structural re-evolution after handoff** — after a component is retired, use freed harness capacity to search for a new frontier capability, testing whether improvement compounds across generations.

## Current ICML-level question

> Can we define and estimate a **general model–harness capability boundary** that distinguishes external utility from genuine parameter-side capability transfer, and use it to drive stable handoff / re-evolution across heterogeneous harness components?

This is a substantially harder and more defensible question than simple usage annealing or skill pruning.
