# Harness / Skill Internalization and Capability Handoff

This category tracks work asking whether an external scaffold can become an **intrinsic model capability** after post-training.

## OPHSD — strongest direct internalization evidence so far

**Training with Harnesses: On-Policy Harness Self-Distillation for Complex Reasoning**
- arXiv: https://arxiv.org/abs/2605.08741
- Code: https://github.com/zzy1127/OPHSD-On-Policy-Harness-Self-Distillation
- Teacher: current model running inside a deterministic stateful harness.
- Student: direct model rollout.
- Training: on-policy self-distillation using the harness-produced terminal context as stronger supervision.
- Evidence: standalone model performance improves; the paper reports that reattaching the harness can provide no further gain or even hurt, supporting real capability transfer.

## EvoHarness-RL — suggestive but not causal evidence

- arXiv: https://arxiv.org/abs/2608.05446
- Observation: harness calls decrease during cost-aware GRPO ("Harness Annealing").
- Problem: usage decay can also be caused by action cost, reward shaping, or strategy substitution.
- Missing key test: evaluate the trained checkpoint under paired **with-component vs without-component** conditions to determine whether the model has actually absorbed the capability.

## Co-Harness

- arXiv: https://arxiv.org/abs/2607.22688
- Improved harness trajectories are fine-tuned back into the model, so scaffolding is intentionally distilled into weights.
- However, the paper is primarily about alternating harness/model optimization rather than detecting the exact moment when a component has become unnecessary.

## Candidate measurement for capability handoff

For model checkpoint `M_t`, full harness `H`, and component `h_i`:

`D_i(t) = R(M_t, H) - R(M_t, H \ h_i)`

A falling call frequency alone is not sufficient. Stronger evidence of internalization requires that:

1. performance without `h_i` improves over training;
2. the paired dependency gap `D_i(t)` approaches zero;
3. the result persists across seeds / held-out tasks;
4. removing `h_i` does not merely induce a different but weaker shortcut policy;
5. ideally, compare against a control policy trained without assistance from `h_i`.

## Current research question

> When has a harness capability actually transferred into an RL-trained policy, and can that transfer be measured well enough to trigger automatic component retirement / boundary movement?

This question must be continuously checked against new work on skill pruning, paired rollouts, external-to-internal memory transfer, and harness-policy co-evolution.
