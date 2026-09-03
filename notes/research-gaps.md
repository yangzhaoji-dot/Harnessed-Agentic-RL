# Research Gaps / ICML Idea Tracking

This note is deliberately stricter than a general literature summary. It tracks claims that might still support a strong paper and claims that are already occupied by prior work.

## Claims that are no longer sufficient by themselves

- **"Train an agent with its real harness."** — Agent Lightning v1.0 already formalizes harnessed agentic RL.
- **"Optimize the harness and model together."** — Co-Harness already alternates harness and model optimization; SafeEvolve explicitly performs harness-policy co-evolution with SFT+RL.
- **"A harness can be internalized into the model."** — OPHSD provides direct self-distillation and standalone-model evidence.
- **"Harness use decreases during RL."** — EvoHarness-RL reports harness annealing under cost-aware GRPO.
- **"Use counterfactual signals to adapt a harness."** — CHILL-Harness uses paired counterfactual intervention learning for workflow orchestration.

## Candidate gap: measured capability handoff

A narrower and more defensible question is:

> **When has a particular external harness capability actually transferred into the policy, and can measured transfer be used to move the model–harness boundary automatically?**

The important distinction is between **usage** and **dependency**.

- Usage signal: `P(call h_i)`
- Dependency signal: `D_i(t) = R(M_t,H) - R(M_t,H \ h_i)`

A component can be called less often without being internalized. Cost penalties, reward shaping, exploration schedules, or a substitute policy can all reduce usage.

## MVP falsifiable hypotheses

### H1 — Usage annealing is not a reliable proxy for internalization
Across RL checkpoints, there exist components / tasks for which harness usage decreases substantially while paired removal dependency remains non-zero.

### H2 — Paired component-removal evaluation detects handoff more reliably
A checkpointed with-component vs without-component comparison predicts whether permanent component retirement preserves task reward better than call-frequency or training-time heuristics.

### H3 — Different capability classes exhibit different handoff behavior
- procedural reasoning scaffolds (plan / critique / verify / decomposition) may become internalizable;
- live search, calculators, persistent external databases or environment state should remain externally necessary.

A convincing method should discover this distinction rather than hard-code it.

## What would make the paper stronger than a component-pruning heuristic

1. Show a real empirical failure mode of existing usage / cost-based annealing.
2. Define internalization operationally and measure it causally or with a strong paired intervention protocol.
3. Demonstrate automatic handoff / retirement improves or preserves reward while lowering harness cost.
4. Compare against fixed harness, usage-threshold annealing, scheduled annealing, and closest prior methods.
5. Evaluate on multiple models and at least two qualitatively different environments.
6. Separate "model learned the capability" from "model learned a different shortcut" with held-out / transfer tests.

## Longer-term extension: capability ratchet / compounding

Only after the handoff phenomenon is established, consider the larger loop:

`evolve stronger external harness -> RL / post-training internalizes transferable parts -> retire absorbed components -> evolve new scaffolds beyond the improved model frontier`

This would be a genuine **capability ratchet**, but it should not be the first MVP because it confounds harness search quality, RL quality, internalization measurement and boundary control.

## Immediate experimental plan

1. Finish closest-prior reading: EvoHarness-RL and relevant pruning/internalization work.
2. Reproduce or implement a minimal Qwen + ALFWorld harnessed GRPO setup.
3. Instrument per-checkpoint:
   - full-harness reward;
   - component-removed reward;
   - component call frequency;
   - trajectory length / token cost;
   - harness-free or reduced-harness performance.
4. Plot all signals over training and test whether usage and dependency actually diverge.
5. Decide whether the phenomenon is strong enough before building a full method.
