# Qwen3-8B GiGPO ALFWorld — observable failure-signal evolution

> Public rollout dumps omit `traj_uid/uid/anchor_obs`, so this report only quantifies signals directly identifiable from the dumped model-visible decision rows. It does **not** claim automatic ground-truth labels for planning, memory, or goal-relevance failures.

## Main table — all rows

| train step | rows | high score | protocol invalid | semantic non-admissible* | `Nothing happens` | 2-step state revisit | strong ABA loop | env step >=40 | branch candidates | same-action mixed controls |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5120 | 22.0% | 17.1% | 2.1% | 18.1% | 5.9% | 1.4% | 18.8% | 40 | 68 |
| 20 | 4480 | 22.9% | 0.4% | 1.4% | 1.7% | 2.9% | 1.3% | 17.1% | 18 | 44 |
| 80 | 4544 | 22.4% | 0.0% | 1.9% | 1.9% | 3.6% | 1.7% | 17.9% | 39 | 68 |
| 160 | 3584 | 17.6% | 0.0% | 0.3% | 0.4% | 1.6% | 0.3% | 18.4% | 21 | 31 |
| 300 | 1536 | 73.4% | 0.1% | 0.8% | 0.8% | 0.5% | 0.1% | 6.2% | 0 | 0 |
| 400 | 1920 | 75.8% | 0.0% | 0.3% | 0.2% | 1.0% | 0.1% | 6.4% | 22 | 29 |

\* Semantic non-admissible = parsed action text is not in the prompt's current admissible-action list. This is separate from verl-agent's protocol-validity flag.

## Conditional on low outcome (`score <= 0`)

| train step | low rows | protocol invalid | semantic non-admissible* | `Nothing happens` | 2-step state revisit | repeat previous action | strong ABA loop | env step >=40 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3994 | 19.6% | 2.5% | 20.8% | 6.4% | 0.4% | 1.1% | 22.0% |
| 20 | 3452 | 0.4% | 1.5% | 1.8% | 1.8% | 0.7% | 0.5% | 22.0% |
| 80 | 3525 | 0.0% | 2.3% | 2.3% | 3.5% | 1.6% | 1.5% | 22.0% |
| 160 | 2955 | 0.0% | 0.3% | 0.3% | 1.9% | 1.6% | 0.4% | 22.0% |
| 300 | 409 | 0.0% | 0.0% | 0.0% | 0.2% | 0.0% | 0.0% | 21.5% |
| 400 | 465 | 0.0% | 0.6% | 0.4% | 3.2% | 0.0% | 0.2% | 21.7% |

## Definitions

- **Protocol invalid**: missing `<think>...</think>` or `<action>...</action>` pair, or contains Chinese characters, matching the verl-agent ALFWorld projection validity convention.
- **Semantic non-admissible**: the extracted action string is absent from the current prompt's admissible-action list.
- **2-step state revisit**: current observation text equals the observation from two decisions earlier, a conservative visible-window loop signal.
- **Strong ABA loop**: a 2-step state revisit plus the current sampled action repeats the action from two decisions earlier.
- **Branch candidate**: identical visible prompt context appears with different sampled actions and both high and low eventual scores. It is an intervention candidate, not causal proof.
- **Same-action mixed control**: identical visible context and identical current action can still have high and low eventual scores, demonstrating downstream-confounding of trajectory score.

## What this dump cannot identify reliably

- `history loss / memory failure`: full trajectory identity and pre-window observations are not preserved in the public dump.
- `planning failure` and `goal-affordance failure`: these require replay, a task-state parser, or manual/LLM labeling; they should not be inferred solely from final score.
