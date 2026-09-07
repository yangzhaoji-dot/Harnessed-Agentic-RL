# Tail Objectives and Replay in Generative / Agentic RL

This file tracks adjacent RL work that can materially affect diversity-preserving research-idea generation and persistent-trajectory / idea-graph training.

## Tail-Likelihood Reinforcement Learning (TailRL)
- arXiv: https://arxiv.org/abs/2609.02987
- Date: 2026-09-02
- Category: Generative RL; Diversity Preservation; Coverage; Rare-Success Learning; Inference-Time Scaling
- Core idea: standard expected-reward RL can collapse probability mass away from rare but exceptionally high-reward rollouts. TailRL instead maximizes the log-probability of exceeding randomly sampled reward thresholds, effectively optimizing the upper tail of the reward distribution rather than only its mean.
- Implementation: the paper derives a finite-rollout estimator that changes the advantage weighting, making the method compatible with existing policy-gradient / group-based RL pipelines without introducing a critic.
- Empirical scope: object localization, text-maze navigation, GUI grounding, and code optimization. In low-success and shortcut-prone regimes, TailRL can preserve exploration and exploit rare high-reward samples where standard GRPO/RLOO stall or collapse.
- Why it matters for research idea generation: scientific ideation is naturally heavy-tailed — most ideas are mediocre, while rare branches may be unusually novel and high-value. Optimizing mean idea reward can suppress those rare modes. Tail-style objectives provide a principled alternative to simple entropy bonuses for diversity-preserving RL.
- Key research question: should idea-generation RL optimize a tail/coverage objective over idea quality, or a multi-dimensional tail over novelty, feasibility, and significance, rather than only mean scalar reward?
- Code: https://github.com/Zanette-Labs/TailRL
- Priority: **Must track / highly relevant to diversity-preserving RL**.

## Headroom-Drift Replay: A Primitive for Principled Replay Control in GRPO
- arXiv: https://arxiv.org/abs/2609.03941
- Date: 2026-09-03
- Venue: COLM 2026
- Category: GRPO; Replay; Agentic RL; Historical Trajectories; Policy Drift
- Core idea: separates replay selection into (1) **Headroom**, which ranks stored rollout groups by remaining learning value, and (2) **Drift**, which filters groups by compatibility with the current policy. The fresh on-policy stream is left unchanged.
- Empirical scope: mathematical reasoning, multimodal reasoning, and Agentic Search. It improves over naive replay and can match broader replay schemes while reducing rollout cost in agentic search.
- Why it matters for persistent idea graphs: if an idea graph or trajectory memory persists across policy updates, historical branches cannot simply be replayed indiscriminately. Their usefulness depends both on remaining learning value and on whether they are still compatible with the current policy. This complements TIGPO's persistent-graph view with a concrete stale-experience control principle.
- Key research question: can persistent idea-graph nodes be assigned a `headroom × policy-compatibility` score to decide which historical branches should be revisited or used for training?
- Priority: **High adjacent**.
