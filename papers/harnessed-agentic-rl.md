# Harnessed Agentic RL

This category tracks work where the **deploy-time harness participates directly in RL rollout / post-training**.

## Core papers

### Agent Lightning v1.0: Towards Harnessed Agentic RL
- arXiv: https://arxiv.org/abs/2608.17528
- Code: https://github.com/microsoft/agent-lightning
- Model reported: Qwen3.5-9B
- Benchmarks: SWE-bench Verified; instruction-following, search and coding-agent evaluations
- Key abstraction: the **harness owns the environment interaction loop**, while the trainer observes LLM request-response sequences through a proxy.
- Main contribution: practical and reproducible infrastructure for RL with arbitrary real agent harnesses; studies retokenization, sample merging, advantage calculation, loss normalization and backend scheduling.
- Relation to our direction: strong infrastructure baseline, but the harness is not itself the main evolving optimization object.

### EvoHarness-RL: Learning Self-Evolving Runtime Harness for Long-Horizon LLM Agents
- arXiv: https://arxiv.org/abs/2608.05446
- Code: no official GitHub repository confirmed yet
- Model: Qwen3-8B
- Benchmark: ALFWorld
- Training: harness SFT + **cost-aware GRPO**
- Harness: Belief / Progress / Experience (BPE) external state with policy-facing harness actions.
- Key result: 96.9% reported success on ALFWorld; observes harness usage annealing during GRPO.
- Critical question: usage decay is not by itself causal evidence of capability internalization. A harness-free / component-removal evaluation is needed to distinguish internalization from cost avoidance or strategy substitution.

### SafeEvolve: Harness-Policy Co-Evolution from Agent Experience for Safety Alignment
- arXiv: https://arxiv.org/abs/2609.02786
- Code: https://github.com/MaoPopovich/SafeEvolve
- Model reported: Qwen3.5-4B
- Benchmark: AgentDojo + agentic safety evaluations
- Training: harness-use SFT + harness-augmented RL
- Harness: safety prompt + hierarchical skills; component-level updates derived from on-policy safety experience.
- Relevance: explicit harness-policy co-evolution with SFT+RL. This means a generic claim of "harness-policy co-evolution" is no longer sufficient novelty by itself.

## What to extract when reading code

For each codebase, locate:
1. rollout / environment loop;
2. harness entrypoints and state representation;
3. LLM proxy / request capture layer;
4. trajectory-to-training-sample conversion;
5. reward computation;
6. GRPO/PPO/DAPO implementation or veRL config;
7. evaluation with and without harness components;
8. model size, GPU count, rollout count and training steps.
