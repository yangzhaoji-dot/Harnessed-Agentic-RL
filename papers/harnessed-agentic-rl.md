# Harnessed Agentic RL

This category tracks work where the **deploy-time harness participates directly in RL rollout / post-training**.

## Core infrastructure

### Agent Lightning v1.0: Towards Harnessed Agentic RL
- arXiv: https://arxiv.org/abs/2608.17528
- Code: https://github.com/microsoft/agent-lightning
- Model reported: Qwen3.5-9B
- Benchmarks: SWE-bench Verified; instruction-following, search and coding-agent evaluations
- Key abstraction: the **harness owns the environment interaction loop**, while the trainer observes LLM request-response sequences through a proxy.
- Main contribution: practical infrastructure for RL with arbitrary real agent harnesses; studies retokenization, sample merging, advantage calculation, loss normalization and backend scheduling.
- Relation to our direction: strong infrastructure baseline, but the harness is not itself the main evolving optimization object.

### OpenForgeRL: Train Harness-native Agents in Any Environment
- arXiv: https://arxiv.org/abs/2607.21557
- Status: Microsoft Research; listed as ICLR 2027
- Harness: arbitrary stateful / multi-process harnesses, connected through an OpenAI-compatible model-call proxy.
- RL backend: standard training stack such as veRL.
- Systems contribution: Kubernetes orchestration runs each rollout in its own remote container.
- Relation: another important implementation route for **real harness in rollout + standard RL trainer**.
- Gap: harness is normally fixed during policy training; no explicit harness evolution/internalization lifecycle.

## RL algorithms for complex harness trajectories

### Harness-RL: Black-Box Reinforcement Learning with Action-Args Decoupling for Central-Agent Multi-Agent Harnesses
- arXiv: https://arxiv.org/abs/2608.29641
- Venue/status: accepted at PCC 2026 (English version)
- Code: https://github.com/jiangxinke/Harness-RL
- Models: Qwen2.5-1.5B and Qwen2.5-3B
- Harness: central agent coordinating sub-agents, tools and environments; branching, parallel calls and rewritten contexts.
- Method: Interface Call Records + per-session prefix trees to reconstruct trainable trajectories.
- RL: **CAPO (Conflict-Aware Policy Optimization)** decouples action-token and argument-token gradients.
- Benchmarks: seven multi-hop QA / agentic retrieval benchmarks.
- Relation: important if an evolving harness generates non-flat / multi-session trajectories.
- Gap: harness structure is given rather than evolved; internalization is not studied.

## Harness-use / harness-state RL

### EvoHarness-RL: Learning Self-Evolving Runtime Harness for Long-Horizon LLM Agents
- arXiv: https://arxiv.org/abs/2608.05446
- Code: no official GitHub repository confirmed yet
- Model: Qwen3-8B
- Benchmark: ALFWorld
- Training: harness SFT + **cost-aware GRPO**
- Harness: Belief / Progress / Experience (BPE) external state with policy-facing harness actions.
- Key result: 96.9% reported success on ALFWorld seen split and 86.6% unseen.
- Compute reported: 8 NVIDIA H200 GPUs; 150 RL epochs; 128 trajectories per step; vLLM TP=4.
- Key observation: harness usage anneals during GRPO.
- Critical question: usage decay is not by itself causal evidence of capability internalization.

## Harness-policy / skill-policy co-evolution under RL

### SafeEvolve: Harness-Policy Co-Evolution from Agent Experience for Safety Alignment
- arXiv: https://arxiv.org/abs/2609.02786
- Code: https://github.com/MaoPopovich/SafeEvolve
- Model reported: Qwen3.5-4B
- Benchmark: AgentDojo + agentic safety evaluations
- Training: harness-use SFT + harness-augmented RL
- Harness: safety prompt + hierarchical skills; component-level updates derived from on-policy safety experience.
- Relevance: explicit harness-policy co-evolution with SFT+RL; generic "harness-policy co-evolution" is not sufficient novelty by itself.

### ReSkill: Reconciling Skill Creation with Policy Optimization in Agentic RL
- arXiv: https://arxiv.org/abs/2606.01619
- Code: https://github.com/amazon-science/reskill
- Framework: veRL extension
- Models: Qwen3-4B-Instruct-2507 and Qwen3-8B
- RL: **GRPO**
- Skill side: assertion-driven skill creation/revision; within-group version testing; Thompson Sampling; skill add/delete/pruning.
- Benchmarks: ALFWorld, Search, ScienceWorld, InterCode-SQL, WANDS.
- Relevance: major prior on RL-in-the-loop skill-policy co-evolution.
- Critical distinction: a skill being pruned because it no longer improves reward is not automatically proof that its capability was causally internalized into model weights.

### D2Skill: Dynamic Dual-Granularity Skill Bank for Agentic RL
- arXiv: https://arxiv.org/abs/2603.28716
- Venue: EMNLP 2026 Main
- Code: https://github.com/TU2021/D2Skill-AgenticRL
- Models: Qwen2.5-7B-Instruct, Qwen3-4B-Instruct-2507
- Benchmarks: ALFWorld, WebShop
- Skill bank: task-level + step-level skills.
- Key methodology: paired baseline and skill-injected rollouts under the **same policy**, with their performance gap used as a hindsight utility signal for skill-bank updates and policy optimization.
- Relevance: very close prior for paired assisted/unassisted measurement.

## Harness-aware post-training

### TaoLive: Training Agents to Evolve with Their Harness
- arXiv: https://arxiv.org/abs/2608.15763
- Model: compact 35B production model
- Harness: Skills, Hooks, prompts, tools evolve independently of weights.
- Training: Harness-State-Augmented SFT -> General On-Policy Distillation -> Harness-State-Augmented Agentic RL.
- Relevance: industrial evidence that model post-training must explicitly expose changing harness states.
- Gap: harness evolution itself is largely external/developer-driven rather than a learned co-evolution mechanism.

## What to extract when reading code

For each codebase, locate:
1. rollout / environment loop;
2. harness entrypoints and state representation;
3. LLM proxy / request capture layer;
4. trajectory-to-training-sample conversion;
5. reward computation;
6. GRPO/PPO/DAPO/CAPO implementation or veRL config;
7. evaluation with and without harness components;
8. model size, GPU count, rollout count and training steps.
