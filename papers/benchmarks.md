# Benchmarks and Environments

This file tracks benchmark choice for Harness × Agentic RL experiments.

| Benchmark | Environment type | Papers in this repo | Why it matters | Caveats |
|---|---|---|---|---|
| ALFWorld | Long-horizon embodied/text interaction | EvoHarness-RL | Cheap, interpretable action trajectories; directly aligned with harness-use learning; good MVP environment | Saturating / narrow domain; not enough alone for a broad ICML claim |
| SWE-bench Verified | Coding agent | Agent Lightning v1.0; CHILL-Harness | Realistic long-horizon harness, tools, context and code execution; strong deployment relevance | Expensive rollouts; infrastructure complexity |
| GAIA | General tool-use / reasoning agent | CHILL-Harness | Diverse long-horizon orchestration and efficiency evaluation | API-heavy; less controlled for causal component studies |
| Terminal-Bench 2.0 | Terminal / computer-use agent | CHILL-Harness | Explicit cost / workflow / execution structure; good for harness orchestration | Rollout cost and environment setup |
| AgentDojo | Agentic safety / tool use | SafeEvolve | Useful for harness-policy co-evolution under safety/utility tradeoffs | Specialized safety objective rather than general capability |
| HMMT25 | Mathematical reasoning | OPHSD | Clean reasoning task for harness-to-model internalization | Not an interactive agent environment |

## MVP benchmark selection criteria

For capability-handoff experiments, prefer an environment where:

1. the harness can be decomposed into explicit components;
2. each component can be removed or ablated without changing the task definition;
3. rollouts are cheap enough for paired counterfactual evaluation across checkpoints;
4. rewards are reliable and reproducible;
5. the model has non-trivial room to learn;
6. at least one external capability is plausibly internalizable and one is plausibly non-internalizable.

## Current recommendation

- **MVP / falsification:** ALFWorld, because EvoHarness-RL provides the closest reference setting and Qwen-scale models are feasible.
- **Second environment:** choose a genuinely tool-dependent environment (coding / terminal / web) to test whether the method correctly keeps irreducibly external capabilities outside the model.

A strong paper should avoid concluding from a single environment that all harness capabilities behave the same way.
