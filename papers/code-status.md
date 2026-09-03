# Source Code Status

Official / likely-official implementations are tracked here. We do **not** mirror third-party repositories into this repo by default.

| Paper | Repository | Status | Notes |
|---|---|---|---|
| OPHSD | https://github.com/zzy1127/OPHSD-On-Policy-Harness-Self-Distillation | Public | Official code link is stated by the paper. Need deeper source map: training entrypoint, harness implementation, reverse-KL objective, configs. |
| Agent Lightning v1.0 | https://github.com/microsoft/agent-lightning | Public, MIT | High-priority code base. Uses a real-harness proxy architecture; current repo provides veRL setup and coding-agent training workflow. |
| CHILL-Harness | https://github.com/csdstar/CHILL-Harness | Public | README exposes base agent, CF agent, counterfactual planner, learned policy hooks, run scripts and offline training utilities. |
| SafeEvolve | https://github.com/MaoPopovich/SafeEvolve | Public | New 2026-09 codebase; needs source-path extraction for harness evolution and RL stages. |
| EvoHarness-RL | — | No official repo confirmed in GitHub search (2026-09-04) | Keep monitoring. |
| Co-Harness | — | No official repo confirmed in GitHub search (2026-09-04) | Keep monitoring. |

## What we record for every released codebase

- license
- key commit / release tag used for our notes
- dependency / environment setup
- model and checkpoint requirements
- training framework (veRL / Ray / TRL / custom)
- harness implementation paths
- environment / benchmark paths
- rollout / trajectory processing paths
- reward computation paths
- GRPO / PPO / DAPO / custom optimizer paths
- SFT / distillation entrypoints
- evaluation scripts, especially harness-free or component-removal evaluation
- rough compute requirements

## CHILL-Harness quick source map

From the public README:
- Base agent: native tool-calling agent for Harbor runs
- CHILL / CF agent: wrapper around the counterfactual planner
- Counterfactual planner: intervention generation / scoring / selection
- Learned policy support: router/change/stop/intervention JSON policies
- Offline policy training: `scripts/train_chill_offline.py`
- Paired-run comparison: `scripts/compare_runs.py`
- CF event summary: `scripts/summarize_cf_events.py`

## Agent Lightning quick source map

High-level architecture:
- Trainer: veRL + vLLM, builds training samples and updates policy
- API Gateway: proxies model requests and captures training data
- Rollout Controller: launches agents locally or as Kubernetes jobs

This is currently the most attractive reusable infrastructure baseline for our own harnessed Agentic RL MVP.
