# Source Code Status

Official / likely-official implementations are tracked here. We do **not** mirror third-party repositories into this repo by default.

| Paper | Repository | Status | Notes |
|---|---|---|---|
| OPHSD | https://github.com/zzy1127/OPHSD-On-Policy-Harness-Self-Distillation | Public | Need deeper source map: harness implementation, reverse-KL objective, configs. |
| Agent Lightning v1.0 | https://github.com/microsoft/agent-lightning | Public, MIT | High-priority code base for real-harness proxy + RL training. |
| OpenForgeRL | official code stated by paper; exact repo URL still to pin | Open-source per paper | Track proxy, Kubernetes orchestrator, veRL integration. |
| Harness-RL | https://github.com/jiangxinke/Harness-RL | Public | CAPO + Interface Call Records + prefix-tree trajectory construction. Default branch is `Harness-RL`. |
| CHILL-Harness | https://github.com/csdstar/CHILL-Harness | Public | Base agent, CF agent, counterfactual planner, learned policy hooks, offline training utilities. |
| SafeEvolve | https://github.com/MaoPopovich/SafeEvolve | Public | Needs source-path extraction for harness evolution and RL stages. |
| JIT-Agent | https://github.com/bingreeky/JIT | Public | Dedicated harness-intelligence model; inspect generation/repair/evolution archive implementation. |
| AutoSaddler | https://github.com/microsoft/AutoSaddler | Public, MIT | V1 reproduces paper experiments; V2 is current durable plugin-based engine. `src/autosaddler/v1/` and `src/autosaddler/v2/`. |
| StarHarness | https://github.com/ServiceNow/StarHarness | Public / very early release | Repository existed at verification time; inspect again as code lands. |
| EnvHarness | https://github.com/google-research/envharness | Public | Environment-side harness; designer/EnvRigger + benchmark bridges/components. |
| ReSkill | https://github.com/amazon-science/reskill | Public, Apache-2.0 | veRL extension. Training entrypoint `scripts/train.py`; configs under `configs/`; environment extras include ALFWorld/Search/ScienceWorld. |
| D2Skill | https://github.com/TU2021/D2Skill-AgenticRL | Public, MIT | ALFWorld/WebShop launchers in `examples_d2skill/`; separate embedding retrieval service. |
| SKILLC | — | No official code confirmed | Critical internalization paper; keep monitoring. |
| TaoLive HAT | — | No public code confirmed | Industrial technical report. |
| EvoHarness-RL | — | No official repo confirmed | Keep monitoring. |
| Co-Harness | — | No official repo confirmed | Keep monitoring. |

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
- GRPO / PPO / DAPO / CAPO / custom optimizer paths
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

## AutoSaddler quick source map

- V1 paper implementation: `src/autosaddler/v1/`
- V2 engine: `src/autosaddler/v2/`
- Configs / split manifests: `configs/`
- Scenario/data scripts: `scripts/`
- Architecture docs: `docs/`

## ReSkill quick source map

- Framework: veRL submodule + ReSkill extension
- Training entrypoint: `scripts/train.py --config-name <env>`
- Environment/data prep: `scripts/data_prep/prepare_<env>.py`
- Configs: `configs/`
- Launch examples: `scripts/launch/`
- Key thing to inspect: where skill versions are assigned within GRPO groups and how add/delete operations are accepted.

## D2Skill quick source map

- ALFWorld training: `examples_d2skill/run_alfworld_d2skill.sh`
- WebShop training: `examples_d2skill/run_webshop_d2skill.sh`
- Skill retrieval service: `examples_d2skill/skill_retrieval_launch.sh`
- Key thing to inspect: paired baseline vs skill-injected rollout allocation and utility-aware pruning.

## Current best reusable code bases for our MVP

1. **Agent Lightning** — simplest conceptual base for arbitrary real-harness RL.
2. **ReSkill** — strongest immediately relevant veRL codebase for GRPO + evolving external skills.
3. **D2Skill** — directly relevant paired assisted/unassisted rollout logic.
4. **Harness-RL** — relevant if our harness produces branching/multi-session trajectories.
5. **AutoSaddler** — structural harness evolution baseline if we add harness-code optimization later.
