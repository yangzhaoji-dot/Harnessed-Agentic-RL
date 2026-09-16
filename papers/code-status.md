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

## ALFWorld empirical / algorithmic codebases added 2026-09-16

These are tracked because they expose implementation patterns or public data that can constrain our Harness design, even when the paper is not itself a Harness paper.

| Method / resource | Repository / artifact | Status | What to inspect for our project |
|---|---|---|---|
| GiGPO / verl-agent | https://github.com/langfengQ/verl-agent | Public | ALFWorld group rollout, anchor-state grouping, macro + micro advantage, environment instrumentation. |
| T²PO | https://github.com/WillDreamer/T2PO | Public, Apache-2.0 | Token uncertainty dynamics, intervention trigger, turn resampling, tracing hooks. |
| HiPER | https://github.com/JonP07/HiPER-agent | Public, Apache-2.0 | Explicit Plan–Execute interface, subgoal boundaries, Hierarchical Advantage Estimation. |
| OCSD | https://github.com/yiy1x/OCSD | Public | Matched Full vs Observation-Ablated replay views; high-uncertainty local calibration. |
| SDAR | https://github.com/ZJU-REAL/SDAR | Public, Apache-2.0 | Privileged self-distillation gate, GRPO backbone, ALFWorld skill-conditioned guidance. |
| SKILL0 | https://github.com/ZJU-REAL/SkillZero | Public | Dynamic skill curriculum and progressive withdrawal / internalization. |
| SIRI | https://github.com/kirito618/SIRI | Public | Self-skill mining, paired skill/free validation, action-token distillation. |
| Skill0.5 | https://github.com/JasonZhujp/Skill0_5 | Public | Difficulty-aware router; general-skill internalization vs task-specific external utilization. |
| SkillRise | https://github.com/Within-yao/SkillRise | Public | Solve/Curate alternation; cross-task skill-document evolution and decoupled credit. |
| GRSD | https://github.com/BinbZheng1/GRSD | Public, Apache-2.0 | Same-group success/failure reflection, DO/AVOID privileged context, turn-level credit modulation. |
| Q-Evolve | https://github.com/QEvolve/q-evolve | Public, MIT | In-distribution critic, process reward labeling, policy/critic/data co-evolution. |
| TCOD | https://github.com/kokolerk/TCOD | Public | Trajectory-depth curriculum for multi-turn OPD; ALFWorld workflows and data builders. |
| ATOD | https://github.com/TanQitai/ATOD | Public, Apache-2.0 | OPD→GRPO annealing and turn-level disagreement/uncertainty reweighting. |
| SPEAR | https://github.com/TencentYoutuResearch/SPEAR | Public | Progressive exploration / self-imitation; GRPO and GiGPO ALFWorld launchers. |
| Selective Rollout | https://github.com/zhiyuanZhai20/selective-rollout | Public | Mid-trajectory group gate; partial-prefix diversity and zero-advantage compute prediction. |
| AgentPRM | https://github.com/sanjibanc/agent_prm | Public | Monte-Carlo process reward targets, exploration, process reward shaping, ALFWorld PRM baseline. |
| MemRL | https://github.com/yanan1116/MemRL | Public | Utility/Q-value-based episodic memory selection; useful Information-Harness baseline. |
| vanilla ALFWorld GRPO reproducibility | https://github.com/boerz-coding/alfworld-vanilla-grpo | Public, Apache-2.0 | Reference curves, three-run telemetry, valid-action ratio, prompt protocol, fixed eval set. |
| small transparent GRPO lab | https://github.com/KMnO4-zx/agentic-rl-lab/tree/main/08-alfworld | Public | Easy-to-read group rollout object, per-step tokens/logprobs/actions/observations and PPO update. |
| SkillOS independent reproduction | https://github.com/belt-sh/skillos | Public | Paired ALFWorld eval arms, negative-result audit, training/eval reproducibility. |

### Public trajectory / transition artifacts

| Artifact | URL | Evidence type | Immediate use |
|---|---|---|---|
| PRM-agent RL artifacts | https://huggingface.co/datasets/wckwan/PRM-agent-rl-artifacts | True on-policy training rollouts | Behavioral evolution across optimizer steps. |
| ALFWorld transition pairs | https://huggingface.co/datasets/XinnanZhang/alfworld-transition-pairs-1.5b | Replayable `(state, action, next_state)` transitions | Branch points, action validity, cost-to-go, replay audit. |
| SkillOS paired eval arms | https://huggingface.co/datasets/inference-sh/skillos-alfworld-eval-arms | Paired checkpoint / memory eval trajectories | With/without assistance matched-task comparisons. |
| RWML ALFWorld data | https://huggingface.co/datasets/erv1n/e2l-rwml-alfworld-data | Next-state/world-model transitions | Information-Harness / dynamics-error signal. |
| TCOD ALFWorld data | https://huggingface.co/datasets/explcre/tcod-v1-alfworld-data | Teacher rollouts + turn-level pairs | Turn-depth / assistance curriculum analysis. |
| TALE Suite trajectories | https://huggingface.co/datasets/talesuite/tale_suite_trajectories | Multi-model eval trajectories | Cross-model failure taxonomy. |
| MATM trajectories | https://huggingface.co/datasets/toeunkim/matm-trajectories | Multi-model success/failure trajectories | Memory/retrieval behavior and broad trajectory statistics. |

See `papers/alfworld-empirical-design-map-2026-09-16.md` for the detailed design synthesis and proposed trajectory studies.

## Current best reusable code bases for our MVP

For the immediate **ALFWorld dynamic heterogeneous Harness** prototype, the highest-leverage stack is now:

1. **verl-agent / GiGPO** — base group-rollout and step-credit infrastructure.
2. **T²PO** — strongest code reference for online trigger / resampling logic.
3. **HiPER** — explicit Plan Harness representation and hierarchical credit.
4. **OCSD** — matched privileged-context control for measuring information contribution.
5. **SKILL0 / SDAR / SIRI** — internalization, paired skill effects, privileged guidance and retirement patterns.
6. **GRSD** — successful-vs-failed group contrast into turn credit.
7. **Selective Rollout** — budget-aware mid-trajectory gating.
8. **ReSkill / D2Skill** — evolving external skills and paired assisted/unassisted rollout allocation.
9. **Harness-RL** — branching / multi-session trajectory baseline when Harness creates counterfactual branches.
10. **vanilla ALFWorld GRPO reproducibility repo** — clean telemetry and instrumentation reference for our empirical study.
