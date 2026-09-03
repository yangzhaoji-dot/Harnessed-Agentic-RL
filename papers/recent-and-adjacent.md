# Recent and Adjacent Papers

This file captures high-priority papers that were discussed during literature tracking but were missing from the first repository import. They should also be reflected in the thematic files and master table.

## Harnessed Agentic RL / RL infrastructure

### OpenForgeRL: Train Harness-native Agents in Any Environment
- Authors: Xiao Yu, Baolin Peng, Ruize Xu, Hao Zou, Qianhui Wu, Hao Cheng, Wenlin Yao, Nikhil Singh, Zhou Yu, Jianfeng Gao
- arXiv: https://arxiv.org/abs/2607.21557
- Date: 2026-07-23
- Venue/status: Microsoft Research; listed as ICLR 2027
- Core idea: an OpenAI-compatible proxy serves model calls made by arbitrary real harnesses and records them for a standard RL backend such as veRL; Kubernetes isolates/scales rollouts.
- Harness: arbitrary stateful / multi-process deploy-time harness; normally fixed while the policy is trained.
- RL: standard RL backend such as veRL; the contribution is infrastructure rather than a new optimizer.
- Relevance: major infrastructure baseline for training a model inside its actual deploy-time harness.
- Gap for our direction: does not optimize the harness or study capability handoff / retirement.

### Harness-RL: Black-Box Reinforcement Learning with Action-Args Decoupling for Central-Agent Multi-Agent Harnesses
- Authors: Xinke Jiang, Zhixin Zhang, Zhibang Yang, Jiaran Gao, Rihong Qiu, Shijin Chen, Xu Chu, Junfeng Zhao, Yasha Wang
- arXiv: https://arxiv.org/abs/2608.29641
- Date: 2026-08-30
- Venue/status: accepted at PCC 2026 (English version)
- Code: https://github.com/jiangxinke/Harness-RL
- Models: Qwen2.5-1.5B, Qwen2.5-3B
- Harness: central agent coordinating sub-agents/tools/environments; dynamic sessions include branching, parallel calls and rewritten contexts.
- RL algorithm: CAPO (Conflict-Aware Policy Optimization), with action-token / argument-token gradient decoupling plus Interface Call Records and prefix-tree trajectory construction.
- Benchmarks: seven multi-hop QA / agentic retrieval benchmarks.
- Relevance: important algorithmic baseline if our future evolving harness produces non-flat trajectories.
- Gap: harness structure itself is not evolved and internalization is not studied.

## Harness generation / optimization / evolution

### JIT-Agent: Scaling Harness Intelligence via Just-in-Time Harness Evolution
- Authors: Guibin Zhang, Leo Lu, Fangzhou Xie, Kang Zhu, Junhao Wang, Zhifei Xie, Zhaochen Yu, Zihang Liu, Zhongxiang Sun, Qiankun Li, Yue Liao, Heng Chang, Xiaobin Hu, Qibing Ren, Wangchunshu Zhou, Shuicheng Yan
- arXiv: https://arxiv.org/abs/2608.25593
- Date: 2026-08-26
- Code: https://github.com/bingreeky/JIT
- Core idea: train a dedicated **Harness Intelligence Model** that synthesizes task-adaptive harnesses on the fly, repairs them, and self-evolves by distilling performance signals from an archive of previous harnesses.
- Harness modules: memory management, planning strategy, action protocol, tool/skill orchestration under a fixed compositional protocol.
- Evaluated serving models include DeepSeek-V4-Flash, GLM-5.2, Qwen-family models and others.
- Benchmarks include DeepSearchQA and OdysseyBench.
- Relevance: strong alternative thesis to internalization — make harness intelligence itself trainable, transferable and compounding.
- Gap: served base model is largely off-the-shelf; no joint policy-harness RL / causal handoff.

### AutoSaddler: Automatic Harness Optimization with Durable Updates from Agent Execution Traces
- Authors: Sungho Park, Wonjoong Kim, Rongyuan Tan, Jue Zhang, Wook-Shin Han, Pengfei Gao, Chanyoung Park, Yongqiang Yao, Rao Fu, Elsie Nallipogu, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang
- arXiv: https://arxiv.org/abs/2608.23041
- Date: 2026-08-24
- Code: https://github.com/microsoft/AutoSaddler
- License: MIT
- Harness search space: prompts, tool definitions/implementations, middleware hooks, agent-loop logic.
- Method: failure-trace diagnosis -> structured Capability/Steering patch generation -> validation-based selection; reflection/evolution are stored in an EvoDAG.
- Benchmarks/results: GAIA2 +9.0 pp, SWE-Bench Pro +9.6 pp, Terminal-Bench 2.0 +10.0 pp over corresponding base harnesses.
- Relevance: strong **structural harness optimization** baseline; more general than memory/skill-only evolution.
- Gap: base model weights remain fixed; no Agentic RL or model-side internalization.

### StarHarness: Evolving Harnesses with Stratified Search for Enterprise Environments
- Authors: Esakkivel Esakkiraja, Denis Akhiyarov, Vikas Yadav, Sai Rajeswar, Patrice Bechard, Sridhar Nemala, Sagar Davasam
- arXiv: https://arxiv.org/abs/2608.24804
- Date: 2026-08-25
- Code: https://github.com/ServiceNow/StarHarness
- Harness search space: task framing/prompts, tool interfaces, skills, MCP providers, subagent structure, agent-loop configuration.
- Model weights: frozen.
- Search design: stratify tasks by baseline failure, separate proposer-visible search tasks from hidden selection tasks, and reserve held-out evaluation tasks.
- Benchmarks: ITBench SRE, EnterpriseOps-Gym ITSM, AutomationBench Finance.
- Reported gain: roughly 20–35 percentage points over default harnesses after 4–12 accepted changes per environment; transfer is tested across GPT/Qwen families.
- Relevance: strong baseline for structural harness evolution and generalization-aware selection.
- Gap: no policy learning / internalization.

### EnvHarness: Awakening Static Worlds for Agent Learning
- Authors: Chengsong Huang, Zifeng Wang, Rujun Han, Jun Yan, Yanfei Chen, Zoey CuiZhu, Ke Jiang, Peng Xia, Han Yu, Yufan Zhuang, Yifei Ming, Jiaqi Pan, Bhavana Dalvi Mishra, Jiaxin Huang, Burak Gokturk, Tomas Pfister, Chen-Yu Lee
- arXiv: https://arxiv.org/abs/2608.19880
- Date: 2026-08-20
- Code: https://github.com/google-research/envharness
- Core idea: apply the harness abstraction to the **environment side**. A programmable component layer reshapes a frozen environment while retaining its verifier; EnvRigger diagnoses policy weaknesses and synthesizes/validates environment-harness components.
- Evaluation: five benchmarks in four domains; reported up to +9.0 points on held-out instances with fewer execution steps.
- Relevance: extends the co-evolution picture from Model <-> Harness to Policy <-> Environment, and provides adaptive RL curricula/signals.
- Gap: not a deploy-time model harness and not model-harness capability handoff.

## Harness-aware post-training

### TaoLive Digital Avatar Agent Technical Report: Training Agents to Evolve with Their Harness
- Authors: TaoLive AIGC LLM Team, Yuhan Sun, Wenhao Lin, Yongdong Luo, Yibo Hu, Meiguang Jin, Junfeng Ma, Weihang Pan, Jiaxin Zhao, Zulong Chen
- arXiv: https://arxiv.org/abs/2608.15763
- Date: 2026-08-16 (later revisions in August)
- Code: no public implementation confirmed
- Harness: Skills, Hooks, system prompts and tools can change independently from model weights.
- Model: compact 35B production model.
- Training pipeline: Harness-State-Augmented SFT -> General On-Policy Distillation -> Harness-State-Augmented Agentic RL.
- Benchmarks/evaluation: Live-Stream QA, Harness-Variant QA, IFEval and production-informed evaluations; complete-agent replay reports latency on one NVIDIA H20.
- Relevance: combines evolving-harness exposure, OPD and Agentic RL in one industrial recipe.
- Gap: harness evolution is primarily an external/developer process; no learned joint handoff/retirement rule.

## Skill-policy co-evolution / internalization (very relevant to our novelty)

### ReSkill: Reconciling Skill Creation with Policy Optimization in Agentic RL
- Authors: Zelin He, Haotian Lin, Boran Han, Wei Zhu, Haoyang Fang, Bernie Wang, Xuan Zhu, Runze Li, Matthew Reimherr
- arXiv: https://arxiv.org/abs/2606.01619
- Date: 2026-06-01
- Code: https://github.com/amazon-science/reskill
- License: Apache-2.0
- Framework: veRL extension.
- Models: Qwen3-4B-Instruct-2507 and Qwen3-8B.
- RL: GRPO.
- Skill evolution: assertion-driven skill creator, within-GRPO-group version comparisons, Thompson Sampling with adaptive discounting; skills can be added, revised, versioned and pruned during training.
- Benchmarks: ALFWorld, Search, plus generalization tests including ScienceWorld, InterCode-SQL and WANDS.
- Relevance: **major collision with generic skill-policy co-evolution and in-loop add/delete claims**.
- Critical distinction to test: utility-based skill pruning is not automatically causal evidence that the model has internalized the skill.

### Dynamic Dual-Granularity Skill Bank for Agentic RL (D2Skill)
- Authors: Songjun Tu, Chengdong Xu, Qichao Zhang, Yaocheng Zhang, Xiangyuan Lan, Linjing Li, Dongbin Zhao
- arXiv: https://arxiv.org/abs/2603.28716
- Date: 2026-03-30
- Venue: EMNLP 2026 Main
- Code: https://github.com/TU2021/D2Skill-AgenticRL
- License: MIT
- Models: Qwen2.5-7B-Instruct and Qwen3-4B-Instruct-2507.
- Benchmarks: ALFWorld, WebShop.
- Skill bank: task-level + step-level skills; continuously expands, retrieves, updates and prunes.
- Key methodological point: **paired baseline and skill-injected rollouts under the same policy**, using their performance gap as a hindsight utility signal for both skill updates and policy optimization.
- Relevance: very close prior to any simple `with component vs without component` measurement idea.
- Gap: paired utility/pruning is not necessarily the same as causal model-side capability transfer.

### SKILLC: Learning Autonomous Skill Internalization in LLM Agents via Contrastive Credit Assignment
- Authors: Hongxiang Lin, Zhirui Kuai, Erpeng Xue, Lei Wang
- arXiv: https://arxiv.org/abs/2605.27899
- Date: 2026-05-27
- Code: no official code confirmed
- Benchmarks: ALFWorld, WebShop.
- Core method: paired skill-injected and skill-free rollouts **inside the same policy update**; Contrastive Skill Credit Assignment injects the task-level contrast into a dual-stream advantage estimator.
- Curriculum: validation-level signal adapts attribution strength, rollout allocation and a monotonic active skill set.
- Evaluation: runtime skill access is removed; the goal is explicitly **autonomous skill internalization**.
- Reported improvement over strongest prior skill-internalization RL baseline: +5.5% on ALFWorld and +4.4% on WebShop.
- Relevance: **critical direct prior**. It substantially narrows novelty for a simple counterfactual assisted/unassisted handoff method.
- Remaining question for our direction: can internalization be generalized beyond skill prompts to heterogeneous harness components, and can the boundary movement be identified more rigorously / temporally rather than through a skill-specific curriculum?

## Takeaway for the current ICML direction

The literature boundary is tighter than the initial six-paper import suggested:

- Arbitrary fixed-harness RL: Agent Lightning / OpenForgeRL.
- Structured multi-agent harness RL algorithms: Harness-RL.
- Runtime state + harness-use RL: EvoHarness-RL.
- Harness structural optimization: CHILL, AutoSaddler, StarHarness, JIT-Agent.
- Harness-policy co-evolution: Co-Harness, SafeEvolve, ReSkill.
- Paired assisted/unassisted skill signals and autonomous internalization: D2Skill and especially SKILLC.

Therefore, novelty cannot be claimed merely as **Harness + RL**, **harness-policy co-evolution**, **paired removal tests**, or **skill retirement**. A stronger ICML contribution must identify a more general or more principled problem than these existing mechanisms.