# 动态 Harness 干预：续查、阅读优先级与设计边界

日期：2026-09-09。延续 [详细阅读地图](policy-adaptive-interventions-2026-09-09.md)。研究方向保持不变：Experience / Planning / Action Filter 是候选操作，训练反馈决定何处、采用何种模式和预算进行干预。本页区分论文机制、工程迁移、待验证的研究假设；未复现训练，不承诺加速比或穷尽性 novelty 检索。

## 1. 新增重点：ARPO 比纯 token 工作更接近我们

[Agentic Reinforced Policy Optimization](https://arxiv.org/html/2507.19849v1)，2025，§3.1–3.2。

ARPO 在固定 rollout 数预算内分配完整轨迹和局部分支，在工具反馈后的高熵阶段增加探索，并研究共享段与分支段的 advantage。它已经覆盖“定位不确定环节—复用前缀—追加分支—训练”的重要部分。它没有因此解决我们的异质 Harness 操作选择。

对我们的实验要求：必须比较同预算的 **无 Harness 分支探索**，不能只对比从初始状态采样的普通 GRPO。否则增益可能来自多次重试，而非干预。

代码核验：本次通过 GitHub 连接读取了 [RUC-NLPIR/ARPO 的 README](https://github.com/RUC-NLPIR/ARPO/blob/main/README.md)，其中同时整理 ARPO 与 AEPO。仅确认公开入口，未测试训练。

## 2. 四轴论文筛选表

| 轴 | 论文 / 原始来源 | 优先借鉴 | 不能直接声称解决了什么 |
|---|---|---|---|
| 位置 | [Beyond the 80/20 Rule](https://arxiv.org/abs/2506.01939), 2025 | 高熵 token 作为候选信号 | 梯度关键位置不等于 Harness 最佳介入点 |
| 位置 | [Critical Tokens Matter](https://arxiv.org/abs/2411.19943), 2024 | rollout/对比估计得到错误相关 token | 不能直接给出哪个组件能挽救状态 |
| 位置 | [CURE](https://arxiv.org/html/2508.11016v2), 2025 | 前缀转 prompt、重采后缀、原始与分支样本管理 | 文本续写不等于恢复 ALFWorld 完整状态 |
| 位置/credit | [ARPO](https://arxiv.org/html/2507.19849v1), 2025 | 工具交互后的分支选择与共享段 advantage | 未替我们定义 Experience/Plan/Filter 的边际学习价值 |
| 组件 | [JIT-Agent](https://arxiv.org/abs/2608.25593), 2026 | 固定接口、有限候选 Harness 模式 | 整个生成 Harness 的 meta-model 不是必要移植对象 |
| 组件 | [AdaPlanner](https://arxiv.org/abs/2305.16653), 2023 | in-plan / out-of-plan refinement、反馈驱动修正 | 推理时规划收益不等于训练收益 |
| 组件/预算 | [MaAS](https://arxiv.org/abs/2502.04180), 2025 | 输入相关、资源约束下的架构选择 | 组件数不是可靠的计算成本度量 |
| credit | [SkillC](https://arxiv.org/html/2605.27899v1), 2026 | assisted/autonomous 双流 advantage 与平滑调度信号 | 不是每一个 Harness 调用的无偏因果分解 |
| credit | [GiGPO](https://arxiv.org/abs/2505.10978), 2025 | episode group 与 anchor-state local group | 同 observation 文本不代表同完整信息状态 |
| credit | [What Does Multi-Harness RL Learn?](https://arxiv.org/abs/2609.04518), 2026 | 同数据比较组内/跨 Harness 分组，固定评估协议 | 不是跨条件分组一概无效的定理 |
| adapt | [CHILL-Harness](https://arxiv.org/html/2607.25825v1), 2026 | 配对延续效果标签、成本与不确定性门槛 | offline/frozen 效果模型不能自动适应持续变化的 policy |
| adapt | [ReSkill](https://arxiv.org/abs/2606.01619), 2026 | 同组版本比较、Thompson Sampling、陈旧证据折扣 | skill 版本竞争不等于已解决位置×模式×预算的选择 |
| adapt/预算 | [SKILL0](https://arxiv.org/abs/2604.02268), 2026 | on-policy helpfulness + 递减预算 | 含视觉上下文，不是 text-only 配置的直接替换 |
| adapt目标 | [Teacher-Student Curriculum Learning](https://arxiv.org/abs/1707.00183), 2017 | learning progress 和遗忘驱动的课程选择 | 还需定义低成本、可归因的 Harness 学习增益信号 |

## 3. 实现与此前讨论的更正

### 3.1 JIT 的周期 summary 不等于覆盖旧 plan

本次直接读了 [planning.py](https://github.com/bingreeky/JIT/blob/main/harness_factory/harnesses/plan_and_execute/planning.py)，blob `685976267fb098b6be736e4486dc84aaf6cc2b58`。初始调用保存 `_current_plan`；默认正整数 8 的倍数触发更新；`update_plan()` 返回 `SummaryState`，没有自己重写 `_current_plan`。因此可借鉴接口，但不能将该 seed 实现描述为完整的事件驱动 plan replacement。

### 3.2 CURE 的代码状态更新

此前详细地图记录 README 请求 404。本次成功读取 [bytedance/CURE 根目录](https://github.com/bytedance/CURE)，可见代码目录，包括 `recipe/`；不能再据 README 404 推断整个仓库不可用。尚未测试其训练配置或确认和论文版本的完全对应关系。

### 3.3 8+8 不是只能分别归一化

ReSkill 本身就比较同一组内的不同 skill 版本。SkillC 使用全局排序与条件内校正，而非仅将两个条件各自归一化。应把不同设计视为目标函数与估计器选择，保存真实 conditioning 和行为 log probability，再实验比较。

一个需先写进单元测试的例子：NoH 的 8 条全 0，H 的 8 条全 1。分别采用纯 outcome-relative GRPO 时，两组 advantage 都为 0；这不表示其他 KL/entropy/辅助项为 0。混合排序能产生对比，却不自动回答收益来自 policy 还是外部辅助。

### 3.4 共享前缀可以保留一次学习

“不重复训练前缀”不等于“任何前缀都不训练”。可先让原始自主轨迹正常更新，只对新增分支中复制的前缀设 context-only / loss mask=0。也可研究聚合子分支回报的共享段 credit。相同前缀出现不同 Monte Carlo 结局本身并非错误；重复计权、干预改变目标才是需要控制的事。

### 3.5 选中失败轨迹以后，旧失败回报不是无偏对照

原始失败用于定位；评估干预效果时，需要从匹配 checkpoint 重新采样 untreated 与 assisted continuation，或另设随机分配/独立估计协议。否则普通无 Harness 重试的成功也会被算作 Harness 功劳。新对照有预算成本，不能假装仍然只需原本的 8+8。

### 3.6 软 shortlist 与硬 action mask 分开

只在 prompt 显示较短列表并不强制模型只能生成其中动作。硬约束需要解码或执行层实现及相应概率记录。不要把执行器改写后的动作，当成原策略实际采样出的 token。

## 4. 三篇近邻：收紧 novelty 表述

- [HarnessBridge: Learnable Bidirectional Controller for LLM Agent Harness](https://arxiv.org/abs/2606.12882)：已经明确使用 observation projection 与 action projection。不能把 Observation/Action 二分当成我们的原创核心。
- [Harness-R1: Learning to Edit Executable Runtime Harnesses from Agent Failure Trajectories](https://arxiv.org/abs/2608.02276)：训练 harness engineer 从失败数据生成可执行补丁。因此“根据失败动态干预”也不能单独构成新颖性。
- [EvoTrainer: Co-Evolving LLM Policies and Training Harnesses for Autonomous Agentic Reinforcement Learning](https://arxiv.org/abs/2606.03108)：研究训练版本级诊断、干预与训练流程演化。需区分这种 training-system harness 与我们轨迹内部的 runtime assistance。

## 5. 可借用、需迁移、需研究

| 问题 | 可借用 | 工程迁移 | 待研究的核心 |
|---|---|---|---|
| 干预位置 | 熵/对比打分、分支采样 | token→完整行动边界；环境与 agent 状态恢复 | 什么位置对特定组件有正收益，而不只是“不确定” |
| 组件模式 | 规划/经验接口与有限模式目录 | Plan/Experience/Filter 统一日志和生命周期 | 持续变化 policy 下的模式、持续期、预算联合选择 |
| 样本分组 | SkillC、GiGPO、ARPO 的参考估计器 | policy/task/prefix/condition ID 与 loss mask | 定义 policy 学习 credit 和 intervention effect 两种不同估计对象 |
| adaptive信号 | CHILL效果标签、ReSkill证据折扣、TSCL学习进展 | 控制器数据与验证数据隔离、计算成本统计 | 如何用廉价即时收益预测真正的 policy-learning gain |

建议的最小 operator 规格是 `(family, mode, parameters, persistence, call_budget)`，另有 `NONE`。这是我们的实验设计建议，不是任何单篇论文已经验证的最佳配置。先一次选择一个操作，后续再测组合交互；预算同时记录 model calls、生成 token、注入 context、env steps、回放和验证费用。

最优前半程位置尚未成立。离线可用完整轨迹作诊断，但需与只见 prefix 的在线 trigger 区分。用剩余预算定义候选窗口；保留随机、低熵或循环检测候选，以免只覆盖犹豫而遗漏自信错误。

## 6. 阅读顺序与产出

先 **ARPO → CURE → SkillC → CHILL → ReSkill**，分别输出：分支数据流、前缀 masking、三种 8+8 estimator 的边界例子、可信配对标签协议、非平稳 selector 与证据更新规则。JIT/AdaPlanner 随实现按需读；MaAS/SKILL0/TSCL补预算与学习目标；三篇 novelty 近邻在确定 claim 前读。

首个可证伪目标：相同计算预算下，针对位置和组件的干预比额外 NoH rollout、无 Harness 分支重试、随机干预、固定 Harness 带来更好的固定部署协议下的 policy 学习。若主张内化，还必须单独评估 held-out tasks 上的无 Harness policy，不能只展示 assisted success 或调用频率下降。
