# Credit Assignment, Process Reward, and Diversity

This thematic tracker covers work that can directly shape **Agentic RL**, **research idea generation**, **idea graphs**, **process-level reward**, **credit assignment**, and **diversity-preserving RL**. Papers here may be adjacent to harness optimization rather than explicit harness papers, but they are tracked because they can change the design space for Harness × Agentic RL.

## 2026-09-02 / 2026-09-03 high-priority additions

### Coverage, Not Targeting: A Structural Regime in Multi-Turn Agent Credit Assignment
- Authors: Chenyu Zhou, Qiliang Jiang, Shuning Wu, Xu Zhou
- arXiv: https://arxiv.org/abs/2609.02417
- Date: 2026-09-02
- Category: Agentic RL; Credit Assignment; Process Reward; Reward Coverage
- Core idea: challenges the common assumption that better credit assignment means concentrating reward on a few “important” turns. It introduces verifier information density `V_d = k/C` and argues that, when the verifier exposes only a small fraction of the true causal chain, broad reward coverage can matter more than precise targeting.
- Benchmarks: tau^2-bench; BFCL V3; additional cross-model experiments reported on ToolACE-2-8B.
- Reward / credit design: compares sparse terminal reward, uniform dense redistribution, progress-targeted redistribution, random matched-concentration controls, and reward-to-go variants.
- Model scale: multiple model families; exact complete scale table should be extracted from the paper when we do a full read.
- Why it matters for our direction: an important counterpoint to node/turn-level credit methods. For an idea graph, the correct objective may sometimes be **coverage over the whole hypothesis→critique→revision→verification chain**, not just identifying a single decisive node.
- Key research question opened: when should a Harness controller target critical turns, and when should it instead ensure broad learning-signal coverage across the trajectory?
- Priority: **Must read**.

### PGPO: Potential-Guided Policy Optimization for Multi-Turn Agentic Tasks
- Authors: Yuyao Zheng, Haipeng Sun, Junwei Bao, Lemao Liu, Hongfei Jiang, Yang Song, Dejing Dou
- arXiv: https://arxiv.org/abs/2609.02236
- Date: 2026-09-02
- Category: Agentic RL; Credit Assignment; Process Reward; Group-Based RL
- Core idea: estimates empirical state potentials from anchor-state-group return statistics and derives step/action advantages from potential differences between adjacent states. This enables **cross-trajectory credit propagation** rather than assigning all intermediate actions the same terminal outcome signal.
- RL algorithm: PGPO (Potential-Guided Policy Optimization), positioned against group-based methods such as GRPO/GiGPO.
- Benchmarks: ALFWorld; WebShop.
- Reward / credit design: state-potential difference provides fine-grained failure-side credit; useful local actions inside globally failed trajectories can receive more favorable credit.
- Model / scale: exact backbone/scale should be pinned from the full experimental section during detailed reading.
- Why it matters for our direction: particularly relevant to research idea generation, where a rejected final idea may still contain a valuable sub-hypothesis, evidence link, or graph branch that should not inherit the entire trajectory’s negative outcome.
- Key research question opened: can state/branch potential be defined over an **idea graph** so that partial scientific insights survive even when the final proposal fails?
- Priority: **Must read**.

### CHIME: Credit-Aware Hierarchical Memory Evolution for Long-Horizon Agentic Planning
- Authors: Yongshi Ye, Tian Lan, Feihu Jiang, Muyang Ye, Bin Zhu, Qianghuai Jia, Longyue Wang, Zhao Xu, Weihua Luo, Xiaodong Shi
- arXiv: https://arxiv.org/abs/2609.02074
- Date: 2026-09-02
- Category: Agent Memory; Credit Assignment; Long-Horizon Planning; Self-Evolution
- Code: announced for https://github.com/ATH-MaaS/Marco-DeepResearch
- Core idea: follows an **attribute-before-memorize** principle. It maintains separate planning and execution memory banks, attributes the final outcome to plan quality, execution quality, both, or neither, and updates only the relevant bank.
- Training: self-evolving external memory; no parameter update is required for the core mechanism.
- Benchmarks: four long-horizon agent benchmarks (full benchmark names to pin during detailed reading).
- Reward / credit design: decomposes outcome attribution before memory update instead of indiscriminately treating the whole trace as good/bad experience.
- Why it matters for our direction: highly transferable to Harness / idea-graph evolution. A failed research trajectory should first be attributed to hypothesis quality, retrieval/evidence, experimental design, execution, or evaluation before changing the corresponding component.
- Key research question opened: can Harness adaptation use **factorized attribution** before editing memory, planner, search policy, verifier, or graph structure?
- Priority: **High adjacent**.

### Do Large Language Models Capture the Diversity in their Training Data?
- Authors: Youqi Wu, Farzan Farnia
- arXiv: https://arxiv.org/abs/2609.02275
- Date: 2026-09-02
- Category: Diversity; Generative Modeling; Information Theory; Diversity-Preserving Post-Training
- Core idea: measures conditional output diversity using conditional entropy and a matrix-based von Neumann entropy analogue. Across several model families, generated outputs are systematically less diverse than the corresponding training data.
- Models / data: OLMo; Pythia; GPT-Neo; the analysis is also extended beyond language modeling to ImageNet and MS-COCO generative settings.
- Method: post-hoc entropy-constrained projection reweights multiple outputs to increase conditional diversity while remaining close to the original model distribution.
- RL status: not an Agentic RL paper; tracked because it offers a stronger diversity formalism than token entropy / distinct-n.
- Why it matters for our direction: research idea generation needs **semantic / conditional diversity**, not merely high token entropy. This suggests measuring diversity over idea embeddings, verified solution classes, or graph branches and potentially imposing a diversity constraint during RL.
- Key research question opened: can diversity-preserving Agentic RL optimize task reward subject to an information-theoretic minimum-diversity constraint over idea trajectories or graph branches?
- Priority: **High adjacent**.

## Synthesis for research idea generation

These papers expose a useful tension:

1. **PGPO:** make credit more local and informative using cross-trajectory state potentials.
2. **Coverage, Not Targeting:** precise targeting can be the wrong objective when verifier information density is low; broad causal-chain coverage may dominate.
3. **CHIME:** before updating a component, attribute the outcome to the correct subsystem.
4. **Conditional-diversity work:** preserve distributional/semantic diversity rather than relying on surface-level entropy.

For scientific ideation / idea graphs, a promising abstraction is therefore:

`idea graph exploration -> verifier/evidence signals -> attribution/coverage decision -> node/branch/component credit -> diversity-constrained policy or harness update`

This is more general than simply adding a step-level process reward to GRPO.