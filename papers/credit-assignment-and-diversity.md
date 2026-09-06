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

## 2026-09-03 / 2026-09-04 newly surfaced high-priority work

### TIGPO: Temporal Instance-Graph Policy Optimization for Long-Horizon LLM Agents
- Author: Jinwei Gan
- arXiv: https://arxiv.org/abs/2609.03383
- Date: 2026-09-03
- Category: Agentic RL; Graph Credit Assignment; Historical Experience; Exploration
- Core idea: keeps a **persistent per-task state-transition graph across policy updates** instead of rebuilding and discarding a graph every batch. Historical transitions provide detached structural references for current on-policy credit, while an Exploration–Revisit schedule reconnects present rollouts to useful earlier states.
- Benchmarks: ALFWorld; WebShop.
- Why it matters for our direction: this is unusually close to an **idea-graph RL** abstraction. Different policy versions can discover different fragments of a useful reasoning/idea path, and a persistent graph can connect them without directly replaying stale policy trajectories.
- Key research question opened: should a research-idea graph be persistent across training iterations, with old hypothesis/evidence transitions used to estimate credit for new branches?
- Priority: **Must read**.

### FlowBalance: Verifier-Grounded Self-Improvement from On-Policy Reasoning Experience
- Authors: Zixun Huang, Kishan Panaganti, Haitao Mi, Leowei Liang
- arXiv: https://arxiv.org/abs/2609.03241
- Date: 2026-09-03
- Category: On-Policy Self-Improvement; RL; Diversity Preservation; Verifier-Grounded Guidance
- Models: Qwen3-4B; Qwen3-8B.
- Core idea: obtains privileged same-policy self-guidance but **calibrates it with verifier-derived group advantage**: retain guidance for positive-advantage trajectories, reverse it for negative-advantage trajectories, and turn it off when the group gives no outcome preference.
- Diversity result: reports higher correct-strategy diversity than FlowRL in a controlled AIME24 diagnostic while avoiding direct OPSD response-length collapse.
- Why it matters for our direction: provides a concrete recipe for using dense/self-generated guidance without letting it collapse exploration onto one apparently good mode. This is directly relevant to diversity-preserving research-idea RL.
- Key research question opened: can idea critique/self-guidance be verifier-gated so that it sharpens good idea modes without homogenizing the idea population?
- Priority: **Must read**.

### SciLENS: RL-Driven Autonomous Agents for Scientific Localized Evidence Navigation and Synthesis
- Authors: Leqi Zheng, Jinbo Su, Yuying Li, Chaokun Wang, Weiping Wang, Haitao Li, Jiajun Zhang, Shannan Yan, Zhaolu Kang, Rong Fu, Jie Wu, Fang Niu, Hang Zhang
- arXiv: https://arxiv.org/abs/2609.03338
- Date: 2026-09-03
- Category: Scientific Agent; Citation Graph; Process Reward; Evidence Grounding
- Core idea: a fully local scientific-evidence agent over ~12M academic records. It makes **structural visualization of citation topology an actionable tool inside the reasoning loop**, and creates training data from multi-hop citation-graph substructures.
- Reward / credit design: reverse-decomposition rubrics provide fine-grained process rewards for early planning and evidence grounding.
- Why it matters for our direction: this is not research-idea generation itself, but it gives a practical precedent for making a scientific graph an **active reasoning state/tool** rather than a passive visualization, and for attaching process reward to scientific evidence navigation.
- Key research question opened: can an idea graph jointly represent hypotheses and citation/evidence topology, with RL deciding when to expand, merge, verify, or abandon branches?
- Priority: **High**.

### Gradients Know What Outcomes Don't: Gradient-Aligned Rewards (GAR)
- Authors: Leqi Zheng, Jinbo Su, Fang Niu, Chaokun Wang, Weiping Wang, Jiajun Zhang, Shannan Yan, Jie Wu, Zhaolu Kang, Rong Fu, Hang Zhang
- arXiv: https://arxiv.org/abs/2609.03342
- Date: 2026-09-03
- Category: Dense Reward; RLVR; Process Supervision; Gradient-Space Credit
- Models: Qwen3-4B; Qwen3-8B.
- Core idea: builds a dense reward from cosine alignment between a rollout gradient and an expert-anchor gradient, using truncated backpropagation through the output projection; reported wall-clock overhead is under 9%.
- Why it matters for our direction: introduces a third family of process signal beyond **LLM-judge textual critique** and **outcome redistribution**: the model’s own gradient geometry. This may be useful when multiple successful idea trajectories receive the same terminal score but differ in learning value.
- Priority: **High adjacent**.

### Legibility is Not Interpretability: Comparing Judged and Actual Importance in Chain-Of-Thought Reasoning
- Authors: Kevin Du, Alexander Hoyle, Laura Ruis, Acyr Locatelli
- arXiv: https://arxiv.org/abs/2609.04194
- Date: 2026-09-03; COLM 2026
- Category: Process Reward; Credit Assignment; LLM Judge Reliability; Counterfactual Importance
- Core idea: defines the importance of a reasoning step by its change in expected reward estimated with Monte Carlo rollouts, then tests whether LLM judges can identify those high-advantage steps. Capable judges beat a prevalence baseline but remain well below a noise ceiling; a trained step critic also remains far from ceiling on correct trajectories.
- Why it matters for our direction: a strong warning against assuming that a readable critique of an idea node is equivalent to its **causal contribution**. Process rewards based only on node text/LLM judgment may systematically misassign credit.
- Key research question opened: for idea graphs, should node importance be estimated with counterfactual rollout/intervention tests rather than solely by an LLM evaluator?
- Priority: **Must read / methodological caution**.

### Where Does Harness-Optimization Value Live? (HARNESSEVO)
- Authors: Michael Nguyen, Wei Chen Tan, Nurul Aisyah Hassan, Arvind Raman, Li Hua Lim, Ahmad Faiz Razak
- arXiv: https://arxiv.org/abs/2609.02889
- Newly listed in the 2026-09-04 cs.CL feed; arXiv submission-history metadata currently displays an earlier date.
- Category: Harness Evolution; Component Credit Assignment; Search Budget
- Core idea: decomposes a textual harness into role, task-strategy, tool/format-rules, and reflection/control, then uses leave-one-in / leave-one-out attribution to identify where evolution value actually resides.
- Benchmarks: ALFWorld; WebShop; frozen 7B backbone reported.
- Result: on ALFWorld, most useful optimization value localizes to reflection/control; uniform search-budget splitting across slots can fall below the optimizer’s effective search floor, while concentrating budget on the high-credit slot recovers large gains.
- Why it matters for our direction: suggests **credit assignment should precede harness evolution**. If we later let an idea-generation Harness evolve retrieval, critique, branching, memory, and verification, equal optimization budget across components may be a bad default.
- Priority: **High / directly relevant to Harness design**.

## Synthesis for research idea generation

The newest work sharpens the design space into four distinct questions:

1. **Persistent graph credit (TIGPO):** accumulate useful state/branch structure across policy updates rather than treating every rollout batch as isolated.
2. **Reliable dense signal (GAR + Legibility):** process reward can come from gradient/counterfactual evidence, while plain LLM-judged textual importance is not necessarily causal importance.
3. **Diversity-preserving guidance (FlowBalance):** dense self-guidance should be anchored to reliable outcome/verifier evidence so it does not collapse onto one reasoning mode.
4. **Scientific graph as an action space (SciLENS):** citation/evidence topology can live inside the agent loop, suggesting an idea graph should be executable and editable rather than only visualized.

A useful research abstraction is now:

`persistent idea/evidence graph -> exploration/revisit -> verifier/counterfactual/gradient evidence -> node/branch/component credit -> diversity-aware policy or harness update`

This is materially richer than simply attaching a scalar process reward to each idea-generation step.