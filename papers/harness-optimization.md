# Harness Optimization / Evolution

This category tracks work where the **harness itself is adapted, searched, edited, or evolved**.

## Co-Harness
- arXiv: https://arxiv.org/abs/2607.22688
- Harness components: prompts, tools, skills, middleware, memory.
- Update mechanism: an LLM-based HarnessCritic analyzes failed trajectories, identifies harness-level failure modes, proposes local updates, and validates them.
- Model update: improved harness trajectories are used to fine-tune the model, alternating harness and model optimization.
- Key distinction: genuine harness-model alternating optimization, but not a general RL-based capability handoff / retirement mechanism.

## CHILL-Harness
- arXiv: https://arxiv.org/abs/2607.25825
- Code: https://github.com/csdstar/CHILL-Harness
- Core method: checkpoint execution state, create paired factual/counterfactual continuations, estimate intervention effect offline, train lightweight routing/change/stop/intervention predictors, then use them for runtime workflow adaptation.
- Main goal: reduce reasoning/resource cost while preserving task success.
- Important distinction:
  - CHILL asks **which workflow should be used now?**
  - capability handoff asks **does the model still need this external component at all?**

## AutoSaddler
- arXiv: https://arxiv.org/abs/2608.23041
- Code: https://github.com/microsoft/AutoSaddler
- License: MIT
- Search space: prompts, tool definitions and implementations, middleware hooks, agent-loop logic.
- Method: deep failure-trace diagnosis -> structured Capability/Steering patches -> validation-based selection; reflection and candidate evolution are retained in an EvoDAG.
- Benchmarks: GAIA2, SWE-Bench Pro, Terminal-Bench 2.0.
- Reported gains: +9.0 / +9.6 / +10.0 percentage points over the corresponding base harnesses.
- Relevance: a strong **structural harness optimization** baseline. Any co-evolution paper should show why changing model weights adds value beyond a strong fixed-model harness optimizer.

## StarHarness
- arXiv: https://arxiv.org/abs/2608.24804
- Code: https://github.com/ServiceNow/StarHarness
- Search space: prompts/task framing, tool interfaces, skills, MCP providers, subagent structure, agent-loop configuration.
- Model weights: fixed.
- Search/evaluation design: baseline-failure stratification; proposer-visible search tasks; proposer-hidden selection tasks; held-out evaluation tasks.
- Benchmarks: ITBench SRE, EnterpriseOps-Gym ITSM, AutomationBench Finance.
- Reported improvement: roughly 20–35 percentage points after 4–12 accepted changes per environment; cross-model transfer tested on GPT and Qwen families.
- Relevance: strong prior for broad structural harness evolution and generalization-aware selection.

## JIT-Agent
- arXiv: https://arxiv.org/abs/2608.25593
- Code: https://github.com/bingreeky/JIT
- Core idea: a dedicated **Harness Intelligence Model** synthesizes task-adaptive harnesses, repairs them, and self-evolves from an archive of prior harness configurations and their performance signals.
- Harness abstraction: composable four-module protocol covering memory, planning, action protocol and tool/skill orchestration.
- Evaluated serving models include DeepSeek-V4-Flash, GLM-5.2 and Qwen-family models.
- Benchmarks include DeepSearchQA and OdysseyBench.
- Relevance: important alternative thesis — instead of absorbing harness capability into the base model, scale a reusable harness-intelligence layer as an orthogonal capability dimension.
- Gap: served base model is largely off-the-shelf; no joint policy-harness Agentic RL.

## EvoHarness-RL
- arXiv: https://arxiv.org/abs/2608.05446
- Harness evolution is primarily external-state evolution: Belief / Progress / Experience are updated online and Experience is consolidated.
- Core BPE interface and harness action set remain fixed.
- Therefore it is closer to **runtime state evolution + learned harness-use policy** than unrestricted structural harness-program evolution.

## ReSkill
- arXiv: https://arxiv.org/abs/2606.01619
- Code: https://github.com/amazon-science/reskill
- External skill library evolves *inside* the GRPO loop: skills are created, tested, revised, versioned and pruned while the policy changes.
- Important methodological point: competing skill versions are evaluated within the same GRPO rollout group, and Thompson Sampling controls exploration/selection.
- Relevance: makes generic claims of "RL-in-the-loop skill/harness evolution" insufficient by themselves.

## SafeEvolve
- arXiv: https://arxiv.org/abs/2609.02786
- Code: https://github.com/MaoPopovich/SafeEvolve
- Harness side: safety prompts and hierarchical skills receive bounded, reversible updates from trajectory-level safety evidence.
- Policy side: SFT + harness-augmented RL.
- Relevance: confirms that full harness-policy co-evolution is becoming an active research direction.

## Environment-side harness evolution: EnvHarness
- arXiv: https://arxiv.org/abs/2608.19880
- Code: https://github.com/google-research/envharness
- Core idea: wrap a frozen environment with programmable components; EnvRigger diagnoses the current policy and synthesizes/validates components that target its weaknesses.
- Evaluation: five benchmarks across four domains; reported up to +9.0 points on held-out instances with fewer execution steps.
- Relevance: broadens the co-evolution picture to **Policy <-> Environment** and offers adaptive training signals for RL.
- Important distinction: this is environment-side rather than deploy-time model harnessing.

## Open questions

- Can control-flow structure, verification loops, routing, tool policies and sub-agent topology be evolved jointly with model learning rather than only optimized around frozen models?
- How should harness edits be credited without overfitting to the trajectory set used for evolution?
- Which external capabilities should remain permanent tools and which should be temporary scaffolds?
- Can harness evolution be coupled to measured model-side capability transfer rather than a fixed schedule or a local utility heuristic?
- How should we distinguish **skill/harness utility pruning** from genuine **model-side internalization**?
