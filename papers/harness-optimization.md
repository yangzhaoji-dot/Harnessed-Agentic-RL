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
- Important distinction from our target question:
  - CHILL asks **which workflow should be used now?**
  - Capability handoff asks **does the model still need this external component at all?**

## EvoHarness-RL
- arXiv: https://arxiv.org/abs/2608.05446
- Harness evolution is primarily external-state evolution: Belief / Progress / Experience are updated online and Experience is consolidated.
- Core BPE interface and harness action set remain fixed.
- Therefore it is closer to **runtime state evolution + learned harness-use policy** than unrestricted structural harness-program evolution.

## SafeEvolve
- arXiv: https://arxiv.org/abs/2609.02786
- Code: https://github.com/MaoPopovich/SafeEvolve
- Harness side: safety prompts and hierarchical skills receive bounded, reversible updates from trajectory-level safety evidence.
- Policy side: SFT + harness-augmented RL.
- Relevance: confirms that full harness-policy co-evolution is becoming an active research direction.

## Open questions

- Can control-flow structure, verification loops, routing, tool policies and sub-agent topology be evolved, not only prompts/memory/skills?
- How should harness edits be credited without overfitting to the trajectory set used for evolution?
- Which external capabilities should be permanent tools and which should be temporary scaffolds?
- Can harness evolution be coupled to measured model-side capability transfer rather than a fixed schedule?
