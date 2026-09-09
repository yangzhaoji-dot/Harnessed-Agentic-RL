# Policy-Adaptive Harness Interventions: Reading Map and Design Gaps

Updated: 2026-09-09. Scope: the current proposal to use on-policy training evidence to select **where**, **which configuration**, and **how much** runtime Harness assistance to apply, then learn from autonomous and assisted trajectories. We retain Experience / Planning / Action filtering as operator families; this is not a recommendation to change research direction.

This note distinguishes primary-source mechanisms from our proposed adaptations. It does not claim exhaustive novelty verification, reproduced results, or measured training speedups. arXiv dates below identify papers, not acceptance status. Newly added and previously tracked papers are deliberately mixed by methodological relevance.

## 0. Main assessment

Much of the machinery already exists: entropy-guided branch sampling (CURE), executable component interfaces (JIT), feedback-triggered plan repair (AdaPlanner), cost-aware architecture selection (MaAS), paired assisted/autonomous credit (SkillC), local state grouping (GiGPO), counterfactual workflow evaluation (CHILL), and adaptation to a changing policy (ReSkill).

The unresolved integration question for us is narrower: **under a fixed compute budget, can a training-time controller select useful interventions at actionable trajectory boundaries and construct updates that improve the policy under a fixed deployment protocol, rather than merely improve the currently assisted trajectory?** Harness-free evaluation is necessary when claiming internalization, but is not the only legitimate deployment objective.

## 1. Where to intervene

### 1.1 Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning
- Primary paper: https://arxiv.org/abs/2506.01939 (2025).
- Mechanism: studies high-entropy reasoning forks and restricts policy-gradient updates to a minority of tokens. Experiments include Qwen3-8B/14B/32B reasoning models.
- Reusable: token-distribution entropy as a candidate-position signal.
- Boundary: identifying useful *gradient locations* is not the same as identifying useful *Harness intervention locations*. High entropy does not identify which component will help.
- Read for: entropy computation, token selection, random/low-entropy controls. Full-vocabulary entropy is not identical to sampled-token negative log probability.

### 1.2 CURE: Critical-Token-Guided Re-Concatenation for Entropy-Collapse Prevention
- Primary paper: https://arxiv.org/html/2508.11016v2 (2025), especially §3.2.1 and Appendix A.
- Mechanism: original rollouts -> sample a position among top-K entropy locations -> use preceding prefix as a new prompt -> generate branches -> optimize original and branched samples. The branched prompt prefix does not receive policy-gradient likelihood terms; original trajectories can still train their prefixes. Reported experiments use Qwen2.5-Math-7B.
- Reusable: the exact structural idea of shared-prefix continuation, original/branched data bookkeeping, and a branch-only loss mask.
- Boundary: text re-prompting does not restore a live ALFWorld state. CURE does not choose Experience/Plan/Filter, nor establish that prefix selection is causally optimal for those interventions.
- Code caveat: the paper lists https://github.com/bytedance/CURE, but the connected GitHub README fetch returned 404 in this check. Treat code availability as unconfirmed, not as a tested runnable baseline.

### 1.3 Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM's Reasoning Capability
- Primary paper: https://arxiv.org/html/2411.19943v2 (2024), §2–3.
- Mechanism: contrast models fitted to correct versus incorrect reasoning trajectories to identify error-associated tokens; cDPO uses the resulting token-level signal.
- Reusable: a failure-oriented alternative to uncertainty-only candidate selection.
- Boundary: extra scoring/training cost; contrastive correlation is not a direct measurement of the benefit of PLAN or retrieval at that point.

### 1.4 Important control: Extremely Sparse Supervision Incentivizes Reasoning Ability
- Primary paper: https://arxiv.org/html/2609.04565v1 (2026), §4.1 and §5.
- Already present in the 2026-09-08 tracker; reclassified here.
- Mechanism: randomly supervising one token can improve OPD; selection of one/two tokens with extreme teacher–student reward can match or exceed dense supervision in reported settings, with additional PPO/RLVR experiments.
- Implication for us: sparse supervision itself does not prove successful localization of causal bottlenecks. Include random-location, equal-count/equal-budget controls. This paper concerns training masks, not runtime Harness insertion.

### Our design work for axis 1

We need a mapping from token statistics to **legal decision boundaries** (before a complete environment action or between model calls). For an offline branch experiment, the preceding response's token scores can suggest the boundary before the corresponding action, but future response statistics are not available to a deployable online trigger. Entropy is a proposal heuristic; validate candidates using continuation outcomes.

The proposed first-half restriction is a budget/recovery heuristic, not an established optimum. Use remaining environment budget, not a future realized episode length, for an online-compatible rule. Low entropy can accompany confidently wrong behavior, so preserve coverage beyond entropy-only candidates.

## 2. Which components, which modes, and how many

### 2.1 JIT-Agent / HarnessFactory
- Paper: https://arxiv.org/abs/2608.25593.
- Official source read: https://github.com/bingreeky/JIT/blob/main/harness_factory/README.md.
- Source read: https://github.com/bingreeky/JIT/blob/main/harness_factory/harnesses/plan_and_execute/planning.py.
- Mechanism: interchangeable memory, planning, action-loop and tool-policy modules. Seed architectures include a linear plan, DAG planning, plan-free execution, context folding and multi-path execution.
- Implementation correction: `plan_and_execute` calls the model for an initial plan and triggers a summary at a positive multiple of `summary_interval` (default 8). Its `update_plan()` returns a SummaryState and does not itself overwrite `_current_plan`. It should not be described as guaranteed event-driven plan replacement.
- Reusable: a finite, typed operator catalogue and explicit module interfaces. Do not transplant its entire inference runner into the RL collector.

### 2.2 AdaPlanner: Adaptive Planning from Feedback with Language Models
- Primary paper: https://arxiv.org/html/2305.16653v1 (2023), §3.1–3.2.
- Mechanism: separates in-plan refinement (extract information needed by the existing plan) from out-of-plan refinement (revise the plan after an assertion/expectation fails). Refine-then-resume continues within the current episode.
- Reusable: distinguish cheap local correction from full replanning; use explicit expected-versus-observed progress checks.
- Boundary: prompt-based inference method, not a trained intervention controller. Its code-style plans and environment adapters are not directly interchangeable with our textual subgoal plan.
- Official implementation linked by the paper/project: https://github.com/haotiansun14/AdaPlanner. No runtime reproduction in this survey.

### 2.3 MaAS: Multi-agent Architecture Search via Agentic Supernet
- Primary paper: https://arxiv.org/abs/2502.04180 (2025).
- Mechanism: learns a distribution of query-dependent agentic architectures rather than one static architecture; explicitly accounts for inference resources.
- Reusable: select from a bounded operator space while considering cost, instead of always activating every module.
- Boundary: inference architecture optimization is not proof of improved policy-learning efficiency. Component cardinality is not a sufficient compute metric: one plan repeated every turn may cost more than several passive components.
- Official implementation linked by the project: https://github.com/bingreeky/MaAS. Algorithm-level rather than runtime-reproduced review here.

### Our design work for axis 2

Use an operator specification such as `(family, mode, parameters, persistence, call_budget)` rather than an unrestricted new workflow. Initial candidate examples—not inherited paper defaults—could be `NONE`, one retrieved experience, one short plan, local plan repair, and a limited-duration action shortlist. Start with at most one active intervention to make tests interpretable; this is an experimental control, not a claim of optimality.

Record separate budgets for generated tokens, injected-context tokens, model calls, environment steps, replay/probe cost, and wall-clock/GPU-hours. Increasing experience count or plan frequency is not guaranteed to monotonically increase assistance quality.

A displayed action shortlist is **observation/prompt manipulation**, not necessarily a hard action-space mask. A hard filter requires explicit decoding/execution constraints and corresponding behavior-probability bookkeeping. Ranking alone is not action-space reduction.

## 3. Credit assignment and the 8+8 proposal

### 3.1 SkillC: Learning Autonomous Skill Internalization in LLM Agents via Contrastive Credit Assignment
- Primary paper: https://arxiv.org/html/2605.27899v1 (2026), §3.2–3.4.
- Mechanism: paired skill-injected and skill-free rollouts with the same policy; a global-ranking stream plus a condition-normalized contrastive correction; smoothed validation contrast controls attribution strength, assisted rollout fraction and active-skill retirement.
- Reusable: assisted/autonomous data separation, dual-stream advantage as a concrete baseline, and separate timescales for noisy update signals versus curriculum decisions.
- Boundary: task/skill-level paired rollouts are not identical to matched-prefix heterogeneous interventions. The objective favors autonomous success; it is not an unbiased causal decomposition of every plan/action contribution. A small with/without gap alone does not prove internalization.
- Code: no official implementation was confirmed in the primary page during this pass. Full method details are available.

### 3.2 Group-in-Group Policy Optimization for LLM Agent Training (GiGPO)
- Primary paper: https://arxiv.org/abs/2505.10978 (2025).
- Mechanism: combines episode-level relative advantage with anchor-state-based local action groups; no auxiliary critic or additional rollout is required by the core grouping mechanism. Includes Qwen2.5-1.5B and 7B on ALFWorld/WebShop.
- Reusable: distinguish whole-task grouping from local decision-state grouping. It is particularly relevant to our current verl-agent baseline.
- Boundary: identical environment observation text is not necessarily identical agent information state. Plan, memory, previous context, available tools and remaining budget can change the continuation distribution.

### 3.3 What Does Multi-Harness RL Learn? Credit Assignment and Portability in Coding Agents
- Primary paper: https://arxiv.org/html/2609.04518v1 (2026), §3–5.
- Mechanism: controlled comparison of within-task-harness and cross-harness GRPO grouping, keeping the training corpus/budget fixed in the main experiment and including an on-policy re-collection check.
- Reusable: same-data grouping ablations, fixed evaluation harness, held-out-harness evaluation.
- Boundary: its coding result is a warning against assuming pooling implies portable capability, not a theorem banning mixed-condition groups or a direct result for our ALFWorld setup.

### 3.4 Conditional background: LUFFY
- Primary paper: https://arxiv.org/abs/2504.14945 (2025), Learning to Reason under Off-Policy Guidance.
- Mechanism: combines off-policy demonstrations and current-policy rollouts with importance-sampling/policy-shaping design.
- Reusable only when relevant: if an external teacher, stale policy, forced action replacement or foreign generated plan is trained as though sampled from our policy, the behavior distribution must be addressed explicitly.
- Boundary: a policy's own rollout conditioned on an added experience is not automatically off-policy. The conditioning and actual action-sampling procedure matter.

### Corrections and open choices for axis 3

**Mixed grouping is not categorically forbidden.** ReSkill explicitly assigns different skill versions inside a GRPO group; CURE merges original and re-prompted samples. We must define the intended objective, behavior condition and normalization rule instead of treating 'same prompt' as a universal validity requirement.

**Separate normalization is not automatically sufficient.** With a binary terminal reward, eight autonomous failures and eight assisted successes give zero outcome-relative advantage within *both* constant-reward subgroups. This example ignores KL/entropy/auxiliary losses. Pooling may restore reward contrast but does not identify which policy decisions versus external assistance created the gap.

Separate three statistical objects:
1. Policy-update group: relative quality of sampled actions/trajectories.
2. Intervention comparison set: matched-state assisted versus fresh untreated continuations.
3. Controller evidence buffer: estimates indexed by policy version, task/prefix, operator specification and cost.

Preserve task ID, trajectory ID, prefix/checkpoint ID, policy version, full prompt, Harness specification, behavior log probabilities, environment-call count and loss masks. If branching eight original trajectories produces eight different prefix states, they are not automatically one identical-state local group.

**Prefix masking is a choice of training objective, not a mathematical necessity.** A useful initial design is to train original autonomous trajectories normally and mask the copied prefix only on extra branches. Masking all prefixes removes learning to reach useful states. Opposite Monte Carlo outcomes for the same prefix are not intrinsically invalid gradients; naïve repeated weighting is the practical problem.

## 4. Signals for adaptive Harness control

### 4.1 CHILL-Harness: Counterfactual Harness Learning for Efficient Reasoning in Long-Horizon Agents
- Primary paper: https://arxiv.org/html/2607.25825v1 (2026), especially Appendix A.2–A.4 and offline-training details.
- Mechanism: finite intervention families are realized as workflows; paired checkpoint continuations yield utility-effect labels; effect models and margin/authorization rules guide execution. Labels include resource accounting and explicit replay assumptions.
- Reusable: checkpoint completeness, isolated branches, common continuation/evaluation protocol, fresh post-checkpoint randomness, pre-specified inclusion rules, and cost/uncertainty-aware authorization.
- Boundary: its effect estimation is trained offline and frozen for evaluation. A controller for an actively changing RL policy must handle drift. Workflow-utility improvement is not automatically a policy-learning gain.
- Code caveat: the checked paper links https://github.com/csdstar/chill-dev, whereas older tracker entries pointed to CHILL-Harness. The connected fetch of chill-dev/README.md returned 404. Do not claim current code availability is verified.

### 4.2 ReSkill: Reconciling Skill Creation with Policy Optimization in Agentic RL
- Primary paper: https://arxiv.org/html/2606.01619v1 (2026), §3.1, §3.3 and Appendix C.5.
- Mechanism: tests old/new skill versions within the GRPO rollout group, allocates samples with Thompson Sampling, and discounts stale evidence while the policy trains.
- Reusable: small candidate competition, minimum exploration, policy-drift-aware evidence forgetting, periodic accept/reject decisions.
- Boundary: its tested objects are skill-bank versions. Applying the same mechanism to PLAN modes, filters, durations and branch positions requires defining comparable contexts and budgets.
- Official code: https://github.com/amazon-science/reskill. Its README was previously read; this survey does not claim training reproduction.

### 4.3 Skill0: In-Context Agentic Reinforcement Learning for Skill Internalization
- Primary paper: https://arxiv.org/html/2604.02268v1 (2026), §3.3.
- Mechanism: estimates on-policy skill helpfulness, filters/ranks active skills, and selects under a stage-wise decaying budget. ALFWorld's reported stage budget includes 6 -> 3 -> 0.
- Reusable: distinguish 'which skills' from 'how much budget'; periodic paired evaluation plus a simple budget schedule is a strong baseline.
- Boundary: it includes visual context rendering, so the full implementation is not a drop-in recipe for a text-only 1.5B model. Monotonic withdrawal can fail if the policy initially cannot use a useful component or later encounters distribution shift.
- Official code linked by authors: https://github.com/ZJU-REAL/SkillZero.

### 4.4 Teacher-Student Curriculum Learning (TSCL)
- Primary paper: https://arxiv.org/abs/1707.00183 (2017).
- Mechanism: prioritizes subtasks by learning progress and revisits deteriorating performance.
- Reusable: distinguish assistance that raises today's score from training experiences that increase tomorrow's ability; include forgetting in adaptation.
- Boundary: this is curriculum background, not a Harness-specific solution. Substituting Harness operators for subtasks still requires an estimable learning-progress reward.

### Our design work for axis 4

Maintain distinct signals:
- Runtime trigger evidence: action/semantic uncertainty, repetition, explicit plan-feedback mismatch, invalid actions, remaining budget. Terminal reward remaining zero is not a reliable stagnation detector in a sparse-reward task.
- Current intervention utility: fresh paired return difference, minus explicitly accounted intervention cost, with uncertainty.
- Training usefulness: fixed-protocol validation progress after updates, group information content, and retained autonomous performance. Group reward variance is a diagnostic proxy, not a sufficient target; more variance may represent noise.

The long-term target could be improvement of fixed-protocol policy performance per unit compute, while immediate counterfactual uplift is a cheaper proxy. How to estimate and connect those two quantities is a candidate research contribution, not something established by merely naming a controller.

## 5. Two close priors requiring an updated novelty check

### Harness-R1: Learning to Edit Executable Runtime Harnesses from Agent Failure Trajectories
- Primary paper: https://arxiv.org/html/2608.02276v1 (2026), §3 and Appendix C.
- Trains a dedicated harness engineer by SFT + online GRPO using outcomes of patches generated from failure batches; target policy is frozen within engineer training. It also studies target-specific engineers after target fine-tuning.
- Four hook sites: episode initialization, pre-decision hint, pre-action intervention, post-feedback recovery.
- Our distinction must not be simply 'diagnose failures and dynamically modify a harness.' Potential distinction: a bounded training-time intervention policy for a simultaneously learning target, with position-conditioned branch credit and fixed-budget policy-learning evaluation.
- Author-linked code: https://github.com/DeepExperience/Harness-R1; not runtime-tested here.

### EvoTrainer: Co-Evolving LLM Policies and Training Harnesses for Autonomous Agentic Reinforcement Learning
- Primary paper: https://arxiv.org/html/2606.03108v1 (2026), §3.1–3.6 and diagnostic appendices.
- Operates across training versions: inspects rollout evidence, revises diagnostics, backtests interventions and stores reusable training skills. Its training-side diagnostic harness is different from a task agent's runtime harness; gated human approval remains in the described execution workflow.
- Useful signals include truncation, low-information/dead groups and behavior-collapse diagnostics.
- Our distinction must be explicit: within-training, trajectory-local runtime assistance versus an experiment/version-level autonomous training system. 'Adapt based on training feedback' alone is too broad.

## 6. Gap matrix

| Question | Mostly borrowable | Adaptation/engineering required | Candidate original design |
|---|---|---|---|
| Where to branch? | CURE / entropy / contrastive critical-token heuristics | Token-to-environment-boundary mapping; deterministic replay or full checkpoint restoration | Component-specific intervention value versus generic uncertainty under a remaining-budget constraint |
| Which component/mode? | JIT interfaces; AdaPlanner refinement modes; bounded architecture choice | RL-call logging, masks, parser/adapter tests; shortlist versus hard-mask semantics | Small state-conditioned operator portfolio selected for learning value rather than only immediate performance |
| How many? | MaAS cost-aware selection; Skill0 budget; ReSkill candidate competition | Normalize tokens/calls/context/replay cost, not only component count | Reversible, drift-aware allocation across heterogeneous intervention types and durations |
| How to group 8+8? | SkillC / GiGPO / Multi-Harness grouping controls | Exact behavior conditions, local versus task groups, prefix deduplication | Jointly specify policy credit and intervention-effect estimation without collapsing one into the other |
| How to adapt? | CHILL effect labels; ReSkill forgetting; SkillC smoothing; TSCL progress | Separate training/controller/evaluation data and timescales | Reliable link between immediate rescue and subsequent fixed-protocol policy improvement |

## 7. Experimental safeguards before claiming gains

1. **Selection bias:** selecting originally failed suffixes makes their observed return a biased untreated reference. Even a fresh no-Harness retry can improve. Use original failures for diagnosis; use newly sampled untreated and assisted continuations, with pre-specified candidate selection, for causal labels. If the controller sees future diagnostic evidence, label the experiment retrospective training-time intervention, not an online trigger.
2. **Equal budgets:** compare ordinary NoH rollout, additional NoH rollout, NoH matched-prefix branching, fixed Harness, budget-matched random interventions, and the proposed selector. A full Harness is a baseline, not a guaranteed upper bound.
3. **State restoration:** same task/seed is insufficient without consistent environment state, agent context, memory, tool state and budget accounting. Replaying recorded actions avoids LLM regeneration but does not make environment execution free.
4. **Who generated what:** train current-policy-generated plan/action tokens under their actual inputs; do not treat controller-forced choices or external tool text as sampled policy actions. Preserve valid behavior probabilities for constrained generation and any off-policy training.
5. **Evaluation:** keep test tasks out of the experience bank/controller selection loop. Hindsight information from training tasks can be a declared privileged-training design, but is not evidence of deployable intervention or generalization. Measure fixed-Harness and no-Harness performance separately when relevant.
6. **No automatic internalization claim:** a shrinking assistance gap can mean both conditions fail, the component became worse, or another component substituted for it. Verify maintained/increased autonomous performance on held-out tasks.

## 8. Reading sequence with concrete deliverables

Start with **CURE -> SkillC -> CHILL -> ReSkill**. Produce, respectively: (1) a prefix/branch/mask data-flow sketch; (2) three competing 8+8 advantage definitions and their constant-reward edge cases; (3) a state-restoration and unbiased-label checklist; (4) an evidence-decay and candidate-budget protocol. Then read JIT/AdaPlanner for implementations and Harness-R1/EvoTrainer for novelty boundaries. MaAS, Skill0, GiGPO, cDPO, 80/20, TSCL and sparse-supervision work fill specific remaining questions.

The proposed experiment is not established merely by combining these papers. The first falsifiable target is that **position-aware, budget-matched assistance yields better fixed-protocol policy learning than equally costly random assistance or extra autonomous exploration**.
