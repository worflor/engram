# Research design

## Aim

Study when paired oscillator dynamics provide useful, compact descriptions of trajectories, and which additional summaries preserve distinctions that a downstream task needs. Keep Engram inspectable as a mathematical object. Host applications may assign meaning to coordinates, wells and actions, but that meaning is not inferred from oscillator regularity alone.

The research program follows a simple chain:

```text
choose a representation → state what information it retains → test a task
                                      ↓
                         find aliases and failure cases
                                      ↓
                   add only a justified mathematical view
```

The object is not a text-only n-gram model. Byte histograms, token vectors, NPC witnesses and physical sensors are input representations; each carries its own assumptions and information loss.

## Questions

- Which trajectory families are well described by paired AR(2), and when do affine, first-order, static, raw or higher-order alternatives better explain the data?
- How do coordinate pairing, normalization, sampling cadence and embedding affect coefficient comparability and downstream discrimination?
- When do macro/micro prediction and residual coding improve decoded accuracy or serialized size against raw storage and simple predictors?
- How do quantization and recurrent conditioning affect error across time and block boundaries?
- Which distinctions survive in well statistics, temporal banks, relational moments and ordered history summaries? Which task-relevant histories become observationally identical?
- Does learning, feedback, consolidation or dreaming improve an external held-out outcome beyond internal geometry measurements?
- Which computations agree with the CPU reference for a stated workload, numeric tolerance and runtime?
- For an NPC, can experience change responses to the same current witness, and can the model relearn when consequences change?

## Method

Begin with exact constructions and synthetic signals whose generating rules are known. Use adversarial pairs to test what a summary preserves, including changes of phase, scale, offset, order, role identity, noise and sampling. Then evaluate on explicitly licensed, versioned external data or a native host integration. Pin source revision, input digest, dimensions, sample count, coordinate basis, random seed, runtime, hardware and parameters. Keep downloaded data, working checkpoints and personal data outside version control unless a specific frozen artifact is selected and documented.

Separate measurements by meaning:

- fitting residual, predictive capture, decoded source error and serialized size;
- representation quality, policy outcomes and retrieval quality;
- accuracy, memory, latency, throughput and energy where measurable;
- exact algebraic claims, floating implementation checks, finite observer results and application experiments.

Include metadata in byte counts. Compare against raw storage, persistence, local fits and simple feature baselines where relevant. A smooth trajectory may still be ill-conditioned; a quantization step does not bound recurrent reconstruction error globally. A well distance is geometry, not a calibrated probability or task score.

For learned systems, reserve data by trajectory, subject, document or world as the task requires. Fit calibration and representation choices on training data; use validation data for selection; keep final evaluation data untouched. Report seeds and aggregation choices. Preserve negative controls and failures. For temporal tasks, distinguish fit sufficiency from decision sufficiency: two histories can have equal fitted moments and different correct actions.

For encoding parity, compare decoders on the same packet under explicit tolerances. Encoder segmentation, floating-point evaluation order and cross-platform replay need not be bitwise identical. For memory continuation, compare canonical state and subsequent decisions across checkpoint boundaries, not merely serialized bytes.

## Evidence and provenance

`research/provenance.json` pins imported inputs by digest and source revision when one exists. Historical portfolio code and results under `research/portfolio/` are contextual evidence; they are not automatically reproduced by the maintained test suite. Frozen artifacts under `research/artifacts/` have separate provenance and scope entries.

The offline instrument is a measured finite gallery. Synthetic checks establish behavior for declared inputs, not whole-codec equivalence or universal compression. GPU, corpus ingestion, language-model loops and Java/Minecraft execution require their own environment and receipts. Exact Scryer findings under `research/exact/` record requests, outcomes and invocation metadata. An exact adjudication concerns the stated claim; a replayed proof requires its own proof DAG and successful verification. Neither establishes floating-point package correctness by itself.

Promote recurring, useful findings into API references or executable regressions. Keep exploratory claims in the relevant brief until their scope and implementation contract are clear. A benchmark result alone is not a reason to alter the object.

## Integrated research areas

The [learning reference](reference/LEARNING.md) describes reusable training, refinement, geometric wells, graph/meta behavior, batching, feedback and evaluation. The source map in [research integration](../research/integration-map.json) records imported materials and review boundaries. Mechanisms inherited from prior experiments retain their own evidence; their inclusion does not certify every historical script or reproduce Alexandria's original training run.

The crystallography audit in portfolio research falsified broad claims about transformer predictability and compute skipping. Its measurements have not been rerun as a maintained acceptance result. Its methodological lessons appear as controls: centered R² beside energy capture, persistence and local-fit baselines, original-identity permutation tests, and time-shuffled inputs. Energy-capture or sorted-eigenvalue correlations do not establish semantic understanding or identical output.

The [well-network experiments](reference/LEARNING.md) evaluate held-out synthetic recognition and forecasting with local-fit baselines, independent training/evaluation phase and amplitude, and negative controls. Corpus-trained coordinates use training documents only. Embedding refinement requires a fresh held-out comparison. No convergence theorem, monotonic class purity or general benefit from dreaming is presumed.

The [NPC and Minecraft work](reference/NPC.md) treats learned experience as a data structure: witness contexts, temporal fits, growing wells, action consequences and response values. Observed-model planning does not invent visits or physical trajectories. Delayed-cue tasks compare identical current witnesses with different histories and include a temporal ablation. Native Minecraft observations and actions provide a separate host-integration lane. Compatible Alexandria coordinates can provide a general dynamics prior, not task-specific action knowledge.

The [real-signal study](briefs/npc-data-and-stress.md) uses a subject-separated UCI Human Activity Recognition split. Calibration and feature choices are restricted to training and validation subjects; held-out subjects measure generalization. Simple window statistics and time-scramble controls remain in the report, including cases where simple statistics outperform oscillator features. This is representation readout evidence, not NPC reward-learning evidence. Its separate stress campaign measures deterministic checkpoint continuation, streaming moments, state growth and observed-model planning.

The [affine forecasting reference](reference/AFFINE.md) separates an exact translation identity and rank counterexample from numerical forecasting evidence. Comparisons include independently centered lag regressions, persistence, synthetic families and real-signal controls. On the recorded real-data task, the homogeneous fit has lower aggregate mean and median error than the affine extension; the latter remains an explicit optional view.

The [history and composition study](briefs/history-and-composition.md) covers ordered-summary algebra, multiscale NPC history and stable-coordinate generation. Its finite observer concerns a bounded sliding-window domain, distinct from an unbounded cumulative NPC. Fit sufficiency, goal sufficiency and learned decision behavior remain separate claims. A numerically richer history view can still collapse through routing. Successor features and host-owned decision-expiry are research directions, not implemented shortcuts.

The [relational fidelity study](briefs/relational-memory.md) compares cross-channel moments, role-conditioned evidence and structured path/braid/link tasks. It reports where each representation succeeds or aliases under authored controls. Its results are bounded to those task constructions and seeds; they do not establish a universal semantic representation, an emotion model or causality in uncontrolled observations.
