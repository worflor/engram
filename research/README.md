# Research and evidence

`provenance.json` records imported file hashes, source paths, revisions and the local-only status of the portfolio sources. It identifies extraction inputs; it is not a current-file integrity manifest.

[Source lineage](../docs/briefs/source-lineage.md) records import and compatibility
boundaries. [Reference unification](../docs/briefs/reference-unification.md) maps
consumer contributions to the maintained mathematical and format contracts.

## Alexandria research artifact

[artifacts/alexandria.engram](artifacts/alexandria.engram) is the original frozen Alexandria ZIP used to produce [Manifold](https://github.com/worflor/git-desktop-premium-ultra-promax-plus-R)'s bundled `alexandria.endb`. It is an accumulated geometric memory, retained as a research artifact.

The snapshot has **300 dimensions and 225 wells**. Its manifest records **5,518,095 absorptions** and a recent-history buffer of 1,200 entries. Those are stored state counters, not an independently reproduced dataset or evaluation score. It contains a manifest, well statistics, recent parameter history and annotations. The file is 3,750,381 bytes; SHA-256 is `408850caea5dcaa475b1609becea835af85adac33127b2895757e547ec26f3f2`.

The ZIP's four payloads match the committed Manifold ENDB v2 export after decoding that format's layout. The snapshot uses ZIP format version **4**, supported by the reference Python reader. Manifold's runtime uses the corresponding ENDB representation; Rust's ENBR reader does not read this ZIP.

Only the frozen snapshot is included here: no adjacent corpus, intermediate checkpoints, ingestion logs or export procedure. Source identity, content hash, matched Manifold revision and verification are recorded in the existing [provenance manifest](provenance.json). The artifact retains the WLCSL-1.0 terms attached to the same Alexandria research content in Manifold; see [license boundaries](../LICENSES/README.md).

## Research sources and findings

The [relational fidelity study](../docs/briefs/relational-memory.md) compares eleven
memory views on six controlled tasks, with independent fresh-seed confirmation,
exact path/braid/link counterexamples, two replayed Scryer observer DAGs and a
native Minecraft context/action binding control. It distinguishes relational
evidence, explicit context keys and geometric invariants from a general semantic
memory claim. The [conditional-binding study](conditional-binding.json) evaluates
streaming `ContextRelations` summaries on eight seeds, including exact parity
marginals and retained level information. [Recency readouts](recency-decoding.json)
measure which distinctions are accessible to different readouts of the same
temporal bank. These studies do not change codec or NPC checkpoint semantics.

The [history and composition study](../docs/briefs/history-and-composition.md) covers exact and floating history concatenation, learned recency distinctions, strict interpolation and a quintic model-mismatch fixture. [Finite observer requests/bindings](exact/witness-observer-requests.json), [deployed construction and DAG replays](exact/witness-observer.json), and [algebra/dwell component findings](exact/history-composition-cases.json) have distinct proof scopes. The observer source uses 64 six-sample sliding windows; it is not a whole-NPC certificate. Shared-motive successor features and Tiercel scheduling remain research proposals.

The [affine extension](../docs/reference/AFFINE.md) separates motion from its origin through a free intercept and independently centered lag columns. [affine-cases.json](exact/affine-cases.json) retains deployed Scryer studies and six confirmed component claims, including a general real translation identity and a finite rank obstruction to homogeneous fitting. [affine-forecast.json](affine-forecast.json) retains translated/untranslated controls in six synthetic families and real-signal forecasting results. The extension helps translated motion; the original homogeneous fit wins the real-data aggregate comparison. No whole-codec proof or new affine packet format is claimed.

The [NPC data and stress study](../docs/briefs/npc-data-and-stress.md) measures fixed training-stream calibration, deterministic continuation, 100,000-sample histories, growing state/action vocabularies and an analytical planning control. [stress.json](npc/stress.json) retains those results. [real-signals.json](npc/real-signals.json) records the UCI HAR official subject-separated split, all six supervised geometric-readout comparisons, validation choices, per-subject outcomes and confusion matrices. Dataset bytes and trained readouts are excluded. This is representation research, distinct from the NPC's action-learning evidence.

The [NPC evidence](npc/minecraft.json) retains successful real-game learning, temporal ablation, checkpoint transfer and adaptation, followed by a second fresh Minecraft process using the learned seed. Passing JUnit reports and native identical-current-witness controls accompany the measured outcomes. [NPC model and host contract](../docs/reference/NPC.md), [Minecraft reproduction and limits](../docs/reference/MINECRAFT-NPC.md). World saves, working memories and complete command logs remain outside version control.

The [projection/continuation investigation](../docs/briefs/projection-and-continuation-audit.md) examines coordinate mixing, nonlinear warps, missing offsets, coefficient averaging, switched stability and transient gain. Its [numerical receipt](projection-audit.json) and [Scryer component findings](exact/projection-cases.json) distinguish model limitations, finite counterexamples and a bounded negative search. It changes no production algorithms.

The [learning reference](../docs/reference/LEARNING.md) documents maintained mechanisms and examples: streaming training, byte coordinate learning/refinement, full-statistic well networks, consolidation, per-well/meta/graph dreams, bounded feedback, relationship retrieval and optional real-device fitting. [integration-map.json](integration-map.json) inventories 202 original portfolio text sources by mechanism, including Engram-related files outside the old `engram/` directory. It preserves hashes and dispositions without importing host-specific procedures into the supported surface.

[network-demo.json](network-demo.json) retains the synthetic held-out experiment and its input/source identities. Rebuild it and the measured README figure with `python tools/build_network_demo.py`. The local well lab runs these mechanisms interactively. The portfolio crystallography audit's negative findings inform centered-capture and shuffle controls here; its transformer executions and model-level conclusions are not reproduced in this repository.

`portfolio/engram/` contains the source-only historical research workspace:

| Area | Purpose | Requirements / authority |
|---|---|---|
| `experiments/`, `experiment_*.py` | Geometry, dynamics, generation, feedback and language-model explorations | Historical exploration; often external embeddings, data or a local model. |
| `ingest*`, `hetero*`, `sources.py` | Historical ingestion and memory formation | Portable mechanisms now supported by the trainer; the three original Alexandria/genesis run scripts are retired. |
| `gpu/` | wgpu engines, probes and WGSL shaders | Optional wgpu/device; not verified by offline library acceptance. |
| `mass/` | CPU/GPU batching and throughput variants | Benchmark/research code; environment-specific dependencies. |
| `test_*.py` | Historical experiment validation | Different generations of APIs; not the maintained default test suite. |
| `ENGRAM_FORMAT_V1.md` | Historical format proposal | Current formats are documented in `docs/reference/FORMATS.md`. |

Inspect a script's imports, endpoints, datasets and output paths before running it. These programs can download material, call localhost models, use GPU memory or mutate brain files. Preserving a script does not assert that it is portable or that its findings were replicated here.

The repository excludes the historical `alexandria.py`, `genesis.py` and `genesis_chain.py` run procedures. Their hashes and dispositions are listed under `retired_files` in [provenance](provenance.json); the portfolio is their source lineage. Historical experiments importing those procedures require adaptation. Maintained corpus workflows use `tools/train_memory.py` and `engram_training`; the research runner does not restore retired dependencies.

Historical scripts may import `engram_codec` and a namespace named `engram` for research helpers. After inspecting a script, `python tools/run_research.py engram/<script.py> [arguments]` supplies that import topology. It executes the requested program without sandboxing or changing its behavior. No research dependency is silently installed by the library. Absolute dataset paths in migrated scripts use a Downloads-folder default; override inputs as needed.

`tools/parity.py` is the maintained finite-case packet comparison; `tools/build_instrument.py` is the reproducible measured gallery. RDD owns methodology and the distinction between numerical observations, hypotheses and claims about a task.

`exact/oscillator-cases.json` preserves deployed Stithy Scryer studies of four real companion matrices and eight confirmed scalar checks. It includes the engine's unrelated-reading declines and invocation-reported version, not credentials or local connection configuration. The [dynamics reference](../docs/reference/DYNAMICS.md) states the scope: exact component findings, with no replayed proof DAG or whole-codec certification.

Cross-language kernel parity is grounded in source comparisons. Maintained learning paths have synthetic training and held-out evaluation; the original external-corpus training campaign has not been rerun. `tests/fixtures/source-kernels.json` retains actual Manifold Dart and WDYM Java kernel outputs on identical float32 inputs, with source hashes and revisions. Both maintained runtimes check those receipts; Java's float32 accumulation uses a stated wider tolerance than Dart/Rust/Python float64 fitting. Historical packet fixtures pin earlier wire behavior. The [technical design](../docs/TDD.md) owns the selected contracts and supported runtime profiles.
