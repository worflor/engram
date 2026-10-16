# Technical design

This document describes how the mathematical object is represented across the codec, memory, generation and host-facing layers. Detailed behavior belongs to the linked API references; research evidence and open questions belong to [RDD](RDD.md) and the studies under [briefs](briefs/).

## Mathematical hierarchy

Engram begins with an ordered real trajectory whose coordinates are paired into complex channels. A fitted channel follows

```text
z[n] = K · z[n−1] − G · z[n−2]
```

The pair basis, units and sampling cadence define the interpretation of those coefficients. The first two samples provide initial conditions. Regularized fitting estimates K and G; residuals measure the part not captured by that recurrence. Macro and micro layers can represent broad motion and residual motion. Linear, static and raw modes handle blocks where the oscillator fit is unsuitable.

The resulting data flow is:

```text
trajectory + coordinate basis
        ↓
paired dynamics + initial conditions
        ↓
residual and reconstruction data ──→ packet
        ↓
sufficient statistics in shared coordinates ──→ memory wells
        ↓
explicit analysis/generation views ──→ host task or decision
```

Each arrow loses or adds particular information. A packet preserves approximate trajectory reconstruction. A well preserves selected sufficient statistics, not its source observations. Affine state adds an offset; relational summaries retain simultaneous cross-channel evidence; temporal summaries preserve selected lag structure. NPC policy and outcome records add task experience. None of these views is interchangeable with the codec packet.

## Runtime shape

`engram_codec.py` is the Python facade. Supporting modules separate wire formats, analysis, memory import, embeddings, geometric networks, training, retrieval, metrics, acceleration, witness history, NPC learning, affine fitting, stability views and relational summaries. Rust implements the shared trajectory/signature/replay contracts and its smaller native memory profile. There is no runtime dependency on a host application, network service, model or corpus.

| Capability | Functional Python reference | Native Rust |
|---|---|---|
| Paired AR(2), explicit AR(1), diagnostics and K/G signatures | Yes | Yes |
| Trajectory blocks, streaming, V4 full/compact replay and error budget | Yes | Yes |
| Memory wells and snapshots | Full `EngramFile` model | Smaller `Brain` profile |
| ZIP v4–v8, ENBR v1 and ENDB v2 imports | Yes | Native ENBR v1 |
| Geometric network, training, retrieval and generation | Yes | Not implemented |
| Witness histories, temporal banks and NPC learning/planning | Yes | Not implemented |
| Affine forecast and ordered-history concatenation | Yes | Not implemented |
| Schur-coordinate generation views | Yes | Not implemented |
| Context-indexed relational moments | Yes | Not implemented |
| Optional WebGPU signature fitter | Yes | Not implemented |

These are capability boundaries, not a claim that the two implementations have identical in-memory structures. See [STATUS](STATUS.md) for the concise support summary.

## Trajectory codec

Rust accepts flat row-major `[T × D]` float32 buffers; Python accepts `[T, D]` arrays and converts to float32 input. Fitting uses float64/complex128 accumulation. D must be positive and even. Entry points validate finite data, sizes and coordinate permutations. Rust `try_encode` and `try_decode` return checked errors; convenience entry points assert their contracts.

The codec selects a coordinate pairing from absolute co-movement when one is not supplied. It estimates block scales from velocity autocorrelation and divides at prediction-error changes or the block limit. Fitting solves regularized complex AR(2) least squares; prediction applies the recurrence. Encoding selects cascaded or linear prediction after static/raw checks and quantizes residuals per pair. Decoding restores dynamics and original coordinate order.

Both implementations reserve the first two samples as initial conditions. Micro spans include partial tails. V4 carries each block's actual seeds. Rust stores quantized residual values and steps; Python stores floating residuals in memory and quantizes them for wire serialization. The in-memory layouts differ while V4 wire semantics agree. Operations across independent blocks can run in parallel; recurrence within a pair remains sequential.

Full V4 packets preserve reconstruction state. Compact V4 packets retain pairing and statistics but omit data required for reconstruction. Historical V2 compact packets lack a basis; Python's `with_pairing` attaches the caller's original pairing before serialization or absorption. Unknown coordinates remain unknown. [FORMATS](reference/FORMATS.md) owns compatibility details.

`max_error` and Rust `encode_with_tolerance` measure actual full-wire reconstruction against float32 input. If prediction and quantization exceed the requested absolute error, raw storage preserves those samples exactly, potentially increasing packet size. The default predictor has no universal reconstruction-error bound.

## Memory and learned experience

Memory uses a compatible coordinate pairing for parameter comparisons. `EngramFile` retains Python well statistics, routing anchors, transitions and per-well histories; Rust `Brain` stores a smaller K-centered well and bounded recent-history profile. Storage grows with well count. Drift, texture and nearest-well distance are measurements of the selected geometry, not probabilities or task accuracy.

`engram_history.py` owns `WitnessHistory`, a centered streaming lag-moment summary for triples `(z[t], z[t−1], z[t−2])`. It retains first and last endpoints for ordered concatenation and supports explicit exponential decay. Concatenation merges moment summaries and adds the boundary triples required at the seam; it is associative in exact arithmetic but generally noncommutative. The summary supports its declared fit, not arbitrary chronology queries. See [HISTORY](reference/HISTORY.md).

`engram_npc.py` combines schema-defined witness vectors, temporal history, context wells, action values and observed outcome models. `EngramNPC` learns from environmental feedback and can rehearse an observed model without counting planning as physical experience. `WitnessSchema.calibrate` freezes training-stream means and scales as part of the input representation. The schema, normalization, actions and reward/outcome meaning remain host-owned. Its standalone implementation is Python-only. See [NPC](reference/NPC.md).

`engram_relations.py` retains context-indexed real-channel means and centered scatter. Covariance and correlation are projections over the stored levels and scatter; representation identity is explicit. This companion summary supports odd real dimensions and independent compatible merges. It is not automatically attached to every well or NPC checkpoint. See [RELATIONS](reference/RELATIONS.md).

## Analysis and generation views

`engram_affine.py` fits an explicit intercept, `z[t] = K z[t−1] − G z[t−2] + b`, by centering the lag variables independently. Its anchored float64 forecast and snapshot from `WitnessHistory` preserve the offset state required by translated motion. A bare K/G signature omits that state; the affine model has no V4 packet flag and does not replace the homogeneous fit or NPC policy. See [AFFINE](reference/AFFINE.md).

`engram_stability.py` maps strict stable AR(2) coefficients to Schur/reflection coordinates. Convex blends in the product of disks remain stable as frozen recurrences. `WellNetwork.blend` and `dream(blend=...)` use this as an explicit generation view without changing learned statistics. The optional requested root radius changes the interpolation geometry and allows growing modes when set above one. Rounded coefficients are checked against the requested disk. Frozen-root stability does not ensure small transient gain, forecast quality or safety under arbitrary switching. See [DYNAMICS](reference/DYNAMICS.md).

`engram_network.py` provides radial well learning, consolidation and oscillator/graph dreams. `engram_training.py` provides caller-supplied data adapters, byte-coordinate learning and ordered training/refinement. `engram_retrieval.py` and `engram_metrics.py` provide relationship scoring and evaluation controls. `engram_acceleration.py` packages a bounded WebGPU signature fitter; its results feed the same CPU-authoritative memory updates. See [LEARNING](reference/LEARNING.md) for formulas and limits.

Establish a common pairing and representation before comparing K/G. Arbitrary per-document pairings refer to different coordinate systems. Source labels, embedding functions, action semantics and constraints remain with the host. The term “brain” refers to these data structures; it makes no claim about consciousness or general intelligence.

## Serialization and compatibility

V4 is the shared full/compact packet format. Full coefficients retain float64 precision; compact coefficients use float32. Full raw blocks preserve float32 samples. Python and Rust convert their internal residual forms at the boundary. Historical V3 full and V2 compact packets remain readable. Ambiguous V1 streams require explicit Python producer selection and are rejected by Rust.

Python memory loading supports ZIP v4–v8, ENBR v1 and ENDB v2. Imports identify unavailable statistics instead of fabricating them. ZIP v8 preserves double-precision histories, routing continuation state and supported unknown manifest fields/opaque sections. Rust `Brain` uses ENBR v1.

Readers enforce byte/value/dimension budgets, finite values, valid counts and permutations, coverage, micro spans and exact consumption. Tests include truncated prefixes, hostile lengths and deterministic mutations. This establishes bounded parser behavior, not a general security certification. Public mutable in-memory structures still require valid state.

## Integrations and verification

The Minecraft lane in `integrations/minecraft/` is a pinned development fixture. `tools/npc_lane.py` controls fresh worlds with acknowledged file requests; its Java adapter performs server-thread observations and native movement. Behavioral gates, history-disabled controls, checkpoint transfer, adaptation and JUnit completion provide separate evidence from compilation. See [MINECRAFT-NPC](reference/MINECRAFT-NPC.md).

`python tools/verify.py` runs Rust tests, focused Python acceptance, runnable examples, bidirectional packet checks and instrument regeneration. `npm test` exercises offline gallery and live-lab workflows. `cargo clippy --all-targets --locked -- -D warnings` checks Rust linting. `python -m pip wheel . --no-deps -w dist` and `python tools/install_smoke.py` check the Python distribution outside the source checkout. `python tools/verify.py --gpu` includes the real-device comparison when a supported adapter is available. Research scripts are excluded from automatic test collection; each study has its own dependencies and evidence scope.

`tools/parity.py` compares independent decoders of the same packets and separately reports error against the source. Agreement is not evidence of accurate reconstruction. `tools/build_instrument.py` embeds actual samples, encoded bytes and decoded values; the browser gallery does not execute Python or Rust. `tools/build_network_demo.py` generates the measured network figure and receipt. Optional GPU precision evidence does not imply a throughput advantage or validate every historical shader.

## Information architecture

README introduces the object and its applications. This document defines technical structure and implementation boundaries. [RDD](RDD.md) sets research questions and evidence standards; [STATUS](STATUS.md) summarizes capabilities and limits. Reference pages define APIs and formats. Briefs contain bounded studies with explicit domains; `research/` retains data and provenance; `legacy/` retains historical implementations. Imported source and licensing boundaries are in [LICENSES](../LICENSES/README.md) and the relevant research inventories.
