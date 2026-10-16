# Applications and representation choices

Engram describes motion through a coordinate system. A useful application starts by deciding what one row means, how time advances, and which distinctions the coordinates should retain.

## Choose a representation

| Application | One sample | Useful output | Host responsibility |
|---|---|---|---|
| Drawn gestures | A normalized 2D point or an even-dimensional motion feature vector | K/G signatures, prediction error, memory distances | Capture boundaries, normalization, cadence, time alignment and recognition thresholds. |
| Sensors and telemetry | Simultaneous channels in consistent units and order | Blocks, modes, residual measurements, decoded signal | Missing values, sampling/resampling, unit scaling, error budget and downstream decisions. |
| Text memory | Byte-frequency windows or external embedding rows | Well geometry, drift/texture and history | Windowing/tokenization, embedding version, shared pairing, labels and task evaluation. |
| Procedural continuation | An observed motion/control trajectory | Generated candidate trajectory from parameter histories | Allowed range, rate limits, semantics, acceptance and fallback. |
| Codec experiments | A pinned synthetic or recorded trajectory | Full/compact size, modes, fit and final decoded error | Baselines, workload selection, timing protocol and reproducibility. |

Dimensions are inputs to paired oscillators, not independent labelled output channels. Two trajectories encoded with unrelated pairings or embedding functions do not automatically inhabit the same parameter space.

## Existing integrations

**Neuron** contains the Rust library's source lineage. Its `rhythm.rs` uses fitting and prediction primitives; `twin.rs` integrates `Brain` and snapshots into a rhythm-interaction system. Neuron's gesture and device behavior also includes other primitives and host logic. Do not attribute the complete recognition/dispatch system to Engram. Exact extraction revision: `research/provenance.json`.

**What Do You Mean?** contains a Java memory implementation and service integration. The preserved component snapshot is under `integrations/wdym`; the surrounding application remains responsible for Minecraft context, embeddings, prompts, state and actions. The snapshot is not a standalone supported Java distribution.

**[Manifold](https://github.com/worflor/git-desktop-premium-ultra-promax-plus-R)** uses Engram-derived K/G features, together with identifier overlap, when weighting code-hunk relationships. It loads a frozen Alexandria memory and a separate embedding vocabulary. The original [Alexandria research artifact](../../research/README.md#alexandria-research-artifact) behind its bundled export is retained here; the hunk-graph decisions and app-specific representation stay with Manifold.

These integrations establish use of the object. Package-level accuracy and consumer migration require separate validation.

## Runnable demonstrations

`examples/well_network.py` learns an unlabeled network of oscillator wells from synthetic motion and tests unseen phases/amplitudes with time-shuffle and forecast baselines. `tools/serve_lab.py` exposes the actual reference through a live local instrument. `examples/code_retrieval.py` retrieves actual authored code hunks with learned byte coordinates plus explicit identifier evidence; `examples/train_bytes.py` demonstrates reusable training, checkpointing and coordinate refinement. These supported paths and their evidence limits live in [LEARNING](LEARNING.md).

`examples/motion.rs` compares K/G signatures of simple 2D paths after explicit centering and scale normalization. The query uses the same number of samples and cadence as the references. It demonstrates a host-defined comparison, not calibrated gesture recognition or time-warp invariance.

`examples/text_memory.py` measures and absorbs authored offline text under named wells, then saves and restores a Python memory. Labels are supplied by the example. Byte-frequency similarities can reflect language, formatting or character distributions rather than topic; no semantic retrieval accuracy is asserted.

`examples/continuation.py` encodes synthetic motions with a common pairing and generates a short Python continuation using its own seeded random source. It reports finite output and scale. Any usefulness beyond that is an evaluation question.

The general trajectory examples show packet reconstruction and memory persistence. Run the commands in [examples/README.md](../../examples/README.md); each program uses disposable state or no persistent state.

## Measure before adopting

For compression, compare total serialized bytes with raw storage and measure final decoded RMS/maximum error. Predictive capture alone does not answer either question. For matching or memory, use held-out examples and measure the host task separately from internal geometry. For generation, assess the candidate against the host's actual constraints and goals.

The [instrument](../../instrument/engram.html) reports actual current reconstruction errors for its finite gallery. Applications requiring a bound can supply an explicit error budget and accept raw fallback. The [research design](../RDD.md) owns evidence standards; [STATUS](../STATUS.md) tracks remaining limits.
