# Capabilities and limits

This page summarizes the maintained implementation and points to the contracts and evidence that define its scope. Commands and environment requirements are in [Technical design](TDD.md); source identities are in [research provenance](../research/provenance.json).

## Mathematical object

Engram represents trajectories with paired complex coordinates and second-order oscillator fits. Packets retain enough information for approximate reconstruction; memory wells accumulate statistics in a shared coordinate basis. The Python reference adds analysis views and learning systems around this core. Those layers have separate guarantees and should not be treated as one universal predictor.

## Available capabilities

- **Trajectory codec:** Rust and Python support paired AR(2), an explicit short-observation AR(1) path, segmentation, prediction, full and compact V4 packets, streaming and optional absolute reconstruction-error fallback.
- **Analysis:** K/G signatures, characteristic roots, residual measures, affine fits with an explicit intercept, and checked Schur-coordinate generation blends.
- **Memory:** Rust `Brain` provides a smaller K-well and recent-history profile. Python `EngramFile` supports fuller per-well statistics, routing anchors, transitions, histories, metadata and parameter-history generation. Supported memory imports include ZIP v4–v8, ENBR v1 and ENDB v2.
- **Temporal experience:** Python `WitnessHistory` streams centered lag moments, supports selected decay scales and ordered summary concatenation. `EngramNPC` associates witness contexts with learned actions, outcome models and observed-model planning.
- **Relational evidence:** Python `ContextRelations` retains context-indexed levels and centered scatter, with covariance/correlation projections and independent compatible merges. It is a mathematical summary, not an emotional or semantic policy.
- **Learning and research tools:** Python includes geometric well networks, training and refinement paths, retrieval, feedback, evaluation controls, a local lab, and an optional WebGPU signature fitter.
- **Examples and application evidence:** motion, text, retrieval, feedback, NPC and Minecraft tasks have runnable examples or pinned integration receipts. Alexandria is retained as a frozen research artifact; its training process and corpus are not included.

Detailed interfaces live in the [API](reference/API.md), [formats](reference/FORMATS.md), [memory](reference/MEMORY.md), [history](reference/HISTORY.md), [NPC](reference/NPC.md), [relations](reference/RELATIONS.md), and [learning](reference/LEARNING.md) references.

## Implementation boundaries

| Capability | Python reference | Rust library |
|---|---|---|
| Paired AR(2), explicit AR(1), signatures and diagnostics | Yes | Yes |
| Segmentation, streaming, V4 packets and error-budget encoding | Yes | Yes |
| Memory wells and snapshots | Full `EngramFile` profile | Smaller `Brain` profile |
| ZIP v4–v8, ENBR v1 and ENDB v2 imports | Yes | Native ENBR v1 |
| Geometric network, training, retrieval and generation | Yes | Not implemented |
| Streaming witness history, NPC learning and planning | Yes | Not implemented |
| Affine forecasting and ordered summary concatenation | Yes | Not implemented |
| Schur-coordinate generation views | Yes | Not implemented |
| Context-indexed relational moments | Yes | Not implemented |

Python and Rust share packet semantics, not identical in-memory layouts or complete memory features. See [Technical design](TDD.md) for the module and runtime map.

## Limits and evidence

- Coordinate meaning, units, sampling cadence, normalization and action semantics belong to the host. K/G comparisons require a compatible pairing and representation.
- Pairwise oscillator fits do not model arbitrary coupling between all coordinates. Memory summaries are lossy; a task may depend on distinctions that a fit omits.
- Learned wells, graph routes and generated continuations have no universal accuracy, stability or safety guarantee. Stable frozen coefficients do not guarantee stable switching or small transient gain.
- State grows with well count. Packet and history readers enforce finite structural budgets, but this is not a security certification.
- NPC results concern specified controlled tasks. The delayed-cue experiment and Minecraft lane do not establish optimal navigation, arbitrary-world transfer or a complete emotion policy. The standalone NPC is Python-only.
- Real human-motion results are subject-separated readout experiments, not NPC reward-learning results. Simpler statistics outperform oscillator features on some recorded comparisons; see the [data and stress study](briefs/npc-data-and-stress.md).
- The Minecraft lane targets Fabric and Minecraft 1.21.1. It is an integration fixture, not a general mod-loader SDK.
- Historical Scryer checks address selected exact mathematical claims. They do not certify numerical fitting, whole-package behavior or every runtime path.
- Package metadata is 0.4.0; installation is from source. No registry package or hosted release exists.

The [relational fidelity study](briefs/relational-memory.md), [history and composition study](briefs/history-and-composition.md), and [projection and continuation audit](briefs/projection-and-continuation-audit.md) preserve task-specific evidence and counterexamples. Each document states its own experimental domain; none should be generalized beyond it.

## Validation map

Use `python tools/verify.py` for the repository's local acceptance workflow. `npm test` covers the browser lab and downloads, `cargo clippy --all-targets --locked -- -D warnings` checks Rust linting, and wheel plus `tools/install_smoke.py` check installation outside the source tree. Optional GPU validation uses `python tools/verify.py --gpu` on a supported adapter. These commands establish only the checks they execute on the host where they run; a local pass does not imply hosted or cross-platform CI completion. See [TDD](TDD.md#integrations-and-verification) for details.
