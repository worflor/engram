# API reference

## Rust

Generate the complete API with `cargo doc --no-deps --open`. The stable orientation points are:

| Module | Entry points | Contract |
|---|---|---|
| `encode` | `encode`, `try_encode`, `encode_with_tolerance` | Finite bounded row-major input; even positive dim; optional permutation and explicit replay error budget. |
| `decode` | `decode`, `try_decode`, `decode_block` | Full-packet reconstruction; checked entry rejects malformed/statistics-only state. |
| `wire` | `to_wire`, `from_wire` | V4 full packets; bounded reads also accept historical V3. |
| `wire` | `to_wire_compact`, `from_wire_compact` | V4 statistics with basis; historical V2 reads lack a known basis. |
| `analysis` | `Signature::from_trajectory`, `compare`, `dynamics`, `fit_ar1` | Shared K/G signatures, disjoint radius diagnostics, envelope half-life/period and short-observation fitting. |
| `fit` / `predict` | `fit_pair`, `fit_all`, `predict_pair`, `predict_all` | Complex AR(2) components with causal initial conditions. |
| `segment` | `derive_pairing`, `derive_block_sizes`, `segment` | Data-derived coordinate and block choices. |
| `histogram` | `text_to_trajectory`, `bytes_to_trajectory` | Non-overlapping byte windows; default 512 bytes, at least four complete windows; trailing incomplete bytes omitted. |
| `brain` | `Brain::new`, `absorb`, `fast_absorb`, `measure` | Shared-coordinate K statistics, well routing, recent history. |
| `brain_io` | `save`, `load` | ENBR v1 brain snapshots; different from Python ZIP brains. |
| `streaming` | `StreamEncoder`, `StreamDecoder` | Ordered block transport; decoder history includes initial seeds and tiny flushes retain all samples. |

`Packet`, `Block`, `Mode` and fitting-result types live in `types`. `Block.micro_size` records the actual micro span; `Packet.reconstructable` and `basis_known` distinguish available information. `with_pairing` attaches the original basis to an unknown-basis historical compact packet. Types remain publicly mutable: checked entry points validate their invariants. `FitAllResult` includes per-pair RMS as well as its mean.

## Python

`engram_codec` is the packet/memory facade. Explicit mathematical views live in
the modules listed below. The distribution name is `woflo-engram`; there is no
`import engram` SDK package. Real observation arrays must be finite and fit the
entry point's precision. Complex arrays reject rather than losing their imaginary
part; complex coordinates are supplied as explicit real/imaginary pairs.

- `Engram(dim=256, alpha=0.0, max_error=None)`: encode `[T,D]` arrays, decode full packets, and generate trajectories using parameter histories. A supplied absolute error budget is checked after full-wire replay relative to the float32 input; raw storage is the exact fallback.
- `encode(trajectory, pairing=None)`: derives pairing unless one is provided. Retain a fixed pairing for cross-observation comparison.
- `to_wire(packet, q_bits=8, compact=False)` and `from_wire(data, q_bits=8, legacy_v1=None)`: V4 full/compact; only eight-bit residual quantization is supported. Read historical V3/V2 directly; historical Python V1 requires `legacy_v1='python'` because old producer conventions differed.
- `ByteHistogram(chunk_size=512).text_to_trajectory(text)`: offline 256-bin byte-frequency input.
- `EngramFile.create(path, name='', dim=256, alpha=.005, metadata=None, annotations=None)`: create memory state in memory; `save()` writes it.
- `EngramFile.load(path_or_bytes)`: import ZIP v4–v8, ENBR v1 or ENDB v2. `import_info` identifies unavailable older fields. `save()` writes ZIP v8; byte imports require an explicit save path. `copy()` creates independent state.
- `absorb(packet, source=None)`, `absorb_flat(K,G,RMS,source=None)` and `absorb_batch_flat(K,G,RMS,boundaries,source=None)` share the same full-statistic routing/update path. Packet pairings must match the memory. Direct feature arrays use the memory's established basis, or identity for an empty memory.
- `measure(codec, observation)` returns geometry measurements and an observation packet when available; codec and memory dimensions must agree. `fast_absorb` fits a shared-basis signature without segmentation, including AR(1) for a two-row observation.
- `measure_text` and `absorb_text`: byte-histogram convenience paths.
- `Signature.from_trajectory(trajectory, pairing=None)`: K/G/RMS and order, with `compare(other)` for matching bases/orders. `dynamics(K,G,tolerance=1e-6)` reports roots, radius, repeated-root flag and exponential-envelope half-life / real conjugate-pair period.
- `EmbeddingTable(tokens,vectors)`, `from_glv(bytes)`, `project(tokens)` and `signature(tokens,pairing,window=10)`: optional vocabulary input; unknown tokens are omitted. Tokenization and vocabulary choice remain host responsibilities.
- `dream(packets, n_blocks=8, samples_per_block=None, rng=None)`: parameter-history generation with an optional independent NumPy generator. All packets must share a known basis; output returns to original coordinate order.
- `merge(a,b,overlap=0,overlap_tolerance=0.0)`: concatenate full packets with preserved block state. Two shared boundary samples are removed only with explicit `overlap=2` and a successful reconstructed-value comparison.

`TanhProjection`, `Identity`, `StreamEncoder`, `StreamDecoder`, `TrajectoryFeed`, `Loop`, `PartnerLoop`, `EchoWorld` and `MirrorWorld` remain available research/integration primitives. Their presence is not a parity commitment with Rust.

## Python mathematical views

- `engram_history.WitnessHistory(dim, decay=1.0)`: `push`, `signature(center=True)`,
  `concat`, `snapshot`, and `restore(dim, decay, data)` retain and compose ordered lag evidence.
  The NPC import re-exports the same class. [History contract](HISTORY.md).
- `engram_affine.AffineOscillator.fit(trajectory, *, pairing=None, relative_ridge=1e-4)`
  and `from_history(history, *, relative_ridge=1e-4)`: explicit intercept fitting,
  `forecast(steps, *, seeds=None)` and `translated(offset)`. [Affine contract](AFFINE.md).
- `engram_stability.schur_coordinates(K,G,radius=1.0,margin=1e-8)` and
  `schur_coefficients(kappa,g,radius=1.0,margin=1e-8)`: strict reflection-coordinate
  views. `blend_schur(K,G,weights,radius=1.0,margin=1e-8,dtype=complex128)` checks
  the requested root disk after rounding. Optional arguments are keyword-only.
  [Generation geometry](DYNAMICS.md#strict-stable-generation-coordinates).
- `engram_relations.ContextRelations(channels, representation=...)`: `push`,
  `moments`, `merge`, `contexts`, `to_dict` and `from_dict` expose context-indexed
  levels and cross-channel moments. This unordered companion state does not
  implicitly change an oscillator or NPC archive. [Relational contract](RELATIONS.md).

The [construction](CONSTRUCTION.md) derives these views and explains why their
summary types and composition operations preserve different information.

## Python learning modules

- `engram_npc.WitnessSchema`, `WitnessHistory`, `EngramNPC`, `Decision` and `DynamicsPrior`: named host observations, incremental centered fitting, experience wells, learned responses, observed-model planning and checkpoint/seed contracts. `EngramNPC.observe`, `decide`, `feedback`, `dream`, `begin_episode`, `responses`, `save` and `load` are the complete host loop. [NPC reference](NPC.md).

- `engram_network.WellNetwork(dim, *, pairing=None, novelty_radius=.15, temperature=.1, residual_weight=1.0, memory=None)`: `learn`, `learn_signature`, `activate`, `activate_signature`, `begin_sequence`, `transition_weights`, `oscillator`, `describe`, `consolidate`, `dream`, atomic `save` and `load`. Learning needs AR(2), at least three input rows, and one common coordinate convention.
- `dream(context, *, well=None, steps=32, noise=0, rng=None, max_radius=None, walk_every=None, blend=None, blend_radius=1.0)` returns `Dream(trajectory,itinerary,original_radius,stabilized,mixture,mixture_radius)`. It continues from the last two observed rows and returns future positions in original coordinate order. A frozen `blend` cannot also select a single well or walk the graph. Its normalized sources and requested disk are recorded in the mixture fields. Generated rows are not implicitly absorbed.
- `blend(weights, *, radius=1.0, margin=1e-8, dtype=complex128)` returns an explicit strict-disk K/G and residual-power view without changing learned well statistics.
- `engram_training.Trainer(network, source=None, source_id=None).fit(items,batch_size=32,workers=1,on_item=None,fitter=None)`: yields committed events. `TrainingItem(key,payload,well=None,sequence_start=True)` gives explicit episode boundaries; default independent documents do not form temporal edges.
- `LookupSource(matrix,to_indices,name='lookup',window=1)`, `LookupSource.bytes`, `smooth_trajectory`, `learn_byte_embedding` and `refine_byte_embedding`: validated sources and explicit coordinate-learning proposals. The caller persists embeddings/tokenizer definitions; source fingerprints reject changed coordinates on resume.
- `engram_retrieval.SignatureIndex(dim,pairing=None).add(key,trajectory,identifiers=())` and `search(trajectory,identifiers=(),lexical_weight=0,limit=5)`: retained keys, joint K/G score, optional Jaccard evidence and distances; `code_tokens` splits identifier conventions.
- `engram_metrics.forecast_metrics(actual,predicted,last_observation=...)` and `permutation_test(left,right,rng=...,repeats=999)`: centered/error/baseline and original-identity shuffle controls.
- `engram_acceleration.GpuFitter().fit(trajectories,dim=...,pairing=None)`: optional GPU float32 batch signatures; `Trainer.fit(fitter=...)` keeps memory commits on the reference path. No device fallback is implicit.

`EngramFile.per_well_meta` fits lazily from sufficient per-well history; new learning invalidates it. Feedback `Loop`/`PartnerLoop` accept `rng` and `capacity` (default 64), preserve their seed pairing and bound retained packets. The [learning reference](LEARNING.md) provides formulas, practical workflows and evidence limits.

NumPy is required. Numba and the maintained WebGPU fitter are optional extras. External embedding dictionaries, corpora and local-model endpoints are caller supplied. `python -m pytest` collects only the maintained offline acceptance suite. `engram_corpus.documents` streams local text/JSONL/wiki XML or bz2 into independent training items; `tools/train_memory.py` provides explicit corpus identity, paired embedding/checkpoint output and cursor-based resume.

Use the [application guide](APPLICATIONS.md) to choose a representation and the [memory reference](MEMORY.md) to distinguish measurement, absorption and generation. [Runnable examples](../../examples/README.md) cover these interfaces without external datasets.
