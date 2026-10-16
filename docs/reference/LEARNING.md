# Learning with wells

An Engram well can do more than measure a distance. Its K/G centroid is an oscillator, its residual RMS describes a noise scale, and its outgoing transition counts form a directed network. The maintained Python learning path turns these pieces into an online geometric model.

## Run the object

```sh
python examples/well_network.py
python tools/serve_lab.py
```

Open the local address printed by the second command. Train, change the novelty radius, probe a new phase/amplitude, choose a well, add residual noise, walk remembered edges, and export both measured session data and the learned `.engram` checkpoint. The lab binds to `127.0.0.1`. Its page runs the actual Python reference through that local process. The separate offline codec explorer remains a stored measurement gallery.

The default experiment learns 120 trajectories from three two-mode oscillator families **without class labels**. The training clusters receive majority labels afterward, solely for evaluation. Sixty held-out trajectories use new random phases, amplitudes, coefficient jitter and observation noise. All use the same sampling cadence and identity pairing. These controlled families are a useful test of motion dynamics; they do not represent the full diversity of physical motion or language.

The retained [measurement receipt](../../research/network-demo.json) reports recognition, confusion, time-shuffle control, learned graph, and actual versus generated positions. Its default run recognized 60/60 held-out inputs; shuffled time reduced recognition to 20/60. Mean 24-step forecast RMS was approximately 0.021 for learned wells, 0.018 for fitting the observed context directly, and 0.830 for persistence. Sharing memory did **not** beat the local-fit baseline on this run. Increasing the novelty radius to 2 collapses the network to one well and recognition to one third; that failure is a regression case too.

## The learning rule

`WellNetwork` fits one whole-trajectory AR(2) `Signature` per observation. For node j, compare its running K/G/RMS means with the new signature:

```text
d_j² = mean_pairs(|K − K_j|² + |G − G_j|² + residual_weight · (RMS − RMS_j)²)
```

If the nearest distance exceeds `novelty_radius`, create a node. Otherwise update that node's sufficient statistics. An explicit `well=` name can select a node instead. Nearest-node choice uses distance; radial activations use normalized `exp(−d²/(2·temperature²))`. Those activations are inspectable weights, not calibrated confidence. Sampling, units, normalization, embedding and these hyperparameters determine the geometry.

This is online prototype learning with oscillator nodes, with no backpropagation. The number of nodes can grow. It uses running centroids for routing; the older `EngramFile` automatic router uses fixed anchors and a maturity/escape rule. Both update the same complete memory state, but they are distinct routing policies. Do not compare their node counts as though they ran the same learning algorithm. Codec packet absorption also weights fitted blocks, whereas this network counts one signature per observation.

```python
import numpy as np
from engram_network import WellNetwork

network = WellNetwork(4, novelty_radius=.15, temperature=.1)
trajectory = np.sin(np.arange(72)[:, None] * .12 + np.arange(4)).astype('float32')
well = network.learn(trajectory)
activations = network.activate(trajectory)  # no learning
future = network.dream(trajectory, steps=24, rng=np.random.default_rng(42))
network.save('motion.engram')
restored = WellNetwork.load('motion.engram')
```

Network observations need at least three rows. Two-row AR(1) signatures remain supported by the lower-level signature API, but are not mixed with AR(2) nodes. Use a fixed pairing. A loaded historical well with missing G or RMS cannot silently become a complete node.

## Generation modes

The explicit `network.dream(context, blend={'well_a': 1, 'well_b': 1})` view interpolates strict stable wells in reflection coordinates. `network.blend(weights)` returns its K/G and weighted residual-power RMS without changing observed statistics. At the default unit radius, static/boundary/unstable inputs are rejected; a frozen blend cannot simultaneously walk the graph. The [dynamics reference](DYNAMICS.md#strict-stable-generation-coordinates) gives the complex proof, numerical margin/rounding checks and the distinction between frozen stability and switching. This view does not replace statistical consolidation or the existing dream defaults.

Pass `radius=ρ` to `blend`, or `blend_radius=ρ` to a blended dream, to declare a different root disk. Its inputs must already satisfy that bound. Values above one explicitly permit growing modes; the default remains one. A blended dream records both source weights and the requested radius. The existing `max_radius` option remains a separate coefficient projection.

- **Parameter-history dream:** `Engram.dream` fits meta-oscillators to recent K/G/RMS histories, predicts parameters and generates trajectory blocks with Laplacian residuals. It can return `None` when history is insufficient.
- **Per-well continuation:** `WellNetwork.dream` uses the chosen node's actual K/G means and the last **two observed positions**. Output contains only future rows, in the original coordinate order. Optional complex Gaussian innovation has expected pair RMS `noise × learned RMS`. K is a dynamics parameter, not a spatial attractor or a remembered position.
- **Graph-guided dream:** supply `walk_every=N` to choose the next node from learned edge frequencies every N future samples. Position carries across the switch. This is procedural generation; a switched itinerary need not forecast the unchanged held-out signal accurately.

`EngramFile.per_well_meta` fits lazily from that well's own recent K history, with at least three entries. `per_well_dream_predict` forecasts the next **K signature**, not the next position. A new observation invalidates that derived fit. `crystallize_dream` fits both global and available local histories.

`max_radius=.999` explicitly projects roots above the requested radius and reports whether coefficients changed. No projection is applied by default. Finite range overflow raises an error. Small root radius does not exclude transient amplification, particularly near repeated roots. Dreams do not automatically train the model or reconstruct the original corpus.

## Train a corpus, resume and refine

`Trainer.fit` accepts an iterable of `TrainingItem(key, payload, well=None, sequence_start=True)` and yields committed `TrainingEvent`s. A `LookupSource` converts caller-tokenized vocabulary indices to trajectories and optionally applies a moving average. This covers byte, word-vector, code-token and other discrete inputs. Existing `EmbeddingTable` imports GLV1 vectors from a caller-supplied file; no vocabulary is bundled.

```python
from engram_training import LookupSource, Trainer, TrainingItem, learn_byte_embedding
from engram_network import WellNetwork

documents = ['the river moves through the valley ' * 20,
             'for key in table: update(key)\n' * 20]
matrix = learn_byte_embedding(documents, dim=16)
source = LookupSource.bytes(matrix, window=4)
network = WellNetwork(16)
events = list(Trainer(network, source).fit(
    (TrainingItem(str(i), text) for i, text in enumerate(documents)),
    batch_size=32, workers=2))
```

Byte coordinates come from document-local symmetric co-occurrence counts, positive PMI and truncated SVD. Unseen byte rows remain zero. Signs are oriented for reproducibility, but degenerate singular subspaces can rotate across linear algebra implementations. Coordinates are learned on the **training split only** for any held-out task.

Feature extraction can use CPU workers; commits retain input order. Batches are bounded by item count, with per-trajectory scalar bounds. A malformed item rejects its whole prepared batch before learning. Short items are explicitly skipped. Callbacks run after each commit. Set `sequence_start=False` only where adjacent records belong to the same temporal episode; independent documents default to breaking graph edges. Identifiers do not deduplicate data or magically resume corpus offsets: the caller owns cursor/order and exactly-once policy.

Network checkpoints atomically replace a file and preserve the reference memory, graph, pairing, recent histories and network configuration in ZIP v8. Source matrix, shape, window and source name are fingerprinted in training metadata. Caller-defined tokenizers also need a stable source name/version and separately saved implementation. Reload with the same representation before continuing; changed coordinates require a fresh memory.

`refine_byte_embedding(memory, previous)` uses accumulated byte composition to propose new coordinates from well geometry. It preserves unseen bytes through a prior. It never changes an existing model in place. Train a **new** network in the proposed space, then compare held-out performance. No monotonic purity or convergence guarantee is assumed. `examples/train_bytes.py` demonstrates both passes and disposable checkpoints on authored text, without importing the historical Alexandria script, its corpus schedule or its intermediates.

For an actual local corpus, use the maintained command:

```sh
python tools/train_memory.py --input my-notes --format text --output notes.engram --corpus-id notes-v1
python tools/train_memory.py --input my-notes --format text --output notes.engram --corpus-id notes-v1 --resume
```

Inputs may be a text file/directory, JSONL (`text` and optional string `id`), or Wikipedia XML/bz2 (`--format wiki`). Readers stream independent documents, reject oversized text and skip wiki redirects. They perform no downloads. The command learns byte coordinates from a bounded initial sample, saves `notes.embedding.npz` beside the memory, and checkpoints a committed document cursor. `--limit N` visits at most N additional documents, including reported short/skipped items; `--gpu` explicitly selects the optional fitter. Resume uses the saved coordinates/configuration and rejects a different corpus ID/format. Keep the corpus and iteration order unchanged under an ID; the ID is caller-declared, not a corpus content certificate. After interruption, resume from the last fully saved checkpoint. The paired embedding file is needed to reproduce the representation. Neither output belongs in Git by default.

`network.consolidate(radius)` explicitly merges nearby nodes and remaps graph counts, preserving K/G/RMS mass and raw composition. Local histories for merged nodes are dropped because their relative chronological order cannot be recovered from separate bounded buffers. Evaluate the changed model before retaining it. `describe()` derives numeric spectral/topology descriptions; it does not invent semantic names.

## Accelerate without changing the brain

```sh
python -m pip install -e ".[dev,gpu]"
python examples/gpu_training.py
python tools/verify.py --gpu
```

`GpuFitter` packages the portfolio's coalesced per-pair fitting kernel and two-pass residual calculation. `Trainer.fit(..., fitter=GpuFitter())` accelerates a batch's signatures, then performs the same ordered reference-memory updates on CPU. No alternate GPU memory state or silently different router is used. Buffers are bounded by both library and device limits and destroyed after fitting. Float32 normal equations are approximate, especially for near-collinear modes; use the float64 reference when precision matters. There is no automatic fallback or unmeasured throughput claim.

On the recorded Windows AMD Radeon/Vulkan run, 26 fitting cases had a maximum K/G/RMS difference of 0.000679; 24 ordered training assignments and graph edges matched CPU. That evidence concerns this kernel, workload and adapter. The historical fused segmentation, gather and routing shader variants remain research comparisons, not additional supported backends.

Nine additional minimum/short/long observations exercise nonidentity pairings at dimensions 2, 8 and 256. Their K/G RMS differences were below 0.000622. Optional Numba acceptance is also run separately from the no-Numba path; neither accelerator is required by the base installation.

## Retrieve relationships and close a loop

`SignatureIndex` retains document keys and signatures. `search` exposes K cosine, G cosine through their joint positive geometric score, K/G RMS distance and optional identifier Jaccard overlap. `lexical_weight` makes the blend explicit. `code_tokens` handles camelCase, acronym boundaries, underscores and digits following Manifold's identifier preprocessing idea. `examples/hunk_search.py` isolates relationship scoring with authored trajectories; real code retrieval supplies vocabulary trajectories and needs a corpus-specific evaluation. A memory well does not contain recoverable document passages, so hosts retain evidence content themselves.

`examples/code_retrieval.py` runs actual authored Python hunks through byte coordinates learned only from indexed code, then ranks three separate queries and returns retained source evidence. All three expected matches are checked. This demonstrates a complete retrieval path on a tiny corpus, with geometric and identifier contributions visible; it is not a general code-search accuracy claim. `examples/alexandria_snapshot.py` also loads the real frozen artifact and generates finite coordinates from its existing recent-history buffer, without retraining or changing the artifact.

`Loop` and `PartnerLoop` preserve one pairing, accept independent RNGs and retain a bounded packet pool (`capacity=64` by default). `EchoWorld`, `MirrorWorld` and `TrajectoryFeed` provide reproducible feedback fixtures. `examples/feedback.py` prints both reconstruction surprise and dream/observation error: low reconstruction surprise alone does not make a good forecast. Loop diagnostic records grow with the requested run; packet retention is bounded.

## Evaluation contracts

`forecast_metrics` reports RMSE, persistence error, mean-baseline error, centered R² and energy capture separately. Centered R² can be negative and is undefined for a constant target. `permutation_test` tests original entity correspondence against a shuffled-right correlation null. Independently sorting both sides creates rank correspondence and invalidates that test.

These controls incorporate lessons from the portfolio's transformer audit: a large constant offset can produce excellent energy capture while explaining none of the varying signal. Transformer compute-skipping, local-model prompting and language quality remain consumer experiments with additional dependencies. The [integration inventory](../../research/integration-map.json) records their relationship to the supported mechanisms; preserving or indexing an experiment does not reproduce its conclusions.
