# Memory and generation

## Measurement and learning

A packet represents one trajectory. A well represents accumulated statistics of fitted blocks from multiple observations. The buffer retains recent parameter histories. These are different forms of state with different information loss.

| Object | Retained evidence | Natural operation |
|---|---|---|
| Full packet | Block laws, causal seeds and residuals | Approximate replay or checked packet concatenation |
| `WitnessHistory` | Per-pair lag means/scatter and boundary observations | Fit a temporal law or concatenate consecutive summaries |
| Well | Weighted fitted K/G/RMS statistics and routing state | Compare observations and accumulate parameter geometry |
| `ContextRelations` | Full real means/scatter within each declared context | Project levels/coupling or merge unordered evidence |
| NPC experience | Current witness, temporal features and observed action consequences | Learn or query host responses |

These states share the [moment construction](CONSTRUCTION.md), but their measures
are different. Averaging fitted coefficients is not equivalent to fitting pooled
lag moments. Correlation is not a replacement for a retained mean, and an ordered
history summary is not an arbitrary policy checkpoint.

Use one representation and coordinate pairing throughout a memory's observations. The memory measurement path derives a reference pairing on its first usable observation and reuses it. If the host encodes packets directly before calling `absorb`, it must enforce that shared convention itself.

`measure` compares an observation with current memory and returns geometry measurements. It does not absorb the observation. It can initialize the reference pairing and update derived caches; Python can also fit a history predictor lazily. Measuring is therefore not a promise of a fully immutable object.

`absorb(packet, source)` changes memory. With a source label, statistics join that named well; without one, the implementation applies its routing rules. A label supplies a category name, not an embedding of the category's meaning. Packet, flat and ordered batch inputs share the same K/G/RMS update path, including fixed anchors, transition counts and per-well histories. Blocks without usable predictive statistics do not necessarily create a well.

## Reading the outputs

| Output | Interpretation | Does not establish |
|---|---|---|
| `drift` | A normalized distance from the observation's fitted K coordinates to a nearby well | Probability of novelty, correctness or relevance. |
| `texture` | `100 − predictive capture`, derived from residual energy | Final wire reconstruction error or a linguistic quality judgement. |
| nearest well / profile | Current geometric neighborhood under this memory's representation | A ground-truth semantic class. |
| dream surprise (Python) | Disagreement with an available parameter-history prediction | A validated prediction of a future event. |

The vocabulary of drift, dreams and brains names these mathematical structures. Host metadata, category labels and measured state should remain distinguishable.

## Runtime differences

Rust `Brain` keeps K sums/counts, a shared pairing and a bounded recent K/G/residual history. Its public measurement also exposes energy and a weighted combined score.

Python `EngramFile` additionally stores G/S and raw-byte statistics, fixed routing anchors, transition counts, per-well histories and meta-oscillators. Its measurement and continuation interfaces cover a larger research surface. Do not expect matching memory states or routing from the two implementations.

Recent-history bounds do not bound the number of wells. Well creation and labels can still grow stored state. These structures summarize observations; they cannot reconstruct the complete original corpus.

## Generation

`Engram.dream(packets, ...)` collects macro K/G/residual histories from packets, fits dynamics to those histories, predicts further parameters and synthesizes trajectory blocks. It needs enough usable history and may return `None` when there is too little. Use a common pairing for the contributing packets.

Synthesis adds sampled Laplacian residual noise. Supply `rng=np.random.default_rng(seed)` for an independent reproducible source; the continuation example does so. The legacy default still uses NumPy's global random source. Generation is not a deterministic consequence of packet history alone. Full-packet generation starts from the last reconstructed endpoint; statistics-only histories have no endpoint and use zero initial conditions. Output returns to the original coordinate order.

The implementation applies parameter and residual-scale guards. They do not constitute a general theorem about decoded accuracy, bounded outputs or meaningful application behavior. Generated rows live in the supplied trajectory representation; text generation requires additional machinery to turn those rows into words.

`EngramFile` can retain history used to construct dream packets and per-well forecasts. These capabilities remain exploratory. A changing internal score or a plausible curve is not evidence that a host's task improved.

The supported `WellNetwork` layer adds direct continuation under learned well oscillators, plus optional graph-driven switching. It starts from real observed positions rather than treating K coefficients as spatial attractors. It shares the reference memory state while using its own explicit radial learning policy. `per_well_meta` fits lazily when enough local history exists; its forecast concerns the next K signature. The [learning reference](LEARNING.md) owns these interfaces, training, checkpoint rules and held-out evaluation.

## Persistence

Rust `brain_io::save/load` uses the smaller ENBR v1 profile. The functional reference's `EngramFile.load` reads ZIP v4–v8, ENBR v1 and Manifold ENDB v2, reporting fields absent from those representations. It saves ZIP v8: existing per-well state plus double-precision recent histories, the previous routed well and basis-initialization state. It preserves supported unknown manifest fields and opaque sections. Importing a smaller representation does not manufacture missing learned G/S or graph history, and there is no lossless export of the full reference state to the smaller native profile.

Save full packets separately if approximate replay of individual trajectories matters. Save the host's preprocessing and embedding definition with its state. Consult [FORMATS](FORMATS.md) for compatibility and [API](API.md) for callable entry points.
