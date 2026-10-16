# Runnable examples

Run from the repository root after installing the Python development package. All examples use synthetic or authored local data and need no model, GPU or corpus download.

| Example | Command | What to inspect |
|---|---|---|
| Rust trajectory and memory | `cargo run --locked --example trajectory` | Full packet bytes, decoded RMS and a restored memory. |
| Python trajectory and memory | `python examples/trajectory.py` | Full packet reconstruction and a disposable ZIP memory. |
| Motion signatures | `cargo run --locked --example motion` | K/G distances after host-defined normalization under a fixed sampling convention. |
| Text wells | `python examples/text_memory.py` | Measured drift/texture, supplied source labels and persistence of accumulated statistics. |
| Parameter-history continuation | `python examples/continuation.py` | Whether enough history yields a finite generated trajectory and how its scale compares with the source. |
| Geometric learning network | `python examples/well_network.py` | Unlabeled training, 60 held-out inputs, confusion, time-shuffle control and well/local-fit/persistence forecast errors. |
| Live well lab | `python tools/serve_lab.py` | Open its local address; train, probe, select oscillators, dream along graph edges and export a session/checkpoint. |
| Alexandria-style training | `python examples/train_bytes.py` | Learn byte coordinates, stream documents, checkpoint/resume, propose refined coordinates and train a fresh model. |
| Manifold-style relationships | `python examples/hunk_search.py` | Geometric scores, distances and explicit identifier overlap. Authored signals isolate the scoring mechanism. |
| Actual code-hunk retrieval | `python examples/code_retrieval.py` | Byte coordinates learned from indexed authored code, held-out queries, scored relationships and retained source evidence. |
| Frozen Alexandria | `python examples/alexandria_snapshot.py` | Load the original 225-well memory, inspect counters and generate finite coordinates from recent history without changing its bytes. |
| Feedback | `python examples/feedback.py` | Dream/observation error separately from reconstruction surprise, shared pairing and bounded packet retention. |
| Experience NPC | `python examples/npc_experience.py --training 120 --planning 5` | A delayed cue, response learning through fitted history, frozen evaluation and a retained action/experience graph. |
| Motion with an origin | `python examples/affine_motion.py` | Explicit affine continuation of a translated orbit, homogeneous baseline and a measured translation-defect check. |
| Temporal distinctions | `python examples/temporal_witness.py` | An exact cumulative-fit collision, declared recency goal and actual learning with a native temporal bank. |
| Stable generation view | `python examples/oscillator_blend.py` | An unstable coefficient average, strict reflection interpolation, retained learned counts and visible transient gain. |
| Relational evidence | `python examples/relational_wells.py` | Streamed role-conditioned correlations, full retained levels, checkpoint restoration and an unknown-context control. |
| Alexandria dynamics prior | `python examples/npc_seeded.py --vocabulary <matching-glove300.bin>` | Read-only affinities in the supplied vocabulary coordinates; distinguishes general dynamics from learned environment actions. |
| Optional GPU training | `python examples/gpu_training.py` | Requires `.[gpu]`; actual device fitting versus CPU, then identical ordered learning/transition updates. |

Output numbers can vary with runtime and floating-point libraries. Generative examples supply independent seeded generators. The well-network example is a controlled synthetic recognition/forecast benchmark; its numbers describe those signal families. The other examples demonstrate interfaces and choices. None establishes semantic retrieval or general forecasting quality. Memory examples use temporary directories; generated continuations are not applied to hardware or a host system.

The [application guide](../docs/reference/APPLICATIONS.md) explains representation choices and the [memory reference](../docs/reference/MEMORY.md) distinguishes measurement, learning and generation.

The [learning reference](../docs/reference/LEARNING.md) owns input contracts, training/resume rules, graph dreams, GPU precision and evaluation controls. Changing the network demo's novelty radius to 2 intentionally demonstrates one-well collapse; increasing observation noise exposes the limits of fitted dynamics rather than hiding them.

The [NPC reference](../docs/reference/NPC.md) documents embedding this memory in a game or other environment. The [Minecraft laboratory](../docs/reference/MINECRAFT-NPC.md) runs the same reference controller against a real villager and native world observations; it needs separate game/JDK dependencies and a scratch world.

For reproducible NPC stress and real sensor recordings, follow the [data and stress study](../docs/briefs/npc-data-and-stress.md). The real-data runner requires an explicitly downloaded, hash-pinned UCI archive and keeps it outside version control. It evaluates a supervised geometric readout of representations, separately from NPC action learning.

To train your own local corpus, run `python tools/train_memory.py --input <text-directory> --format text --output <memory.engram> --corpus-id <dataset-version>`. JSONL and wiki XML/bz2 are also supported; `--resume` uses a saved document cursor and paired embedding file. See the learning reference before changing coordinates or corpus order.
