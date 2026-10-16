# Engram: contributor map

A trajectory codec built from paired complex AR(2) oscillators, with statistical memory wells. This repository owns the reference contracts. Python is the complete functional reference for memory/generation and cross-format import; Rust implements the shared trajectory/signature/replay contract and a smaller native memory profile. Use the capability map in TDD rather than assuming every API exists in every language.

- `README.md`: purpose, installation, runnable examples and navigation.
- `docs/TDD.md`, `docs/RDD.md`: technical model and research intent.
- `docs/reference/`: mathematical construction, API and format boundaries.
- `docs/STATUS.md`: current limitations and useful next work.
- `src/`, `tests/`: Rust runtime, unit tests and self-contained integration fixtures.
- `engram_codec.py`, `engram_wire.py`, `engram_analysis.py`, `engram_memory_io.py`, `engram_embeddings.py`: Python reference facade, replay, diagnostics, memory import and input adapters.
- `engram_network.py`, `engram_training.py`, `engram_retrieval.py`, `engram_metrics.py`: supported geometric learning, streaming training, relationships and evaluation. `docs/reference/LEARNING.md` owns these contracts.
- `engram_acceleration.py`, `engram_kernels/`: optional WebGPU signature fitting; CPU remains authoritative for ordered memory updates. Retain the portfolio kernel's license.
- `engram_corpus.py`, `tools/train_memory.py`: local streaming corpus readers and reusable training/resume command. Retired Alexandria/genesis procedures stay retired; the original research artifact stays frozen.
- `engram_npc.py`: streaming witness fits, learned experience/action consequences, planning, tailored checkpoints and explicit dynamics priors. `docs/reference/NPC.md` owns the host contract.
- `engram_affine.py`: explicit affine AR(2) analysis/forecasting with an unpenalized intercept, causal seeds and snapshots from streaming history. `docs/reference/AFFINE.md` owns the contract; this does not add an affine V4 packet or change NPC policy features.
- `engram_history.py`: root ordered witness summaries, prefix/suffix endpoints, decayed concatenation and versioned snapshots; NPCs may declare native temporal banks. `docs/reference/HISTORY.md` owns the distinction between fit evidence and sufficient decision state.
- `engram_stability.py`: strict reflection-coordinate generation views and checked coefficient rounding. Learned statistics are never replaced by transformed-space averages. `docs/reference/DYNAMICS.md` owns the proof and switching boundaries.
- `engram_relations.py`: schema-bound per-context means/scatter and read-only covariance/correlation views, independent merges and versioned dictionary snapshots. `docs/reference/RELATIONS.md` owns the contract; this does not implicitly change well routing or NPC archives.
- `integrations/minecraft/`, `tools/npc_lane.py`: optional isolated real-Minecraft lane. Consult `docs/reference/MINECRAFT-NPC.md`, use the machine's shared heavy-build lock, and never stop unrelated JVMs or use production worlds. Live sources must remain unchanged during a sealed run.
- `tests/python/`, `tests/fixtures/source-kernels.json`: offline acceptance and actual Java/Dart kernel receipts.
- `instrument/`: generated offline explorer; rebuild using `tools/build_instrument.py`.
- `instrument/wells.html`, `tools/serve_lab.py`: live reference well lab, loopback only. `tools/build_network_demo.py` regenerates the measured README figure/receipt.
- `examples/`, `tools/`: runnable integration and repeatable verification.
- `research/`, `legacy/`, `integrations/`: provenance, experiments, historical implementations and consumer snapshots. These are not automatically supported APIs.
- `research/experiments/relational_*.py`, `tests/research/`: opt-in relational-memory, exact topology and native-world studies, with fast offline evidence checks. The default verification runs those checks without launching games or mass campaigns; `docs/briefs/relational-memory.md` owns their scope.

## Verify

```sh
python -m pip install -e ".[dev]"
python tools/verify.py
npm ci --ignore-scripts
npm test
cargo clippy --all-targets --locked -- -D warnings
cargo package --locked --allow-dirty
python -m pip wheel . --no-deps -w dist
python tools/install_smoke.py
```

The verification runner uses explicit offline test targets, examples, cross-language checks and instrument regeneration. Do not recursively collect the research tree: some scripts download corpora, call local models or write long-lived brain files. Use disposable files for persistence tests. Optional research dependencies and hardware are separate from library acceptance.

`python tools/verify.py --gpu` explicitly runs real-device fitting after installing `.[gpu]`. `npm test` also starts and closes a temporary local well-lab process. The research integration map accounts for original portfolio sources by mechanism; it does not certify every historical script or transformer experiment. Do not silently change coordinates of accumulated wells, feed dreams back as training, or invent chronological order when merging separate per-well histories.

## Invariants and working style

Preserve the AR(2) recurrence, causal seeds, pairing and reconstruction semantics. V4 packets carry explicit block state and their basis. Compact packets cannot reconstruct; legacy compact packets need an explicitly attached original basis before reuse. ZIP v8 preserves the functional reference state; older ZIP/ENBR/ENDB imports must not invent missing statistics. Numerical agreement is tolerance-based and limited to measured cases. Keep Alexandria's frozen original bytes intact.

Document behavioral changes with discriminating regressions. Keep source comments about non-obvious constraints and intent. Update the document that owns a changed contract; put one-off investigations in `docs/briefs/`. Provenance hashes identify extraction inputs, not a promise that current files remain byte-identical.

Preserve license notices. Do not track credentials, corpora, working brain states, machine-specific paths, dependency trees or build outputs. The owner-selected frozen Alexandria snapshot under `research/artifacts/` is the explicit research-artifact exception; keep its hash and context in the existing research index and provenance manifest. Keep registry publication disabled. Public hosting, releases, consumer migrations and license changes require their own user authorization.
