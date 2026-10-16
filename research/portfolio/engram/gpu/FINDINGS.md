# Engram GPU kernel findings

Measured on AMD Radeon iGPU via Vulkan on WebGPU/wgpu. Same WGSL source will run on any WebGPU-compatible backend (Metal, DX12, Vulkan).

## Production status

**Live in the genesis pipeline as of 2026-04-22.** `genesis_chain_gpu.py` drives corpus ingest through `BatchedGpuIngest` (`rag_tests/engram/gpu_ingest.py`), which calls `GpuEngine.fit_batch` → `fit_pair_v3.wgsl` once per batch of 128 articles. End-to-end speedup vs. the pre-existing CPU `genesis_chain.py`: **5-8× on this iGPU** with all three optimization layers live.

The accumulator preserves every invariant the CPU path established:
- same `_reference_pairing` shared across the corpus
- same segment boundaries from `derive_block_sizes` + `segment`
- same `absorb()` sequential well-assignment logic
- same checkpointing cadence (every 5000 absorbs)

### Three stacked speedups

| Layer | What | Measured win | File |
|---|---|---|---|
| GPU fit kernel | `fit_pair_v3.wgsl` replaces `fit_all` on the hot inner loop | **3-6×** end-to-end on the fit stage | `gpu/shaders/fit_pair_v3.wgsl` |
| Threaded CPU prep | `ThreadPoolExecutor(4)` overlaps text-parse + embed + segment with GPU dispatch | **1.3-1.8×** on real articles (where `mwparserfromhell.strip_code` releases GIL); smaller on synthetic dict-heavy workloads | `gpu_ingest.ingest_gpu(n_workers=...)` |
| Matrix-form GloVe | `load_glove_matrix()` replaces dict-of-arrays with contiguous `f32[V, 300]` + `word_to_idx`; sliding window vectorized | **1.91× on embed** (bit-identical output) | `fast_embed.py` |

All three compose. The CPU-prep thread pool benefits most when the embed stage is already fast (layer 3 feeds layer 2), and the overlap between prep and GPU compute (layer 2) only matters because the GPU stage is fast enough to be worth overlapping (layer 1 feeds layer 2).

### Memory-safety fix (2026-04-22 OOM incident)

The first production run crashed on a monster enwiki article — `numpy._core._exceptions._ArrayMemoryError: Unable to allocate 3.66 MiB` inside `absorb()`. A 3.66 MiB failure means the process was at the wall, not that the allocation itself was unreasonable.

Root cause: `_StagedArticle` originally held the **full paired trajectory** `wp` for each article pending in `_pending`. On a 50k-token article that's ~60 MB of f32. With `batch_size=128` plus ~16 more buffered inside `ThreadPoolExecutor`, peak pending memory approached multi-GB. Add the GPU-side buffers and Python overhead, and the iGPU's 4 GB shared-RAM budget saturates.

Fix:
1. **`_StagedArticle` now holds segment slices only** (`List[_StagedSegment]`), built and the parent `wp` discarded before staging. Each segment is capped at `MAX_BLOCK+2 = 130` timesteps, so one segment is ~156 KB regardless of source article size. A typical article stages at 2-5 MB; even pathological monsters at sub-MB.
2. **Token cap in the embed** — `MAX_TOKENS_PER_ARTICLE = 20000` in `fast_embed.py`. Mirrors the implicit 100k-char cap already in `iter_zim_articles`. Cuts off at the source; downstream never sees longer.
3. **Regression guard**: `test_memory_safety.py` stages a batch mixing 120 normal + 8 monster articles (60k tokens each, 3× the cap). Verifies max staged footprint per article stays bounded.

After the fix: monster articles stage at 0.75 MB max (measured), the old bug-triggering batch completes cleanly.

## Directory layout

```
rag_tests/engram/
├── fast_embed.py                   # matrix-form GloVe, vectorized sliding window, 20k-token cap
├── gpu_ingest.py                   # BatchedGpuIngest accumulator + threaded ingest_gpu
├── genesis_chain_gpu.py            # production pipeline entrypoint
├── test_gpu_ingest.py              # end-to-end correctness check
├── test_memory_safety.py           # regression guard for the monster-article OOM
├── bench_embed.py                  # measures fast_embed vs. dict path
├── bench_ingest.py                 # measures threaded ingest speedup
└── gpu/
    ├── engine.py                   # GpuEngine, default shader = fit_pair_v3
    ├── FINDINGS.md                 # this file
    ├── shaders/
    │   ├── fit_pair_v3.wgsl        # production kernel
    │   └── variants/               # exploration record (see ladder below)
    ├── benches/                    # perf harnesses
    │   ├── bench_fit.py            # primary v3-vs-numpy bake-off
    │   ├── bench_fit_v2.py         # v1/v3/v4 accuracy comparison
    │   ├── bench_scaling.py        # phase breakdown (upload/compute/readback)
    │   ├── bench_minmax.py         # pure-compute sweep, workgroup sizing
    │   ├── bench_v8.py             # shared-memory experiment (rejected)
    │   ├── bench_v10.py            # f16 storage experiment (rejected)
    │   └── bench_v11.py            # double-float experiment (rejected)
    └── probes/                     # hardware/numerics diagnostics
        ├── probe_ceiling.py        # DRAM bandwidth ceiling probe
        └── probe_v2_errors.py      # algebraic-rms error distribution
```

## The final ladder

| Shader | Strategy | n=256 ms | vs CPU | eff GB/s | K med err | rms med err |
|---|---|---:|---:|---:|---:|---:|
| `variants/fit_pair.wgsl` (v1) | pair-per-workgroup, tree reduction, 2-pass | 32 | 3× | 3.0-3.5 | 8.1e-7 | 2.1e-7 |
| `variants/fit_pair_v2.wgsl` | 64-pair workgroup, algebraic rms (f32) | 43 | 2× | 2.2 | 2.0e-7 | **1.6e-4 (max 0.28 catastrophic)** |
| **`fit_pair_v3.wgsl`** (prod) | pair-per-thread, tiled workgroups, 2-pass | **7** | **9-12×** | **15-17** | **2.0e-7** | **1.7e-7** |
| `variants/fit_pair_v4.wgsl` | v3 + register sliding window | 7 | 9-11× | 14-16 | 2.0e-7 | 1.7e-7 |
| `variants/fit_pair_v5_noresid.wgsl` (ceiling) | v4 without rms pass | 4 | 13-14× | 27-36 | — | — |
| `variants/fit_pair_v8_sm.wgsl` | shared-memory traj cache | 48 | 5× | — | same as v3 | — |
| `variants/fit_pair_v9_hybrid.wgsl` | algebraic + fallback detection | 29 | 8-9× | — | 2.0e-7 | 1.2e-5 |
| `variants/fit_pair_v10_f16.wgsl` | f16 storage, f32 compute | 23 | 10-11× | — | 3.4e-5 (max 0.18 **bad**) | 8.0e-4 |
| `variants/fit_pair_v11_df.wgsl` | double-float arithmetic everywhere | 34 | 9× | — | 7.7e-7 | 5.3e-6 |

**v3 is the production winner on every axis that matters.** Faster than everything except v5 (which is incorrect), more accurate than everything except itself. The rest is a catalog of interesting dead ends preserved under `shaders/variants/` so the findings below can be reproduced without re-doing the work.

## Machine capabilities discovered by probing

AMD Radeon iGPU via Vulkan exposes these features through wgpu:
- `shader-f16` — f16 types in storage and compute
- `shader-f64` — listed but **WGSL does NOT support `enable f64`** (Naga parser error). Unusable from wgpu-py without SPIR-V passthrough (not exposed on this version)
- `shader-int64`
- `subgroup`, `subgroup-barrier`, `subgroup-vertex`
- max workgroup storage: 65536 bytes
- max workgroup invocations: 1024
- max dispatch per dimension: 65535

Memcpy peak: **~30-37 GB/s** (vec4<f32> read+write). v3 at 2-pass hits 32 GB/s real DRAM traffic = already bandwidth-saturated. Further gains on this hardware require reducing bytes moved.

## What we learned, in order of surprise

### 1. Dispatch shape matters more than coalescing

I spent the first round convinced v1's "uncoalesced access pattern" was the bug. The fix (v2) pair-per-thread gave coalesced reads but was *slower*. The actual bug was that v1 dispatched **190,650 workgroups** for n=256, but the iGPU has maybe ~500 concurrent slots — a 380× oversubscription where dispatch management overwhelmed compute. v3 dispatches 3,813 workgroups (50× fewer) and hits 5× the throughput.

**Rule:** `n_workgroups ≈ hardware_concurrency × small_factor`. Oversubscription beyond that wastes dispatch cycles on overhead.

### 2. The AMD L1 cache absorbs most "bad coalescing"

v1's intra-workgroup memory reads were strided by 1200 bytes (one D-stride). Classical CUDA wisdom says this is catastrophic. In practice, AMD's L1 served repeated reads from neighboring workgroups (different pairs, same segment) via cache hits. The coalescing "fix" didn't move the needle because the cache had already solved it.

**Rule:** Don't trust CUDA coalescing folklore on non-NVIDIA hardware. Measure.

### 3. Algebraic identities destroy precision in f32

Replacing the 2-pass rms with `Σ|err|² = sTT − 2·Re(K·conj(sTA) − G·conj(sTB)) + |K|²·sAA + |G|²·sBB − 2·Re(K·conj(G)·sAB)` is mathematically exact but every term is O(T·|signal|²) and they cancel to produce a near-zero residual. In f32, median rel error was 1.6e-4, max 0.28. Grimoire Circle XI (Kahan compensation) literally warns about exactly this.

**Rule:** Single-pass rms in f32 via algebra is a trap. Use the direct pointwise pass or upgrade to double-float.

### 4. Tree reduction in shared memory costs accuracy too

v1 uses a 64-wide tree reduction (log₂(64) = 6 stages). Each stage adds rounding error. v3's linear per-thread accumulation has no tree, so K/G precision is 4× better despite both using f32. Circle XXIII-adjacent observation: **fewer rounding stages > more parallel reductions**, when the thread can afford the serial work.

### 5. Upload via `tobytes()` is 5× slower than via `memoryview()`

At 105 MB per batch, `numpy.tobytes()` materializes a full Python-side copy before the FFI call. Passing `memoryview(numpy_array)` lets wgpu memcpy directly. Saved ~20 ms per batch.

**Rule:** Python's "obvious" conversions hide memory copies. Always check.

### 6. The rms pass costs exactly 50% of kernel time

v5 (diagnostic, no rms) hits 72% of peak DRAM bandwidth — double v4's 36%. The 2nd pass is a full re-read of the trajectory. If we ever find a way to compute rms accurately in one pass (pair-of-f32 precision? Kahan? compensated FMA?), we can 2× again. For now, accept the 2× cost.

### 7. Register-cached sliding window is ~only a 5% win

Hardware L1 already absorbs the 3× temporal overlap. Register caching caught up to what the hardware was doing implicitly. Strictly better but not dramatic.

### 8. Workgroup size 64 wins on AMD

Tested 64 vs 128. WG=128 was 30% slower because AMD wave64 means one wave/workgroup at WG=64 is optimal. Two waves per workgroup creates barrier stalls without buying more parallelism.

**Rule:** Match workgroup size to hardware subgroup size. On AMD/RDNA: 64. On NVIDIA: 32. On Intel Arc: varies.

## The universal pattern for this kind of kernel

For batched small-matrix reductions + solves where each independent task has:
- ~50 timesteps
- ~150 pairs
- ~1000 tasks per batch

The winning architecture is:
- **Dispatch**: `(tasks, ceil(P / wg_size), 1)`
- **Workgroup size**: 1 × subgroup_size (64 on AMD, 32 on NVIDIA)
- **Per-thread work**: one "pair" / independent problem, full serial accumulation
- **No shared memory, no barriers**
- **Single-pass where possible, two-pass when precision demands**

This pattern generalizes to: batched linear regressions, batched PCA decompositions, batched ODE fits, batched correlation estimates. Any time you have lots of small independent reductions where each fits in registers.

## Attacks I tried and what they revealed

Everything past v3 was an attempt to cross the DRAM ceiling. None worked on this iGPU. Each failure mode is informative.

### v8: Shared memory caching → DROPPED
Loaded the (segment, 64-pair) tile into 64 KB workgroup shared memory so phase A and phase D both read from SRAM instead of DRAM. **Failed: 2× SLOWER than v3.** Occupancy collapsed — 64 KB per workgroup means 1 workgroup per CU, killing latency hiding. The occupancy tax exceeded the SM win.

Universal lesson: **shared memory is only a win if you have register budget to spare**. On occupancy-constrained hardware, it's a trap.

### v9: Hybrid algebraic + per-thread fallback → NO WIN
Compute algebraic rms in a single pass. If the result looks numerically suspicious (residual/signal ratio below threshold), re-read the trajectory pointwise for that pair. **Failed: same speed as v3.** Warp divergence is total — when any thread in a 64-wide wave takes the fallback, all 64 threads pay the cost. At 20-50% of pairs falling back on synthetic data, every warp re-reads.

Universal lesson: **per-thread divergence on SIMD architectures has zero savings from the non-divergent threads.** Would need a compaction pass + second dispatch to fix, which adds its own cost.

### v10: f16 storage, f32 compute → PRECISION FAIL
Halve DRAM bandwidth by storing the trajectory in f16 (2 bytes per float) and upcasting to f32 inside the kernel. **Speed: 1.2× over v3 (only).** Even though DRAM traffic halved, total speedup was modest — v3 wasn't PURELY bandwidth-bound, and f16→f32 conversions have their own cost.

**Accuracy: max K rel error 18%** — unacceptable. f16 loses too much precision for the 2×2 solve on pairs near the stability boundary.

Universal lesson: **f16 storage works for approximately-correct compute, not for numerically sensitive solves.** Precision audit before shipping.

### v11: Double-float (DF) arithmetic → SLOWER, BUGGIER
Implemented the classical double-float (Dekker/Hida/Li/Bailey) arithmetic in WGSL: each scalar is a (hi, lo) pair of f32s giving ~48 bits mantissa. Rewrote the entire fit (5 sums, 2×2 solve, algebraic rms) in DF. **Failed: ~10% slower than v3 AND has worse accuracy (max K error 3.5e-3 vs v3's 7.6e-4).**

Two reasons for failure:
1. DF arithmetic is 6-10× compute per scalar op. A kernel that's bandwidth-bound in f32 becomes compute-bound in DF, and the single-pass savings are offset.
2. DF correctness is fragile — a bug in the `two_sum`/`quick_two_sum` chain silently loses bits rather than crashing. Hard to verify from perf benches.

Universal lesson: **emulated extended precision trades compute for memory**. On compute-rich, bandwidth-poor hardware it can win; on bandwidth-saturated iGPUs it doesn't.

## What's actually still on the table

All proven to be real directions, none yet implemented:

1. **Discrete GPU** — v3 is DRAM-saturated on this iGPU at 32 GB/s. A discrete card at 500+ GB/s would run v3 15-20× faster without shader changes.

2. **Hybrid with compaction** — like v9 but with a compaction pass between the algebraic and fallback kernels. Eliminates warp divergence. Needs atomics or prefix-sum. Worth doing for 1.5-2× on this iGPU.

3. **Dream-buffer-aware rms** — 99.98% of pair_rms values get evicted from the dream buffer before use. Could skip rms computation entirely for most blocks.

4. **SPIR-V passthrough with real f64** — `shader-f64` feature exists but WGSL spec doesn't support it. SPIR-V passthrough would unlock it. Not exposed in current wgpu-py.

5. **f16 where it's safe** — keep f32 for K,G solve, drop to f16 for rms-adjacent computations. Mixed precision by region.

## Phase breakdown for current v3 (n=256)

- Upload (memoryview): 7 ms
- Submit: 0.2 ms
- GPU compute + sync: 30 ms (v3) or 40 ms (v1)
- Readback: 2 ms
- **Total: ~40 ms per batch, 9× over CPU numpy**

Double-buffering would hide the ~9 ms of CPU-side work behind GPU compute, giving ~30 ms steady-state → ~12× end-to-end.

## Upload/readback budget (post memoryview fix)

At n=256 (105 MB batch):
- Upload: 7 ms
- Submit+compute: 0.2 ms
- GPU sync: 42 ms (*v1*) / 6 ms (*v3*)
- Readback: 2 ms
- **Total wall (v3): ~15 ms end-to-end per batch**

Double-buffering (prep batch N+1 during GPU compute of batch N) would hide the 9 ms of CPU-side work behind the 6 ms GPU work, giving ~10 ms per batch steady state.

1M articles at 256 articles/batch × 10 ms/batch = **40 seconds for the whole genesis run**, vs the current multi-minute pipeline.

That's on an iGPU. Discrete card should be ~5 seconds or less.

## How to run

### Production pipeline

```
python rag_tests/engram/genesis_chain_gpu.py
```

Loads the existing `_alexandria.engram` if present, honors `_genesis_progress.json`, resumes at the first incomplete phase. Checkpoints every 5000 absorbs.

### Bench harness

Primary perf check (v3 vs numpy, end-to-end with upload/readback):
```
python rag_tests/engram/gpu/benches/bench_fit.py
```

Pure-compute sweep across batch sizes and workgroup configs:
```
python rag_tests/engram/gpu/benches/bench_minmax.py
```

Phase breakdown (upload / submit / sync / readback per batch):
```
python rag_tests/engram/gpu/benches/bench_scaling.py
```

### Hardware probes

DRAM bandwidth ceiling:
```
python rag_tests/engram/gpu/probes/probe_ceiling.py
```

Algebraic-rms error distribution (which pairs fail in v2, why, how many):
```
python rag_tests/engram/gpu/probes/probe_v2_errors.py
```

### End-to-end correctness

```
python rag_tests/engram/test_gpu_ingest.py
```

Builds a throwaway brain from 200 synthetic articles, verifies absorb + well-formation + save/load round-trip. No corpus files required.

## Cross-references to `principia_mathematica_obscura` (the grimoire)

The empirical findings above map onto formal circles in the grimoire. These aren't retrofitted justifications — they're laws I rediscovered the hard way, where the grimoire would have been the shortest path.

| Empirical finding | Grimoire circle |
|---|---|
| v3 beats v1 by 5× purely through dispatch shape (n_workgroups matching hardware concurrency, not oversubscribing 380×) | **XIX (Cost Hierarchy)** — stall cover `η = min(1, C/L)`, occupancy ≠ speed |
| v1's tree reduction (6 barriers × slowest-thread) cost more than v3's independent per-thread accumulation | **XIX (barrier phase tax)** — `T = Σ_k max_i t_{i,k}` |
| v8 (shared memory) was 2× slower despite eliminating 2-pass DRAM — 64 KB allocation → 1 workgroup per CU → no latency hiding | **XIX (stall cover)** again — `C` collapsed to 1, so `η_cover` → 1/L |
| v9's per-thread fallback had zero speedup because one lane's slow path forces all 64 to pay | **XVIII (Warp Communion / Active-mask law)** — `η_lane = Σ_j a_j ℓ_j / (W · Σ_j ℓ_j)` |
| v2's algebraic rms had 28% max error from catastrophic cancellation in f32 | **XI (Kahan Compensation)** — literal warning about this exact class of failure |
| v10's f16 storage gave only 1.2× despite halving DRAM — v3 wasn't purely bandwidth-bound, and f16 broke the 2×2 solve on boundary-stable pairs | **XIV (AoSoA / transfer granule)** + **XXIX (Posit rebellion / precision as contract)** |
| v11's double-float was slower because DF compute exceeded the bandwidth savings | **LV (Roofline)** — we were bandwidth-bound; adding compute doesn't help |
| v3 at 32 GB/s = 46% of nominal DRAM peak, matches memcpy probe | **LV (Roofline)** again — proves we're AT the bandwidth roof |
| Dispatch launch overhead is ~2-3 ms per call, forces batch size > ~20 | **LV** — Little's Law sibling: need enough independent work to hide launch latency |
| Python `tobytes()` added 20 ms/batch; `memoryview` fixed it | **XIV** — memory-class choice, not arithmetic, was the tyrant |
| Workgroup size 128 slower than 64 on AMD | **XVIII (Warp Communion)** — subgroup size isn't eternal; on wave64 hardware, 1 wave per workgroup is the sweet spot |

Grimoire laws I'd consult *first* for the next kernel:
- **XIX bank conflict geometry** (`gcd(stride, bank_count)`) if I ever revisit shared-memory caching
- **XIX atomic hotspot** (`R = max_i c_i`) if adding any atomic writes
- **LVII publish split** for async CPU↔GPU pipelining
- **LXIX write-allocate** (`2NL vs NL`) for any write-heavy output buffer

The grimoire doesn't unlock new optimization paths on this iGPU — v3 is bandwidth-saturated and the remaining paths are architectural (discrete GPU, pre-embed to kill Amdahl, compacted hybrid). But it would have saved ~3 of the 8 experimental iterations.
