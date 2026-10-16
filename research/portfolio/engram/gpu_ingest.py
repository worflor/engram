"""
gpu_ingest.py — batched GPU ingest accumulator for the Alexandria genesis
pipeline.

architectural contract:
    - CPU does text embedding (GloVe/fastText), dim-pairing, segmentation.
      these stages either are fundamentally python-bound (dict lookup)
      or are already numba-jit'd and negligible next to the GPU kernel.
    - the engram codec's fit_all — the hot inner loop — runs as a single
      batched GPU dispatch over many articles' segments at once.
    - absorb stays sequential because well assignment depends on brain
      state mutated by previous absorbs. each absorb is ~microseconds.

this mirrors what `_ingest_worker` + the `ingest(...)` outer loop in
`genesis_chain.py` do, but replaces the per-article fit_all call with
one GPU dispatch per batch.

why the batched shape is necessary:
    fit_pair_v3.wgsl has a per-dispatch launch overhead of ~2-3 ms. at
    n=8 articles the kernel is launch-bound (1.4x vs numpy); at n=128+
    it's compute-bound (4-5x end-to-end, 9-12x pure compute). the
    accumulator exists so callers don't have to think about this:
    they call .queue(text) and the flush fires automatically when the
    batch is full.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from engram_codec import (
    Engram, EngramFile, Block, Packet, Mode,
    _to_complex, fit_all, derive_pairing, apply_pairing,
    derive_block_sizes, segment, SEED_COUNT, MIN_BLOCK,
)
from engram.gpu.engine import GpuEngine, pack_batch, gpu_available


# ── per-article staged work (CPU) ────────────────────────────────────
#
# we extract only the segment-level slices (each capped at MAX_BLOCK
# timesteps via the segmenter) and discard the full paired trajectory
# as soon as the segments are materialized. this is load-bearing for
# memory: wiki articles can run 50k+ tokens, which after paired-f32
# storage is ~60 MB per article. with 128 articles queued in a batch
# and several more buffered inside ThreadPoolExecutor, retaining `wp`
# would balloon to multiple GB — OOM-crashing on any machine without
# a huge working set.
#
# segment slices in contrast are bounded: each is ≤ MAX_BLOCK+2
# timesteps (2 seeds + up to MAX_BLOCK data rows), so one segment is
# at most ~150 KB at D=300. a typical article has a handful of
# segments, so one staged article is single-digit MB regardless of
# source article length.

@dataclass
class _StagedSegment:
    """one segment ready for GPU fit: z_full is [seed2, seed1, data...]."""
    z_full: np.ndarray          # [L+2, D] float32
    ss: int                     # segment start in the original trajectory
    se: int                     # segment end
    L: int                      # length (se - ss)
    seed1: np.ndarray           # [D] float32
    seed2: np.ndarray           # [D] float32
    signal_energy: float


@dataclass
class _StagedArticle:
    """one article's CPU-prepped inputs, waiting for its GPU fits."""
    title: str
    T: int                      # total paired trajectory length (for Packet.length)
    segments: List[_StagedSegment]
    macro_size: int


def _cpu_prepare(item, embed_fn, reference_pairing_holder):
    """
    CPU stage for one article: embed + pair + segment + materialize
    segment slices, then drop the full paired trajectory. safe to run
    in a worker thread — mwparserfromhell (via embed_fn) releases the
    GIL during parse, numpy releases it during derive_pairing /
    apply_pairing / segment, so N workers genuinely parallelize.

    reference_pairing_holder is a 1-element list acting as a shared
    cell. if it's empty when we arrive, we derive pairing from our
    own trajectory and install it. concurrent workers may both install
    (one wins); any valid pairing is correct because downstream absorb
    only cares that K centroids live in one consistent coordinate
    system — which is the case as long as every article uses the same
    pairing going forward.
    """
    title, text = item
    traj = embed_fn(text)
    if traj.shape[0] < 10:
        return None
    w = traj.astype(np.float32)
    T, D = w.shape
    if T <= SEED_COUNT:
        return None

    if not reference_pairing_holder:
        reference_pairing_holder.append(derive_pairing(w))
    pairing = reference_pairing_holder[0]

    wp = apply_pairing(w, pairing)
    macro_size, _ = derive_block_sizes(wp)
    segs = segment(wp, macro_size)
    if not segs:
        return None

    staged_segments: List[_StagedSegment] = []
    for ss, se in segs:
        L = se - ss
        if L < MIN_BLOCK + 1:
            continue
        seed1 = wp[ss - 1] if ss >= 1 else wp[0]
        seed2 = wp[ss - 2] if ss >= 2 else (wp[0] if ss <= 1 else seed1)
        data = wp[ss:se]
        z_full = np.vstack([seed2[np.newaxis], seed1[np.newaxis], data]).copy()
        signal_energy = float(np.sum(data.astype(np.float64) ** 2))
        staged_segments.append(_StagedSegment(
            z_full=z_full, ss=ss, se=se, L=L,
            seed1=seed1.copy(), seed2=seed2.copy(),
            signal_energy=signal_energy,
        ))

    if not staged_segments:
        return None

    # wp and w go out of scope here and can be GC'd immediately —
    # only the small per-segment slices survive into the queue.
    return _StagedArticle(
        title=title, T=T, segments=staged_segments, macro_size=macro_size,
    )


# ── the accumulator ─────────────────────────────────────────────────

class BatchedGpuIngest:
    """
    accumulator pattern: queue articles, flush when full.

    typical usage:
        eng = GpuEngine()
        ingestor = BatchedGpuIngest(ef, codec, eng, batch_size=128)
        for title, text in articles:
            ingestor.queue(title, text, embed_fn)
        ingestor.flush()
    """

    def __init__(
        self,
        ef: EngramFile,
        codec: Engram,
        engine: GpuEngine,
        batch_size: int = 128,
        progress_every: int = 1000,
        checkpoint_every: int = 5000,
        checkpoint_path: Optional[str] = None,
    ):
        self.ef = ef
        self.codec = codec
        self.engine = engine
        self.batch_size = batch_size
        self.progress_every = progress_every
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path = checkpoint_path

        self._pending: List[_StagedArticle] = []
        self._absorbed = 0
        self._since_checkpoint = 0
        self._t_start = time.time()
        self._t_last = self._t_start
        self._last_title = ""

    # ── CPU staging ─────────────────────────────────────────────────

    def queue(self, title: str, text: str, embed_fn: Callable[[str], np.ndarray]) -> None:
        """embed, pair, segment. defer fit_all to the GPU batch."""
        # delegate to the shared CPU-prepare helper so single-threaded
        # and threaded paths produce identically shaped staged articles.
        holder: List[np.ndarray] = []
        if self.ef._reference_pairing is not None:
            holder.append(self.ef._reference_pairing)
        staged = _cpu_prepare((title, text), embed_fn, holder)
        if staged is None:
            return
        if self.ef._reference_pairing is None and holder:
            self.ef._reference_pairing = holder[0]

        self._pending.append(staged)
        self._last_title = staged.title

        if len(self._pending) >= self.batch_size:
            self.flush()

    # ── GPU dispatch + absorb ───────────────────────────────────────

    def flush(self) -> None:
        if not self._pending:
            return

        # Staged segments are already z_full slices; just flatten
        # across articles and keep a parallel list of article indices
        # so we can rebuild blocks per article after the dispatch.
        traj_slices: List[np.ndarray] = []
        # (article_idx, segment_obj)
        block_spec: List[Tuple[int, _StagedSegment]] = []

        for art_idx, art in enumerate(self._pending):
            for seg in art.segments:
                traj_slices.append(seg.z_full)
                block_spec.append((art_idx, seg))

        if not traj_slices:
            self._pending = []
            return

        traj_flat, seg_table = pack_batch(traj_slices)
        K_flat, G_flat, rms_flat = self.engine.fit_batch(traj_flat, seg_table)

        P = self.ef.dim // 2
        K_per_seg = K_flat.reshape(-1, P)
        G_per_seg = G_flat.reshape(-1, P)
        rms_per_seg = rms_flat.reshape(-1, P)

        # ── assemble packets per article, absorb sequentially ────────

        by_article: dict[int, List[Block]] = {}
        D = self.ef.dim
        for seg_idx, (art_idx, seg) in enumerate(block_spec):
            # cast to complex128 to match the CPU absorb path's dtype.
            # The GPU kernel computes in f32, so this is a widening
            # no-op precision-wise but keeps downstream invariants.
            K = K_per_seg[seg_idx].astype(np.complex128)
            G = G_per_seg[seg_idx].astype(np.complex128)
            pair_rms = rms_per_seg[seg_idx].astype(np.float32)

            blk = Block(
                mode=Mode.CASCADED, start=seg.ss, length=seg.L,
                macro_K=K, macro_G=G,
                micro_Ks=[], micro_Gs=[],
                residuals=np.empty((0, D), dtype=np.float32),
                seed1=seg.seed1, seed2=seg.seed2,
                scales=np.zeros(P, dtype=np.float32),
                signal_energy=seg.signal_energy, residual_energy=0.0,
            )
            blk.pair_rms = pair_rms
            by_article.setdefault(art_idx, []).append(blk)

        pairing = self.ef._reference_pairing
        for art_idx, art in enumerate(self._pending):
            blocks = by_article.get(art_idx, [])
            if not blocks:
                continue
            pkt = Packet(
                dim=D, n_pairs=P, length=art.T, alpha=self.ef.alpha,
                macro_block=art.macro_size, micro_block=MIN_BLOCK,
                pairing=pairing, blocks=blocks,
            )
            self.ef.absorb(pkt)
            self._absorbed += 1
            self._since_checkpoint += 1

            if self._absorbed % self.progress_every == 0:
                self._print_progress()

        # Checkpoint gate (saves the brain to disk once per N absorbs).
        if (self.checkpoint_path is not None
                and self._since_checkpoint >= self.checkpoint_every):
            self.ef.save(self.checkpoint_path)
            self._since_checkpoint = 0

        self._pending = []

    # ── reporting ───────────────────────────────────────────────────

    def _print_progress(self) -> None:
        now = time.time()
        dt = now - self._t_last
        self._t_last = now
        rate = self._absorbed / max(now - self._t_start, 1e-9)
        n_wells = len(self.ef.wells())
        print(f"  [{self._absorbed:>7,}] wells={n_wells:3d}  "
              f"{rate:.0f}/s  dt={dt:.0f}s  {self._last_title[:45]}")

    def stats(self) -> dict:
        elapsed = time.time() - self._t_start
        return {
            "absorbed": self._absorbed,
            "elapsed_s": elapsed,
            "rate_per_s": self._absorbed / max(elapsed, 1e-9),
            "wells": len(self.ef.wells()),
        }


# ── convenience: threaded ingest with CPU/GPU overlap ───────────────

def ingest_gpu(
    ef: EngramFile,
    codec: Engram,
    articles,
    embed_fn: Callable[[str], np.ndarray],
    label: str,
    engine: Optional[GpuEngine] = None,
    batch_size: int = 128,
    checkpoint_every: int = 5000,
    checkpoint_path: Optional[str] = None,
    n_workers: int = 4,
) -> int:
    """
    threaded GPU ingest. CPU workers prep (embed+pair+segment) in
    parallel while the main thread drives GPU dispatches. the typical
    pattern on the iGPU is that CPU prep (text_to_trajectory dominates)
    takes ~70-100 ms per 128-article batch and GPU dispatch takes
    ~50-80 ms, so overlap yields ~1.5-2× over the single-threaded path.

    the worker count default of 4 is tuned for "don't starve the GPU,
    don't over-subscribe cache." on a machine with 8+ cores it's safe
    to bump this to 6 or 8; past that, the GIL-held dict-lookup tax
    inside text_to_trajectory starts to fight itself.

    the ThreadPoolExecutor preserves order, which keeps absorb
    decisions deterministic across runs.
    """
    from concurrent.futures import ThreadPoolExecutor

    if engine is None:
        if not gpu_available():
            raise RuntimeError("wgpu not available; install wgpu or use the CPU path")
        engine = GpuEngine()

    t0 = time.time()
    accumulator = BatchedGpuIngest(
        ef=ef, codec=codec, engine=engine,
        batch_size=batch_size,
        checkpoint_every=checkpoint_every,
        checkpoint_path=checkpoint_path,
    )

    # shared reference-pairing cell. seeded from the brain's existing
    # pairing (every loaded brain has one) so workers use it directly.
    # if the brain is fresh and has no pairing yet, the first worker
    # to return a valid trajectory installs one.
    ref_holder: List[np.ndarray] = []
    if ef._reference_pairing is not None:
        ref_holder.append(ef._reference_pairing)

    def _prep(item):
        return _cpu_prepare(item, embed_fn, ref_holder)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        # chunksize=4 keeps the pool's internal queue small enough
        # that workers stay warm without buffering too much memory
        # ahead of the GPU.
        for staged in pool.map(_prep, articles, chunksize=4):
            if staged is None:
                continue
            # install pairing into the brain once the first worker
            # has produced one (deferred from workers to avoid any
            # racing mutation of ef state from multiple threads).
            if ef._reference_pairing is None and ref_holder:
                ef._reference_pairing = ref_holder[0]

            accumulator._pending.append(staged)
            accumulator._last_title = staged.title
            if len(accumulator._pending) >= batch_size:
                accumulator.flush()

    accumulator.flush()

    if checkpoint_path is not None:
        ef.save(checkpoint_path)

    elapsed = time.time() - t0
    count = accumulator.stats()["absorbed"]
    print(f"  {count:,} articles from {label} in {elapsed:.0f}s "
          f"({count/max(elapsed,1):.0f}/s)")
    return count
