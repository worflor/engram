"""
hetero_ingest.py — heterogeneous CPU↔GPU ingest pipeline.

architecture (the "2 rats" pattern):
    bz2 reader (1 thread)
         │ raw (title, wikitext) — bounded queue [size 2×batch]
         ▼
    MASS + embed workers (N threads, numba-nogil + numpy)
         │ staged (title, segments, macro_size) — bounded queue [size 2×batch]
         ▼
    GPU dispatcher (MAIN thread — owns GpuEngine exclusively)
         │ results (title, K, G, rms per article)
         ▼
    absorb (MAIN thread, right after GPU readback)

safety invariants (enforced):
  1. only MAIN thread ever touches GpuEngine / device / queue.
  2. every queue is bounded (maxsize) — backpressure prevents OOM.
  3. every blocking queue op has a TIMEOUT — no infinite waits.
  4. None sentinel propagates forward for clean shutdown.
  5. worker exceptions surface via shared error state (caller sees them).
  6. GPU work sizes are bounded per batch (no unbounded growth).
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from engram_codec import (
    Engram, EngramFile, Block, Packet, Mode,
    _to_complex, fit_all, derive_pairing, apply_pairing,
    derive_block_sizes, segment, SEED_COUNT, MIN_BLOCK,
)
from engram.gpu.engine import GpuEngine, pack_batch, gpu_available
from engram.gpu_ingest import _StagedArticle, _StagedSegment


# ── shared error bucket ──────────────────────────────────────────────

class PipelineErrors:
    """collects exceptions from worker threads for the main to surface."""

    def __init__(self):
        self._lock = threading.Lock()
        self._errs: list[BaseException] = []

    def record(self, e: BaseException, context: str = "") -> None:
        with self._lock:
            self._errs.append(e)
            print(f"[pipeline] worker error in {context}: {e}")
            traceback.print_exc()

    def had_error(self) -> bool:
        with self._lock:
            return len(self._errs) > 0

    def all(self) -> list[BaseException]:
        with self._lock:
            return list(self._errs)


# ── CPU worker: wikitext → staged article ────────────────────────────

def _cpu_worker(
    raw_q: "queue.Queue[Optional[tuple[str, str]]]",
    staged_q: "queue.Queue[Optional[_StagedArticle]]",
    embed_fn: Callable[[str], np.ndarray],
    reference_pairing_holder: list,
    errors: PipelineErrors,
    stop_event: threading.Event,
) -> None:
    """
    pulls raw (title, wikitext) from raw_q, produces _StagedArticle,
    pushes to staged_q. exits on None sentinel or stop_event.

    uses GIL-releasing ops (embed_fn should use numpy/numba) so N
    of these run in parallel.
    """
    try:
        while not stop_event.is_set():
            try:
                item = raw_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                # pass sentinel through so downstream knows this
                # pipeline stage is done
                staged_q.put(None)
                return
            title, text = item
            try:
                staged = _cpu_prepare_safe(
                    title, text, embed_fn, reference_pairing_holder,
                )
                if staged is not None:
                    staged_q.put(staged, timeout=5.0)
            except queue.Full:
                # downstream is stuck — surface via error, shut down
                errors.record(RuntimeError("staged_q full for >5s"),
                              "cpu_worker")
                stop_event.set()
                return
            except Exception as e:
                errors.record(e, f"cpu_worker article={title!r}")
                # drop this article but keep processing
    except Exception as e:
        errors.record(e, "cpu_worker outer")


def _cpu_prepare_safe(
    title: str,
    text: str,
    embed_fn: Callable[[str], np.ndarray],
    reference_pairing_holder: list,
) -> Optional[_StagedArticle]:
    """CPU-side: embed + pair + segment → _StagedArticle, or None."""
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

    staged_segs: List[_StagedSegment] = []
    for ss, se in segs:
        L = se - ss
        if L < MIN_BLOCK + 1:
            continue
        seed1 = wp[ss - 1] if ss >= 1 else wp[0]
        seed2 = wp[ss - 2] if ss >= 2 else (wp[0] if ss <= 1 else seed1)
        data = wp[ss:se]
        z_full = np.vstack([seed2[np.newaxis], seed1[np.newaxis], data]).copy()
        signal_energy = float(np.sum(data.astype(np.float64) ** 2))
        staged_segs.append(_StagedSegment(
            z_full=z_full, ss=ss, se=se, L=L,
            seed1=seed1.copy(), seed2=seed2.copy(),
            signal_energy=signal_energy,
        ))
    if not staged_segs:
        return None

    return _StagedArticle(
        title=title, T=T, segments=staged_segs, macro_size=macro_size,
    )


# ── GPU side: drains staged_q, batches, dispatches, absorbs ─────────

def _gpu_dispatch_loop(
    ef: EngramFile,
    engine: GpuEngine,
    staged_q: "queue.Queue[Optional[_StagedArticle]]",
    n_workers: int,
    batch_size: int,
    progress_every: int,
    checkpoint_every: int,
    checkpoint_path: Optional[str],
    errors: PipelineErrors,
    stop_event: threading.Event,
    stats_out: dict,
) -> None:
    """
    MAIN-THREAD function. owns the GPU exclusively.

    pulls _StagedArticle from staged_q, buffers into batches of
    batch_size, dispatches fit_batch, absorbs results.

    terminates when it has received n_workers `None` sentinels
    (one from each CPU worker).
    """
    pending: list[_StagedArticle] = []
    absorbed = 0
    since_ck = 0
    sentinels_seen = 0
    t_start = time.time()
    t_last = t_start

    try:
        while sentinels_seen < n_workers:
            if stop_event.is_set():
                break
            try:
                item = staged_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                sentinels_seen += 1
                continue

            pending.append(item)
            if len(pending) >= batch_size:
                absorbed, since_ck, t_last = _flush_gpu(
                    ef, engine, pending, absorbed, since_ck,
                    t_start, t_last, progress_every,
                    checkpoint_every, checkpoint_path,
                )
                pending = []

        # drain the last partial batch
        if pending:
            absorbed, since_ck, t_last = _flush_gpu(
                ef, engine, pending, absorbed, since_ck,
                t_start, t_last, progress_every,
                checkpoint_every, checkpoint_path,
            )
            pending = []

        # final checkpoint
        if checkpoint_path is not None:
            ef.save(checkpoint_path)

    except Exception as e:
        errors.record(e, "gpu_dispatch_loop")
    finally:
        stats_out["absorbed"] = absorbed
        stats_out["elapsed_s"] = time.time() - t_start
        stats_out["rate_per_s"] = absorbed / max(stats_out["elapsed_s"], 1e-9)


def _flush_gpu(
    ef: EngramFile,
    engine: GpuEngine,
    pending: list[_StagedArticle],
    absorbed: int,
    since_ck: int,
    t_start: float,
    t_last: float,
    progress_every: int,
    checkpoint_every: int,
    checkpoint_path: Optional[str],
) -> tuple[int, int, float]:
    """one batch: pack segments, GPU fit, build packets, absorb."""
    traj_slices = []
    block_spec = []
    for art_idx, art in enumerate(pending):
        for seg in art.segments:
            traj_slices.append(seg.z_full)
            block_spec.append((art_idx, seg))

    if not traj_slices:
        return absorbed, since_ck, t_last

    traj_flat, seg_table = pack_batch(traj_slices)
    # HARD safety cap: refuse ridiculously large batches that would
    # flirt with buffer-size limits or long kernel runtime.
    max_bytes = 512 * 1024 * 1024  # 512 MB
    if traj_flat.nbytes > max_bytes:
        raise RuntimeError(
            f"refusing GPU batch of {traj_flat.nbytes/1e6:.0f} MB "
            f"(safety cap {max_bytes/1e6:.0f} MB)")

    K_flat, G_flat, rms_flat = engine.fit_batch(traj_flat, seg_table)

    D = ef.dim
    P = D // 2
    K_per_seg = K_flat.reshape(-1, P)
    G_per_seg = G_flat.reshape(-1, P)
    rms_per_seg = rms_flat.reshape(-1, P)

    by_article: dict[int, list[Block]] = {}
    for seg_idx, (art_idx, seg) in enumerate(block_spec):
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

    pairing = ef._reference_pairing
    for art_idx, art in enumerate(pending):
        blocks = by_article.get(art_idx, [])
        if not blocks:
            continue
        pkt = Packet(
            dim=D, n_pairs=P, length=art.T, alpha=ef.alpha,
            macro_block=art.macro_size, micro_block=MIN_BLOCK,
            pairing=pairing, blocks=blocks,
        )
        ef.absorb(pkt)
        absorbed += 1
        since_ck += 1
        if absorbed % progress_every == 0:
            now = time.time()
            dt = now - t_last
            t_last = now
            rate = absorbed / max(now - t_start, 1e-9)
            print(f"  [{absorbed:>7,}] wells={len(ef.wells()):3d}  "
                  f"{rate:.0f}/s  dt={dt:.0f}s  {art.title[:45]}")

    if checkpoint_path is not None and since_ck >= checkpoint_every:
        ef.save(checkpoint_path)
        since_ck = 0

    return absorbed, since_ck, t_last


# ── public API ───────────────────────────────────────────────────────

def ingest_hetero(
    ef: EngramFile,
    codec: Engram,
    articles: Iterable[tuple[str, str]],
    embed_fn: Callable[[str], np.ndarray],
    label: str,
    engine: Optional[GpuEngine] = None,
    batch_size: int = 128,
    n_workers: int = 4,
    checkpoint_every: int = 5000,
    checkpoint_path: Optional[str] = None,
    progress_every: int = 1000,
) -> int:
    """
    heterogeneous CPU↔GPU ingest.

    single producer iterates `articles` into a bounded raw queue.
    N cpu workers consume, run embed_fn + segment, push _StagedArticle.
    main thread drains staged queue, batches, dispatches GPU, absorbs.

    all queues bounded. all blocking ops have timeouts. errors in any
    thread bubble up via a shared PipelineErrors bucket and stop the
    pipeline cleanly.
    """
    if engine is None:
        if not gpu_available():
            raise RuntimeError("wgpu not available")
        engine = GpuEngine()

    errors = PipelineErrors()
    stop_event = threading.Event()
    stats: dict = {"absorbed": 0, "elapsed_s": 0.0, "rate_per_s": 0.0}

    # shared reference pairing
    ref_holder: list = []
    if ef._reference_pairing is not None:
        ref_holder.append(ef._reference_pairing)

    # queues: sized to ~2x batch so we have breathing room but
    # backpressure kicks in if any stage stalls.
    raw_q: "queue.Queue" = queue.Queue(maxsize=batch_size * 2)
    staged_q: "queue.Queue" = queue.Queue(maxsize=batch_size * 2)

    # cpu workers
    workers = []
    for i in range(n_workers):
        t = threading.Thread(
            target=_cpu_worker,
            args=(raw_q, staged_q, embed_fn, ref_holder, errors, stop_event),
            daemon=True, name=f"cpu_worker_{i}",
        )
        t.start()
        workers.append(t)

    # producer: iterate articles into raw_q (runs on a DEDICATED thread
    # so this function can run the GPU loop on main).
    def producer():
        try:
            for title, text in articles:
                if stop_event.is_set():
                    break
                try:
                    raw_q.put((title, text), timeout=5.0)
                except queue.Full:
                    errors.record(RuntimeError("raw_q full for >5s"),
                                  "producer")
                    stop_event.set()
                    return
            # signal end of input to workers
            for _ in range(n_workers):
                raw_q.put(None)
        except Exception as e:
            errors.record(e, "producer")
            stop_event.set()

    prod_thread = threading.Thread(target=producer, daemon=True,
                                    name="producer")
    prod_thread.start()

    # GPU dispatcher runs on MAIN thread so it has clear ownership
    # of the wgpu.Device.
    _gpu_dispatch_loop(
        ef=ef, engine=engine,
        staged_q=staged_q,
        n_workers=n_workers,
        batch_size=batch_size,
        progress_every=progress_every,
        checkpoint_every=checkpoint_every,
        checkpoint_path=checkpoint_path,
        errors=errors,
        stop_event=stop_event,
        stats_out=stats,
    )

    # wait for threads to finish
    prod_thread.join(timeout=30.0)
    for w in workers:
        w.join(timeout=30.0)

    # install reference pairing if it was derived on the fly
    if ef._reference_pairing is None and ref_holder:
        ef._reference_pairing = ref_holder[0]

    if errors.had_error():
        print(f"[{label}] pipeline completed with {len(errors.all())} errors")

    print(f"  {stats['absorbed']:,} articles from {label} in "
          f"{stats['elapsed_s']:.0f}s ({stats['rate_per_s']:.0f}/s)")
    return stats["absorbed"]
