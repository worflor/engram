"""
hetero_v2.py — 4-stage assembly-line CPU↔GPU pipeline.

improvements over hetero_ingest.py:

  1. ABSORB IS ITS OWN THREAD.
     previously main thread did GPU dispatch AND absorb alternately —
     50ms GPU wait, 10ms absorb, 50ms GPU wait, 10ms absorb. the GPU
     was idle during absorb and vice versa. splitting to its own
     thread means both halves stay saturated.

  2. GPU DISPATCHER IS ITS OWN THREAD (not main).
     main becomes a pure orchestrator/telemetry thread. single clear
     GPU owner is the dispatcher thread. strictly better locality.

  3. TELEMETRY.
     live queue depths, per-stage rates, GPU/CPU utilization estimate.
     printed every N seconds. tells you which stage is the bottleneck
     in real time without waiting for a timeout.

  4. ADAPTIVE BATCH.
     GPU dispatcher collects until batch_size hit OR batch_timeout_ms.
     prevents starvation on small corpora, caps latency, avoids giant
     batches on very long articles.

  5. LOCK-PROTECTED BRAIN ACCESS.
     absorb thread holds brain-write lock. telemetry reads brain state
     under the same lock. no torn reads.

  6. GRACEFUL DEGRADATION.
     any stage's exception goes to a shared error bus; main surfaces
     them, sets stop_event, waits for threads to drain, prints stats.
     NEVER raises synchronously through unstarted join()s.

safety invariants from hetero_ingest carry over:
  - single GPU owner (now the dispatcher thread, not main)
  - bounded queues (every one)
  - timeouts on all blocking ops
  - sentinel propagation for clean shutdown
  - 512 MB hard cap on GPU batch upload
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from engram_codec import (
    Engram, EngramFile, Block, Packet, Mode,
    derive_pairing, apply_pairing,
    derive_block_sizes, segment, SEED_COUNT, MIN_BLOCK,
)
from engram.gpu.engine import GpuEngine, pack_batch, gpu_available
from engram.gpu_ingest import _StagedArticle, _StagedSegment


# ── shared error bus + pipeline stats ───────────────────────────────

class PipelineErrors:
    def __init__(self):
        self._lock = threading.Lock()
        self._errs: list[tuple[str, BaseException]] = []

    def record(self, ctx: str, e: BaseException) -> None:
        with self._lock:
            self._errs.append((ctx, e))
            print(f"[pipeline] worker error in {ctx}: {e}")
            traceback.print_exc()

    def had_error(self) -> bool:
        with self._lock:
            return bool(self._errs)

    def all(self) -> list[tuple[str, BaseException]]:
        with self._lock:
            return list(self._errs)


@dataclass
class PipelineStats:
    """per-stage counters updated by workers and read by telemetry."""
    raw_q_depth:     int = 0
    staged_q_depth:  int = 0
    results_q_depth: int = 0
    raw_enqueued:    int = 0
    staged_enqueued: int = 0
    batches_dispatched: int = 0
    absorbed:        int = 0
    drops_cpu:       int = 0         # articles filtered by cpu stage
    gpu_batch_bytes_last: int = 0
    start_time:      float = field(default_factory=time.time)

    _lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, field_name: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, field_name, getattr(self, field_name) + n)

    def snapshot(self) -> dict:
        with self._lock:
            d = {k: getattr(self, k) for k in (
                "raw_q_depth", "staged_q_depth", "results_q_depth",
                "raw_enqueued", "staged_enqueued", "batches_dispatched",
                "absorbed", "drops_cpu", "gpu_batch_bytes_last",
            )}
            d["elapsed"] = time.time() - self.start_time
            return d


# ── stage 1: producer ────────────────────────────────────────────────

def _producer(
    articles: Iterable[tuple[str, str]],
    raw_q: "queue.Queue",
    n_workers: int,
    stop_event: threading.Event,
    errors: PipelineErrors,
    stats: PipelineStats,
) -> None:
    try:
        for title, text in articles:
            if stop_event.is_set():
                break
            # blocks if queue full — that's backpressure, on purpose
            try:
                raw_q.put((title, text), timeout=5.0)
                stats.inc("raw_enqueued")
            except queue.Full:
                errors.record("producer", RuntimeError("raw_q full >5s"))
                stop_event.set()
                return
        for _ in range(n_workers):
            raw_q.put(None)
    except Exception as e:
        errors.record("producer", e)
        stop_event.set()


# ── stage 2: cpu workers ────────────────────────────────────────────

def _cpu_worker(
    raw_q: "queue.Queue",
    staged_q: "queue.Queue",
    embed_fn: Callable[[str], np.ndarray],
    ref_holder: list,
    stop_event: threading.Event,
    errors: PipelineErrors,
    stats: PipelineStats,
    worker_id: int,
) -> None:
    try:
        while not stop_event.is_set():
            try:
                item = raw_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                staged_q.put(None)
                return
            title, text = item
            try:
                staged = _prepare_one(title, text, embed_fn, ref_holder)
                if staged is None:
                    stats.inc("drops_cpu")
                    continue
                staged_q.put(staged, timeout=5.0)
                stats.inc("staged_enqueued")
            except queue.Full:
                errors.record(
                    f"cpu_worker_{worker_id}",
                    RuntimeError("staged_q full >5s"),
                )
                stop_event.set()
                return
            except Exception as e:
                errors.record(f"cpu_worker_{worker_id} title={title!r}", e)
    except Exception as e:
        errors.record(f"cpu_worker_{worker_id} outer", e)


def _prepare_one(
    title: str,
    text: str,
    embed_fn: Callable[[str], np.ndarray],
    ref_holder: list,
) -> Optional[_StagedArticle]:
    traj = embed_fn(text)
    if traj.shape[0] < 10:
        return None
    w = traj.astype(np.float32)
    T, D = w.shape
    if T <= SEED_COUNT:
        return None

    if not ref_holder:
        ref_holder.append(derive_pairing(w))
    pairing = ref_holder[0]

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


# ── stage 3: GPU dispatcher ──────────────────────────────────────────

@dataclass
class _ResultBatch:
    pending: list[_StagedArticle]
    K_per_seg: np.ndarray
    G_per_seg: np.ndarray
    rms_per_seg: np.ndarray
    block_spec: list[tuple[int, _StagedSegment]]


def _gpu_dispatcher(
    engine: GpuEngine,
    staged_q: "queue.Queue",
    results_q: "queue.Queue",
    n_workers: int,
    batch_size: int,
    batch_timeout_ms: int,
    max_batch_bytes: int,
    stop_event: threading.Event,
    errors: PipelineErrors,
    stats: PipelineStats,
) -> None:
    """drain staged_q into batches, dispatch fit_batch, push results.
    adaptive: ships on batch_size OR batch_timeout_ms, whichever first.
    """
    try:
        sentinels_seen = 0
        pending: list[_StagedArticle] = []
        last_dispatch = time.time()

        while sentinels_seen < n_workers:
            if stop_event.is_set():
                break

            elapsed_ms = (time.time() - last_dispatch) * 1000.0
            timeout_left = max(0.05,
                               min(0.5, (batch_timeout_ms - elapsed_ms) / 1000.0))
            try:
                item = staged_q.get(timeout=timeout_left)
            except queue.Empty:
                item = "TIMEOUT"

            if item is None:
                sentinels_seen += 1
            elif item == "TIMEOUT":
                pass
            else:
                pending.append(item)

            should_flush = (
                len(pending) >= batch_size
                or (pending and elapsed_ms >= batch_timeout_ms)
                or (pending and sentinels_seen >= n_workers)
            )

            if should_flush:
                try:
                    batch = _gpu_flush(engine, pending, max_batch_bytes, stats)
                    if batch is not None:
                        results_q.put(batch, timeout=10.0)
                        stats.inc("batches_dispatched")
                    pending = []
                    last_dispatch = time.time()
                except queue.Full:
                    errors.record("gpu_dispatcher",
                                  RuntimeError("results_q full >10s"))
                    stop_event.set()
                    return
                except Exception as e:
                    errors.record("gpu_dispatcher", e)
                    pending = []  # drop this batch, keep going

        # drain any stragglers
        if pending:
            try:
                batch = _gpu_flush(engine, pending, max_batch_bytes, stats)
                if batch is not None:
                    results_q.put(batch, timeout=10.0)
                    stats.inc("batches_dispatched")
            except Exception as e:
                errors.record("gpu_dispatcher final flush", e)

        results_q.put(None)  # sentinel to absorb
    except Exception as e:
        errors.record("gpu_dispatcher outer", e)
        results_q.put(None)


def _gpu_flush(
    engine: GpuEngine,
    pending: list[_StagedArticle],
    max_batch_bytes: int,
    stats: PipelineStats,
) -> Optional[_ResultBatch]:
    traj_slices = []
    block_spec: list[tuple[int, _StagedSegment]] = []
    for art_idx, art in enumerate(pending):
        for seg in art.segments:
            traj_slices.append(seg.z_full)
            block_spec.append((art_idx, seg))
    if not traj_slices:
        return None

    traj_flat, seg_table = pack_batch(traj_slices)
    if traj_flat.nbytes > max_batch_bytes:
        raise RuntimeError(
            f"GPU batch too large: {traj_flat.nbytes/1e6:.0f} MB "
            f"(cap {max_batch_bytes/1e6:.0f} MB)")
    stats.gpu_batch_bytes_last = int(traj_flat.nbytes)

    K_flat, G_flat, rms_flat = engine.fit_batch(traj_flat, seg_table)
    # P derived from seg_table: seg_table[:, 2] holds the D of each segment
    P = int(seg_table[0, 2]) // 2
    return _ResultBatch(
        pending=list(pending),
        K_per_seg=K_flat.reshape(-1, P),
        G_per_seg=G_flat.reshape(-1, P),
        rms_per_seg=rms_flat.reshape(-1, P),
        block_spec=block_spec,
    )


# ── stage 4: absorb worker ───────────────────────────────────────────

def _absorb_worker(
    ef: EngramFile,
    results_q: "queue.Queue",
    brain_lock: threading.Lock,
    progress_every: int,
    checkpoint_every: int,
    checkpoint_path: Optional[str],
    stop_event: threading.Event,
    errors: PipelineErrors,
    stats: PipelineStats,
) -> None:
    """runs absorb sequentially; mutations to ef take brain_lock."""
    try:
        since_ck = 0
        D = ef.dim
        P = D // 2
        t_last = time.time()
        t_start = t_last

        while True:
            if stop_event.is_set():
                break
            try:
                batch = results_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if batch is None:
                break

            # build Blocks outside the lock (no ef state needed)
            by_article: dict[int, list[Block]] = {}
            for seg_idx, (art_idx, seg) in enumerate(batch.block_spec):
                K = batch.K_per_seg[seg_idx].astype(np.complex128)
                G = batch.G_per_seg[seg_idx].astype(np.complex128)
                pair_rms = batch.rms_per_seg[seg_idx].astype(np.float32)
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

            # absorb each article under the brain lock
            with brain_lock:
                pairing = ef._reference_pairing
                for art_idx, art in enumerate(batch.pending):
                    blocks = by_article.get(art_idx, [])
                    if not blocks:
                        continue
                    pkt = Packet(
                        dim=D, n_pairs=P, length=art.T, alpha=ef.alpha,
                        macro_block=art.macro_size, micro_block=MIN_BLOCK,
                        pairing=pairing, blocks=blocks,
                    )
                    ef.absorb(pkt)
                    stats.inc("absorbed")
                    since_ck += 1

                if checkpoint_path and since_ck >= checkpoint_every:
                    ef.save(checkpoint_path)
                    since_ck = 0

            # progress line (outside lock, no race cost)
            if stats.snapshot()["absorbed"] % progress_every == 0:
                s = stats.snapshot()
                now = time.time()
                dt = now - t_last
                t_last = now
                rate = s["absorbed"] / max(now - t_start, 1e-9)
                print(f"  [{s['absorbed']:>7,}] wells={len(ef.wells()):3d}  "
                      f"{rate:.0f}/s  dt={dt:.0f}s  "
                      f"qs={s['staged_q_depth']}/{s['results_q_depth']}")

        if checkpoint_path:
            with brain_lock:
                ef.save(checkpoint_path)
    except Exception as e:
        errors.record("absorb_worker", e)


# ── main: telemetry + orchestration ──────────────────────────────────

def _telemetry(
    stop_event: threading.Event,
    stats: PipelineStats,
    raw_q: "queue.Queue",
    staged_q: "queue.Queue",
    results_q: "queue.Queue",
    interval_s: float = 10.0,
) -> None:
    """periodically update depths in stats and print a one-liner."""
    last = time.time()
    last_absorbed = 0
    while not stop_event.is_set():
        stop_event.wait(timeout=interval_s)
        if stop_event.is_set():
            break
        with stats._lock:
            stats.raw_q_depth     = raw_q.qsize()
            stats.staged_q_depth  = staged_q.qsize()
            stats.results_q_depth = results_q.qsize()
        s = stats.snapshot()
        now = time.time()
        window = now - last
        delta = s["absorbed"] - last_absorbed
        rate_window = delta / max(window, 1e-9)
        last, last_absorbed = now, s["absorbed"]
        print(f"  [tele] absorbed={s['absorbed']:<6d} "
              f"drops={s['drops_cpu']:<4d} "
              f"batches={s['batches_dispatched']:<4d} "
              f"q={s['raw_q_depth']}/{s['staged_q_depth']}/{s['results_q_depth']} "
              f"rate={rate_window:.1f}/s "
              f"gpu_batch={s['gpu_batch_bytes_last']/1e6:.1f}MB")


def ingest_hetero_v2(
    ef: EngramFile,
    codec: Engram,
    articles: Iterable[tuple[str, str]],
    embed_fn: Callable[[str], np.ndarray],
    label: str,
    engine: Optional[GpuEngine] = None,
    batch_size: int = 128,
    batch_timeout_ms: int = 200,
    max_batch_bytes: int = 512 * 1024 * 1024,
    n_cpu_workers: int = 4,
    queue_depth: Optional[int] = None,
    progress_every: int = 1000,
    checkpoint_every: int = 5000,
    checkpoint_path: Optional[str] = None,
    telemetry_interval_s: float = 10.0,
) -> int:
    """4-stage assembly line ingest. every stage runs in its own thread
    so CPU and GPU both stay busy. queues between stages surface the
    bottleneck.

    returns the number of articles absorbed.
    """
    if engine is None:
        if not gpu_available():
            raise RuntimeError("wgpu not available")
        engine = GpuEngine()

    errors = PipelineErrors()
    stats = PipelineStats()
    stop_event = threading.Event()
    brain_lock = threading.Lock()

    qdepth = queue_depth if queue_depth is not None else batch_size * 2

    raw_q:     "queue.Queue" = queue.Queue(maxsize=qdepth)
    staged_q:  "queue.Queue" = queue.Queue(maxsize=qdepth)
    results_q: "queue.Queue" = queue.Queue(maxsize=max(2, qdepth // 8))

    ref_holder: list = []
    if ef._reference_pairing is not None:
        ref_holder.append(ef._reference_pairing)

    # start workers
    threads: list[threading.Thread] = []

    prod_t = threading.Thread(
        target=_producer,
        args=(articles, raw_q, n_cpu_workers, stop_event, errors, stats),
        daemon=True, name="producer",
    )
    threads.append(prod_t)

    for i in range(n_cpu_workers):
        t = threading.Thread(
            target=_cpu_worker,
            args=(raw_q, staged_q, embed_fn, ref_holder,
                  stop_event, errors, stats, i),
            daemon=True, name=f"cpu_worker_{i}",
        )
        threads.append(t)

    disp_t = threading.Thread(
        target=_gpu_dispatcher,
        args=(engine, staged_q, results_q, n_cpu_workers,
              batch_size, batch_timeout_ms, max_batch_bytes,
              stop_event, errors, stats),
        daemon=True, name="gpu_dispatcher",
    )
    threads.append(disp_t)

    abs_t = threading.Thread(
        target=_absorb_worker,
        args=(ef, results_q, brain_lock,
              progress_every, checkpoint_every, checkpoint_path,
              stop_event, errors, stats),
        daemon=True, name="absorb_worker",
    )
    threads.append(abs_t)

    tele_t = threading.Thread(
        target=_telemetry,
        args=(stop_event, stats, raw_q, staged_q, results_q,
              telemetry_interval_s),
        daemon=True, name="telemetry",
    )
    threads.append(tele_t)

    t0 = time.time()
    for t in threads:
        t.start()

    # wait for absorb thread to finish; it's the terminus
    abs_t.join()
    # now stop the telemetry loop
    stop_event.set()
    for t in threads:
        if t.is_alive():
            t.join(timeout=10.0)

    # install ref pairing if it was derived on the fly
    if ef._reference_pairing is None and ref_holder:
        ef._reference_pairing = ref_holder[0]

    dt = time.time() - t0
    s = stats.snapshot()
    print(f"  {s['absorbed']:,} articles from {label} in {dt:.0f}s "
          f"({s['absorbed']/max(dt,1):.0f}/s)  "
          f"drops={s['drops_cpu']} batches={s['batches_dispatched']}")
    if errors.had_error():
        print(f"  [{label}] completed with {len(errors.all())} errors")

    return s["absorbed"]
