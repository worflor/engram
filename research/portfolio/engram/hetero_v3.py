"""
hetero_v3.py — 4-stage pipeline with GPU-resident GloVe matrix.

the v2 pipeline moved fit_pair off-CPU but the CPU workers still did:
    idxs = MASS.extract(wikitext)      # fast, all C
    raw  = matrix[idxs]                # 480 MB random-access gather
    cs   = cumsum(raw, axis=0)          # cache-hostile
    wp   = smoothed[:, pairing]         # another random gather
these three numpy ops dominated per-article CPU time — telemetry
showed raw_q backing up while GPU sat idle.

v3 fuses those three into a single GPU kernel (gather_smooth_pair.wgsl)
owned by ChainEngine. the GloVe matrix is uploaded ONCE at pipeline
start, pairing is uploaded once, and each batch only uploads the
tiny int32 indices.

CPU worker output is just (title, idxs: int32[N]) — kilobytes instead
of megabytes. staged_q can hold ~100× more articles for the same
memory budget.

invariants carried from v2:
  - single GPU owner (dispatcher thread; chain + engine both touched
    here and nowhere else)
  - bounded queues, timeouts, sentinel shutdown, shared error bus
  - 512 MB cap on gather output AND on fit_batch upload (checked
    independently in ChainEngine and GpuEngine)
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

from engram_codec import Engram, EngramFile
from engram.gpu.engine import GpuEngine, gpu_available
from engram.gpu.chain_engine import ChainEngine, GatherHandle
from engram.gpu_ingest import _StagedSegment


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
class StageTimings:
    """per-stage wall-clock EWMA for the GPU dispatcher.

    distinguishes three classes of time:

    CPU-blocked-on-GPU (feat_wait_ms, fit_wait_ms, seeds_wait_ms):
        map_sync calls. if these dominate, GPU is the bottleneck — the
        CPU literally could not make progress. attack the corresponding
        kernel or reduce the amount of data it processes.
    CPU work (cpu_segment_ms, block_build_ms):
        numpy / Python / numba. if these dominate, CPU is bottleneck.
        jit/vectorize/reduce allocations.
    submit (gather_submit_ms, feat_submit_ms, fit_submit_ms, seeds_submit_ms):
        time spent dispatching a kernel (encoder + submit). near zero
        in steady state; high values usually mean driver overhead or
        buffer resize.

    the report flags TOP=<stage>. optimization lever:
        feat_wait dominant  → gather or features kernel is slow
        cpu_segment dominant→ jit-segment loop / reduce cumsum allocs
        fit_wait dominant   → fit kernel is slow
        block_build dominant→ Python object construction overhead
    """
    gather_submit_ms:   float = 0.0
    feat_submit_ms:     float = 0.0
    feat_wait_ms:       float = 0.0   # waits for gather + features GPU
    cpu_segment_ms:     float = 0.0
    fit_submit_ms:      float = 0.0
    fit_wait_ms:        float = 0.0
    seeds_submit_ms:    float = 0.0
    seeds_wait_ms:      float = 0.0
    block_build_ms:     float = 0.0
    absorb_ms:          float = 0.0   # per-batch absorb-worker time
    # GPU-measured kernel times (via timestamp-query). these are the
    # TRUE times, not wait-for-queue-drain.
    gpu_gather_ms:      float = 0.0
    gpu_features_ms:    float = 0.0
    # pipelined prefetch: submit of NEXT batch that overlapped with
    # CURRENT batch's CPU work. separated so it doesn't double-count.
    prefetch_submit_ms: float = 0.0
    # useful CPU work that happened while GPU was chewing on prefetch
    prefetch_overlap_ms:float = 0.0
    n_samples:          int = 0
    _alpha:             float = 0.3
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, **kwargs) -> None:
        with self._lock:
            self.n_samples += 1
            for k, v in kwargs.items():
                if not hasattr(self, k):
                    continue
                cur = getattr(self, k)
                if self.n_samples <= 1:
                    setattr(self, k, float(v))
                else:
                    setattr(self, k, cur * (1 - self._alpha) + v * self._alpha)

    def report(self) -> str:
        with self._lock:
            pairs = [
                ("gather_submit", self.gather_submit_ms),
                ("feat_submit",   self.feat_submit_ms),
                ("feat_wait",     self.feat_wait_ms),
                ("cpu_segment",   self.cpu_segment_ms),
                ("fit_submit",    self.fit_submit_ms),
                ("fit_wait",      self.fit_wait_ms),
                ("seeds_submit",  self.seeds_submit_ms),
                ("seeds_wait",    self.seeds_wait_ms),
                ("block_build",   self.block_build_ms),
                ("prefetch_sub",  self.prefetch_submit_ms),
                ("absorb",        self.absorb_ms),
            ]
            total = sum(v for _, v in pairs)
            if total <= 0:
                return "[timings] (no samples yet)"
            pairs.sort(key=lambda kv: -kv[1])
            head = pairs[0]
            parts = []
            for k, v in pairs:
                pct = 100.0 * v / total
                parts.append(f"{k}={v:.0f}ms({pct:.0f}%)")
            overlap_str = (f" overlap={self.prefetch_overlap_ms:.0f}ms"
                           if self.prefetch_overlap_ms > 0 else "")
            # GPU truth: what the kernels ACTUALLY took per timestamp
            # queries. if gpu_* << feat_wait, the rest is queue drain /
            # prefetch contention / driver overhead.
            gpu_str = ""
            if self.gpu_gather_ms > 0 or self.gpu_features_ms > 0:
                gpu_str = (f" gpu_truth:gather={self.gpu_gather_ms:.1f}ms "
                           f"features={self.gpu_features_ms:.1f}ms")
            return (f"[timings] TOP={head[0]} TOTAL={total:.0f}ms/batch{overlap_str}{gpu_str}  "
                    + " ".join(parts))


@dataclass
class PipelineStats:
    raw_q_depth:     int = 0
    staged_q_depth:  int = 0
    results_q_depth: int = 0
    raw_enqueued:    int = 0
    staged_enqueued: int = 0
    batches_dispatched: int = 0
    absorbed:        int = 0
    drops_cpu:       int = 0
    drops_dispatcher: int = 0
    gpu_gather_bytes_last: int = 0
    gpu_fit_bytes_last: int = 0
    start_time:      float = field(default_factory=time.time)

    timings: StageTimings = field(default_factory=StageTimings)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, field_name: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, field_name, getattr(self, field_name) + n)

    def snapshot(self) -> dict:
        with self._lock:
            d = {k: getattr(self, k) for k in (
                "raw_q_depth", "staged_q_depth", "results_q_depth",
                "raw_enqueued", "staged_enqueued", "batches_dispatched",
                "absorbed", "drops_cpu", "drops_dispatcher",
                "gpu_gather_bytes_last", "gpu_fit_bytes_last",
            )}
            d["elapsed"] = time.time() - self.start_time
            return d


# ── staged item for v3: title + raw indices ──────────────────────────

@dataclass
class _IdxArticle:
    title: str
    idxs: np.ndarray  # int32[N]; empty array means "dropped" tombstone
    seq: int = 0      # producer-assigned monotonic sequence number
    # byte histogram (256-dim f64) for byte-level Sources. None when
    # the Source's vocab isn't byte-shaped — i.e. idxs outside [0,255)
    # or >256-entry matrix. the absorb worker uses this to update
    # ef._well_sum_raw per assigned well, enabling compositional
    # naming. cpu_worker populates it; everywhere else it flows
    # through unchanged.
    byte_hist: Optional[np.ndarray] = None


def _is_drop_tombstone(item: "_IdxArticle") -> bool:
    return item.idxs.shape[0] == 0


# ── stage 1: producer ────────────────────────────────────────────────

def _producer(
    articles: Iterable[tuple[str, str]],
    raw_q: "queue.Queue",
    n_workers: int,
    stop_event: threading.Event,
    errors: PipelineErrors,
    stats: PipelineStats,
) -> None:
    # tag each article with a monotonic sequence number. with
    # n_cpu_workers > 1 the workers produce results out of order; the
    # dispatcher uses these seq numbers to reorder before batching so
    # routing is deterministic across runs.
    try:
        seq = 0
        for title, text in articles:
            while not stop_event.is_set():
                try:
                    raw_q.put((seq, title, text), timeout=1.0)
                    stats.inc("raw_enqueued")
                    break
                except queue.Full:
                    continue
            if stop_event.is_set():
                break
            seq += 1
        for _ in range(n_workers):
            while not stop_event.is_set():
                try:
                    raw_q.put(None, timeout=1.0)
                    break
                except queue.Full:
                    continue
    except Exception as e:
        errors.record("producer", e)
        stop_event.set()


# ── stage 2: CPU workers run MASS only ──────────────────────────────

def _cpu_worker(
    raw_q: "queue.Queue",
    staged_q: "queue.Queue",
    idx_fn: Callable[[str], np.ndarray],
    min_tokens: int,
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
                # propagate sentinel, retry-on-full like real items
                while not stop_event.is_set():
                    try:
                        staged_q.put(None, timeout=1.0)
                        break
                    except queue.Full:
                        continue
                return
            seq, title, text = item
            try:
                idxs = idx_fn(text)
                # cap article length — pathologically long articles
                # (e.g. wikiquote full-work pages with 100k+ tokens)
                # make the features + segment GPU kernels exceed
                # Windows' 2s TDR threshold, losing the device. mirror
                # fast_embed.MAX_TOKENS_PER_ARTICLE = 20000.
                # hard cap matching fast_embed.MAX_TOKENS_PER_ARTICLE.
                # large articles (wikiquote full-author pages, gutenberg
                # chapters) would push chain's + unified's kernels into
                # Windows TDR (2s GPU timeout) territory.
                if idxs.shape[0] > 20000:
                    idxs = idxs[:20000]
                if idxs.shape[0] < min_tokens:
                    # emit as tombstone — dispatcher needs to see every
                    # seq so it can advance past it during reorder.
                    # using idxs=empty as the sentinel keeps the path
                    # uniform without a separate Queue.
                    stats.inc("drops_cpu")
                    staged = _IdxArticle(
                        title=title,
                        idxs=np.empty(0, dtype=np.int32),
                        seq=seq)
                else:
                    if idxs.dtype != np.int32:
                        idxs = idxs.astype(np.int32)
                    # byte histogram from the raw text's UTF-8 bytes.
                    # language-agnostic (UTF-8 covers every Unicode
                    # codepoint) and cheap (single bincount per article).
                    # populates ef._well_sum_raw for EVERY Source type
                    # in the unified absorb path — so the refine-from-
                    # wells bootstrap has usable mass even when the
                    # ingest tokenizer was word-level (MUSE, GloVe, etc.)
                    # rather than byte-level.
                    if isinstance(text, (bytes, bytearray)):
                        raw = bytes(text)
                    elif isinstance(text, str):
                        raw = text.encode("utf-8", errors="replace")
                    else:
                        raw = str(text).encode("utf-8", errors="replace")
                    byte_hist = np.bincount(
                        np.frombuffer(raw, dtype=np.uint8),
                        minlength=256).astype(np.float64)
                    staged = _IdxArticle(
                        title=title, idxs=idxs, seq=seq,
                        byte_hist=byte_hist)
                # backpressure loop — first real-enwiki batch on the GPU
                # can run 10+ seconds, during which staged_q stays full.
                # retry until the dispatcher drains it or the pipeline
                # is shutting down.
                while not stop_event.is_set():
                    try:
                        staged_q.put(staged, timeout=1.0)
                        stats.inc("staged_enqueued")
                        break
                    except queue.Full:
                        continue
            except Exception as e:
                errors.record(f"cpu_worker_{worker_id} title={title!r}", e)
    except Exception as e:
        errors.record(f"cpu_worker_{worker_id} outer", e)


# ── stage 3: GPU dispatcher (chain + fit) ────────────────────────────

@dataclass
class _ResultBatch:
    """unified batch result from the single-submit pipeline.

    K_sums/G_sums/rms_sums are per-article sums produced directly by
    gpu/shaders/per_article_reduce.wgsl — no CPU reduceat required.
    consumed by Brain.absorb_from_sums.

    byte_hists carries the per-article byte histogram for byte-level
    Sources (None otherwise). the absorb worker uses it to maintain
    ef._well_sum_raw per well, which powers compositional well naming.
    """
    titles: list[str]
    Ts: list[int]
    K_sums: np.ndarray          # complex64[n_art, P]
    G_sums: np.ndarray          # complex64[n_art, P]
    rms_sums: np.ndarray        # float32[n_art, P]
    n_segs_per_art: np.ndarray  # int64[n_art]
    # per-article byte histograms when the Source is byte-level; None
    # when the Source doesn't produce byte-shaped indices. same length
    # as titles when present. entries are None for individual articles
    # that had no byte_hist computed (mixed-source batches).
    byte_hists: Optional[List[Optional[np.ndarray]]] = None


@dataclass
class _ReorderState:
    """per-dispatcher reorder buffer. items arrive from staged_q in
    whatever order the CPU workers finished tokenizing; we stash them
    in buf keyed by seq, and only ever emit items in seq order (with
    tombstones advanced past silently). this makes routing deterministic
    across runs regardless of n_cpu_workers.
    """
    buf: dict = field(default_factory=dict)      # seq → _IdxArticle
    next_seq: int = 0


def _collect_batch(
    staged_q: "queue.Queue",
    sentinels_state: list,
    batch_size: int,
    batch_timeout_ms: int,
    n_workers: int,
    stop_event: threading.Event,
    window: int = 10,
    holdover: Optional[list] = None,
    reorder: Optional["_ReorderState"] = None,
) -> tuple[list["_IdxArticle"], bool]:
    """pull items off staged_q until batch_size is reached, OR the
    aggregate total_smooth would exceed the dispatch cap, OR the
    batch_timeout expires, OR all sentinels are observed.

    items are deterministically ordered by producer-assigned sequence
    numbers via `reorder`. an item waiting for an earlier seq to arrive
    stays in reorder.buf and is not dispatched this call; it will be
    emitted the moment the predecessor arrives, keeping the dispatcher
    order identical across runs regardless of CPU-worker scheduling.

    holdover is a mutable list the caller owns; items that arrive too
    large to fit the current batch get pushed onto it and consumed by
    the next call. this prevents both dropping articles and the dispatch
    cap errors at batch_size=48+ on real enwiki.
    """
    pending: list["_IdxArticle"] = []
    total_smooth = 0
    t_first = None

    if reorder is None:
        # fallback for callers that don't pass a reorder state — e.g.
        # single-worker runs where order is preserved naturally. not
        # recommended for n_cpu_workers > 1.
        reorder = _ReorderState()

    def _drain_in_order():
        """emit contiguous items starting from next_seq. tombstones are
        consumed silently (already counted as drops_cpu by workers).
        stops when next_seq is not in buf, batch_size reached, or
        total_smooth cap hit."""
        nonlocal total_smooth, t_first
        while reorder.next_seq in reorder.buf:
            item = reorder.buf.pop(reorder.next_seq)
            reorder.next_seq += 1
            if _is_drop_tombstone(item):
                continue
            est = max(0, int(item.idxs.shape[0]) - window + 1)
            if total_smooth + est > _GATHER_MAX_TOTAL_SMOOTH:
                if pending:
                    if holdover is not None:
                        holdover.append(item)
                    return False  # batch full (by size cap)
                else:
                    pending.append(item)
                    return False  # oversized singleton
            pending.append(item)
            total_smooth += est
            if t_first is None:
                t_first = time.time()
            if len(pending) >= batch_size:
                return False  # batch full (by count)
        return True  # emitted all contiguous items currently in buf

    # drain holdover first (items already in order, from prior call)
    if holdover:
        while holdover:
            item = holdover[0]
            est = max(0, int(item.idxs.shape[0]) - window + 1)
            if total_smooth + est > _GATHER_MAX_TOTAL_SMOOTH and pending:
                break
            pending.append(holdover.pop(0))
            total_smooth += est
            if len(pending) >= batch_size:
                break

    # drain anything already in the reorder buffer
    _drain_in_order()

    while not stop_event.is_set():
        if sentinels_state[0] >= n_workers and not reorder.buf and pending:
            return pending, False
        if sentinels_state[0] >= n_workers and not reorder.buf and not pending:
            return pending, True
        if len(pending) >= batch_size:
            return pending, False

        if t_first is None:
            remaining_ms = batch_timeout_ms
        else:
            remaining_ms = batch_timeout_ms - (time.time() - t_first) * 1000.0
            if remaining_ms <= 0 and pending:
                return pending, False

        timeout_s = max(0.01, min(0.25, remaining_ms / 1000.0))
        try:
            item = staged_q.get(timeout=timeout_s)
        except queue.Empty:
            if pending:
                return pending, False
            continue

        if item is None:
            sentinels_state[0] += 1
            if sentinels_state[0] >= n_workers:
                # drain any remaining in-order items; out-of-order
                # items in buf are stranded (missing earlier seq) —
                # this can only happen if a cpu_worker crashed; skip them.
                _drain_in_order()
                return pending, (len(pending) == 0)
        else:
            reorder.buf[item.seq] = item
            if not _drain_in_order():
                # batch is full or pending has oversized singleton
                return pending, False

    return pending, False


# wgpu dispatch per-dim cap (WebGPU baseline: 65535). gather_smooth_pair
# v2 dispatch uses (ceil(total_smooth/4), ceil(D/64)); so the per-batch
# total_smooth cap is 65535 * 4 = 262140. if a batch exceeds this, the
# dispatcher pre-caps the collected batch and holds overflow items for
# the next batch.
_GATHER_MAX_TOTAL_SMOOTH = 65535 * 4


def _submit_gather_features(
    chain: ChainEngine, items: list, max_batch_bytes: int,
    stats: PipelineStats,
) -> Optional[dict]:
    """fused submit: gather + features in ONE queue submit.

    segmentation stays on CPU. we tried a GPU segmentation kernel
    (single-workgroup, then parallel-per-article with atomic slot
    reservation) and both lost to numba-jit CPU segmentation on this
    iGPU — 310 vs 352 art/s. the GPU kernel's setup+reductions+atomic
    overhead on ~24 k small sequential emissions beats CPU numba,
    which runs the same logic in ~1 ms of in-cache work.
    """
    import time as _t
    if not items:
        return None

    idx_list = [a.idxs for a in items]
    t0 = _t.perf_counter()
    combined = chain.submit_gather_and_features(
        idx_list, max_batch_bytes=max_batch_bytes)
    g_ms = (_t.perf_counter() - t0) * 1000.0
    if combined is None:
        for _ in items:
            stats.inc("drops_dispatcher")
        return None
    stats.gpu_gather_bytes_last = combined["handle"].out_buf_size_bytes
    return {
        "items": items,
        "handle": combined["handle"],
        "feat_ctx": combined["feat_ctx"],
        "g_submit_ms": g_ms,
        "f_submit_ms": 0.0,
    }


def _finish_batch(
    prefetched: dict,
    chain: ChainEngine,
    engine: GpuEngine,
    unified,  # gpu.unified.UnifiedStage
    stats: PipelineStats,
) -> Optional["_ResultBatch"]:
    """run the unified single-submit pipeline on the batch's gather+
    features output: segment + finalize_indirect + fit + per-article
    reduce in one GPU encoder, yielding per-article K_sums/G_sums/
    rms_sums directly. Brain.absorb_from_sums consumes these with no
    CPU reduceat.
    """
    import time as _t
    items = prefetched["items"]
    handle = prefetched["handle"]
    feat_ctx = prefetched["feat_ctx"]
    n_in = len(items)

    t0 = _t.perf_counter()
    result = unified.submit_fused(
        chain, engine, handle, feat_ctx,
        n_art=n_in,
    )
    fused_ms = (_t.perf_counter() - t0) * 1000.0
    gpu_gather_ms = chain.last_gather_ns / 1e6
    gpu_features_ms = chain.last_features_ns / 1e6

    if result is None or result["total_segs"] == 0:
        stats.timings.update(
            gather_submit_ms=prefetched["g_submit_ms"],
            feat_submit_ms=prefetched["f_submit_ms"],
            fit_wait_ms=fused_ms,
            gpu_gather_ms=gpu_gather_ms,
            gpu_features_ms=gpu_features_ms,
        )
        for _ in items:
            stats.inc("drops_dispatcher")
        return None

    # n_segs_per_art[a] > 0 means the article emitted at least one
    # segment. zero-segment articles are drops.
    n_segs_arr = result["n_segs_per_art"]
    accepted_mask = n_segs_arr > 0
    titles: list[str] = []
    Ts: list[int] = []
    byte_hists: list[Optional[np.ndarray]] = []
    any_hist = False
    for a in range(n_in):
        if accepted_mask[a]:
            titles.append(items[a].title)
            Ts.append(int(items[a].idxs.shape[0]))
            bh = getattr(items[a], "byte_hist", None)
            byte_hists.append(bh)
            if bh is not None:
                any_hist = True
        else:
            stats.inc("drops_dispatcher")

    if not titles:
        return None

    # tight-pack per-article arrays to only accepted articles
    K_sums = result["K_sums"][accepted_mask]
    G_sums = result["G_sums"][accepted_mask]
    rms_sums = result["rms_sums"][accepted_mask]
    n_segs_packed = n_segs_arr[accepted_mask].astype(np.int64)

    stats.timings.update(
        gather_submit_ms=prefetched["g_submit_ms"],
        feat_submit_ms=prefetched["f_submit_ms"],
        fit_wait_ms=fused_ms,
        gpu_gather_ms=gpu_gather_ms,
        gpu_features_ms=gpu_features_ms,
    )

    return _ResultBatch(
        titles=titles, Ts=Ts,
        K_sums=K_sums, G_sums=G_sums, rms_sums=rms_sums,
        n_segs_per_art=n_segs_packed,
        byte_hists=byte_hists if any_hist else None,
    )


class _BatchSizeTuner:
    """hill-climb controller for batch_size.

    every N batches, looks at art/s since the last adjustment and
    decides whether to keep current batch_size, grow it (if fixed
    overheads dominate), or shrink it (if per-batch cost scales with
    size). uses a simple momentum-direction-reversal strategy: if the
    last change helped, try more of the same; if it hurt, reverse.

    why this makes sense: with GPU timestamp queries showing actual
    kernel time ≈ 4 ms out of 370 ms batches, the per-batch fixed
    overhead is ~99% of time. bigger batches amortize that overhead
    across more articles. but beyond a point, absorb and segment work
    scale with batch size and caps dominate. the optimum depends on
    workload + hardware and shifts at runtime.
    """

    def __init__(self, initial: int, lo: int = 8, hi: int = 64,
                 window_batches: int = 3):
        self.cur = initial
        self.lo = lo
        self.hi = hi
        self.window = window_batches
        self.last_dir = +1  # +1 = grow, -1 = shrink
        self.last_rate: float = 0.0
        self._batches_since_change = 0
        self._t_window_start = time.time()
        self._absorbed_at_window_start = 0

    def maybe_update(self, absorbed_now: int) -> int:
        self._batches_since_change += 1
        if self._batches_since_change < self.window:
            return self.cur
        now = time.time()
        dt = max(now - self._t_window_start, 1e-9)
        rate = (absorbed_now - self._absorbed_at_window_start) / dt
        # decide direction
        if rate > self.last_rate * 1.02:
            # improving, continue
            new = self.cur + self.last_dir * 4
        elif rate < self.last_rate * 0.98:
            # regressing, reverse
            self.last_dir = -self.last_dir
            new = self.cur + self.last_dir * 4
        else:
            # within 2% — hold
            new = self.cur
        new = max(self.lo, min(self.hi, new))
        self.cur = new
        self.last_rate = rate
        self._batches_since_change = 0
        self._t_window_start = now
        self._absorbed_at_window_start = absorbed_now
        return self.cur


def _gpu_dispatcher(
    chain: "ChainEngine",
    engine: "GpuEngine",
    unified,  # gpu.unified.UnifiedStage
    staged_q: "queue.Queue",
    results_q: "queue.Queue",
    n_workers: int,
    batch_size: int,
    batch_timeout_ms: int,
    max_batch_bytes: int,
    stop_event: threading.Event,
    errors: PipelineErrors,
    stats: PipelineStats,
    auto_tune: bool = True,
) -> None:
    """single-chain dispatcher for the unified single-submit pipeline.

    batch lifecycle: collect items → submit gather+features → run
    _finish_batch (unified submit) → push _ResultBatch to absorb_worker.
    prefetch overlap is disabled — on this iGPU, stacking the next
    batch's gather+features BEFORE the current batch's unified submit
    occasionally loses the device (AMD driver TDR on deep queues with
    long-running segment kernels). serial submits still produce ~1000
    art/s because the unified kernel replaces the CPU numba segment
    bottleneck that forced the legacy path to overlap in the first place.
    """
    sentinels_state = [0]

    tuner = _BatchSizeTuner(batch_size) if auto_tune else None
    current_batch_size = batch_size

    try:
        holdover: list = []
        window = chain.window
        # per-dispatcher reorder state — makes article ordering
        # deterministic across runs regardless of n_cpu_workers.
        reorder = _ReorderState()

        while not stop_event.is_set():
            items, done = _collect_batch(
                staged_q, sentinels_state, current_batch_size,
                batch_timeout_ms, n_workers, stop_event,
                window=window, holdover=holdover, reorder=reorder)
            if not items:
                if done:
                    break
                continue
            try:
                current = _submit_gather_features(
                    chain, items, max_batch_bytes, stats)
            except Exception as e:
                errors.record("gpu_dispatcher submit", e)
                continue
            if current is None:
                continue

            try:
                batch = _finish_batch(current, chain, engine, unified, stats)
            except Exception as e:
                errors.record("gpu_dispatcher finish", e)
                batch = None

            if batch is not None:
                # 60s tolerates a slow checkpoint save on the absorb
                # side. at W=10k+ a save can serialize ~150MB which is
                # several seconds on a busy disk; the dispatcher must
                # wait through that without crashing the pipeline.
                try:
                    results_q.put(batch, timeout=60.0)
                    stats.inc("batches_dispatched")
                except queue.Full:
                    errors.record("gpu_dispatcher",
                                  RuntimeError("results_q full >60s"))
                    stop_event.set()
                    return

            if tuner is not None:
                current_batch_size = tuner.maybe_update(
                    stats.snapshot()["absorbed"])

        results_q.put(None)
    except Exception as e:
        errors.record("gpu_dispatcher outer", e)
        try:
            results_q.put(None, timeout=1.0)
        except queue.Full:
            pass


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
    brain,
    on_article: Optional[Callable[[str, str, dict], None]] = None,
) -> None:
    """consume _ResultBatch and route it through brain.absorb_from_sums.
    the GPU already produced per-article K/G/rms sums via
    per_article_reduce.wgsl, so the CPU apply is a trivial dict update
    — no reduceat, no K_stack upload.

    on_article: optional callback (title, well_name, measurement) fired
    per absorbed article. measurement is the EngramFile.measure_from_sums
    dict captured BEFORE absorb mutates the brain (matching measure()'s
    semantics) — carries dream_surprise, well_attention, well_profile,
    drift, dream_state. the full bi-processor signal (wells × dream ×
    attention) that the batched GPU path used to hide.

    CALLBACK ASYNC: when a callback is wired, a dedicated worker thread
    drains a queue and fires on_article per item. this keeps slow
    observers (e.g. writing to disk, pushing to a socket, heavy
    analysis) from throttling the absorb thread — the absorb path
    publishes (title, name, measurement) tuples and moves on. order is
    preserved because the queue is FIFO and only one callback worker
    consumes it.
    """
    cb_q: "queue.Queue" = queue.Queue(maxsize=10000)
    cb_stop = threading.Event()

    def _cb_worker():
        # fire on_article(title, name, measurement) for each item in
        # order. None is the shutdown sentinel. exceptions from the
        # user's callback are recorded and swallowed; one bad observer
        # shouldn't tear down ingestion.
        while True:
            try:
                item = cb_q.get(timeout=0.5)
            except queue.Empty:
                if cb_stop.is_set():
                    # drain any remaining work already queued
                    while True:
                        try:
                            item = cb_q.get_nowait()
                        except queue.Empty:
                            return
                        if item is None:
                            return
                        try:
                            on_article(*item)  # type: ignore
                        except Exception as cb_err:
                            errors.record("on_article", cb_err)
                    return
                continue
            if item is None:
                return
            try:
                on_article(*item)  # type: ignore
            except Exception as cb_err:
                errors.record("on_article", cb_err)

    cb_thread: Optional[threading.Thread] = None
    if on_article is not None:
        cb_thread = threading.Thread(
            target=_cb_worker, daemon=True, name="on_article_worker")
        cb_thread.start()

    try:
        since_ck = 0
        t_last = time.time()
        t_start = t_last
        # transition graph pointer: tracks the previous absorbed well so
        # we can record edges prev→curr as articles stream through. the
        # graph is a Markov chain over concepts — each ingest order edge
        # is one observation of "after reading X, we read Y."
        prev_well: Optional[str] = None

        while True:
            if stop_event.is_set():
                break
            try:
                batch = results_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if batch is None:
                break

            _t_absorb0 = time.perf_counter()
            n_art_out = len(batch.titles)

            with brain_lock:
                if n_art_out > 0:
                    # measure BEFORE absorb — captures pre-absorb
                    # bi-processor signal (well_attention, dream_surprise
                    # reflect the brain at time-of-observation, before
                    # this batch is folded in). measurement only runs
                    # when the callback is wired, so the hot path has
                    # zero cost when no observer is attached.
                    measurements: list[dict] = []
                    if on_article is not None:
                        try:
                            measurements = ef.measure_from_sums(
                                batch.K_sums, batch.G_sums,
                                batch.rms_sums, batch.n_segs_per_art)
                        except Exception as meas_err:
                            errors.record("measure_from_sums", meas_err)
                            measurements = [{} for _ in batch.titles]

                    names = brain.absorb_from_sums(
                        batch.K_sums, batch.G_sums,
                        batch.rms_sums, batch.n_segs_per_art)

                    # accumulate per-well byte histograms when the
                    # Source is byte-level. this is what populates
                    # ef._well_sum_raw in the unified path (the legacy
                    # absorb_text path had its own update; the unified
                    # path was dropping this signal, starving the
                    # compositional side of well naming and the raw-
                    # space neuroplastic embedding).
                    if (names is not None
                            and batch.byte_hists is not None):
                        for n, bh in zip(names, batch.byte_hists):
                            if bh is None or n == "unknown":
                                continue
                            if n not in ef._well_sum_raw:
                                ef._well_sum_raw[n] = np.zeros(
                                    256, dtype=np.float64)
                            ef._well_sum_raw[n] += bh

                    # transition graph: record prev_well → curr edges.
                    # "unknown" routes are skipped (no well to anchor to)
                    # but don't break the chain — we keep prev_well
                    # pointing at the last real well until another real
                    # route arrives, so a single unknown doesn't sever
                    # an otherwise meaningful thematic sequence.
                    if names is not None:
                        for n in names:
                            if n == "unknown":
                                continue
                            if prev_well is not None:
                                targets = ef._well_transitions.setdefault(
                                    prev_well, {})
                                targets[n] = targets.get(n, 0) + 1
                            prev_well = n

                    if on_article is not None and names is not None:
                        # enqueue for the callback worker. absorb
                        # doesn't block on observer latency.
                        if len(measurements) != len(names):
                            measurements = [{} for _ in names]
                        for t, n, m in zip(batch.titles, names, measurements):
                            try:
                                cb_q.put((t, n, m), timeout=5.0)
                            except queue.Full:
                                errors.record("on_article",
                                              RuntimeError("cb_q full >5s"))
                                break
                    stats.inc("absorbed", n_art_out)
                    since_ck += n_art_out

                if checkpoint_path and since_ck >= checkpoint_every:
                    ef.save(checkpoint_path)
                    since_ck = 0

            absorb_ms = (time.perf_counter() - _t_absorb0) * 1000.0
            stats.timings.update(absorb_ms=absorb_ms)

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
        # absorb_worker is the terminal stage — if it dies, the
        # pipeline cannot make progress. signal shutdown so the
        # dispatcher drains and the caller sees the failure instead
        # of a silent stall + partial absorb count.
        stop_event.set()
    finally:
        # drain callback queue and stop the worker. the sentinel is
        # picked up even if cb_stop hasn't fired yet, so this joins
        # cleanly even if absorb crashed partway through a batch.
        if cb_thread is not None:
            cb_stop.set()
            try:
                cb_q.put(None, timeout=1.0)
            except queue.Full:
                pass
            cb_thread.join(timeout=30.0)


# ── telemetry ──────────────────────────────────────────────────────────

def _telemetry(
    stop_event: threading.Event,
    stats: PipelineStats,
    raw_q: "queue.Queue",
    staged_q: "queue.Queue",
    results_q: "queue.Queue",
    interval_s: float = 10.0,
) -> None:
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
              f"drops={s['drops_cpu']}/{s['drops_dispatcher']:<4d} "
              f"batches={s['batches_dispatched']:<4d} "
              f"q={s['raw_q_depth']}/{s['staged_q_depth']}/{s['results_q_depth']} "
              f"rate={rate_window:.1f}/s "
              f"gather={s['gpu_gather_bytes_last']/1e6:.1f}MB "
              f"fit={s['gpu_fit_bytes_last']/1e6:.1f}MB")
        # per-stage breakdown — the TOP= field is the next-best lever
        if stats.timings.n_samples > 0:
            print(f"  {stats.timings.report()}")


# ── public API ────────────────────────────────────────────────────────

def ingest_hetero_v3(
    ef: EngramFile,
    codec: Engram,
    articles: Iterable[tuple[str, str]],
    idx_fn: Callable[[str], np.ndarray],
    chain: ChainEngine,
    label: str,
    engine: Optional[GpuEngine] = None,
    batch_size: int = 128,
    batch_timeout_ms: int = 200,
    max_batch_bytes: int = 512 * 1024 * 1024,
    n_cpu_workers: int = 4,
    queue_depth: Optional[int] = None,
    progress_every: int = 1000,
    # checkpoint cost scales with brain size: at W=8k+ a save serializes
    # ~150MB and blocks the absorb worker for several seconds (it holds
    # brain_lock during write). 5000 was fine for hundreds-of-wells
    # brains; for production-scale ingest where W reaches 10k+, save
    # every 50k articles instead. crash recovery loses up to 50k articles
    # of work, but the pipeline doesn't backpressure-crash.
    checkpoint_every: int = 50000,
    checkpoint_path: Optional[str] = None,
    telemetry_interval_s: float = 10.0,
    min_tokens: int = 10,
    on_article: Optional[Callable[[str, str, dict], None]] = None,
    escape_energy: Optional[float] = None,
    finalize: bool = True,
) -> int:
    """4-stage pipeline with GPU-resident embedding matrix.

    caller supplies a ChainEngine already holding the matrix + pairing
    on GPU. pipeline overlap is achieved by the dispatcher submitting
    the NEXT batch's gather+features BEFORE waiting on the current
    batch's features — the wgpu queue is FIFO so current completes
    before next even starts executing.

    pairing consistency: the reference pairing used by the brain's
    Packet must match the one uploaded to ChainEngine; this function
    enforces that by writing chain.pairing into ef._reference_pairing
    if the brain has none set.
    """
    if engine is None:
        if not gpu_available():
            raise RuntimeError("wgpu not available")
        engine = GpuEngine()

    # enforce pairing consistency between brain and chain
    if ef._reference_pairing is None:
        ef._reference_pairing = chain.pairing.astype(np.int32, copy=True)
    else:
        if not np.array_equal(ef._reference_pairing.astype(np.int32),
                              chain.pairing.astype(np.int32)):
            raise ValueError(
                "ef._reference_pairing disagrees with chain.pairing — "
                "rebuild ChainEngine from ef._reference_pairing")

    errors = PipelineErrors()
    stats = PipelineStats()
    stop_event = threading.Event()
    brain_lock = threading.Lock()

    qdepth = queue_depth if queue_depth is not None else batch_size * 2
    raw_q:     "queue.Queue" = queue.Queue(maxsize=qdepth)
    staged_q:  "queue.Queue" = queue.Queue(maxsize=qdepth)
    results_q: "queue.Queue" = queue.Queue(maxsize=max(2, qdepth // 8))

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
            args=(raw_q, staged_q, idx_fn, min_tokens,
                  stop_event, errors, stats, i),
            daemon=True, name=f"cpu_worker_{i}",
        )
        threads.append(t)

    # unified single-submit pipeline: fuses segment + finalize_indirect
    # + fit (indirect dispatch) + per-article reduce into one encoder.
    # on this iGPU it comfortably beats legacy at ~1000 art/s even without
    # batch-overlap prefetch (overlap caused queue-depth-driven device
    # losses on long-running segment kernels).
    from engram.gpu.unified import UnifiedStage
    unified_stage = UnifiedStage(engine.device, engine, P=chain.D // 2)

    disp_t = threading.Thread(
        target=_gpu_dispatcher,
        args=(chain, engine, unified_stage, staged_q, results_q,
              n_cpu_workers, batch_size, batch_timeout_ms, max_batch_bytes,
              stop_event, errors, stats),
        daemon=True, name="gpu_dispatcher",
    )
    threads.append(disp_t)

    # absorb via Brain — one unified CPU+GPU path. routes on GPU
    # (parallel argmin + escape decision), mutates ef's authoritative
    # state (well sums, dream buffer) on CPU. grows well capacity on
    # demand; no hardcoded ceiling.
    from engram.gpu.brain import Brain
    brain = Brain(ef, engine, escape_energy=escape_energy)
    print(f"  [absorb] Brain ready: {len(ef.wells())} wells synced")

    abs_t = threading.Thread(
        target=_absorb_worker,
        args=(ef, results_q, brain_lock,
              progress_every, checkpoint_every, checkpoint_path,
              stop_event, errors, stats, brain),
        kwargs={"on_article": on_article},
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

    abs_t.join()
    stop_event.set()
    for t in threads:
        if t.is_alive():
            t.join(timeout=10.0)

    dt = time.time() - t0
    s = stats.snapshot()
    print(f"  {s['absorbed']:,} articles from {label} in {dt:.0f}s "
          f"({s['absorbed']/max(dt,1):.0f}/s)  "
          f"drops={s['drops_cpu']}/{s['drops_dispatcher']} "
          f"batches={s['batches_dispatched']}")
    if stats.timings.n_samples > 0:
        print(f"  {stats.timings.report()}")

    # if a worker thread died, surface it to the caller rather than
    # silently returning a partial count. critical-stage failures
    # (absorb_worker, gpu_dispatcher) should raise; producer/cpu_worker
    # per-item failures are already soft-logged as drops.
    critical_stages = ("absorb_worker", "gpu_dispatcher")
    fatal = [e for e in errors.all()
             if any(stg in e[0] for stg in critical_stages)]
    if fatal:
        tag, exc = fatal[0]
        raise RuntimeError(
            f"[{label}] pipeline stage {tag!r} failed: {exc!r}; "
            f"absorbed {s['absorbed']} articles before failure")
    if errors.had_error():
        print(f"  [{label}] completed with {len(errors.all())} non-fatal errors")

    # end-of-ingest polish: crystallize the meta-oscillator so the
    # saved brain ships with its temporal-prediction layer ready,
    # consolidate near-duplicate wells, and compute physics+byte
    # labels. all operations are cheap (ms for typical W) and
    # derive everything from the brain's own state — no hardcoded
    # thresholds. opt out via finalize=False.
    if finalize:
        from engram.finalize import finalize_brain
        fin_t0 = time.time()
        fin_report = finalize_brain(ef)
        fin_dt = time.time() - fin_t0
        print(f"  [finalize] {fin_dt*1000:.0f}ms: "
              f"wells {fin_report['wells_before']}->"
              f"{fin_report['wells_after']} "
              f"(merged {fin_report['wells_merged']})  "
              f"crystallized={fin_report['crystallized']}  "
              f"named={fin_report['wells_named']}")

    return s["absorbed"]
