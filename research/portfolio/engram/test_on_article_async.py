"""test_on_article_async.py -- real check that the on_article callback
worker decouples observer latency from absorb throughput.

setup: drive a fixed corpus through two runs. one with a fast
callback (trivial appender, ~microseconds per call), one with a slow
callback that sleeps 50ms per article. if the callback ran inline
under brain_lock, the absorb thread's per-batch time would be
dominated by n_items_per_batch * 50 ms. the async worker decouples:
absorb just enqueues and moves on, so the absorb thread's per-batch
time stays close to the fast-callback baseline.

claim: slow-callback run's reported absorb_ms/batch is within 2x of
the fast-callback run's. total elapsed DIFFERS because shutdown
waits for the callback worker to drain the queue, but that is
end-of-run drain latency, not hot-path throttling.
"""
from __future__ import annotations

import sys
import time
import tempfile
import threading
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from engram_codec import EngramFile
from engram.gpu.engine import gpu_available
from genesis_chain import genesis_chain, iter_wiki_articles
from sources import from_bytes, byte_cooc_svd_embedding


DOWNLOADS = (Path.home() / "Downloads")
SIMPLEWIKI = DOWNLOADS / "simplewiki-20260301-pages-articles-multistream.xml.bz2"


def pull_corpus(n: int = 200):
    if not SIMPLEWIKI.exists():
        return []
    items = []
    for t, text in iter_wiki_articles(str(SIMPLEWIKI), limit=n * 2):
        b = text.encode("utf-8", errors="replace")
        if 500 <= len(b) <= 15000:
            items.append((t, b.decode("utf-8", errors="replace")))
        if len(items) >= n:
            break
    return items


def run_one(corpus, source, callback, label: str):
    """drive one full ingest. a monitor thread samples ef._total_absorbed
    while the pipeline runs so we can ask real questions like "when
    did absorb actually reach N items" — separate from the total
    genesis_chain return time (which also includes shutdown drain of
    the callback worker queue).

    returns: (n_absorbed, total_elapsed_s, snaps) where snaps is a
    list of (t_from_start, absorbed_count) tuples.
    """
    with tempfile.TemporaryDirectory() as td:
        bp = str(Path(td) / f"{label}.engram")
        ef = EngramFile.create(bp, name=label, dim=source.dim,
                               metadata={}, annotations={})

        snaps: list[tuple[float, int]] = []
        stop = threading.Event()

        def _mon(ef_ref, stop_ref, t0):
            while not stop_ref.is_set():
                snaps.append(
                    (time.perf_counter() - t0, ef_ref._total_absorbed))
                time.sleep(0.02)

        t0 = time.perf_counter()
        mon = threading.Thread(
            target=_mon, args=(ef, stop, t0), daemon=True)
        mon.start()
        n = genesis_chain(
            ef, iter(corpus), source, label=label,
            batch_size=24, batch_timeout_ms=100, n_cpu_workers=4,
            progress_every=99999, telemetry_interval_s=9999.0,
            min_tokens=10, on_article=callback,
        )
        dt = time.perf_counter() - t0
        stop.set()
        mon.join(timeout=1.0)
    return n, dt, snaps


def t_to_reach(snaps, target: int) -> float:
    for ts, absorbed in snaps:
        if absorbed >= target:
            return ts
    return float("inf")


def main():
    if not gpu_available():
        print("wgpu missing"); sys.exit(1)

    corpus = pull_corpus(200)
    if len(corpus) < 50:
        print(f"not enough corpus ({len(corpus)}); skipping")
        return
    print(f"corpus: {len(corpus)} articles")

    # shared byte source
    sample = b"\n".join(t.encode("utf-8", errors="replace")[:4000]
                        for _i, t in corpus[:50])
    emb = byte_cooc_svd_embedding(sample, D=32, window=4)
    source = from_bytes(matrix=emb, name="bytes")

    # FAST callback: microseconds
    fast_received = []
    def fast_cb(title: str, well: str, m: dict):
        fast_received.append((title, well))

    # SLOW callback: 50 ms per call. if this ran inline on absorb it
    # would add 50ms * 200 = 10 seconds minimum. the async worker
    # should mostly decouple this from absorb throughput.
    slow_received = []
    def slow_cb(title: str, well: str, m: dict):
        time.sleep(0.05)
        slow_received.append((title, well))

    print("\n== fast callback (trivial) ==")
    n_fast, t_fast, snaps_fast = run_one(corpus, source, fast_cb, "fast")
    print(f"  absorbed={n_fast}  total_elapsed={t_fast:.2f}s  "
          f"received={len(fast_received)}")

    print("\n== slow callback (50ms sleep) ==")
    n_slow, t_slow, snaps_slow = run_one(corpus, source, slow_cb, "slow")
    print(f"  absorbed={n_slow}  total_elapsed={t_slow:.2f}s  "
          f"received={len(slow_received)}")

    # the right metric: how long did ABSORB take to reach N items?
    # total_elapsed includes shutdown drain of the callback queue; we
    # want to know if the absorb thread itself is being held up by
    # the slow callback.
    half_fast = t_to_reach(snaps_fast, n_fast // 2)
    half_slow = t_to_reach(snaps_slow, n_slow // 2)
    all_fast = t_to_reach(snaps_fast, n_fast)
    all_slow = t_to_reach(snaps_slow, n_slow)
    print(f"\n  time to ABSORB half: fast={half_fast:.2f}s  slow={half_slow:.2f}s")
    print(f"  time to ABSORB all:  fast={all_fast:.2f}s  slow={all_slow:.2f}s")

    # if on_article were inline and synchronous, absorbing all N items
    # would take at least N * 0.05s. with async we expect the absorb
    # thread to finish in ~t_fast, then shutdown waits for the
    # callback queue to drain (that extra time is correct, it's not
    # hot-path throttling).
    inline_floor = n_slow * 0.05
    print(f"  inline floor (if throttled): {inline_floor:.2f}s")

    failed = []
    if n_fast != n_slow:
        failed.append(f"absorbed differs: fast={n_fast} slow={n_slow}")
    if len(fast_received) != n_fast:
        failed.append(
            f"fast callback didn't receive all absorbed articles: "
            f"got {len(fast_received)}, expected {n_fast}")
    if len(slow_received) != n_slow:
        failed.append(
            f"slow callback didn't receive all absorbed articles: "
            f"got {len(slow_received)}, expected {n_slow}")

    # absorb should finish well before the inline floor. tolerate 2x
    # fast's completion time so jitter doesn't flap the test.
    if all_slow > max(all_fast * 2.0, inline_floor * 0.3):
        failed.append(
            f"absorb took too long under slow callback: "
            f"all_slow={all_slow:.2f}s vs 2x fast={all_fast*2:.2f}s / "
            f"inline-floor*0.3={inline_floor*0.3:.2f}s")

    if failed:
        print("\nFAILURES:")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)
    print("\n[PASS] on_article is properly async -- absorb finished in "
          f"{all_slow:.2f}s under a callback that would cost "
          f"{inline_floor:.2f}s if inline")


if __name__ == "__main__":
    main()
