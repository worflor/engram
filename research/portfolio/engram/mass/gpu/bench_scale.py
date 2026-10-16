"""bench_scale.py — see if GPU beats numba at larger inputs."""

from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from numba import njit
from nano_sm import build_transition_table
from run_gpu_v3 import NanoScanGPUv3
from test_real import read_first_articles, ENWIKI


@njit(cache=True, boundscheck=False, nogil=True)
def _numba_scan(ta, T, out):
    s = np.uint8(0)
    out[0] = s
    for i in range(ta.shape[0]):
        s = T[s, ta[i]]
        out[i + 1] = s


def run():
    articles = read_first_articles(ENWIKI, n_articles=400)
    total_len = sum(len(wt) for _, wt in articles)
    print(f"have {len(articles)} articles, {total_len/1024/1024:.1f} MB total")
    print()

    # build several sizes by concatenating progressively
    texts = []
    acc = b""
    for _, wt in articles:
        acc += wt.encode("utf-8") + b"\n\n"
        if len(acc) >= 20 * 1024 * 1024:
            break
    # pull out test sizes
    sizes_mb = [0.5, 1, 2, 5, 10, 15]
    inputs = []
    for sz in sizes_mb:
        n = int(sz * 1024 * 1024)
        if n <= len(acc):
            inputs.append(acc[:n])
    print(f"using sizes: {[len(x)/1024/1024 for x in inputs]} MB")
    print()

    T = build_transition_table()
    gpu = NanoScanGPUv3()

    # warmup
    warm = inputs[0][:1024]
    ta = np.frombuffer(warm, dtype=np.uint8)
    out = np.empty(ta.shape[0] + 1, dtype=np.uint8)
    _numba_scan(ta, T, out)
    gpu.run(warm); gpu.run(warm)

    def timeit(fn, k=3):
        best = float("inf")
        for _ in range(k):
            t0 = time.perf_counter()
            fn()
            dt = time.perf_counter() - t0
            if dt < best:
                best = dt
        return best

    print(f"{'size MB':>8s} {'numba ms':>10s} {'numba MB/s':>11s} "
          f"{'gpu ms':>10s} {'gpu MB/s':>11s} {'speedup':>9s}")
    print("-" * 70)

    for data in inputs:
        ta = np.frombuffer(data, dtype=np.uint8)
        out = np.empty(ta.shape[0] + 1, dtype=np.uint8)

        t_cpu = timeit(lambda: _numba_scan(ta, T, out))
        t_gpu = timeit(lambda: gpu.run(data))

        n = len(data)
        print(f"{n/1024/1024:>8.1f} {t_cpu*1000:>10.2f} "
              f"{n/t_cpu/1e6:>11.1f} {t_gpu*1000:>10.2f} "
              f"{n/t_gpu/1e6:>11.1f} {t_cpu/t_gpu:>8.2f}x")


if __name__ == "__main__":
    run()
