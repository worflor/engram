"""bench_gpu_v2.py — time the GPU parallel scan vs numba scalar."""

from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from nano_sm import build_transition_table, S_PROSE
from run_gpu_v2 import NanoScanGPUv2
from test_real import read_first_articles, ENWIKI


def numba_scalar(text: bytes, T):
    from numba import njit
    text_arr = np.frombuffer(text, dtype=np.uint8)
    states = np.empty(text_arr.shape[0] + 1, dtype=np.uint8)

    @njit(cache=True, boundscheck=False, nogil=True)
    def _inner(ta, T, out):
        s = np.uint8(0)
        out[0] = s
        for i in range(ta.shape[0]):
            s = T[s, ta[i]]
            out[i + 1] = s
    _inner(text_arr, T, states)
    return states


def run():
    if not Path(ENWIKI).exists():
        print("enwiki missing"); return
    articles = read_first_articles(ENWIKI, n_articles=40)
    big = b"".join(wt.encode("utf-8") + b"\n\n" for _, wt in articles)
    print(f"input: {len(big)/1024:.0f} KB")
    print()

    T = build_transition_table()
    gpu = NanoScanGPUv2()

    # warmup
    numba_scalar(big[:1024], T)
    gpu.run(big[:1024])
    gpu.run(big[:1024])

    # best-of-3
    def timeit(fn, k=3):
        best = float("inf")
        for _ in range(k):
            t0 = time.perf_counter()
            fn()
            dt = time.perf_counter() - t0
            if dt < best: best = dt
        return best

    t_cpu = timeit(lambda: numba_scalar(big, T))
    t_gpu = timeit(lambda: gpu.run(big))

    print(f"{'path':<20s} {'ms':>10s} {'MB/s':>10s}")
    print("-" * 50)
    print(f"{'numba scalar':<20s} {t_cpu*1000:>10.2f} {len(big)/t_cpu/1e6:>10.1f}")
    print(f"{'gpu parallel scan':<20s} {t_gpu*1000:>10.2f} {len(big)/t_gpu/1e6:>10.1f}")
    print(f"speedup: {t_cpu/t_gpu:.2f}x")

    # correctness
    cpu_s = numba_scalar(big, T)
    gpu_s = gpu.run(big)
    print()
    print(f"correctness: {(cpu_s == gpu_s).all()}")


if __name__ == "__main__":
    run()
