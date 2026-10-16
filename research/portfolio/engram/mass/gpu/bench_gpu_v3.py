"""bench_gpu_v3.py — time recursive-spine GPU scan vs numba."""

from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from nano_sm import build_transition_table, S_PROSE
from run_gpu_v3 import NanoScanGPUv3
from test_real import read_first_articles, ENWIKI


def numba_scalar(text, T):
    from numba import njit
    ta = np.frombuffer(text, dtype=np.uint8)
    out = np.empty(ta.shape[0] + 1, dtype=np.uint8)

    @njit(cache=True, boundscheck=False, nogil=True)
    def _inner(ta, T, out):
        s = np.uint8(0)
        out[0] = s
        for i in range(ta.shape[0]):
            s = T[s, ta[i]]
            out[i + 1] = s
    _inner(ta, T, out)
    return out


def run():
    if not Path(ENWIKI).exists():
        print("enwiki missing"); return

    articles = read_first_articles(ENWIKI, n_articles=40)
    big = b"".join(wt.encode("utf-8") + b"\n\n" for _, wt in articles)
    print(f"input: {len(big)/1024:.0f} KB ({len(big):,} bytes)")
    print()

    T = build_transition_table()
    gpu = NanoScanGPUv3()

    # warmup
    numba_scalar(big[:1024], T)
    gpu.run(big[:1024])
    gpu.run(big[:1024])

    def timeit(fn, k=3):
        best = float("inf")
        for _ in range(k):
            t0 = time.perf_counter()
            fn()
            dt = time.perf_counter() - t0
            if dt < best:
                best = dt
        return best

    t_cpu = timeit(lambda: numba_scalar(big, T))
    t_gpu = timeit(lambda: gpu.run(big))

    print(f"{'path':<30s} {'ms':>10s} {'MB/s':>10s}")
    print("-" * 55)
    print(f"{'numba scalar':<30s} {t_cpu*1000:>10.2f} "
          f"{len(big)/t_cpu/1e6:>10.1f}")
    print(f"{'gpu recursive-spine':<30s} {t_gpu*1000:>10.2f} "
          f"{len(big)/t_gpu/1e6:>10.1f}")
    print(f"speedup: {t_cpu/t_gpu:.2f}x")
    print()

    cpu_s = numba_scalar(big, T)
    gpu_s = gpu.run(big)
    match = (cpu_s == gpu_s).all()
    print(f"correctness: {match}")
    if not match:
        diff = np.where(cpu_s != gpu_s)[0][:5]
        print(f"  first diff: {diff}")
        for d in diff:
            print(f"  pos {d}: cpu={cpu_s[d]} gpu={gpu_s[d]}")


if __name__ == "__main__":
    run()
