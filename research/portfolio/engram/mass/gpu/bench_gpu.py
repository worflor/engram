"""
bench_gpu.py — measure GPU parallel-scan MASS on real enwiki.

we run three things on the SAME byte buffer:
  A. numba scalar MASS (sequential state machine — the "CPU ceiling")
  B. Python reference scan (for correctness spot-check)
  C. GPU Hillis-Steele scan (the eldritch one)

check: A and C produce identical state arrays.
time: throughput in MB/s for each.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nano_sm import build_transition_table, run_sequential, S_PROSE, N_STATES
from run_gpu import NanoScanGPU

# we need real wikitext
from test_real import read_first_articles, ENWIKI


def numba_scalar_mass(text: bytes, T: np.ndarray) -> np.ndarray:
    """same as run_sequential but vectorized for speed."""
    n = len(text)
    text_arr = np.frombuffer(text, dtype=np.uint8)
    states = np.empty(n + 1, dtype=np.uint8)
    states[0] = S_PROSE
    # can't vectorize due to sequential dependency, but we can JIT
    try:
        from numba import njit

        @njit(cache=True, boundscheck=False, nogil=True)
        def _inner(text_arr, T, out):
            s = np.uint8(0)  # S_PROSE
            out[0] = s
            for i in range(text_arr.shape[0]):
                s = T[s, text_arr[i]]
                out[i + 1] = s
        _inner(text_arr, T, states)
    except Exception:
        for i in range(n):
            states[i + 1] = T[states[i], text_arr[i]]
    return states


def run():
    if not Path(ENWIKI).exists():
        print("enwiki bz2 missing"); return

    print("streaming enwiki samples...")
    articles = read_first_articles(ENWIKI, n_articles=40)
    # concatenate into one big buffer
    big = b"".join(wt.encode("utf-8") + b"\n\n" for _, wt in articles)
    print(f"  {len(big)/1024:.0f} KB total raw wikitext")
    print()

    T = build_transition_table()
    gpu = NanoScanGPU()

    # warmup
    _ = numba_scalar_mass(big[:1024], T)
    _ = gpu.run(big[:1024])

    # numba scalar (the CPU ceiling for the state machine alone)
    t0 = time.perf_counter()
    cpu_states = numba_scalar_mass(big, T)
    t_cpu = time.perf_counter() - t0

    # gpu
    t0 = time.perf_counter()
    gpu_states = gpu.run(big)
    t_gpu = time.perf_counter() - t0

    print(f"input: {len(big)/1024:.0f} KB  ({len(big):,} bytes)")
    print()
    print(f"{'path':<20s} {'ms':>10s} {'MB/s':>10s}")
    print("-" * 45)
    print(f"{'numba scalar':<20s} {t_cpu*1000:>10.2f} {len(big)/t_cpu/1e6:>10.1f}")
    print(f"{'gpu parallel scan':<20s} {t_gpu*1000:>10.2f} {len(big)/t_gpu/1e6:>10.1f}")
    print(f"speedup: {t_cpu/t_gpu:.1f}x")
    print()

    # correctness
    match = (cpu_states == gpu_states).all()
    print(f"state arrays match: {match}")
    if not match:
        diff = np.where(cpu_states != gpu_states)[0]
        print(f"  first diff at position {diff[0]}")
        print(f"  cpu[{diff[0]-3}:{diff[0]+5}] = {list(cpu_states[max(0,diff[0]-3):diff[0]+5])}")
        print(f"  gpu[{diff[0]-3}:{diff[0]+5}] = {list(gpu_states[max(0,diff[0]-3):diff[0]+5])}")

    # quick repeat to see if second call is faster (buffers cached etc)
    print()
    print("repeat timings (second run uses cached buffers):")
    for _ in range(3):
        t0 = time.perf_counter()
        gpu.run(big)
        print(f"  gpu: {(time.perf_counter()-t0)*1000:.2f} ms")


if __name__ == "__main__":
    run()
