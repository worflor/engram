"""
engram_bench.py

how fast is it? how much memory does it use? how does it scale?

benchmarks encode, decode, dream across trajectory lengths.
measures wall time, peak memory, and throughput.
compares against the cost of what it replaces (transformer inference).
"""

from __future__ import annotations

import sys
import os
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from engram_codec import Engram, summary, encode_block, decode_block, fit_pair, _to_complex, MIN_BLOCK


def bench_encode_decode(codec: Engram, T: int, D: int, n_runs: int = 3) -> dict:
    """benchmark encode + decode at a given trajectory length."""
    rng = np.random.RandomState(42)
    w = rng.randn(T, D).astype(np.float32) * 0.3

    # warmup
    codec.encode(w[:min(50, T)])

    # encode
    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        pkt = codec.encode(w)
    enc_time = (time.perf_counter() - t0) / n_runs
    enc_mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    s = summary(pkt)

    # decode
    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        dec = codec.decode(pkt)
    dec_time = (time.perf_counter() - t0) / n_runs
    dec_mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    err = float(np.sqrt(np.mean((w - dec) ** 2)))

    return {
        "T": T, "D": D,
        "enc_ms": enc_time * 1000,
        "dec_ms": dec_time * 1000,
        "enc_mem_kb": enc_mem / 1024,
        "dec_mem_kb": dec_mem / 1024,
        "blocks": s["blocks"],
        "capture": s["capture"],
        "err": err,
        "samples_per_sec_enc": T / max(enc_time, 1e-9),
        "samples_per_sec_dec": T / max(dec_time, 1e-9),
    }


def bench_dream(codec: Engram, n_seed_packets: int, n_blocks: int, D: int) -> dict:
    """benchmark dream generation."""
    rng = np.random.RandomState(42)
    packets = []
    for i in range(n_seed_packets):
        w = rng.randn(100, D).astype(np.float32) * 0.3
        packets.append(codec.encode(w))

    tracemalloc.start()
    t0 = time.perf_counter()
    gen = codec.dream(packets, n_blocks=n_blocks, samples_per_block=12)
    dream_time = time.perf_counter() - t0
    dream_mem = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    gen_samples = gen.shape[0] if gen is not None else 0
    energy = float(np.sum(gen ** 2)) if gen is not None else 0.0

    return {
        "n_seeds": n_seed_packets,
        "n_blocks": n_blocks,
        "dream_ms": dream_time * 1000,
        "dream_mem_kb": dream_mem / 1024,
        "gen_samples": gen_samples,
        "energy": energy,
    }


def bench_fit_pair_raw(T: int, n_runs: int = 100) -> dict:
    """benchmark the core algorithm: fit_pair on a single complex series."""
    rng = np.random.RandomState(42)
    z = (rng.randn(T) + 1j * rng.randn(T)).astype(np.complex128)

    # warmup
    fit_pair(z)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        fit_pair(z)
    total = time.perf_counter() - t0

    return {
        "T": T,
        "fit_us": total / n_runs * 1e6,
        "fits_per_sec": n_runs / total,
    }


def run():
    print("=" * 70)
    print("  ENGRAM BENCHMARK")
    print("  time, memory, throughput, scaling")
    print("=" * 70)
    print()

    D = 300  # GloVe dimension
    codec = Engram(dim=D)

    # 1. core algorithm: fit_pair
    print("--- fit_pair (the one algorithm) ---\n")
    print(f"  {'T':>8s} {'time_us':>10s} {'fits/sec':>12s}")
    for T in [10, 50, 100, 500, 1000]:
        r = bench_fit_pair_raw(T)
        print(f"  {r['T']:8d} {r['fit_us']:10.1f} {r['fits_per_sec']:12.0f}")

    # 2. encode/decode scaling with trajectory length
    print("\n--- encode/decode scaling ---\n")
    print(f"  {'T':>8s} {'enc_ms':>10s} {'dec_ms':>10s} {'enc_KB':>10s} {'dec_KB':>10s} "
          f"{'blocks':>8s} {'cap%':>8s} {'enc samp/s':>12s} {'dec samp/s':>12s}")

    for T in [50, 100, 500, 1000, 2000, 5000, 10000]:
        r = bench_encode_decode(codec, T, D, n_runs=2)
        print(f"  {r['T']:8d} {r['enc_ms']:10.1f} {r['dec_ms']:10.1f} "
              f"{r['enc_mem_kb']:10.0f} {r['dec_mem_kb']:10.0f} "
              f"{r['blocks']:8d} {r['capture']:8.1f} "
              f"{r['samples_per_sec_enc']:12.0f} {r['samples_per_sec_dec']:12.0f}")

    # 3. dream scaling with pool size
    print("\n--- dream scaling with pool size ---\n")
    print(f"  {'seeds':>8s} {'blocks':>8s} {'dream_ms':>10s} {'mem_KB':>10s} {'samples':>10s}")
    for n_seeds in [3, 5, 10, 20, 50]:
        r = bench_dream(codec, n_seeds, n_blocks=6, D=D)
        print(f"  {r['n_seeds']:8d} {r['n_blocks']:8d} {r['dream_ms']:10.1f} "
              f"{r['dream_mem_kb']:10.0f} {r['gen_samples']:10d}")

    # 4. dimension scaling
    print("\n--- dimension scaling (T=500) ---\n")
    print(f"  {'D':>8s} {'enc_ms':>10s} {'dec_ms':>10s} {'pairs':>8s}")
    for D_test in [16, 64, 128, 256, 300, 512]:
        c = Engram(dim=D_test)
        r = bench_encode_decode(c, 500, D_test, n_runs=2)
        print(f"  {D_test:8d} {r['enc_ms']:10.1f} {r['dec_ms']:10.1f} {D_test//2:8d}")

    # 5. comparison: what does this replace?
    print("\n--- cost comparison ---\n")
    r_1k = bench_encode_decode(codec, 1000, 300, n_runs=2)
    r_5k = bench_encode_decode(codec, 5000, 300, n_runs=2)
    r_10k = bench_encode_decode(codec, 10000, 300, n_runs=2)

    print(f"  Engram encode 1K samples:   {r_1k['enc_ms']:8.1f} ms  ({r_1k['enc_mem_kb']/1024:.1f} MB peak)")
    print(f"  Engram encode 5K samples:   {r_5k['enc_ms']:8.1f} ms  ({r_5k['enc_mem_kb']/1024:.1f} MB peak)")
    print(f"  Engram encode 10K samples:  {r_10k['enc_ms']:8.1f} ms  ({r_10k['enc_mem_kb']/1024:.1f} MB peak)")
    print()
    print(f"  for comparison (typical LLM costs on CPU):")
    print(f"    Qwen 0.8B inference 1K tokens:   ~300,000 ms  (~2,000 MB)")
    print(f"    sentence-transformers 1K chunks:   ~10,000 ms  (~500 MB)")
    print(f"    GloVe lookup 1K words:                  <1 ms  (~400 MB for vocab)")
    print()
    print(f"  Engram encode is {300000 / max(r_1k['enc_ms'], 0.01):.0f}x faster than Qwen inference")
    print(f"  Engram encode is {10000 / max(r_1k['enc_ms'], 0.01):.0f}x faster than sentence-transformers")
    print(f"  Engram peak memory is {r_10k['enc_mem_kb']/1024:.1f} MB for 10K samples")

    # 6. throughput summary
    print("\n--- throughput summary ---\n")
    print(f"  encode: {r_5k['samples_per_sec_enc']:.0f} samples/sec at D=300")
    print(f"  decode: {r_5k['samples_per_sec_dec']:.0f} samples/sec at D=300")
    print(f"  that's {r_5k['samples_per_sec_enc'] * 300 / 1e6:.1f}M float ops/sec for encode")


if __name__ == "__main__":
    run()
