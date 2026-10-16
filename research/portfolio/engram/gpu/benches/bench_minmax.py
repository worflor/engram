"""
bench_minmax.py

push the gpu. the goal is to know, for this exact workload on this exact
hardware, how close we can get to peak bandwidth and what the blocking
factor is.

experiments:
  E1 — pure compute time (data pre-uploaded, persistent buffers)
  E2 — workgroup size sweep (32, 64, 128, 256)
  E3 — amortized cost (same data, N dispatches back to back)
  E4 — what if we never read back? (proves readback isn't the bottleneck)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

import numpy as np
import wgpu

from engram_codec import MIN_BLOCK, MAX_BLOCK
from engram.gpu.engine import GpuEngine, pack_batch, gpu_available


def synth_segment(T, D, seed):
    rng = np.random.default_rng(seed)
    P = D // 2
    freqs = rng.uniform(0.1, 2.0, size=P)
    damps = rng.uniform(0.90, 0.995, size=P)
    K = 2.0 * damps * np.cos(freqs)
    G = damps * damps
    z = np.zeros((T, P), dtype=np.complex128)
    z[0] = rng.standard_normal(P) + 1j * rng.standard_normal(P)
    z[1] = rng.standard_normal(P) + 1j * rng.standard_normal(P)
    for n in range(2, T):
        z[n] = K * z[n - 1] - G * z[n - 2]
    z = z + 0.01 * (rng.standard_normal((T, P)) + 1j * rng.standard_normal((T, P)))
    out = np.empty((T, D), dtype=np.float32)
    out[:, 0::2] = z.real.astype(np.float32)
    out[:, 1::2] = z.imag.astype(np.float32)
    return out


def synth_batch(n_articles, dim, seed=0):
    rng = np.random.default_rng(seed)
    segments = []
    for i in range(n_articles):
        n_segs_here = int(rng.integers(3, 8))
        for j in range(n_segs_here):
            T = int(rng.integers(MIN_BLOCK + 4, MAX_BLOCK + 1))
            segments.append(synth_segment(T, dim, seed + 1000 * i + j))
    return segments


def prep(engine: GpuEngine, traj, segs):
    """upload once, return what the hot path needs."""
    n_segs = int(segs.shape[0])
    D = int(segs[0, 2])
    P = D // 2
    out_len = int((segs[:, 3] + P).max())

    engine._ensure_traj(traj.size)
    engine._ensure_segs(n_segs)
    engine._ensure_outputs(out_len)

    engine.device.queue.write_buffer(engine._traj_buf, 0, memoryview(traj))
    engine.device.queue.write_buffer(engine._seg_buf,  0, memoryview(segs))

    bg = engine.device.create_bind_group(
        layout=engine._bind_layout,
        entries=[
            {"binding": 0, "resource": {"buffer": engine._traj_buf,
                                         "offset": 0, "size": engine._traj_cap}},
            {"binding": 1, "resource": {"buffer": engine._seg_buf,
                                         "offset": 0, "size": engine._seg_cap}},
            {"binding": 2, "resource": {"buffer": engine._k_buf,
                                         "offset": 0, "size": engine._out_cap * 8}},
            {"binding": 3, "resource": {"buffer": engine._g_buf,
                                         "offset": 0, "size": engine._out_cap * 8}},
            {"binding": 4, "resource": {"buffer": engine._rms_buf,
                                         "offset": 0, "size": engine._out_cap * 4}},
        ],
    )
    return n_segs, P, out_len, bg


def dispatch_only(engine: GpuEngine, n_segs, P, out_len, bg,
                  with_readback=True, n_dispatches=1):
    """dispatch n_dispatches times, optionally read back."""
    enc = engine.device.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(engine._pipeline)
    cp.set_bind_group(0, bg)
    for _ in range(n_dispatches):
        shader = engine._shader_name
        if ("v3" in shader or "v4" in shader or "v5" in shader
                or "v8" in shader):
            WG = 64
            pair_tiles = (P + WG - 1) // WG
            cp.dispatch_workgroups(n_segs, pair_tiles, 1)
        elif "v2" in shader:
            cp.dispatch_workgroups(n_segs, 1, 1)
        else:
            cp.dispatch_workgroups(n_segs, P, 1)
    cp.end()
    if with_readback:
        kg_bytes = out_len * 8
        rms_bytes = out_len * 4
        enc.copy_buffer_to_buffer(engine._k_buf,   0, engine._k_rb,   0, kg_bytes)
        enc.copy_buffer_to_buffer(engine._g_buf,   0, engine._g_rb,   0, kg_bytes)
        enc.copy_buffer_to_buffer(engine._rms_buf, 0, engine._rms_rb, 0, rms_bytes)
    engine.device.queue.submit([enc.finish()])

    t0 = time.perf_counter()
    if with_readback:
        engine._k_rb.map_sync(wgpu.MapMode.READ,   size=kg_bytes)
        engine._g_rb.map_sync(wgpu.MapMode.READ,   size=kg_bytes)
        engine._rms_rb.map_sync(wgpu.MapMode.READ, size=rms_bytes)
        engine._k_rb.unmap()
        engine._g_rb.unmap()
        engine._rms_rb.unmap()
    else:
        # to force GPU completion without readback, issue a tiny no-op
        # and sync on that. we just submit and immediately destroy the
        # encoder — wgpu_sync below.
        # actually the cleanest way is a noop writeBuffer + a dummy
        # readback of 1 byte on a separate buffer. but the simplest
        # sync primitive in wgpu-py is poll(). use that.
        engine.device._poll_wait()
    return time.perf_counter() - t0


def run_config(shader, n_articles, with_readback=True, dispatches_per_submit=1, iters=5):
    engine = GpuEngine(shader=shader)
    segments = synth_batch(n_articles, 300, seed=42)
    traj, segs = pack_batch(segments)
    n_segs, P, out_len, bg = prep(engine, traj, segs)

    # warmup
    for _ in range(3):
        dispatch_only(engine, n_segs, P, out_len, bg,
                      with_readback=with_readback,
                      n_dispatches=dispatches_per_submit)

    best = float("inf")
    for _ in range(iters):
        dt = dispatch_only(engine, n_segs, P, out_len, bg,
                           with_readback=with_readback,
                           n_dispatches=dispatches_per_submit)
        if dt < best:
            best = dt
    return {
        "ms": best * 1e3,
        "ms_per_dispatch": best * 1e3 / dispatches_per_submit,
        "data_mb": traj.nbytes / 1e6,
        "eff_gbps": (traj.nbytes / 1e9) / best * dispatches_per_submit,
        "n_segs": n_segs,
        "workgroups": n_segs * P,
    }


def run():
    if not gpu_available():
        print("wgpu not available")
        return

    # basic adapter info
    e0 = GpuEngine()
    print(f"adapter: {e0.describe()}")
    print()

    # E1: pure compute (persistent upload, with readback)
    print("== E1: pure compute (data pre-uploaded, with readback) ==")
    print(f"{'shader':>24} {'n_art':>6} {'segs':>6} {'WGs':>8} {'dMB':>6} {'ms':>8} {'GB/s':>7}")
    for shader in ["fit_pair.wgsl", "fit_pair_v3.wgsl", "fit_pair_v4.wgsl", "fit_pair_v5_noresid.wgsl"]:
        for n_art in (256, 512, 1024, 2048):
            r = run_config(shader, n_art)
            wgs = r['workgroups']
            if "v3" in shader or "v4" in shader or "v5" in shader:
                WG = 64
                P = 150
                pair_tiles = (P + WG - 1) // WG
                wgs = r['n_segs'] * pair_tiles
            print(f"{shader:>24} {n_art:>6} {r['n_segs']:>6} "
                  f"{wgs:>8} {r['data_mb']:>6.1f} "
                  f"{r['ms']:>7.2f} {r['eff_gbps']:>6.2f}")
    print()

    # E2: no-readback (measures dispatch+compute only)
    print("== E2: same, no readback (GPU completion via poll) ==")
    print(f"{'shader':>24} {'n_art':>6} {'ms':>8} {'GB/s':>7}")
    for shader in ["fit_pair_v3.wgsl", "fit_pair_v4.wgsl", "fit_pair_v5_noresid.wgsl"]:
        for n_art in (256, 512, 1024):
            r = run_config(shader, n_art, with_readback=False)
            print(f"{shader:>24} {n_art:>6} {r['ms']:>7.2f} {r['eff_gbps']:>6.2f}")


if __name__ == "__main__":
    run()
