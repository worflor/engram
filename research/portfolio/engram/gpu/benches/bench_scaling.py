"""
bench_scaling.py

push v1 past n=512 to find the real iGPU plateau, and break down
per-batch time into (upload, dispatch, readback, python overhead).
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


def timed_phases(engine: GpuEngine, traj_flat, segments):
    """manual instrumentation of upload / dispatch+readback."""
    n_segs = int(segments.shape[0])
    D = int(segments[0, 2])
    P = D // 2
    out_len = int((segments[:, 3] + P).max())

    engine._ensure_traj(traj_flat.size)
    engine._ensure_segs(n_segs)
    engine._ensure_outputs(out_len)

    # phase 1: upload via memoryview (avoids python-side copy)
    t0 = time.perf_counter()
    engine.device.queue.write_buffer(engine._traj_buf, 0, memoryview(traj_flat))
    engine.device.queue.write_buffer(engine._seg_buf,  0, memoryview(segments))
    t_upload = time.perf_counter() - t0

    # phase 2: dispatch + readback (bundled because wgpu doesn't split)
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

    t0 = time.perf_counter()
    enc = engine.device.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(engine._pipeline)
    cp.set_bind_group(0, bg)
    if "v2" in engine._shader_name:
        cp.dispatch_workgroups(n_segs, 1, 1)
    else:
        cp.dispatch_workgroups(n_segs, P, 1)
    cp.end()
    kg_bytes = out_len * 8
    rms_bytes = out_len * 4
    enc.copy_buffer_to_buffer(engine._k_buf,   0, engine._k_rb,   0, kg_bytes)
    enc.copy_buffer_to_buffer(engine._g_buf,   0, engine._g_rb,   0, kg_bytes)
    enc.copy_buffer_to_buffer(engine._rms_buf, 0, engine._rms_rb, 0, rms_bytes)
    engine.device.queue.submit([enc.finish()])
    t_submit = time.perf_counter() - t0

    # phase 3: map_sync forces GPU to complete, this is where GPU work actually finishes
    t0 = time.perf_counter()
    engine._k_rb.map_sync(wgpu.MapMode.READ,   size=kg_bytes)
    engine._g_rb.map_sync(wgpu.MapMode.READ,   size=kg_bytes)
    engine._rms_rb.map_sync(wgpu.MapMode.READ, size=rms_bytes)
    t_sync = time.perf_counter() - t0

    # phase 4: actually read out the bytes
    t0 = time.perf_counter()
    K = np.frombuffer(engine._k_rb.read_mapped(size=kg_bytes), dtype=np.float32).copy()
    G = np.frombuffer(engine._g_rb.read_mapped(size=kg_bytes), dtype=np.float32).copy()
    rms = np.frombuffer(engine._rms_rb.read_mapped(size=rms_bytes), dtype=np.float32).copy()
    engine._k_rb.unmap()
    engine._g_rb.unmap()
    engine._rms_rb.unmap()
    t_read = time.perf_counter() - t0

    return {
        "upload_ms":  t_upload * 1e3,
        "submit_ms":  t_submit * 1e3,
        "gpu_sync_ms": t_sync * 1e3,
        "readback_ms": t_read * 1e3,
        "total_ms":   (t_upload + t_submit + t_sync + t_read) * 1e3,
    }


def run():
    if not gpu_available():
        print("wgpu not available.")
        return

    engine = GpuEngine(shader="fit_pair.wgsl")
    print(f"adapter: {engine.describe()}")
    print()

    DIM = 300

    print(f"{'n_art':>6} {'segs':>6} {'data(MB)':>9} {'upload':>8} {'submit':>8} "
          f"{'gpu_sync':>9} {'readback':>9} {'total':>8} {'eff GB/s':>9}")
    print("-" * 90)

    for n_articles in (32, 128, 256, 512, 1024, 2048):
        segments = synth_batch(n_articles, DIM, seed=42)
        traj, segs = pack_batch(segments)

        # warmup
        _ = timed_phases(engine, traj, segs)
        _ = timed_phases(engine, traj, segs)

        # best of 3
        best = None
        for _ in range(3):
            p = timed_phases(engine, traj, segs)
            if best is None or p["total_ms"] < best["total_ms"]:
                best = p

        data_mb = traj.nbytes / 1e6
        eff_bw = data_mb / best["total_ms"] * 1000 / 1000  # GB/s
        print(f"{n_articles:>6} {len(segments):>6} {data_mb:>8.1f} "
              f"{best['upload_ms']:>7.2f} {best['submit_ms']:>7.2f} "
              f"{best['gpu_sync_ms']:>8.2f} {best['readback_ms']:>8.2f} "
              f"{best['total_ms']:>7.2f} {eff_bw:>8.2f}")


if __name__ == "__main__":
    run()
