"""bench_v10.py — f16 storage with f32 compute.

halves DRAM bandwidth for the trajectory reads. only the input storage
format changes; K, G, rms outputs stay f32.
"""

from __future__ import annotations
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

import numpy as np
import wgpu
from engram_codec import fit_all, _to_complex, MIN_BLOCK, MAX_BLOCK
from engram.gpu.engine import GpuEngine, pack_batch


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
        for j in range(int(rng.integers(3, 8))):
            T = int(rng.integers(MIN_BLOCK + 4, MAX_BLOCK + 1))
            segments.append(synth_segment(T, dim, seed + 1000 * i + j))
    return segments


def f16_fit_batch(device, pipeline, layout, traj_f16, segments, out_len, P):
    """dispatch the v10 kernel with an f16 trajectory buffer."""
    n_segs = int(segments.shape[0])

    traj_buf = device.create_buffer(
        size=max(traj_f16.nbytes, 256),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    seg_buf = device.create_buffer(
        size=max(segments.nbytes, 256),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    k_buf = device.create_buffer(size=out_len * 8,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    g_buf = device.create_buffer(size=out_len * 8,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    rms_buf = device.create_buffer(size=out_len * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    k_rb = device.create_buffer(size=out_len * 8,
        usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)
    g_rb = device.create_buffer(size=out_len * 8,
        usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)
    rms_rb = device.create_buffer(size=out_len * 4,
        usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)

    device.queue.write_buffer(traj_buf, 0, memoryview(traj_f16))
    device.queue.write_buffer(seg_buf, 0, memoryview(segments))

    bg = device.create_bind_group(
        layout=layout,
        entries=[
            {"binding": 0, "resource": {"buffer": traj_buf, "offset": 0, "size": traj_f16.nbytes}},
            {"binding": 1, "resource": {"buffer": seg_buf, "offset": 0, "size": segments.nbytes}},
            {"binding": 2, "resource": {"buffer": k_buf, "offset": 0, "size": out_len * 8}},
            {"binding": 3, "resource": {"buffer": g_buf, "offset": 0, "size": out_len * 8}},
            {"binding": 4, "resource": {"buffer": rms_buf, "offset": 0, "size": out_len * 4}},
        ],
    )
    enc = device.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    WG = 64
    pair_tiles = (P + WG - 1) // WG
    cp.dispatch_workgroups(n_segs, pair_tiles, 1)
    cp.end()
    enc.copy_buffer_to_buffer(k_buf,   0, k_rb,   0, out_len * 8)
    enc.copy_buffer_to_buffer(g_buf,   0, g_rb,   0, out_len * 8)
    enc.copy_buffer_to_buffer(rms_buf, 0, rms_rb, 0, out_len * 4)
    device.queue.submit([enc.finish()])

    k_rb.map_sync(wgpu.MapMode.READ, size=out_len * 8)
    g_rb.map_sync(wgpu.MapMode.READ, size=out_len * 8)
    rms_rb.map_sync(wgpu.MapMode.READ, size=out_len * 4)
    K = np.frombuffer(k_rb.read_mapped(size=out_len * 8), dtype=np.float32).copy().view(np.complex64)
    G = np.frombuffer(g_rb.read_mapped(size=out_len * 8), dtype=np.float32).copy().view(np.complex64)
    rms = np.frombuffer(rms_rb.read_mapped(size=out_len * 4), dtype=np.float32).copy()
    k_rb.unmap(); g_rb.unmap(); rms_rb.unmap()
    return K, G, rms


def run():
    # enable f16
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync(required_features=["shader-f16"])

    wgsl = (Path(__file__).resolve().parents[1] / "shaders" / "variants" / "fit_pair_v10_f16.wgsl").read_text()
    module = device.create_shader_module(code=wgsl)
    layout = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.storage}},
    ])
    pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[layout]),
        compute={"module": module, "entry_point": "main"},
    )
    print(f"adapter: {dict(adapter.info).get('device', '?')} — f16 enabled")
    print()

    eng_v3 = GpuEngine(shader="fit_pair_v3.wgsl")

    print(f"{'n_art':>6} {'segs':>6} {'cpu':>8} {'v3':>8} {'v10':>8} {'v10x':>6}")
    print("-" * 52)

    for n_art in (32, 128, 256, 512):
        segments = synth_batch(n_art, 300, seed=42)
        n_segs = len(segments)
        P = segments[0].shape[1] // 2

        traj_f32, segs = pack_batch(segments)
        # convert f32 traj to f16 in-place (same layout, half memory)
        traj_f16 = traj_f32.astype(np.float16)
        # the offsets in segs were computed for f32 indexing; they still
        # work for f16 indexing because "index of k-th element" doesn't
        # depend on element size when the offset is in element units.
        out_len = int((segs[:, 3] + P).max())

        # cpu reference
        K_cpu = np.empty((n_segs, P), dtype=np.complex128)
        G_cpu = np.empty((n_segs, P), dtype=np.complex128)
        rms_cpu = np.empty((n_segs, P), dtype=np.float32)
        t0 = time.perf_counter()
        for i, s in enumerate(segments):
            z_c = _to_complex(s.astype(np.float32))
            K, G, _, rms = fit_all(z_c)
            K_cpu[i] = K; G_cpu[i] = G; rms_cpu[i] = rms
        t_cpu = time.perf_counter() - t0

        # v3 reference
        _ = eng_v3.fit_batch(traj_f32, segs)
        _ = eng_v3.fit_batch(traj_f32, segs)
        t0 = time.perf_counter()
        K3, G3, rms3 = eng_v3.fit_batch(traj_f32, segs)
        t_v3 = time.perf_counter() - t0

        # v10
        _ = f16_fit_batch(device, pipeline, layout, traj_f16, segs, out_len, P)
        _ = f16_fit_batch(device, pipeline, layout, traj_f16, segs, out_len, P)
        t0 = time.perf_counter()
        K10, G10, rms10 = f16_fit_batch(device, pipeline, layout, traj_f16, segs, out_len, P)
        t_v10 = time.perf_counter() - t0

        K10 = K10.reshape(n_segs, P)
        G10 = G10.reshape(n_segs, P)
        rms10 = rms10.reshape(n_segs, P)

        print(f"{n_art:>6} {n_segs:>6} "
              f"{t_cpu*1e3:>7.1f} {t_v3*1e3:>7.1f} {t_v10*1e3:>7.1f} "
              f"{t_cpu/t_v10:>5.2f}x")

        if n_art == 256:
            def err(a, b):
                denom = np.maximum(np.abs(b), 1e-6)
                return float(np.median(np.abs(a.astype(np.complex128) - b) / denom))
            def errmax(a, b):
                denom = np.maximum(np.abs(b), 1e-6)
                return float(np.max(np.abs(a.astype(np.complex128) - b) / denom))
            print(f"\n  v10 accuracy (vs cpu f64):")
            print(f"    K   med={err(K10, K_cpu):.2e}  max={errmax(K10, K_cpu):.2e}")
            print(f"    G   med={err(G10, G_cpu):.2e}  max={errmax(G10, G_cpu):.2e}")
            print(f"    rms med={err(rms10.astype(np.complex128), rms_cpu.astype(np.complex128)):.2e}  "
                  f"max={errmax(rms10.astype(np.complex128), rms_cpu.astype(np.complex128)):.2e}")
            print()


if __name__ == "__main__":
    run()
