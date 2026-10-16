"""
test_gather_kernel.py — correctness for gather_smooth_pair.wgsl.

builds a synthetic (glove, indices, pairing), runs the kernel, compares
to numpy reference. must match to float32 precision.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import wgpu
from wgpu.utils.device import get_default_device


def numpy_reference(glove, indices_list, pairing, W=10):
    """reference: per article, gather → sliding mean → apply pairing."""
    D = glove.shape[1]
    results = []
    for idxs in indices_list:
        if len(idxs) < W:
            results.append(np.empty((0, D), dtype=np.float32))
            continue
        # gather: [T, D]
        raw = glove[idxs]  # f32[T, D]
        # sliding window mean via cumsum
        T, _ = raw.shape
        cs = np.empty((T + 1, D), dtype=np.float32)
        cs[0] = 0
        np.cumsum(raw, axis=0, out=cs[1:])
        smoothed = (cs[W:] - cs[:-W]) / np.float32(W)  # [T-W+1, D]
        # apply pairing: out[t, d] = smoothed[t, pairing[d]]
        paired = smoothed[:, pairing]
        results.append(paired)
    return results


def pack_batch(indices_list, W):
    """
    flatten into buffers for the GPU.
      indices: i32[N_total] all indices concatenated
      offsets: u32[n+1]     cumulative token offsets
      out_off: u32[n+1]     cumulative smoothed-timestep offsets
    """
    offsets = [0]
    out_off = [0]
    all_idx = []
    for idxs in indices_list:
        all_idx.extend(idxs)
        offsets.append(offsets[-1] + len(idxs))
        n_smooth = max(0, len(idxs) - W + 1)
        out_off.append(out_off[-1] + n_smooth)
    return (np.asarray(all_idx, dtype=np.int32),
            np.asarray(offsets, dtype=np.uint32),
            np.asarray(out_off, dtype=np.uint32))


def run_gpu(glove, indices_list, pairing, W=10):
    device = get_default_device()

    # flatten glove
    glove_flat = np.ascontiguousarray(glove, dtype=np.float32).ravel()
    V, D = glove.shape
    n_articles = len(indices_list)

    all_idx, offsets, out_off = pack_batch(indices_list, W)
    total_out = int(out_off[-1])
    if total_out == 0:
        return [np.empty((0, D), dtype=np.float32) for _ in indices_list]

    # out_pair is flat f32[total_out * D]
    out_buf_size = total_out * D * 4

    # --- buffers ---
    glove_buf = device.create_buffer(
        size=max(glove_flat.nbytes, 256),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    device.queue.write_buffer(glove_buf, 0, memoryview(glove_flat))

    idx_buf = device.create_buffer(
        size=max(all_idx.nbytes, 256),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    device.queue.write_buffer(idx_buf, 0, memoryview(all_idx))

    off_buf = device.create_buffer(
        size=max(offsets.nbytes, 256),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    device.queue.write_buffer(off_buf, 0, memoryview(offsets))

    pair_buf = device.create_buffer(
        size=max(pairing.nbytes, 256),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    pairing32 = np.asarray(pairing, dtype=np.int32)
    device.queue.write_buffer(pair_buf, 0, memoryview(pairing32))

    out_off_buf = device.create_buffer(
        size=max(out_off.nbytes, 256),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    device.queue.write_buffer(out_off_buf, 0, memoryview(out_off))

    out_buf = device.create_buffer(
        size=max(out_buf_size, 256),
        usage=(wgpu.BufferUsage.STORAGE
               | wgpu.BufferUsage.COPY_SRC
               | wgpu.BufferUsage.COPY_DST),
    )
    # pre-zero in case some cells are never written (shouldn't happen)
    zeros = np.zeros(total_out * D, dtype=np.float32)
    device.queue.write_buffer(out_buf, 0, memoryview(zeros))

    uniform_buf = device.create_buffer(
        size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )
    params = np.array([W, D, n_articles, 0], dtype=np.uint32)
    device.queue.write_buffer(uniform_buf, 0, memoryview(params))

    rb = device.create_buffer(
        size=max(out_buf_size, 256),
        usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
    )

    # --- shader + pipeline ---
    src = (Path(__file__).resolve().parent / "shaders" /
           "gather_smooth_pair.wgsl").read_text()
    module = device.create_shader_module(code=src)

    layout = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 5, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 6, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.uniform}},
    ])
    pl_layout = device.create_pipeline_layout(bind_group_layouts=[layout])
    pipe = device.create_compute_pipeline(
        layout=pl_layout,
        compute={"module": module, "entry_point": "main"},
    )

    bg = device.create_bind_group(
        layout=layout,
        entries=[
            {"binding": 0, "resource": {"buffer": glove_buf, "offset": 0, "size": glove_flat.nbytes}},
            {"binding": 1, "resource": {"buffer": idx_buf,   "offset": 0, "size": max(all_idx.nbytes, 256)}},
            {"binding": 2, "resource": {"buffer": off_buf,   "offset": 0, "size": max(offsets.nbytes, 256)}},
            {"binding": 3, "resource": {"buffer": pair_buf,  "offset": 0, "size": max(pairing32.nbytes, 256)}},
            {"binding": 4, "resource": {"buffer": out_off_buf, "offset": 0, "size": max(out_off.nbytes, 256)}},
            {"binding": 5, "resource": {"buffer": out_buf,   "offset": 0, "size": max(out_buf_size, 256)}},
            {"binding": 6, "resource": {"buffer": uniform_buf, "offset": 0, "size": 16}},
        ],
    )

    enc = device.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipe)
    cp.set_bind_group(0, bg)
    # dispatch: (ceil(total_out / 64), ceil(D / 4), 1)
    wg_x = (total_out + 63) // 64
    wg_y = (D + 3) // 4
    cp.dispatch_workgroups(wg_x, wg_y, 1)
    cp.end()
    enc.copy_buffer_to_buffer(out_buf, 0, rb, 0, out_buf_size)
    device.queue.submit([enc.finish()])

    rb.map_sync(wgpu.MapMode.READ, size=out_buf_size)
    data = np.frombuffer(rb.read_mapped(size=out_buf_size),
                          dtype=np.float32).copy()
    rb.unmap()

    # split back into per-article arrays
    flat = data.reshape(total_out, D)
    results = []
    for a in range(n_articles):
        s = int(out_off[a]); e = int(out_off[a + 1])
        results.append(flat[s:e].copy())
    return results


def run():
    # synthetic
    V = 1000
    D = 300
    W = 10
    rng = np.random.default_rng(42)
    glove = (rng.standard_normal((V, D)) * 0.3).astype(np.float32)
    pairing = rng.permutation(D).astype(np.int32)

    # 5 articles of varying length
    articles = []
    for seed in range(5):
        r = np.random.default_rng(seed + 100)
        n = int(r.integers(50, 400))
        idxs = r.integers(0, V, size=n).astype(np.int32)
        articles.append(idxs)

    ref = numpy_reference(glove, articles, pairing, W=W)
    gpu = run_gpu(glove, articles, pairing, W=W)

    print(f"{'article':<8s} {'ref shape':<12s} {'gpu shape':<12s} {'max abs diff':>14s}")
    print("-" * 55)
    for i, (r, g) in enumerate(zip(ref, gpu)):
        if r.shape != g.shape:
            print(f"{i:<8d} {str(r.shape):<12s} {str(g.shape):<12s} SHAPE MISMATCH")
            continue
        diff = float(np.abs(r - g).max()) if r.size else 0.0
        print(f"{i:<8d} {str(r.shape):<12s} {str(g.shape):<12s} {diff:>14.2e}")

    all_match = all(r.shape == g.shape and
                    (r.size == 0 or np.abs(r - g).max() < 1e-4)
                    for r, g in zip(ref, gpu))
    print()
    print(f"overall: {'OK' if all_match else 'FAIL'}")


if __name__ == "__main__":
    run()
