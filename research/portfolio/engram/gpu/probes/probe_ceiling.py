"""
probe_ceiling.py — establish the machine's actual compute/bandwidth ceiling.

1) measure raw DRAM throughput with a vec4 memcpy kernel
2) list adapter features (subgroup ops, f16, etc.)
3) test various memcpy dispatch shapes to find the saturation point
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import wgpu
from wgpu.utils.device import get_default_device


def run():
    device = get_default_device()
    adapter = device.adapter
    print("adapter info:")
    info = dict(adapter.info)
    for k, v in info.items():
        print(f"  {k}: {v}")
    print()

    print("adapter.features (FULL):")
    try:
        feats = set(adapter.features)
        for f in sorted(feats):
            print(f"  {f}")
        # explicit checks for what we care about
        print()
        print("  subgroup check:", "subgroups" in feats)
        print("  f16 check:", "shader-f16" in feats)
        print("  f64 check:", "shader-f64" in feats)
        print("  atomic f32 check:", any("atomic" in f and "float" in f for f in feats))
    except Exception as e:
        print(f"  (could not enumerate: {e})")
    print()

    print("device.features:")
    try:
        dfeats = set(device.features)
        for f in sorted(dfeats):
            print(f"  {f}")
    except Exception as e:
        print(f"  (could not enumerate: {e})")
    print()

    print("adapter.limits (selected):")
    try:
        lim = dict(adapter.limits)
        for k in ["max_storage_buffer_binding_size",
                  "max_compute_workgroup_size_x",
                  "max_compute_workgroup_storage_size",
                  "max_compute_invocations_per_workgroup",
                  "max_compute_workgroups_per_dimension",
                  "subgroup_min_size", "subgroup_max_size"]:
            if k in lim:
                print(f"  {k}: {lim[k]}")
    except Exception as e:
        print(f"  (could not: {e})")
    print()

    # ── memcpy bandwidth probe ───────────────────────────────────────
    shader_src = (Path(__file__).resolve().parents[1] / "shaders" / "variants" / "memcpy_probe.wgsl").read_text()
    module = device.create_shader_module(code=shader_src)

    layout = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.storage}},
    ])
    pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[layout]),
        compute={"module": module, "entry_point": "main"},
    )

    print("memcpy probe (pure bandwidth):")
    print(f"{'size_MB':>10} {'elems':>10} {'WGs':>10} {'ms':>8} {'GB/s':>8}")

    for size_mb in [4, 16, 32, 64, 128, 256]:
        n_vec4 = (size_mb * 1024 * 1024) // 16
        n_wgs = (n_vec4 + 63) // 64
        # dispatch dim limit is 65535 per axis; use 2D dispatch to scale
        wg_x = min(n_wgs, 65535)
        wg_y = (n_wgs + wg_x - 1) // wg_x

        src_buf = device.create_buffer(
            size=n_vec4 * 16,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        dst_buf = device.create_buffer(
            size=n_vec4 * 16,
            usage=wgpu.BufferUsage.STORAGE,
        )

        bg = device.create_bind_group(
            layout=layout,
            entries=[
                {"binding": 0, "resource": {"buffer": src_buf, "offset": 0, "size": n_vec4 * 16}},
                {"binding": 1, "resource": {"buffer": dst_buf, "offset": 0, "size": n_vec4 * 16}},
            ],
        )

        # warmup
        for _ in range(3):
            enc = device.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(pipeline)
            cp.set_bind_group(0, bg)
            cp.dispatch_workgroups(wg_x, wg_y, 1)
            cp.end()
            device.queue.submit([enc.finish()])
        device._poll_wait()

        # measured (5 runs, best)
        best = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            enc = device.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(pipeline)
            cp.set_bind_group(0, bg)
            cp.dispatch_workgroups(wg_x, wg_y, 1)
            cp.end()
            device.queue.submit([enc.finish()])
            device._poll_wait()
            dt = time.perf_counter() - t0
            if dt < best:
                best = dt

        # each thread reads 16 B and writes 16 B = 32 B total traffic per thread
        total_bytes = n_vec4 * 16 * 2  # read + write
        gbps = total_bytes / 1e9 / best
        print(f"{size_mb:>9}M {n_vec4:>10} {n_wgs:>10} {best*1000:>7.2f} {gbps:>7.2f}")


if __name__ == "__main__":
    run()
