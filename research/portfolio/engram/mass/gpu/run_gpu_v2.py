"""
run_gpu_v2.py — global Hillis-Steele over the entire scan array.

drops the workgroup-local + spine structure in favor of log2(N)
doubling-stride passes. simpler, scales to any N, O(N log N) work.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import wgpu
from wgpu.utils.device import get_default_device

from nano_sm import build_transition_table, S_PROSE, N_STATES
from run_gpu import pack_transition_table, identity_packed


WG_SIZE = 256
SHADER = Path(__file__).parent / "shaders" / "nano_scan_v2.wgsl"


class NanoScanGPUv2:
    def __init__(self):
        self.device = get_default_device()
        src = SHADER.read_text()
        module = self.device.create_shader_module(code=src)

        self.layout = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ])
        pl_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.layout])

        self.pipe_init = self.device.create_compute_pipeline(
            layout=pl_layout, compute={"module": module, "entry_point": "init"})
        self.pipe_step = self.device.create_compute_pipeline(
            layout=pl_layout, compute={"module": module, "entry_point": "scan_step"})

        trans = pack_transition_table(build_transition_table())
        self.trans_buf = self.device.create_buffer(
            size=trans.nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.device.queue.write_buffer(self.trans_buf, 0, memoryview(trans))

    def run(self, text: bytes) -> np.ndarray:
        n = len(text)
        if n == 0:
            return np.array([S_PROSE], dtype=np.uint8)

        # pad bytes to multiple of 4
        bytes_arr = np.frombuffer(text, dtype=np.uint8)
        n_padded = ((n + 3) // 4) * 4
        if n_padded != n:
            padded = np.zeros(n_padded, dtype=np.uint8)
            padded[:n] = bytes_arr
            bytes_arr = padded
        bytes_u32 = bytes_arr.view(np.uint32)

        bytes_buf = self.device.create_buffer(
            size=bytes_u32.nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.device.queue.write_buffer(bytes_buf, 0, memoryview(bytes_u32))

        # two scan buffers for ping-pong
        scan_a = self.device.create_buffer(
            size=max(n * 4, 256),
            usage=(wgpu.BufferUsage.STORAGE
                   | wgpu.BufferUsage.COPY_SRC
                   | wgpu.BufferUsage.COPY_DST),
        )
        scan_b = self.device.create_buffer(
            size=max(n * 4, 256),
            usage=(wgpu.BufferUsage.STORAGE
                   | wgpu.BufferUsage.COPY_SRC
                   | wgpu.BufferUsage.COPY_DST),
        )
        rb = self.device.create_buffer(
            size=max(n * 4, 256),
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
        )

        # uniform for params (d, n, pad, pad) — minimum size 16 B
        uniform_buf = self.device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        wg_count = (n + WG_SIZE - 1) // WG_SIZE
        # 2D dispatch if we exceed 65535 per-dim
        def dispatch_shape(wgc):
            if wgc <= 65535:
                return (wgc, 1, 1)
            # 2D: (65535, ceil(wgc/65535))
            import math
            y = (wgc + 65534) // 65535
            return (min(65535, wgc), y, 1)

        dshape = dispatch_shape(wg_count)

        def make_bg(a_buf, b_buf):
            return self.device.create_bind_group(
                layout=self.layout,
                entries=[
                    {"binding": 0, "resource": {"buffer": self.trans_buf,
                                                 "offset": 0, "size": 256 * 4}},
                    {"binding": 1, "resource": {"buffer": bytes_buf,
                                                 "offset": 0, "size": bytes_u32.nbytes}},
                    {"binding": 2, "resource": {"buffer": a_buf,
                                                 "offset": 0, "size": n * 4}},
                    {"binding": 3, "resource": {"buffer": b_buf,
                                                 "offset": 0, "size": n * 4}},
                    {"binding": 4, "resource": {"buffer": uniform_buf,
                                                 "offset": 0, "size": 16}},
                ],
            )

        # --- init: write initial fns into scan_a ---
        # bind a TINY dummy to the unused in_buf slot (wgpu forbids
        # binding the same buffer as both read-only-storage and
        # storage-read-write in one dispatch).
        dummy = self.device.create_buffer(
            size=max(n * 4, 256),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        params = np.array([0, n, 0, 0], dtype=np.uint32)
        self.device.queue.write_buffer(uniform_buf, 0, memoryview(params))
        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self.pipe_init)
        cp.set_bind_group(0, make_bg(dummy, scan_a))  # dummy=in, scan_a=out
        cp.dispatch_workgroups(*dshape)
        cp.end()
        self.device.queue.submit([enc.finish()])

        # --- log2(n) scan steps, ping-pong ---
        d = 1
        src = scan_a; dst = scan_b
        while d < n:
            params = np.array([d, n, 0, 0], dtype=np.uint32)
            self.device.queue.write_buffer(uniform_buf, 0, memoryview(params))
            enc = self.device.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(self.pipe_step)
            cp.set_bind_group(0, make_bg(src, dst))
            cp.dispatch_workgroups(*dshape)
            cp.end()
            self.device.queue.submit([enc.finish()])
            src, dst = dst, src
            d *= 2

        # src now has the final scan (after the last swap)
        enc = self.device.create_command_encoder()
        enc.copy_buffer_to_buffer(src, 0, rb, 0, n * 4)
        self.device.queue.submit([enc.finish()])

        rb.map_sync(wgpu.MapMode.READ, size=n * 4)
        packed = np.frombuffer(rb.read_mapped(size=n * 4),
                                dtype=np.uint32).copy()
        rb.unmap()

        states = np.empty(n + 1, dtype=np.uint8)
        states[0] = S_PROSE
        states[1:] = (packed >> (S_PROSE * 4)) & 0xF
        return states


if __name__ == "__main__":
    from nano_sm import run_sequential

    gpu = NanoScanGPUv2()
    T = build_transition_table()

    import random
    rng = random.Random(7)
    for n in (100, 1000, 10000, 100000, 500000, 1000000):
        parts = []
        while sum(len(p) for p in parts) < n:
            c = rng.random()
            if c < 0.5:
                parts.append("some words and more words ")
            elif c < 0.75:
                parts.append("{{template arg=val}}")
            else:
                parts.append("[[link text]]")
        text = "".join(parts)[:n].encode()
        seq, _ = run_sequential(text, T)
        gpu_s = gpu.run(text)
        ok = (seq == gpu_s).all()
        if not ok:
            diff = np.where(seq != gpu_s)[0][:3]
            print(f"n={n:<8d} FAIL at {list(diff)}")
        else:
            print(f"n={n:<8d} OK")
