"""
run_gpu_v3.py — recursive local+spine scan driver.

scheme: allocate one buffer per pyramid level. climb the pyramid doing
scan_local at each (which writes block_totals = next level's input),
then descend doing apply_prefix.

for input N = 3.6 MB at WG_SIZE = 256:
  level 0: N          = 3.6 M
  level 1: N/256      = 14 K
  level 2: N/256/256  = 55
  level 3: 1          (root; nothing to do)

dispatch count: 3 scan_local + 2 apply_prefix + 1 init = 6. memory
traffic: 1 read + 1 write of each level = O(N) total.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import wgpu
from wgpu.utils.device import get_default_device

from nano_sm import build_transition_table, S_PROSE
from run_gpu import pack_transition_table, identity_packed


WG_SIZE = 256
SHADER = Path(__file__).parent / "shaders" / "nano_scan_v3.wgsl"


def _dispatch_shape(wgc):
    if wgc <= 65535:
        return (wgc, 1, 1)
    y = (wgc + 65534) // 65535
    return (65535, y, 1)


class NanoScanGPUv3:
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
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ])
        pl_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.layout])

        self.pipe_init = self.device.create_compute_pipeline(
            layout=pl_layout, compute={"module": module, "entry_point": "init"})
        self.pipe_scan = self.device.create_compute_pipeline(
            layout=pl_layout,
            compute={"module": module, "entry_point": "scan_local"})
        self.pipe_apply = self.device.create_compute_pipeline(
            layout=pl_layout,
            compute={"module": module, "entry_point": "apply_prefix"})

        trans = pack_transition_table(build_transition_table())
        self.trans_buf = self.device.create_buffer(
            size=trans.nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.device.queue.write_buffer(self.trans_buf, 0, memoryview(trans))

        self._dummy = None

    def _dummy_buf(self, size):
        """a tiny scratch bound to unused slots."""
        if self._dummy is None or self._dummy.size < size:
            self._dummy = self.device.create_buffer(
                size=max(size, 256),
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
        return self._dummy

    def _bind(self, bytes_buf, scan_buf, bt_buf, uniform_buf,
              scan_size, bt_size, bytes_size):
        return self.device.create_bind_group(
            layout=self.layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self.trans_buf,
                                             "offset": 0, "size": 256 * 4}},
                {"binding": 1, "resource": {"buffer": bytes_buf,
                                             "offset": 0, "size": bytes_size}},
                {"binding": 2, "resource": {"buffer": scan_buf,
                                             "offset": 0, "size": scan_size}},
                {"binding": 3, "resource": {"buffer": bt_buf,
                                             "offset": 0, "size": bt_size}},
                {"binding": 4, "resource": {"buffer": uniform_buf,
                                             "offset": 0, "size": 16}},
            ],
        )

    def _set_params(self, uniform_buf, n):
        params = np.array([n, 0, 0, 0], dtype=np.uint32)
        self.device.queue.write_buffer(uniform_buf, 0, memoryview(params))

    def run(self, text: bytes) -> np.ndarray:
        n = len(text)
        if n == 0:
            return np.array([S_PROSE], dtype=np.uint8)

        # bytes
        bytes_arr = np.frombuffer(text, dtype=np.uint8)
        n_pad = ((n + 3) // 4) * 4
        if n_pad != n:
            p = np.zeros(n_pad, dtype=np.uint8)
            p[:n] = bytes_arr
            bytes_arr = p
        bytes_u32 = bytes_arr.view(np.uint32)
        bytes_buf = self.device.create_buffer(
            size=bytes_u32.nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.device.queue.write_buffer(bytes_buf, 0, memoryview(bytes_u32))

        uniform_buf = self.device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # build pyramid sizes
        sizes = [n]
        while sizes[-1] > 1:
            sizes.append((sizes[-1] + WG_SIZE - 1) // WG_SIZE)
        # level_bufs[L] holds the scan at level L (size sizes[L] u32s).
        # we need one more than levels so the TOP level (size 1) has a
        # place to write its "block total" even though we won't use it.
        level_bufs = []
        for L, sz in enumerate(sizes):
            sz_eff = max(sz, 1)
            b = self.device.create_buffer(
                size=max(sz_eff * 4, 256),
                usage=(wgpu.BufferUsage.STORAGE
                       | wgpu.BufferUsage.COPY_SRC
                       | wgpu.BufferUsage.COPY_DST),
            )
            idents = np.full(sz_eff, identity_packed(), dtype=np.uint32)
            self.device.queue.write_buffer(b, 0, memoryview(idents))
            level_bufs.append(b)
        # need one extra slot for the BT of the top scan_local call
        # (even if only 1 element). append an extra dummy-sized buf.
        overflow_bt = self.device.create_buffer(
            size=256,
            usage=(wgpu.BufferUsage.STORAGE
                   | wgpu.BufferUsage.COPY_SRC
                   | wgpu.BufferUsage.COPY_DST),
        )
        self.device.queue.write_buffer(
            overflow_bt, 0,
            memoryview(np.full(64, identity_packed(), dtype=np.uint32)))
        level_bufs.append(overflow_bt)

        # readback buffer for final result
        rb = self.device.create_buffer(
            size=max(n * 4, 256),
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
        )

        # ── pass 1: init bytes → level 0 ──
        self._set_params(uniform_buf, n)
        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self.pipe_init)
        cp.set_bind_group(0, self._bind(
            bytes_buf, level_bufs[0], level_bufs[1],
            uniform_buf,
            scan_size=max(n * 4, 256),
            bt_size=max(sizes[1] * 4, 256) if len(sizes) > 1 else 256,
            bytes_size=bytes_u32.nbytes,
        ))
        cp.dispatch_workgroups(*_dispatch_shape((n + WG_SIZE - 1) // WG_SIZE))
        cp.end()

        # ── climb pyramid: scan_local at levels 0..n_levels-1 ──
        n_levels = len(sizes) - 1  # top level has size 1 — nothing to do
        for L in range(n_levels):
            n_this = sizes[L]
            self._set_params(uniform_buf, n_this)
            cp = enc.begin_compute_pass()
            cp.set_pipeline(self.pipe_scan)
            cp.set_bind_group(0, self._bind(
                bytes_buf, level_bufs[L], level_bufs[L + 1],
                uniform_buf,
                scan_size=max(sizes[L] * 4, 256),
                bt_size=max(sizes[L + 1] * 4, 256),
                bytes_size=bytes_u32.nbytes,
            ))
            cp.dispatch_workgroups(*_dispatch_shape(
                (n_this + WG_SIZE - 1) // WG_SIZE))
            cp.end()

        # ── descend pyramid: apply_prefix at levels n_levels-2 down to 0 ──
        # at level L, apply level_bufs[L+1] (now globally scanned) as the
        # block-prefix to level_bufs[L].
        for L in range(n_levels - 2, -1, -1):
            n_this = sizes[L]
            self._set_params(uniform_buf, n_this)
            cp = enc.begin_compute_pass()
            cp.set_pipeline(self.pipe_apply)
            cp.set_bind_group(0, self._bind(
                bytes_buf, level_bufs[L], level_bufs[L + 1],
                uniform_buf,
                scan_size=max(sizes[L] * 4, 256),
                bt_size=max(sizes[L + 1] * 4, 256),
                bytes_size=bytes_u32.nbytes,
            ))
            cp.dispatch_workgroups(*_dispatch_shape(
                (n_this + WG_SIZE - 1) // WG_SIZE))
            cp.end()

        # copy final to readback
        enc.copy_buffer_to_buffer(level_bufs[0], 0, rb, 0, n * 4)
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

    gpu = NanoScanGPUv3()
    T = build_transition_table()

    import random
    rng = random.Random(7)
    for n in (100, 1000, 10000, 100000, 500000, 1_000_000, 3_000_000):
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
        if n <= 200_000:  # sequential is too slow for > 200K
            seq, _ = run_sequential(text, T)
            gpu_s = gpu.run(text)
            ok = (seq == gpu_s).all()
            if not ok:
                diff = np.where(seq != gpu_s)[0][:3]
                print(f"n={n:<8d} FAIL at {list(diff)}")
            else:
                print(f"n={n:<8d} OK (vs sequential)")
        else:
            gpu_s = gpu.run(text)
            print(f"n={n:<8d} gpu ran, output[:8]={list(gpu_s[:8])}")
