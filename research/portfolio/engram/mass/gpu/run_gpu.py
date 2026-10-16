"""
run_gpu.py — invoke the WGSL parallel-scan MASS state machine.

builds the transition table, uploads byte input, dispatches init +
scan_local + scan_spine + apply_prefix, reads back the prefix array,
unpacks states. verifies against the Python reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import wgpu
from wgpu.utils.device import get_default_device

from nano_sm import build_transition_table, S_PROSE, N_STATES


WG_SIZE = 256
SHADER = Path(__file__).parent / "shaders" / "nano_scan.wgsl"


def pack_transition_table(T: np.ndarray) -> np.ndarray:
    """
    T: shape (8, 256) uint8. returns uint32[256] where each entry
    is the packed transition function for that byte:
        trans[b] = T[0,b] | T[1,b]<<4 | T[2,b]<<8 | ... | T[7,b]<<28
    """
    result = np.zeros(256, dtype=np.uint32)
    for b in range(256):
        packed = 0
        for s in range(N_STATES):
            packed |= int(T[s, b]) << (s * 4)
        result[b] = packed
    return result


def identity_packed() -> int:
    """packed identity transition: state s → s for all s."""
    v = 0
    for s in range(N_STATES):
        v |= s << (s * 4)
    return v


class NanoScanGPU:
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
        ])
        pl_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.layout])

        self.pipe_init = self.device.create_compute_pipeline(
            layout=pl_layout, compute={"module": module, "entry_point": "init"})
        self.pipe_scan_local = self.device.create_compute_pipeline(
            layout=pl_layout,
            compute={"module": module, "entry_point": "scan_local"})
        self.pipe_scan_spine = self.device.create_compute_pipeline(
            layout=pl_layout,
            compute={"module": module, "entry_point": "scan_spine"})
        self.pipe_apply = self.device.create_compute_pipeline(
            layout=pl_layout,
            compute={"module": module, "entry_point": "apply_prefix"})

        # persistent trans table (built once)
        trans = pack_transition_table(build_transition_table())
        self.trans_buf = self.device.create_buffer(
            size=trans.nbytes,
            usage=(wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
        )
        self.device.queue.write_buffer(self.trans_buf, 0, memoryview(trans))

    def run(self, text: bytes) -> np.ndarray:
        """
        returns states array, length N+1, with state[0] = S_PROSE and
        state[i+1] = scan[i] applied to S_PROSE for i in [0, N).
        """
        n = len(text)
        if n == 0:
            return np.array([S_PROSE], dtype=np.uint8)

        # pad byte input to multiple of 4 (packed as u32)
        bytes_arr = np.frombuffer(text, dtype=np.uint8)
        n_padded = ((n + 3) // 4) * 4
        if n_padded != n:
            padded = np.zeros(n_padded, dtype=np.uint8)
            padded[:n] = bytes_arr
            bytes_arr = padded
        bytes_u32 = bytes_arr.view(np.uint32)

        # scan buffer length = n (exact) — we pad with identity during scan
        n_blocks = (n + WG_SIZE - 1) // WG_SIZE
        scan_len = n_blocks * WG_SIZE  # padded to full blocks

        # Buffers
        bytes_buf = self.device.create_buffer(
            size=bytes_u32.nbytes,
            usage=(wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
        )
        self.device.queue.write_buffer(bytes_buf, 0, memoryview(bytes_u32))

        scan_buf = self.device.create_buffer(
            size=max(scan_len * 4, 256),
            usage=(wgpu.BufferUsage.STORAGE
                   | wgpu.BufferUsage.COPY_SRC
                   | wgpu.BufferUsage.COPY_DST),
        )
        # pre-fill scan_buf with IDENTITY so padded positions are safe
        identity_arr = np.full(scan_len, identity_packed(), dtype=np.uint32)
        self.device.queue.write_buffer(scan_buf, 0, memoryview(identity_arr))

        block_buf = self.device.create_buffer(
            size=max(n_blocks * 4, 256),
            usage=(wgpu.BufferUsage.STORAGE
                   | wgpu.BufferUsage.COPY_SRC
                   | wgpu.BufferUsage.COPY_DST),
        )
        block_init = np.full(n_blocks, identity_packed(), dtype=np.uint32)
        self.device.queue.write_buffer(block_buf, 0, memoryview(block_init))

        scan_rb = self.device.create_buffer(
            size=max(n * 4, 256),
            usage=(wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST),
        )

        # Bind group (shared across all pipeline calls since bindings match)
        bg = self.device.create_bind_group(
            layout=self.layout,
            entries=[
                {"binding": 0, "resource": {"buffer": bytes_buf,
                                             "offset": 0, "size": bytes_u32.nbytes}},
                {"binding": 1, "resource": {"buffer": self.trans_buf,
                                             "offset": 0, "size": 256 * 4}},
                {"binding": 2, "resource": {"buffer": scan_buf,
                                             "offset": 0, "size": scan_len * 4}},
                {"binding": 3, "resource": {"buffer": block_buf,
                                             "offset": 0, "size": n_blocks * 4}},
            ],
        )

        # Only dispatch `n` threads for init (not scan_len) — we only
        # have `n` input bytes; the rest of scan_buf stays IDENTITY.
        enc = self.device.create_command_encoder()

        cp = enc.begin_compute_pass()
        cp.set_bind_group(0, bg)
        # init: dispatch ceil(n / WG_SIZE) workgroups
        cp.set_pipeline(self.pipe_init)
        cp.dispatch_workgroups((n + WG_SIZE - 1) // WG_SIZE, 1, 1)
        # local scan: same
        cp.set_pipeline(self.pipe_scan_local)
        cp.dispatch_workgroups(n_blocks, 1, 1)
        # spine scan: 1 workgroup
        cp.set_pipeline(self.pipe_scan_spine)
        cp.dispatch_workgroups(1, 1, 1)
        # apply prefix: same as local
        cp.set_pipeline(self.pipe_apply)
        cp.dispatch_workgroups(n_blocks, 1, 1)
        cp.end()

        enc.copy_buffer_to_buffer(scan_buf, 0, scan_rb, 0, n * 4)
        self.device.queue.submit([enc.finish()])

        scan_rb.map_sync(wgpu.MapMode.READ, size=n * 4)
        packed = np.frombuffer(scan_rb.read_mapped(size=n * 4),
                                dtype=np.uint32).copy()
        scan_rb.unmap()

        # unpack: state after byte i = (packed[i] >> (S_PROSE*4)) & 0xF
        states = np.empty(n + 1, dtype=np.uint8)
        states[0] = S_PROSE
        states[1:] = (packed >> (S_PROSE * 4)) & 0xF
        return states


if __name__ == "__main__":
    from nano_sm import run_sequential

    gpu = NanoScanGPU()

    tests = [
        b"hello world",
        b"aa {{t}} bb",
        b"aa [[link]] bb",
        b"nested {{outer {{inner}} still outer}} done",
        b"unclosed {{ template no end",
        (b"abc " * 100) + b"{{tmpl}} " + (b"def " * 100),
    ]

    T = build_transition_table()
    for t in tests:
        seq_states, _ = run_sequential(t, T)
        gpu_states = gpu.run(t)
        ok = (seq_states == gpu_states).all()
        print(f"{'OK' if ok else 'FAIL':4} len={len(t):4d}  {t[:40]!r}")
        if not ok:
            diff = np.where(seq_states != gpu_states)[0]
            print(f"     first diff at position {diff[0]}")
            print(f"     seq[:20] = {list(seq_states[:20])}")
            print(f"     gpu[:20] = {list(gpu_states[:20])}")
