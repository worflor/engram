"""
GpuEngine: wgpu-backed substrate for engram codec kernels.

owns the wgpu device, compiled pipelines, and a reusable buffer pool.
callers pass numpy arrays in and get numpy arrays out; internally, all
work happens in one dispatch per batch, with readback only at the end.

the engine is designed around one principle: pay kernel-launch overhead
once per batch, not once per segment. the storage buffers are sized to a
max-batch envelope and reused across calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List
import struct

import numpy as np

try:
    import wgpu
    from wgpu.utils.device import get_default_device
    _HAS_WGPU = True
except Exception:
    _HAS_WGPU = False


def _request_device_with_features(features):
    """try to acquire a device with the given wgpu features. falls back
    to a featureless device if the adapter doesn't expose them."""
    if not _HAS_WGPU:
        return None
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        available = set(adapter.features)
        requested = [f for f in features if f in available]
        return adapter.request_device_sync(required_features=requested)
    except Exception:
        return get_default_device()


# optional features that enhance profiling / performance. these are
# all currently available on AMD iGPU/Vulkan but every one is
# safely fallback-checked at use site.
DEFAULT_FEATURES = (
    "shader-f64",
    "shader-f16",
    "timestamp-query",
    "timestamp-query-inside-encoders",
    "subgroup",
)


SHADER_DIR = Path(__file__).parent / "shaders"
SHADER_VARIANT_DIR = SHADER_DIR / "variants"
WG_SIZE = 64
SEGMENT_STRUCT_SIZE = 16  # 4 × u32


def _resolve_shader(name: str) -> Path:
    """look up a WGSL file in shaders/ first, then shaders/variants/.
    production kernels live at the top level; exploratory variants
    (v1/v2/v4/v5/v8/v9/v10/v11, memcpy_probe, wg128) are archived
    under variants/ so they stay available for reproducing the
    findings without cluttering the production path."""
    primary = SHADER_DIR / name
    if primary.exists():
        return primary
    fallback = SHADER_VARIANT_DIR / name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"shader '{name}' not found in {SHADER_DIR} or {SHADER_VARIANT_DIR}"
    )


def gpu_available() -> bool:
    """true if wgpu-py is importable and an adapter can be acquired."""
    if not _HAS_WGPU:
        return False
    try:
        get_default_device()
        return True
    except Exception:
        return False


class GpuEngine:
    """
    wgpu compute substrate. one instance per process.

    usage:
        eng = GpuEngine()
        K, G, rms = eng.fit_batch(traj_flat, segments)

    where:
        traj_flat: float32 [N_floats], packed paired trajectories
                   (pair k of timestep t lives at index
                    seg_offset + t * D + 2k, seg_offset + t * D + 2k + 1)
        segments:  list of (traj_offset, length, dim, out_offset) tuples.
                   out_offset is the base index into K/G/rms for this
                   segment's P = dim/2 pairs.

    returns:
        K, G   : complex64 [out_len]
        rms    : float32   [out_len]
    """

    def __init__(self, power_preference: str = "high-performance",
                 shader: str = "fit_pair_v3.wgsl",
                 want_features=DEFAULT_FEATURES):
        if not _HAS_WGPU:
            raise RuntimeError("wgpu-py not installed. `pip install wgpu`")
        dev = _request_device_with_features(want_features)
        self.device = dev if dev is not None else get_default_device()
        self._info = dict(self.device.adapter.info)
        self._features = set(self.device.features)
        self._shader_name = shader

        # cached storage buffers, grow on demand
        self._traj_buf: Optional[wgpu.GPUBuffer] = None
        self._traj_cap: int = 0
        self._seg_buf: Optional[wgpu.GPUBuffer] = None
        self._seg_cap: int = 0
        self._k_buf: Optional[wgpu.GPUBuffer] = None
        self._g_buf: Optional[wgpu.GPUBuffer] = None
        self._rms_buf: Optional[wgpu.GPUBuffer] = None
        self._out_cap: int = 0  # in pair slots (not bytes)

        # readback buffers (separate because they need MAP_READ usage)
        self._k_rb: Optional[wgpu.GPUBuffer] = None
        self._g_rb: Optional[wgpu.GPUBuffer] = None
        self._rms_rb: Optional[wgpu.GPUBuffer] = None

        self._pipeline: Optional[wgpu.GPUComputePipeline] = None
        self._bind_layout: Optional[wgpu.GPUBindGroupLayout] = None
        self._build_fit_pipeline()

    # ── introspection ──────────────────────────────────────────────────

    @property
    def adapter_info(self) -> dict:
        return self._info

    def describe(self) -> str:
        info = self._info
        return (f"{info.get('device', '?')} "
                f"[{info.get('adapter_type', '?')}/{info.get('backend_type', '?')}] "
                f"driver={info.get('description', '?')}")

    # ── pipeline construction ──────────────────────────────────────────

    def _build_fit_pipeline(self) -> None:
        wgsl = _resolve_shader(self._shader_name).read_text()
        module = self.device.create_shader_module(code=wgsl)

        # explicit bind group layout so we can pre-allocate buffer pool
        self._bind_layout = self.device.create_bind_group_layout(entries=[
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
        layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._bind_layout])

        self._pipeline = self.device.create_compute_pipeline(
            layout=layout,
            compute={"module": module, "entry_point": "main"},
        )

    # ── buffer pool ────────────────────────────────────────────────────

    def _ensure_traj(self, n_floats: int) -> None:
        need = max(n_floats * 4, 256)
        if self._traj_buf is None or self._traj_cap < need:
            self._traj_cap = max(need, self._traj_cap * 2)
            self._traj_buf = self.device.create_buffer(
                size=self._traj_cap,
                usage=(wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
            )

    def _ensure_segs(self, n_segs: int) -> None:
        need = max(n_segs * SEGMENT_STRUCT_SIZE, 256)
        if self._seg_buf is None or self._seg_cap < need:
            self._seg_cap = max(need, self._seg_cap * 2)
            self._seg_buf = self.device.create_buffer(
                size=self._seg_cap,
                usage=(wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
            )

    def _ensure_outputs(self, out_len: int) -> None:
        """grow output + readback buffer pool to hold `out_len` pair slots.
        classic staging: kernel writes to DEVICE_LOCAL storage, encoder
        copies to MAP_READ staging, host maps. on AMD iGPU the
        direct-mapped STORAGE+MAP_READ variant was benchmarked as
        slightly slower (write-combining / uncached path in the driver)
        so we use staging uniformly.
        """
        if self._out_cap >= out_len and self._k_buf is not None:
            return
        new_cap = max(out_len, self._out_cap * 2, 256)
        self._out_cap = new_cap
        kg_bytes = new_cap * 8
        rms_bytes = new_cap * 4

        usage_out = (wgpu.BufferUsage.STORAGE
                     | wgpu.BufferUsage.COPY_SRC
                     | wgpu.BufferUsage.COPY_DST)
        usage_rb = wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
        self._k_buf   = self.device.create_buffer(size=kg_bytes,  usage=usage_out)
        self._g_buf   = self.device.create_buffer(size=kg_bytes,  usage=usage_out)
        self._rms_buf = self.device.create_buffer(size=rms_bytes, usage=usage_out)
        self._k_rb   = self.device.create_buffer(size=kg_bytes,  usage=usage_rb)
        self._g_rb   = self.device.create_buffer(size=kg_bytes,  usage=usage_rb)
        self._rms_rb = self.device.create_buffer(size=rms_bytes, usage=usage_rb)

    # ── the hot call ──────────────────────────────────────────────────

    # ── split submit/wait variants for pipeline overlap ─────────────
    #
    # one engine instance uses a single set of K/G/rms readback buffers,
    # so only ONE fit can be in flight at a time per GpuEngine. callers
    # that want to overlap two batches run two separate GpuEngine
    # instances, one per batch-in-flight.

    def submit_fit_on_buffer(
        self,
        traj_buf: "wgpu.GPUBuffer",
        traj_buf_size: int,
        segments: np.ndarray,
    ) -> dict:
        """dispatch fit + queue readback copies. returns a context dict
        that wait_fit() consumes. no map_sync here — GPU work is async."""
        if segments.dtype != np.uint32 or not segments.flags.c_contiguous:
            segments = np.ascontiguousarray(segments, dtype=np.uint32)

        n_segs = int(segments.shape[0])
        if n_segs == 0:
            return {
                "empty": True,
                "out_len": 0,
            }

        dims = segments[:, 2]
        D = int(dims[0])
        if not np.all(dims == D):
            raise ValueError("all segments in a batch must share dim")
        P = D // 2
        if P == 0:
            raise ValueError("dim must be >= 2")

        out_len = int((segments[:, 3] + P).max())
        max_end_floats = int((segments[:, 0] + segments[:, 1] * D).max())
        if max_end_floats * 4 > traj_buf_size:
            raise ValueError(
                f"segment reads past traj_buf: end={max_end_floats*4} B, "
                f"traj_buf_size={traj_buf_size} B")

        self._ensure_segs(n_segs)
        self._ensure_outputs(out_len)
        self.device.queue.write_buffer(self._seg_buf, 0, memoryview(segments))

        bg = self.device.create_bind_group(
            layout=self._bind_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": traj_buf,
                                             "offset": 0, "size": traj_buf_size}},
                {"binding": 1, "resource": {"buffer": self._seg_buf,
                                             "offset": 0, "size": self._seg_cap}},
                {"binding": 2, "resource": {"buffer": self._k_buf,
                                             "offset": 0, "size": self._out_cap * 8}},
                {"binding": 3, "resource": {"buffer": self._g_buf,
                                             "offset": 0, "size": self._out_cap * 8}},
                {"binding": 4, "resource": {"buffer": self._rms_buf,
                                             "offset": 0, "size": self._out_cap * 4}},
            ],
        )

        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self._pipeline)
        cp.set_bind_group(0, bg)
        shader = self._shader_name
        if ("v3" in shader or "v4" in shader or "v5" in shader
                or "v8" in shader or "v9" in shader or "v11" in shader):
            WG = 64
            pair_tiles = (P + WG - 1) // WG
            cp.dispatch_workgroups(n_segs, pair_tiles, 1)
        elif "v2" in shader:
            cp.dispatch_workgroups(n_segs, 1, 1)
        else:
            cp.dispatch_workgroups(n_segs, P, 1)
        cp.end()

        kg_bytes = out_len * 8
        rms_bytes = out_len * 4
        enc.copy_buffer_to_buffer(self._k_buf,   0, self._k_rb,   0, kg_bytes)
        enc.copy_buffer_to_buffer(self._g_buf,   0, self._g_rb,   0, kg_bytes)
        enc.copy_buffer_to_buffer(self._rms_buf, 0, self._rms_rb, 0, rms_bytes)
        self.device.queue.submit([enc.finish()])

        return {
            "empty": False,
            "out_len": out_len,
            "kg_bytes": kg_bytes,
            "rms_bytes": rms_bytes,
            "P": P,
        }

    def wait_fit(self, ctx: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """block on map_sync, return (K, G, rms). must be called with
        the dict from submit_fit_on_buffer on the same engine.

        POLLER TRICK: the first buffer uses `map_async` + sync_wait —
        this waits on THIS buffer only, routing completion through the
        wgpu-py background poller thread (5-10 % faster + tighter
        variance than map_sync's internal spin). subsequent map_syncs
        are then instant since all work is already drained. crucially,
        using map_async rather than `_poll_wait()` means we don't stall
        on the prefetch worker's next-batch submits that may already be
        in flight.
        """
        if ctx.get("empty"):
            return (np.empty(0, dtype=np.complex64),
                    np.empty(0, dtype=np.complex64),
                    np.empty(0, dtype=np.float32))
        kg_bytes = ctx["kg_bytes"]
        rms_bytes = ctx["rms_bytes"]

        self._k_rb.map_async(wgpu.MapMode.READ, size=kg_bytes).sync_wait()
        self._g_rb.map_sync(wgpu.MapMode.READ,   size=kg_bytes)
        self._rms_rb.map_sync(wgpu.MapMode.READ, size=rms_bytes)

        K = np.frombuffer(self._k_rb.read_mapped(size=kg_bytes), dtype=np.float32).copy()
        G = np.frombuffer(self._g_rb.read_mapped(size=kg_bytes), dtype=np.float32).copy()
        rms = np.frombuffer(self._rms_rb.read_mapped(size=rms_bytes), dtype=np.float32).copy()

        self._k_rb.unmap()
        self._g_rb.unmap()
        self._rms_rb.unmap()

        return K.view(np.complex64).copy(), G.view(np.complex64).copy(), rms


# ── batch builder helper ──────────────────────────────────────────────

def pack_batch(
    paired_segments: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    pack a list of per-segment paired trajectories [T_i, D] into a single
    flat float32 buffer plus a uint32 [n_segs, 4] metadata table.

    all segments must share D. returns (traj_flat, segments).
    """
    if not paired_segments:
        return (np.empty(0, dtype=np.float32),
                np.empty((0, 4), dtype=np.uint32))

    D = int(paired_segments[0].shape[1])
    P = D // 2

    # total size
    total_floats = sum(int(s.shape[0]) * D for s in paired_segments)
    traj = np.empty(total_floats, dtype=np.float32)
    segs = np.empty((len(paired_segments), 4), dtype=np.uint32)

    f_off = 0
    p_off = 0
    for i, s in enumerate(paired_segments):
        if s.shape[1] != D:
            raise ValueError(f"segment {i} has dim {s.shape[1]}, expected {D}")
        T = int(s.shape[0])
        flat = np.ascontiguousarray(s, dtype=np.float32).ravel()
        traj[f_off:f_off + flat.size] = flat
        segs[i, 0] = f_off
        segs[i, 1] = T
        segs[i, 2] = D
        segs[i, 3] = p_off
        f_off += flat.size
        p_off += P

    return traj, segs
