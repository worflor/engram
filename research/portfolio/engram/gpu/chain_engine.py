"""
chain_engine.py — the GPU chain: gather_smooth_pair → fit_pair.

wraps GpuEngine's device and adds:
  - persistent glove matrix + pairing buffers (uploaded once)
  - gather_paired_batch(indices_list) → list of paired np arrays

the pipeline then:
  1. CPU emits int32 indices per article (just MASS, nothing else)
  2. ChainEngine.gather_paired_batch(indices_list) returns paired
     trajectories as np arrays — fast because upload is tiny
     (just indices) and the huge matrix lives on GPU
  3. CPU segments each trajectory with derive_block_sizes + segment
  4. GpuEngine.fit_batch(traj_flat, seg_table) runs as before

the heavy CPU work (matrix[idxs] on 480 MB matrix + cumsum + pairing)
vanishes from the worker threads — those ops happen on GPU where
they parallelize naturally.

safety: persistent buffers sized ONCE to max corpus vocab; per-batch
buffers pooled and capped at 512 MB like the rest of the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import wgpu
    _HAS_WGPU = True
except Exception:
    _HAS_WGPU = False

from engram.gpu.engine import (
    GpuEngine, gpu_available, _resolve_shader,
)


GATHER_SHADER = "gather_smooth_pair.wgsl"
GATHER_SHADER_F16 = "gather_smooth_pair_f16.wgsl"
GATHER_SHADER_I8 = "gather_smooth_pair_i8.wgsl"
FEATURES_SHADER = "traj_features.wgsl"
SEEDS_SHADER = "gather_seeds.wgsl"
SEGMENT_SHADER = "segment.wgsl"


@dataclass
class GatherHandle:
    """zero-copy handoff from gather to fit.

    out_buf: GPU buffer holding the packed paired trajectory for all
             articles in the batch, contiguous as [timestep][dim] row-major.
             lives in the ChainEngine's buffer pool; caller must NOT hold
             it across a subsequent gather call (pool may reuse or resize).
    out_buf_size_bytes: valid byte range of out_buf (may be ≤ the actual
             allocation if the pool overallocated).
    out_off: cumulative smoothed-timestep offsets, shape [n_articles+1]
             uint32. article a's data spans rows [out_off[a]..out_off[a+1]).
    D:       dimension (columns per timestep in out_buf).
    """
    out_buf: "wgpu.GPUBuffer"
    out_buf_size_bytes: int
    out_off: np.ndarray
    D: int
    # batch-pinned input buffers. the unified pipeline reads err/vel/
    # out_off_buf/pwr_buf from chain AFTER the dispatcher may have
    # prefetched the next batch (growing chain._err_buf etc. to a new,
    # larger GPUBuffer). capturing the refs here pins this batch's
    # buffers so a subsequent _grow_storage can't swap them out from
    # under us. bindgroups built from the handle read the CORRECT
    # buffers for this batch's data.
    err_buf: Optional["wgpu.GPUBuffer"] = None
    err_buf_size_bytes: int = 0
    vel_buf: Optional["wgpu.GPUBuffer"] = None
    vel_buf_size_bytes: int = 0
    out_off_buf: Optional["wgpu.GPUBuffer"] = None
    out_off_buf_size_bytes: int = 0
    pwr_buf: Optional["wgpu.GPUBuffer"] = None
    pwr_buf_size_bytes: int = 0


class ChainEngine:
    """composes with GpuEngine to add persistent matrix + chained gather.

    matrix_dtype controls the GPU-side storage of the GloVe matrix:
      "f32" (default) — stored as float32, 4 bytes per value.
      "f16"           — stored as float16, 2 bytes per value. on iGPU
                         with a 47 MB random-access surface this 2×
                         footprint reduction doubles cache density and
                         measurably beats the f32 path. requires the
                         shader-f16 adapter feature (always present on
                         AMD iGPU/Vulkan; falls back to f32 if missing).
                         quantization noise is averaged out by the
                         W=10 sliding-window mean.
    """

    def __init__(self, engine: GpuEngine, matrix: np.ndarray,
                 pairing: np.ndarray, window: int = 10,
                 matrix_dtype: str = "f32"):
        if not _HAS_WGPU:
            raise RuntimeError("wgpu not available")
        self.engine = engine
        self.device = engine.device
        self.window = int(window)
        self.matrix_dtype = matrix_dtype

        # auto-fallback if shader-f16 isn't available
        if matrix_dtype == "f16" and "shader-f16" not in set(self.device.features):
            matrix_dtype = "f32"
            self.matrix_dtype = "f32"

        # normalize inputs
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.V, self.D = self.matrix.shape

        if matrix_dtype == "f16":
            # store as f16 on GPU
            self.matrix_gpu = np.ascontiguousarray(
                self.matrix.astype(np.float16))
            self.matrix_scales = None
        elif matrix_dtype == "i8":
            # per-row int8 quantization with float scale. D must divide by 4.
            if self.D % 4 != 0:
                raise ValueError(
                    f"matrix_dtype='i8' requires D % 4 == 0, got D={self.D}")
            row_max = np.max(np.abs(self.matrix), axis=1)
            row_max = np.where(row_max < 1e-12, 1e-12, row_max)
            scales = (row_max / 127.0).astype(np.float32)
            quantized = np.round(
                self.matrix / scales[:, None]).clip(-128, 127).astype(np.int8)
            # pack 4 consecutive int8 into one u32 column
            q_u8 = quantized.view(np.uint8).reshape(self.V, self.D)
            # reshape to (V, D/4, 4) then combine as u32 little-endian
            q4 = q_u8.reshape(self.V, self.D // 4, 4)
            packed = (q4[..., 0].astype(np.uint32)
                      | (q4[..., 1].astype(np.uint32) << 8)
                      | (q4[..., 2].astype(np.uint32) << 16)
                      | (q4[..., 3].astype(np.uint32) << 24))
            self.matrix_gpu = np.ascontiguousarray(packed)
            self.matrix_scales = np.ascontiguousarray(scales)
        else:
            self.matrix_gpu = self.matrix
            self.matrix_scales = None
        self.pairing = np.ascontiguousarray(pairing, dtype=np.int32)
        assert self.pairing.shape == (self.D,), \
            f"pairing must be shape ({self.D},), got {self.pairing.shape}"

        # build pipelines
        self._build_pipeline()
        self._build_features_pipeline()

        # persistent buffers — populated by _upload_persistent
        self._glove_buf: Optional[wgpu.GPUBuffer] = None
        self._pairing_buf: Optional[wgpu.GPUBuffer] = None
        self._uniform_buf: Optional[wgpu.GPUBuffer] = None
        self._scales_buf: Optional[wgpu.GPUBuffer] = None

        # GPU timestamp infrastructure. the "timestamp-query" feature
        # lets us bracket each compute pass with GPU-recorded timestamps,
        # which measure ACTUAL kernel runtime (not wait-for-queue-drain
        # time). 8 slots: gather begin/end, features begin/end, plus
        # spares for future kernels. unlocks truth-telling: we can finally
        # tell gather-kernel-time apart from "everything queued before
        # gather's readback completes."
        self._has_timestamps = ("timestamp-query" in set(self.device.features))
        self._ts_query_set: Optional[wgpu.GPUQuerySet] = None
        self._ts_resolve_buf: Optional[wgpu.GPUBuffer] = None
        self._ts_readback_buf: Optional[wgpu.GPUBuffer] = None
        self._ts_slots = 8
        if self._has_timestamps:
            self._ts_query_set = self.device.create_query_set(
                type=wgpu.QueryType.timestamp,
                count=self._ts_slots,
            )
            self._ts_resolve_buf = self.device.create_buffer(
                size=self._ts_slots * 8,  # u64 per timestamp
                usage=(wgpu.BufferUsage.QUERY_RESOLVE
                       | wgpu.BufferUsage.COPY_SRC
                       | wgpu.BufferUsage.STORAGE),
            )
            self._ts_readback_buf = self.device.create_buffer(
                size=self._ts_slots * 8,
                usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
            )
        # last-observed GPU-measured kernel durations in nanoseconds.
        # populated by wait_features when timestamps are available.
        self.last_gather_ns: int = 0
        self.last_features_ns: int = 0

        # unified fit+seeds readback buffer. one buffer + one map_sync
        # replaces 4 separate syncs (K, G, rms, seeds). on iGPU with
        # ~10 ms driver cost per sync, that saves ~30 ms per batch.
        self._fused_rb_buf: Optional[wgpu.GPUBuffer] = None
        self._fused_rb_cap: int = 0

        # GPU segmentation buffers. seg_table_buf holds the final
        # (traj_offset, length, D, pair_off) records fit reads. seg_art
        # holds the per-segment article index for CPU-side absorb
        # grouping. seg_count_buf holds an atomic u32 counter of
        # segments emitted (we reset to 0 per batch via write_buffer).
        # fit_indirect_buf holds (n_segs, P_tiles, 1) for indirect
        # dispatch of fit — eliminates a CPU roundtrip to know n_segs.
        self._seg_table_buf: Optional[wgpu.GPUBuffer] = None
        self._seg_table_cap: int = 0
        self._seg_art_buf: Optional[wgpu.GPUBuffer] = None
        self._seg_art_cap: int = 0
        self._seg_count_buf: Optional[wgpu.GPUBuffer] = None
        self._fit_indirect_buf: Optional[wgpu.GPUBuffer] = None
        # readback buffers for the tiny per-batch seg metadata
        self._seg_count_rb: Optional[wgpu.GPUBuffer] = None
        self._seg_art_rb: Optional[wgpu.GPUBuffer] = None
        self._seg_art_rb_cap: int = 0

        # reusable per-batch buffers — grow on demand
        self._idx_buf: Optional[wgpu.GPUBuffer] = None
        self._idx_cap = 0
        self._off_buf: Optional[wgpu.GPUBuffer] = None
        self._off_cap = 0
        self._out_off_buf: Optional[wgpu.GPUBuffer] = None
        self._out_off_cap = 0
        self._out_buf: Optional[wgpu.GPUBuffer] = None
        self._out_cap = 0
        self._rb_buf: Optional[wgpu.GPUBuffer] = None
        self._rb_cap = 0

        # features kernel buffers (tiny: one float per timestep)
        self._err_buf: Optional[wgpu.GPUBuffer] = None
        self._err_cap = 0
        self._vel_buf: Optional[wgpu.GPUBuffer] = None
        self._vel_cap = 0
        self._pwr_buf: Optional[wgpu.GPUBuffer] = None
        self._pwr_cap = 0
        self._feat_rb_buf: Optional[wgpu.GPUBuffer] = None
        self._feat_rb_cap = 0  # sized for all three concatenated

        # seeds kernel buffers
        self._row_src_buf: Optional[wgpu.GPUBuffer] = None
        self._row_src_cap = 0
        self._seeds_buf: Optional[wgpu.GPUBuffer] = None
        self._seeds_cap = 0
        self._seeds_rb_buf: Optional[wgpu.GPUBuffer] = None
        self._seeds_rb_cap = 0

        self._upload_persistent()

    # ── pipeline ────────────────────────────────────────────────────

    def _build_pipeline(self) -> None:
        if self.matrix_dtype == "i8":
            shader_name = GATHER_SHADER_I8
        elif self.matrix_dtype == "f16":
            shader_name = GATHER_SHADER_F16
        else:
            shader_name = GATHER_SHADER
        wgsl = _resolve_shader(shader_name).read_text()
        module = self.device.create_shader_module(code=wgsl)

        layout_entries = [
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
        ]
        if self.matrix_dtype == "i8":
            # extra binding for per-row scales
            layout_entries.append(
                {"binding": 7, "visibility": wgpu.ShaderStage.COMPUTE,
                 "buffer": {"type": wgpu.BufferBindingType.read_only_storage}})
        self._layout = self.device.create_bind_group_layout(entries=layout_entries)
        layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._layout])
        self._pipeline = self.device.create_compute_pipeline(
            layout=layout,
            compute={"module": module, "entry_point": "main"},
        )

    def _build_features_pipeline(self) -> None:
        wgsl = _resolve_shader(FEATURES_SHADER).read_text()
        module = self.device.create_shader_module(code=wgsl)
        self._feat_layout = self.device.create_bind_group_layout(entries=[
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
            {"binding": 5, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ])
        layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._feat_layout])
        self._feat_pipeline = self.device.create_compute_pipeline(
            layout=layout,
            compute={"module": module, "entry_point": "main"},
        )

    def _upload_persistent(self) -> None:
        # self.matrix_gpu is:
        #   f32 path → float32[V, D]
        #   f16 path → float16[V, D]
        #   i8 path  → uint32[V, D/4]  (4 packed int8 per u32)
        m = self.matrix_gpu
        self._glove_buf = self.device.create_buffer(
            size=m.nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.device.queue.write_buffer(
            self._glove_buf, 0, memoryview(m.ravel()))

        # per-row scales for the i8 path
        self._scales_buf: Optional[wgpu.GPUBuffer] = None
        if self.matrix_scales is not None:
            self._scales_buf = self.device.create_buffer(
                size=self.matrix_scales.nbytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
            self.device.queue.write_buffer(
                self._scales_buf, 0,
                memoryview(self.matrix_scales))

        self._pairing_buf = self.device.create_buffer(
            size=max(self.pairing.nbytes, 256),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.device.queue.write_buffer(
            self._pairing_buf, 0, memoryview(self.pairing))

        self._uniform_buf = self.device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

    # ── buffer pool helpers ─────────────────────────────────────────

    def _grow_storage(self, buf, cap, need, extra_usage=0):
        if buf is not None and cap >= need:
            return buf, cap
        new_cap = max(need, cap * 2, 256)
        new_buf = self.device.create_buffer(
            size=new_cap,
            usage=(wgpu.BufferUsage.STORAGE
                   | wgpu.BufferUsage.COPY_DST
                   | wgpu.BufferUsage.COPY_SRC
                   | extra_usage),
        )
        return new_buf, new_cap

    def _grow_readback(self, buf, cap, need):
        if buf is not None and cap >= need:
            return buf, cap
        new_cap = max(need, cap * 2, 256)
        new_buf = self.device.create_buffer(
            size=new_cap,
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
        )
        return new_buf, new_cap

    # ── public API ──────────────────────────────────────────────────

    # ── internal dispatch helpers ───────────────────────────────────

    def submit_gather_and_features(
        self, indices_list: List[np.ndarray],
        max_batch_bytes: int = 512 * 1024 * 1024,
    ) -> Optional[dict]:
        """fuse gather + features into ONE encoder + ONE submit.

        the compute passes are sequenced within the encoder (gather
        writes out_pair, features reads it), so there's no extra
        synchronization needed — wgpu guarantees happens-before within
        a single submit.

        saves one queue-lock acquisition per batch relative to calling
        gather_paired_batch_gpu_only followed by submit_features.

        returns a combined dict {handle, feat_ctx} or None if the
        batch was empty.
        """
        W = self.window
        D = self.D
        n_articles = len(indices_list)
        if n_articles == 0:
            return None

        # ── build gather inputs (CPU side) ─────────────────────────
        lens = np.fromiter((len(x) for x in indices_list),
                           dtype=np.int32, count=n_articles)
        total_tokens = int(lens.sum())
        offsets = np.empty(n_articles + 1, dtype=np.uint32)
        out_off = np.empty(n_articles + 1, dtype=np.uint32)
        offsets[0] = 0
        out_off[0] = 0
        c_tok = 0
        c_smooth = 0
        for i, L in enumerate(lens):
            c_tok += int(L)
            offsets[i + 1] = c_tok
            n_smooth = max(0, int(L) - W + 1)
            c_smooth += n_smooth
            out_off[i + 1] = c_smooth
        total_smooth = int(out_off[-1])
        if total_smooth == 0:
            return None

        all_idx = np.empty(total_tokens, dtype=np.int32)
        c = 0
        for x in indices_list:
            if len(x) == 0:
                continue
            all_idx[c:c + len(x)] = x
            c += len(x)

        out_bytes = total_smooth * D * 4
        if out_bytes > max_batch_bytes:
            raise RuntimeError(
                f"gather output too big: {out_bytes/1e6:.0f} MB "
                f"(cap {max_batch_bytes/1e6:.0f} MB)")

        # grow all buffers (gather + features)
        self._idx_buf, self._idx_cap = self._grow_storage(
            self._idx_buf, self._idx_cap, all_idx.nbytes)
        self._off_buf, self._off_cap = self._grow_storage(
            self._off_buf, self._off_cap, offsets.nbytes)
        self._out_off_buf, self._out_off_cap = self._grow_storage(
            self._out_off_buf, self._out_off_cap, out_off.nbytes)
        self._out_buf, self._out_cap = self._grow_storage(
            self._out_buf, self._out_cap, out_bytes)

        feat_bytes = total_smooth * 4
        feat_total = feat_bytes * 3
        self._err_buf, self._err_cap = self._grow_storage(
            self._err_buf, self._err_cap, feat_bytes)
        self._vel_buf, self._vel_cap = self._grow_storage(
            self._vel_buf, self._vel_cap, feat_bytes)
        self._pwr_buf, self._pwr_cap = self._grow_storage(
            self._pwr_buf, self._pwr_cap, feat_bytes)
        self._feat_rb_buf, self._feat_rb_cap = self._grow_readback(
            self._feat_rb_buf, self._feat_rb_cap, feat_total)

        # uploads. the gather uniform (W, D, n_articles) differs from
        # the features uniform (D, n_articles), so we can only share
        # if we rewrite the uniform between passes — which a single
        # encoder can't do. use two separate uniform buffers.
        self.device.queue.write_buffer(self._idx_buf, 0, memoryview(all_idx))
        self.device.queue.write_buffer(self._off_buf, 0, memoryview(offsets))
        self.device.queue.write_buffer(self._out_off_buf, 0, memoryview(out_off))

        # first-time allocation of a second uniform for features
        if getattr(self, "_uniform_feat_buf", None) is None:
            self._uniform_feat_buf = self.device.create_buffer(
                size=16,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )

        gather_params = np.array([W, D, n_articles, 0], dtype=np.uint32)
        self.device.queue.write_buffer(
            self._uniform_buf, 0, memoryview(gather_params))
        feat_params = np.array([D, n_articles, 0, 0], dtype=np.uint32)
        self.device.queue.write_buffer(
            self._uniform_feat_buf, 0, memoryview(feat_params))

        # gather bind group (cached)
        gb_key = (
            id(self._idx_buf), self._idx_cap,
            id(self._off_buf), self._off_cap,
            id(self._out_off_buf), self._out_off_cap,
            id(self._out_buf), self._out_cap,
        )
        if getattr(self, "_gather_bg_key", None) != gb_key:
            entries = [
                {"binding": 0, "resource": {"buffer": self._glove_buf,
                                             "offset": 0, "size": self.matrix_gpu.nbytes}},
                {"binding": 1, "resource": {"buffer": self._idx_buf,
                                             "offset": 0, "size": self._idx_cap}},
                {"binding": 2, "resource": {"buffer": self._off_buf,
                                             "offset": 0, "size": self._off_cap}},
                {"binding": 3, "resource": {"buffer": self._pairing_buf,
                                             "offset": 0, "size": max(self.pairing.nbytes, 256)}},
                {"binding": 4, "resource": {"buffer": self._out_off_buf,
                                             "offset": 0, "size": self._out_off_cap}},
                {"binding": 5, "resource": {"buffer": self._out_buf,
                                             "offset": 0, "size": self._out_cap}},
                {"binding": 6, "resource": {"buffer": self._uniform_buf,
                                             "offset": 0, "size": 16}},
            ]
            if self._scales_buf is not None:
                entries.append(
                    {"binding": 7, "resource": {"buffer": self._scales_buf,
                                                 "offset": 0,
                                                 "size": self.matrix_scales.nbytes}})
            self._gather_bg = self.device.create_bind_group(
                layout=self._layout, entries=entries)
            self._gather_bg_key = gb_key

        # features bind group (cached, uses out_buf as input)
        fb_key = (
            id(self._out_buf), out_bytes,
            id(self._out_off_buf), self._out_off_cap,
            id(self._err_buf), self._err_cap,
            id(self._vel_buf), self._vel_cap,
            id(self._pwr_buf), self._pwr_cap,
        )
        if getattr(self, "_feat_bg_key", None) != fb_key:
            self._feat_bg = self.device.create_bind_group(
                layout=self._feat_layout,
                entries=[
                    {"binding": 0, "resource": {"buffer": self._out_buf,
                                                 "offset": 0, "size": out_bytes}},
                    {"binding": 1, "resource": {"buffer": self._out_off_buf,
                                                 "offset": 0, "size": self._out_off_cap}},
                    {"binding": 2, "resource": {"buffer": self._err_buf,
                                                 "offset": 0, "size": self._err_cap}},
                    {"binding": 3, "resource": {"buffer": self._vel_buf,
                                                 "offset": 0, "size": self._vel_cap}},
                    {"binding": 4, "resource": {"buffer": self._pwr_buf,
                                                 "offset": 0, "size": self._pwr_cap}},
                    {"binding": 5, "resource": {"buffer": self._uniform_feat_buf,
                                                 "offset": 0, "size": 16}},
                ],
            )
            self._feat_bg_key = fb_key

        # dispatch sizes
        gather_wg_x = (total_smooth + 3) // 4
        gather_wg_y = (D + 63) // 64
        if gather_wg_x > 65535 or gather_wg_y > 65535:
            raise RuntimeError(
                f"gather dispatch too large: ({gather_wg_x}, {gather_wg_y})")
        feat_wg_x = (total_smooth + 63) // 64
        if feat_wg_x > 65535:
            raise RuntimeError(
                f"features dispatch too large: {feat_wg_x}")

        # ── single encoder: gather pass → features pass → readback ──
        enc = self.device.create_command_encoder()

        # timestamp slots: 0=gather_start, 1=gather_end,
        #                  2=features_start, 3=features_end
        if self._has_timestamps:
            cp = enc.begin_compute_pass(timestamp_writes={
                "query_set": self._ts_query_set,
                "beginning_of_pass_write_index": 0,
                "end_of_pass_write_index": 1,
            })
        else:
            cp = enc.begin_compute_pass()
        cp.set_pipeline(self._pipeline)
        cp.set_bind_group(0, self._gather_bg)
        cp.dispatch_workgroups(gather_wg_x, gather_wg_y, 1)
        cp.end()

        if self._has_timestamps:
            cp = enc.begin_compute_pass(timestamp_writes={
                "query_set": self._ts_query_set,
                "beginning_of_pass_write_index": 2,
                "end_of_pass_write_index": 3,
            })
        else:
            cp = enc.begin_compute_pass()
        cp.set_pipeline(self._feat_pipeline)
        cp.set_bind_group(0, self._feat_bg)
        cp.dispatch_workgroups(feat_wg_x, 1, 1)
        cp.end()

        enc.copy_buffer_to_buffer(
            self._err_buf, 0, self._feat_rb_buf, 0, feat_bytes)
        enc.copy_buffer_to_buffer(
            self._vel_buf, 0, self._feat_rb_buf, feat_bytes, feat_bytes)
        enc.copy_buffer_to_buffer(
            self._pwr_buf, 0, self._feat_rb_buf, feat_bytes * 2, feat_bytes)

        if self._has_timestamps:
            # resolve timestamps into staging buffer, then copy to readback.
            enc.resolve_query_set(
                query_set=self._ts_query_set, first_query=0, query_count=4,
                destination=self._ts_resolve_buf, destination_offset=0)
            enc.copy_buffer_to_buffer(
                self._ts_resolve_buf, 0, self._ts_readback_buf, 0, 4 * 8)

        self.device.queue.submit([enc.finish()])

        handle = GatherHandle(
            out_buf=self._out_buf,
            out_buf_size_bytes=out_bytes,
            out_off=out_off,
            D=D,
            # pin this batch's chain input buffers so a prefetch submit
            # of the next (possibly larger) batch can't replace them.
            err_buf=self._err_buf,
            err_buf_size_bytes=feat_bytes,
            vel_buf=self._vel_buf,
            vel_buf_size_bytes=feat_bytes,
            out_off_buf=self._out_off_buf,
            out_off_buf_size_bytes=out_off.nbytes,
            pwr_buf=self._pwr_buf,
            pwr_buf_size_bytes=feat_bytes,
        )
        feat_ctx = {
            "total_smooth": total_smooth,
            "out_bytes": feat_bytes,
            "feat_bytes_total": feat_total,
            "has_timestamps": self._has_timestamps,
        }
        return {"handle": handle, "feat_ctx": feat_ctx}

    def wait_features(
        self, ctx: dict,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """map_sync on the features readback buffer and split the three
        streams into numpy arrays. if timestamps were captured in the
        corresponding submit, also update self.last_gather_ns and
        self.last_features_ns with the GPU-measured kernel runtimes.
        returns (err, vel, pwr, readback_ms).

        POLLER TRICK: use `map_async` + promise.sync_wait instead of
        `map_sync`. the async path routes completion through the wgpu-py
        background poller thread and is 5-10% faster + far more
        consistent at production buffer sizes. critically, `map_async`
        waits only on THIS buffer, so if the prefetch worker has already
        queued the next batch's gather+features (typical in this
        pipeline), we don't stall on its work — unlike `_poll_wait()`
        which drains the entire queue. see bench_async_completion.py.
        """
        import time
        total_smooth = ctx["total_smooth"]
        feat_bytes_total = ctx["feat_bytes_total"]

        t0 = time.perf_counter()
        self._feat_rb_buf.map_async(
            wgpu.MapMode.READ, size=feat_bytes_total).sync_wait()
        raw = np.frombuffer(
            self._feat_rb_buf.read_mapped(size=feat_bytes_total),
            dtype=np.float32,
        )
        err = raw[:total_smooth].copy()
        vel = raw[total_smooth:2 * total_smooth].copy()
        pwr = raw[2 * total_smooth:3 * total_smooth].copy()
        self._feat_rb_buf.unmap()
        readback_ms = (time.perf_counter() - t0) * 1000.0

        # harvest GPU timestamps. the feat_rb map_sync above already
        # guarantees all prior GPU work is done, so the ts_readback
        # buffer is valid to map now (no extra sync needed).
        if ctx.get("has_timestamps") and self._has_timestamps:
            try:
                self._ts_readback_buf.map_sync(wgpu.MapMode.READ, size=4 * 8)
                ts = np.frombuffer(
                    self._ts_readback_buf.read_mapped(size=4 * 8),
                    dtype=np.uint64,
                ).copy()
                self._ts_readback_buf.unmap()
                # timestamp ticks on AMD Vulkan are typically 1 ns each;
                # even if they're not, deltas between adjacent queries
                # are proportional, so "ns" here is treated as a relative
                # unit for profiler bookkeeping.
                self.last_gather_ns = int(ts[1] - ts[0])
                self.last_features_ns = int(ts[3] - ts[2])
            except Exception:
                pass

        return err, vel, pwr, readback_ms

