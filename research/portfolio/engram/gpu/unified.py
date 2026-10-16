"""
unified.py — fuses segment + fit + per_article_reduce into one GPU submit.

OLD PIPELINE (3 GPU submits per batch):
  1. chain.submit_gather_and_features  (gather + features)
  2. engine.submit_fit_on_buffer        (fit after CPU numba segment)
  3. brain.absorb_batch                 (centroid + route)

  Between each: CPU wait_features + CPU numba segment (1-2 ms kernel,
  ~50 ms driver sync) then CPU reduceat (2-3 ms) then brain's CPU work.

NEW PIPELINE (2 GPU submits per batch):
  1. chain.submit_gather_and_features  (unchanged)
  2. unified.submit_fused              (segment → finalize → fit[indirect]
                                         → per_article_reduce) — reads
                                         features directly from chain's
                                         GPU buffers, never round-trips
                                         through CPU numba.
  3. brain.absorb_batch                 (unchanged but takes K_sums
                                         instead of K_stack — much
                                         cheaper CPU apply)

One driver round-trip eliminated (numba sync ≈ 50 ms). The readback
from submit #2 carries per-article sums (K_sums, G_sums, rms_sums,
n_segs_per_art) so Brain's apply stage does trivial dict updates
instead of a [total_segs, P] reduceat.

Ownership:
  - segment.wgsl and per_article_reduce.wgsl live in shaders/
  - This module owns the matching pipelines + per-batch scratch buffers
  - chain's output buffers (err, vel, out_buf) are consumed as-is
  - engine's fit pipeline is consumed as-is via indirect dispatch
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import wgpu
    _HAS_WGPU = True
except Exception:
    _HAS_WGPU = False

from engram_codec import MIN_BLOCK, MAX_BLOCK


SHADER_DIR = Path(__file__).parent / "shaders"


def _read_wgsl(name: str) -> str:
    return (SHADER_DIR / name).read_text(encoding="utf-8")


class UnifiedStage:
    """owns segment + finalize_indirect + per_article_reduce kernels,
    composes them with engine's fit pipeline into one encoder/submit.

    usage (after chain.submit_gather_and_features has been queued):

        result = unified.submit_fused(
            chain, engine,
            handle=gather_handle,    # from chain's submit_gather_and_features
            feat_ctx=feat_ctx,       # from chain's submit_gather_and_features
            n_art=n_articles,
            max_block=macro_block_ceiling,
        )
        # result: {'K_sums', 'G_sums', 'rms_sums', 'n_segs_per_art',
        #          'seg_count', 'seg_art_idx'}
        # values are host-side numpy; one map_async.sync_wait backed them.

    the submit records 4 compute passes in order:
      1. segment      reads chain.err/vel/out_off → seg_table,
                      seg_art_idx, seg_count, seg_range
      2. finalize     reads seg_count → fit_indirect(n_segs, P_tiles, 1)
      3. fit          reads chain.out_buf + seg_table → K/G/rms per seg
      4. reduce       reads K/G/rms per seg + seg_range → per-article sums

    WebGPU inserts automatic storage-buffer RAW barriers between passes,
    so the compute-pass sequence is race-free without explicit syncs.
    """

    def __init__(self, device, engine, P: int):
        if not _HAS_WGPU:
            raise RuntimeError("wgpu-py required")
        self.device = device
        self.engine = engine
        self.P = int(P)

        # persistent state buffers (small, fixed-size)
        self._seg_count_buf = self.device.create_buffer(
            size=16,  # one u32 + padding
            usage=(wgpu.BufferUsage.STORAGE
                   | wgpu.BufferUsage.COPY_DST
                   | wgpu.BufferUsage.COPY_SRC),
        )
        self._fit_indirect_buf = self.device.create_buffer(
            size=16,
            usage=(wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
                   | wgpu.BufferUsage.INDIRECT),
        )
        # uniform buffers (16 bytes each)
        self._seg_params = self.device.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._fin_params = self.device.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._par_params = self.device.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        # scratch caps (grow on demand)
        self._n_art_cap = 0
        self._seg_cap = 0
        self._seg_table_buf = None
        self._seg_art_buf = None
        self._seg_range_buf = None
        self._K_per_seg_buf = None
        self._G_per_seg_buf = None
        self._rms_per_seg_buf = None
        self._K_sums_buf = None
        self._G_sums_buf = None
        self._rms_sums_buf = None
        self._n_segs_per_art_buf = None
        # unified readback buffer
        self._rb_buf = None
        self._rb_cap = 0

        self._build_pipelines()

    # ── pipelines ──────────────────────────────────────────────────

    def _build_pipelines(self):
        # segment kernel
        module = self.device.create_shader_module(code=_read_wgsl("segment.wgsl"))
        self._seg_layout = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 5, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 6, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 7, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 8, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
        ])
        self._seg_pipe = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(
                bind_group_layouts=[self._seg_layout]),
            compute={"module": module, "entry_point": "main"},
        )

        # finalize_indirect kernel
        module = self.device.create_shader_module(
            code=_read_wgsl("finalize_indirect.wgsl"))
        self._fin_layout = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ])
        self._fin_pipe = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(
                bind_group_layouts=[self._fin_layout]),
            compute={"module": module, "entry_point": "main"},
        )

        # per_article_reduce kernel
        module = self.device.create_shader_module(
            code=_read_wgsl("per_article_reduce.wgsl"))
        self._par_layout = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 5, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 6, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 7, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 8, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ])
        self._par_pipe = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(
                bind_group_layouts=[self._par_layout]),
            compute={"module": module, "entry_point": "main"},
        )

    # ── scratch sizing ─────────────────────────────────────────────

    def _ensure(self, n_art: int, max_segs: int):
        """grow scratch buffers on demand. old buffers are left to be
        garbage-collected — explicit .destroy() races with in-flight
        commands on some driver versions and can lose the device.
        """
        P = self.P

        if n_art > self._n_art_cap:
            self._n_art_cap = max(n_art * 2, 64)
            cap = self._n_art_cap
            S = wgpu.BufferUsage.STORAGE
            D = wgpu.BufferUsage.COPY_DST
            R = wgpu.BufferUsage.COPY_SRC
            self._seg_range_buf = self.device.create_buffer(
                size=cap * 8, usage=S | D)
            self._K_sums_buf = self.device.create_buffer(
                size=cap * 2 * P * 4, usage=S | R)
            self._G_sums_buf = self.device.create_buffer(
                size=cap * 2 * P * 4, usage=S | R)
            self._rms_sums_buf = self.device.create_buffer(
                size=cap * P * 4, usage=S | R)
            self._n_segs_per_art_buf = self.device.create_buffer(
                size=cap * 4, usage=S | R)

        if max_segs > self._seg_cap:
            self._seg_cap = max(max_segs * 2, 256)
            cap = self._seg_cap
            S = wgpu.BufferUsage.STORAGE
            D = wgpu.BufferUsage.COPY_DST
            R = wgpu.BufferUsage.COPY_SRC
            self._seg_table_buf = self.device.create_buffer(
                size=cap * 16, usage=S | D)
            self._seg_art_buf = self.device.create_buffer(
                size=cap * 4, usage=S | R)
            self._K_per_seg_buf = self.device.create_buffer(
                size=cap * 2 * P * 4, usage=S)
            self._G_per_seg_buf = self.device.create_buffer(
                size=cap * 2 * P * 4, usage=S)
            self._rms_per_seg_buf = self.device.create_buffer(
                size=cap * P * 4, usage=S)

    def _ensure_rb(self, n_art: int) -> dict:
        """layout offsets + ensure unified readback buffer is large enough."""
        P = self.P
        kg = n_art * 2 * P * 4
        rms = n_art * P * 4
        n32 = n_art * 4
        # K_sums | G_sums | rms_sums | n_segs_per | seg_art (n_art*4) | seg_count (4)
        # seg_art_idx only needed per-seg, not per-art — but we want total_segs
        # so we read seg_count, seg_art_idx via separate readback if needed.
        # for now, include just seg_count for sanity.
        total = kg + kg + rms + n32 + 16
        if total > self._rb_cap:
            self._rb_cap = total * 2
            self._rb_buf = self.device.create_buffer(
                size=self._rb_cap,
                usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
            )
        return {
            "K_sums":    (0,                        kg),
            "G_sums":    (kg,                       kg),
            "rms_sums":  (2 * kg,                   rms),
            "n_segs":    (2 * kg + rms,             n32),
            "seg_count": (2 * kg + rms + n32,       4),
            "_total":    total,
        }

    # ── the fused submit ──────────────────────────────────────────

    def submit_fused(self, chain, engine, handle, feat_ctx,
                     n_art: int,
                     max_block: int = MAX_BLOCK) -> Optional[dict]:
        """record segment+finalize+fit+per_article_reduce into one
        encoder, submit, readback. handle + feat_ctx come from a
        prior chain.submit_gather_and_features call.
        """
        P = self.P
        D = handle.D
        total_smooth = feat_ctx["total_smooth"]

        # per-article smoothed-length cap. segment.wgsl runs its ACF
        # + emission loops in a single workgroup per article; each
        # thread's work is O(T * MAX_LAG). at T > ~20k rows, kernel
        # time crosses Windows' 2s TDR threshold and the driver
        # terminates the device. legacy path doesn't hit this because
        # numba segmentation is CPU-bound. for oversized articles we
        # bail out of the unified path; the dispatcher logs drops.
        # per-article size guard (TDR protection)
        MAX_SMOOTH_PER_ART = 20000
        art_sizes = np.diff(handle.out_off)
        max_size = int(art_sizes.max())
        if max_size > MAX_SMOOTH_PER_ART:
            return None

        # safe upper bound on emitted segments. two ceilings apply:
        #   1. one segment per MIN_BLOCK rows  -> total_smooth // MIN_BLOCK
        #   2. segment.wgsl caps per-article emission at MAX_CAND=2048
        #      via the workgroup scratch buffer (wg_cand_ss / wg_cand_L)
        # the tighter of the two is the true ceiling. earlier code used
        # only ceiling #1 which overshoots by ~10-15x on realistic
        # batches and forced byte-level ingestion to tiny batch_size=8;
        # with the tighter min() we can run batch_size up to ~30 on
        # long-article corpora. WebGPU's indirect-dispatch axis limit
        # is 65535 so we bail whole batches whose WORST case would
        # exceed it (real emission is typically 6-8% of this bound).
        MAX_CAND_PER_ART = 2048  # must match segment.wgsl's MAX_CAND
        max_segs_est = (min(total_smooth // MIN_BLOCK,
                            n_art * MAX_CAND_PER_ART)
                        + n_art + 2)
        if max_segs_est < 16:
            max_segs_est = 16
        if max_segs_est > 65000:
            return None

        self._ensure(n_art, max_segs_est)
        off = self._ensure_rb(n_art)

        fit_WG = 64
        P_tiles = (P + fit_WG - 1) // fit_WG

        # segment params: (D, n_art, P_tiles, max_block)
        self.device.queue.write_buffer(self._seg_params, 0, memoryview(
            np.array([D, n_art, P_tiles, max_block], dtype=np.uint32)))
        self.device.queue.write_buffer(self._fin_params, 0, memoryview(
            np.array([P_tiles, 0, 0, 0], dtype=np.uint32)))
        self.device.queue.write_buffer(self._par_params, 0, memoryview(
            np.array([n_art, P, 0, 0], dtype=np.uint32)))

        # reset seg_count each batch
        self.device.queue.write_buffer(
            self._seg_count_buf, 0,
            memoryview(np.zeros(4, dtype=np.uint32)))

        # ── build encoder ──────────────────────────────────────────
        enc = self.device.create_command_encoder()

        # 1. segment (one WG of 128 per article, reads this batch's pinned
        # err/vel/out_off buffers from the handle — NOT chain.*_buf, which
        # the dispatcher may have swapped to a grown buffer when prefetching
        # the next batch between gather submit and unified submit).
        err_buf = handle.err_buf if handle.err_buf is not None else chain._err_buf
        vel_buf = handle.vel_buf if handle.vel_buf is not None else chain._vel_buf
        out_off_buf = (handle.out_off_buf if handle.out_off_buf is not None
                        else chain._out_off_buf)
        err_sz = handle.err_buf_size_bytes or chain._err_cap
        vel_sz = handle.vel_buf_size_bytes or chain._vel_cap
        out_off_sz = handle.out_off_buf_size_bytes or chain._out_off_cap
        bg_seg = self.device.create_bind_group(
            layout=self._seg_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": err_buf,
                    "offset": 0, "size": err_sz}},
                {"binding": 1, "resource": {"buffer": vel_buf,
                    "offset": 0, "size": vel_sz}},
                {"binding": 2, "resource": {"buffer": out_off_buf,
                    "offset": 0, "size": out_off_sz}},
                {"binding": 3, "resource": {"buffer": self._seg_table_buf,
                    "offset": 0, "size": self._seg_table_buf.size}},
                {"binding": 4, "resource": {"buffer": self._seg_art_buf,
                    "offset": 0, "size": self._seg_art_buf.size}},
                {"binding": 5, "resource": {"buffer": self._seg_count_buf,
                    "offset": 0, "size": 16}},
                {"binding": 6, "resource": {"buffer": self._fit_indirect_buf,
                    "offset": 0, "size": 16}},
                {"binding": 7, "resource": {"buffer": self._seg_params,
                    "offset": 0, "size": 16}},
                {"binding": 8, "resource": {"buffer": self._seg_range_buf,
                    "offset": 0, "size": self._seg_range_buf.size}},
            ],
        )
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self._seg_pipe)
        cp.set_bind_group(0, bg_seg)
        cp.dispatch_workgroups(n_art, 1, 1)
        cp.end()

        # 2. finalize_indirect (1 WG of 1 thread)
        bg_fin = self.device.create_bind_group(
            layout=self._fin_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self._seg_count_buf,
                    "offset": 0, "size": 16}},
                {"binding": 1, "resource": {"buffer": self._fit_indirect_buf,
                    "offset": 0, "size": 16}},
                {"binding": 2, "resource": {"buffer": self._fin_params,
                    "offset": 0, "size": 16}},
            ],
        )
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self._fin_pipe)
        cp.set_bind_group(0, bg_fin)
        cp.dispatch_workgroups(1, 1, 1)
        cp.end()

        # 3. fit (indirect dispatch, reads chain.out_buf + seg_table)
        bg_fit = self.device.create_bind_group(
            layout=engine._bind_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": handle.out_buf,
                    "offset": 0, "size": handle.out_buf_size_bytes}},
                {"binding": 1, "resource": {"buffer": self._seg_table_buf,
                    "offset": 0, "size": self._seg_table_buf.size}},
                {"binding": 2, "resource": {"buffer": self._K_per_seg_buf,
                    "offset": 0, "size": self._K_per_seg_buf.size}},
                {"binding": 3, "resource": {"buffer": self._G_per_seg_buf,
                    "offset": 0, "size": self._G_per_seg_buf.size}},
                {"binding": 4, "resource": {"buffer": self._rms_per_seg_buf,
                    "offset": 0, "size": self._rms_per_seg_buf.size}},
            ],
        )
        cp = enc.begin_compute_pass()
        cp.set_pipeline(engine._pipeline)
        cp.set_bind_group(0, bg_fit)
        cp.dispatch_workgroups_indirect(self._fit_indirect_buf, 0)
        cp.end()

        # 4. per_article_reduce
        bg_par = self.device.create_bind_group(
            layout=self._par_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self._K_per_seg_buf,
                    "offset": 0, "size": self._K_per_seg_buf.size}},
                {"binding": 1, "resource": {"buffer": self._G_per_seg_buf,
                    "offset": 0, "size": self._G_per_seg_buf.size}},
                {"binding": 2, "resource": {"buffer": self._rms_per_seg_buf,
                    "offset": 0, "size": self._rms_per_seg_buf.size}},
                {"binding": 3, "resource": {"buffer": self._seg_range_buf,
                    "offset": 0, "size": self._seg_range_buf.size}},
                {"binding": 4, "resource": {"buffer": self._K_sums_buf,
                    "offset": 0, "size": self._K_sums_buf.size}},
                {"binding": 5, "resource": {"buffer": self._G_sums_buf,
                    "offset": 0, "size": self._G_sums_buf.size}},
                {"binding": 6, "resource": {"buffer": self._rms_sums_buf,
                    "offset": 0, "size": self._rms_sums_buf.size}},
                {"binding": 7, "resource": {"buffer": self._n_segs_per_art_buf,
                    "offset": 0, "size": self._n_segs_per_art_buf.size}},
                {"binding": 8, "resource": {"buffer": self._par_params,
                    "offset": 0, "size": 16}},
            ],
        )
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self._par_pipe)
        cp.set_bind_group(0, bg_par)
        cp.dispatch_workgroups(n_art, 1, 1)
        cp.end()

        # 5. pack unified readback
        enc.copy_buffer_to_buffer(self._K_sums_buf,   0, self._rb_buf,
                                   off["K_sums"][0],   off["K_sums"][1])
        enc.copy_buffer_to_buffer(self._G_sums_buf,   0, self._rb_buf,
                                   off["G_sums"][0],   off["G_sums"][1])
        enc.copy_buffer_to_buffer(self._rms_sums_buf, 0, self._rb_buf,
                                   off["rms_sums"][0], off["rms_sums"][1])
        enc.copy_buffer_to_buffer(self._n_segs_per_art_buf, 0, self._rb_buf,
                                   off["n_segs"][0],   off["n_segs"][1])
        enc.copy_buffer_to_buffer(self._seg_count_buf, 0, self._rb_buf,
                                   off["seg_count"][0], 4)
        self.device.queue.submit([enc.finish()])

        # 6. ONE sync_wait retrieves everything
        self._rb_buf.map_async(
            wgpu.MapMode.READ, size=off["_total"]).sync_wait()
        raw = self._rb_buf.read_mapped(size=off["_total"])
        flat = np.frombuffer(raw, dtype=np.uint8)

        K_sums = np.frombuffer(
            flat[off["K_sums"][0]:off["K_sums"][0]+off["K_sums"][1]].tobytes(),
            dtype=np.complex64).reshape(n_art, P).copy()
        G_sums = np.frombuffer(
            flat[off["G_sums"][0]:off["G_sums"][0]+off["G_sums"][1]].tobytes(),
            dtype=np.complex64).reshape(n_art, P).copy()
        rms_sums = np.frombuffer(
            flat[off["rms_sums"][0]:off["rms_sums"][0]+off["rms_sums"][1]].tobytes(),
            dtype=np.float32).reshape(n_art, P).copy()
        n_segs_per_art = np.frombuffer(
            flat[off["n_segs"][0]:off["n_segs"][0]+off["n_segs"][1]].tobytes(),
            dtype=np.uint32).copy()
        seg_count = int(np.frombuffer(
            flat[off["seg_count"][0]:off["seg_count"][0]+4].tobytes(),
            dtype=np.uint32)[0])
        self._rb_buf.unmap()

        return {
            "K_sums": K_sums,
            "G_sums": G_sums,
            "rms_sums": rms_sums,
            "n_segs_per_art": n_segs_per_art,
            "total_segs": seg_count,
        }
