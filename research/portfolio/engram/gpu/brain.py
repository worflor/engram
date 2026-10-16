"""
brain.py — the engram brain. one unified CPU+GPU object.

a Brain wraps an EngramFile and a GpuEngine. absorb decisions (which
well each observation joins, when a new well is seeded) run on the GPU
via three kernels; absorbing state (well sums, counts, dream buffer)
lives canonically in the EngramFile so persistence and inspection stay
simple.

one API:

    brain = Brain(ef, engine)
    names = brain.absorb_from_sums(K_sums, G_sums, rms_sums, n_segs_per_art)

consumes per-article sums as produced by the unified GPU pipeline's
per_article_reduce kernel. works for any Source — the brain only sees
complex K/G sums + float rms + per-article segment counts, not what
produced them.

no fallbacks, no thresholds, no env flags. this is the only path.

state layout
============
  GPU (bindings mirror the CPU-authoritative side):
    well_centroids [W_cap, 2*P] f32 interleaved re/im
    well_count     [W_cap]      u32
    gap_nn_dists   [W_cap]      f32   — kernel scratch
    obs_centroid   [n_art, 2*P] f32   — CPU writes pre-computed per-
                                        article centroids (K_sum/n_seg)
    route outputs  [n_art]      three arrays, coalesced into one MAP_READ

  CPU (EngramFile):
    _well_sum_K, _well_count — authoritative for persistence
    _dream_Ks/Gs/Ss          — circular buffer
    _next_well_id            — well naming

the two kernels
===============
  absorb_gap.wgsl   — pairwise well distances, median on host gives
                      typical_gap (recomputed only when the
                      established set changes size)
  absorb_route.wgsl — parallel per-article argmin + escape decision

constants that used to be hardcoded in WGSL (WELL_ESTABLISHED,
ESCAPE_ENERGY) now come from engram_codec via uniforms. the shaders
are pure physics; numerical thresholds are one source of truth.
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

# engram_codec owns the decision constants. shaders receive them as
# uniforms; python side reads them here.
from engram_codec import (
    EngramFile, WELL_ESTABLISHED, ESCAPE_ENERGY, BOOTSTRAP_WELLS,
)


SHADER_DIR = Path(__file__).parent / "shaders"


class Brain:
    """GPU-accelerated unified absorb engine over an EngramFile.

    usage:
        brain = Brain(ef, engine)
        names = brain.absorb_batch(K, G, rms, boundaries)

    `ef` is the authoritative CPU state (persistence, dream buffers).
    `engine` is a GpuEngine providing the device; Brain holds its own
    state buffers and pipelines on that device.

    the well capacity grows on demand — the first batch that exceeds
    the current capacity triggers an in-place resize. no explicit
    ceiling, no hard failure.

    dim comes from ef. P = dim // 2.
    """

    def __init__(self, ef: EngramFile, engine, well_cap: int = 256,
                 escape_energy: Optional[float] = None):
        if not _HAS_WGPU:
            raise RuntimeError("wgpu-py required")
        self.ef = ef
        self.engine = engine
        self.device = engine.device
        self.dim = int(ef.dim)
        self.P = self.dim // 2

        # escape threshold multiplier. see absorb_route.wgsl:
        #     spawn if nn_d > typical_gap * escape_energy * sqrt(...)
        # default (2.0) comes from engram_codec's ESCAPE_ENERGY; lower
        # values make the brain more willing to spawn new wells (finer
        # clustering), higher values lock it into a few dominant
        # attractors (stable identity). tunable per brain for workloads
        # that need different clustering granularity.
        self.escape_energy = (
            float(escape_energy) if escape_energy is not None
            else float(ESCAPE_ENERGY))

        # grow-on-demand capacity. starts small; doubles when exceeded.
        self._well_cap = max(well_cap, 256)

        # GPU state (built below; resizable)
        self._centroid_buf = None
        self._count_buf = None
        self._nn_dists_buf = None
        self._nn_dists_rb = None
        self._alloc_well_state()

        # per-batch scratch (grows on demand)
        self._obs_cap = 0
        self._obs_centroid_buf = None
        self._route_idx_buf = None
        self._route_dist_buf = None
        self._route_new_buf = None
        self._route_rb = None

        # uniforms (16 bytes each)
        self._params_gap      = self._mkuniform()
        self._params_route_u  = self._mkuniform()
        self._params_route_f  = self._mkuniform()

        # pipelines
        self._build_pipelines()

        # host-side routing tables — map ef's well-name dict to dense
        # GPU row indices.
        self._name_to_row: dict[str, int] = {}
        self._row_to_name: list[str] = []
        self._n_wells = 0

        # typical_gap: median nearest-neighbor distance among wells.
        # this is a SCALE — meaningful with any >= 2 non-empty wells,
        # not just mature ones. tying its computation to
        # WELL_ESTABLISHED (as prior versions did) caused a catastrophic
        # bootstrap bug: while fewer than 2 mature wells existed,
        # typical_gap stayed +inf, the escape check was always "join,
        # never spawn", and one bootstrap well accidentally grew into a
        # mega-attractor absorbing 90%+ of early corpus. cached on
        # n_wells since that's what drives re-computation now.
        #
        # median_count stays tied to WELL_ESTABLISHED — it's a maturity
        # reference for the escape_mult scaling, a genuinely different
        # signal from the gap scale.
        self._typical_gap = float('inf')
        self._median_count = 1.0
        self._gap_cached_n_wells = -1

        # per-stage timing (for telemetry; updated each absorb_from_sums
        # call). stage names match the five phases of the hot path:
        # upload (obs_centroid), gap (typical_gap recompute), compute
        # (route kernel), readback, apply (CPU-side well mutation).
        self.last_stage_ms: dict = {k: 0.0 for k in
            ("upload", "gap", "compute", "readback", "apply")}

        # initial sync from ef
        self._sync_state_from_ef()

    # ── GPU buffer helpers ─────────────────────────────────────────

    def _mkuniform(self):
        return self.device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

    # routing feature width in f32 per pair: K.re, K.im, G.re, G.im, rms.
    # this is the FULL AR(2)-state-plus-residual signature of a well or
    # article. routing on K alone collapses by LLN on long articles
    # (per-article K-centroid → population mean); the (K, G, rms) tuple
    # is per-pair higher-dimensional and doesn't suffer the same fate.
    FEATURE_PER_PAIR = 5

    def _row_bytes(self) -> int:
        return self._well_cap * self.FEATURE_PER_PAIR * self.P * 4

    def _pack_well_row(self, name: str) -> np.ndarray:
        """build the 5*P f32 routing-feature row for a well.

        per-pair layout: [K.re, K.im, G.re, G.im, rms].
        ROUTING USES THE FIXED ANCHOR, NOT THE RUNNING CENTROID. anchors
        are pinned at the spawning article's K/G/rms and never updated;
        this prevents the centroid from drifting to the population mean
        as the well absorbs articles, which would otherwise let one
        well sit at the data's center-of-mass and absorb every future
        article (the geometric mega-well failure observed at 5M+
        articles).

        anchors fall back to the running centroid for wells loaded
        from format v6 brains (no anchor field) — best effort, but
        fresh ingests on format v7 have anchors recorded at well birth.
        """
        P = self.P
        anchor_K = self.ef._well_anchor_K.get(name)
        if anchor_K is not None:
            K = anchor_K.astype(np.complex128)
            G = self.ef._well_anchor_G.get(
                name, np.zeros(P, dtype=np.complex128)
            ).astype(np.complex128)
            S = self.ef._well_anchor_S.get(
                name, np.zeros(P, dtype=np.float64)
            ).astype(np.float64)
        else:
            # v6 fallback: use running centroid as the anchor proxy.
            cnt = max(int(self.ef._well_count.get(name, 0)), 1)
            K = (self.ef._well_sum_K[name] / cnt).astype(np.complex128)
            G_sum = self.ef._well_sum_G.get(name)
            G = ((G_sum / cnt).astype(np.complex128)
                 if G_sum is not None else np.zeros(P, dtype=np.complex128))
            S_sum = self.ef._well_sum_S.get(name)
            S = ((S_sum / cnt).astype(np.float64)
                 if S_sum is not None else np.zeros(P, dtype=np.float64))
        out = np.empty(self.FEATURE_PER_PAIR * P, dtype=np.float32)
        out[0::5] = K.real.astype(np.float32)
        out[1::5] = K.imag.astype(np.float32)
        out[2::5] = G.real.astype(np.float32)
        out[3::5] = G.imag.astype(np.float32)
        out[4::5] = S.astype(np.float32)
        return out

    def _pack_obs_rows(self, K_sums, G_sums, rms_sums,
                       n_segs_safe) -> np.ndarray:
        """build the [n_art, 5*P] f32 routing-feature matrix for a batch.

        same per-pair layout as _pack_well_row: per article the feature
        is (K, G, rms) all divided by n_segs (mean-per-segment).
        """
        n_art = K_sums.shape[0]
        P = self.P
        K_mean = K_sums / n_segs_safe[:, None]      # complex64 [n_art, P]
        G_mean = G_sums / n_segs_safe[:, None]      # complex64 [n_art, P]
        S_mean = rms_sums / n_segs_safe[:, None]    # f32 [n_art, P]
        out = np.empty((n_art, self.FEATURE_PER_PAIR * P), dtype=np.float32)
        out[:, 0::5] = K_mean.real
        out[:, 1::5] = K_mean.imag
        out[:, 2::5] = G_mean.real
        out[:, 3::5] = G_mean.imag
        out[:, 4::5] = S_mean
        return np.ascontiguousarray(out)

    def _alloc_well_state(self):
        centroid_bytes = (self._well_cap * self.FEATURE_PER_PAIR
                          * self.P * 4)
        count_bytes = self._well_cap * 4
        self._centroid_buf = self.device.create_buffer(
            size=centroid_bytes,
            usage=(wgpu.BufferUsage.STORAGE
                   | wgpu.BufferUsage.COPY_DST
                   | wgpu.BufferUsage.COPY_SRC),
        )
        self._count_buf = self.device.create_buffer(
            size=count_bytes,
            usage=(wgpu.BufferUsage.STORAGE
                   | wgpu.BufferUsage.COPY_DST
                   | wgpu.BufferUsage.COPY_SRC),
        )
        self._nn_dists_buf = self.device.create_buffer(
            size=self._well_cap * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        self._nn_dists_rb = self.device.create_buffer(
            size=self._well_cap * 4,
            usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
        )

    def _grow_well_cap(self, needed: int) -> None:
        """double (or more) and rebuild. called when the brain adds
        enough new wells that it outgrows its GPU capacity."""
        new_cap = max(needed, self._well_cap * 2)
        self._well_cap = new_cap
        self._alloc_well_state()
        # re-upload every well — the dict state is authoritative.
        self._reupload_all_wells()

    def _build_pipelines(self):
        def _read(name):
            return (SHADER_DIR / name).read_text(encoding="utf-8")

        # gap
        module = self.device.create_shader_module(
            code=_read("absorb_gap.wgsl"))
        self._gap_layout = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ])
        self._gap_pipe = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(
                bind_group_layouts=[self._gap_layout]),
            compute={"module": module, "entry_point": "main"},
        )

        # route
        module = self.device.create_shader_module(
            code=_read("absorb_route.wgsl"))
        self._route_layout = self.device.create_bind_group_layout(entries=[
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
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
            {"binding": 7, "visibility": wgpu.ShaderStage.COMPUTE,
             "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ])
        self._route_pipe = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(
                bind_group_layouts=[self._route_layout]),
            compute={"module": module, "entry_point": "main"},
        )

    def _ensure_scratch(self, n_art: int, _unused: int = 0):
        # _unused kept positional-compatible with the legacy two-arg
        # call pattern (absorb_batch passed total_segs); absorb_from_sums
        # only needs per-article capacity.
        if n_art > self._obs_cap:
            self._obs_cap = max(n_art * 2, 64)
            cap = self._obs_cap
            self._obs_centroid_buf = self.device.create_buffer(
                size=cap * self.FEATURE_PER_PAIR * self.P * 4,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
            )
            self._route_idx_buf = self.device.create_buffer(
                size=cap * 4,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
            )
            self._route_dist_buf = self.device.create_buffer(
                size=cap * 4,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
            )
            self._route_new_buf = self.device.create_buffer(
                size=cap * 4,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
            )
            self._route_rb = self.device.create_buffer(
                size=cap * 4 * 3,
                usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
            )

    # ── initial / resync state from ef ─────────────────────────────

    def _sync_state_from_ef(self):
        wells = self.ef.wells()
        if len(wells) > self._well_cap:
            self._grow_well_cap(len(wells) * 2)
        self._name_to_row = {}
        self._row_to_name = []
        P = self.P
        feat_w = self.FEATURE_PER_PAIR * P
        centroids_f32 = np.zeros((self._well_cap, feat_w), dtype=np.float32)
        counts_u32 = np.zeros(self._well_cap, dtype=np.uint32)
        for i, (name, _) in enumerate(wells.items()):
            centroids_f32[i] = self._pack_well_row(name)
            counts_u32[i] = int(self.ef._well_count.get(name, 0))
            self._name_to_row[name] = i
            self._row_to_name.append(name)
        self.device.queue.write_buffer(
            self._centroid_buf, 0, memoryview(centroids_f32))
        self.device.queue.write_buffer(
            self._count_buf, 0, memoryview(counts_u32))
        self._n_wells = len(wells)
        self._gap_cached_n_est = -1  # force recompute next call

    def _reupload_all_wells(self):
        # same as _sync_state_from_ef but rebuilds against freshly
        # allocated buffers (_grow_well_cap just ran).
        self._sync_state_from_ef()

    # ── hot path ───────────────────────────────────────────────────

    def absorb_from_sums(
        self,
        K_sums: np.ndarray,       # [n_art, P] complex64
        G_sums: np.ndarray,       # [n_art, P] complex64
        rms_sums: np.ndarray,     # [n_art, P] f32
        n_segs_per_art: np.ndarray,  # [n_art] u32/i32
        source: Optional[str] = None,
    ) -> List[str]:
        """absorb one batch through the GPU router.

        takes per-article K/G/rms sums from the unified pipeline's
        per_article_reduce kernel — CPU never sees raw per-segment
        K/G stacks. mutates self.ef in place (well sums, counts,
        dream buffer, total_absorbed). returns one well name per
        article, ordered to match the input.
        """
        import time as _t
        stage = self.last_stage_ms
        for k in stage:
            stage[k] = 0.0

        n_art = int(K_sums.shape[0])
        if n_art == 0:
            return []
        P = self.P

        # cheap check: coerce dtypes and compute per-article centroids
        _t0 = _t.perf_counter()
        if K_sums.dtype != np.complex64:
            K_sums = K_sums.astype(np.complex64, copy=False)
        if G_sums.dtype != np.complex64:
            G_sums = G_sums.astype(np.complex64, copy=False)
        if rms_sums.dtype != np.float32:
            rms_sums = rms_sums.astype(np.float32, copy=False)
        n_segs_arr = np.asarray(n_segs_per_art, dtype=np.int64)
        n_segs_safe = np.maximum(n_segs_arr, 1).astype(np.float32)
        # K-only centroid: still tracked separately for new-well init,
        # dream buffers, and the in-batch CPU rewire path which reasons
        # about K-centroid distance directly. routing uses the FULL
        # 5*P feature row (K, G, rms) — see _pack_obs_rows.
        centroids = (K_sums / n_segs_safe[:, None]).astype(np.complex64)
        feat_rows = self._pack_obs_rows(
            K_sums, G_sums, rms_sums, n_segs_safe)
        self._ensure_scratch(n_art, 1)  # only n_art matters here
        self.device.queue.write_buffer(
            self._obs_centroid_buf, 0, memoryview(feat_rows))
        stage["upload"] = (_t.perf_counter() - _t0) * 1000.0

        nonempty_mask = n_segs_arr > 0

        # bootstrap pads the brain to BOOTSTRAP_WELLS distinct anchored
        # wells before falling through to argmin routing. matches the
        # CPU absorb_flat path's `< BOOTSTRAP_WELLS` condition so the
        # two routes produce bit-equivalent well sets on identical
        # inputs. for resumed brains with > BOOTSTRAP_WELLS wells, this
        # is a no-op; for resumed brains with < BOOTSTRAP_WELLS, it
        # tops them off (rare and benign).
        if self._n_wells < self.BOOTSTRAP_WELLS:
            return self._bootstrap_from_sums(
                K_sums, G_sums, rms_sums, n_segs_arr, centroids)

        # typical_gap scale. recompute whenever the well count grows —
        # a new well changes the pairwise-distance distribution. we
        # deliberately include ALL wells (count >= 1) so the scale is
        # defined from the first post-bootstrap batch onward; see the
        # __init__ comment for the bootstrap-bug rationale. median_count
        # still tracks only WELL_ESTABLISHED wells — that's the router's
        # maturity reference, a distinct signal from the gap scale.
        # gap-recompute throttle. the gap kernel is W² in well count and
        # 5*P in feature width — at W=3000 it does ~7 billion memory
        # reads per call, saturating iGPU memory bandwidth (which is
        # shared with the rest of the system, so the WHOLE computer
        # stalls). without throttling it fires every batch since wells
        # grow ~1/batch on real corpora.
        #
        # rule: recompute only when well count grew by max(50, 5%) since
        # the last cached gap. typical_gap is a ROBUST median statistic
        # (not sensitive to small changes in W), so a 5% margin is
        # within its inherent noise. cost amortizes to O(1/N) per batch
        # at large N — at W=10k a recompute every 500 new wells is
        # ~once per 500-batch run, basically free.
        _t0 = _t.perf_counter()
        if self._n_wells >= 2:
            cached_W = self._gap_cached_n_wells
            grew_by = self._n_wells - max(cached_W, 0)
            threshold = max(50, max(cached_W, 1) // 20)
            if cached_W < 0 or grew_by >= threshold:
                self._recompute_typical_gap()
        else:
            self._typical_gap = float('inf')
            self._median_count = 1.0
            self._gap_cached_n_wells = -1
        stage["gap"] = (_t.perf_counter() - _t0) * 1000.0

        # route kernel (centroid kernel is skipped — we pre-computed
        # centroids on CPU and uploaded them directly to obs_centroid_buf)
        _t0 = _t.perf_counter()
        self._submit_route_only(n_art)
        stage["compute"] = (_t.perf_counter() - _t0) * 1000.0

        _t0 = _t.perf_counter()
        idx_arr, new_arr = self._read_route_outputs(n_art)
        stage["readback"] = (_t.perf_counter() - _t0) * 1000.0

        # apply — same semantics as legacy, but K_sums is already
        # per-article so no reduceat needed.
        _t0 = _t.perf_counter()
        names = self._apply_from_sums(
            n_art, idx_arr, new_arr,
            K_sums, G_sums, rms_sums, n_segs_arr, centroids,
            nonempty_mask)
        stage["apply"] = (_t.perf_counter() - _t0) * 1000.0
        return names

    def _submit_route_only(self, n_art: int):
        """route kernel without the centroid kernel — centroids are
        pre-uploaded by the caller into obs_centroid_buf.
        """
        P = self.P
        uparams = np.array(
            [n_art, self._n_wells, P, WELL_ESTABLISHED], dtype=np.uint32)
        fparams = np.array(
            [self._typical_gap, self._median_count,
             float(self.escape_energy), 0.0],
            dtype=np.float32,
        )
        self.device.queue.write_buffer(
            self._params_route_u, 0, memoryview(uparams))
        self.device.queue.write_buffer(
            self._params_route_f, 0, memoryview(fparams))

        bg = self.device.create_bind_group(
            layout=self._route_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self._obs_centroid_buf,
                    "offset": 0, "size": self._obs_centroid_buf.size}},
                {"binding": 1, "resource": {"buffer": self._centroid_buf,
                    "offset": 0, "size": self._centroid_buf.size}},
                {"binding": 2, "resource": {"buffer": self._count_buf,
                    "offset": 0, "size": self._count_buf.size}},
                {"binding": 3, "resource": {"buffer": self._route_idx_buf,
                    "offset": 0, "size": self._route_idx_buf.size}},
                {"binding": 4, "resource": {"buffer": self._route_dist_buf,
                    "offset": 0, "size": self._route_dist_buf.size}},
                {"binding": 5, "resource": {"buffer": self._route_new_buf,
                    "offset": 0, "size": self._route_new_buf.size}},
                {"binding": 6, "resource": {"buffer": self._params_route_u,
                    "offset": 0, "size": 16}},
                {"binding": 7, "resource": {"buffer": self._params_route_f,
                    "offset": 0, "size": 16}},
            ],
        )
        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self._route_pipe)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups(n_art, 1, 1)
        cp.end()
        nab = n_art * 4
        enc.copy_buffer_to_buffer(self._route_idx_buf,  0, self._route_rb, 0,       nab)
        enc.copy_buffer_to_buffer(self._route_dist_buf, 0, self._route_rb, nab,     nab)
        enc.copy_buffer_to_buffer(self._route_new_buf,  0, self._route_rb, nab * 2, nab)
        self.device.queue.submit([enc.finish()])

    # imported from engram_codec so CPU absorb_flat and GPU Brain agree.
    BOOTSTRAP_WELLS = BOOTSTRAP_WELLS

    def _bootstrap_from_sums(self, K_sums, G_sums, rms_sums,
                              n_segs_arr, centroids) -> List[str]:
        """create up to BOOTSTRAP_WELLS distinct wells from the first
        non-empty articles, each anchored at its spawning article's
        feature vector. anchors don't drift — they pin the well's
        routing position so the centroid-of-cluster collapse is
        impossible. once we hit BOOTSTRAP_WELLS or run out of articles
        in this batch, falls through to the normal route path.
        """
        P = self.P
        n_art = K_sums.shape[0]
        names: List[str] = []
        for a in range(n_art):
            if self._n_wells >= self.BOOTSTRAP_WELLS:
                # recurse on remaining articles using the main path
                remaining_names = self.absorb_from_sums(
                    K_sums[a:], G_sums[a:], rms_sums[a:], n_segs_arr[a:])
                names.extend(remaining_names)
                return names
            n_segs = int(n_segs_arr[a])
            if n_segs == 0:
                self.ef._total_absorbed += 1
                names.append("unknown")
                continue
            name = f"well_{self.ef._next_well_id}"
            self.ef._next_well_id += 1
            self.ef._well_sum_K[name] = np.zeros(P, dtype=np.complex128)
            self.ef._well_sum_G[name] = np.zeros(P, dtype=np.complex128)
            self.ef._well_sum_S[name] = np.zeros(P, dtype=np.float64)
            self.ef._well_count[name] = 0
            self.ef._well_sum_K[name] += K_sums[a].astype(np.complex128)
            self.ef._well_sum_G[name] += G_sums[a].astype(np.complex128)
            self.ef._well_sum_S[name] += rms_sums[a].astype(np.float64)
            self.ef._well_count[name] += n_segs
            # anchor: pin routing position to THIS article's mean K/G/S.
            # never updated — the well's identity in routing space is
            # whatever spawned it. centroid (sum_X/count) still tracks
            # for analysis, dream, naming.
            self.ef._well_anchor_K[name] = centroids[a].astype(np.complex128)
            self.ef._well_anchor_G[name] = (
                G_sums[a] / max(n_segs, 1)).astype(np.complex128)
            self.ef._well_anchor_S[name] = (
                rms_sums[a] / max(n_segs, 1)).astype(np.float64)
            self.ef._dream_Ks.append(centroids[a].astype(np.complex128))
            self.ef._dream_Gs.append((G_sums[a] / max(n_segs, 1)).astype(np.complex128))
            self.ef._dream_Ss.append((rms_sums[a] / max(n_segs, 1)).astype(np.float64))
            self.ef._total_absorbed += 1
            row = self._n_wells
            if row >= self._well_cap:
                self._grow_well_cap(row + 1)
            self._name_to_row[name] = row
            self._row_to_name.append(name)
            self._n_wells += 1
            self._upload_one_well(row, name, n_segs)
            names.append(name)
        self.ef._invalidate_caches()
        self._trim_dream()
        return names

    def _apply_from_sums(self, n_art, idx_arr, new_arr,
                          K_sums, G_sums, rms_sums, n_segs_arr,
                          centroids, nonempty_mask) -> List[str]:
        """apply updates from pre-aggregated per-article sums. no
        reduceat needed; this is ~5× faster than _apply_updates.
        """
        P = self.P
        names: List[str] = []

        # in-batch new-well rewire: same logic as legacy, matching host
        # sequential semantics. O(M²) where M = candidates.
        # distance uses the FULL routing feature (K, G, rms) — same
        # metric the GPU route shader uses, so in-batch matching agrees
        # with what the next batch's argmin would find.
        batch_new_wells: list = []
        typical_gap = self._typical_gap
        median_count = self._median_count
        # n_segs_safe was computed for the obs upload; recompute locally
        # to keep this function decoupled from absorb_from_sums internals.
        _n_segs_safe = np.maximum(n_segs_arr, 1).astype(np.float32)
        feat_rows = self._pack_obs_rows(
            K_sums, G_sums, rms_sums, _n_segs_safe)  # [n_art, 5*P]
        feat_w = self.FEATURE_PER_PAIR * P
        for a in range(n_art):
            if not nonempty_mask[a] or not new_arr[a]:
                continue
            c = centroids[a]
            f = feat_rows[a]
            cnt = int(n_segs_arr[a])
            best_row = -1
            best_d = float('inf')
            for (nw_row, nw_c, nw_f, nw_cnt) in batch_new_wells:
                # mean per-pair squared diff over all 5 features (K.re,
                # K.im, G.re, G.im, rms), divided by P to match the
                # route shader's d2 scaling exactly.
                diff = f - nw_f
                d2 = float(np.dot(diff, diff)) / P
                d = np.sqrt(max(d2, 0.0))
                # match the WGSL router: cap the depth-ratio at 1.0 so
                # mature wells don't get runaway-stickiness basins. see
                # absorb_route.wgsl for the rationale.
                ratio_raw = (np.log1p(nw_cnt)
                             / max(np.log1p(median_count), 1e-20))
                escape_mult = self.escape_energy * np.sqrt(
                    min(ratio_raw, 1.0))
                if d <= typical_gap * escape_mult and d < best_d:
                    best_d = d
                    best_row = nw_row
            if best_row >= 0:
                new_arr[a] = 0
                idx_arr[a] = best_row
            else:
                batch_new_wells.append(
                    (self._n_wells + len(batch_new_wells), c, f, cnt))

        dirty: set = set()
        grew = False
        for a in range(n_art):
            if not nonempty_mask[a]:
                self.ef._total_absorbed += 1
                names.append("unknown")
                continue

            if new_arr[a]:
                name = f"well_{self.ef._next_well_id}"
                self.ef._next_well_id += 1
                self.ef._well_sum_K[name] = np.zeros(P, dtype=np.complex128)
                self.ef._well_sum_G[name] = np.zeros(P, dtype=np.complex128)
                self.ef._well_sum_S[name] = np.zeros(P, dtype=np.float64)
                self.ef._well_count[name] = 0
                # anchor: pin routing position to THIS spawning article's
                # K/G/S. fixed for the well's lifetime — see _bootstrap_
                # from_sums and _pack_well_row for rationale.
                n_segs_a = max(int(n_segs_arr[a]), 1)
                self.ef._well_anchor_K[name] = (
                    centroids[a].astype(np.complex128))
                self.ef._well_anchor_G[name] = (
                    G_sums[a] / n_segs_a).astype(np.complex128)
                self.ef._well_anchor_S[name] = (
                    rms_sums[a] / n_segs_a).astype(np.float64)
                row = self._n_wells
                if row >= self._well_cap:
                    self._grow_well_cap(row + 1)
                    grew = True
                self._name_to_row[name] = row
                self._row_to_name.append(name)
                self._n_wells += 1
            else:
                row = int(idx_arr[a])
                name = self._row_to_name[row]
                # defensive: a well loaded from a v4 brain doesn't have
                # sum_G / sum_S entries. initialize them on first write
                # so older brains transparently upgrade after one more
                # absorb pass.
                if name not in self.ef._well_sum_G:
                    self.ef._well_sum_G[name] = np.zeros(P, dtype=np.complex128)
                if name not in self.ef._well_sum_S:
                    self.ef._well_sum_S[name] = np.zeros(P, dtype=np.float64)

            # core update — accumulate the full AR(2) state per well.
            # K alone gave us a centroid; K+G gives us the oscillator
            # eigenvalues (λ² - Kλ + G = 0), so each well's attractor
            # character is now described by its actual dynamical
            # parameters, not a proxy.
            self.ef._well_sum_K[name] = (
                self.ef._well_sum_K[name] + K_sums[a].astype(np.complex128))
            self.ef._well_sum_G[name] = (
                self.ef._well_sum_G[name] + G_sums[a].astype(np.complex128))
            self.ef._well_sum_S[name] = (
                self.ef._well_sum_S[name] + rms_sums[a].astype(np.float64))
            self.ef._well_count[name] = (
                self.ef._well_count[name] + int(n_segs_arr[a]))
            dirty.add(row)

            n_segs_a = max(int(n_segs_arr[a]), 1)

            # per-well dream buffer: append this article's K-centroid
            # to the dream history of the well it routed to. ring-
            # buffer capped at ef.WELL_DREAM_CAP; older entries drop.
            # this is what lets each well fit its OWN meta-oscillator
            # and predict its next incoming article locally — rather
            # than all wells sharing the brain-wide dream prediction.
            well_dream = self.ef._well_dream_Ks.setdefault(name, [])
            well_dream.append(centroids[a].astype(np.complex128))
            if len(well_dream) > self.ef.WELL_DREAM_CAP:
                # trim the oldest; keep the rolling tail. slice-assign
                # is cheap for a list of complex128 arrays at the
                # sizes we care about (a few hundred entries max).
                self.ef._well_dream_Ks[name] = well_dream[
                    -self.ef.WELL_DREAM_CAP:]

            self.ef._dream_Ks.append(centroids[a].astype(np.complex128))
            self.ef._dream_Gs.append(
                (G_sums[a] / n_segs_a).astype(np.complex128))
            self.ef._dream_Ss.append(
                (rms_sums[a] / n_segs_a).astype(np.float64))
            self.ef._total_absorbed += 1
            names.append(name)

        if dirty and not grew:
            self._flush_dirty_rows(sorted(dirty))

        self.ef._invalidate_caches()
        self._trim_dream()
        return names

    # ── internals ──────────────────────────────────────────────────

    @staticmethod
    def _complex_interleave(z: np.ndarray) -> np.ndarray:
        """[..., P] complex -> [..., 2*P] float32 (re, im, re, im, ...)."""
        if z.dtype == np.complex64:
            return z.view(np.float32).reshape(*z.shape[:-1], 2 * z.shape[-1])
        z64 = z.astype(np.complex64, copy=False)
        return z64.view(np.float32).reshape(*z64.shape[:-1], 2 * z64.shape[-1])

    def _count_established(self) -> int:
        """number of wells that count as 'established' under the
        codec's threshold. threshold is owned by engram_codec, not
        this module."""
        return sum(
            1 for n in self._row_to_name
            if self.ef._well_count.get(n, 0) >= WELL_ESTABLISHED
        )

    # cap on workgroups dispatched + inner-loop scan length in the
    # gap kernel. typical_gap is a robust median; sampling 256 wells
    # out of 6k+ gives a < 1% bias estimate in 5*P-dim feature space.
    # the cap keeps the kernel under Windows' 2-second TDR threshold:
    # 256² × 5*150 ≈ 50M ops × 4 bytes = 200 MB memory traffic, ~10ms
    # on iGPU. without it, at W=6000 the kernel does 100+ GB of
    # bandwidth and crashes with "Parent device is lost".
    GAP_SAMPLE_MAX = 256

    def _recompute_typical_gap(self) -> None:
        # at large W we sample wells at a stride to bound kernel cost.
        # shader filters participating wells by count >= params.z (we
        # pass 1, so every non-empty sampled well contributes). the
        # median_count below still uses WELL_ESTABLISHED — the two
        # statistics have different jobs.
        W = self._n_wells
        if W <= self.GAP_SAMPLE_MAX:
            stride = 1
            n_active = W
        else:
            stride = (W + self.GAP_SAMPLE_MAX - 1) // self.GAP_SAMPLE_MAX
            n_active = (W + stride - 1) // stride
        params = np.array(
            [W, self.P, 1, stride], dtype=np.uint32)
        self.device.queue.write_buffer(
            self._params_gap, 0, memoryview(params))
        bg = self.device.create_bind_group(
            layout=self._gap_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self._centroid_buf,
                    "offset": 0, "size": self._centroid_buf.size}},
                {"binding": 1, "resource": {"buffer": self._count_buf,
                    "offset": 0, "size": self._count_buf.size}},
                {"binding": 2, "resource": {"buffer": self._nn_dists_buf,
                    "offset": 0, "size": self._nn_dists_buf.size}},
                {"binding": 3, "resource": {"buffer": self._params_gap,
                    "offset": 0, "size": 16}},
            ],
        )
        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self._gap_pipe)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups(n_active, 1, 1)
        cp.end()
        nbytes = n_active * 4
        enc.copy_buffer_to_buffer(
            self._nn_dists_buf, 0, self._nn_dists_rb, 0, nbytes)
        self.device.queue.submit([enc.finish()])

        self._nn_dists_rb.map_async(
            wgpu.MapMode.READ, size=nbytes).sync_wait()
        nn_dists = np.frombuffer(
            self._nn_dists_rb.read_mapped(size=nbytes),
            dtype=np.float32,
        ).copy()
        self._nn_dists_rb.unmap()

        # the shader emits f32_max as a "no valid distance" sentinel
        # for unestablished wells — WGSL f32 has no true infinity.
        # isfinite() returns True for that value, so we must filter it
        # explicitly. use np.finfo(float32).max directly as the
        # reference, not a magic 1e30; threshold at 0.5x of it so any
        # near-sentinel noise from shader arithmetic also gets
        # filtered. upcast to f64 before median so averaging two
        # sentinel rows can't overflow the f32 accumulator.
        f32_max = float(np.finfo(np.float32).max)
        finite = nn_dists[nn_dists < f32_max * 0.5]
        if finite.size:
            self._typical_gap = float(np.median(finite.astype(np.float64)))
        else:
            self._typical_gap = 1.0

        est_counts = [
            self.ef._well_count.get(n, 0)
            for n in self._row_to_name
            if self.ef._well_count.get(n, 0) >= WELL_ESTABLISHED
        ]
        self._median_count = (
            float(np.median(est_counts)) if est_counts else 1.0)
        self._gap_cached_n_wells = self._n_wells

    def _read_route_outputs(self, n_art: int):
        nab = n_art * 4
        self._route_rb.map_async(
            wgpu.MapMode.READ, size=nab * 3).sync_wait()
        raw = self._route_rb.read_mapped(size=nab * 3)
        flat = np.frombuffer(raw, dtype=np.uint8)
        # .copy() makes the views writable — the rewire pass in
        # _apply_from_sums mutates these arrays.
        idx_arr = np.frombuffer(flat[0:nab].tobytes(),
                                dtype=np.uint32).copy()
        new_arr = np.frombuffer(flat[nab * 2:nab * 3].tobytes(),
                                dtype=np.uint32).copy()
        self._route_rb.unmap()
        return idx_arr, new_arr


    # ── GPU state flushes ──────────────────────────────────────────

    def _upload_one_well(self, row: int, name: str, count: int):
        f = self._pack_well_row(name)
        stride = self.FEATURE_PER_PAIR * self.P * 4
        self.device.queue.write_buffer(
            self._centroid_buf, row * stride, memoryview(f))
        count_arr = np.array([count], dtype=np.uint32)
        self.device.queue.write_buffer(
            self._count_buf, row * 4, memoryview(count_arr))

    def _flush_dirty_rows(self, sorted_rows: list) -> None:
        """batch-upload updated well rows. groups contiguous row runs
        into one write_buffer each — for the common case of a few
        popular wells, this is ≤ 5 driver calls per batch."""
        P = self.P
        feat_w = self.FEATURE_PER_PAIR * P
        row_stride_bytes = feat_w * 4

        runs: list = []
        run_start = sorted_rows[0]
        run_len = 1
        for r in sorted_rows[1:]:
            if r == run_start + run_len:
                run_len += 1
            else:
                runs.append((run_start, run_len))
                run_start = r
                run_len = 1
        runs.append((run_start, run_len))

        for start, length in runs:
            centroid_buf = np.empty((length, feat_w), dtype=np.float32)
            count_buf = np.empty(length, dtype=np.uint32)
            for i in range(length):
                row = start + i
                name = self._row_to_name[row]
                centroid_buf[i] = self._pack_well_row(name)
                count_buf[i] = int(self.ef._well_count[name])
            self.device.queue.write_buffer(
                self._centroid_buf,
                start * row_stride_bytes,
                memoryview(centroid_buf),
            )
            self.device.queue.write_buffer(
                self._count_buf, start * 4, memoryview(count_buf),
            )

    def _trim_dream(self):
        cap = self.ef.dim * self.ef.DREAM_BUFFER_CAPACITY_PER_DIM
        if len(self.ef._dream_Ks) > cap:
            excess = len(self.ef._dream_Ks) - cap
            self.ef._dream_Ks = self.ef._dream_Ks[excess:]
            self.ef._dream_Gs = self.ef._dream_Gs[excess:]
            self.ef._dream_Ss = self.ef._dream_Ss[excess:]
