// absorb_gap.wgsl — per-well nearest-neighbor distance over a strided
// SAMPLE of the well population. distance is over the FULL routing
// feature (FEATURE_PER_PAIR=5 floats per pair: K.re, K.im, G.re, G.im,
// rms), matching absorb_route.wgsl exactly so typical_gap is a true
// scale estimate of the routing metric.
//
// dispatch: (n_active, 1, 1) — one workgroup per SAMPLED well. 64
// threads cooperate on the inner P-dim reduction. typical_gap is a
// robust median statistic; sampling 256 wells out of W gives an
// estimate within median sampling noise (5*P-dim space → bias < 1%
// for any reasonable W). this caps gap-kernel cost at ~50ms regardless
// of W, keeping the kernel well under Windows' 2-second TDR threshold.
//
// stride sampling = uniform coverage across well-id space. wells are
// spawned in ingest order so well_0 differs from well_3000; striding
// by W/n_active picks one well per epoch instead of clustering on
// early bootstrap wells.
//
// participating wells are those with count >= params.z. host passes
// 1 so every non-empty well contributes; outer wells below threshold
// emit +inf into the OUTPUT (host filter), inner wells below threshold
// are skipped when scanned.
//
// uniforms: (W_total, P, count_threshold, well_stride)

const WG_SIZE : u32 = 64u;
const INF : f32 = 3.4028235e38;
const FEATURE_PER_PAIR : u32 = 5u;

@group(0) @binding(0) var<storage, read> well_centroids : array<f32>;  // [W, 5*P]
@group(0) @binding(1) var<storage, read> well_count     : array<u32>;  // [W]
@group(0) @binding(2) var<storage, read_write> nn_dists : array<f32>;  // [n_active] output (packed)
@group(0) @binding(3) var<uniform> params : vec4<u32>;  // (W_total, P, count_threshold, well_stride)

var<workgroup> wg_partial : array<f32, WG_SIZE>;

fn wg_reduce(tid: u32) {
    var step : u32 = WG_SIZE >> 1u;
    loop {
        if (step == 0u) { break; }
        if (tid < step) {
            wg_partial[tid] = wg_partial[tid] + wg_partial[tid + step];
        }
        workgroupBarrier();
        step = step >> 1u;
    }
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    let tid = lid.x;
    let W = params.x;
    let P = params.y;
    let count_threshold = params.z;
    let stride = params.w;
    let a_dispatch = wg.x;
    let a = a_dispatch * stride;

    if (a >= W) {
        if (tid == 0u) { nn_dists[a_dispatch] = INF; }
        return;
    }

    let a_count = well_count[a];
    if (a_count < count_threshold) {
        if (tid == 0u) { nn_dists[a_dispatch] = INF; }
        return;
    }

    let a_base = a * FEATURE_PER_PAIR * P;

    // direct (a-b)·(a-b) per pair — 5 features per pair (K.re, K.im,
    // G.re, G.im, rms). we don't use the matmul-trick |a|² + |b|² -
    // 2·Re(a·b*) here since rms is real and adding a third asymmetric
    // term complicates the bookkeeping; the direct difference is just
    // as cheap with 5 features and stays bit-aligned with the route
    // shader (same per-pair ordering, same divisor by P).
    var best : f32 = INF;

    for (var b : u32 = 0u; b < W; b = b + stride) {
        if (b == a) { continue; }
        let b_count = well_count[b];
        if (b_count < count_threshold) { continue; }
        let b_base = b * FEATURE_PER_PAIR * P;

        var sum_term : f32 = 0.0;
        for (var k : u32 = tid; k < P; k = k + WG_SIZE) {
            let pair_off = FEATURE_PER_PAIR * k;
            let dKr = well_centroids[a_base + pair_off + 0u]
                    - well_centroids[b_base + pair_off + 0u];
            let dKi = well_centroids[a_base + pair_off + 1u]
                    - well_centroids[b_base + pair_off + 1u];
            let dGr = well_centroids[a_base + pair_off + 2u]
                    - well_centroids[b_base + pair_off + 2u];
            let dGi = well_centroids[a_base + pair_off + 3u]
                    - well_centroids[b_base + pair_off + 3u];
            let dS  = well_centroids[a_base + pair_off + 4u]
                    - well_centroids[b_base + pair_off + 4u];
            sum_term = sum_term
                + dKr * dKr + dKi * dKi
                + dGr * dGr + dGi * dGi
                + dS * dS;
        }
        wg_partial[tid] = sum_term;
        workgroupBarrier();
        wg_reduce(tid);
        if (tid == 0u) {
            let d2 = wg_partial[0] / f32(P);
            let d2_clamped = max(d2, 0.0);
            if (d2_clamped < best) { best = d2_clamped; }
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        if (best >= INF) {
            nn_dists[a_dispatch] = INF;
        } else {
            nn_dists[a_dispatch] = sqrt(best);
        }
    }
}
