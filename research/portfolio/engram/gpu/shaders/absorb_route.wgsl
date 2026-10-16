// absorb_route.wgsl — parallel per-article nearest-well argmin plus
// escape decision. one workgroup per observation, 64 threads cooperate
// on the argmin scan across all W wells.
//
// routing feature (FEATURE_PER_PAIR=5 floats per pair, P pairs per row):
//     [K.re, K.im, G.re, G.im, rms]
// well rows are FIXED ANCHORS — the K/G/rms of the article that spawned
// each well, frozen at creation. with running centroids, sum_K/count
// drifts to the population mean as a well absorbs articles, and
// eventually one well sits at the cluster's geometric center where
// it's closest to every future article (mega-well failure observed
// at 5M+ articles). anchors don't drift, so wells partition K-space
// by their bootstrap positions.
//
// raw-distance argmin (no mass weighting):
//     d_raw[w]  = sqrt(mean(d2_per_pair))   -- argmin by d_raw
// historical implementations divided d_raw by log(1+count), letting
// "massive wells pull harder." with anchors that's strictly harmful:
// it lets an old well attract articles that are 27× closer to a
// younger well, defeating the whole point of fixed anchors.
//
// escape rule (matches _route_observation on host):
//     escape_mult = ESCAPE_ENERGY * sqrt(min(log1p(nn_c)/log1p(med_c), 1.0))
//     if d_raw > typical_gap * escape_mult -> spawn a new well
// typical_gap and median_count come from the host as f32 uniforms.
//
// outputs per article:
//     nearest_idx[a] : u32   — index into the well array
//     dist[a]        : f32   — raw rms distance to nearest well
//     is_new[a]      : u32   — 0 join nearest, 1 spawn new

const WG_SIZE : u32 = 64u;
const INF : f32 = 3.4028235e38;
const FEATURE_PER_PAIR : u32 = 5u;

@group(0) @binding(0) var<storage, read>  obs_centroids  : array<f32>;  // [n_art, 5*P]
@group(0) @binding(1) var<storage, read>  well_centroids : array<f32>;  // [W, 5*P]
@group(0) @binding(2) var<storage, read>  well_count     : array<u32>;  // [W]
@group(0) @binding(3) var<storage, read_write> nearest_idx : array<u32>;  // [n_art]
@group(0) @binding(4) var<storage, read_write> dist_out    : array<f32>;  // [n_art]
@group(0) @binding(5) var<storage, read_write> is_new      : array<u32>;  // [n_art]
@group(0) @binding(6) var<uniform> uparams : vec4<u32>;
//   (n_art, W, P, WELL_ESTABLISHED_unused_but_reserved)
@group(0) @binding(7) var<uniform> fparams : vec4<f32>;
//   (typical_gap, median_count, escape_energy, 0)

var<workgroup> wg_d2    : array<f32, WG_SIZE>;
var<workgroup> wg_idx   : array<u32, WG_SIZE>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    let tid = lid.x;
    let n_art = uparams.x;
    let W = uparams.y;
    let P = uparams.z;
    let a = wg.x;
    if (a >= n_art) { return; }

    let typical_gap = fparams.x;
    let median_count = fparams.y;
    let escape_energy = fparams.z;
    let obs_base = a * FEATURE_PER_PAIR * P;

    var best_d2   : f32 = INF;
    var best_w    : u32 = 0u;

    for (var w : u32 = tid; w < W; w = w + WG_SIZE) {
        let wc = well_count[w];
        if (wc == 0u) { continue; }
        let w_base = w * FEATURE_PER_PAIR * P;
        var sum_sq : f32 = 0.0;
        for (var k : u32 = 0u; k < P; k = k + 1u) {
            let pair_off = FEATURE_PER_PAIR * k;
            let dKr = obs_centroids[obs_base + pair_off + 0u]
                    - well_centroids[w_base + pair_off + 0u];
            let dKi = obs_centroids[obs_base + pair_off + 1u]
                    - well_centroids[w_base + pair_off + 1u];
            let dGr = obs_centroids[obs_base + pair_off + 2u]
                    - well_centroids[w_base + pair_off + 2u];
            let dGi = obs_centroids[obs_base + pair_off + 3u]
                    - well_centroids[w_base + pair_off + 3u];
            let dS  = obs_centroids[obs_base + pair_off + 4u]
                    - well_centroids[w_base + pair_off + 4u];
            sum_sq = sum_sq + dKr * dKr + dKi * dKi
                            + dGr * dGr + dGi * dGi
                            + dS * dS;
        }
        // mean over P pairs. note this divides total by P, not by
        // P*FEATURE_PER_PAIR — keeps the unit comparable to the old
        // K-only path (which also divided by P), so typical_gap
        // calibration carries over.
        let d2 = sum_sq / f32(P);
        if (d2 < best_d2) {
            best_d2 = d2;
            best_w = w;
        }
    }

    wg_d2[tid]   = best_d2;
    wg_idx[tid]  = best_w;
    workgroupBarrier();

    var step : u32 = WG_SIZE >> 1u;
    loop {
        if (step == 0u) { break; }
        if (tid < step) {
            let other_d2 = wg_d2[tid + step];
            if (other_d2 < wg_d2[tid]) {
                wg_d2[tid]   = other_d2;
                wg_idx[tid]  = wg_idx[tid + step];
            }
        }
        workgroupBarrier();
        step = step >> 1u;
    }

    if (tid == 0u) {
        let nn_w = wg_idx[0];
        let nn_d = sqrt(wg_d2[0]);
        nearest_idx[a] = nn_w;
        dist_out[a]    = nn_d;

        // escape: compare raw distance to gap * escape_mult. host
        // has ensured typical_gap is finite before invoking this
        // kernel — it's recomputed as soon as >= 2 wells exist
        // (count >= 1), not when >= 2 wells are mature.
        let nn_c = well_count[nn_w];
        let denom = max(log(1.0 + median_count), 1e-20);
        // ratio captures relative well depth: fresh wells (nn_c <
        // median_count) get scaled-DOWN escape mult so members can
        // leave; mature wells used to get scaled-UP mult, which gave
        // them runaway basins (each absorption made the basin grow,
        // attracting yet more mass — observed as 99%+ concentration in
        // a single mega-well on real corpora). cap the ratio at 1.0
        // so mature wells get the BASE escape_mult, not extra
        // stickiness. fresh-well penalty is preserved.
        let ratio_raw = log(1.0 + f32(nn_c)) / denom;
        let ratio = min(ratio_raw, 1.0);
        let escape_mult = escape_energy * sqrt(ratio);
        if (nn_d > typical_gap * escape_mult) {
            is_new[a] = 1u;
        } else {
            is_new[a] = 0u;
        }
    }
}
