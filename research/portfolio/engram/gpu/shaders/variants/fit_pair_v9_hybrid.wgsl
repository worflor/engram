// fit_pair_v9_hybrid.wgsl
//
// hybrid rms strategy:
//   - accumulate all five complex sums + sTT in a single DRAM pass
//     (like v5), then algebraically derive Σ|err|² in closed form
//   - check whether the algebraic residual is numerically safe. if the
//     residual is pathologically small relative to the signal (i.e. the
//     closed-form subtraction has catastrophically cancelled), fall
//     back to a second pointwise pass just for THAT pair
//   - in the common case (~98% of pairs from our error probe) the fit
//     isn't near the stability boundary and the algebraic rms is fine
//
// expected: close to v5 throughput when catastrophic cancellation is
// rare, gracefully degrades to v3-like on pathological workloads.

const WG_SIZE     : u32 = 64u;
const MACHINE_EPS : f32 = 1.1920929e-7;
const RIDGE_SCALE : f32 = 1.0e-4;

// cancellation detection threshold: if the algebraic |err|² is less
// than this fraction of |tgt|², the subtraction lost too many bits.
// empirical from v2 error probe: catastrophic cases had residual/sTT
// ratios below ~5e-4; at 1e-3 we catch them without over-flagging.
const CANCEL_RATIO : f32 = 1.0e-3;

struct Segment {
    traj_offset : u32,
    length      : u32,
    dim         : u32,
    out_offset  : u32,
};

@group(0) @binding(0) var<storage, read>       traj     : array<f32>;
@group(0) @binding(1) var<storage, read>       segments : array<Segment>;
@group(0) @binding(2) var<storage, read_write> K_out    : array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> G_out    : array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> rms_out  : array<f32>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
fn cconj(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x, -a.y);
}
fn cmul_conj(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
}
fn load_z(base: u32, t: u32, D: u32, p2: u32) -> vec2<f32> {
    let off = base + t * D + p2;
    return vec2<f32>(traj[off], traj[off + 1u]);
}

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    let seg_idx   = wg.x;
    let pair_base = wg.y * WG_SIZE;
    let pair_idx  = pair_base + lid.x;
    let seg = segments[seg_idx];
    let T   = seg.length;
    let D   = seg.dim;
    let P   = D / 2u;
    if (pair_idx >= P) { return; }

    let base = seg.traj_offset;
    let p2   = pair_idx * 2u;

    // ── single pass: five cross-sums + sTT ────────────────────────────

    var sAA : f32       = 0.0;
    var sBB : f32       = 0.0;
    var sTT : f32       = 0.0;
    var sAB : vec2<f32> = vec2<f32>(0.0, 0.0);
    var sTA : vec2<f32> = vec2<f32>(0.0, 0.0);
    var sTB : vec2<f32> = vec2<f32>(0.0, 0.0);

    if (T >= 3u) {
        var z_tm2 = load_z(base, 0u, D, p2);
        var z_tm1 = load_z(base, 1u, D, p2);
        var t : u32 = 2u;
        loop {
            if (t >= T) { break; }
            let z_t = load_z(base, t, D, p2);
            sAA = sAA + dot(z_tm1, z_tm1);
            sBB = sBB + dot(z_tm2, z_tm2);
            sTT = sTT + dot(z_t,   z_t);
            sAB = sAB + cmul_conj(z_tm1, z_tm2);
            sTA = sTA + cmul_conj(z_t,   z_tm1);
            sTB = sTB + cmul_conj(z_t,   z_tm2);
            z_tm2 = z_tm1;
            z_tm1 = z_t;
            t = t + 1u;
        }
    }

    // ── solve K, G ────────────────────────────────────────────────────

    let trace  = sAA + sBB;
    let ridge  = max(MACHINE_EPS, RIDGE_SCALE * trace * 0.5);
    let sAA_r  = sAA + ridge;
    let sBB_r  = sBB + ridge;
    let absAB2 = sAB.x * sAB.x + sAB.y * sAB.y;
    let det    = sAA_r * sBB_r - absAB2;
    let det_thresh = MACHINE_EPS * trace * trace;

    var K_v : vec2<f32> = vec2<f32>(2.0, 0.0);
    var G_v : vec2<f32> = vec2<f32>(1.0, 0.0);
    if (T >= 3u && abs(det) >= det_thresh) {
        K_v = (sTA * sBB_r - cmul(cconj(sAB), sTB)) / det;
        G_v = (cmul(sAB, sTA) - sTB * sAA_r) / det;
        let all_finite = all(vec4<f32>(K_v.x, K_v.y, G_v.x, G_v.y) ==
                             vec4<f32>(K_v.x, K_v.y, G_v.x, G_v.y));
        if (!all_finite) {
            K_v = vec2<f32>(2.0, 0.0);
            G_v = vec2<f32>(1.0, 0.0);
        }
    }

    // ── algebraic residual first (cheap, no extra DRAM) ───────────────
    //
    // Σ|err|² = sTT − 2·Re(K·conj(sTA) − G·conj(sTB))
    //         + |K|²·sAA + |G|²·sBB − 2·Re(K·conj(G)·sAB)

    let K_mag2 = K_v.x * K_v.x + K_v.y * K_v.y;
    let G_mag2 = G_v.x * G_v.x + G_v.y * G_v.y;
    let re_K_conjsTA  = K_v.x * sTA.x + K_v.y * sTA.y;
    let re_G_conjsTB  = G_v.x * sTB.x + G_v.y * sTB.y;
    let KconjG        = cmul(K_v, cconj(G_v));
    let re_KconjG_sAB = KconjG.x * sAB.x - KconjG.y * sAB.y;

    let alg_resid_sq = sTT
                     - 2.0 * (re_K_conjsTA - re_G_conjsTB)
                     + K_mag2 * sAA + G_mag2 * sBB
                     - 2.0 * re_KconjG_sAB;

    // ── cancellation detection ────────────────────────────────────────
    //
    // if the algebraic residual is either (a) negative (guaranteed
    // cancellation disaster because err² ≥ 0) or (b) unreasonably
    // small relative to sTT (signal magnitude), don't trust the
    // closed form. fall back to the direct pointwise computation.

    var sq_err : f32 = max(alg_resid_sq, 0.0);
    let suspicious = (alg_resid_sq < CANCEL_RATIO * sTT) || (alg_resid_sq < 0.0);

    if (suspicious && T >= 3u) {
        // fallback: pointwise pass on THIS pair only. the other threads
        // in the workgroup will skip this branch; warp divergence is
        // accepted because the pathological fraction is small.
        var acc : f32 = 0.0;
        var z_tm2 = load_z(base, 0u, D, p2);
        var z_tm1 = load_z(base, 1u, D, p2);
        var t : u32 = 2u;
        loop {
            if (t >= T) { break; }
            let z_t  = load_z(base, t, D, p2);
            let pred = cmul(K_v, z_tm1) - cmul(G_v, z_tm2);
            let err  = z_t - pred;
            acc = acc + dot(err, err);
            z_tm2 = z_tm1;
            z_tm1 = z_t;
            t = t + 1u;
        }
        sq_err = acc;
    }

    var denom : u32 = 1u;
    if (T >= 3u) { denom = T - 2u; }
    let rms_v = sqrt(max(sq_err, 0.0) / f32(denom));

    K_out[seg.out_offset + pair_idx]   = K_v;
    G_out[seg.out_offset + pair_idx]   = G_v;
    rms_out[seg.out_offset + pair_idx] = rms_v;
}
