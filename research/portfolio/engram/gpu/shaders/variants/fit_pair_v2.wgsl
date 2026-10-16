// fit_pair_v2.wgsl
//
// corrections over v1:
//   1. dispatch = (n_segments, 1, 1), workgroup = 64 threads.
//      each thread owns up to PAIRS_PER_THREAD pairs (stride-WG_SIZE
//      assignment). coalesced reads: adjacent threads read adjacent
//      complex slots at the same timestep, so one subgroup read spans
//      contiguous memory. no shared memory, no barriers, no tree
//      reduction.
//   2. residual rms computed algebraically from the same sums used
//      for the 2x2 solve. no second pass over the trajectory:
//          Σ|err|² = sTT
//                  - 2·Re(K·conj(sTA) - G·conj(sTB))
//                  + |K|²·sAA + |G|²·sBB
//                  - 2·Re(K·conj(G)·sAB)
//      only cost: one extra scalar (sTT) per pair in the accumulator.
//      this halves global memory traffic in the hot kernel.

const WG_SIZE          : u32 = 64u;
const PAIRS_PER_THREAD : u32 = 3u;
const MACHINE_EPS      : f32 = 1.1920929e-7;
const RIDGE_SCALE      : f32 = 1.0e-4;

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

// a * conj(b)
fn cmul_conj(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
}

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    let seg_idx = wg.x;
    let tid     = lid.x;
    let seg     = segments[seg_idx];
    let T       = seg.length;
    let base    = seg.traj_offset;
    let D       = seg.dim;
    let P       = D / 2u;

    // per-slot accumulators. PAIRS_PER_THREAD = 3 covers P up to 192
    // with workgroup size 64. for P=150, slots 0 and 1 are full,
    // slot 2 covers 22 threads (pair indices 128..149).
    var sAA : array<f32,       PAIRS_PER_THREAD>;
    var sBB : array<f32,       PAIRS_PER_THREAD>;
    var sTT : array<f32,       PAIRS_PER_THREAD>;
    var sAB : array<vec2<f32>, PAIRS_PER_THREAD>;
    var sTA : array<vec2<f32>, PAIRS_PER_THREAD>;
    var sTB : array<vec2<f32>, PAIRS_PER_THREAD>;
    for (var s : u32 = 0u; s < PAIRS_PER_THREAD; s = s + 1u) {
        sAA[s] = 0.0;
        sBB[s] = 0.0;
        sTT[s] = 0.0;
        sAB[s] = vec2<f32>(0.0, 0.0);
        sTA[s] = vec2<f32>(0.0, 0.0);
        sTB[s] = vec2<f32>(0.0, 0.0);
    }

    // ── single pass over the time axis ────────────────────────────────

    if (T >= 3u) {
        var t : u32 = 2u;
        loop {
            if (t >= T) { break; }
            let t_off = base + t * D;
            let a_off = base + (t - 1u) * D;
            let b_off = base + (t - 2u) * D;

            for (var s : u32 = 0u; s < PAIRS_PER_THREAD; s = s + 1u) {
                let pair_idx = tid + s * WG_SIZE;
                if (pair_idx >= P) { continue; }
                let p2 = pair_idx * 2u;

                let tgt = vec2<f32>(traj[t_off + p2], traj[t_off + p2 + 1u]);
                let A   = vec2<f32>(traj[a_off + p2], traj[a_off + p2 + 1u]);
                let B   = vec2<f32>(traj[b_off + p2], traj[b_off + p2 + 1u]);

                sAA[s] = sAA[s] + dot(A, A);
                sBB[s] = sBB[s] + dot(B, B);
                sTT[s] = sTT[s] + dot(tgt, tgt);
                sAB[s] = sAB[s] + cmul_conj(A, B);
                sTA[s] = sTA[s] + cmul_conj(tgt, A);
                sTB[s] = sTB[s] + cmul_conj(tgt, B);
            }

            t = t + 1u;
        }
    }

    // ── per-slot 2x2 solve + algebraic rms ────────────────────────────

    var denom : u32 = 1u;
    if (T >= 3u) { denom = T - 2u; }

    for (var s : u32 = 0u; s < PAIRS_PER_THREAD; s = s + 1u) {
        let pair_idx = tid + s * WG_SIZE;
        if (pair_idx >= P) { continue; }

        let fsAA = sAA[s];
        let fsBB = sBB[s];
        let fsTT = sTT[s];
        let fsAB = sAB[s];
        let fsTA = sTA[s];
        let fsTB = sTB[s];

        let trace = fsAA + fsBB;
        let ridge = max(MACHINE_EPS, RIDGE_SCALE * trace * 0.5);
        let sAA_r = fsAA + ridge;
        let sBB_r = fsBB + ridge;
        let absAB2 = fsAB.x * fsAB.x + fsAB.y * fsAB.y;
        let det    = sAA_r * sBB_r - absAB2;
        let det_thresh = MACHINE_EPS * trace * trace;

        var K_v : vec2<f32> = vec2<f32>(2.0, 0.0);
        var G_v : vec2<f32> = vec2<f32>(1.0, 0.0);

        if (T >= 3u && abs(det) >= det_thresh) {
            // K = (sTA * sBB_r - conj(sAB) * sTB) / det
            let t1 = fsTA * sBB_r;
            let t2 = cmul(cconj(fsAB), fsTB);
            K_v = (t1 - t2) / det;
            // G = (sAB * sTA - sAA_r * sTB) / det
            let t3 = cmul(fsAB, fsTA);
            let t4 = fsTB * sAA_r;
            G_v = (t3 - t4) / det;

            let all_finite = all(vec4<f32>(K_v.x, K_v.y, G_v.x, G_v.y) ==
                                 vec4<f32>(K_v.x, K_v.y, G_v.x, G_v.y));
            if (!all_finite) {
                K_v = vec2<f32>(2.0, 0.0);
                G_v = vec2<f32>(1.0, 0.0);
            }
        }

        // algebraic residual: Σ|err|² = sTT
        //                             - 2·Re(K·conj(sTA) - G·conj(sTB))
        //                             + |K|²·sAA + |G|²·sBB
        //                             - 2·Re(K·conj(G)·sAB)
        //
        // Re(K · conj(sTA)) = K.x·sTA.x + K.y·sTA.y   (conj flips imag sign)
        // Re(K · conj(G) · sAB) = Re((K·conj(G)) · sAB), computed after cmul.

        let K_mag2 = K_v.x * K_v.x + K_v.y * K_v.y;
        let G_mag2 = G_v.x * G_v.x + G_v.y * G_v.y;
        let re_K_conjsTA = K_v.x * fsTA.x + K_v.y * fsTA.y;
        let re_G_conjsTB = G_v.x * fsTB.x + G_v.y * fsTB.y;
        let KconjG       = cmul(K_v, cconj(G_v));
        let re_KconjG_sAB = KconjG.x * fsAB.x - KconjG.y * fsAB.y;

        let resid_sq = fsTT
                     - 2.0 * (re_K_conjsTA - re_G_conjsTB)
                     + K_mag2 * fsAA + G_mag2 * fsBB
                     - 2.0 * re_KconjG_sAB;

        let mean_sq = max(resid_sq, 0.0) / f32(denom);
        let rms_v = sqrt(mean_sq);

        K_out[seg.out_offset + pair_idx]   = K_v;
        G_out[seg.out_offset + pair_idx]   = G_v;
        rms_out[seg.out_offset + pair_idx] = rms_v;
    }
}
