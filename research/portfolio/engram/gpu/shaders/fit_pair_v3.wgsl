// fit_pair_v3.wgsl
//
// third architecture:
//   - dispatch (n_segs, ceil(P / WG_SIZE), 1)
//   - workgroup_size = 64, each thread handles ONE pair
//   - thread does its own full time loop, no shared memory, no barriers
//   - within a workgroup, adjacent threads read adjacent pairs of the
//     same timestep → coalesced (8-byte stride)
//   - total workgroups = n_segs * ceil(P/64) — matches iGPU concurrency
//     better than v1's n_segs * P (massive oversubscription)
//   - keeps v1's numerically stable 2-pass rms (fix v2 learned the
//     hard way that algebraic rms is catastrophic in f32)

const WG_SIZE       : u32 = 64u;
const MACHINE_EPS   : f32 = 1.1920929e-7;
const RIDGE_SCALE   : f32 = 1.0e-4;

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

    var sAA : f32       = 0.0;
    var sBB : f32       = 0.0;
    var sAB : vec2<f32> = vec2<f32>(0.0, 0.0);
    var sTA : vec2<f32> = vec2<f32>(0.0, 0.0);
    var sTB : vec2<f32> = vec2<f32>(0.0, 0.0);

    if (T >= 3u) {
        var t : u32 = 2u;
        loop {
            if (t >= T) { break; }
            let off_tgt = base + t * D + p2;
            let off_A   = base + (t - 1u) * D + p2;
            let off_B   = base + (t - 2u) * D + p2;
            let tgt = vec2<f32>(traj[off_tgt],     traj[off_tgt + 1u]);
            let A   = vec2<f32>(traj[off_A],       traj[off_A + 1u]);
            let B   = vec2<f32>(traj[off_B],       traj[off_B + 1u]);
            sAA = sAA + dot(A, A);
            sBB = sBB + dot(B, B);
            sAB = sAB + cmul_conj(A, B);
            sTA = sTA + cmul_conj(tgt, A);
            sTB = sTB + cmul_conj(tgt, B);
            t = t + 1u;
        }
    }

    let trace = sAA + sBB;
    let ridge = max(MACHINE_EPS, RIDGE_SCALE * trace * 0.5);
    let sAA_r = sAA + ridge;
    let sBB_r = sBB + ridge;
    let absAB2 = sAB.x * sAB.x + sAB.y * sAB.y;
    let det    = sAA_r * sBB_r - absAB2;
    let det_thresh = MACHINE_EPS * trace * trace;

    var K_v : vec2<f32> = vec2<f32>(2.0, 0.0);
    var G_v : vec2<f32> = vec2<f32>(1.0, 0.0);

    if (T >= 3u && abs(det) >= det_thresh) {
        let t1 = sTA * sBB_r;
        let t2 = cmul(cconj(sAB), sTB);
        K_v = (t1 - t2) / det;
        let t3 = cmul(sAB, sTA);
        let t4 = sTB * sAA_r;
        G_v = (t3 - t4) / det;
        let all_finite = all(vec4<f32>(K_v.x, K_v.y, G_v.x, G_v.y) ==
                             vec4<f32>(K_v.x, K_v.y, G_v.x, G_v.y));
        if (!all_finite) {
            K_v = vec2<f32>(2.0, 0.0);
            G_v = vec2<f32>(1.0, 0.0);
        }
    }

    // numerically stable 2nd pass for rms (pointwise err, no catastrophic cancel)
    var sq_err : f32 = 0.0;
    if (T >= 3u) {
        var t : u32 = 2u;
        loop {
            if (t >= T) { break; }
            let off_tgt = base + t * D + p2;
            let off_A   = base + (t - 1u) * D + p2;
            let off_B   = base + (t - 2u) * D + p2;
            let tgt = vec2<f32>(traj[off_tgt],     traj[off_tgt + 1u]);
            let A   = vec2<f32>(traj[off_A],       traj[off_A + 1u]);
            let B   = vec2<f32>(traj[off_B],       traj[off_B + 1u]);
            let pred = cmul(K_v, A) - cmul(G_v, B);
            let err  = tgt - pred;
            sq_err = sq_err + dot(err, err);
            t = t + 1u;
        }
    }
    var denom : u32 = 1u;
    if (T >= 3u) { denom = T - 2u; }
    let rms_v = sqrt(max(sq_err, 0.0) / f32(denom));

    K_out[seg.out_offset + pair_idx]   = K_v;
    G_out[seg.out_offset + pair_idx]   = G_v;
    rms_out[seg.out_offset + pair_idx] = rms_v;
}
