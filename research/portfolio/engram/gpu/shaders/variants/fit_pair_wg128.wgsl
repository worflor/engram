// same as v1 fit_pair.wgsl but workgroup_size = 128.
// two subgroups (on wave64 hardware) per workgroup, four (on wave32).
// tests whether more threads per workgroup amortizes launch overhead.

const WG_SIZE       : u32 = 128u;
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

var<workgroup> sh : array<f32, WG_SIZE * 8u>;
var<workgroup> K_bcast : vec2<f32>;
var<workgroup> G_bcast : vec2<f32>;

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
    let seg_idx  = wg.x;
    let pair_idx = wg.y;
    let tid      = lid.x;
    let seg       = segments[seg_idx];
    let T         = seg.length;
    let base      = seg.traj_offset;
    let D         = seg.dim;
    let pair_base = pair_idx * 2u;

    var sAA : f32       = 0.0;
    var sBB : f32       = 0.0;
    var sAB : vec2<f32> = vec2<f32>(0.0, 0.0);
    var sTA : vec2<f32> = vec2<f32>(0.0, 0.0);
    var sTB : vec2<f32> = vec2<f32>(0.0, 0.0);

    if (T >= 3u) {
        var t : u32 = 2u + tid;
        loop {
            if (t >= T) { break; }
            let off_tgt = base + t * D + pair_base;
            let off_A   = base + (t - 1u) * D + pair_base;
            let off_B   = base + (t - 2u) * D + pair_base;
            let tgt = vec2<f32>(traj[off_tgt],     traj[off_tgt + 1u]);
            let A   = vec2<f32>(traj[off_A],       traj[off_A + 1u]);
            let B   = vec2<f32>(traj[off_B],       traj[off_B + 1u]);
            sAA = sAA + dot(A, A);
            sBB = sBB + dot(B, B);
            sAB = sAB + cmul_conj(A, B);
            sTA = sTA + cmul_conj(tgt, A);
            sTB = sTB + cmul_conj(tgt, B);
            t = t + WG_SIZE;
        }
    }

    let i = tid * 8u;
    sh[i + 0u] = sAA;
    sh[i + 1u] = sBB;
    sh[i + 2u] = sAB.x; sh[i + 3u] = sAB.y;
    sh[i + 4u] = sTA.x; sh[i + 5u] = sTA.y;
    sh[i + 6u] = sTB.x; sh[i + 7u] = sTB.y;
    workgroupBarrier();

    var stride : u32 = WG_SIZE >> 1u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            let src = (tid + stride) * 8u;
            let dst = tid * 8u;
            sh[dst + 0u] = sh[dst + 0u] + sh[src + 0u];
            sh[dst + 1u] = sh[dst + 1u] + sh[src + 1u];
            sh[dst + 2u] = sh[dst + 2u] + sh[src + 2u];
            sh[dst + 3u] = sh[dst + 3u] + sh[src + 3u];
            sh[dst + 4u] = sh[dst + 4u] + sh[src + 4u];
            sh[dst + 5u] = sh[dst + 5u] + sh[src + 5u];
            sh[dst + 6u] = sh[dst + 6u] + sh[src + 6u];
            sh[dst + 7u] = sh[dst + 7u] + sh[src + 7u];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (tid == 0u) {
        let fsAA = sh[0u];
        let fsBB = sh[1u];
        let fsAB = vec2<f32>(sh[2u], sh[3u]);
        let fsTA = vec2<f32>(sh[4u], sh[5u]);
        let fsTB = vec2<f32>(sh[6u], sh[7u]);

        let trace = fsAA + fsBB;
        let ridge = max(MACHINE_EPS, RIDGE_SCALE * trace * 0.5);
        let sAA_r = fsAA + ridge;
        let sBB_r = fsBB + ridge;
        let absAB2 = fsAB.x * fsAB.x + fsAB.y * fsAB.y;
        let det   = sAA_r * sBB_r - absAB2;
        let det_thresh = MACHINE_EPS * trace * trace;

        var K_v : vec2<f32> = vec2<f32>(2.0, 0.0);
        var G_v : vec2<f32> = vec2<f32>(1.0, 0.0);

        if (T >= 3u && abs(det) >= det_thresh) {
            let t1 = fsTA * sBB_r;
            let t2 = cmul(cconj(fsAB), fsTB);
            K_v = (t1 - t2) / det;
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
        K_bcast = K_v;
        G_bcast = G_v;
        K_out[seg.out_offset + pair_idx] = K_v;
        G_out[seg.out_offset + pair_idx] = G_v;
    }
    workgroupBarrier();

    let K_use = K_bcast;
    let G_use = G_bcast;
    var sq_err : f32 = 0.0;
    if (T >= 3u) {
        var t : u32 = 2u + tid;
        loop {
            if (t >= T) { break; }
            let off_tgt = base + t * D + pair_base;
            let off_A   = base + (t - 1u) * D + pair_base;
            let off_B   = base + (t - 2u) * D + pair_base;
            let tgt = vec2<f32>(traj[off_tgt],     traj[off_tgt + 1u]);
            let A   = vec2<f32>(traj[off_A],       traj[off_A + 1u]);
            let B   = vec2<f32>(traj[off_B],       traj[off_B + 1u]);
            let pred = cmul(K_use, A) - cmul(G_use, B);
            let err  = tgt - pred;
            sq_err = sq_err + dot(err, err);
            t = t + WG_SIZE;
        }
    }
    sh[tid * 8u] = sq_err;
    workgroupBarrier();

    var stride2 : u32 = WG_SIZE >> 1u;
    loop {
        if (stride2 == 0u) { break; }
        if (tid < stride2) {
            sh[tid * 8u] = sh[tid * 8u] + sh[(tid + stride2) * 8u];
        }
        workgroupBarrier();
        stride2 = stride2 >> 1u;
    }

    if (tid == 0u) {
        var denom : u32 = 1u;
        if (T >= 3u) { denom = T - 2u; }
        let mean_sq = sh[0u] / f32(denom);
        rms_out[seg.out_offset + pair_idx] = sqrt(max(mean_sq, 0.0));
    }
}
