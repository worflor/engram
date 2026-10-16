// fit_pair_v8_sm.wgsl
//
// shared memory caching: load the workgroup's (segment, 64-pair) tile
// into on-chip SRAM once. then BOTH phase A (solve) and phase D (rms)
// read from shared memory at L1/SRAM speed instead of DRAM.
//
// memory budget: T_MAX=128 × 64 pairs × sizeof(vec2<f32>)=8B = 65536 B,
// exactly at the WebGPU workgroup storage limit on this device.
//
// dispatch: (n_segs, ceil(P/64), 1), workgroup_size=64.
// each workgroup:
//   1. cooperatively loads its (segment, 64-pair) slab into SM (all 64
//      threads load 128 timesteps of their pair in parallel).
//   2. barrier.
//   3. each thread runs its pair's full fit (phase A + solve + phase D)
//      reading exclusively from SM. no barriers inside the fit itself.

const WG_SIZE     : u32 = 64u;
const T_MAX       : u32 = 128u;
const MACHINE_EPS : f32 = 1.1920929e-7;
const RIDGE_SCALE : f32 = 1.0e-4;

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

// layout: sm[t * WG_SIZE + tid] = z at time t for tid's pair.
// this layout keeps thread-t reads coalesced in SM (they're consecutive
// banks modulo a typical 32-bank layout).
var<workgroup> sm : array<vec2<f32>, T_MAX * WG_SIZE>;

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
    let tid       = lid.x;
    let pair_idx  = pair_base + tid;

    let seg = segments[seg_idx];
    let T   = seg.length;
    let D   = seg.dim;
    let P   = D / 2u;
    let base = seg.traj_offset;
    let in_range = pair_idx < P;
    let p2 = pair_idx * 2u;

    // ── cooperative load into shared memory ────────────────────────────
    //
    // each thread loads all T timesteps of its own pair. inin_range threads
    // skip the load but must still reach the barrier.
    if (in_range) {
        var t : u32 = 0u;
        loop {
            if (t >= T) { break; }
            let off = base + t * D + p2;
            sm[t * WG_SIZE + tid] = vec2<f32>(traj[off], traj[off + 1u]);
            t = t + 1u;
        }
    }
    workgroupBarrier();

    if (!in_range) { return; }

    // ── phase A: solve for K, G from shared memory ─────────────────────

    var sAA : f32       = 0.0;
    var sBB : f32       = 0.0;
    var sAB : vec2<f32> = vec2<f32>(0.0, 0.0);
    var sTA : vec2<f32> = vec2<f32>(0.0, 0.0);
    var sTB : vec2<f32> = vec2<f32>(0.0, 0.0);

    if (T >= 3u) {
        var z_tm2 = sm[0u * WG_SIZE + tid];
        var z_tm1 = sm[1u * WG_SIZE + tid];
        var t : u32 = 2u;
        loop {
            if (t >= T) { break; }
            let z_t = sm[t * WG_SIZE + tid];
            sAA = sAA + dot(z_tm1, z_tm1);
            sBB = sBB + dot(z_tm2, z_tm2);
            sAB = sAB + cmul_conj(z_tm1, z_tm2);
            sTA = sTA + cmul_conj(z_t,   z_tm1);
            sTB = sTB + cmul_conj(z_t,   z_tm2);
            z_tm2 = z_tm1;
            z_tm1 = z_t;
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

    // ── phase D: pointwise rms, from shared memory (not DRAM) ──────────

    var sq_err : f32 = 0.0;
    if (T >= 3u) {
        var z_tm2 = sm[0u * WG_SIZE + tid];
        var z_tm1 = sm[1u * WG_SIZE + tid];
        var t : u32 = 2u;
        loop {
            if (t >= T) { break; }
            let z_t = sm[t * WG_SIZE + tid];
            let pred = cmul(K_v, z_tm1) - cmul(G_v, z_tm2);
            let err  = z_t - pred;
            sq_err = sq_err + dot(err, err);
            z_tm2 = z_tm1;
            z_tm1 = z_t;
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
