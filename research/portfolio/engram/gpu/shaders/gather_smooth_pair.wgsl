// gather_smooth_pair.wgsl — v2 with coalesced glove reads.
//
// v1 design: workgroup_size(64, 4). lid.x varied out_t, lid.y varied d.
// within a warp (64 threads, lid.x = 0..63), every thread had a
// different out_t ⇒ different token indices ⇒ different glove ROWS.
// glove reads scattered across the 47 MB matrix: cache-miss per read,
// iGPU effective bandwidth dropped to ~3 GB/s random.
//
// v2 design: same workgroup_size(64, 4), but SWAPPED meaning:
//   lid.x = 0..63  → varies d   (COLUMN within glove)
//   lid.y = 0..3   → varies out_t (4 timesteps per workgroup)
// within a warp, all 64 threads have the same out_t ⇒ same row in
// glove ⇒ 64 consecutive column reads hit the same cache line(s).
// every thread fetches W=10 rows' worth of (idx, row_start+d_src)
// pairs, but since row and src_base are shared per lid.y, we read
// them cooperatively into shared memory.
//
// memory shared per workgroup:
//   shared_idx  [4 × 10]  (40 × 4 bytes = 160 B)  — W indices per out_t
//   shared_dsrc [64]      (64 × 4 bytes = 256 B)  — pairing[d] per d
// total 416 B ≪ 64 KiB limit.
//
// dispatch: (ceil(total_smooth/4), ceil(D/64), 1)

@group(0) @binding(0) var<storage, read>       glove   : array<f32>;
@group(0) @binding(1) var<storage, read>       indices : array<i32>;
@group(0) @binding(2) var<storage, read>       offsets : array<u32>;
@group(0) @binding(3) var<storage, read>       pairing : array<i32>;
@group(0) @binding(4) var<storage, read>       out_off : array<u32>;
@group(0) @binding(5) var<storage, read_write> out_pair: array<f32>;
@group(0) @binding(6) var<uniform>             params  : vec4<u32>;
// params.x = W (must be ≤ 10 for this shader)
// params.y = D
// params.z = n_articles

const MAX_W : u32 = 10u;

var<workgroup> shared_idx  : array<i32, 40>;  // 4 out_t × 10 indices
var<workgroup> shared_dsrc : array<i32, 64>;  // 64 d → pairing[d]

@compute @workgroup_size(64, 4, 1)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let W = params.x;
    let D = params.y;
    let n_articles = params.z;

    let out_t = wg.x * 4u + lid.y;
    let d     = wg.y * 64u + lid.x;
    let total_smooth = out_off[n_articles];

    // ── cooperative shared-memory loads ──────────────────────────
    // thread (lid.x, lid.y = 0) loads pairing for this workgroup's d.
    if (lid.y == 0u) {
        if (d < D) {
            shared_dsrc[lid.x] = pairing[d];
        } else {
            shared_dsrc[lid.x] = 0;
        }
    }
    // thread (lid.x < W, each lid.y) loads its out_t's k-th index.
    if (lid.x < W) {
        if (out_t < total_smooth) {
            var lo : u32 = 0u;
            var hi : u32 = n_articles;
            while (lo + 1u < hi) {
                let mid = (lo + hi) >> 1u;
                if (out_off[mid] <= out_t) { lo = mid; } else { hi = mid; }
            }
            let a = lo;
            let local_t = out_t - out_off[a];
            let src_base = offsets[a] + local_t;
            shared_idx[lid.y * MAX_W + lid.x] = indices[src_base + lid.x];
        } else {
            shared_idx[lid.y * MAX_W + lid.x] = -1;
        }
    }
    workgroupBarrier();

    // ── real work ─────────────────────────────────────────────────
    if (d >= D) { return; }
    if (out_t >= total_smooth) { return; }

    // recompute article (can't easily share across the workgroup for
    // the 4 different out_t values; the binary search is cheap).
    var lo : u32 = 0u;
    var hi : u32 = n_articles;
    while (lo + 1u < hi) {
        let mid = (lo + hi) >> 1u;
        if (out_off[mid] <= out_t) { lo = mid; } else { hi = mid; }
    }
    let a = lo;
    let local_t = out_t - out_off[a];
    let out_base = out_off[a] * D;

    let d_src = u32(shared_dsrc[lid.x]);

    var acc : f32 = 0.0;
    for (var k : u32 = 0u; k < W; k = k + 1u) {
        let idx = shared_idx[lid.y * MAX_W + k];
        if (idx < 0) { continue; }
        let row_start = u32(idx) * D;
        acc = acc + glove[row_start + d_src];
    }
    acc = acc * (1.0 / f32(W));

    out_pair[out_base + local_t * D + d] = acc;
}
