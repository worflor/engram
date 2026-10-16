// gather_smooth_pair_f16.wgsl — f16 matrix variant.
//
// the GloVe matrix is the hottest random-access surface in the whole
// pipeline. on iGPU with shared DRAM, its 47 MB footprint blows every
// cache level, giving an effective ~3 GB/s random-read bandwidth vs
// the ~30 GB/s peak the hardware claims. f16 storage halves the
// matrix to 24 MB, doubles cache density, and keeps the values in
// the ~half-precision range GloVe embeddings were always happy in.
//
// the f16 feature ("shader-f16") is requested at device init; AMD
// iGPU supports it natively (no emulation penalty).
//
// output is still f32 — we only compress the MATRIX, not the outputs.
// after the window-mean reduction, the f16 quantization noise is
// averaged over W=10 samples, further suppressing it.

enable f16;

@group(0) @binding(0) var<storage, read>       glove_h : array<f16>;
@group(0) @binding(1) var<storage, read>       indices : array<i32>;
@group(0) @binding(2) var<storage, read>       offsets : array<u32>;
@group(0) @binding(3) var<storage, read>       pairing : array<i32>;
@group(0) @binding(4) var<storage, read>       out_off : array<u32>;
@group(0) @binding(5) var<storage, read_write> out_pair: array<f32>;
@group(0) @binding(6) var<uniform>             params  : vec4<u32>;
// params.x = W (≤ 10)
// params.y = D
// params.z = n_articles

const MAX_W : u32 = 10u;

var<workgroup> shared_idx  : array<i32, 40>;
var<workgroup> shared_dsrc : array<i32, 64>;

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

    if (lid.y == 0u) {
        if (d < D) {
            shared_dsrc[lid.x] = pairing[d];
        } else {
            shared_dsrc[lid.x] = 0;
        }
    }
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

    if (d >= D) { return; }
    if (out_t >= total_smooth) { return; }

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

    // accumulate in f32 (output is f32, intermediate precision matters
    // when W=10 values are averaged).
    var acc : f32 = 0.0;
    for (var k : u32 = 0u; k < W; k = k + 1u) {
        let idx = shared_idx[lid.y * MAX_W + k];
        if (idx < 0) { continue; }
        let row_start = u32(idx) * D;
        acc = acc + f32(glove_h[row_start + d_src]);
    }
    acc = acc * (1.0 / f32(W));

    out_pair[out_base + local_t * D + d] = acc;
}
