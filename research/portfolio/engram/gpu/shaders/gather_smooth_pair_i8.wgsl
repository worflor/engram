// gather_smooth_pair_i8.wgsl — int8 packed matrix variant.
//
// layout: matrix is stored as u32[V * D/4], where each u32 packs 4
//   consecutive signed int8 values from one row. per-row scale[V] f32
//   converts the quantized byte back to the original range. this
//   squeezes the 47 MB float matrix down to 12 MB + 160 KB of scales,
//   fitting comfortably inside the iGPU's cache hierarchy.
//
// quantization: scale[v] = max(abs(matrix[v])) / 127. matrix_i8[v,d]
//   = round(matrix[v,d] / scale[v]). reconstruction:
//   matrix[v,d] ≈ matrix_i8[v,d] * scale[v].
//
// D must be divisible by 4 for clean packing. at D=300 this holds.
//
// same workgroup layout as the f32/f16 gathers: 64 threads in d, 4 in
// out_t, all coalescing the same row reads on the same u32 when
// possible.

@group(0) @binding(0) var<storage, read>       glove_i8 : array<u32>;
@group(0) @binding(1) var<storage, read>       indices  : array<i32>;
@group(0) @binding(2) var<storage, read>       offsets  : array<u32>;
@group(0) @binding(3) var<storage, read>       pairing  : array<i32>;
@group(0) @binding(4) var<storage, read>       out_off  : array<u32>;
@group(0) @binding(5) var<storage, read_write> out_pair : array<f32>;
@group(0) @binding(6) var<uniform>             params   : vec4<u32>;
@group(0) @binding(7) var<storage, read>       scales   : array<f32>;
// params.x = W (≤ 10)
// params.y = D (must be % 4 == 0)
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
    let D_words = D >> 2u;  // D / 4

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
    let byte_pos = d_src & 3u;        // 0..3
    let u32_col  = d_src >> 2u;       // d_src / 4
    let shift    = byte_pos * 8u;

    var acc : f32 = 0.0;
    for (var k : u32 = 0u; k < W; k = k + 1u) {
        let idx = shared_idx[lid.y * MAX_W + k];
        if (idx < 0) { continue; }
        let row = u32(idx);
        let packed = glove_i8[row * D_words + u32_col];
        let byte_u = (packed >> shift) & 0xFFu;
        // sign extend: interpret 8-bit unsigned as signed
        var byte_i : i32 = i32(byte_u);
        if (byte_u >= 128u) {
            byte_i = byte_i - 256;
        }
        let s = scales[row];
        acc = acc + f32(byte_i) * s;
    }
    acc = acc * (1.0 / f32(W));

    out_pair[out_base + local_t * D + d] = acc;
}
