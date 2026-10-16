// per_article_reduce.wgsl — sum K/G/rms across each article's segments.
//
// in the single-submit pipeline, fit writes per-segment [K, G, rms],
// and we want per-article sums (for well_sum_K updates and centroid
// computation) WITHOUT sending the full [total_segs, P] stack back
// to CPU. one workgroup per article, threads cooperate on the
// P-dimensional reduction across that article's segments.
//
// inputs:
//   K_per_seg    [total_segs, P]   complex64 stored as interleaved f32
//                                  (matches fit_pair_v3's output)
//   G_per_seg    [total_segs, P]   complex64 interleaved
//   rms_per_seg  [total_segs, P]   f32
//   seg_range    [n_art, 2]        (start, end_exclusive) of segments
//                                  for each article; written by segment.wgsl
//
// outputs:
//   K_sums   [n_art, 2*P]  f32 interleaved (sum of K across article's segments)
//   G_sums   [n_art, 2*P]  f32 interleaved
//   rms_sums [n_art, P]    f32
//   n_segs_per_art [n_art] u32   derived count (end - start)
//
// dispatch: (n_art, 1, 1) workgroups of 64 threads.
//
// for article a, the range [start, end) is a contiguous slice of
// seg_table (segment kernel emits in article order via atomic base).
// each thread strides P to accumulate its slice, then tree-reduces
// to thread 0.

const WG_SIZE : u32 = 64u;

@group(0) @binding(0) var<storage, read>  K_per_seg   : array<f32>;  // [total_segs * 2P]
@group(0) @binding(1) var<storage, read>  G_per_seg   : array<f32>;  // [total_segs * 2P]
@group(0) @binding(2) var<storage, read>  rms_per_seg : array<f32>;  // [total_segs * P]
@group(0) @binding(3) var<storage, read>  seg_range   : array<vec2<u32>>;  // [n_art]
@group(0) @binding(4) var<storage, read_write> K_sums      : array<f32>;  // [n_art * 2P]
@group(0) @binding(5) var<storage, read_write> G_sums      : array<f32>;  // [n_art * 2P]
@group(0) @binding(6) var<storage, read_write> rms_sums    : array<f32>;  // [n_art * P]
@group(0) @binding(7) var<storage, read_write> n_segs_per_art : array<u32>;  // [n_art]
@group(0) @binding(8) var<uniform> params : vec4<u32>;  // (n_art, P, 0, 0)

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    let tid = lid.x;
    let n_art = params.x;
    let P = params.y;
    let a = wg.x;
    if (a >= n_art) { return; }

    let range = seg_range[a];
    let s = range.x;
    let e = range.y;
    let n_segs = e - s;

    if (tid == 0u) {
        n_segs_per_art[a] = n_segs;
    }

    if (n_segs == 0u) {
        // zero out this article's sums
        for (var k : u32 = tid; k < 2u * P; k = k + WG_SIZE) {
            K_sums[a * 2u * P + k] = 0.0;
            G_sums[a * 2u * P + k] = 0.0;
        }
        for (var k : u32 = tid; k < P; k = k + WG_SIZE) {
            rms_sums[a * P + k] = 0.0;
        }
        return;
    }

    // K, G: each thread walks its slice of the 2P row and accumulates
    // across n_segs segments. dim is [n_segs, 2P] row-major.
    let K_out_base = a * 2u * P;
    let G_out_base = a * 2u * P;
    for (var k : u32 = tid; k < 2u * P; k = k + WG_SIZE) {
        var k_sum : f32 = 0.0;
        var g_sum : f32 = 0.0;
        for (var si : u32 = s; si < e; si = si + 1u) {
            let row_base = si * 2u * P;
            k_sum = k_sum + K_per_seg[row_base + k];
            g_sum = g_sum + G_per_seg[row_base + k];
        }
        K_sums[K_out_base + k] = k_sum;
        G_sums[G_out_base + k] = g_sum;
    }

    // rms is only P floats per segment (no complex)
    let R_out_base = a * P;
    for (var k : u32 = tid; k < P; k = k + WG_SIZE) {
        var r_sum : f32 = 0.0;
        for (var si : u32 = s; si < e; si = si + 1u) {
            r_sum = r_sum + rms_per_seg[si * P + k];
        }
        rms_sums[R_out_base + k] = r_sum;
    }
}
