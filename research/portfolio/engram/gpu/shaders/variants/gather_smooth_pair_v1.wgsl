// gather_smooth_pair.wgsl — the CPU↔GPU bus crossing.
//
// input:
//   glove      f32[V * D]        persistent GloVe matrix (480 MB at V=400k, D=300)
//   indices    i32[N_total]      concatenated article token indices (-1 = skip/pad)
//   offsets    u32[n_articles+1] cumulative token offsets into indices
//   pairing    i32[D]            dim permutation to apply
//   out_pair   f32[N_total_smooth * D]  output paired+smoothed trajectory, packed
//   out_off    u32[n_articles+1] cumulative smoothed-timestep offsets (pre-filled CPU side)
//   params.x   window size (usually 10)
//   params.y   D (usually 300)
//
// per output (t, d):
//   read W = window consecutive indices[offset[a]..offset[a]+W]
//   accumulate glove[idx[k], pairing[d]] / W for k = t..t+W-1
//   write out_pair[a_out_base + t * D + d]
//
// this fuses three stages into ONE kernel:
//   - gather (from GloVe matrix at random idx)
//   - sliding window mean (over W consecutive rows)
//   - apply pairing (dim permutation on output)
//
// each thread owns one (t, d) output cell. coalesced within a workgroup
// on the d axis (adjacent threads = adjacent output dims).

@group(0) @binding(0) var<storage, read>       glove   : array<f32>;
@group(0) @binding(1) var<storage, read>       indices : array<i32>;
@group(0) @binding(2) var<storage, read>       offsets : array<u32>;
@group(0) @binding(3) var<storage, read>       pairing : array<i32>;
@group(0) @binding(4) var<storage, read>       out_off : array<u32>;
@group(0) @binding(5) var<storage, read_write> out_pair: array<f32>;
@group(0) @binding(6) var<uniform>             params  : vec4<u32>;
// params.x = window (W)
// params.y = D
// params.z = n_articles

@compute @workgroup_size(64, 4, 1)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let W = params.x;
    let D = params.y;
    let n_articles = params.z;

    // 2D dispatch: gid.x = (output timestep in smoothed stream)
    //              gid.y = (output dim)
    let out_t = wg.x * 64u + lid.x;
    let d = wg.y * 4u + lid.y;
    if (d >= D) { return; }

    // binary search for which article this out_t belongs to.
    // out_off is cumulative smoothed timesteps; find a such that
    // out_off[a] <= out_t < out_off[a+1].
    var lo : u32 = 0u;
    var hi : u32 = n_articles;
    while (lo + 1u < hi) {
        let mid = (lo + hi) >> 1u;
        if (out_off[mid] <= out_t) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let a = lo;
    if (out_t >= out_off[n_articles]) { return; }

    // local timestep within article: local_t = out_t - out_off[a]
    // source indices slice: indices[offsets[a] + local_t ... offsets[a] + local_t + W]
    let local_t = out_t - out_off[a];
    let src_base = offsets[a] + local_t;

    // the source dim (apply pairing): d_src = pairing[d]
    let d_src = u32(pairing[d]);

    // accumulate window mean
    var acc : f32 = 0.0;
    let inv_w = 1.0 / f32(W);
    for (var k : u32 = 0u; k < W; k = k + 1u) {
        let idx = indices[src_base + k];
        if (idx < 0) { continue; }  // skip missing-vocab tokens (treat as zero)
        let row_start = u32(idx) * D;
        acc = acc + glove[row_start + d_src];
    }
    acc = acc * inv_w;

    // write to output
    let out_base = out_off[a] * D;
    out_pair[out_base + local_t * D + d] = acc;
}
