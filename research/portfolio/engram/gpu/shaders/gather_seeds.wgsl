// gather_seeds.wgsl — compact copy of per-segment seed rows.
//
// for each segment a CPU dispatcher selects from err/vel_norm, it needs
// seed1 = wp[ss-1] and seed2 = wp[ss-2] to populate the Block dataclass.
// reading these back one-by-one from the GPU would be n_segs round
// trips; instead we upload one u32[n_rows] array of absolute timestep
// indices (n_rows = 2*n_segs: [seg0.seed2, seg0.seed1, seg1.seed2, ...])
// and this kernel packs them into a contiguous f32[n_rows*D] buffer
// ready for a single map_sync.
//
// one thread per output float. at D=300 and 16k segments (32 articles
// of real enwiki), that's ~10M threads — well-tiled by workgroup_size
// 256, bandwidth-bound.

@group(0) @binding(0) var<storage, read>       in_traj   : array<f32>;
@group(0) @binding(1) var<storage, read>       row_src   : array<u32>;
@group(0) @binding(2) var<storage, read_write> out_seeds : array<f32>;
@group(0) @binding(5) var<uniform>             params    : vec4<u32>;
// params.x = D
// params.y = n_rows (i.e. 2 * n_segments)

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let D = params.x;
    let n_rows = params.y;
    let total = n_rows * D;
    if (flat >= total) { return; }

    let row = flat / D;
    let d = flat - row * D;
    let src_t = row_src[row];
    out_seeds[flat] = in_traj[src_t * D + d];
}
