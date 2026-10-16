// finalize_indirect.wgsl — compose fit_indirect dispatch args after
// segment.wgsl has finished writing to seg_count.
//
// segment.wgsl used to try to write this from workgroup 0, but that
// was racy (article 0 can finish before article N-1, reading a partial
// seg_count total). the correct pattern is to run a tiny kernel AFTER
// segment across the compute-pass boundary, so WebGPU's automatic
// RAW barrier guarantees segment completed first.
//
// one workgroup, one thread.
//
// inputs:
//   seg_count [1] u32  — final total from segment kernel
//
// outputs:
//   fit_indirect [3] u32 = (n_workgroups_x=seg_count, P_tiles, 1)
//
// the P_tiles value comes via uniform from the host (fit kernel is
// dispatched as (n_segs, P_tiles, 1) to tile over the pair axis).

@group(0) @binding(0) var<storage, read>       seg_count    : array<u32>;
@group(0) @binding(1) var<storage, read_write> fit_indirect : array<u32>;
@group(0) @binding(2) var<uniform>             params       : vec4<u32>;  // (P_tiles, 0, 0, 0)

@compute @workgroup_size(1, 1, 1)
fn main() {
    fit_indirect[0] = seg_count[0];
    fit_indirect[1] = params.x;
    fit_indirect[2] = 1u;
}
