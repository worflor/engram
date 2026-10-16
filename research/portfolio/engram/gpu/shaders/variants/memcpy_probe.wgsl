// memcpy_probe.wgsl
//
// bare-metal memory throughput probe. each thread reads N vec4s and
// writes them to a sink buffer. no compute. no reductions. pure
// bandwidth measurement. tells us the physical ceiling we're fighting.

@group(0) @binding(0) var<storage, read>       src  : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst  : array<vec4<f32>>;

// use 2D dispatch + 2D invocation id so we can scale past the 65535
// per-axis limit for large buffers.
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>,
        @builtin(num_workgroups) nwg : vec3<u32>) {
    let wg_stride = nwg.x * 64u;
    let i = gid.y * wg_stride + gid.x;
    if (i >= arrayLength(&src)) { return; }
    dst[i] = src[i];
}
