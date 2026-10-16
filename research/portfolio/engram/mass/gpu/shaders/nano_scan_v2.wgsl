// nano_scan_v2.wgsl — global Hillis-Steele parallel scan of MASS
// state-machine transition functions.
//
// simpler than the local+spine approach: one kernel, dispatched
// log2(N) times with a doubling stride. O(N log N) total work but
// O(log N) dispatches. ping-pong between two buffers to avoid races.
//
// input: bytes (packed u32s)
// output: scan[i] = fn[i] ∘ fn[i-1] ∘ ... ∘ fn[0] (inclusive scan)
//         state_after_i = scan[i] applied to initial state

@group(0) @binding(0) var<storage, read>       trans    : array<u32>;
@group(0) @binding(1) var<storage, read>       bytes    : array<u32>;
@group(0) @binding(2) var<storage, read>       in_buf   : array<u32>;
@group(0) @binding(3) var<storage, read_write> out_buf  : array<u32>;
@group(0) @binding(4) var<uniform>             params   : vec4<u32>;
//  params.x = d (stride), params.y = n (length)

fn get_byte(i: u32) -> u32 {
    let word = bytes[i >> 2u];
    let shift = (i & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

fn compose(f: u32, g: u32) -> u32 {
    var out : u32 = 0u;
    for (var s : u32 = 0u; s < 8u; s = s + 1u) {
        let sp = (g >> (s * 4u)) & 0xFu;
        let t  = (f >> (sp * 4u)) & 0xFu;
        out = out | (t << (s * 4u));
    }
    return out;
}

// ── init: byte[i] → fn[i] ──
@compute @workgroup_size(256, 1, 1)
fn init(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let wg_stride = nwg.x * 256u;
    let i = gid.y * wg_stride + gid.x;
    let n = params.y;
    if (i >= n) { return; }
    out_buf[i] = trans[get_byte(i)];
}

// ── global Hillis-Steele step ──
// for each i: if i >= d, out[i] = in[i] ∘ in[i-d]. else out[i] = in[i].
@compute @workgroup_size(256, 1, 1)
fn scan_step(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let wg_stride = nwg.x * 256u;
    let i = gid.y * wg_stride + gid.x;
    let n = params.y;
    if (i >= n) { return; }
    let d = params.x;
    if (i >= d) {
        out_buf[i] = compose(in_buf[i], in_buf[i - d]);
    } else {
        out_buf[i] = in_buf[i];
    }
}
