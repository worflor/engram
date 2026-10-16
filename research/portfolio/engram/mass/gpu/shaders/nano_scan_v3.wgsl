// nano_scan_v3.wgsl — local scan + recursive spine.
//
// bindings (all RW so kernels can work in-place):
//   trans        : per-byte transition table (RO)
//   bytes        : packed byte input (RO, unused by non-init kernels)
//   scan         : the array being scanned (RW)
//   block_totals : one entry per workgroup, filled by scan_local,
//                  scanned by the next level's scan_local (RW)
//
// the driver picks different buffers for `scan` and `block_totals`
// at each recursion level. the shader itself is level-agnostic.

const WG_SIZE : u32 = 256u;
const IDENTITY : u32 = 0x76543210u;

@group(0) @binding(0) var<storage, read>       trans    : array<u32>;
@group(0) @binding(1) var<storage, read>       bytes    : array<u32>;
@group(0) @binding(2) var<storage, read_write> scan     : array<u32>;
@group(0) @binding(3) var<storage, read_write> block_totals : array<u32>;
@group(0) @binding(4) var<uniform>             params   : vec4<u32>;
//  params.x = n (length of `scan`)

fn get_byte(i: u32) -> u32 {
    let word = bytes[i >> 2u];
    return (word >> ((i & 3u) * 8u)) & 0xFFu;
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

var<workgroup> sdata : array<u32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn init(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let wg_stride = nwg.x * WG_SIZE;
    let i = gid.y * wg_stride + gid.x;
    let n = params.x;
    if (i >= n) { return; }
    scan[i] = trans[get_byte(i)];
}

@compute @workgroup_size(WG_SIZE, 1, 1)
fn scan_local(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let wg_stride = nwg.x * WG_SIZE;
    let wg_linear = wg.y * nwg.x + wg.x;
    let i = gid.y * wg_stride + gid.x;
    let tid = lid.x;
    let n = params.x;

    if (i < n) {
        sdata[tid] = scan[i];
    } else {
        sdata[tid] = IDENTITY;
    }
    workgroupBarrier();

    var d : u32 = 1u;
    for (var step : u32 = 0u; step < 8u; step = step + 1u) {
        var val : u32 = IDENTITY;
        var combine : bool = false;
        if (tid >= d) {
            val = compose(sdata[tid], sdata[tid - d]);
            combine = true;
        }
        workgroupBarrier();
        if (combine) {
            sdata[tid] = val;
        }
        workgroupBarrier();
        d = d * 2u;
    }

    if (i < n) {
        scan[i] = sdata[tid];
    }
    if (tid == WG_SIZE - 1u) {
        block_totals[wg_linear] = sdata[tid];
    }
}

@compute @workgroup_size(WG_SIZE, 1, 1)
fn apply_prefix(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let wg_stride = nwg.x * WG_SIZE;
    let wg_linear = wg.y * nwg.x + wg.x;
    let i = gid.y * wg_stride + gid.x;
    let n = params.x;
    if (i >= n) { return; }
    if (wg_linear == 0u) { return; }
    let prefix = block_totals[wg_linear - 1u];
    scan[i] = compose(scan[i], prefix);
}
