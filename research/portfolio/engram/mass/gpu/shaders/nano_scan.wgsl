// nano_scan.wgsl — parallel-scan MASS state machine on GPU.
//
// three kernels (one compute pipeline per pass):
//   init:       byte[i] → fn[i]  (per-byte transition function, packed u32)
//   scan_local: Hillis-Steele scan within each workgroup (256 bytes/wg)
//   scan_spine: scan the workgroup-level block totals
//   apply:      propagate block prefix to all elements + derive state[i]
//
// state packing: 8 states, 4 bits each, packed into one u32.
//   lane s's next state = (fn >> (s*4)) & 0xF
//   composition f∘g: for each s, s' = (g>>(s*4))&0xF ; t = (f>>(s'*4))&0xF
//
// transition table: flat u32[256] indexed by byte value, each entry
// is the packed transition function for that byte.

const WG_SIZE : u32 = 256u;

@group(0) @binding(0) var<storage, read>       bytes    : array<u32>;   // input bytes (packed 4/u32)
@group(0) @binding(1) var<storage, read>       trans    : array<u32>;   // [256] per-byte transition functions
@group(0) @binding(2) var<storage, read_write> scan     : array<u32>;   // [N] per-position prefix (output of scan)
@group(0) @binding(3) var<storage, read_write> block_totals : array<u32>; // [n_blocks] last scan per workgroup

// helper: read byte i from packed bytes array
fn get_byte(i: u32) -> u32 {
    let word = bytes[i >> 2u];
    let shift = (i & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

// compose two packed transition functions: result = f ∘ g, i.e., (f∘g)[s] = f[g[s]]
fn compose(f: u32, g: u32) -> u32 {
    var out : u32 = 0u;
    for (var s : u32 = 0u; s < 8u; s = s + 1u) {
        let s_prime = (g >> (s * 4u)) & 0xFu;
        let t = (f >> (s_prime * 4u)) & 0xFu;
        out = out | (t << (s * 4u));
    }
    return out;
}

// workgroup shared memory for the local scan
var<workgroup> sdata : array<u32, WG_SIZE>;

// ── pass 1: init ──
// each thread: read its byte, look up the transition function, write to scan[i].
@compute @workgroup_size(WG_SIZE, 1, 1)
fn init(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let wg_stride = nwg.x * WG_SIZE;
    let i = gid.y * wg_stride + gid.x;
    let total = arrayLength(&scan);
    if (i >= total) { return; }
    let b = get_byte(i);
    scan[i] = trans[b];
}

// ── pass 2: local scan ──
// each workgroup does a Hillis-Steele inclusive scan over its WG_SIZE
// contiguous elements using shared memory. writes back to scan, and
// writes the last element (block total) to block_totals.
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
    let total = arrayLength(&scan);

    let IDENTITY : u32 = 0x76543210u;  // identity: state s maps to s (0,1,2,...,7 packed)

    // load
    if (i < total) {
        sdata[tid] = scan[i];
    } else {
        sdata[tid] = IDENTITY;
    }
    workgroupBarrier();

    // Hillis-Steele: log2(WG_SIZE) = 8 passes
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

    // write back
    if (i < total) {
        scan[i] = sdata[tid];
    }
    // last thread stores the workgroup total
    if (tid == WG_SIZE - 1u) {
        block_totals[wg_linear] = sdata[tid];
    }
}

// ── pass 3: scan the spine (block totals) ──
// single workgroup handles up to WG_SIZE blocks. for large inputs, we'd
// need a recursive spine scan; for MVP, assume n_blocks <= WG_SIZE.
@compute @workgroup_size(WG_SIZE, 1, 1)
fn scan_spine(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let n = arrayLength(&block_totals);
    let IDENTITY : u32 = 0x76543210u;

    if (tid < n) {
        sdata[tid] = block_totals[tid];
    } else {
        sdata[tid] = IDENTITY;
    }
    workgroupBarrier();

    var d : u32 = 1u;
    for (var step : u32 = 0u; step < 8u; step = step + 1u) {
        var val : u32 = IDENTITY;
        var combine : bool = false;
        if (tid >= d && tid < n) {
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

    if (tid < n) {
        block_totals[tid] = sdata[tid];
    }
}

// ── pass 4: apply prefix to each element ──
// each element's scan[i] = compose(scan[i], block_prefix[wg-1]) where
// block_prefix for workgroup 0 is identity.
@compute @workgroup_size(WG_SIZE, 1, 1)
fn apply_prefix(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let wg_stride = nwg.x * WG_SIZE;
    let wg_linear = wg.y * nwg.x + wg.x;
    let i = gid.y * wg_stride + gid.x;
    let total = arrayLength(&scan);
    if (i >= total) { return; }
    if (wg_linear == 0u) { return; }  // first workgroup has identity prefix
    let prefix = block_totals[wg_linear - 1u];
    scan[i] = compose(scan[i], prefix);
}
