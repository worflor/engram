// segment.wgsl — GPU-side segmentation, parallel across articles.
//
// one workgroup per article. 128 threads cooperate on parallel
// reductions via shared-memory tree reductions (subgroup ops would
// be cleaner, but Naga's WGSL frontend does not implement subgroup
// intrinsics as of this wgpu-py version — see engram/gpu/shaders
// notes for SPIR-V injection attempts that hit Naga-less paths).
// thread 0 handles the sequential boundary-emission walk and
// atomically reserves global seg_table slots.

const MAX_LAG     : u32 = 128u;
const MIN_BLOCK   : u32 = 4u;
const SEED_COUNT  : u32 = 2u;
const MACHINE_EPS : f32 = 1.1920929e-7;
const SIGMA       : f32 = 3.0;

@group(0) @binding(0) var<storage, read>       err      : array<f32>;
@group(0) @binding(1) var<storage, read>       vel      : array<f32>;
@group(0) @binding(2) var<storage, read>       out_off  : array<u32>;
@group(0) @binding(3) var<storage, read_write> seg_table   : array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> seg_art_idx : array<u32>;
@group(0) @binding(5) var<storage, read_write> seg_count   : array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> fit_indirect : array<u32>;
@group(0) @binding(7) var<uniform>             params      : vec4<u32>;
// per-article [start, end_exclusive) range into seg_table — needed by
// per_article_reduce to find each article's segments without CPU help.
// default (0, 0) means "no segments emitted for this article".
@group(0) @binding(8) var<storage, read_write> seg_range   : array<vec2<u32>>;

var<workgroup> wg_mean   : f32;
var<workgroup> wg_var    : f32;
var<workgroup> wg_macro  : u32;
var<workgroup> wg_thresh : f32;
var<workgroup> wg_acf_done : u32;
var<workgroup> wg_partial : array<f32, 128>;

const MAX_CAND : u32 = 2048u;
var<workgroup> wg_cand_ss : array<u32, MAX_CAND>;
var<workgroup> wg_cand_L  : array<u32, MAX_CAND>;

fn reduce_sum(tid: u32, val: f32) -> f32 {
    wg_partial[tid] = val;
    workgroupBarrier();
    var step : u32 = 64u;
    loop {
        if (step == 0u) { break; }
        if (tid < step) {
            wg_partial[tid] = wg_partial[tid] + wg_partial[tid + step];
        }
        workgroupBarrier();
        step = step >> 1u;
    }
    return wg_partial[0];
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    let tid = lid.x;
    let D = params.x;
    let n_articles = params.y;
    let P_tiles = params.z;
    let max_block = params.w;
    let a = wg.x;
    if (a >= n_articles) { return; }

    let P = D / 2u;
    let art_start = out_off[a];
    let art_end   = out_off[a + 1u];
    let T = art_end - art_start;

    if (T <= SEED_COUNT) {
        // article too short — emit empty range so per_article_reduce
        // knows there's nothing to sum for this slot.
        if (tid == 0u) {
            seg_range[a] = vec2<u32>(0u, 0u);
        }
        return;
    }

    let vn_len = T - 1u;
    let vn_start = art_start + 1u;

    var local_sum : f32 = 0.0;
    for (var i : u32 = tid; i < vn_len; i = i + 128u) {
        local_sum = local_sum + vel[vn_start + i];
    }
    let mean = reduce_sum(tid, local_sum) / f32(vn_len);
    if (tid == 0u) { wg_mean = mean; }
    workgroupBarrier();

    var local_var : f32 = 0.0;
    for (var i : u32 = tid; i < vn_len; i = i + 128u) {
        let d = vel[vn_start + i] - wg_mean;
        local_var = local_var + d * d;
    }
    let var_sum = reduce_sum(tid, local_var);
    if (tid == 0u) { wg_var = var_sum; }
    workgroupBarrier();

    if (wg_var < MACHINE_EPS || wg_var / f32(vn_len) < MACHINE_EPS * MACHINE_EPS) {
        if (tid == 0u) { wg_macro = MIN_BLOCK; }
    } else {
        if (tid == 0u) {
            wg_macro = MIN_BLOCK;
            wg_acf_done = 0u;
        }
        workgroupBarrier();

        var ml : u32 = MAX_LAG;
        if (ml > vn_len / 2u) { ml = vn_len / 2u; }

        for (var lag : u32 = 1u; lag < ml; lag = lag + 1u) {
            if (wg_acf_done == 1u) { break; }
            var lac : f32 = 0.0;
            let limit = vn_len - lag;
            for (var i : u32 = tid; i < limit; i = i + 128u) {
                let d0 = vel[vn_start + i] - wg_mean;
                let d1 = vel[vn_start + i + lag] - wg_mean;
                lac = lac + d0 * d1;
            }
            let acf_num = reduce_sum(tid, lac);
            if (tid == 0u) {
                if (acf_num / wg_var <= 0.0) {
                    var m = lag;
                    if (m < MIN_BLOCK) { m = MIN_BLOCK; }
                    wg_macro = m;
                    wg_acf_done = 1u;
                }
            }
            workgroupBarrier();
        }
        if (tid == 0u) {
            if (wg_acf_done == 0u) {
                var m = ml;
                if (m < MIN_BLOCK) { m = MIN_BLOCK; }
                wg_macro = m;
            }
            if (wg_macro > max_block) { wg_macro = max_block; }
        }
        workgroupBarrier();
    }

    let e_start = art_start + SEED_COUNT;
    let e_len = T - SEED_COUNT;
    if (e_len <= 2u) {
        if (tid == 0u) { wg_thresh = 1e30; }
    } else {
        var e_sum : f32 = 0.0;
        var e_sq  : f32 = 0.0;
        for (var i : u32 = tid; i < e_len; i = i + 128u) {
            let v = err[e_start + i];
            e_sum = e_sum + v;
            e_sq  = e_sq  + v * v;
        }
        let tot_sum = reduce_sum(tid, e_sum);
        let tot_sq  = reduce_sum(tid, e_sq);
        if (tid == 0u) {
            let em = tot_sum / f32(e_len);
            var ev = tot_sq / f32(e_len) - em * em;
            if (ev < 0.0) { ev = 0.0; }
            let es = sqrt(ev) + MACHINE_EPS;
            wg_thresh = em + SIGMA * es;
        }
    }
    workgroupBarrier();

    if (tid == 0u) {
        var block_start : u32 = SEED_COUNT;
        let macro_v = wg_macro;
        let thresh_v = wg_thresh;
        var local_n : u32 = 0u;

        for (var t : u32 = SEED_COUNT + 1u; t < T; t = t + 1u) {
            let bl = t - block_start;
            var emit : bool = false;
            if (bl >= macro_v) { emit = true; }
            else if (bl >= MIN_BLOCK && err[art_start + t] > thresh_v) {
                emit = true;
            }
            if (emit) {
                if (local_n < MAX_CAND) {
                    wg_cand_ss[local_n] = block_start;
                    wg_cand_L[local_n]  = bl;
                    local_n = local_n + 1u;
                }
                block_start = t;
            }
        }
        if (block_start < T) {
            if (local_n < MAX_CAND) {
                wg_cand_ss[local_n] = block_start;
                wg_cand_L[local_n]  = T - block_start;
                local_n = local_n + 1u;
            }
        }

        if (local_n > 1u) {
            let last = local_n - 1u;
            if (wg_cand_L[last] < MIN_BLOCK) {
                wg_cand_L[last - 1u] = wg_cand_L[last - 1u] + wg_cand_L[last];
                local_n = local_n - 1u;
            }
        }

        var kept : u32 = 0u;
        for (var k : u32 = 0u; k < local_n; k = k + 1u) {
            if (wg_cand_L[k] >= MIN_BLOCK + 1u) { kept = kept + 1u; }
        }

        // write seg_range unconditionally (only thread 0 ever gets here).
        // kept == 0 → empty range, else → (base, base+kept). the empty-
        // range branch MUST NOT live in an `else` branch of `tid == 0u`
        // because threads tid > 0 would race a (0,0) write here against
        // thread 0's real emission above.
        if (kept == 0u) {
            seg_range[a] = vec2<u32>(0u, 0u);
            return;
        }

        let base = atomicAdd(&seg_count[0], kept);

        // record this article's range in seg_table so
        // per_article_reduce can find its segments without scanning.
        seg_range[a] = vec2<u32>(base, base + kept);

        var w_idx : u32 = 0u;
        for (var k : u32 = 0u; k < local_n; k = k + 1u) {
            let L = wg_cand_L[k];
            if (L < MIN_BLOCK + 1u) { continue; }
            let ss = wg_cand_ss[k];
            let traj_offset = (art_start + ss - 2u) * D;
            let length_rows = L + 2u;
            let pair_off = (base + w_idx) * P;
            seg_table[base + w_idx] =
                vec4<u32>(traj_offset, length_rows, D, pair_off);
            seg_art_idx[base + w_idx] = a;
            w_idx = w_idx + 1u;
        }
    }

    // fit_indirect is composed by a separate finalize kernel after
    // segment completes across all workgroups. we leave it alone here
    // — writing it from workgroup_id==0 was racy (that workgroup may
    // finish before others and capture a partial seg_count).
}
