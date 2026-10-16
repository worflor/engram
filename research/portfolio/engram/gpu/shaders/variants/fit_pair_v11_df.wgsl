// fit_pair_v11_df.wgsl
//
// double-float arithmetic everywhere. each scalar is represented as
// (hi, lo) with value = hi + lo and |lo| << ulp(hi). this gives ~48
// bits of mantissa — enough to break the cancellation catastrophe in
// the algebraic rms identity that killed v2.
//
// result: single memory pass, correct K/G AND correct rms, no warp
// divergence. ~2× compute vs v3 but bandwidth-bound hardware hides it.
//
// references:
//   Dekker, "A floating-point technique for extending the available
//   precision" (1971).
//   Hida/Li/Bailey, "Library for double-double and quad-double
//   arithmetic" (2001).

const WG_SIZE     : u32 = 64u;
const MACHINE_EPS : f32 = 1.1920929e-7;
const RIDGE_SCALE : f32 = 1.0e-4;

struct Segment {
    traj_offset : u32,
    length      : u32,
    dim         : u32,
    out_offset  : u32,
};

@group(0) @binding(0) var<storage, read>       traj     : array<f32>;
@group(0) @binding(1) var<storage, read>       segments : array<Segment>;
@group(0) @binding(2) var<storage, read_write> K_out    : array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> G_out    : array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> rms_out  : array<f32>;

// ── double-float primitives ─────────────────────────────────────────
//
// a DF scalar is a vec2<f32> with x = hi, y = lo.
// a DF complex is vec4<f32> with (r_hi, r_lo, i_hi, i_lo).

// exact addition of two f32s that may have any magnitude ordering
fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    return vec2<f32>(s, e);
}

// when |a| >= |b|, cheaper version (3 ops vs 6)
fn quick_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let e = b - (s - a);
    return vec2<f32>(s, e);
}

// exact product using fma: e = a*b - p holds when p = fl(a*b)
fn two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    let e = fma(a, b, -p);
    return vec2<f32>(p, e);
}

// DF + f32
fn df_add_f(x: vec2<f32>, y: f32) -> vec2<f32> {
    let s = two_sum(x.x, y);
    return quick_two_sum(s.x, s.y + x.y);
}

// DF + DF (Dekker's add)
fn df_add(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    let s = two_sum(x.x, y.x);
    let t = two_sum(x.y, y.y);
    let c = s.y + t.x;
    let v = quick_two_sum(s.x, c);
    return quick_two_sum(v.x, t.y + v.y);
}

// DF * f32
fn df_mul_f(x: vec2<f32>, y: f32) -> vec2<f32> {
    let p = two_prod(x.x, y);
    return quick_two_sum(p.x, p.y + x.y * y);
}

// DF * DF
fn df_mul(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    let p = two_prod(x.x, y.x);
    return quick_two_sum(p.x, p.y + x.x * y.y + x.y * y.x);
}

// DF / DF via long division (Newton step)
fn df_div(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    let q1 = x.x / y.x;
    let t  = df_mul_f(y, q1);
    let r  = df_add(x, vec2<f32>(-t.x, -t.y));
    let q2 = r.x / y.x;
    return quick_two_sum(q1, q2);
}

fn df_neg(x: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(-x.x, -x.y);
}

fn f32_to_df(x: f32) -> vec2<f32> {
    return vec2<f32>(x, 0.0);
}

fn df_to_f32(x: vec2<f32>) -> f32 {
    return x.x + x.y;
}

// ── double-float complex (vec4: r_hi, r_lo, i_hi, i_lo) ──────────────

fn dfc_zero() -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

fn dfc_add(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    let r = df_add(a.xy, b.xy);
    let i = df_add(a.zw, b.zw);
    return vec4<f32>(r.x, r.y, i.x, i.y);
}

// DF complex + f32 complex (reals in .x, .y)
fn dfc_add_c(a: vec4<f32>, b_r: f32, b_i: f32) -> vec4<f32> {
    let r = df_add_f(a.xy, b_r);
    let i = df_add_f(a.zw, b_i);
    return vec4<f32>(r.x, r.y, i.x, i.y);
}

fn dfc_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    // (a_r + i a_i) * (b_r + i b_i) = (a_r b_r - a_i b_i) + i (a_r b_i + a_i b_r)
    let rr = df_mul(a.xy, b.xy);
    let ii = df_mul(a.zw, b.zw);
    let ri = df_mul(a.xy, b.zw);
    let ir = df_mul(a.zw, b.xy);
    let real_ = df_add(rr, df_neg(ii));
    let imag_ = df_add(ri, ir);
    return vec4<f32>(real_.x, real_.y, imag_.x, imag_.y);
}

fn dfc_conj(a: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.x, a.y, -a.z, -a.w);
}

fn dfc_mul_scalar(a: vec4<f32>, s: vec2<f32>) -> vec4<f32> {
    let r = df_mul(a.xy, s);
    let i = df_mul(a.zw, s);
    return vec4<f32>(r.x, r.y, i.x, i.y);
}

fn dfc_div_scalar(a: vec4<f32>, s: vec2<f32>) -> vec4<f32> {
    let r = df_div(a.xy, s);
    let i = df_div(a.zw, s);
    return vec4<f32>(r.x, r.y, i.x, i.y);
}

// |c|² as DF scalar
fn dfc_abs2(a: vec4<f32>) -> vec2<f32> {
    let rr = df_mul(a.xy, a.xy);
    let ii = df_mul(a.zw, a.zw);
    return df_add(rr, ii);
}

// Re(a * b)
fn dfc_re_mul(a: vec4<f32>, b: vec4<f32>) -> vec2<f32> {
    let rr = df_mul(a.xy, b.xy);
    let ii = df_mul(a.zw, b.zw);
    return df_add(rr, df_neg(ii));
}

// accumulate |c|² onto a DF scalar (c is in f32)
fn df_acc_abs2_f32(acc: vec2<f32>, r: f32, i: f32) -> vec2<f32> {
    let rr = two_prod(r, r);
    let ii = two_prod(i, i);
    let s = quick_two_sum(rr.x + ii.x, rr.y + ii.y);
    return df_add(acc, s);
}

// accumulate a*conj(b) (f32 inputs) onto DF complex
fn dfc_acc_mul_conj(acc: vec4<f32>, a_r: f32, a_i: f32, b_r: f32, b_i: f32) -> vec4<f32> {
    // a * conj(b) = (a_r b_r + a_i b_i) + i (a_i b_r - a_r b_i)
    let rr = two_prod(a_r, b_r);
    let ii = two_prod(a_i, b_i);
    let ir = two_prod(a_i, b_r);
    let ri = two_prod(a_r, b_i);
    let real_add = df_add(rr, ii);
    let imag_add = df_add(ir, vec2<f32>(-ri.x, -ri.y));
    let r_new = df_add(acc.xy, real_add);
    let i_new = df_add(acc.zw, imag_add);
    return vec4<f32>(r_new.x, r_new.y, i_new.x, i_new.y);
}

fn load_z(base: u32, t: u32, D: u32, p2: u32) -> vec2<f32> {
    let off = base + t * D + p2;
    return vec2<f32>(traj[off], traj[off + 1u]);
}

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    let seg_idx   = wg.x;
    let pair_base = wg.y * WG_SIZE;
    let pair_idx  = pair_base + lid.x;
    let seg = segments[seg_idx];
    let T   = seg.length;
    let D   = seg.dim;
    let P   = D / 2u;
    if (pair_idx >= P) { return; }

    let base = seg.traj_offset;
    let p2   = pair_idx * 2u;

    // all six sums as DF
    var sAA : vec2<f32> = vec2<f32>(0.0, 0.0);      // DF real
    var sBB : vec2<f32> = vec2<f32>(0.0, 0.0);      // DF real
    var sTT : vec2<f32> = vec2<f32>(0.0, 0.0);      // DF real
    var sAB : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);  // DF complex
    var sTA : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var sTB : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    if (T >= 3u) {
        var z_tm2 = load_z(base, 0u, D, p2);
        var z_tm1 = load_z(base, 1u, D, p2);
        var t : u32 = 2u;
        loop {
            if (t >= T) { break; }
            let z_t = load_z(base, t, D, p2);

            sAA = df_acc_abs2_f32(sAA, z_tm1.x, z_tm1.y);
            sBB = df_acc_abs2_f32(sBB, z_tm2.x, z_tm2.y);
            sTT = df_acc_abs2_f32(sTT, z_t.x,   z_t.y);
            sAB = dfc_acc_mul_conj(sAB, z_tm1.x, z_tm1.y, z_tm2.x, z_tm2.y);
            sTA = dfc_acc_mul_conj(sTA, z_t.x,   z_t.y,   z_tm1.x, z_tm1.y);
            sTB = dfc_acc_mul_conj(sTB, z_t.x,   z_t.y,   z_tm2.x, z_tm2.y);

            z_tm2 = z_tm1;
            z_tm1 = z_t;
            t = t + 1u;
        }
    }

    // solve 2x2 in DF
    let trace_df = df_add(sAA, sBB);
    let trace    = df_to_f32(trace_df);
    let ridge    = max(MACHINE_EPS, RIDGE_SCALE * trace * 0.5);
    let sAA_r    = df_add_f(sAA, ridge);
    let sBB_r    = df_add_f(sBB, ridge);

    // absAB2 = sAB.r² + sAB.i² (DF)
    let sAB_r_sq = df_mul(sAB.xy, sAB.xy);
    let sAB_i_sq = df_mul(sAB.zw, sAB.zw);
    let absAB2   = df_add(sAB_r_sq, sAB_i_sq);

    // det = sAA_r * sBB_r - absAB2 (DF)
    let sAAsBB = df_mul(sAA_r, sBB_r);
    let det    = df_add(sAAsBB, df_neg(absAB2));

    let det_thresh = MACHINE_EPS * trace * trace;

    var K_v : vec4<f32> = vec4<f32>(2.0, 0.0, 0.0, 0.0);  // DF complex
    var G_v : vec4<f32> = vec4<f32>(1.0, 0.0, 0.0, 0.0);

    if (T >= 3u && abs(det.x) >= det_thresh) {
        // K = (sTA * sBB_r - conj(sAB) * sTB) / det
        let t1 = dfc_mul_scalar(sTA, sBB_r);
        let t2 = dfc_mul(dfc_conj(sAB), sTB);
        let K_num = dfc_add(t1, vec4<f32>(-t2.x, -t2.y, -t2.z, -t2.w));
        K_v = dfc_div_scalar(K_num, det);
        // G = (sAB * sTA - sAA_r * sTB) / det
        let t3 = dfc_mul(sAB, sTA);
        let t4 = dfc_mul_scalar(sTB, sAA_r);
        let G_num = dfc_add(t3, vec4<f32>(-t4.x, -t4.y, -t4.z, -t4.w));
        G_v = dfc_div_scalar(G_num, det);

        let all_finite = all(vec4<bool>(K_v.x == K_v.x, K_v.z == K_v.z,
                                         G_v.x == G_v.x, G_v.z == G_v.z));
        if (!all_finite) {
            K_v = vec4<f32>(2.0, 0.0, 0.0, 0.0);
            G_v = vec4<f32>(1.0, 0.0, 0.0, 0.0);
        }
    }

    // ── algebraic rms in DF (no catastrophic cancellation) ─────────────
    //
    // Σ|err|² = sTT − 2·Re(K·conj(sTA) − G·conj(sTB))
    //         + |K|²·sAA + |G|²·sBB − 2·Re(K·conj(G)·sAB)

    let K_mag2 = dfc_abs2(K_v);
    let G_mag2 = dfc_abs2(G_v);
    let re_K_conjsTA = dfc_re_mul(K_v, dfc_conj(sTA));
    let re_G_conjsTB = dfc_re_mul(G_v, dfc_conj(sTB));
    let KconjG       = dfc_mul(K_v, dfc_conj(G_v));
    let re_KconjG_sAB = dfc_re_mul(KconjG, sAB);

    // resid = sTT - 2*(re_K_conjsTA - re_G_conjsTB) + K_mag2*sAA + G_mag2*sBB - 2*re_KconjG_sAB
    let term1 = df_add(re_K_conjsTA, df_neg(re_G_conjsTB));
    let term1_2 = df_mul_f(term1, 2.0);
    let sum1 = df_add(sTT, df_neg(term1_2));
    let term2 = df_mul(K_mag2, sAA);
    let term3 = df_mul(G_mag2, sBB);
    let sum2 = df_add(term2, term3);
    let term4_2 = df_mul_f(re_KconjG_sAB, 2.0);
    let resid_df = df_add(df_add(sum1, sum2), df_neg(term4_2));
    let resid_sq = df_to_f32(resid_df);

    var denom : u32 = 1u;
    if (T >= 3u) { denom = T - 2u; }
    let rms_v = sqrt(max(resid_sq, 0.0) / f32(denom));

    // output: collapse DF to f32
    let K_out_v = vec2<f32>(K_v.x + K_v.y, K_v.z + K_v.w);
    let G_out_v = vec2<f32>(G_v.x + G_v.y, G_v.z + G_v.w);

    K_out[seg.out_offset + pair_idx]   = K_out_v;
    G_out[seg.out_offset + pair_idx]   = G_out_v;
    rms_out[seg.out_offset + pair_idx] = rms_v;
}
