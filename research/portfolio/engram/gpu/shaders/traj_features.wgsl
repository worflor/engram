// traj_features.wgsl — per-timestep features for segmentation.
//
// computes three scalar features for every timestep in a packed batch
// of paired trajectories. all three derive from the same w[t], w[t-1],
// w[t-2] neighborhood, so we fuse them into one pass — the trajectory
// read is the expensive part on iGPU.
//
//   err[t]      = ‖w[t] - 2·w[t-1] + w[t-2]‖₂   (t >= 2, else 0)
//     drives the 3σ block-boundary detector.
//   vel_norm[t] = ‖w[t] - w[t-1]‖₂               (t >= 1, else 0)
//     drives the autocorrelation-length → macro block size.
//   pwr[t]      = ‖w[t]‖₂²                       (always)
//     cumsum on CPU → signal_energy lookup per segment.
//
// the three outputs together are 3 × 4 × total_smooth bytes — for a
// typical 200k-timestep batch that's 2.4 MB vs the 200 MB wp itself.
// pair this with gather_seeds.wgsl and the full wp never leaves the GPU.
//
// dispatch: (ceil(total_smooth/64), 1, 1), workgroup_size(64).
// each thread owns one timestep. binary-searches in_off[] to find its
// article so we don't need per-article dispatches.

@group(0) @binding(0) var<storage, read>       in_traj  : array<f32>;
@group(0) @binding(1) var<storage, read>       in_off   : array<u32>;
@group(0) @binding(2) var<storage, read_write> err      : array<f32>;
@group(0) @binding(3) var<storage, read_write> vel_norm : array<f32>;
@group(0) @binding(4) var<storage, read_write> pwr      : array<f32>;
@group(0) @binding(5) var<uniform>             params   : vec4<u32>;
// params.x = D
// params.y = n_articles

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t_global = gid.x;
    let D = params.x;
    let n_articles = params.y;

    let total_smooth = in_off[n_articles];
    if (t_global >= total_smooth) { return; }

    // find the article this timestep belongs to. in_off is cumulative:
    // article a owns [in_off[a], in_off[a+1]).
    var lo : u32 = 0u;
    var hi : u32 = n_articles;
    while (lo + 1u < hi) {
        let mid = (lo + hi) >> 1u;
        if (in_off[mid] <= t_global) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let a = lo;
    let local_t = t_global - in_off[a];
    let art_base = in_off[a] * D;
    let t_base = art_base + local_t * D;

    let can_vel = local_t >= 1u;
    let can_err = local_t >= 2u;

    // one pass over d. three accumulators. the v1/v2 reads are only
    // issued when they'll actually be used — most timesteps take all
    // three branches, the first two per article take the short path.
    var sum_sq : f32 = 0.0;
    var vel_sq : f32 = 0.0;
    var err_sq : f32 = 0.0;

    if (can_err) {
        let t_base_1 = art_base + (local_t - 1u) * D;
        let t_base_2 = art_base + (local_t - 2u) * D;
        for (var d : u32 = 0u; d < D; d = d + 1u) {
            let v0 = in_traj[t_base + d];
            let v1 = in_traj[t_base_1 + d];
            let v2 = in_traj[t_base_2 + d];
            sum_sq = sum_sq + v0 * v0;
            let dv = v0 - v1;
            vel_sq = vel_sq + dv * dv;
            let e = v0 - 2.0 * v1 + v2;
            err_sq = err_sq + e * e;
        }
    } else if (can_vel) {
        let t_base_1 = art_base + (local_t - 1u) * D;
        for (var d : u32 = 0u; d < D; d = d + 1u) {
            let v0 = in_traj[t_base + d];
            let v1 = in_traj[t_base_1 + d];
            sum_sq = sum_sq + v0 * v0;
            let dv = v0 - v1;
            vel_sq = vel_sq + dv * dv;
        }
    } else {
        for (var d : u32 = 0u; d < D; d = d + 1u) {
            let v0 = in_traj[t_base + d];
            sum_sq = sum_sq + v0 * v0;
        }
    }

    pwr[t_global] = sum_sq;
    vel_norm[t_global] = select(0.0, sqrt(vel_sq), can_vel);
    err[t_global] = select(0.0, sqrt(err_sq), can_err);
}
