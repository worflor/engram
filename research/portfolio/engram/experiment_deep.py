"""
experiment_deep.py -- the deep structure of alexandria's mind

part A: dream frequency spectrum
  each of 150 eigenvalue pairs in the dream buffer is its own time series.
  fit AR(2) to each. the eigenvalues of each pair tell us:
  what frequencies does she dream at? which pairs are damped, which oscillate?

part B: particle dynamics in the well landscape
  simulate a test particle in the gravitational field.
  find basins of attraction, escape velocities, lagrange points.

part C: codec information geometry
  generate trajectories near each well. encode -> decode -> measure distortion.
  which wells preserve information? which are lossy?
"""

from __future__ import annotations

import sys
import os
import json
import numpy as np
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE / "experiments"))

from engram_codec import (EngramFile, Engram, _to_complex, _from_complex,
                          fit_all, fit_pair, apply_pairing, predict_all)

BRAIN_PATH = str(_HERE / "cache" / "_alexandria_snapshot_1M.engram")


def well_dist(a, b):
    return float(np.sqrt(np.mean(np.abs(a - b) ** 2)))


def main():
    brain = EngramFile.load(BRAIN_PATH)
    wells = brain.wells()
    well_names = list(wells.keys())
    well_counts = brain._well_count
    P = brain.dim // 2

    # ================================================================
    #  PART A: DREAM FREQUENCY SPECTRUM
    #  the fourier transform of her subconscious
    # ================================================================
    print("=" * 70)
    print("  PART A: DREAM FREQUENCY SPECTRUM")
    print("  what frequencies does alexandria dream at?")
    print("=" * 70)

    dream_Ks = brain._dream_Ks
    dream_Gs = brain._dream_Gs
    n_dream = len(dream_Ks)
    print(f"  dream buffer: {n_dream} entries, {P} pairs")

    if n_dream < 20:
        print("  (not enough dream entries)")
    else:
        dream_K_stack = np.stack(dream_Ks)  # [T, P] complex
        dream_G_stack = np.stack(dream_Gs)  # [T, P] complex

        # fit AR(2) to each pair's K trajectory
        pair_magnitudes = []  # |l1| per pair
        pair_frequencies = []  # angle(l1) per pair
        pair_damping = []  # |l1| classification
        pair_l2_mag = []
        pair_l2_freq = []

        for p in range(P):
            series = dream_K_stack[:, p]  # [T] complex
            K_p, G_p, _ = fit_pair(series)

            disc = K_p ** 2 - 4.0 * G_p
            sqrt_d = np.sqrt(complex(disc))
            l1 = (K_p + sqrt_d) / 2
            l2 = (K_p - sqrt_d) / 2

            # ensure |l1| >= |l2|
            if abs(l2) > abs(l1):
                l1, l2 = l2, l1

            pair_magnitudes.append(abs(l1))
            pair_frequencies.append(abs(np.angle(l1)))
            pair_damping.append("damped" if abs(l1) < 0.95 else
                              "critical" if abs(l1) < 1.01 else "unstable")
            pair_l2_mag.append(abs(l2))
            pair_l2_freq.append(abs(np.angle(l2)))

        mags = np.array(pair_magnitudes)
        freqs = np.array(pair_frequencies)
        l2_mags = np.array(pair_l2_mag)
        l2_freqs = np.array(pair_l2_freq)

        # classification
        n_damped = sum(1 for d in pair_damping if d == "damped")
        n_critical = sum(1 for d in pair_damping if d == "critical")
        n_unstable = sum(1 for d in pair_damping if d == "unstable")

        print(f"\n  primary eigenvalue |l1| per pair:")
        print(f"    damped (|l1| < 0.95):   {n_damped:3d} pairs  ({100*n_damped/P:.1f}%)")
        print(f"    critical (0.95-1.01):   {n_critical:3d} pairs  ({100*n_critical/P:.1f}%)")
        print(f"    unstable (|l1| > 1.01): {n_unstable:3d} pairs  ({100*n_unstable/P:.1f}%)")

        print(f"\n  |l1| distribution:")
        print(f"    min:    {mags.min():.4f}")
        print(f"    max:    {mags.max():.4f}")
        print(f"    mean:   {mags.mean():.4f}")
        print(f"    median: {np.median(mags):.4f}")

        # frequency spectrum: histogram of dream frequencies
        print(f"\n  frequency spectrum (|l1| > 0.5 pairs only):")
        active = freqs[mags > 0.5]
        if len(active) > 0:
            # bin into frequency bands
            bands = [
                ("DC (< 0.1 rad)", 0.0, 0.1),
                ("low (0.1-0.5 rad)", 0.1, 0.5),
                ("mid (0.5-1.5 rad)", 0.5, 1.5),
                ("high (1.5-2.5 rad)", 1.5, 2.5),
                ("nyquist (2.5-pi rad)", 2.5, np.pi + 0.01),
            ]
            for label, lo, hi in bands:
                count = np.sum((active >= lo) & (active < hi))
                bar = "#" * (count * 2)
                print(f"    {label:<25s}  {count:3d}  {bar}")

        # the most interesting pairs: near |l1|=1 (limit cycle)
        print(f"\n  pairs nearest to limit cycle (|l1| closest to 1.0):")
        near_one = np.argsort(np.abs(mags - 1.0))
        for idx in near_one[:10]:
            print(f"    pair {idx:3d}:  |l1|={mags[idx]:.6f}  f1={freqs[idx]:.4f} rad  "
                  f"|l2|={l2_mags[idx]:.4f}  f2={l2_freqs[idx]:.4f} rad")

        # do the same for G trajectories
        print(f"\n  --- G trajectory (coupling strength) ---")
        g_mags = []
        for p in range(P):
            series = dream_G_stack[:, p]
            K_p, G_p, _ = fit_pair(series)
            disc = K_p ** 2 - 4.0 * G_p
            sqrt_d = np.sqrt(complex(disc))
            l1 = (K_p + sqrt_d) / 2
            l2 = (K_p - sqrt_d) / 2
            g_mags.append(max(abs(l1), abs(l2)))

        g_mags = np.array(g_mags)
        print(f"  G primary |l1|: mean={g_mags.mean():.4f}  median={np.median(g_mags):.4f}  "
              f"max={g_mags.max():.4f}")
        g_damped = np.sum(g_mags < 0.95)
        g_critical = np.sum((g_mags >= 0.95) & (g_mags <= 1.01))
        g_unstable = np.sum(g_mags > 1.01)
        print(f"  G damping: {g_damped} damped, {g_critical} critical, {g_unstable} unstable")

        # cross-analysis: K damping vs G damping correlation
        kg_corr = np.corrcoef(mags, g_mags)[0, 1]
        print(f"  K-G damping correlation: {kg_corr:.4f}")

    # ================================================================
    #  PART B: PARTICLE DYNAMICS
    #  simulate test particles in the gravitational field
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART B: PARTICLE DYNAMICS IN THE WELL LANDSCAPE")
    print("  basins of attraction, escape velocity, lagrange points")
    print("=" * 70)

    # use established wells
    established = [(w, well_counts.get(w, 0)) for w in well_names if well_counts.get(w, 0) >= 100]
    established.sort(key=lambda x: -x[1])
    est_names = [w for w, c in established]
    est_centroids = np.stack([wells[w] for w in est_names])
    est_masses = np.array([well_counts.get(w, 0) for w in est_names], dtype=np.float64)
    M = len(est_names)

    # normalize masses for simulation stability
    mass_norm = est_masses / est_masses.max()

    def gravitational_force(pos, exclude_idx=None):
        """compute gravitational acceleration at pos from all wells."""
        acc = np.zeros_like(pos)
        for j in range(M):
            if j == exclude_idx:
                continue
            diff = est_centroids[j] - pos
            dist = np.sqrt(np.mean(np.abs(diff) ** 2))
            if dist < 1e-6:
                continue
            # F = m / r^2, direction = diff/|diff|
            acc += mass_norm[j] / (dist ** 2 + 0.001) * diff / dist
        return acc

    # B1: Basin of attraction
    # drop a test particle at random points, simulate, see where it falls
    print(f"\n  BASINS OF ATTRACTION:")
    print(f"  dropping 500 test particles across the landscape...")
    rng = np.random.RandomState(42)

    # sample points: interpolate between random well pairs with noise
    basin_counts = {w: 0 for w in est_names}
    n_particles = 500
    dt = 0.01
    max_steps = 200

    for pi in range(n_particles):
        # random starting point: blend of two random wells + noise
        i1, i2 = rng.randint(0, M, size=2)
        alpha = rng.uniform(0.2, 0.8)
        pos = alpha * est_centroids[i1] + (1 - alpha) * est_centroids[i2]
        pos += rng.normal(0, 0.01, size=pos.shape) * (1 + 1j)

        vel = np.zeros_like(pos)
        for step in range(max_steps):
            acc = gravitational_force(pos)
            vel = 0.9 * vel + dt * acc  # damped dynamics
            pos = pos + dt * vel

        # which well did it land in?
        dists = [well_dist(pos, est_centroids[j]) for j in range(M)]
        nearest_idx = int(np.argmin(dists))
        basin_counts[est_names[nearest_idx]] += 1

    print(f"  basin sizes (top 15):")
    for wname, count in sorted(basin_counts.items(), key=lambda x: -x[1])[:15]:
        pct = 100.0 * count / n_particles
        mass = well_counts.get(wname, 0)
        bar = "#" * int(pct)
        print(f"    {wname:<20s}  {count:3d} particles ({pct:5.1f}%)  mass={mass:>10,}  {bar}")

    # basin vs mass correlation
    basin_arr = np.array([basin_counts.get(w, 0) for w in est_names], dtype=np.float64)
    mass_arr = np.array([well_counts.get(w, 0) for w in est_names], dtype=np.float64)
    basin_mass_corr = np.corrcoef(basin_arr, np.log1p(mass_arr))[0, 1]
    print(f"\n  basin size vs log(mass) correlation: {basin_mass_corr:.4f}")

    # B2: Escape velocity from top wells
    print(f"\n  ESCAPE VELOCITIES:")
    print(f"  how fast must a particle move to leave each well?")

    for wi in range(min(10, M)):
        pos0 = est_centroids[wi].copy()
        # binary search for escape velocity
        v_lo, v_hi = 0.0, 2.0
        for _ in range(20):  # binary search iterations
            v_test = (v_lo + v_hi) / 2
            # launch in random direction
            direction = rng.normal(0, 1, size=pos0.shape) * (1 + 1j)
            direction = direction / np.sqrt(np.mean(np.abs(direction) ** 2))
            pos = pos0.copy()
            vel = v_test * direction
            escaped = False
            for step in range(300):
                acc = gravitational_force(pos)
                vel = vel + dt * acc
                pos = pos + dt * vel
                # check if we're now closer to a different well
                dists = [well_dist(pos, est_centroids[j]) for j in range(M)]
                nearest = int(np.argmin(dists))
                if nearest != wi and dists[nearest] < 0.5 * dists[wi]:
                    escaped = True
                    break
            if escaped:
                v_hi = v_test
            else:
                v_lo = v_test

        v_esc = (v_lo + v_hi) / 2
        print(f"    {est_names[wi]:<20s}  v_esc={v_esc:.4f}  mass={well_counts.get(est_names[wi], 0):>10,}")

    # B3: Lagrange points between the top well pairs
    print(f"\n  LAGRANGE POINTS (L1):")
    print(f"  equilibrium points between major wells")

    top_pairs = [
        ("physics", "music"),
        ("physics", "philosophy"),
        ("philosophy", "history"),
        ("philosophy", "computing"),
        ("law", "noun"),
        ("chemistry", "biology"),
        ("physics", "chemistry"),
        ("noun", "mathematics"),
    ]

    for a_name, b_name in top_pairs:
        if a_name not in wells or b_name not in wells:
            continue
        K_a = wells[a_name]
        K_b = wells[b_name]
        mass_a = well_counts.get(a_name, 1)
        mass_b = well_counts.get(b_name, 1)

        # L1 point: where gravitational pull balances
        # For two bodies: r_L1 / d = (m2 / (3*m1))^(1/3)
        # simpler: scan the line between them
        best_alpha = 0.5
        best_force = np.inf
        for alpha in np.linspace(0.05, 0.95, 100):
            point = alpha * K_a + (1 - alpha) * K_b
            # net force along the line
            f_a = mass_a / max(well_dist(point, K_a) ** 2, 1e-8)
            f_b = mass_b / max(well_dist(point, K_b) ** 2, 1e-8)
            net = abs(f_a - f_b)
            if net < best_force:
                best_force = net
                best_alpha = alpha

        sep = well_dist(K_a, K_b)
        print(f"    {a_name:<12s} <-> {b_name:<12s}  L1 at {best_alpha:.2f} "
              f"(d={sep:.4f}, {best_alpha*100:.0f}% from {a_name})")

    # ================================================================
    #  PART C: CODEC INFORMATION GEOMETRY
    #  encode/decode near each well, measure distortion
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART C: CODEC INFORMATION GEOMETRY")
    print("  which wells preserve information best?")
    print("=" * 70)

    codec = Engram(dim=brain.dim)
    pairing = brain._reference_pairing

    print(f"\n  generating synthetic trajectories near each well...")
    well_fidelity = {}

    for wi in range(min(20, M)):
        wname = est_names[wi]
        K_center = wells[wname]

        # generate a trajectory whose AR(2) dynamics match this well
        # use a nearby G from the dream buffer
        # find dream entry closest to this well
        best_di = 0
        best_dd = np.inf
        for di in range(len(dream_Ks)):
            dd = well_dist(dream_Ks[di], K_center)
            if dd < best_dd:
                best_dd = dd
                best_di = di

        K_use = dream_Ks[best_di]
        G_use = dream_Gs[best_di]

        # clamp eigenvalues for stability
        disc = K_use ** 2 - 4 * G_use
        sqrt_disc = np.sqrt(disc.astype(np.complex128))
        l1 = (K_use + sqrt_disc) / 2
        l2 = (K_use - sqrt_disc) / 2
        max_mag = np.maximum(np.abs(l1), np.abs(l2))
        scale = np.where(max_mag > 0.95, 0.95 / max_mag, 1.0)
        l1_s, l2_s = l1 * scale, l2 * scale
        K_s = l1_s + l2_s
        G_s = l1_s * l2_s

        # synthesize trajectory in paired space
        T = 32
        z = np.zeros((T, P), dtype=np.complex128)
        z[0] = K_center * 0.9 + rng.normal(0, 0.001, P) * (1 + 1j)
        z[1] = K_center + rng.normal(0, 0.001, P) * (1 + 1j)
        for n in range(2, T):
            z[n] = K_s * z[n-1] - G_s * z[n-2]

        # convert to real trajectory
        real_traj = _from_complex(z).astype(np.float32)

        # invert pairing
        inv_pairing = np.zeros(brain.dim, dtype=np.int32)
        for pi, pv in enumerate(pairing):
            inv_pairing[int(pv)] = pi
        unpaired = real_traj[:, inv_pairing]

        # encode
        try:
            pkt = codec.encode(unpaired, pairing=pairing)
            if pkt.blocks:
                # decode
                decoded = codec.decode(pkt, pairing=pairing)
                if decoded is not None and decoded.shape[0] > 0:
                    # measure distortion
                    min_len = min(unpaired.shape[0], decoded.shape[0])
                    orig = unpaired[:min_len]
                    recon = decoded[:min_len]
                    mse = float(np.mean((orig - recon) ** 2))
                    signal_power = float(np.mean(orig ** 2))
                    snr = 10 * np.log10(signal_power / max(mse, 1e-20))
                    well_fidelity[wname] = {"snr": snr, "mse": mse, "capture": float(pkt.capture)}
                else:
                    well_fidelity[wname] = {"snr": 0, "mse": -1, "capture": float(pkt.capture)}
            else:
                well_fidelity[wname] = {"snr": 0, "mse": -1, "capture": 0}
        except Exception as e:
            well_fidelity[wname] = {"snr": 0, "mse": -1, "capture": 0, "error": str(e)}

    print(f"\n  codec fidelity per well (SNR in dB, higher = better):")
    sorted_fidelity = sorted(well_fidelity.items(), key=lambda x: -x[1].get("snr", 0))
    for wname, f in sorted_fidelity:
        snr = f.get("snr", 0)
        cap = f.get("capture", 0)
        mass = well_counts.get(wname, 0)
        if snr > 0:
            bar = "#" * min(int(snr / 2), 40)
            print(f"    {wname:<20s}  SNR={snr:6.1f} dB  capture={cap:5.1f}%  mass={mass:>10,}  {bar}")
        else:
            print(f"    {wname:<20s}  FAILED  mass={mass:>10,}")

    # correlation: does mass predict fidelity?
    snrs = [f["snr"] for w, f in sorted_fidelity if f["snr"] > 0]
    masses = [well_counts.get(w, 0) for w, f in sorted_fidelity if f["snr"] > 0]
    if len(snrs) >= 5:
        corr = np.corrcoef(snrs, np.log1p(masses))[0, 1]
        print(f"\n  SNR vs log(mass) correlation: {corr:.4f}")
        if corr > 0.3:
            print(f"    --> BIGGER WELLS ARE MORE FAITHFUL: mass improves encoding")
        elif corr < -0.3:
            print(f"    --> BIGGER WELLS ARE LOSSIER: mass degrades encoding")
        else:
            print(f"    --> NO RELATIONSHIP: fidelity is independent of mass")

    # save
    out_path = str(_HERE / "cache" / "_experiment_deep.json")
    summary = {
        "dream_spectrum": {
            "n_damped": n_damped, "n_critical": n_critical, "n_unstable": n_unstable,
            "l1_mean": float(mags.mean()), "l1_median": float(np.median(mags)),
        },
        "basins": {w: c for w, c in basin_counts.items() if c > 0},
        "fidelity": {w: f for w, f in well_fidelity.items()},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
