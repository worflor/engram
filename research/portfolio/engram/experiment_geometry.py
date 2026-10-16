"""
experiment_geometry.py -- the shape of alexandria's mind

no GloVe. no model. just the brain file and math.

what is the topology of the well landscape?
what are the brain's own eigenvalues?
is the mass distribution a power law?
"""

from __future__ import annotations

import sys
import os
import json
import numpy as np
from pathlib import Path
from collections import Counter

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE / "experiments"))

from engram_codec import (EngramFile, Engram, _to_complex, fit_all, fit_pair,
                          apply_pairing)

BRAIN_PATH = str(_HERE / "cache" / "_alexandria_snapshot_1M.engram")


def well_dist(a, b):
    """RMS distance between two complex eigenvalue centroids."""
    return float(np.sqrt(np.mean(np.abs(a - b) ** 2)))


def main():
    brain = EngramFile.load(BRAIN_PATH)
    wells = brain.wells()
    well_names = list(wells.keys())
    well_counts = brain._well_count
    N = len(well_names)
    P = brain.dim // 2

    print("=" * 70)
    print("  THE SHAPE OF ALEXANDRIA'S MIND")
    print("=" * 70)
    print(f"  articles: {brain._total_absorbed:,}")
    print(f"  wells: {N}")
    print(f"  eigenvalue dimension: {P} complex pairs")
    print(f"  dream buffer: {len(brain._dream_Ks)} entries")

    # ================================================================
    #  PART 1: MASS DISTRIBUTION
    #  is it zipf? power law? exponential?
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART 1: MASS DISTRIBUTION")
    print("  how is knowledge distributed across wells?")
    print("=" * 70)

    counts = sorted([well_counts.get(w, 0) for w in well_names], reverse=True)
    total_blocks = sum(counts)
    print(f"\n  total blocks: {total_blocks:,}")
    print(f"  top 10 wells:")
    for i, wname in enumerate(sorted(well_names, key=lambda w: -well_counts.get(w, 0))[:10]):
        c = well_counts.get(wname, 0)
        pct = 100.0 * c / total_blocks if total_blocks > 0 else 0
        print(f"    {i+1:3d}. {wname:<20s}  {c:>12,} blocks  ({pct:5.1f}%)")

    # concentration: what fraction of blocks are in top-K wells?
    cum = np.cumsum(counts) / total_blocks if total_blocks > 0 else np.zeros(len(counts))
    for k in [1, 3, 5, 10, 20, 50]:
        if k <= len(cum):
            print(f"  top {k:2d} wells contain {cum[k-1]*100:.1f}% of all blocks")

    # fit power law: log(rank) vs log(count)
    valid_counts = [c for c in counts if c > 0]
    if len(valid_counts) >= 5:
        log_rank = np.log(np.arange(1, len(valid_counts) + 1, dtype=np.float64))
        log_count = np.log(np.array(valid_counts, dtype=np.float64))
        slope, intercept = np.polyfit(log_rank, log_count, 1)
        print(f"\n  power law fit: count ~ rank^{slope:.2f}")
        print(f"    zipf exponent: {-slope:.2f} (zipf=1.0, empirical language~1.07)")
        if -1.3 < slope < -0.7:
            print(f"    --> ZIPFIAN: classic power-law distribution")
        elif slope > -0.7:
            print(f"    --> FLATTER than zipf: knowledge more evenly spread")
        else:
            print(f"    --> STEEPER than zipf: extreme concentration in top wells")

    # ================================================================
    #  PART 2: WELL LANDSCAPE TOPOLOGY
    #  pairwise distances, clustering, diameter
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART 2: WELL LANDSCAPE TOPOLOGY")
    print("  what is the shape of eigenvalue space?")
    print("=" * 70)

    # only analyze wells with significant mass (>= 100 blocks)
    established = [(w, well_counts.get(w, 0)) for w in well_names if well_counts.get(w, 0) >= 100]
    established.sort(key=lambda x: -x[1])
    est_names = [w for w, c in established]
    est_centroids = np.stack([wells[w] for w in est_names])  # [M, P] complex
    M = len(est_names)

    print(f"\n  established wells (>= 100 blocks): {M}")

    # pairwise distance matrix
    dist_matrix = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        for j in range(i + 1, M):
            d = well_dist(est_centroids[i], est_centroids[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # statistics
    triu = dist_matrix[np.triu_indices(M, k=1)]
    print(f"  pairwise distances ({len(triu)} pairs):")
    print(f"    min:    {np.min(triu):.4f}")
    print(f"    max:    {np.max(triu):.4f}  (diameter)")
    print(f"    mean:   {np.mean(triu):.4f}")
    print(f"    median: {np.median(triu):.4f}")
    print(f"    std:    {np.std(triu):.4f}")

    # nearest neighbor for each well
    print(f"\n  nearest neighbors (top 15 by mass):")
    for i in range(min(15, M)):
        dists_i = dist_matrix[i].copy()
        dists_i[i] = np.inf
        nn_idx = np.argmin(dists_i)
        nn_dist = dists_i[nn_idx]
        print(f"    {est_names[i]:<20s} <-> {est_names[nn_idx]:<20s}  d={nn_dist:.4f}")

    # furthest pairs (most different domains)
    print(f"\n  most distant pairs (top 10):")
    flat_idx = np.argsort(triu)[::-1]
    shown = 0
    for fi in flat_idx[:10]:
        # convert flat index back to i,j
        idx_pairs = np.triu_indices(M, k=1)
        i, j = idx_pairs[0][fi], idx_pairs[1][fi]
        print(f"    {est_names[i]:<20s} <-> {est_names[j]:<20s}  d={dist_matrix[i,j]:.4f}")
        shown += 1

    # closest pairs (most similar domains)
    print(f"\n  closest pairs (top 10):")
    for fi in flat_idx[-10:][::-1]:
        idx_pairs = np.triu_indices(M, k=1)
        i, j = idx_pairs[0][fi], idx_pairs[1][fi]
        print(f"    {est_names[i]:<20s} <-> {est_names[j]:<20s}  d={dist_matrix[i,j]:.4f}")

    # ================================================================
    #  PART 3: SPECTRAL ANALYSIS OF THE WELL GRAPH
    #  build adjacency from distances, find eigenvalues of the landscape
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART 3: SPECTRAL ANALYSIS OF THE WELL GRAPH")
    print("  the eigenvalues of the eigenvalues")
    print("=" * 70)

    # Gaussian kernel affinity: A[i,j] = exp(-d^2 / (2*sigma^2))
    sigma = np.median(triu)
    affinity = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(affinity, 0)

    # graph Laplacian: L = D - A
    degree = np.sum(affinity, axis=1)
    laplacian = np.diag(degree) - affinity

    # normalized Laplacian eigenvalues
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-10)))
    L_norm = D_inv_sqrt @ laplacian @ D_inv_sqrt
    eigvals = np.sort(np.real(np.linalg.eigvalsh(L_norm)))

    print(f"\n  sigma (median distance): {sigma:.4f}")
    print(f"  first 15 Laplacian eigenvalues:")
    for i, ev in enumerate(eigvals[:15]):
        bar = "#" * int(min(ev, 2.0) * 30)
        print(f"    l{i:2d} = {ev:8.4f}  {bar}")

    # spectral gap = l1 (first nonzero eigenvalue)
    spectral_gap = eigvals[1] if len(eigvals) > 1 else 0
    print(f"\n  spectral gap (l1): {spectral_gap:.4f}")
    if spectral_gap > 0.1:
        print(f"    --> WELL CONNECTED: wells form a cohesive landscape")
    elif spectral_gap > 0.01:
        print(f"    --> CLUSTERED: wells form distinct groups")
    else:
        print(f"    --> FRAGMENTED: wells are nearly disconnected")

    # number of near-zero eigenvalues ~ number of clusters
    n_clusters = np.sum(eigvals < 0.01)
    print(f"  approximate clusters: {n_clusters}")

    # ================================================================
    #  PART 4: DREAM BUFFER DYNAMICS
    #  the dream buffer is a trajectory. what are ITS eigenvalues?
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART 4: DREAM BUFFER DYNAMICS")
    print("  the brain's own trajectory through eigenvalue space")
    print("=" * 70)

    dream_Ks = brain._dream_Ks
    dream_Gs = brain._dream_Gs
    n_dream = len(dream_Ks)

    if n_dream >= 10:
        # treat dream_Ks as a time series of complex vectors
        # fit AR(2) to the centroid trajectory
        dream_stack = np.stack(dream_Ks)  # [T, P] complex

        # per-pair: fit AR(2) to each eigenvalue pair's trajectory
        print(f"\n  dream buffer: {n_dream} entries, {P} pairs")

        # global centroid trajectory: mean across pairs per timestep
        # (reduces to 1D complex trajectory for meta-analysis)
        dream_mean = np.mean(dream_stack, axis=1)  # [T] complex

        # fit AR(2) to the mean trajectory
        if n_dream >= 20:
            # use last 100 entries for stable fit
            recent = dream_mean[-min(100, n_dream):]
            K_meta, G_meta, _ = fit_pair(recent)
            disc = K_meta ** 2 - 4.0 * G_meta
            l1 = (K_meta + np.sqrt(disc.astype(np.complex128))) / 2
            l2 = (K_meta - np.sqrt(disc.astype(np.complex128))) / 2

            print(f"\n  AR(2) fit of dream centroid trajectory:")
            print(f"    K = {K_meta.real:.6f} + {K_meta.imag:.6f}i")
            print(f"    G = {G_meta.real:.6f} + {G_meta.imag:.6f}i")
            print(f"    l1: |l1| = {abs(l1):.4f}  freq = {np.angle(l1):.4f} rad")
            print(f"    l2: |l2| = {abs(l2):.4f}  freq = {np.angle(l2):.4f} rad")

            if abs(l1) < 1 and abs(l2) < 1:
                period = 2 * np.pi / max(abs(np.angle(l1)), 1e-10)
                print(f"    --> DAMPED: dream trajectory converges")
                print(f"    period ~ {period:.1f} dream entries")
            elif abs(l1) > 1 or abs(l2) > 1:
                print(f"    --> DIVERGENT: dream buffer is expanding")
            else:
                print(f"    --> SUSTAINED: dream buffer oscillates")

        # which wells does the dream buffer visit?
        print(f"\n  dream buffer well visits (last 50):")
        recent_dreams = dream_Ks[-min(50, n_dream):]
        visit_counts = Counter()
        visit_sequence = []
        for K in recent_dreams:
            nearest, dist = brain.nearest_well(K)
            visit_counts[nearest] += 1
            visit_sequence.append(nearest)

        for wname, count in visit_counts.most_common(10):
            pct = 100.0 * count / len(recent_dreams)
            print(f"    {wname:<20s}  {count:3d} visits  ({pct:5.1f}%)")

        # transitions: how often does the dream change wells?
        transitions = sum(1 for i in range(1, len(visit_sequence))
                         if visit_sequence[i] != visit_sequence[i-1])
        print(f"\n  well transitions in last {len(recent_dreams)} dreams: {transitions}")
        print(f"  transition rate: {transitions/len(recent_dreams)*100:.1f}% per dream entry")

        # dream drift: distance from global centroid over time
        gc = brain.global_centroid()
        if gc is not None:
            drifts = [float(np.sqrt(np.mean(np.abs(K - gc) ** 2))) for K in recent_dreams]
            print(f"\n  dream distance from global centroid:")
            print(f"    mean:  {np.mean(drifts):.4f}")
            print(f"    std:   {np.std(drifts):.4f}")
            print(f"    trend: {np.polyfit(range(len(drifts)), drifts, 1)[0]:+.6f}/entry")

    # ================================================================
    #  PART 5: LYAPUNOV FROM DETERMINISM DATA
    #  how fast do nearby trajectories converge?
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART 5: LYAPUNOV EXPONENT FROM DETERMINISM DATA")
    print("  how fast does the brain pull trajectories together?")
    print("=" * 70)

    det_path = str(_HERE / "cache" / "_experiment_determinism.json")
    if os.path.exists(det_path):
        with open(det_path, "r", encoding="utf-8") as f:
            det_data = json.load(f)

        # extract well_dist trajectories from each run
        runs = []
        for run in det_data:
            dists = [t["well_dist"] for t in run if t["well_dist"] is not None]
            if dists:
                runs.append(np.array(dists))

        if len(runs) >= 2:
            # pairwise divergence at each turn
            n_turns = min(len(r) for r in runs)
            print(f"\n  {len(runs)} runs, {n_turns} turns each")

            # inter-run distance at each turn
            pair_dists = []
            for t in range(n_turns):
                vals = [r[t] for r in runs]
                # mean pairwise distance
                d = 0
                count = 0
                for i in range(len(vals)):
                    for j in range(i+1, len(vals)):
                        d += abs(vals[i] - vals[j])
                        count += 1
                pair_dists.append(d / max(count, 1))

            print(f"  inter-run distance per turn:")
            for t, d in enumerate(pair_dists):
                bar = "#" * int(d * 200)
                print(f"    t{t+1}: {d:.4f}  {bar}")

            # Lyapunov: log(d[t+1]/d[t]) averaged
            if len(pair_dists) >= 3 and pair_dists[0] > 1e-8:
                lyap_samples = []
                for t in range(len(pair_dists) - 1):
                    if pair_dists[t] > 1e-8 and pair_dists[t+1] > 1e-8:
                        lyap_samples.append(np.log(pair_dists[t+1] / pair_dists[t]))

                if lyap_samples:
                    lyap = np.mean(lyap_samples)
                    print(f"\n  Lyapunov exponent: {lyap:.4f}")
                    if lyap < -0.1:
                        half_life = -np.log(2) / lyap
                        print(f"    --> CONTRACTING: trajectories converge")
                        print(f"    half-life: {half_life:.1f} turns (distance halves every {half_life:.1f} turns)")
                    elif lyap > 0.1:
                        print(f"    --> EXPANDING: trajectories diverge (chaotic)")
                    else:
                        print(f"    --> NEUTRAL: trajectories neither converge nor diverge")
    else:
        print(f"  (no determinism data found at {det_path})")

    # ================================================================
    #  PART 6: GRAVITATIONAL FIELD STRENGTH
    #  mass-weighted potential at each well
    # ================================================================
    print("\n" + "=" * 70)
    print("  PART 6: GRAVITATIONAL FIELD MAP")
    print("  the potential landscape of the mind")
    print("=" * 70)

    # for each established well, compute the gravitational potential
    # from all other wells: phi = -sum(mass_j / dist_ij)
    potentials = np.zeros(M, dtype=np.float64)
    for i in range(M):
        for j in range(M):
            if i != j and dist_matrix[i, j] > 1e-8:
                mass_j = well_counts.get(est_names[j], 0)
                potentials[i] -= mass_j / dist_matrix[i, j]

    # normalize
    pot_min, pot_max = potentials.min(), potentials.max()
    pot_range = pot_max - pot_min if pot_max > pot_min else 1

    print(f"\n  gravitational potential (deeper = stronger attractor):")
    ranked = np.argsort(potentials)  # most negative = deepest well
    for idx in ranked[:15]:
        name = est_names[idx]
        mass = well_counts.get(name, 0)
        pot_norm = (potentials[idx] - pot_min) / pot_range  # 0=deepest, 1=shallowest
        bar = "#" * int((1 - pot_norm) * 40)
        print(f"    {name:<20s}  mass={mass:>10,}  phi={potentials[idx]:>12.1f}  {bar}")

    print(f"\n  shallowest wells (easiest to escape):")
    for idx in ranked[-5:][::-1]:
        name = est_names[idx]
        mass = well_counts.get(name, 0)
        print(f"    {name:<20s}  mass={mass:>10,}  phi={potentials[idx]:>12.1f}")

    # escape energy: to leave well i, you need to overcome the potential barrier
    # to the nearest saddle point (the well with the weakest pull)
    print(f"\n  escape energy estimate (potential difference to nearest neighbor):")
    for i in range(min(10, M)):
        dists_i = dist_matrix[i].copy()
        dists_i[i] = np.inf
        nn_idx = np.argmin(dists_i)
        escape_energy = abs(potentials[i] - potentials[nn_idx])
        print(f"    {est_names[i]:<20s} -> {est_names[nn_idx]:<20s}  dE={escape_energy:>10.1f}")

    # save summary
    out_path = str(_HERE / "cache" / "_experiment_geometry.json")
    summary = {
        "n_wells": N,
        "n_established": M,
        "total_blocks": total_blocks,
        "diameter": float(np.max(triu)),
        "mean_dist": float(np.mean(triu)),
        "spectral_gap": float(spectral_gap),
        "n_clusters": int(n_clusters),
        "top_wells": [(est_names[i], int(well_counts.get(est_names[i], 0)))
                      for i in range(min(20, M))],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
