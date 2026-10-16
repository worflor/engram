"""well_simulate.py -- each well is a complete AR(2) oscillator. let it dream.

before the physics completion (format v5), wells stored only K-centroids.
we could route observations to nearest well but a well had no dynamics
of its own. with (K, G, S) per well, every well IS a dynamical system:
eigenvalues of λ² - K*λ + G = 0 determine its spectral character, and S
gives its residual noise floor.

per-well simulation: seed the well's oscillator and spin it up.

    z[0], z[1] = seeds
    z[n] = K_cen * z[n-1] - G_cen * z[n-2]    for n >= 2

this produces a complex P-dim trajectory over n_steps — the literal
signal the well's oscillator would generate from the seeds. for |λ|<1
the trajectory decays toward the well's attractor; for |λ|≈1 it
persists; for |λ|>1 it diverges (shouldn't happen on a stable brain,
but we return it truthfully).

for byte-level Sources the trajectory can be decoded back to bytes
via nearest-neighbor over the Source's embedding matrix. that's each
well's voice — what a well "says" given a starting impulse.

core claim: different wells with different K,G have different
dynamics, so their dreams diverge. verify numerically before trusting
the decoded byte sequences.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from engram_codec import predict_all, eigenvalues_all, _from_complex


def per_well_oscillator(ef, well_name: str) -> dict:
    """extract a well's AR(2) oscillator parameters.

    returns a dict with:
      K           — complex[P]  the well's K centroid (K_sum/count)
      G           — complex[P]  the well's G centroid (G_sum/count)
      sigma       — float       per-pair residual rms geomean (noise floor)
      eigenvalues — (λ1, λ2)    per-pair complex roots
      |λ|_max     — float       max magnitude, >= 1 = unstable
      stable      — bool        all pairs strictly stable
      count       — int         observation count (maturity)
    """
    if well_name not in ef._well_sum_K:
        raise KeyError(f"well {well_name!r} not in ef")
    count = max(int(ef._well_count.get(well_name, 1)), 1)
    K_cen = (ef._well_sum_K[well_name] / count).astype(np.complex128)
    G_sum = ef._well_sum_G.get(well_name)
    if G_sum is None or np.abs(G_sum).sum() < 1e-12:
        # v4-loaded well or never populated via unified path; G = 0
        G_cen = np.zeros_like(K_cen)
    else:
        G_cen = (G_sum / count).astype(np.complex128)
    S_sum = ef._well_sum_S.get(well_name)
    if S_sum is None or np.abs(S_sum).sum() < 1e-12:
        sigma = 0.0
    else:
        sigma = float(np.sqrt(np.mean(np.abs(S_sum) / count)))

    lam1, lam2 = eigenvalues_all(K_cen, G_cen)
    abs_max = float(np.maximum(np.abs(lam1), np.abs(lam2)).max())

    return {
        "K": K_cen,
        "G": G_cen,
        "sigma": sigma,
        "eigenvalues": (lam1, lam2),
        "|λ|_max": abs_max,
        "stable": bool(abs_max < 1.0),
        "count": count,
    }


def dream_well(
    ef,
    well_name: str,
    n_steps: int = 32,
    seeds: Optional[str] = "centroid",
    seed1: Optional[np.ndarray] = None,
    seed2: Optional[np.ndarray] = None,
    rng_seed: int = 0,
) -> np.ndarray:
    """spin up a well's oscillator for n_steps and return the generated
    complex [n_steps, P] trajectory.

    seeds control the initial conditions:
      "centroid" — both s1 and s2 are the well's K centroid. the
                   oscillator starts AT the attractor; trajectory
                   shows how it responds to being perturbed from
                   steady state.
      "noise"    — small gaussian noise per pair (magnitude matched
                   to well's sigma). shows the well's impulse
                   response — pure eigenmode dynamics.
      "explicit" — caller passes seed1, seed2 directly. used for
                   "continue this observation" style dreams where
                   seeds come from the last observed state.

    for stable wells (|λ|<1) the trajectory decays; for wells near the
    stability boundary it persists; for unstable wells it diverges.
    we return the raw trajectory either way — the caller can check
    |λ|_max from per_well_oscillator() if they care.
    """
    osc = per_well_oscillator(ef, well_name)
    K = osc["K"]
    G = osc["G"]
    P = K.shape[0]

    if seeds == "centroid":
        s1 = K.copy()
        s2 = K.copy()
    elif seeds == "noise":
        rng = np.random.default_rng(rng_seed)
        scale = max(osc["sigma"], 1e-6)
        s1 = (rng.standard_normal(P) + 1j * rng.standard_normal(P)) * scale
        s2 = (rng.standard_normal(P) + 1j * rng.standard_normal(P)) * scale
    elif seeds == "explicit":
        if seed1 is None or seed2 is None:
            raise ValueError("seeds='explicit' requires seed1 and seed2")
        s1 = np.asarray(seed1, dtype=np.complex128)
        s2 = np.asarray(seed2, dtype=np.complex128)
    else:
        raise ValueError(f"unknown seeds mode: {seeds!r}")

    return predict_all(s1, s2, K, G, length=n_steps)


def dream_well_bytes(
    ef,
    well_name: str,
    source_matrix: np.ndarray,
    n_steps: int = 32,
    seeds: str = "centroid",
    rng_seed: int = 0,
) -> np.ndarray:
    """decode a well's dream trajectory to bytes via nearest-neighbor.

    source_matrix is the Source's [256, D] byte embedding matrix (real
    float32). the dream trajectory is complex [n_steps, P]; convert to
    real [n_steps, D] via _from_complex and find the nearest byte per
    step. returns int32[n_steps] of byte values.

    caveat: the K-space complex trajectory lives in eigenvalue-dynamics
    space, not byte-embedding space directly. _from_complex just
    interleaves real/imag parts — that's a FAITHFUL inverse of the
    pairing used at ingest time, but it assumes the trajectory is on
    the same manifold as the original trajectory. for a stable well
    the dream should land near the well's centroid in embedding space;
    nearest-byte decode then returns bytes that are characteristic of
    that centroid's region.
    """
    traj_complex = dream_well(
        ef, well_name, n_steps=n_steps, seeds=seeds, rng_seed=rng_seed)
    traj_real = _from_complex(traj_complex)  # [n_steps, D]

    D = source_matrix.shape[1]
    if traj_real.shape[1] != D:
        raise ValueError(
            f"source matrix D={D} doesn't match trajectory D="
            f"{traj_real.shape[1]}")

    # nearest-byte per step via squared-distance argmin. vectorize:
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    # for each step we want argmin over the 256 byte rows.
    M = source_matrix.astype(np.float32)
    M_sq = (M * M).sum(axis=1)  # [256]
    T_real = traj_real.astype(np.float32)
    T_sq = (T_real * T_real).sum(axis=1)  # [n_steps]
    inner = T_real @ M.T  # [n_steps, 256]
    dists_sq = T_sq[:, None] + M_sq[None, :] - 2 * inner
    return np.argmin(dists_sq, axis=1).astype(np.int32)


def dream_well_byte_distribution(
    ef,
    well_name: str,
    source_matrix: np.ndarray,
    *,
    n_steps: int = 256,
    n_samples: int = 8,
    softmax_temp: Optional[float] = None,
) -> np.ndarray:
    """aggregate a well's byte dream into a 256-dim byte distribution.

    two decoding modes, selected by softmax_temp:

      softmax_temp=None — hard nearest-byte per step. counts bytes
        that decode to the trajectory's closest row in source_matrix.
        fast but loses information about "nearly equal" bytes and
        collapses to a tight set when the trajectory stays near
        one embedding cluster (slow-decay oscillators always end
        up here, giving degenerate distributions).

      softmax_temp=<float> — soft decode via softmax over negative
        squared distances per step, scaled by 1/softmax_temp. each
        trajectory point contributes a probability DISTRIBUTION over
        bytes, not a single winner. aggregating these gives a
        smoother byte signature that preserves the "shape" of the
        trajectory's neighborhood in embedding space — which is
        what we want for discrimination between wells whose
        centroids are close but whose eigenvalue PHASES differ.

    noise seeds give different trajectories each call; averaging
    over n_samples reveals the stationary distribution.
    """
    if softmax_temp is None:
        hist = np.zeros(256, dtype=np.float64)
        for s in range(n_samples):
            bytes_seq = dream_well_bytes(
                ef, well_name, source_matrix,
                n_steps=n_steps, seeds="noise", rng_seed=s)
            binc = np.bincount(bytes_seq, minlength=256).astype(np.float64)
            hist += binc
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist

    # softmax decode: per trajectory point, compute a probability
    # distribution over bytes via softmax(-d² / temp), then average.
    M = source_matrix.astype(np.float32)
    M_sq = (M * M).sum(axis=1)  # [256]
    hist = np.zeros(256, dtype=np.float64)
    for s in range(n_samples):
        traj_complex = dream_well(
            ef, well_name, n_steps=n_steps, seeds="noise", rng_seed=s)
        traj_real = _from_complex(traj_complex).astype(np.float32)  # [n_steps, D]
        # distance²: ||t - b||² = ||t||² + ||b||² - 2 t·b
        T_sq = (traj_real * traj_real).sum(axis=1)  # [n_steps]
        inner = traj_real @ M.T  # [n_steps, 256]
        dists_sq = T_sq[:, None] + M_sq[None, :] - 2 * inner  # [n_steps, 256]
        # softmax(-d²/τ) per step. subtract row min for numerical
        # stability before the exp.
        scores = -dists_sq.astype(np.float64) / float(softmax_temp)
        scores -= scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=1, keepdims=True) + 1e-20
        hist += probs.sum(axis=0)  # sum over time steps
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist
