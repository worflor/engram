"""
engram_mirror_test.py

can the codec recognize itself in a mirror?

two loops running in parallel, seeded with different knowledge:
    loop A: biology (photosynthesis, DNA, evolution, ecology, genetics)
    loop B: music (jazz, blues, classical, opera, symphony)

both dream for 10 iterations with residuals.

then the mirror test:
    feed A's dream into A's encoder → measure eigenvalue distance (self)
    feed A's dream into B's encoder → measure eigenvalue distance (other)

if self < other, the system recognizes its own output.
if self ≈ other, it's just oscillating without identity.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from engram_codec import Engram, summary, eigenvalues_all
from engram_glove_test import load_glove, text_to_trajectory, tokenize_simple
from engram_babel_test import fetch_article


def eigenvalue_histogram(packet) -> np.ndarray:
    """spectral distribution fingerprint."""
    mag_edges = np.array([0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99, 1.01, 1.05, 1.1, 1.5, 1e6])
    freq_edges = np.array([0, 0.1, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5, np.pi + 0.01])
    all_mags, all_freqs = [], []
    for blk in packet.blocks:
        if blk.macro_K is not None and len(blk.macro_K) > 0:
            l1, l2 = eigenvalues_all(blk.macro_K, blk.macro_G)
            all_mags.extend(np.abs(l1).tolist())
            all_mags.extend(np.abs(l2).tolist())
            all_freqs.extend(np.abs(np.angle(l1)).tolist())
            all_freqs.extend(np.abs(np.angle(l2)).tolist())
    if not all_mags:
        return np.zeros(19, dtype=np.float32)
    mh, _ = np.histogram(all_mags, bins=mag_edges)
    fh, _ = np.histogram(all_freqs, bins=freq_edges)
    return np.concatenate([mh / max(mh.sum(), 1), fh / max(fh.sum(), 1)]).astype(np.float32)


def full_fingerprint(packet, s) -> np.ndarray:
    """combined fingerprint: block size + capture + eigenvalue histogram."""
    eh = eigenvalue_histogram(packet)
    return np.concatenate([
        np.array([s["macro_block"], s["capture"]], dtype=np.float32),
        eh,
    ])


def cosine(a, b):
    if len(a) < len(b): a = np.pad(a, (0, len(b) - len(a)))
    elif len(b) < len(a): b = np.pad(b, (0, len(a) - len(b)))
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 1e-8 else 0.0


BIOLOGY_ARTICLES = [
    "Photosynthesis", "DNA", "Evolution", "Ecology", "Genetics",
]

MUSIC_ARTICLES = [
    "Jazz", "Blues", "Classical music", "Opera", "Symphony",
]


def run_loop(name: str, titles: list[str], glove: dict, codec: Engram, n_iter: int = 10) -> tuple[list, list]:
    """run a dream loop. returns (packets, dream_trajectories)."""
    print(f"\n  [{name}] seeding...")
    packets = []
    for title in titles:
        text = fetch_article(title, "en")
        traj = text_to_trajectory(text, glove)
        if traj.shape[0] >= 20:
            packets.append(codec.encode(traj))
            print(f"    {title}: {traj.shape[0]} words")

    dreams = []
    print(f"  [{name}] dreaming {n_iter} iterations...")
    for i in range(n_iter):
        gen = codec.dream(packets, n_blocks=6, samples_per_block=12)
        if gen is None or gen.shape[0] < 4:
            print(f"    [{i}] dream returned nothing")
            continue
        pkt = codec.encode(gen)
        s = summary(pkt)
        energy = float(np.sum(gen ** 2))
        print(f"    [{i:2d}] energy={energy:8.1f}, block={s['macro_block']:3d}, capture={s['capture']:5.1f}%")
        packets.append(pkt)
        dreams.append(gen)

    return packets, dreams


def run():
    print("=" * 70)
    print("  ENGRAM MIRROR TEST")
    print("  can the codec recognize itself?")
    print("=" * 70)

    glove = load_glove()
    codec = Engram(dim=300)

    # run both loops
    pkts_A, dreams_A = run_loop("biology", BIOLOGY_ARTICLES, glove, codec)
    pkts_B, dreams_B = run_loop("music", MUSIC_ARTICLES, glove, codec)

    if not dreams_A or not dreams_B:
        print("\n  not enough dreams to test.")
        return

    print()
    print("=" * 70)
    print("  THE MIRROR")
    print("=" * 70)
    print()

    # take the last 3 dreams from each loop
    n_test = min(3, len(dreams_A), len(dreams_B))

    self_scores = []   # A dream encoded by A's pool
    other_scores = []  # A dream encoded by B's pool
    self_B = []        # B dream encoded by B's pool
    other_B = []       # B dream encoded by A's pool

    # fingerprint the pools themselves (average of all packets)
    fp_A = np.mean([full_fingerprint(p, summary(p)) for p in pkts_A], axis=0)
    fp_B = np.mean([full_fingerprint(p, summary(p)) for p in pkts_B], axis=0)

    print(f"  pool A (biology) vs pool B (music) centroid cosine: {cosine(fp_A, fp_B):.4f}")
    print()

    for i in range(n_test):
        dream_A = dreams_A[-(i + 1)]
        dream_B = dreams_B[-(i + 1)]

        # encode A's dream
        pkt_A_self = codec.encode(dream_A)
        fp_A_self = full_fingerprint(pkt_A_self, summary(pkt_A_self))

        # encode B's dream
        pkt_B_self = codec.encode(dream_B)
        fp_B_self = full_fingerprint(pkt_B_self, summary(pkt_B_self))

        # compare against pool centroids
        a_self = cosine(fp_A_self, fp_A)   # A's dream vs A's pool
        a_other = cosine(fp_A_self, fp_B)  # A's dream vs B's pool
        b_self = cosine(fp_B_self, fp_B)   # B's dream vs B's pool
        b_other = cosine(fp_B_self, fp_A)  # B's dream vs A's pool

        self_scores.append(a_self)
        other_scores.append(a_other)
        self_B.append(b_self)
        other_B.append(b_other)

        sa = summary(pkt_A_self)
        sb = summary(pkt_B_self)

        print(f"  dream A[{i}]: block={sa['macro_block']:3d}, capture={sa['capture']:5.1f}%")
        print(f"    A→A (self):  {a_self:.4f}")
        print(f"    A→B (other): {a_other:.4f}")
        print(f"    margin:      {a_self - a_other:+.4f}")
        print()
        print(f"  dream B[{i}]: block={sb['macro_block']:3d}, capture={sb['capture']:5.1f}%")
        print(f"    B→B (self):  {b_self:.4f}")
        print(f"    B→A (other): {b_other:.4f}")
        print(f"    margin:      {b_self - b_other:+.4f}")
        print()

    # verdict
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print()

    mean_self_A = np.mean(self_scores)
    mean_other_A = np.mean(other_scores)
    mean_self_B = np.mean(self_B)
    mean_other_B = np.mean(other_B)

    mean_self = (mean_self_A + mean_self_B) / 2
    mean_other = (mean_other_A + mean_other_B) / 2
    margin = mean_self - mean_other

    print(f"  A dreams → A pool (self):   {mean_self_A:.4f}")
    print(f"  A dreams → B pool (other):  {mean_other_A:.4f}")
    print(f"  B dreams → B pool (self):   {mean_self_B:.4f}")
    print(f"  B dreams → A pool (other):  {mean_other_B:.4f}")
    print()
    print(f"  mean self:  {mean_self:.4f}")
    print(f"  mean other: {mean_other:.4f}")
    print(f"  margin:     {margin:+.4f}")
    print()

    if margin > 0.01:
        print(f"  the codec recognizes itself.")
        print(f"  biology dreams are closer to biology. music dreams are closer to music.")
        print(f"  the loop has identity.")
    elif margin > 0.0:
        print(f"  faint signal. self is slightly closer than other.")
    else:
        print(f"  no self-recognition. the dreams are indistinguishable from each other.")


if __name__ == "__main__":
    run()
