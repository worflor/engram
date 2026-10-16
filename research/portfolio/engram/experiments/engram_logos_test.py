"""
engram_logos_test.py

ablation study: which Logos axes carry semantic signal for Engram wells?

tests every combination of L, U, X axes to find the sweet spot.
each axis is a different way of listening to byte streams:
    L (255d): intra-byte bit tree. the 8D Möbius lattice.
    U (32d):  AR(2) per bit lane. temporal oscillation.
    X (255d): cross-subspace. prev byte class × bit tree.

for each combination, builds wells from 20 articles across 5 domains
and measures well-matching accuracy.
"""

from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from engram_codec import Engram, EngramFile
from engram_babel_test import fetch_article

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*a, **kw):
        def w(f): return f
        return w


@njit(cache=True)
def _logos_extract(data, stride, use_L, use_U, use_X):
    """extract trajectory from Logos axes. returns [N, D] float32."""
    n_bytes = len(data)
    n_points = n_bytes // stride

    # count dimensions
    dim = 0
    if use_L: dim += 255
    if use_U: dim += 32
    if use_X: dim += 255
    if dim % 2 == 1:
        dim += 1  # pad to even

    lC = np.zeros((256, 2), dtype=np.float64)
    uC = np.zeros((32, 2), dtype=np.float64)
    xC = np.zeros((8192, 2), dtype=np.float64)
    p1 = 0
    p2 = 0

    out = np.zeros((n_points, dim), dtype=np.float32)
    out_idx = 0

    for i in range(n_bytes):
        b = data[i]
        ctx = 1
        for k in range(8):
            bit = (b >> (7 - k)) & 1
            u_slot = (((p1 >> (7 - k)) & 1) << 1) | ((p2 >> (7 - k)) & 1)
            u_idx = u_slot * 8 + k
            x_idx = (p1 >> 3) * 256 + ctx
            lC[ctx, bit] += 1.0
            uC[u_idx, bit] += 1.0
            if x_idx < 8192:
                xC[x_idx, bit] += 1.0
            ctx = (ctx << 1) | bit
        p2 = p1
        p1 = b

        if (i + 1) % stride == 0 and out_idx < n_points:
            col = 0
            if use_L:
                for j in range(1, 256):
                    c0 = lC[j, 0]
                    c1 = lC[j, 1]
                    out[out_idx, col] = np.float32((c0 + 0.5) / (c0 + c1 + 1.0))
                    col += 1
            if use_U:
                for j in range(32):
                    c0 = uC[j, 0]
                    c1 = uC[j, 1]
                    out[out_idx, col] = np.float32((c0 + 0.5) / (c0 + c1 + 1.0))
                    col += 1
            if use_X:
                x_class = p1 >> 3
                for j in range(1, 256):
                    xi = x_class * 256 + j
                    if xi < 8192:
                        c0 = xC[xi, 0]
                        c1 = xC[xi, 1]
                        out[out_idx, col] = np.float32((c0 + 0.5) / (c0 + c1 + 1.0))
                    else:
                        out[out_idx, col] = 0.5
                    col += 1
            out_idx += 1

    return out[:out_idx]


def make_trajectory(text: str, stride: int, use_L: bool, use_U: bool, use_X: bool) -> np.ndarray:
    """text → Logos trajectory with selected axes."""
    data = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
    if len(data) < stride * 10:
        return np.empty((0, 2), dtype=np.float32)
    states = _logos_extract(data, stride, use_L, use_U, use_X)
    if states.shape[0] < 5:
        return np.empty((0, 2), dtype=np.float32)
    return states


ARTICLES = [
    ("Physics", "physics"), ("Quantum mechanics", "physics"),
    ("Classical mechanics", "physics"), ("Thermodynamics", "physics"),
    ("Jazz", "music"), ("Blues", "music"),
    ("Classical music", "music"), ("Opera", "music"),
    ("DNA", "biology"), ("Evolution", "biology"),
    ("Photosynthesis", "biology"), ("Genetics", "biology"),
    ("Algorithm", "cs"), ("Machine learning", "cs"),
    ("Cryptography", "cs"), ("Database", "cs"),
    ("Roman Empire", "history"), ("French Revolution", "history"),
    ("World War II", "history"), ("Cold War", "history"),
]


def test_config(name: str, use_L: bool, use_U: bool, use_X: bool, stride: int = 16):
    """test one axis configuration and return accuracy."""
    # compute dim
    dim = 0
    if use_L: dim += 255
    if use_U: dim += 32
    if use_X: dim += 255
    if dim % 2 == 1: dim += 1
    if dim < 4: return 0, 0, dim

    codec = Engram(dim=dim)
    ef = EngramFile.create(f'/tmp/logos_{name}.engram', name=name, dim=dim)

    for title, domain in ARTICLES:
        text = fetch_article(title, "en")
        if len(text) < 100: continue
        traj = make_trajectory(text, stride, use_L, use_U, use_X)
        if traj.shape[0] < 10: continue
        m = ef.measure(codec, traj)
        if "packet" in m:
            ef.absorb(m["packet"], source=domain)

    # accuracy
    correct = 0
    total = 0
    for title, domain in ARTICLES:
        text = fetch_article(title, "en")
        traj = make_trajectory(text, stride, use_L, use_U, use_X)
        if traj.shape[0] < 10: continue
        m = ef.measure(codec, traj)
        if str(m.get("nearest_well", "?")) == domain:
            correct += 1
        total += 1

    # well separation
    wells = ef.wells()
    gc = ef.global_centroid()
    sep = 0.0
    if gc is not None and len(wells) >= 2:
        scale = float(np.sqrt(np.mean(np.abs(gc) ** 2))) + 1e-7
        dists = []
        wn = sorted(wells.keys())
        for i in range(len(wn)):
            for j in range(i + 1, len(wn)):
                d = float(np.sqrt(np.mean(np.abs(wells[wn[i]] - wells[wn[j]]) ** 2))) / scale * 100
                dists.append(d)
        sep = np.mean(dists) if dists else 0.0

    return correct, total, dim, sep


def run():
    print("=" * 70)
    print("  LOGOS AXIS ABLATION STUDY")
    print("=" * 70)
    print()

    # warmup numba with smallest config
    print("  compiling numba...", end=" ", flush=True)
    _ = _logos_extract(np.array([65, 66, 67] * 20, dtype=np.uint8), 4, True, False, False)
    print("done")
    print()

    STRIDE = 16

    configs = [
        ("L",       True,  False, False),
        ("U",       False, True,  False),
        ("X",       False, False, True),
        ("L+U",     True,  True,  False),
        ("L+X",     True,  False, True),
        ("U+X",     False, True,  True),
        ("L+U+X",   True,  True,  True),
    ]

    print(f"  {'config':10s} {'dim':>5s} {'accuracy':>10s} {'sep%':>8s}")
    print(f"  {'-'*10} {'-'*5} {'-'*10} {'-'*8}")

    results = []
    for name, L, U, X in configs:
        t0 = time.time()
        correct, total, dim, sep = test_config(name, L, U, X, stride=STRIDE)
        dt = time.time() - t0
        acc = correct / total * 100 if total > 0 else 0
        print(f"  {name:10s} {dim:5d} {correct:3d}/{total:<3d} {acc:5.0f}% {sep:7.1f}%  ({dt:.0f}s)")
        results.append((name, dim, acc, sep))

    print()
    print("  === WINNER ===")
    best = max(results, key=lambda r: (r[2], r[3]))
    print(f"  {best[0]}: {best[1]}d, {best[2]:.0f}% accuracy, {best[3]:.1f}% mean separation")
    print()
    print("  for reference: GloVe 300d pretrained = ~88% accuracy")


if __name__ == "__main__":
    run()
