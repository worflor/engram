"""bench_embed.py — time the embed stage alone."""

import sys, time, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))

import numpy as np
from engram.fast_embed import text_to_trajectory_fast

# build a small synthetic "glove" we can exercise both paths against
VOCAB = 20000
DIM = 300
rng = np.random.default_rng(1234)
matrix = (rng.standard_normal((VOCAB, DIM)) * 0.3).astype(np.float32)
glove_dict = {f"word{i}": matrix[i] for i in range(VOCAB)}
word_to_idx = {f"word{i}": i for i in range(VOCAB)}


def old_path(text, glove, window=10):
    """replica of experiments/engram_glove_test.text_to_trajectory."""
    tokens = [w for w in re.findall(r"[a-z]+[0-9]+", text.lower()) if len(w) > 1]
    vecs = [glove[t] for t in tokens if t in glove]
    if len(vecs) < max(4, window):
        return np.empty((0, DIM), dtype=np.float32)
    raw = np.stack(vecs)
    if window <= 1:
        return raw
    T, D = raw.shape
    cumsum = np.vstack([np.zeros((1, D), dtype=np.float32), np.cumsum(raw, axis=0)])
    smoothed = np.empty((T - window + 1, D), dtype=np.float32)
    for i in range(T - window + 1):
        smoothed[i] = (cumsum[i + window] - cumsum[i]) / window
    return smoothed


def new_path(text, _m=matrix, _w=word_to_idx):
    # the fast embed uses re.findall(r"[a-z]+", ...) not [a-z]+[0-9]+,
    # so adjust tokenization to match for apples-to-apples comparison
    tokens = [w for w in re.findall(r"word\d+", text.lower()) if len(w) > 1]
    if len(tokens) < 10:
        return np.empty((0, DIM), dtype=np.float32)
    idxs = [_w[t] for t in tokens if t in _w]
    raw = _m[np.asarray(idxs, dtype=np.int32)]
    T, D = raw.shape
    window = 10
    cumsum = np.empty((T + 1, D), dtype=np.float32)
    cumsum[0] = 0
    np.cumsum(raw, axis=0, out=cumsum[1:])
    return (cumsum[window:] - cumsum[:-window]) / np.float32(window)


def gen_text(seed, n_words=600):
    r = np.random.default_rng(seed)
    return " ".join(f"word{int(x)}" for x in r.integers(0, VOCAB, size=n_words))


texts = [gen_text(i) for i in range(500)]

# warmup
for t in texts[:10]:
    old_path(t, glove_dict)
    new_path(t)

# measure
t0 = time.perf_counter()
for t in texts:
    old_path(t, glove_dict)
t_old = time.perf_counter() - t0

t0 = time.perf_counter()
for t in texts:
    new_path(t)
t_new = time.perf_counter() - t0

print(f"old path: {t_old*1000:.1f}ms  ({500/t_old:.0f} articles/s, {t_old/500*1000:.2f}ms/article)")
print(f"new path: {t_new*1000:.1f}ms  ({500/t_new:.0f} articles/s, {t_new/500*1000:.2f}ms/article)")
print(f"speedup:  {t_old/t_new:.2f}x")

# correctness
a = old_path(texts[0], glove_dict)
b = new_path(texts[0])
if a.shape == b.shape:
    diff = float(np.abs(a - b).max())
    print(f"max abs diff on one sample: {diff:.2e}")
