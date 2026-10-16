"""test_genesis_chain_api.py — proves the universal `genesis_chain(ef,
items, source)` entry point fills an engram via the ZOOMIN GPU pipeline.

runs three scenarios:

  1. from_glove_dict on a synthetic random-vector "glove" and a small
     set of fake articles — proves the plumbing works without touching
     real corpora or loading real glove.

  2. two back-to-back genesis_chain() calls on the same ef — proves
     restart / continuation semantics (second call adopts the pairing
     stored on ef._reference_pairing from the first).

  3. from_matrix + custom tokenizer — proves the non-text path works
     (arbitrary integer token sequences over any matrix).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from engram_codec import Engram, EngramFile

from engram.gpu.engine import gpu_available


def fake_glove(V=2000, D=300, seed=42):
    """build a random 'glove' — dict[word → float32[D]]."""
    rng = np.random.default_rng(seed)
    words = [f"w{i:04d}" for i in range(V)]
    vecs = rng.standard_normal((V, D)).astype(np.float32) * 0.3
    return {w: vecs[i] for i, w in enumerate(words)}


def fake_articles(n=120, mean_tokens=500, seed=1):
    """yield (title, text) pairs of random fake words from the fake_glove
    vocab. text is a bag of 'w0042 w1337 ...' strings."""
    rng = np.random.default_rng(seed)
    V = 2000
    for i in range(n):
        T = int(rng.normal(mean_tokens, mean_tokens * 0.2))
        T = max(60, T)
        toks = rng.integers(0, V, size=T)
        text = " ".join(f"w{t:04d}" for t in toks)
        yield f"article_{i:04d}", text


def scenario_1_glove_synthetic():
    print("=" * 60)
    print("SCENARIO 1: from_glove_dict + synthetic corpus")
    print("=" * 60)

    from genesis_chain import genesis_chain, from_glove_dict

    glove = fake_glove(V=2000, D=300)
    src = from_glove_dict(glove, name="fake-glove")
    print(f"source: {src.name}  V={src.vocab_size}  D={src.dim}")

    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "test.engram")
        ef = EngramFile.create(path, name="test", dim=300,
                               metadata={}, annotations={})

        n = genesis_chain(
            ef, fake_articles(n=60), src,
            label="scenario1",
            batch_size=16, batch_timeout_ms=100, n_cpu_workers=2,
            progress_every=999999, telemetry_interval_s=999.0,
        )
        print(f"absorbed {n}/60, wells={len(ef.wells())}")
        assert n > 0, "no articles absorbed"
        assert ef._reference_pairing is not None, "pairing not set"
    print()


def scenario_2_continuation():
    print("=" * 60)
    print("SCENARIO 2: two calls on same ef (continuation)")
    print("=" * 60)

    from genesis_chain import genesis_chain, from_glove_dict

    glove = fake_glove(V=2000, D=300, seed=42)
    src = from_glove_dict(glove, name="fake-glove")

    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "test.engram")
        ef = EngramFile.create(path, name="test", dim=300,
                               metadata={}, annotations={})

        n1 = genesis_chain(
            ef, fake_articles(n=30, seed=1), src,
            label="call1",
            batch_size=16, batch_timeout_ms=100, n_cpu_workers=2,
            progress_every=999999, telemetry_interval_s=999.0,
        )
        pairing1 = ef._reference_pairing.copy()
        wells1 = len(ef.wells())
        absorbed1 = ef._total_absorbed
        print(f"call 1: absorbed={n1} wells={wells1} total={absorbed1}")

        # second call on same ef — should continue filling
        n2 = genesis_chain(
            ef, fake_articles(n=30, seed=2), src,
            label="call2",
            batch_size=16, batch_timeout_ms=100, n_cpu_workers=2,
            progress_every=999999, telemetry_interval_s=999.0,
        )
        pairing2 = ef._reference_pairing.copy()
        wells2 = len(ef.wells())
        absorbed2 = ef._total_absorbed
        print(f"call 2: absorbed={n2} wells={wells2} total={absorbed2}")

        assert np.array_equal(pairing1, pairing2), \
            "pairing drifted across calls — should be stable"
        assert absorbed2 > absorbed1, \
            "second call did not increase ef._total_absorbed"
        print(f"[ok] pairing stable, total grew {absorbed1} -> {absorbed2}")
    print()


def scenario_3_from_matrix():
    print("=" * 60)
    print("SCENARIO 3: from_matrix + custom int-sequence tokenizer")
    print("=" * 60)

    from genesis_chain import genesis_chain, from_matrix

    # synthetic codebook: 512 entries × 300 dim. payloads ARE int
    # sequences (simulating e.g. VQ audio codes, BPE code tokens, etc).
    rng = np.random.default_rng(7)
    V, D = 512, 300
    matrix = rng.standard_normal((V, D)).astype(np.float32) * 0.3

    def tokenize(payload):
        # payload is already an int32 array of indices
        return np.asarray(payload, dtype=np.int32)

    src = from_matrix(matrix, tokenize, name="codebook")
    print(f"source: {src.name}  V={src.vocab_size}  D={src.dim}")

    # generate some fake "code sequences"
    def fake_codes(n=40):
        for i in range(n):
            T = int(rng.integers(200, 500))
            yield f"seq_{i:04d}", rng.integers(0, V, size=T)

    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "test.engram")
        ef = EngramFile.create(path, name="test", dim=300,
                               metadata={}, annotations={})

        n = genesis_chain(
            ef, fake_codes(n=40), src,
            label="codebook",
            batch_size=16, batch_timeout_ms=100, n_cpu_workers=2,
            progress_every=999999, telemetry_interval_s=999.0,
        )
        print(f"absorbed {n}/40, wells={len(ef.wells())}")
        assert n > 0, "no sequences absorbed"
    print()


def main():
    if not gpu_available():
        print("wgpu missing — skipping")
        return
    scenario_1_glove_synthetic()
    scenario_2_continuation()
    scenario_3_from_matrix()
    print("all scenarios passed")


if __name__ == "__main__":
    main()
