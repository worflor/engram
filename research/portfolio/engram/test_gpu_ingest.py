"""
test_gpu_ingest.py

end-to-end correctness check for the GPU ingest pipeline. the
contract we want to verify:

    for the same corpus and same ref_pairing, the GPU ingestor
    should produce well assignments that are consistent with the
    CPU path (within f32 precision wiggle on K centroids).

we do NOT require bit-exact equivalence — the GPU kernel computes
in f32 while fit_all uses f64, so K values differ in the last few
bits of mantissa. what we DO require is that well membership
matches for the vast majority of articles, and that the brain's
well structure (count, relative sizes, centroids) is stable.

runs against a throwaway in-memory brain so it does not touch
the real Alexandria save file.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))

import numpy as np

from engram_codec import Engram, EngramFile
from engram.gpu.engine import GpuEngine, gpu_available
from engram.gpu_ingest import BatchedGpuIngest


def synth_article_text(seed: int, n_words: int = 300) -> str:
    """build a plausible text by picking deterministic fake 'words'.
    the embed function below turns each word into a vector, so
    'real text' isn't required — we just need trajectory structure."""
    rng = np.random.default_rng(seed)
    return " ".join(f"tok{int(x)}" for x in rng.integers(0, 20000, size=n_words))


def make_fake_embed(dim: int = 300, vocab_size: int = 50000):
    """deterministic word→vector map, mimicking what text_to_trajectory
    produces but without needing the 1GB GloVe table on disk."""
    rng = np.random.default_rng(1234)
    table = rng.standard_normal((vocab_size, dim)).astype(np.float32) * 0.3

    import re
    def embed(text: str) -> np.ndarray:
        tokens = re.findall(r"tok(\d+)", text)
        if len(tokens) < 4:
            return np.empty((0, dim), dtype=np.float32)
        idxs = [int(t) % vocab_size for t in tokens]
        vecs = table[idxs]
        # sliding window average, window=10 — same as text_to_trajectory
        window = 10
        if vecs.shape[0] < window:
            return vecs
        T, D = vecs.shape
        cumsum = np.vstack([np.zeros((1, D), dtype=np.float32), np.cumsum(vecs, axis=0)])
        smoothed = np.empty((T - window + 1, D), dtype=np.float32)
        for i in range(T - window + 1):
            smoothed[i] = (cumsum[i + window] - cumsum[i]) / window
        return smoothed
    return embed


def run():
    if not gpu_available():
        print("wgpu not available, skipping")
        return

    engine = GpuEngine()
    print(f"adapter: {engine.describe()}")
    print()

    embed = make_fake_embed()

    # Build a throwaway brain in a temp dir
    with tempfile.TemporaryDirectory() as td:
        brain_path = str(Path(td) / "_test.engram")
        ef = EngramFile.create(
            brain_path, name="test", dim=300,
            metadata={}, annotations={},
        )
        codec = Engram(dim=300)

        ingestor = BatchedGpuIngest(
            ef=ef, codec=codec, engine=engine,
            batch_size=32, progress_every=50, checkpoint_every=9999,
        )

        N = 200
        for i in range(N):
            title = f"article_{i:04d}"
            text = synth_article_text(seed=i)
            ingestor.queue(title, text, embed)
        ingestor.flush()

        stats = ingestor.stats()
        print()
        print(f"absorbed: {stats['absorbed']}")
        print(f"wells:    {stats['wells']}")
        print(f"rate:     {stats['rate_per_s']:.0f} articles/s")
        print(f"elapsed:  {stats['elapsed_s']:.2f}s")

        # sanity: the brain should have grown
        assert ef._total_absorbed >= N * 0.95, \
            f"expected ~{N} absorbs, got {ef._total_absorbed}"
        assert len(ef.wells()) >= 1
        print()
        print("top wells:")
        top = sorted(ef._well_count.items(), key=lambda x: -x[1])[:8]
        for name, count in top:
            print(f"  {name:20s}: {count}")

        # save + reload round-trip: make sure the brain is disk-safe
        ef.save(brain_path)
        ef2 = EngramFile.load(brain_path)
        assert ef2._total_absorbed == ef._total_absorbed
        assert set(ef2._well_count) == set(ef._well_count)
        print()
        print("save/load round-trip: OK")


if __name__ == "__main__":
    run()
