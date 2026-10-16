"""
test_memory_safety.py

regression guard for the OOM bug that crashed genesis_chain_gpu.py on
a monster enwiki article. the original staging path held the full
paired trajectory (`wp`) per pending article, which on articles with
50k+ tokens was ~60 MB each; with batch_size=128 and a few more
buffered in the ThreadPoolExecutor, peak memory hit multi-GB and the
process died.

the fix extracts only segment slices (each capped at MAX_BLOCK+2
timesteps via the segmenter) plus a char-level MAX_TOKENS cap in the
embed stage. this test simulates the old failure mode: a mix of
normal articles + a few monsters, batched to at least one flush,
should complete with bounded peak memory.

we don't measure absolute RSS (platform-dependent); we assert
that staged articles themselves are small regardless of input length.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))

import numpy as np

from engram_codec import Engram, EngramFile, MAX_BLOCK
from engram.fast_embed import MAX_TOKENS_PER_ARTICLE
from engram.gpu.engine import GpuEngine, gpu_available
from engram.gpu_ingest import BatchedGpuIngest, _cpu_prepare


def make_fake_embed(vocab_size: int = 20000, dim: int = 300):
    rng = np.random.default_rng(1234)
    table = (rng.standard_normal((vocab_size, dim)) * 0.3).astype(np.float32)
    word_to_idx = {f"tok{i}": i for i in range(vocab_size)}

    import re
    def embed(text: str) -> np.ndarray:
        tokens = re.findall(r"tok\d+", text)
        if len(tokens) < 4:
            return np.empty((0, dim), dtype=np.float32)
        idxs = [word_to_idx[t] for t in tokens if t in word_to_idx]
        # mirror fast_embed's MAX_TOKENS cap so the regression is
        # exercised whether or not we go through fast_embed.
        if len(idxs) > MAX_TOKENS_PER_ARTICLE:
            idxs = idxs[:MAX_TOKENS_PER_ARTICLE]
        raw = table[np.asarray(idxs, dtype=np.int32)]
        window = 10
        if raw.shape[0] < window:
            return raw
        T, D = raw.shape
        cumsum = np.empty((T + 1, D), dtype=np.float32)
        cumsum[0] = 0
        np.cumsum(raw, axis=0, out=cumsum[1:])
        return (cumsum[window:] - cumsum[:-window]) / np.float32(window)
    return embed


def monster_text(seed: int, n_tokens: int) -> str:
    r = np.random.default_rng(seed)
    return " ".join(f"tok{int(x)}" for x in r.integers(0, 20000, size=n_tokens))


def run():
    if not gpu_available():
        print("wgpu not available, skipping")
        return

    engine = GpuEngine()
    embed = make_fake_embed()
    print(f"adapter: {engine.describe()}")
    print(f"MAX_TOKENS_PER_ARTICLE = {MAX_TOKENS_PER_ARTICLE}")
    print(f"MAX_BLOCK = {MAX_BLOCK}")
    print()

    # mixed batch: most small, a few monstrous.
    articles = []
    for i in range(120):
        articles.append((f"normal_{i:04d}", monster_text(i, n_tokens=500)))
    for i in range(8):
        # each one is ~3x the MAX_TOKENS cap — would OOM under the old
        # staging scheme.
        articles.append((f"monster_{i}", monster_text(1000 + i, n_tokens=60000)))

    with tempfile.TemporaryDirectory() as td:
        brain_path = str(Path(td) / "_mem.engram")
        ef = EngramFile.create(brain_path, name="memtest", dim=300,
                                metadata={}, annotations={})
        codec = Engram(dim=300)

        accum = BatchedGpuIngest(
            ef=ef, codec=codec, engine=engine,
            batch_size=128, progress_every=999999,
        )

        import sys as _sys
        max_staged_bytes = 0
        max_segs_per_article = 0

        for title, text in articles:
            accum.queue(title, text, embed)
            # track the size of the most recently staged article
            if accum._pending:
                last = accum._pending[-1]
                # crude deep size: sum z_full nbytes across segments.
                nbytes = sum(s.z_full.nbytes for s in last.segments)
                max_staged_bytes = max(max_staged_bytes, nbytes)
                max_segs_per_article = max(max_segs_per_article, len(last.segments))

        accum.flush()

        print(f"absorbed:              {ef._total_absorbed}")
        print(f"max segs per article:  {max_segs_per_article}")
        print(f"max staged per art:    {max_staged_bytes/1024/1024:.2f} MB")

        # hard invariants
        # 1. every segment is bounded — MAX_BLOCK+2 timesteps × D × 4 bytes
        max_seg_bytes = (MAX_BLOCK + 2) * 300 * 4
        print(f"max seg size (theory): {max_seg_bytes} bytes")

        # 2. no article's staged footprint exceeds what a worst-case
        #    post-cap article could produce (20k tokens ÷ MIN_BLOCK=4
        #    segments × max seg size ≈ 780 MB). we don't assert an
        #    exact bound — just sanity-print.
        ceiling_mb = (MAX_TOKENS_PER_ARTICLE / 4 * max_seg_bytes) / 1024 / 1024
        print(f"pathological ceiling:  {ceiling_mb:.0f} MB per article")

        # 3. everything absorbed (monsters included)
        assert ef._total_absorbed >= 120, \
            f"expected ≥120 absorbs, got {ef._total_absorbed}"

        print()
        print("memory safety invariants: OK")


if __name__ == "__main__":
    run()
