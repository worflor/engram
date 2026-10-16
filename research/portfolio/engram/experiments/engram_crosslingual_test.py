"""
engram_crosslingual_test.py

the real test: does English photosynthesis have the same block size
as Japanese photosynthesis?

GloVe for English. fastText for Japanese. separate embedding spaces,
separate vocabularies, separate grammars. the ONLY thing they share
is the subject matter.

if the codec outputs (block size, capture rate, eigenvalue distribution)
match across languages for the same topic, then the structure of
knowledge is language-independent.
"""

from __future__ import annotations

import sys
import os
import re
import time
import pickle
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from engram_codec import Engram, summary, eigenvalues_all
from engram_babel_test import fetch_article, cosine

GLOVE_PATH = Path(os.path.dirname(__file__)) / "_glove_cache" / "glove.6B.300d.txt"
FASTTEXT_JA_PATH = Path(os.path.dirname(__file__)) / "_fasttext_cache" / "cc.ja.300.vec"


# ── embedding loaders ─────────────────────────────────────────────────

def load_glove() -> dict[str, np.ndarray]:
    cache = Path(str(GLOVE_PATH) + ".pkl")
    if cache.exists():
        with open(str(cache), "rb") as f:
            return pickle.load(f)
    glove = {}
    with open(str(GLOVE_PATH), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            if vec.shape[0] == 300:
                glove[parts[0]] = vec
    with open(str(cache), "wb") as f:
        pickle.dump(glove, f, protocol=pickle.HIGHEST_PROTOCOL)
    return glove


def load_fasttext_ja(max_words: int = 200000) -> dict[str, np.ndarray]:
    cache = Path(str(FASTTEXT_JA_PATH) + f".{max_words}.pkl")
    if cache.exists():
        print(f"  loading fastText JA from cache...", end=" ", flush=True)
        t0 = time.time()
        with open(str(cache), "rb") as f:
            ft = pickle.load(f)
        print(f"{len(ft)} words in {time.time() - t0:.1f}s")
        return ft

    print(f"  parsing fastText JA ({max_words} words)...", end=" ", flush=True)
    t0 = time.time()
    ft = {}
    with open(str(FASTTEXT_JA_PATH), "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for i, line in enumerate(f):
            if i >= max_words:
                break
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            if vec.shape[0] == 300:
                ft[word] = vec
    print(f"{len(ft)} words in {time.time() - t0:.1f}s")
    with open(str(cache), "wb") as f:
        pickle.dump(ft, f, protocol=pickle.HIGHEST_PROTOCOL)
    return ft


# ── tokenizers ────────────────────────────────────────────────────────

def tokenize_en(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 1]


def tokenize_ja(text: str) -> list[str]:
    """tokenize Japanese with fugashi (MeCab wrapper)."""
    try:
        import fugashi
        tagger = fugashi.Tagger()
        return [word.surface for word in tagger(text) if len(word.surface.strip()) > 0]
    except Exception:
        # fallback: character-level
        return [c for c in text if c.strip() and ord(c) > 127]


def text_to_trajectory(text: str, vocab: dict[str, np.ndarray], tokenizer) -> np.ndarray:
    tokens = tokenizer(text)
    vecs = [vocab[t] for t in tokens if t in vocab]
    if len(vecs) < 4:
        return np.empty((0, 300), dtype=np.float32)
    return np.stack(vecs)


# ── codec fingerprint ────────────────────────────────────────────────

def full_fingerprint(packet, s: dict) -> dict:
    """the FULL codec fingerprint: block size + capture + eigenvalue histogram."""
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

    mag_hist = np.zeros(len(mag_edges) - 1, dtype=np.float32)
    freq_hist = np.zeros(len(freq_edges) - 1, dtype=np.float32)
    if all_mags:
        mh, _ = np.histogram(all_mags, bins=mag_edges)
        fh, _ = np.histogram(all_freqs, bins=freq_edges)
        mag_hist = mh.astype(np.float32) / max(mh.sum(), 1)
        freq_hist = fh.astype(np.float32) / max(fh.sum(), 1)

    return {
        "macro_block": s["macro_block"],
        "micro_block": s["micro_block"],
        "capture": s["capture"],
        "n_blocks": s["blocks"],
        "mag_hist": mag_hist,
        "freq_hist": freq_hist,
    }


# ── test articles ─────────────────────────────────────────────────────

PARALLEL_ARTICLES = [
    {"topic": "photosynthesis",     "en": "Photosynthesis",         "ja": "光合成"},
    {"topic": "DNA",                "en": "DNA",                    "ja": "デオキシリボ核酸"},
    {"topic": "chess",              "en": "Chess",                  "ja": "チェス"},
    {"topic": "volcano",            "en": "Volcano",                "ja": "火山"},
    {"topic": "jazz",               "en": "Jazz",                   "ja": "ジャズ"},
    {"topic": "roman empire",       "en": "Roman Empire",           "ja": "ローマ帝国"},
    {"topic": "violin",             "en": "Violin",                 "ja": "ヴァイオリン"},
    {"topic": "general relativity", "en": "General relativity",     "ja": "一般相対性理論"},
    {"topic": "python language",    "en": "Python (programming language)", "ja": "Python"},
    {"topic": "the moon",           "en": "Moon",                   "ja": "月"},
]


def run():
    print("=" * 70)
    print("  ENGRAM CROSS-LINGUAL TEST")
    print("  same topic, different language, different embedding space.")
    print("  does the codec measure the same structure?")
    print("=" * 70)
    print()

    print("loading embeddings...")
    glove = load_glove()
    print(f"  GloVe EN: {len(glove)} words")
    ft_ja = load_fasttext_ja(200000)
    print()

    codec = Engram(dim=300)

    en_results = {}
    ja_results = {}

    for art in PARALLEL_ARTICLES:
        topic = art["topic"]

        # English
        en_text = fetch_article(art["en"], "en")
        en_traj = text_to_trajectory(en_text, glove, tokenize_en)

        # Japanese
        ja_text = fetch_article(art["ja"], "ja")
        ja_traj = text_to_trajectory(ja_text, ft_ja, tokenize_ja)

        print(f"  {topic}:")

        if en_traj.shape[0] >= 10:
            en_pkt = codec.encode(en_traj)
            en_s = summary(en_pkt)
            en_fp = full_fingerprint(en_pkt, en_s)
            en_results[topic] = en_fp
            print(f"    EN: {en_traj.shape[0]:6d} words, block={en_s['macro_block']:3d}, capture={en_s['capture']:5.1f}%, blocks={en_s['blocks']}")
        else:
            print(f"    EN: too few words ({en_traj.shape[0]})")

        if ja_traj.shape[0] >= 10:
            ja_pkt = codec.encode(ja_traj)
            ja_s = summary(ja_pkt)
            ja_fp = full_fingerprint(ja_pkt, ja_s)
            ja_results[topic] = ja_fp
            print(f"    JA: {ja_traj.shape[0]:6d} words, block={ja_s['macro_block']:3d}, capture={ja_s['capture']:5.1f}%, blocks={ja_s['blocks']}")
        else:
            print(f"    JA: too few words ({ja_traj.shape[0]})")

    print()
    print("=" * 70)
    print("  CROSS-LINGUAL COMPARISON")
    print("=" * 70)
    print()

    # compare each metric across languages
    shared_topics = [t for t in en_results if t in ja_results]
    if not shared_topics:
        print("  no shared topics with enough data.")
        return

    print(f"  {'topic':24s} {'EN block':>10s} {'JA block':>10s} {'EN cap%':>10s} {'JA cap%':>10s} {'eig cos':>10s}")
    print(f"  {'':24s} {'size':>10s} {'size':>10s} {'':>10s} {'':>10s} {'':>10s}")
    print(f"  " + "-" * 74)

    block_diffs = []
    capture_diffs = []
    eig_coses = []

    for topic in shared_topics:
        en = en_results[topic]
        ja = ja_results[topic]

        eig_en = np.concatenate([en["mag_hist"], en["freq_hist"]])
        eig_ja = np.concatenate([ja["mag_hist"], ja["freq_hist"]])
        eig_cos = cosine(eig_en, eig_ja)

        block_diffs.append(abs(en["macro_block"] - ja["macro_block"]))
        capture_diffs.append(abs(en["capture"] - ja["capture"]))
        eig_coses.append(eig_cos)

        print(f"  {topic:24s} {en['macro_block']:10d} {ja['macro_block']:10d} {en['capture']:10.1f} {ja['capture']:10.1f} {eig_cos:10.4f}")

    print()
    print(f"  mean block size difference:  {np.mean(block_diffs):.1f}")
    print(f"  mean capture difference:     {np.mean(capture_diffs):.1f}%")
    print(f"  mean eigenvalue cosine:      {np.mean(eig_coses):.4f}")
    print()

    # compare against random baseline: different topics, same language
    print("  baseline (different EN topics):")
    en_topics = sorted(en_results.keys())
    baseline_block = []
    baseline_cap = []
    for i in range(len(en_topics)):
        for j in range(i + 1, len(en_topics)):
            baseline_block.append(abs(en_results[en_topics[i]]["macro_block"] - en_results[en_topics[j]]["macro_block"]))
            baseline_cap.append(abs(en_results[en_topics[i]]["capture"] - en_results[en_topics[j]]["capture"]))

    if baseline_block:
        print(f"  mean block size difference:  {np.mean(baseline_block):.1f}")
        print(f"  mean capture difference:     {np.mean(baseline_cap):.1f}%")
    print()

    # verdict
    if block_diffs and baseline_block:
        cross_block = np.mean(block_diffs)
        base_block = np.mean(baseline_block)
        cross_cap = np.mean(capture_diffs)
        base_cap = np.mean(baseline_cap)

        print(f"  VERDICT:")
        if cross_block < base_block:
            print(f"    block size: SAME topic across languages ({cross_block:.1f}) is MORE similar")
            print(f"    than DIFFERENT topics in same language ({base_block:.1f}).")
            print(f"    the autocorrelation of explanation is language-independent.")
        else:
            print(f"    block size: cross-lingual ({cross_block:.1f}) vs baseline ({base_block:.1f}).")
            print(f"    no clear cross-lingual signal in block size.")

        if cross_cap < base_cap:
            print(f"    capture: SAME topic across languages ({cross_cap:.1f}%) is MORE similar")
            print(f"    than DIFFERENT topics ({base_cap:.1f}%).")
        else:
            print(f"    capture: cross-lingual ({cross_cap:.1f}%) vs baseline ({base_cap:.1f}%).")


if __name__ == "__main__":
    run()
