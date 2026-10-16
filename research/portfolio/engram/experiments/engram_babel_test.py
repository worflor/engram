"""
engram_babel_test.py

do the eigenvalues of thought transcend language?

take the same Wikipedia article in English, French, and Japanese.
embed each with a multilingual model. encode with Engram.
compare the orbital signatures.

caches fetched articles and embeddings to disk so subsequent runs
skip the network and embedding model entirely.
"""

from __future__ import annotations

import sys
import os
import json
import time
import pickle
import urllib.request
import urllib.parse
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from engram_codec import (
    Engram, Packet, summary, spectral_summary,
    eigenvalues_all, blend_eigenvalues,
)

CACHE_DIR = Path(os.path.dirname(__file__)) / ".." / "cache" / "_babel_cache"


# ── Wikipedia API (with disk cache) ──────────────────────────────────

def fetch_article(title: str, lang: str = "en") -> str:
    """fetch plain text of a Wikipedia article. caches to disk."""
    CACHE_DIR.mkdir(exist_ok=True)
    safe_name = title.replace("/", "_").replace("\\", "_").replace(" ", "_")
    cache_path = CACHE_DIR / f"{lang}_{safe_name}.txt"

    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "1",
        "exlimit": "1",
        "format": "json",
    }
    full_url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(full_url, headers={"User-Agent": "EngramBabelTest/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    pages = data.get("query", {}).get("pages", {})
    text = ""
    for page_id, page in pages.items():
        if page_id != "-1":
            text = page.get("extract", "")
            break

    if text:  # don't cache empty results (redirects, missing pages)
        cache_path.write_text(text, encoding="utf-8")
    return text


def chunk_text(text: str, chunk_size: int = 200) -> list[str]:
    """split text into chunks of ~chunk_size words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
    return chunks


# ── embedding (with disk cache) ──────────────────────────────────────

def load_embedder():
    from sentence_transformers import SentenceTransformer
    print("loading multilingual embedding model...")
    t0 = time.time()
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print(f"  loaded in {time.time() - t0:.1f}s (dim={model.get_sentence_embedding_dimension()})")
    return model


def embed_article(model, text: str, title: str, lang: str, chunk_size: int = 200) -> np.ndarray:
    """embed an article as a trajectory. caches embeddings to disk."""
    CACHE_DIR.mkdir(exist_ok=True)
    safe_name = title.replace("/", "_").replace("\\", "_").replace(" ", "_")
    cache_path = CACHE_DIR / f"{lang}_{safe_name}_emb.npy"

    if cache_path.exists():
        return np.load(str(cache_path))

    chunks = chunk_text(text, chunk_size)
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)
    embeddings = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    np.save(str(cache_path), embeddings)
    return embeddings


# ── articles ──────────────────────────────────────────────────────────

ARTICLES = [
    {"topic": "photosynthesis",     "en": "Photosynthesis",                    "fr": "Photosynthèse",                "ja": "光合成"},
    {"topic": "the moon",           "en": "Moon",                              "fr": "Lune",                         "ja": "月"},
    {"topic": "violin",             "en": "Violin",                            "fr": "Violon",                       "ja": "ヴァイオリン"},
    {"topic": "DNA",                "en": "DNA",                               "fr": "Acide désoxyribonucléique",    "ja": "デオキシリボ核酸"},
    {"topic": "chess",              "en": "Chess",                             "fr": "Échecs",                       "ja": "チェス"},
    {"topic": "volcano",            "en": "Volcano",                           "fr": "Volcan",                       "ja": "火山"},
    {"topic": "python language",    "en": "Python (programming language)",     "fr": "Python (langage)",             "ja": "Python"},
    {"topic": "roman empire",       "en": "Roman Empire",                      "fr": "Empire romain",                "ja": "ローマ帝国"},
    {"topic": "jazz",               "en": "Jazz",                              "fr": "Jazz",                         "ja": "ジャズ"},
    {"topic": "general relativity", "en": "General relativity",                "fr": "Relativité générale",          "ja": "一般相対性理論"},
]

LANGS = ["en", "fr", "ja"]


# ── eigenvalue fingerprint ────────────────────────────────────────────

def eigenvalue_fingerprint(packet: Packet) -> np.ndarray:
    """
    spectral distribution fingerprint for cross-trajectory comparison.

    instead of sorting eigenvalues (which makes every sorted ramp look
    the same), histogram the magnitudes and frequencies into fixed bins.
    this captures the DISTRIBUTION of oscillator dynamics, which is
    pairing-independent and content-discriminative.

    magnitude bins: how many eigenvalues are decaying, sustained, divergent.
    frequency bins: how many oscillate slowly, moderately, rapidly.
    together: the spectral signature of the trajectory.
    """
    # magnitude bins: [0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99, 1.01, 1.05, 1.1, 1.5, inf]
    mag_edges = np.array([0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99, 1.01, 1.05, 1.1, 1.5, 1e6])
    # frequency bins: [0, 0.1, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5, pi]
    freq_edges = np.array([0, 0.1, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5, np.pi + 0.01])

    all_mags = []
    all_freqs = []
    for blk in packet.blocks:
        if blk.macro_K is not None and len(blk.macro_K) > 0:
            l1, l2 = eigenvalues_all(blk.macro_K, blk.macro_G)
            all_mags.extend(np.abs(l1).tolist())
            all_mags.extend(np.abs(l2).tolist())
            all_freqs.extend(np.abs(np.angle(l1)).tolist())
            all_freqs.extend(np.abs(np.angle(l2)).tolist())

    if not all_mags:
        return np.zeros(len(mag_edges) - 1 + len(freq_edges) - 1, dtype=np.float32)

    mag_hist, _ = np.histogram(all_mags, bins=mag_edges)
    freq_hist, _ = np.histogram(all_freqs, bins=freq_edges)

    # normalize to probability distributions
    mag_p = mag_hist.astype(np.float32) / max(mag_hist.sum(), 1)
    freq_p = freq_hist.astype(np.float32) / max(freq_hist.sum(), 1)

    return np.concatenate([mag_p, freq_p])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < len(b):
        a = np.pad(a, (0, len(b) - len(a)))
    elif len(b) < len(a):
        b = np.pad(b, (0, len(a) - len(b)))
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 1e-8 else 0.0


# ── the test ──────────────────────────────────────────────────────────

def run():
    print("=" * 70)
    print("  ENGRAM BABEL TEST (with correlation-derived pairing)")
    print("  cached articles + embeddings in _babel_cache/")
    print("=" * 70)
    print()

    embedder = load_embedder()
    embed_dim = embedder.get_sentence_embedding_dimension()
    codec_dim = embed_dim if embed_dim % 2 == 0 else embed_dim - 1
    codec = Engram(dim=codec_dim)

    print(f"codec dim: {codec_dim}, pairing: correlation-derived")
    print()

    results = {}

    for art in ARTICLES:
        topic = art["topic"]
        for lang in LANGS:
            title = art[lang]
            key = (topic, lang)

            print(f"  [{lang}] {title}...", end=" ", flush=True)
            try:
                text = fetch_article(title, lang)
            except Exception as e:
                print(f"FETCH ERROR: {e}")
                continue

            if len(text) < 100:
                print(f"too short ({len(text)} chars)")
                continue

            traj = embed_article(embedder, text, title, lang)
            if traj.shape[0] < 4:
                print(f"too few chunks ({traj.shape[0]})")
                continue

            traj = traj[:, :codec_dim]
            pkt = codec.encode(traj)
            fp = eigenvalue_fingerprint(pkt)
            s = summary(pkt)

            results[key] = (pkt, fp, traj.shape[0])
            print(f"{traj.shape[0]} chunks, {s['blocks']} blocks, capture={s['capture']}%")

    print()

    # cross-lingual comparison
    print("=" * 70)
    print("  CROSS-LINGUAL EIGENVALUE SIMILARITY")
    print("=" * 70)
    print()

    same_topic_scores = []
    diff_topic_scores = []

    print("SAME TOPIC across languages:")
    for art in ARTICLES:
        topic = art["topic"]
        fps = {}
        for lang in LANGS:
            key = (topic, lang)
            if key in results:
                fps[lang] = results[key][1]

        if len(fps) < 2:
            continue

        lang_list = sorted(fps.keys())
        scores = []
        for i in range(len(lang_list)):
            for j in range(i + 1, len(lang_list)):
                sc = cosine(fps[lang_list[i]], fps[lang_list[j]])
                scores.append(sc)
                same_topic_scores.append(sc)

        print(f"  {topic:24s} [{'/'.join(lang_list)}]: {np.mean(scores):.4f}")

    print()

    print("DIFFERENT TOPICS within English:")
    en_fps = {art["topic"]: results[(art["topic"], "en")][1]
              for art in ARTICLES if (art["topic"], "en") in results}
    topics = sorted(en_fps.keys())
    pairs = []
    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            sc = cosine(en_fps[topics[i]], en_fps[topics[j]])
            diff_topic_scores.append(sc)
            pairs.append((sc, topics[i], topics[j]))
    pairs.sort(reverse=True)
    for sc, t1, t2 in pairs[:5]:
        print(f"  {t1:20s} ~ {t2:20s}: {sc:.4f}")
    print(f"  ...")
    for sc, t1, t2 in pairs[-3:]:
        print(f"  {t1:20s} ~ {t2:20s}: {sc:.4f}")

    print()
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print()

    if same_topic_scores and diff_topic_scores:
        mean_same = np.mean(same_topic_scores)
        mean_diff = np.mean(diff_topic_scores)
        std_same = np.std(same_topic_scores)
        std_diff = np.std(diff_topic_scores)

        print(f"  same topic, different language:  {mean_same:.4f} +/- {std_same:.4f}")
        print(f"  different topic, same language:  {mean_diff:.4f} +/- {std_diff:.4f}")
        print(f"  separation:                     {mean_same - mean_diff:+.4f}")
        print()

        if mean_same > mean_diff + 0.05:
            print(f"  the eigenvalues transcend language.")
        elif mean_same > mean_diff:
            print(f"  weak signal. more data needed.")
        else:
            print(f"  no separation detected.")

    # dream
    print()
    print("--- dream: multilingual generation ---")
    all_packets = [r[0] for r in results.values()]
    if len(all_packets) >= 3:
        gen = codec.dream(all_packets, n_blocks=4, samples_per_block=8)
        if gen is not None:
            print(f"  dreamed {gen.shape[0]} samples from {len(all_packets)} multilingual articles")
            print(f"  range: [{gen.min():.4f}, {gen.max():.4f}]")


if __name__ == "__main__":
    run()
