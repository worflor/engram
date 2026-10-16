"""test_bytes_vs_words.py -- code vs prose differentiation via byte-level
engram + self-bootstrap refinement.

answers three questions:

  1. with a cooc-SVD byte embedding (no pretraining), does engram form
     DIFFERENT wells for code vs prose? or does the trajectory collapse
     the two into one attractor?

  2. does one-pass refinement (use the wells to rebuild the byte
     embedding, re-ingest) sharpen the separation or drift?

  3. how does byte-level compare to GloVe on prose -- does the cold-
     start signal recover the topical structure GloVe captures natively?

we measure:
  - well-count per corpus (more wells = more regime discovery)
  - which corpus's articles route to which wells when mixed
  - corpus-class purity: fraction of articles routed to a well whose
    majority corpus matches the article's source
"""
from __future__ import annotations

import io
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Iterator, Tuple, List

# force stdout to utf-8 so non-ASCII chars don't break on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from engram_codec import EngramFile
from engram.gpu.engine import gpu_available

from genesis_chain import genesis_chain, iter_wiki_articles
from sources import (
    from_bytes,
    byte_cooc_svd_embedding,
    refine_byte_embedding_from_wells,
)


DOWNLOADS = (Path.home() / "Downloads")
SIMPLEWIKI = DOWNLOADS / "simplewiki-20260301-pages-articles-multistream.xml.bz2"
ENGRAM_DIR = Path(__file__).resolve().parent


def section(title: str):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


# ── corpus iterators ─────────────────────────────────────────────────

def iter_code_files(limit: int = 200, min_bytes: int = 500,
                   max_bytes: int = 15000) -> Iterator[Tuple[str, str]]:
    """walk engram/*.py and engram/gpu/**/*.py, yielding (path, text)."""
    roots = [
        ENGRAM_DIR,
        ENGRAM_DIR / "gpu",
        ENGRAM_DIR / "gpu" / "shaders",
    ]
    n = 0
    for root in roots:
        if not root.exists():
            continue
        patterns = ["*.py", "*.wgsl"]
        for pat in patterns:
            for p in sorted(root.glob(pat)):
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                b = text.encode("utf-8", errors="replace")
                if len(b) < min_bytes:
                    continue
                if len(b) > max_bytes:
                    text = b[:max_bytes].decode("utf-8", errors="replace")
                yield (str(p.relative_to(ENGRAM_DIR)), text)
                n += 1
                if n >= limit:
                    return


def iter_prose(limit: int = 200, min_bytes: int = 500,
               max_bytes: int = 15000) -> Iterator[Tuple[str, str]]:
    """simplewiki articles with the same length envelope as code files."""
    if not SIMPLEWIKI.exists():
        print(f"  [skip prose] simplewiki missing at {SIMPLEWIKI}")
        return
    n = 0
    for title, text in iter_wiki_articles(str(SIMPLEWIKI), limit=limit * 3):
        b = text.encode("utf-8", errors="replace")
        if len(b) < min_bytes:
            continue
        if len(b) > max_bytes:
            text = b[:max_bytes].decode("utf-8", errors="replace")
        yield (title, text)
        n += 1
        if n >= limit:
            return


def corpus_sample_bytes(iterator_fn, n: int, per_item_cap: int = 4000) -> bytes:
    """materialize up to n items from iterator, concatenate truncated
    bytes for use as a corpus-statistics seed."""
    pieces: List[bytes] = []
    for i, (_id, text) in enumerate(iterator_fn(n)):
        b = text.encode("utf-8", errors="replace")[:per_item_cap]
        pieces.append(b)
        if len(pieces) >= n:
            break
    return b"\n".join(pieces)


# ── ingestion + analysis helpers ──────────────────────────────────────

def ingest_into_new_ef(articles, source, label: str, td: str,
                       batch_size: int = 24, n_cpu_workers: int = 4):
    """run genesis_chain on a fresh EngramFile. returns (ef, assignments)
    where assignments is dict[title -> well_name].
    """
    path = str(Path(td) / f"{label}.engram")
    ef = EngramFile.create(path, name=label, dim=source.dim,
                           metadata={}, annotations={})
    assignments: dict[str, str] = {}

    def on_article(title: str, well: str, measurement: dict):
        # measurement carries dream_surprise / well_attention etc.;
        # this experiment only cares about well routing, so ignore it.
        assignments[title] = well

    t0 = time.time()
    n = genesis_chain(
        ef, articles, source, label=label,
        batch_size=batch_size, batch_timeout_ms=100,
        n_cpu_workers=n_cpu_workers,
        progress_every=99999, telemetry_interval_s=99999.0,
        min_tokens=10,
        on_article=on_article,
    )
    dt = time.time() - t0
    print(f"  [{label}] absorbed {n} articles in {dt:.1f}s ({n/max(dt,1e-6):.0f}/s) "
          f"wells={len(ef.wells())}  assignments_captured={len(assignments)}")
    return ef, assignments, n


def well_distribution(assignments: dict) -> dict:
    """count how many articles routed to each well."""
    d: dict[str, int] = {}
    for _, w in assignments.items():
        d[w] = d.get(w, 0) + 1
    return d


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def report_well_usage(assignments: dict, label: str, top: int = 5):
    dist = well_distribution(assignments)
    total = sum(dist.values())
    ordered = sorted(dist.items(), key=lambda kv: -kv[1])
    print(f"  [{label}] {len(dist)} wells used, {total} articles total")
    for w, c in ordered[:top]:
        print(f"    {w:20s} {c:5d}  {c/max(total,1)*100:5.1f}%")


# ── experiments ──────────────────────────────────────────────────────

def experiment_1_initial_separation():
    """ingest code and prose SEPARATELY with the same cooc-SVD byte
    embedding trained on a mixed sample. each corpus grows its own
    brain; compare the well sets."""
    section("EXPERIMENT 1 -- initial separation (code vs prose, cooc-SVD init)")

    # build one embedding from a mixed sample, so both ingestions see
    # the same embedding -- any well divergence is from the trajectory
    # dynamics, not embedding skew.
    print("  sampling mixed corpus for cooc-SVD...")
    code_bytes = corpus_sample_bytes(iter_code_files, n=80)
    prose_bytes = corpus_sample_bytes(iter_prose, n=80)
    mixed = code_bytes + b"\n" + prose_bytes
    print(f"    code sample: {len(code_bytes)/1024:.0f} KB, "
          f"prose sample: {len(prose_bytes)/1024:.0f} KB")
    t0 = time.time()
    emb = byte_cooc_svd_embedding(mixed, D=32, window=4)
    print(f"  cooc-SVD [256,32] in {(time.time()-t0)*1000:.0f}ms "
          f"(norm mean={np.linalg.norm(emb, axis=1).mean():.2f})")

    source = from_bytes(matrix=emb, name="bytes-cooc")

    with tempfile.TemporaryDirectory() as td:
        ef_code, code_asg, n_code = ingest_into_new_ef(
            iter_code_files(200), source, "code-only", td)
        ef_prose, prose_asg, n_prose = ingest_into_new_ef(
            iter_prose(200), source, "prose-only", td)

        # a mixed ingest into one shared brain -- show regime mixing
        def mix():
            for x in iter_code_files(200):
                yield ("code/" + x[0], x[1])
            for x in iter_prose(200):
                yield ("prose/" + x[0], x[1])
        ef_mix, mix_asg, n_mix = ingest_into_new_ef(mix(), source, "mixed", td)

        # analysis
        print()
        print(f"  code-only   : {len(ef_code.wells()):3d} wells  "
              f"absorbed={n_code}")
        print(f"  prose-only  : {len(ef_prose.wells()):3d} wells  "
              f"absorbed={n_prose}")
        print(f"  mixed       : {len(ef_mix.wells()):3d} wells  "
              f"absorbed={n_mix}")

        code_wells = set(ef_code.wells().keys())
        prose_wells = set(ef_prose.wells().keys())
        mix_wells = set(ef_mix.wells().keys())
        # separate brains have independent well namespaces -> names will
        # differ trivially (well_0, well_1, ...). the useful comparison
        # is in the MIXED brain: are there wells dominated by one
        # corpus vs the other?
        print()
        print("  mixed brain well attribution (top 10):")
        dist = {}
        corpus_of = {}
        for aid, w in mix_asg.items():
            dist[w] = dist.get(w, 0) + 1
            corpus_of.setdefault(w, {"code": 0, "prose": 0})
            if aid.startswith("code/"):
                corpus_of[w]["code"] += 1
            elif aid.startswith("prose/"):
                corpus_of[w]["prose"] += 1
        ordered = sorted(dist.items(), key=lambda kv: -kv[1])
        total_code = sum(v["code"] for v in corpus_of.values())
        total_prose = sum(v["prose"] for v in corpus_of.values())
        for w, c in ordered[:10]:
            co = corpus_of[w]
            tot = co["code"] + co["prose"]
            purity = max(co["code"], co["prose"]) / max(tot, 1)
            dom = "code" if co["code"] > co["prose"] else "prose"
            print(f"    {w:20s} n={c:4d}  "
                  f"code={co['code']:4d} prose={co['prose']:4d}  "
                  f"purity={purity*100:5.1f}% -> {dom}")

        # overall purity: total fraction of articles whose corpus
        # matches the majority corpus of their well
        correct = 0
        for aid, w in mix_asg.items():
            co = corpus_of[w]
            maj = "code" if co["code"] >= co["prose"] else "prose"
            actual = "code" if aid.startswith("code/") else "prose"
            if actual == maj:
                correct += 1
        acc = correct / max(len(mix_asg), 1)
        print(f"\n  corpus-class purity (mixed brain): {acc*100:.1f}%  "
              f"(baseline 50%=no differentiation)")

        return emb, ef_mix, mix_asg


def experiment_2_refinement():
    """one-pass refinement: use mixed-brain wells to rebuild the byte
    embedding, re-ingest, measure change."""
    section("EXPERIMENT 2 -- self-bootstrap via refine_byte_embedding_from_wells")

    # redo experiment 1's ingestion but capture the full article stream
    # so we can re-run refinement against it. keep a list for reuse.
    code_items = list(iter_code_files(200))
    prose_items = list(iter_prose(200))
    combined = ([("code/" + i, t) for i, t in code_items]
                + [("prose/" + i, t) for i, t in prose_items])

    code_bytes = b"\n".join(
        t.encode("utf-8", errors="replace")[:4000] for _i, t in code_items[:80])
    prose_bytes = b"\n".join(
        t.encode("utf-8", errors="replace")[:4000] for _i, t in prose_items[:80])
    emb_v0 = byte_cooc_svd_embedding(code_bytes + b"\n" + prose_bytes,
                                     D=32, window=4)
    source_v0 = from_bytes(matrix=emb_v0, name="bytes-cooc-v0")

    with tempfile.TemporaryDirectory() as td:
        print("\n  == pass 0 (cooc-SVD init) ==")
        ef_v0, asg_v0, n_v0 = ingest_into_new_ef(
            iter(combined), source_v0, "v0", td, batch_size=8)
        _classify_purity(asg_v0, "v0")

        # refine byte embedding using v0's wells + assignments.
        # scale-match to the source embedding so the trajectory
        # energy stays comparable across refinement iterations —
        # otherwise segment.wgsl drops articles whose variance falls
        # below MACHINE_EPS on the shrunken vectors.
        print("  refining byte embedding from v0 wells...")
        t0 = time.time()
        v0_norm = float(np.linalg.norm(emb_v0, axis=1).mean())
        emb_v1 = refine_byte_embedding_from_wells(
            ef_v0, combined, asg_v0, D=32, blend=None,
            target_norm=v0_norm, prev_emb=emb_v0)
        print(f"    refined [256,32] in {(time.time()-t0)*1000:.0f}ms "
              f"(target_norm={v0_norm:.2f})")
        # diagnostic: average change in byte embeddings
        delta = np.linalg.norm(emb_v1 - emb_v0, axis=1)
        v1_norm = float(np.linalg.norm(emb_v1, axis=1).mean())
        print(f"    per-byte delta: mean={delta.mean():.3f} "
              f"max={delta.max():.3f}  v1 norm mean={v1_norm:.2f}")

        # pass 1: re-ingest with refined embedding into a fresh brain
        source_v1 = from_bytes(matrix=emb_v1, name="bytes-refined-v1")
        print("\n  == pass 1 (well-refined embedding) ==")
        ef_v1, asg_v1, n_v1 = ingest_into_new_ef(
            iter(combined), source_v1, "v1", td, batch_size=8)
        _classify_purity(asg_v1, "v1")

        # compare the two passes
        print()
        print(f"  v0 wells={len(ef_v0.wells())}  v1 wells={len(ef_v1.wells())}")


def _classify_purity(assignments: dict, label: str):
    """classify each article by which well captured it and whether that
    well's majority corpus agrees with the article's corpus."""
    corpus_of: dict[str, dict[str, int]] = {}
    for aid, w in assignments.items():
        c = "code" if aid.startswith("code/") else "prose"
        corpus_of.setdefault(w, {"code": 0, "prose": 0})[c] += 1

    correct = 0
    total = 0
    for aid, w in assignments.items():
        co = corpus_of[w]
        maj = "code" if co["code"] >= co["prose"] else "prose"
        actual = "code" if aid.startswith("code/") else "prose"
        total += 1
        if actual == maj:
            correct += 1

    # purity of the DOMINANT well for each corpus
    biggest_code = max((w for w, co in corpus_of.items()),
                       key=lambda w: corpus_of[w]["code"], default=None)
    biggest_prose = max((w for w, co in corpus_of.items()),
                        key=lambda w: corpus_of[w]["prose"], default=None)
    print(f"  [{label}] wells_used={len(corpus_of)}  "
          f"purity={correct/max(total,1)*100:.1f}%")
    if biggest_code:
        co = corpus_of[biggest_code]
        print(f"    top code well  {biggest_code}: code={co['code']} prose={co['prose']}")
    if biggest_prose and biggest_prose != biggest_code:
        co = corpus_of[biggest_prose]
        print(f"    top prose well {biggest_prose}: code={co['code']} prose={co['prose']}")


# ── driver ────────────────────────────────────────────────────────────

def main():
    if not gpu_available():
        print("wgpu missing"); sys.exit(1)

    t0 = time.time()
    experiment_1_initial_separation()
    experiment_2_refinement()
    section(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
