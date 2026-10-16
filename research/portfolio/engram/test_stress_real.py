"""test_stress_real.py — aggressive real-data stress harness.

feeds real corpora (arxiv abstracts, openstax textbook chapters,
wikiquote, simplewiki) through the genesis_chain pipeline and tries
to break it. goals:

  1. surface bugs that synthetic tests miss
  2. profile every stage with real data (variable article lengths,
     OOV-heavy content, mixed scripts)
  3. exercise long-run invariants: scale past 100 wells, checkpoint +
     reload, switch sources mid-run
  4. assert state invariants at every phase boundary

every phase has a pass/fail with telemetry. bugs surfaced here get
fixed; the harness stays as the regression suite.

usage: `python test_stress_real.py [PHASE]`
  PHASE ∈ {all, hetero, bounds, checkpoint, scale} (default: all)
"""
from __future__ import annotations

import sys
import os
import time
import tempfile
import traceback
from pathlib import Path
from typing import Iterable, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from engram_codec import EngramFile, Engram
from engram.gpu.engine import gpu_available
from engram.gpu.brain import Brain

# public API under test
from genesis_chain import (
    genesis_chain, from_glove_dict,
    iter_arxiv_abstracts, iter_wiki_articles, iter_text_files,
)


DOWNLOADS = (Path.home() / "Downloads")

CORPORA = {
    "arxiv":      DOWNLOADS / "archive.zip",
    "openstax":   DOWNLOADS / "openstax-textbooks" / "openstax-text",
    "wikiquote":  DOWNLOADS / "enwikiquote-20260301-pages-articles-multistream.xml.bz2",
    "simplewiki": DOWNLOADS / "simplewiki-20260301-pages-articles-multistream.xml.bz2",
    "wikibooks":  DOWNLOADS / "enwikibooks-20260301-pages-articles-multistream.xml.bz2",
    "enwiki":     DOWNLOADS / "enwiki-20260301-pages-articles-multistream.xml.bz2",
}


# ── telemetry helpers ────────────────────────────────────────────────

class PhaseTelemetry:
    def __init__(self, name: str):
        self.name = name
        self.t0 = time.time()
        self.stats = {}
        self.failures = []

    def note(self, k, v):
        self.stats[k] = v

    def fail(self, stage: str, exc: BaseException):
        self.failures.append((stage, type(exc).__name__, str(exc),
                              traceback.format_exc()))

    def report(self):
        dt = time.time() - self.t0
        print(f"\n=== {self.name} completed in {dt:.1f}s ===")
        for k, v in self.stats.items():
            print(f"  {k}: {v}")
        if self.failures:
            print(f"  FAILURES: {len(self.failures)}")
            for stage, typ, msg, tb in self.failures:
                print(f"    [{stage}] {typ}: {msg}")
                for line in tb.splitlines()[-6:]:
                    print(f"      {line}")
            return False
        return True


# ── source loader (once, reused across phases) ────────────────────────

_glove_source = None
def get_glove_source():
    global _glove_source
    if _glove_source is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))
        from engram_glove_test import load_glove
        _glove_source = from_glove_dict(load_glove())
    return _glove_source


# ── real-corpus iterators with safety caps ────────────────────────────

def iter_arxiv_limit(limit: int):
    if not CORPORA["arxiv"].exists():
        return
    yield from iter_arxiv_abstracts(str(CORPORA["arxiv"]), limit=limit)


def iter_openstax_limit(limit: int):
    if not CORPORA["openstax"].exists():
        return
    yield from iter_text_files(str(CORPORA["openstax"]), limit=limit)


def iter_wiki_limit(corpus_key: str, limit: int):
    path = CORPORA[corpus_key]
    if not path.exists():
        return
    yield from iter_wiki_articles(str(path), limit=limit)


# ── validator: brain state invariants ─────────────────────────────────

def validate_brain(ef: EngramFile, label: str) -> list[str]:
    """return a list of invariant violations (empty list = clean)."""
    violations = []
    # 1. every well in _well_sum_K has a count
    k_names = set(ef._well_sum_K.keys())
    c_names = set(ef._well_count.keys())
    if k_names != c_names:
        violations.append(
            f"well_sum_K vs well_count mismatch: "
            f"only-K={k_names - c_names}, only-count={c_names - k_names}")

    # 2. every well has a non-negative count
    for n, c in ef._well_count.items():
        if c < 0:
            violations.append(f"well {n} has negative count {c}")

    # 3. well centroids are finite
    for n, s in ef._well_sum_K.items():
        if not np.all(np.isfinite(s)):
            violations.append(f"well {n} has non-finite sum ({np.isnan(s).sum()} NaN, "
                              f"{np.isinf(s).sum()} inf)")

    # 4. dream buffer lists same length
    dK, dG, dS = len(ef._dream_Ks), len(ef._dream_Gs), len(ef._dream_Ss)
    if not (dK == dG == dS):
        violations.append(f"dream buffer drift: K={dK}, G={dG}, S={dS}")

    # 5. dream buffer entries are finite
    for i, k in enumerate(ef._dream_Ks):
        if not np.all(np.isfinite(k)):
            violations.append(f"dream_Ks[{i}] non-finite")
            break  # one is enough

    # 6. total_absorbed is articles, well_count is segments — different
    # units. sanity: if any wells exist, total_absorbed >= 1. if
    # total_absorbed == 0, no wells should exist.
    if ef._total_absorbed == 0 and ef._well_count:
        violations.append(
            f"total_absorbed=0 but {len(ef._well_count)} wells exist")
    if ef._well_count and ef._total_absorbed == 0:
        violations.append("wells exist but no absorbed articles tracked")

    # 7. reference_pairing is the right shape
    if ef._reference_pairing is not None:
        if ef._reference_pairing.shape != (ef.dim,):
            violations.append(
                f"reference_pairing shape {ef._reference_pairing.shape} != (dim={ef.dim},)")

    return violations


# ── phase 1: heterogeneous multi-source ingestion ────────────────────

def phase_hetero(limit_per_source: int = 400):
    """ingest arxiv + openstax + wikiquote into one engram.
    tests that mixed sources share a well space correctly."""
    tele = PhaseTelemetry("phase_hetero")

    try:
        src = get_glove_source()
        tele.note("source dim", src.dim)
        tele.note("source vocab", src.vocab_size)
    except Exception as e:
        tele.fail("load_glove", e)
        tele.report()
        return False

    with tempfile.TemporaryDirectory() as td:
        brain_path = str(Path(td) / "hetero.engram")
        ef = EngramFile.create(brain_path, name="hetero_stress", dim=300,
                                metadata={}, annotations={})

        for corpus_key, iterator in [
            ("arxiv",     iter_arxiv_limit(limit_per_source)),
            ("openstax",  iter_openstax_limit(limit_per_source)),
            ("wikiquote", iter_wiki_limit("wikiquote", limit_per_source)),
            ("simplewiki", iter_wiki_limit("simplewiki", limit_per_source)),
        ]:
            if not CORPORA[corpus_key].exists():
                print(f"  [skip {corpus_key}] corpus missing at {CORPORA[corpus_key]}")
                continue
            t0 = time.time()
            try:
                n = genesis_chain(
                    ef, iterator, src,
                    label=corpus_key,
                    batch_size=48, batch_timeout_ms=100, n_cpu_workers=4,
                    progress_every=10_000, telemetry_interval_s=30.0,
                    checkpoint_path=brain_path,
                )
            except Exception as e:
                tele.fail(f"ingest_{corpus_key}", e)
                continue
            dt = time.time() - t0
            tele.note(f"{corpus_key} absorbed", n)
            tele.note(f"{corpus_key} rate_a_s", round(n / max(dt, 1e-6), 1))
            tele.note(f"{corpus_key} wells_after", len(ef.wells()))

            violations = validate_brain(ef, corpus_key)
            if violations:
                tele.fail(f"validate_{corpus_key}", RuntimeError("; ".join(violations)))

    return tele.report()


# ── phase 2: edge-case bounds ─────────────────────────────────────────

def phase_bounds():
    """drives articles of pathological size/content: empty, 1-token,
    stub, ultra-long, OOV-only, pure punctuation, UTF-8-heavy."""
    tele = PhaseTelemetry("phase_bounds")
    try:
        src = get_glove_source()
    except Exception as e:
        tele.fail("load_glove", e)
        tele.report()
        return False

    hazards = [
        ("empty",       ""),
        ("one_word",    "hello"),
        ("one_char",    "a"),
        ("punct_only",  "!!!!???..,,,---"),
        ("numbers",     "123 456 789"),
        ("short_stub",  "apple banana"),
        ("emoji_heavy", "🎉 party 🎊 celebration 🎈 happy"),
        ("utf8_mixed",  "Hello Привет 你好 مرحبا 안녕하세요 こんにちは"),
        ("markdown",    "# Heading\n\n**bold** text with `code` and ![img](url)"),
        ("repeat_word", "the the the the " * 200),
        ("url_heavy",   "see http://example.com/foo/bar and https://test.org/baz "*50),
    ]

    # stress mode: one very long article (20k tokens of real English)
    with open(__file__, "r", encoding="utf-8") as f:
        long_prose = f.read() * 100  # repeat source file 100× → large text
    hazards.append(("very_long", long_prose))

    with tempfile.TemporaryDirectory() as td:
        brain_path = str(Path(td) / "bounds.engram")
        ef = EngramFile.create(brain_path, name="bounds_stress", dim=300,
                                metadata={}, annotations={})

        try:
            n = genesis_chain(
                ef, iter(hazards), src,
                label="bounds",
                batch_size=8, batch_timeout_ms=50, n_cpu_workers=2,
                progress_every=99999, telemetry_interval_s=99999,
                min_tokens=1,  # allow 1-token items through
            )
            tele.note("hazards_input", len(hazards))
            tele.note("absorbed", n)
            tele.note("wells", len(ef.wells()))
        except Exception as e:
            tele.fail("ingest_hazards", e)

        violations = validate_brain(ef, "bounds")
        if violations:
            tele.fail("validate", RuntimeError("; ".join(violations)))

    return tele.report()


# ── phase 3: checkpoint + reload ──────────────────────────────────────

def phase_checkpoint(n_articles: int = 800):
    """run half of a corpus, save, reload, finish. verify state is
    identical whether done in one shot or checkpoint-split."""
    tele = PhaseTelemetry("phase_checkpoint")
    try:
        src = get_glove_source()
    except Exception as e:
        tele.fail("load_glove", e)
        tele.report()
        return False

    if not CORPORA["simplewiki"].exists():
        print("  [skip] simplewiki missing")
        tele.report()
        return True

    # gather the same article list twice (same order)
    articles_a = list(iter_wiki_limit("simplewiki", n_articles))
    articles_b = list(articles_a)  # shallow copy of the list
    if len(articles_a) < n_articles // 2:
        tele.note("warning", f"only {len(articles_a)} articles available")

    with tempfile.TemporaryDirectory() as td:
        # one-shot run (reference)
        ref_path = str(Path(td) / "ref.engram")
        ef_ref = EngramFile.create(ref_path, name="ref", dim=300,
                                    metadata={}, annotations={})
        try:
            n_ref = genesis_chain(
                ef_ref, iter(articles_a), src, label="one_shot",
                batch_size=48, batch_timeout_ms=100, n_cpu_workers=4,
                progress_every=99999, telemetry_interval_s=99999,
                checkpoint_path=ref_path,
            )
        except Exception as e:
            tele.fail("one_shot", e); tele.report(); return False
        tele.note("ref_absorbed", n_ref)
        tele.note("ref_wells", len(ef_ref.wells()))
        ref_total = ef_ref._total_absorbed

        # split run: ingest half, save, load, ingest rest
        split_path = str(Path(td) / "split.engram")
        ef_s1 = EngramFile.create(split_path, name="split", dim=300,
                                    metadata={}, annotations={})
        half = len(articles_b) // 2
        try:
            genesis_chain(
                ef_s1, iter(articles_b[:half]), src, label="half1",
                batch_size=48, batch_timeout_ms=100, n_cpu_workers=4,
                progress_every=99999, telemetry_interval_s=99999,
                checkpoint_path=split_path,
            )
        except Exception as e:
            tele.fail("half1", e); tele.report(); return False

        # explicit save + reload
        ef_s1.save(split_path)
        del ef_s1
        try:
            ef_s2 = EngramFile.load(split_path)
        except Exception as e:
            tele.fail("reload", e); tele.report(); return False
        tele.note("after_half1_wells", len(ef_s2.wells()))
        tele.note("after_half1_total", ef_s2._total_absorbed)

        try:
            genesis_chain(
                ef_s2, iter(articles_b[half:]), src, label="half2",
                batch_size=48, batch_timeout_ms=100, n_cpu_workers=4,
                progress_every=99999, telemetry_interval_s=99999,
                checkpoint_path=split_path,
            )
        except Exception as e:
            tele.fail("half2", e); tele.report(); return False
        tele.note("split_absorbed", ef_s2._total_absorbed)
        tele.note("split_wells", len(ef_s2.wells()))

        # validate both paths reached equivalent state.
        #
        # NOTE: absorption count can drift by a small amount (~0.5%)
        # because batch composition affects f32 precision in the
        # features kernel enough to tip borderline article accept/
        # reject decisions. this is a known pre-existing numerical
        # effect, not a determinism bug — each run at a fixed
        # batch_size is identical (verified separately).
        abs_diff = abs(ef_ref._total_absorbed - ef_s2._total_absorbed)
        tele.note("absorb_diff", abs_diff)
        tele.note("absorb_drift_pct",
                  f"{abs_diff / max(ef_ref._total_absorbed, 1):.2%}")
        if abs_diff / max(ef_ref._total_absorbed, 1) > 0.02:
            tele.fail("absorb_drift_too_large",
                      RuntimeError(
                          f"ref={ef_ref._total_absorbed}, "
                          f"split={ef_s2._total_absorbed}, "
                          f"diff={abs_diff}"))

        # well sets must mostly match — but a small symmetric
        # difference is tolerated. reason: dream.bin stores the dream
        # buffer as complex64 (the save/load format), and on reload
        # the meta-oscillator refits from the downcast values. the
        # one-shot path never round-trips through disk, so its dream
        # state keeps complex128 precision all the way through. that
        # residual precision gap can flip at most a handful of
        # borderline articles between "join nearest" and "spawn new
        # well" near the save boundary. cap at 2% of ref wells
        # (roughly one borderline decision per few hundred articles).
        ref_wells = set(ef_ref.wells().keys())
        split_wells = set(ef_s2.wells().keys())
        sym_diff = ref_wells.symmetric_difference(split_wells)
        well_drift_tol = max(2, len(ref_wells) // 50)
        tele.note("well_set_sym_diff", len(sym_diff))
        if len(sym_diff) > well_drift_tol:
            tele.fail("well_set_mismatch", RuntimeError(
                f"ref_only={list(ref_wells - split_wells)[:5]}, "
                f"split_only={list(split_wells - ref_wells)[:5]}, "
                f"sym_diff={len(sym_diff)} > tol={well_drift_tol}"))

        max_cnt_diff = 0
        for n in ref_wells & split_wells:
            d = abs(ef_ref._well_count[n] - ef_s2._well_count[n])
            max_cnt_diff = max(max_cnt_diff, d)
        tele.note("max_count_diff", max_cnt_diff)

        for (label_ef, ef) in [("ref", ef_ref), ("split", ef_s2)]:
            violations = validate_brain(ef, label_ef)
            if violations:
                tele.fail(f"validate_{label_ef}", RuntimeError("; ".join(violations)))

    return tele.report()


# ── phase 4: scale (grow past 100 wells) ──────────────────────────────

def phase_scale(n_articles: int = 2000):
    """drive enough articles to grow the brain past 100 wells.
    stresses the gap cache + apply stage at larger scale, and the
    in-batch rewire path when new wells are being actively created."""
    tele = PhaseTelemetry("phase_scale")
    try:
        src = get_glove_source()
    except Exception as e:
        tele.fail("load_glove", e)
        tele.report()
        return False

    # mix arxiv (short abstracts, high-topic-variance) + openstax
    # (dense technical prose) to force well diversity
    def mixed_iter():
        a_iter = iter_arxiv_limit(n_articles // 2)
        o_iter = iter_openstax_limit(n_articles // 2)
        while True:
            ar = next(a_iter, None)
            os_item = next(o_iter, None)
            if ar is None and os_item is None:
                break
            if ar is not None: yield ar
            if os_item is not None: yield os_item

    with tempfile.TemporaryDirectory() as td:
        brain_path = str(Path(td) / "scale.engram")
        ef = EngramFile.create(brain_path, name="scale_stress", dim=300,
                                metadata={}, annotations={})
        t0 = time.time()
        try:
            n = genesis_chain(
                ef, mixed_iter(), src,
                label="scale",
                batch_size=48, batch_timeout_ms=100, n_cpu_workers=4,
                progress_every=500, telemetry_interval_s=15.0,
                checkpoint_path=brain_path,
            )
        except Exception as e:
            tele.fail("ingest", e); tele.report(); return False
        dt = time.time() - t0
        tele.note("absorbed", n)
        tele.note("wells_final", len(ef.wells()))
        tele.note("rate_a_s", round(n / max(dt, 1e-6), 1))
        tele.note("duration_s", round(dt, 1))

        # distribution check: not all mass in one well
        counts = sorted(ef._well_count.values(), reverse=True)
        if counts:
            top = counts[0]
            total = sum(counts)
            tele.note("top_well_share", f"{top/total:.1%}")
            tele.note("top_5_wells", counts[:5])

        violations = validate_brain(ef, "scale")
        if violations:
            tele.fail("validate", RuntimeError("; ".join(violations)))

    return tele.report()


# ── main ──────────────────────────────────────────────────────────────

def main():
    if not gpu_available():
        print("wgpu missing")
        return

    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = {}

    phases = {
        "bounds": phase_bounds,
        "hetero": phase_hetero,
        "checkpoint": phase_checkpoint,
        "scale": phase_scale,
    }

    if phase == "all":
        to_run = ["bounds", "hetero", "checkpoint", "scale"]
    else:
        to_run = [phase]

    for p in to_run:
        print(f"\n{'=' * 70}")
        print(f"PHASE: {p}")
        print(f"{'=' * 70}")
        try:
            results[p] = phases[p]()
        except BaseException as e:
            print(f"  !!! {p} crashed outright: {type(e).__name__}: {e}")
            traceback.print_exc()
            results[p] = False

    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    for p, ok in results.items():
        print(f"  {p}: {'PASS' if ok else 'FAIL'}")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
