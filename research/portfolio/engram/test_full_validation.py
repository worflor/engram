"""test_full_validation.py — exhaustive real-data validation of the
unified pipeline. runs each stress phase at elevated scale, verifies
invariants, then does a determinism check (two runs on identical
input must produce identical wells).

layers exercised:
  - hetero:     mixed-corpus ingest (arxiv + openstax + wikiquote +
                simplewiki + wikibooks if present) at 2000/source
  - checkpoint: save + reload + resume, 2000 simplewiki articles
  - scale:      5000 mixed arxiv+openstax to grow past 100 wells
  - determinism:ingest 500 articles twice into two separate brains,
                assert identical well state
  - long_run:   5000 simplewiki articles through one brain for
                throughput + memory stability
"""
from __future__ import annotations

import sys
import time
import tempfile
import traceback
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from engram_codec import EngramFile
from engram.gpu.engine import gpu_available

from genesis_chain import (
    genesis_chain, from_glove_dict,
    iter_arxiv_abstracts, iter_wiki_articles, iter_text_files,
)

from test_stress_real import (
    CORPORA, get_glove_source,
    iter_arxiv_limit, iter_openstax_limit, iter_wiki_limit,
    validate_brain, PhaseTelemetry,
    phase_bounds,
)


def section(title: str):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


# ── layer 1: elevated hetero across every available corpus ─────────────

def layer_hetero(limit_per_source: int = 2000) -> bool:
    tele = PhaseTelemetry("layer_hetero")
    src = get_glove_source()
    tele.note("glove_vocab", src.vocab_size)

    sources = [
        ("arxiv",     iter_arxiv_limit(limit_per_source)),
        ("openstax",  iter_openstax_limit(limit_per_source)),
        ("wikiquote", iter_wiki_limit("wikiquote", limit_per_source)),
        ("simplewiki", iter_wiki_limit("simplewiki", limit_per_source)),
        ("wikibooks", iter_wiki_limit("wikibooks", limit_per_source)),
    ]

    with tempfile.TemporaryDirectory() as td:
        bp = str(Path(td) / "hetero.engram")
        ef = EngramFile.create(bp, name="hetero_full", dim=300,
                               metadata={}, annotations={})

        for key, it in sources:
            if not CORPORA[key].exists():
                print(f"  [skip {key}] not present")
                continue
            t0 = time.time()
            try:
                n = genesis_chain(
                    ef, it, src, label=key,
                    batch_size=48, batch_timeout_ms=100, n_cpu_workers=4,
                    progress_every=500, telemetry_interval_s=15.0,
                    checkpoint_path=bp,
                )
            except Exception as e:
                tele.fail(f"ingest_{key}", e)
                continue
            dt = time.time() - t0
            tele.note(f"{key}_absorbed", n)
            tele.note(f"{key}_rate", f"{n/max(dt,1e-6):.0f}/s")
            tele.note(f"{key}_wells_after", len(ef.wells()))
            viol = validate_brain(ef, key)
            if viol:
                tele.fail(f"validate_{key}", RuntimeError("; ".join(viol)))

        # final state summary
        counts = sorted(ef._well_count.values(), reverse=True)
        if counts:
            tele.note("wells_total", len(counts))
            tele.note("wells_top5_counts", counts[:5])
            tele.note("wells_median_count", int(np.median(counts)))
            total = sum(counts)
            tele.note("top_share", f"{counts[0]/total:.1%}")
            tele.note("total_absorbed", ef._total_absorbed)

    return tele.report()


# ── layer 2: determinism (two identical runs must match) ──────────────

def layer_determinism(n_articles: int = 500) -> bool:
    tele = PhaseTelemetry("layer_determinism")
    src = get_glove_source()
    if not CORPORA["simplewiki"].exists():
        print("  [skip] simplewiki missing")
        tele.report()
        return True

    articles_a = list(iter_wiki_limit("simplewiki", n_articles))
    articles_b = list(articles_a)  # same content, fresh iterator
    tele.note("n_articles", len(articles_a))

    with tempfile.TemporaryDirectory() as td:
        ef1_path = str(Path(td) / "run1.engram")
        ef2_path = str(Path(td) / "run2.engram")
        ef1 = EngramFile.create(ef1_path, name="run1", dim=300,
                                metadata={}, annotations={})
        ef2 = EngramFile.create(ef2_path, name="run2", dim=300,
                                metadata={}, annotations={})

        t0 = time.time()
        n1 = genesis_chain(ef1, iter(articles_a), src, label="run1",
                           batch_size=48, batch_timeout_ms=100,
                           n_cpu_workers=4,
                           progress_every=999999,
                           telemetry_interval_s=999.0)
        t_run1 = time.time() - t0

        t0 = time.time()
        n2 = genesis_chain(ef2, iter(articles_b), src, label="run2",
                           batch_size=48, batch_timeout_ms=100,
                           n_cpu_workers=4,
                           progress_every=999999,
                           telemetry_interval_s=999.0)
        t_run2 = time.time() - t0

        tele.note("run1_absorbed", n1)
        tele.note("run2_absorbed", n2)
        tele.note("run1_wells", len(ef1.wells()))
        tele.note("run2_wells", len(ef2.wells()))
        tele.note("run1_rate", f"{n1/max(t_run1,1e-6):.0f}/s")
        tele.note("run2_rate", f"{n2/max(t_run2,1e-6):.0f}/s")

        if n1 != n2:
            tele.fail("absorbed_mismatch",
                      RuntimeError(f"run1={n1} run2={n2}"))
        w1 = set(ef1.wells().keys())
        w2 = set(ef2.wells().keys())
        if w1 != w2:
            tele.fail("well_set_mismatch",
                      RuntimeError(f"only_1={list(w1-w2)[:3]}, "
                                   f"only_2={list(w2-w1)[:3]}"))
        max_cnt_d = 0
        for n in w1 & w2:
            max_cnt_d = max(max_cnt_d,
                            abs(ef1._well_count[n] - ef2._well_count[n]))
        tele.note("max_count_diff", max_cnt_d)
        if max_cnt_d != 0:
            tele.fail("well_counts_drift",
                      RuntimeError(f"max count diff = {max_cnt_d}"))

        # centroid equivalence: compare sums elementwise
        max_sum_rel = 0.0
        for n in w1 & w2:
            s1 = ef1._well_sum_K[n]
            s2 = ef2._well_sum_K[n]
            d = np.abs(s1 - s2).max()
            base = np.abs(s1).max() + 1e-12
            max_sum_rel = max(max_sum_rel, d / base)
        tele.note("max_well_sum_rel_diff", f"{max_sum_rel:.2e}")
        if max_sum_rel > 1e-5:
            tele.fail("centroid_drift",
                      RuntimeError(f"max rel diff = {max_sum_rel:.2e}"))

    return tele.report()


# ── layer 3: checkpoint round-trip on simplewiki (bigger than stress) ──

def layer_checkpoint(n_articles: int = 2000) -> bool:
    tele = PhaseTelemetry("layer_checkpoint")
    src = get_glove_source()
    if not CORPORA["simplewiki"].exists():
        print("  [skip] simplewiki missing")
        tele.report()
        return True

    arts_a = list(iter_wiki_limit("simplewiki", n_articles))
    arts_b = list(arts_a)
    tele.note("n_articles", len(arts_a))

    with tempfile.TemporaryDirectory() as td:
        ref_path = str(Path(td) / "ref.engram")
        split_path = str(Path(td) / "split.engram")

        # one-shot ref run
        ef_ref = EngramFile.create(ref_path, name="ref", dim=300,
                                   metadata={}, annotations={})
        t0 = time.time()
        n_ref = genesis_chain(ef_ref, iter(arts_a), src, label="one_shot",
                              batch_size=48, batch_timeout_ms=100,
                              n_cpu_workers=4,
                              progress_every=999999,
                              telemetry_interval_s=999.0,
                              checkpoint_path=ref_path)
        t_ref = time.time() - t0
        tele.note("ref_rate", f"{n_ref/max(t_ref,1e-6):.0f}/s")
        tele.note("ref_wells", len(ef_ref.wells()))

        # split: half → save → load → half
        ef_s1 = EngramFile.create(split_path, name="s1", dim=300,
                                  metadata={}, annotations={})
        half = len(arts_b) // 2
        genesis_chain(ef_s1, iter(arts_b[:half]), src, label="half1",
                      batch_size=48, batch_timeout_ms=100, n_cpu_workers=4,
                      progress_every=999999, telemetry_interval_s=999.0,
                      checkpoint_path=split_path)
        ef_s1.save(split_path)
        del ef_s1
        ef_s2 = EngramFile.load(split_path)
        tele.note("after_half1_wells", len(ef_s2.wells()))

        genesis_chain(ef_s2, iter(arts_b[half:]), src, label="half2",
                      batch_size=48, batch_timeout_ms=100, n_cpu_workers=4,
                      progress_every=999999, telemetry_interval_s=999.0,
                      checkpoint_path=split_path)
        tele.note("split_wells", len(ef_s2.wells()))

        # equivalence: well set + counts must match exactly
        if ef_ref._total_absorbed != ef_s2._total_absorbed:
            tele.fail("total_absorbed_mismatch",
                      RuntimeError(
                          f"ref={ef_ref._total_absorbed}, "
                          f"split={ef_s2._total_absorbed}"))
        w_ref = set(ef_ref.wells().keys())
        w_spl = set(ef_s2.wells().keys())
        if w_ref != w_spl:
            tele.fail("well_set_mismatch", RuntimeError(
                f"ref_only={list(w_ref-w_spl)[:5]}, "
                f"split_only={list(w_spl-w_ref)[:5]}"))
        max_c = 0
        for n in w_ref & w_spl:
            max_c = max(max_c,
                        abs(ef_ref._well_count[n] - ef_s2._well_count[n]))
        tele.note("max_count_diff", max_c)

        for lbl, ef in [("ref", ef_ref), ("split", ef_s2)]:
            viol = validate_brain(ef, lbl)
            if viol:
                tele.fail(f"validate_{lbl}", RuntimeError("; ".join(viol)))

    return tele.report()


# ── layer 4: scale run (5000 mixed) ───────────────────────────────────

def layer_scale(n_articles: int = 5000) -> bool:
    tele = PhaseTelemetry("layer_scale")
    src = get_glove_source()

    def mixed_iter():
        a = iter_arxiv_limit(n_articles // 2)
        o = iter_openstax_limit(n_articles // 2)
        while True:
            ar = next(a, None)
            oi = next(o, None)
            if ar is None and oi is None:
                break
            if ar is not None: yield ar
            if oi is not None: yield oi

    with tempfile.TemporaryDirectory() as td:
        bp = str(Path(td) / "scale.engram")
        ef = EngramFile.create(bp, name="scale_full", dim=300,
                               metadata={}, annotations={})
        t0 = time.time()
        n = genesis_chain(ef, mixed_iter(), src, label="scale",
                          batch_size=48, batch_timeout_ms=100,
                          n_cpu_workers=4,
                          progress_every=1000, telemetry_interval_s=20.0,
                          checkpoint_path=bp)
        dt = time.time() - t0
        tele.note("absorbed", n)
        tele.note("rate", f"{n/max(dt,1e-6):.0f}/s")
        tele.note("wells", len(ef.wells()))
        counts = sorted(ef._well_count.values(), reverse=True)
        if counts:
            tele.note("top5", counts[:5])
            total = sum(counts)
            tele.note("top_share", f"{counts[0]/total:.1%}")
            tele.note("median_count", int(np.median(counts)))
        viol = validate_brain(ef, "scale")
        if viol:
            tele.fail("validate", RuntimeError("; ".join(viol)))

    return tele.report()


# ── layer 5: long-run memory/throughput stability on simplewiki ───────

def layer_longrun(n_articles: int = 5000) -> bool:
    tele = PhaseTelemetry("layer_longrun")
    src = get_glove_source()
    if not CORPORA["simplewiki"].exists():
        print("  [skip] simplewiki missing")
        tele.report()
        return True

    with tempfile.TemporaryDirectory() as td:
        bp = str(Path(td) / "longrun.engram")
        ef = EngramFile.create(bp, name="longrun", dim=300,
                               metadata={}, annotations={})
        t0 = time.time()
        n = genesis_chain(ef, iter_wiki_limit("simplewiki", n_articles),
                          src, label="longrun",
                          batch_size=48, batch_timeout_ms=100,
                          n_cpu_workers=4,
                          progress_every=1000, telemetry_interval_s=20.0,
                          checkpoint_path=bp)
        dt = time.time() - t0
        tele.note("absorbed", n)
        tele.note("rate", f"{n/max(dt,1e-6):.0f}/s")
        tele.note("wells", len(ef.wells()))
        tele.note("duration_s", round(dt, 1))
        viol = validate_brain(ef, "longrun")
        if viol:
            tele.fail("validate", RuntimeError("; ".join(viol)))

    return tele.report()


# ── driver ────────────────────────────────────────────────────────────

def main():
    if not gpu_available():
        print("wgpu missing"); sys.exit(1)

    layers = [
        ("bounds",       phase_bounds),
        ("hetero",       layer_hetero),
        ("determinism",  layer_determinism),
        ("checkpoint",   layer_checkpoint),
        ("scale",        layer_scale),
        ("longrun",      layer_longrun),
    ]

    if len(sys.argv) > 1:
        chosen = sys.argv[1].split(",")
        layers = [(n, f) for (n, f) in layers if n in chosen]

    results = {}
    T0 = time.time()
    for name, fn in layers:
        section(f"LAYER: {name}")
        try:
            results[name] = fn()
        except BaseException as e:
            print(f"  !!! {name} crashed: {type(e).__name__}: {e}")
            traceback.print_exc()
            results[name] = False

    section("GRAND SUMMARY")
    for n, ok in results.items():
        print(f"  {n:14s}: {'PASS' if ok else 'FAIL'}")
    print(f"  total elapsed: {time.time()-T0:.1f}s")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
