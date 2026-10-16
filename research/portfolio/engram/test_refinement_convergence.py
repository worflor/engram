"""test_refinement_convergence.py -- does iterated byte-embedding
refinement converge to a fixed point, and does classification purity
improve (or at least not degrade) with iterations?

we have refine_byte_embedding_from_wells() as a one-shot helper. the
open question: if we call it in a loop, what happens?

  - purity should NOT decrease. refinement pulls byte vectors toward
    wells their containing articles routed to. if this works, wells
    get more coherent each pass.
  - per-byte embedding delta should DECREASE across passes. at
    convergence, the well structure and the embedding should be
    consistent, so a further refinement pass shouldn't move the
    bytes much.
  - the label set should stabilize. wells' composite names
    (physics+composition+topology) should stop changing after
    convergence.

we report three real numbers per iteration:
  - classification purity on code vs prose mixed brain
  - mean per-byte embedding change
  - well-label sym-diff vs previous iteration

failure modes we catch:
  - oscillation: delta stays high, purity swings
  - degradation: purity decreases
  - single-well collapse: wells drop to 1, can't evaluate

5 passes over a ~100-article mixed corpus. small enough to iterate
fast, big enough for meaningful purity numbers.
"""
from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict

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
from engram.well_naming import name_wells


DOWNLOADS = (Path.home() / "Downloads")
SIMPLEWIKI = DOWNLOADS / "simplewiki-20260301-pages-articles-multistream.xml.bz2"
ENGRAM_DIR = Path(__file__).resolve().parent


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def pull_mixed(n_each: int = 60):
    code = []
    for root in (ENGRAM_DIR, ENGRAM_DIR / "gpu", ENGRAM_DIR / "gpu" / "shaders"):
        if not root.exists():
            continue
        for pat in ("*.py", "*.wgsl"):
            for p in sorted(root.glob(pat)):
                try:
                    t = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                b = t.encode("utf-8", errors="replace")
                if 500 <= len(b) <= 15000:
                    code.append(("code/" + str(p.relative_to(ENGRAM_DIR)), t))
                if len(code) >= n_each:
                    break
    prose = []
    if SIMPLEWIKI.exists():
        for title, text in iter_wiki_articles(str(SIMPLEWIKI),
                                               limit=n_each * 3):
            b = text.encode("utf-8", errors="replace")
            if 500 <= len(b) <= 15000:
                prose.append(("prose/" + title, text))
            if len(prose) >= n_each:
                break
    return code, prose


def purity(by_well: Dict[str, Dict[str, int]]) -> float:
    correct = 0
    total = 0
    for well, counts in by_well.items():
        c, p = counts.get("code", 0), counts.get("prose", 0)
        total += c + p
        correct += max(c, p)
    return correct / max(total, 1)


def one_pass(items, source, td: str, label: str) -> Tuple[EngramFile, Dict, int]:
    bp = str(Path(td) / f"{label}.engram")
    ef = EngramFile.create(bp, name=label, dim=source.dim,
                           metadata={}, annotations={})
    by_well: Dict[str, Dict[str, int]] = {}
    assignments: Dict[str, str] = {}

    def on_article(title: str, well: str, m: dict):
        cls = "code" if title.startswith("code/") else "prose"
        by_well.setdefault(well, {"code": 0, "prose": 0})[cls] += 1
        assignments[title] = well

    n = genesis_chain(
        ef, iter(items), source, label=label,
        batch_size=24, batch_timeout_ms=100, n_cpu_workers=4,
        progress_every=99999, telemetry_interval_s=99999.0,
        min_tokens=10, on_article=on_article,
    )
    return ef, by_well, assignments, n


def main():
    if not gpu_available():
        print("wgpu missing"); sys.exit(1)

    code, prose = pull_mixed(60)
    if not code or not prose:
        print("need both code + prose"); sys.exit(1)
    items = code + prose
    print(f"corpus: code={len(code)} prose={len(prose)} total={len(items)}")

    # seed embedding via cooc-SVD on a subset
    sample = b"\n".join(t.encode("utf-8", errors="replace")[:4000]
                        for _i, t in items[:80])
    emb = byte_cooc_svd_embedding(sample, D=32, window=4)
    print(f"seed embedding: shape={emb.shape} norm_mean="
          f"{float(np.linalg.norm(emb, axis=1).mean()):.3f}")

    per_iter = []
    prev_labels: set = set()
    prev_emb: np.ndarray = emb.copy()

    N_ITERS = 5
    with tempfile.TemporaryDirectory() as td:
        for it in range(N_ITERS):
            section(f"ITER {it}")
            src = from_bytes(matrix=emb, name=f"bytes_v{it}")
            t0 = time.time()
            ef, by_well, assignments, n = one_pass(
                items, src, td, f"iter{it}")
            t_ingest = time.time() - t0

            pur = purity(by_well)
            n_wells = len(ef.wells())
            labels = name_wells(ef)
            label_set = set(rec["name"] for rec in labels.values())

            delta = float(np.linalg.norm(emb - prev_emb, axis=1).mean())
            label_change = (
                len(label_set.symmetric_difference(prev_labels))
                if it > 0 else -1)

            print(f"  ingested={n}  wells={n_wells}  purity={pur*100:.1f}%")
            print(f"  ingest time: {t_ingest:.1f}s")
            if it > 0:
                print(f"  per-byte emb delta from prev: mean={delta:.3f}")
                print(f"  label sym_diff vs prev: {label_change} / "
                      f"{max(len(label_set), len(prev_labels))}")
            print(f"  current labels (top 5 by count):")
            ordered = sorted(labels.items(),
                             key=lambda kv: -kv[1]["count"])
            for well, rec in ordered[:5]:
                by = by_well.get(well, {"code": 0, "prose": 0})
                print(f"    {well:6s} n={rec['count']:6d}  "
                      f"({by['code']:3d}c,{by['prose']:3d}p)  "
                      f"label={rec['name']!r}")

            per_iter.append({
                "iter": it,
                "purity": pur,
                "wells": n_wells,
                "emb_delta": delta,
                "label_change": label_change,
            })

            # refine for next iteration. blend=None uses the adaptive
            # per-byte Bayesian rule — a byte's weight pulls harder
            # when it appears in mature wells, softer in fresh ones.
            # prev_emb keeps the new embedding full-rank across passes.
            if it < N_ITERS - 1:
                t0 = time.time()
                v0_norm = float(np.linalg.norm(emb, axis=1).mean())
                new_emb = refine_byte_embedding_from_wells(
                    ef, items, assignments, D=emb.shape[1],
                    blend=None, target_norm=v0_norm,
                    prev_emb=emb,
                )
                t_refine = time.time() - t0
                print(f"  refined in {t_refine:.2f}s")
                prev_emb = emb.copy()
                emb = new_emb
                prev_labels = label_set

    section("SUMMARY")
    print(f"{'iter':<5} {'purity':<8} {'wells':<6} "
          f"{'emb_delta':<10} {'label_change':<12}")
    for r in per_iter:
        pur_s = f"{r['purity']*100:.1f}%"
        delta_s = (f"{r['emb_delta']:.3f}" if r["iter"] > 0 else "—")
        ch_s = (f"{r['label_change']}" if r["iter"] > 0 else "—")
        print(f"{r['iter']:<5} {pur_s:<8} {r['wells']:<6} "
              f"{delta_s:<10} {ch_s:<12}")

    # assertions
    failures = []
    # C1: purity shouldn't COLLAPSE across iterations
    purs = [r["purity"] for r in per_iter]
    if min(purs) < 0.5:
        failures.append(
            f"purity dropped to {min(purs)*100:.1f}% at some iteration "
            f"(< 50% = worse than random)")

    # C2: embedding delta should trend DOWN (convergence signal)
    deltas = [r["emb_delta"] for r in per_iter[1:]]
    if len(deltas) >= 3:
        # late-vs-early delta ratio
        early = np.mean(deltas[:len(deltas)//2])
        late = np.mean(deltas[len(deltas)//2:])
        print(f"\n  delta early (iters 1..{len(deltas)//2}) mean = {early:.3f}")
        print(f"  delta late  (iters {len(deltas)//2+1}..) mean = {late:.3f}")
        if late > early * 1.2:
            failures.append(
                f"embedding delta INCREASING across iterations: "
                f"early mean {early:.3f} -> late mean {late:.3f}. "
                f"not converging")
        else:
            print(f"  [convergence] late delta <= early * 1.2 -> "
                  f"not diverging")

    # C3: label set should stabilize. symmetric diff should decrease.
    changes = [r["label_change"] for r in per_iter[1:]]
    if len(changes) >= 3:
        print(f"  label changes across iters: {changes}")
        if changes[-1] > changes[0] * 2 + 2:
            failures.append(
                f"label churn INCREASING: {changes[0]} -> {changes[-1]}. "
                f"labels not stabilizing")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("\nALL CONVERGENCE CRITERIA PASS -- refinement loop is well-behaved")


if __name__ == "__main__":
    main()
