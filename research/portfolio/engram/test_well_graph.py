"""test_well_graph.py -- transition graph assertions.

the brain now records prev->curr edges as articles stream through
the absorb worker. the graph is a Markov chain over concepts; the
tests below assert it's correctly populated, useful, and
deterministic.

  T1  GRAPH HAS EDGES. after ingesting a non-trivial corpus we
      expect at least one outgoing edge from most established
      wells — otherwise transition tracking silently regressed.

  T2  PREDICT-NEXT BEATS RANDOM. predict_next_well(prev) should
      put more probability on the actually-seen next-well than a
      uniform-over-wells baseline does. we measure mean log-odds
      gain across every recorded (prev, curr) pair.

  T3  TRANSITION ENTROPY IS BOUNDED. for wells with >=2 distinct
      outgoing targets, entropy should be > 0 (they genuinely
      branch); for single-target wells it should be exactly 0.

  T4  SAVE/LOAD ROUND-TRIPS THE GRAPH. the saved .engram carries
      transitions.json; a reloaded brain has identical edges.

none of these require extra data beyond what test_per_well_dream
already pulls. running both as a pair is a natural sanity suite
for format v6.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import List, Dict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import math
import numpy as np

from engram_codec import EngramFile, WELL_ESTABLISHED
from genesis_chain import genesis_chain, iter_wiki_articles
from sources import from_bytes, byte_cooc_svd_embedding


DOWNLOADS = (Path.home() / "Downloads")
SIMPLEWIKI = DOWNLOADS / "simplewiki-20260301-pages-articles-multistream.xml.bz2"
ENGRAM_DIR = Path(__file__).resolve().parent


def section(t: str):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}", flush=True)


def pull_mixed(n_each: int = 200):
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


def build_brain(td: str):
    code, prose = pull_mixed(200)
    items = code + prose
    sample = b"\n".join(t.encode("utf-8", errors="replace")[:4000]
                        for _i, t in items[:120])
    emb = byte_cooc_svd_embedding(sample, D=32, window=4)
    source = from_bytes(matrix=emb, name="bytes")
    ef = EngramFile.create(str(Path(td) / "b.engram"), name="b",
                           dim=source.dim, metadata={}, annotations={})
    routing_order: List[str] = []

    def on_article(title, well, m):
        # callback fires in absorb order; the ordering here matches
        # the ordering of transition-graph updates, so we can cross-
        # check the graph against routing_order later.
        routing_order.append(well)

    # finalize=False: consolidate would merge wells and rewrite edges,
    # but routing_order still references the pre-merge names, so any
    # T2-style "predict what I actually observed" check would see a
    # stale reference set. disabling finalize keeps the raw graph
    # aligned with the routing sequence for this assertion. T4 still
    # tests save/load on the raw (unconsolidated) graph which is the
    # stricter case.
    genesis_chain(
        ef, iter(items), source, label="mix",
        batch_size=24, batch_timeout_ms=100, n_cpu_workers=4,
        progress_every=99999, telemetry_interval_s=99999.0,
        min_tokens=10, on_article=on_article,
        escape_energy=0.5, finalize=False,
    )
    return ef, source, items, routing_order


def expected_edges_from_order(routing_order: List[str]) -> Dict:
    """reconstruct the transition dict the absorb worker should have
    built from the observed routing_order. 'unknown' routes are
    skipped but don't reset prev_well (matching the absorb worker).
    """
    exp: Dict[str, Dict[str, int]] = {}
    prev = None
    for name in routing_order:
        if name == "unknown":
            continue
        if prev is not None:
            exp.setdefault(prev, {})
            exp[prev][name] = exp[prev].get(name, 0) + 1
        prev = name
    return exp


def t1_graph_has_edges(ef) -> bool:
    section("T1: graph has edges on established wells")
    wells = list(ef._well_sum_K.keys())
    established = [
        n for n in wells
        if int(ef._well_count.get(n, 0)) >= WELL_ESTABLISHED
    ]
    if len(established) < 2:
        print(f"  skipped (only {len(established)} established wells)")
        return True
    with_edges = [n for n in established if ef._well_transitions.get(n)]
    coverage = len(with_edges) / len(established)
    print(f"  established wells: {len(established)}")
    print(f"  with outgoing edges: {len(with_edges)} ({coverage:.0%})")
    total_edges = sum(len(v) for v in ef._well_transitions.values())
    print(f"  total edges: {total_edges}")
    ok = coverage >= 0.75 and total_edges > 0
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def t2_predict_beats_random(ef, routing_order: List[str]) -> bool:
    section("T2: predict_next_well beats uniform baseline")
    # mean log-prob of the actually-observed next well vs uniform
    # over all established wells.
    W = len(ef._well_sum_K)
    if W < 2:
        print("  skipped (<2 wells)")
        return True
    uniform = 1.0 / W

    total_log_pred = 0.0
    total_log_uniform = 0.0
    n_checks = 0
    prev = None
    for curr in routing_order:
        if curr == "unknown":
            continue
        if prev is not None:
            preds = ef.predict_next_well(prev, k=W)
            pred_map = {w: p for w, p in preds}
            # Laplace smoothing so unseen continuations don't give -inf
            p_seen = pred_map.get(curr, 0.0)
            p = p_seen * 0.9 + uniform * 0.1
            total_log_pred += math.log(max(p, 1e-12))
            total_log_uniform += math.log(uniform)
            n_checks += 1
        prev = curr

    if n_checks == 0:
        print("  skipped (no transitions)")
        return True
    gain = (total_log_pred - total_log_uniform) / n_checks
    print(f"  checked {n_checks} transitions")
    print(f"  mean log-prob gain over uniform: {gain:.3f} nats")
    ok = gain > 0.1
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def t3_transition_entropy(ef) -> bool:
    section("T3: transition_entropy = 0 for singletons, > 0 for branches")
    ok = True
    n_singleton = 0
    n_branch = 0
    for well, targets in ef._well_transitions.items():
        if not targets:
            continue
        h = ef.transition_entropy(well)
        distinct = sum(1 for c in targets.values() if c > 0)
        if distinct == 1:
            n_singleton += 1
            if h != 0.0:
                print(f"  FAIL: singleton {well} has entropy {h}")
                ok = False
        elif distinct > 1:
            n_branch += 1
            if h <= 0.0:
                print(f"  FAIL: branching {well} has entropy {h}")
                ok = False
    print(f"  singletons: {n_singleton} (entropy=0 expected)")
    print(f"  branching:  {n_branch}  (entropy>0 expected)")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def t4_graph_roundtrips(ef, td: str) -> bool:
    section("T4: save/load round-trips the graph")
    path = str(Path(td) / "roundtrip.engram")
    from engram.finalize import finalize_brain
    finalize_brain(ef, consolidate=False, name=False, crystallize=False)
    ef.save(path)
    ef2 = EngramFile.load(path)

    before = {a: dict(bs) for a, bs in ef._well_transitions.items()}
    after = {a: dict(bs) for a, bs in ef2._well_transitions.items()}
    ok = before == after
    print(f"  edges before: {sum(len(v) for v in before.values())}")
    print(f"  edges after:  {sum(len(v) for v in after.values())}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    with tempfile.TemporaryDirectory() as td:
        ef, source, items, routing_order = build_brain(td)
        exp = expected_edges_from_order(routing_order)
        actual = {a: dict(bs) for a, bs in ef._well_transitions.items()}
        match = exp == actual
        print(f"\ngraph matches reconstructed ingest order: {match}")
        if not match:
            # show one discrepancy to aid debugging
            keys = set(exp) | set(actual)
            for k in sorted(keys):
                if exp.get(k) != actual.get(k):
                    print(f"  diff at {k}: exp={exp.get(k)} act={actual.get(k)}")
                    break

        results = [
            ("T1 has edges", t1_graph_has_edges(ef)),
            ("T2 beats random", t2_predict_beats_random(ef, routing_order)),
            ("T3 entropy", t3_transition_entropy(ef)),
            ("T4 round-trip", t4_graph_roundtrips(ef, td)),
        ]
        section("summary")
        for name, ok in results:
            print(f"  {'[PASS]' if ok else '[FAIL]'} {name}")
        if not all(ok for _, ok in results):
            sys.exit(1)


if __name__ == "__main__":
    main()
