"""profile_well_spawning.py -- measure why byte-level ingestion often
collapses to 2 wells on mixed corpora.

hypothesis to check: once 2 wells form, the mass-weighted argmin plus
the log-growth escape_mult make new-well spawning essentially
impossible even for genuinely distinct content.

escape condition (from absorb_route.wgsl):
    escape_mult = ESCAPE_ENERGY * sqrt(log(1+nn_c) / log(1+median_count))
    spawn if: nn_d > typical_gap * escape_mult

with ESCAPE_ENERGY=2, and nn_c = count of the NEAREST well (often the
dominant one), escape threshold grows roughly as sqrt(log(nn_c)).
dominant wells become attractors that eat everything -- unless the
new content is >2x median well gap away in K-space.

we want to know:
  (a) distribution of nn_d / (typical_gap * escape_mult) per article
      -- how close are we to escaping? are we flirting with the edge
      or are we well below?
  (b) trajectory of well count over articles -- does it grow then
      plateau, or does spawning just stop cold?
  (c) what the mean typical_gap * escape_mult looks like vs actual
      article distances
"""
from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path
from typing import List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from engram_codec import EngramFile, ESCAPE_ENERGY
from engram.gpu.engine import gpu_available
from engram.gpu.brain import Brain
from genesis_chain import genesis_chain, iter_wiki_articles
from sources import from_bytes, byte_cooc_svd_embedding


DOWNLOADS = (Path.home() / "Downloads")
SIMPLEWIKI = DOWNLOADS / "simplewiki-20260301-pages-articles-multistream.xml.bz2"
ENGRAM_DIR = Path(__file__).resolve().parent


def pull_mixed(n_each: int = 100):
    code = []
    for p in sorted(ENGRAM_DIR.glob("*.py"))[:n_each]:
        try:
            t = p.read_text(encoding="utf-8", errors="replace")
            if 500 <= len(t) <= 15000:
                code.append(("code/" + p.name, t))
        except Exception:
            pass
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


def profile_run(items, source, label: str):
    with tempfile.TemporaryDirectory() as td:
        ef = EngramFile.create(str(Path(td) / f"{label}.engram"),
                               name=label, dim=source.dim,
                               metadata={}, annotations={})

        # capture per-article data. we can't snapshot brain state from
        # outside synchronously, but measurement carries well_distance;
        # we correlate against the ef counts AT TIME OF OBSERVATION
        # (post-absorb) which is close enough for the trend we're
        # looking for.
        trace: list[dict] = []
        seen_wells: list[int] = []  # running well count snapshot

        def on_article(title: str, well: str, m: dict):
            d = m.get("well_distance", -1.0)
            drift = m.get("drift", -1.0)
            trace.append({
                "title": title,
                "well": well,
                "dist": float(d),
                "drift_pct": float(drift),
                "surprise": float(m.get("dream_surprise", -1.0)),
                "dream_state": m.get("dream_state", ""),
                "well_count_after": len(ef.wells()),
            })

        t0 = time.time()
        n = genesis_chain(
            ef, iter(items), source, label=label,
            batch_size=24, batch_timeout_ms=100, n_cpu_workers=4,
            progress_every=99999, telemetry_interval_s=99999.0,
            min_tokens=10, on_article=on_article,
        )
        dt = time.time() - t0

        # compute derived stats
        wells = ef.wells()
        counts = np.array([ef._well_count.get(n, 0) for n in wells],
                          dtype=np.float64)
        # typical_gap estimate from the final brain (what it WAS at
        # end-of-run — this is what articles were routed against for
        # the tail of the ingest).
        if len(counts) >= 2:
            diffs = []
            centroids = [wells[n] for n in wells]
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    d = float(np.sqrt(np.mean(
                        np.abs(centroids[i] - centroids[j]) ** 2).real))
                    diffs.append(d)
            typical_gap_final = float(np.median(diffs))
        else:
            typical_gap_final = float("nan")

        established = counts[counts >= 10]
        median_count = float(np.median(established)) if len(established) >= 2 else 1.0

        return {
            "n_absorbed": n,
            "elapsed": dt,
            "wells_final": len(wells),
            "counts": counts,
            "typical_gap_final": typical_gap_final,
            "median_count": median_count,
            "trace": trace,
        }


def summarize(result: dict, label: str):
    trace = result["trace"]
    counts = result["counts"]
    tg = result["typical_gap_final"]
    print(f"\n== {label} ==")
    print(f"  absorbed={result['n_absorbed']} in {result['elapsed']:.1f}s")
    print(f"  final wells: {result['wells_final']}")
    print(f"  top-5 counts: {sorted(counts, reverse=True)[:5]}")
    print(f"  typical_gap (end-of-run): {tg:.4f}")
    print(f"  median_count (established): {result['median_count']:.1f}")

    # well-count growth trajectory
    growth = []
    last_n = 0
    for i, t in enumerate(trace):
        if t["well_count_after"] != last_n:
            growth.append((i, t["well_count_after"]))
            last_n = t["well_count_after"]
    print(f"  well_count growth (article_idx, count) first 8:")
    for i, n in growth[:8]:
        print(f"    @{i:4d}: {n}")
    if len(growth) > 8:
        print(f"    ... {len(growth) - 8} more spawn events ...")
        print(f"    last: @{growth[-1][0]:4d}: {growth[-1][1]}")

    # distances: distribution + near-escape frequency
    dists = np.array([t["dist"] for t in trace if t["dist"] >= 0],
                     dtype=np.float64)
    drifts = np.array([t["drift_pct"] for t in trace if t["drift_pct"] >= 0],
                      dtype=np.float64)
    if len(dists):
        print(f"  article well_distance: "
              f"min={dists.min():.4f} p25={np.percentile(dists,25):.4f} "
              f"med={np.median(dists):.4f} p75={np.percentile(dists,75):.4f} "
              f"max={dists.max():.4f}")
        print(f"  drift (dist/scale*100): "
              f"min={drifts.min():.1f} med={np.median(drifts):.1f} "
              f"max={drifts.max():.1f}")

    # what fraction of articles were "close to escape"? we don't have
    # typical_gap per-article in the trace, but we can use the
    # end-of-run value. escape_mult >= ESCAPE_ENERGY=2 so the
    # threshold is >= 2*typical_gap. articles with dist > typical_gap
    # but < 2*typical_gap were on the bubble.
    if not np.isnan(tg) and tg > 0:
        on_bubble = int(np.sum((dists > tg) & (dists < 2 * tg)))
        over_2g = int(np.sum(dists > 2 * tg))
        print(f"  distance thresholds (w/ typical_gap={tg:.4f}):")
        print(f"    < 1x tg     (nested in well):  {int(np.sum(dists < tg))}")
        print(f"    1x-2x tg    (on the bubble):   {on_bubble}")
        print(f"    > 2x tg     (would escape):    {over_2g}")


def main():
    if not gpu_available():
        print("wgpu missing"); sys.exit(1)

    # fresh runs on different scales to see if scale helps wells grow
    for n_each in (50, 200, 500):
        code, prose = pull_mixed(n_each)
        if not code or not prose:
            print(f"skip n_each={n_each}: insufficient corpus")
            continue
        items = code + prose
        sample = b"\n".join(t.encode("utf-8", errors="replace")[:4000]
                             for _i, t in items[:80])
        emb = byte_cooc_svd_embedding(sample, D=32, window=4)
        source = from_bytes(matrix=emb, name="bytes")
        result = profile_run(items, source, f"mix_{n_each}")
        summarize(result, f"mixed n={n_each}")


if __name__ == "__main__":
    main()
