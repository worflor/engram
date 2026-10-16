"""test_escape_energy_sweep.py -- prove escape_energy is a real knob.

the escape condition in absorb_route.wgsl:
    spawn if nn_d > typical_gap * escape_energy * sqrt(log1p(nn_c) / log1p(med_c))

with the escape_energy multiplier now exposed through genesis_chain,
this test sweeps three values -- LOW (easier to spawn), DEFAULT, HIGH
(harder to spawn) -- on the same corpus and embedding, and asserts
well count changes monotonically. if escape_energy doesn't affect
clustering, something is broken (wrong value reaching the kernel,
cached pipeline state, etc.).

we ingest a mixed code+prose corpus under each setting and record:
  - final well count
  - size distribution
  - per-corpus purity (how well the clustering separates the two
    classes)

claim:
  wells(low) >= wells(default) >= wells(high)
  (monotone in the expected direction -- more willingness to spawn
  means more wells form)

not claiming strict-greater because the corpus is finite; two small
settings might tie. but the default should never produce MORE wells
than the low setting.
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
from sources import from_bytes, byte_cooc_svd_embedding


DOWNLOADS = (Path.home() / "Downloads")
SIMPLEWIKI = DOWNLOADS / "simplewiki-20260301-pages-articles-multistream.xml.bz2"
ENGRAM_DIR = Path(__file__).resolve().parent


def section(t: str):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}", flush=True)


def pull_mixed(n_each: int = 100):
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


def run_with_ee(items, source, ee: float, td: str, label: str):
    bp = str(Path(td) / f"{label}.engram")
    ef = EngramFile.create(bp, name=label, dim=source.dim,
                           metadata={}, annotations={})
    by_well: Dict[str, Dict[str, int]] = {}

    def on_article(title: str, well: str, m: dict):
        cls = "code" if title.startswith("code/") else "prose"
        by_well.setdefault(well, {"code": 0, "prose": 0})[cls] += 1

    t0 = time.time()
    n = genesis_chain(
        ef, iter(items), source, label=label,
        batch_size=24, batch_timeout_ms=100, n_cpu_workers=4,
        progress_every=99999, telemetry_interval_s=99999.0,
        min_tokens=10, on_article=on_article,
        escape_energy=ee,
    )
    dt = time.time() - t0

    counts = sorted(ef._well_count.values(), reverse=True)
    correct = sum(max(v.get("code", 0), v.get("prose", 0))
                  for v in by_well.values())
    total = sum(v.get("code", 0) + v.get("prose", 0)
                for v in by_well.values())
    purity_score = correct / max(total, 1)

    return {
        "ee": ee,
        "absorbed": n,
        "elapsed": dt,
        "wells": len(ef.wells()),
        "top5_counts": counts[:5],
        "purity": purity_score,
        "n_classified": total,
    }


def main():
    if not gpu_available():
        print("wgpu missing"); sys.exit(1)

    code, prose = pull_mixed(100)
    if not code or not prose:
        print("need both code + prose"); sys.exit(1)
    items = code + prose
    print(f"corpus: code={len(code)} prose={len(prose)} total={len(items)}")

    sample = b"\n".join(t.encode("utf-8", errors="replace")[:4000]
                        for _i, t in items[:100])
    emb = byte_cooc_svd_embedding(sample, D=32, window=4)
    source = from_bytes(matrix=emb, name="bytes")

    settings = [
        ("low",     1.0),
        ("default", 2.0),
        ("high",    4.0),
    ]

    results = []
    with tempfile.TemporaryDirectory() as td:
        for label, ee in settings:
            section(f"escape_energy = {ee} ({label})")
            r = run_with_ee(items, source, ee, td, f"ee_{label}")
            results.append(r)
            print(f"  absorbed={r['absorbed']}  wells={r['wells']}  "
                  f"purity={r['purity']*100:.1f}%  top5={r['top5_counts']}  "
                  f"{r['elapsed']:.1f}s")

    section("SUMMARY")
    print(f"{'setting':<10} {'ee':<6} {'wells':<8} {'purity':<10} {'top5 counts'}")
    for r in results:
        label = {1.0: "low", 2.0: "default", 4.0: "high"}[r["ee"]]
        print(f"{label:<10} {r['ee']:<6} {r['wells']:<8} "
              f"{r['purity']*100:<9.1f}% {r['top5_counts']}")

    failures = []
    # monotone well count in the expected direction
    wells_low = next(r["wells"] for r in results if r["ee"] == 1.0)
    wells_def = next(r["wells"] for r in results if r["ee"] == 2.0)
    wells_high = next(r["wells"] for r in results if r["ee"] == 4.0)
    if not (wells_low >= wells_def >= wells_high):
        failures.append(
            f"well count not monotone in escape_energy: "
            f"low({wells_low}) default({wells_def}) high({wells_high}) -- "
            f"expected low >= default >= high")
    else:
        print(f"\n  [monotone] low({wells_low}) >= default({wells_def}) "
              f">= high({wells_high})")

    # the low/default should produce DIFFERENT well counts on this
    # corpus size or the knob is barely doing anything useful. allow
    # tie if corpus is small but warn.
    if wells_low == wells_def == wells_high:
        failures.append(
            f"all three settings produce {wells_low} wells -- the knob "
            f"has no measurable effect on this corpus")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("\n[PASS] escape_energy is a real tunable knob with monotone effect")


if __name__ == "__main__":
    main()
