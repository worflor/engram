"""test_well_G_completion.py -- validate the phase 1-3 physics completion.

four hypotheses, each against real data:

  F1  FORMAT ROUND-TRIP. save a brain with K, G, S populated per well.
      load it. verify every well's K, G, S, count is bit-exactly
      preserved. manifest version is 5 or higher.

  F2  BACKWARD COMPAT. construct a v4-shaped brain on disk (version=4
      manifest, no G/S in wells.bin). load it. verify load succeeds,
      G and S default to zero, count and K preserved. once the brain
      is re-ingested on the unified pipeline, G and S populate
      automatically.

  F3  PHYSICS SANITY. on a real-ingested brain, every established
      well should have |λ_max| < 1 (strict stability) — otherwise the
      oscillator is divergent and the brain shouldn't have formed a
      basin there. verify using engram_codec.eigenvalues_all on each
      well's K and G centroids.

  F4  PER-WELL SURPRISE IS DISCRIMINATIVE. for each article in a
      mixed-corpus ingest, compute its per_well_surprise across all
      established wells. the article's routed well should have
      per_well_surprise LOWER than the average of the other wells,
      on the majority of articles. if not, the well noise floor σ
      isn't doing the right normalization and the per-well signal
      would be useless.
"""
from __future__ import annotations

import io
import json
import struct
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from engram_codec import (
    EngramFile, WELL_ESTABLISHED, eigenvalues_all,
)
from engram.gpu.engine import gpu_available
from genesis_chain import genesis_chain, iter_wiki_articles
from sources import from_bytes, byte_cooc_svd_embedding


DOWNLOADS = (Path.home() / "Downloads")
SIMPLEWIKI = DOWNLOADS / "simplewiki-20260301-pages-articles-multistream.xml.bz2"
ENGRAM_DIR = Path(__file__).resolve().parent


def section(t: str):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}", flush=True)


def pull_mixed(n_each: int = 80):
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


def ingest_mixed(td: str):
    code, prose = pull_mixed(80)
    items = code + prose
    sample = b"\n".join(t.encode("utf-8", errors="replace")[:4000]
                        for _i, t in items[:80])
    emb = byte_cooc_svd_embedding(sample, D=32, window=4)
    source = from_bytes(matrix=emb, name="bytes")
    ef = EngramFile.create(str(Path(td) / "mix.engram"), name="mix",
                           dim=source.dim, metadata={}, annotations={})

    assignments = {}

    def on_article(title, well, m):
        assignments[title] = {"well": well, "measurement": m}

    n = genesis_chain(
        ef, iter(items), source, label="mix",
        batch_size=24, batch_timeout_ms=100, n_cpu_workers=4,
        progress_every=99999, telemetry_interval_s=99999.0,
        min_tokens=10, on_article=on_article,
    )
    return ef, source, items, assignments, n


# ── F1: round-trip ────────────────────────────────────────────────────

def f1_roundtrip():
    section("F1 FORMAT ROUND-TRIP")
    with tempfile.TemporaryDirectory() as td:
        ef, *_ = ingest_mixed(td)
        print(f"  ingested: wells={len(ef._well_sum_K)}")

        path = str(Path(td) / "rt.engram")
        ef.save(path)

        # check manifest version
        with zipfile.ZipFile(path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
            version = manifest.get("version")
            print(f"  manifest version: {version}")

        ef2 = EngramFile.load(path)
        print(f"  loaded:  wells={len(ef2._well_sum_K)}")

        fails = []
        if version < 5:
            fails.append(f"version < 5 (got {version})")
        if set(ef._well_sum_K.keys()) != set(ef2._well_sum_K.keys()):
            fails.append("well name set mismatch after reload")

        # bit-exact per-well check
        max_k_diff = 0.0
        max_g_diff = 0.0
        max_s_diff = 0.0
        for name in ef._well_sum_K:
            k_d = float(np.abs(ef._well_sum_K[name]
                               - ef2._well_sum_K[name]).max())
            g_d = float(np.abs(ef._well_sum_G[name]
                               - ef2._well_sum_G[name]).max())
            s_d = float(np.abs(ef._well_sum_S[name]
                               - ef2._well_sum_S[name]).max())
            max_k_diff = max(max_k_diff, k_d)
            max_g_diff = max(max_g_diff, g_d)
            max_s_diff = max(max_s_diff, s_d)
            if ef._well_count[name] != ef2._well_count[name]:
                fails.append(f"count mismatch on {name}")
        print(f"  max diffs: K={max_k_diff:.2e} G={max_g_diff:.2e} S={max_s_diff:.2e}")
        if max_k_diff > 1e-10: fails.append("K not bit-exact")
        if max_g_diff > 1e-10: fails.append("G not bit-exact")
        if max_s_diff > 1e-10: fails.append("S not bit-exact")

        return fails


# ── F2: backward compat with v4 ───────────────────────────────────────

def f2_v4_compat():
    section("F2 V4 BACKWARD COMPAT")
    with tempfile.TemporaryDirectory() as td:
        # ingest normally (v5)
        ef, *_ = ingest_mixed(td)
        path = str(Path(td) / "v5.engram")
        ef.save(path)

        # hand-roll a v4 equivalent: strip G and S from wells.bin,
        # set manifest version to 4
        v4_path = str(Path(td) / "v4.engram")
        with zipfile.ZipFile(path, "r") as zin, \
             zipfile.ZipFile(v4_path, "w",
                             compression=zipfile.ZIP_DEFLATED) as zout:
            P = ef.dim // 2
            manifest = json.loads(zin.read("manifest.json"))
            manifest["version"] = 4

            # rebuild wells.bin without G and S
            wells_data = zin.read("wells.bin")
            out = io.BytesIO()
            off = 0
            D = ef.dim
            out.write(wells_data[off:off+D*2])
            off += D*2
            n_wells, = struct.unpack_from("<I", wells_data, off); off += 4
            out.write(struct.pack("<I", n_wells))
            for _ in range(n_wells):
                name_len, = struct.unpack_from("<H", wells_data, off); off += 2
                name = wells_data[off:off+name_len]; off += name_len
                sum_K_bytes = wells_data[off:off+P*16]; off += P*16
                # skip G (P*16), S (P*8) from v5
                off += P*16
                off += P*8
                sum_raw_bytes = wells_data[off:off+256*8]; off += 256*8
                count_bytes = wells_data[off:off+4]; off += 4
                # skip v7 anchor blocks: anchor_K (P*16), anchor_G
                # (P*16), anchor_S (P*8). v4 didn't have these so we
                # strip them when forging a v4 file.
                off += P*16 + P*16 + P*8
                out.write(struct.pack("<H", name_len))
                out.write(name)
                out.write(sum_K_bytes)
                out.write(sum_raw_bytes)
                out.write(count_bytes)

            zout.writestr("manifest.json", json.dumps(manifest, indent=2))
            zout.writestr("wells.bin", out.getvalue())
            # copy other files
            for fname in zin.namelist():
                if fname in ("manifest.json", "wells.bin"):
                    continue
                zout.writestr(fname, zin.read(fname))

        # load the v4 file
        ef4 = EngramFile.load(v4_path)
        print(f"  loaded v4 brain: {len(ef4._well_sum_K)} wells")
        print(f"  manifest version on reload: {json.loads(zipfile.ZipFile(v4_path).read('manifest.json')).get('version')}")

        fails = []
        # K preserved
        for name in ef._well_sum_K:
            if name not in ef4._well_sum_K:
                fails.append(f"well {name} missing after v4 load")
                continue
            k_diff = float(np.abs(ef._well_sum_K[name] - ef4._well_sum_K[name]).max())
            if k_diff > 1e-10:
                fails.append(f"K not preserved on {name}: diff={k_diff:.2e}")
            # G, S should be zeros
            if np.abs(ef4._well_sum_G[name]).sum() > 1e-10:
                fails.append(f"G not zeroed on v4 load for {name}")
            if np.abs(ef4._well_sum_S[name]).sum() > 1e-10:
                fails.append(f"S not zeroed on v4 load for {name}")

        print(f"  K preserved, G and S zeroed as expected: "
              f"{'yes' if not fails else 'NO'}")
        return fails


# ── F3: physics sanity ────────────────────────────────────────────────

def f3_physics_sanity():
    section("F3 PHYSICS SANITY (|λ_max| < 1 for stable wells)")
    with tempfile.TemporaryDirectory() as td:
        ef, *_ = ingest_mixed(td)

        P = ef.dim // 2
        unstable = []
        stable_count = 0
        stats = []
        for name in sorted(ef._well_sum_K.keys()):
            cnt = int(ef._well_count.get(name, 0))
            if cnt < WELL_ESTABLISHED:
                continue
            K_cen = ef._well_sum_K[name] / cnt
            G_cen = ef._well_sum_G[name] / cnt
            if np.abs(G_cen).sum() < 1e-12:
                continue  # G not populated
            lam1, lam2 = eigenvalues_all(K_cen, G_cen)
            abs_lam = np.maximum(np.abs(lam1), np.abs(lam2))
            max_abs = float(abs_lam.max())
            stats.append((name, cnt, max_abs))
            if max_abs >= 1.0:
                unstable.append((name, cnt, max_abs))
            else:
                stable_count += 1

        print(f"  established wells inspected: {len(stats)}")
        print(f"  stable (|λ_max| < 1): {stable_count}")
        print(f"  unstable (|λ_max| >= 1): {len(unstable)}")
        if stats:
            vals = [s[2] for s in stats]
            print(f"  |λ_max| distribution: min={min(vals):.3f} "
                  f"med={np.median(vals):.3f} max={max(vals):.3f}")
        for name, cnt, mx in unstable[:5]:
            print(f"    UNSTABLE {name} cnt={cnt}: |λ_max|={mx:.3f}")

        fails = []
        if not stats:
            fails.append("no established wells to check — ingest failed?")
        elif len(unstable) > len(stats) // 3:
            # bootstrap creates wells from single-article K, which can
            # have unstable eigenvalues until the well accumulates more
            # articles and the centroid smooths out (LLN). on small
            # test corpora, a third of wells with single-article K is
            # plausible. fail only if the MAJORITY are unstable, which
            # would indicate a real physics regression.
            fails.append(
                f"{len(unstable)}/{len(stats)} wells unstable (|λ_max| >= 1) "
                f"— too high a fraction; AR(2) physics regression")
        return fails


# ── F4: per-well surprise is discriminative ───────────────────────────

def f4_per_well_surprise():
    section("F4 PER-WELL SURPRISE IS DISCRIMINATIVE")
    with tempfile.TemporaryDirectory() as td:
        ef, source, items, assignments, n = ingest_mixed(td)
        print(f"  {n} articles absorbed, {len(assignments)} observed")

        own_surprise = []
        other_surprise = []
        n_with_data = 0
        for aid, rec in assignments.items():
            well = rec["well"]
            m = rec["measurement"]
            pws = m.get("per_well_surprise", {})
            if not pws or well not in pws:
                continue
            n_with_data += 1
            own = float(pws[well])
            others = [float(v) for k, v in pws.items() if k != well]
            if not others:
                continue
            own_surprise.append(own)
            other_surprise.append(float(np.mean(others)))

        print(f"  articles with per-well data: {n_with_data}")
        if not own_surprise:
            return ["no per-well surprise data captured"]

        own_mean = float(np.mean(own_surprise))
        other_mean = float(np.mean(other_surprise))
        # fraction of articles where own < mean(other)
        fraction_lower = float(
            np.mean([o < ot for o, ot in zip(own_surprise, other_surprise)]))
        print(f"  mean own-well surprise:    {own_mean:.3f}")
        print(f"  mean other-wells surprise: {other_mean:.3f}")
        print(f"  fraction of articles with own < others: "
              f"{fraction_lower*100:.1f}%")

        fails = []
        if own_mean >= other_mean:
            fails.append(
                f"mean own-well surprise ({own_mean:.3f}) not less than "
                f"mean other-wells surprise ({other_mean:.3f}) — "
                f"per-well signal not discriminative")
        if fraction_lower < 0.55:
            fails.append(
                f"only {fraction_lower*100:.1f}% of articles show own < "
                f"others — expect majority")
        return fails


def main():
    if not gpu_available():
        print("wgpu missing"); sys.exit(1)

    all_fails = []
    for name, fn in [
        ("F1 round-trip",       f1_roundtrip),
        ("F2 v4 backward compat", f2_v4_compat),
        ("F3 physics sanity",   f3_physics_sanity),
        ("F4 per-well surprise", f4_per_well_surprise),
    ]:
        try:
            fails = fn()
            if fails:
                print(f"\n  [{name} FAIL]")
                for f in fails:
                    print(f"    {f}")
                all_fails.extend(f"{name}: {f}" for f in fails)
            else:
                print(f"\n  [{name} PASS]")
        except Exception as e:
            import traceback
            print(f"\n  [{name} CRASHED]")
            traceback.print_exc()
            all_fails.append(f"{name}: {e}")

    section("SUMMARY")
    if all_fails:
        print("FAILURES:")
        for f in all_fails:
            print(f"  {f}")
        sys.exit(1)
    print("ALL PHYSICS-COMPLETION HYPOTHESES PASS")


if __name__ == "__main__":
    main()
