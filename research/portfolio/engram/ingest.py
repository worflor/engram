"""
ingest.py

feeds Alexandria from local dumps: Wikipedia bz2 + Gutenberg ZIM.

streams articles from compressed archives without extracting to disk.
each article is converted to a byte histogram trajectory and absorbed
into the brain. progress is tracked so the pipeline can resume.

sources:
    wikipedia:  MediaWiki XML in bz2 (enwiki, jawiki, etc.)
    gutenberg:  ZIM archives (fiction, by Library of Congress class)

usage:
    python ingest.py --wiki path/to/enwiki.xml.bz2
    python ingest.py --zim path/to/gutenberg.zim --source fiction
    python ingest.py --wiki enwiki.bz2 --zim gutenberg.zim  # both
    python ingest.py --brain path/to/brain.engram            # specify brain
    python ingest.py --limit 1000                            # first N articles
"""

from __future__ import annotations

import sys
import os
import re
import json
import time
import bz2
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import lxml.etree as ET
import mwparserfromhell
from engram_codec import EngramFile

# ── MediaWiki XML stream parser ───────────────────────────────────────

def iter_wiki_articles(path: str, limit: int = 0):
    """
    stream articles from a MediaWiki XML bz2 dump.
    yields (title, plaintext) tuples.
    skips redirects, disambiguations, and stubs.
    """
    ns = ""
    count = 0
    with bz2.open(path, "rb") as f:
        for event, elem in ET.iterparse(f, events=("end",)):
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

            if tag == "page":
                title_el = elem.find(f"{ns}title") if ns else None
                text_el = elem.find(f".//{ns}text") if ns else None

                # try without namespace
                if title_el is None:
                    title_el = elem.find("title")
                if text_el is None:
                    for child in elem.iter():
                        if child.tag.endswith("text"):
                            text_el = child
                            break

                if title_el is not None and text_el is not None:
                    title = title_el.text or ""
                    wikitext = text_el.text or ""

                    # skip redirects, stubs, meta pages
                    if wikitext.startswith("#REDIRECT"):
                        elem.clear()
                        continue
                    if len(wikitext) < 2000:
                        elem.clear()
                        continue
                    if ":" in title:  # skip Talk:, User:, etc.
                        elem.clear()
                        continue

                    # parse wikitext to plaintext
                    try:
                        parsed = mwparserfromhell.parse(wikitext)
                        plaintext = parsed.strip_code()
                    except Exception:
                        plaintext = re.sub(r"\{\{[^}]*\}\}", "", wikitext)
                        plaintext = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", plaintext)
                        plaintext = re.sub(r"<[^>]+>", "", plaintext)
                        plaintext = re.sub(r"'{2,}", "", plaintext)

                    # clean up
                    plaintext = re.sub(r"\n{3,}", "\n\n", plaintext).strip()
                    if len(plaintext) < 1000:
                        elem.clear()
                        continue

                    count += 1
                    yield title, plaintext

                    if limit and count >= limit:
                        return

                elem.clear()

            # detect namespace on first element
            if not ns and "}" in elem.tag:
                ns = elem.tag.split("}")[0] + "}"


# ── ZIM (Gutenberg) parser ────────────────────────────────────────────

def iter_zim_articles(path: str, limit: int = 0):
    """
    stream articles from a ZIM archive (Gutenberg dumps).
    yields (title, plaintext) tuples.
    strips HTML to plain text.
    """
    from libzim import Archive

    zim = Archive(path)
    count = 0

    for i in range(zim.entry_count):
        entry = zim._get_entry_by_id(i)
        if entry.is_redirect:
            continue

        item = entry.get_item()
        if item.mimetype != "text/html":
            continue

        html = bytes(item.content).decode("utf-8", errors="ignore")
        if len(html) < 5000:
            continue

        # strip HTML to plaintext
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-z]+;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < 2000:
            continue

        # title from path
        title = entry.path.split("/")[-1].replace("_", " ")
        # clean up Gutenberg path artifacts
        title = re.sub(r"\.\d+$", "", title)

        count += 1
        yield title, text

        if limit and count >= limit:
            return


# ── pipeline ──────────────────────────────────────────────────────────

def run():
    parser = argparse.ArgumentParser(description="ingest dumps into Alexandria")
    parser.add_argument("--wiki", nargs="*", default=[], help="Wikipedia bz2 dump paths")
    parser.add_argument("--zim", nargs="*", default=[], help="ZIM archive paths")
    parser.add_argument("--brain", default=str(Path(__file__).parent / "cache" / "_alexandria.engram"))
    parser.add_argument("--limit", type=int, default=0, help="max articles per source (0=all)")
    parser.add_argument("--source", default=None, help="source tag for ZIM articles")
    parser.add_argument("--fresh", action="store_true", help="create new brain")
    parser.add_argument("--supervised", action="store_true", help="use source tags (vs unsupervised)")
    args = parser.parse_args()

    progress_path = args.brain.replace(".engram", "_ingest_progress.json")

    # load or create brain
    if args.fresh or not os.path.exists(args.brain):
        ef = EngramFile.create(args.brain, name="Alexandria", dim=256,
                               metadata={"description": "built from local dumps"},
                               annotations={"system_card": "You are Alexandria."})
        progress = {"absorbed": set()}
        print(f"created new brain: {args.brain}")
    else:
        ef = EngramFile.load(args.brain)
        if os.path.exists(progress_path):
            with open(progress_path) as f:
                p = json.load(f)
                progress = {"absorbed": set(p.get("absorbed", []))}
        else:
            progress = {"absorbed": set()}
        print(f"loaded brain: {ef.name}, {ef._total_absorbed} articles, {len(ef.wells())} wells")

    total_new = 0
    t_start = time.time()

    # ── process sources ──

    sources = []
    for path in args.wiki:
        tag = Path(path).stem.split("-")[0]  # enwiki, jawiki, etc.
        sources.append(("wiki", path, tag))
    for path in args.zim:
        tag = args.source or Path(path).stem.split("_")[1] if "_" in Path(path).stem else "gutenberg"
        sources.append(("zim", path, tag))

    for kind, path, default_tag in sources:
        print(f"\n{'='*60}")
        print(f"  source: {Path(path).name} ({kind}, tag={default_tag})")
        print(f"{'='*60}\n")

        if kind == "wiki":
            articles = iter_wiki_articles(path, limit=args.limit)
        else:
            articles = iter_zim_articles(path, limit=args.limit)

        source_count = 0
        for title, text in articles:
            key = f"{default_tag}:{title}"
            if key in progress["absorbed"]:
                continue

            source_tag = default_tag if args.supervised else None
            well = ef.absorb_text(text, source=source_tag)
            progress["absorbed"].add(key)
            total_new += 1
            source_count += 1

            if total_new % 100 == 0:
                elapsed = time.time() - t_start
                rate = total_new / elapsed
                print(f"  [{total_new:6d}] {title[:50]:50s}  "
                      f"well={well:15s}  wells={len(ef.wells()):3d}  "
                      f"{rate:.1f} art/s")

            # checkpoint every 500 articles
            if total_new % 500 == 0:
                ef.save(args.brain)
                with open(progress_path, "w") as f:
                    json.dump({"absorbed": list(progress["absorbed"])}, f)
                size = os.path.getsize(args.brain)
                print(f"  -- checkpoint: {ef._total_absorbed} absorbed, "
                      f"{len(ef.wells())} wells, {size/1024:.0f} KB --")

        print(f"  {source_count} articles from {Path(path).name}")

    # final save
    ef.save(args.brain)
    with open(progress_path, "w") as f:
        json.dump({"absorbed": list(progress["absorbed"])}, f)

    elapsed = time.time() - t_start
    size = os.path.getsize(args.brain)

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")
    print(f"  new articles: {total_new}")
    print(f"  total absorbed: {ef._total_absorbed}")
    print(f"  wells: {len(ef.wells())}")
    print(f"  brain: {size:,} bytes ({size/1024:.0f} KB)")
    print(f"  time: {elapsed:.0f}s ({total_new/max(elapsed,1):.1f} art/s)")


if __name__ == "__main__":
    run()
