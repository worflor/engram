"""Check maintained Markdown links and source import accounting."""

import json
import hashlib
from pathlib import Path
import re
from html import unescape
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
errors = []


def heading_anchors(path):
    """GitHub-style ATX heading IDs for the repository's Markdown pages."""
    source = path.read_text(encoding="utf-8")
    anchors = set(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)["\']', source))
    duplicates = {}
    fence = None
    for line in source.splitlines():
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if marker:
            value = marker.group(1)
            if fence is None:
                fence = value
            elif value[0] == fence[0] and len(value) >= len(fence):
                fence = None
            continue
        if fence is not None:
            continue
        match = re.match(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not match:
            continue
        text = unescape(re.sub(r"<[^>]+>", "", match.group(1))).lower()
        slug = re.sub(r"[^\w\- ]", "", text).replace(" ", "-")
        duplicate = duplicates.get(slug, 0)
        duplicates[slug] = duplicate + 1
        anchors.add(slug if duplicate == 0 else f"{slug}-{duplicate}")
    return anchors


docs = [
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    *list((ROOT / "docs").rglob("*.md")),
    *[
        ROOT / path
        for path in [
            "examples/README.md",
            "research/README.md",
            "legacy/README.md",
            "integrations/README.md",
            "LICENSES/README.md",
        ]
    ],
]
anchor_checks = 0
for source in docs:
    for target in re.findall(r"\]\(([^)]+)\)", source.read_text(encoding="utf-8")):
        if "://" in target:
            continue
        target, _, fragment = target.partition("#")
        destination = source.parent / unquote(target) if target else source
        if not destination.exists():
            errors.append(f"{source.relative_to(ROOT)}: missing {target}")
        elif fragment and destination.suffix.lower() == ".md":
            anchor_checks += 1
            if unquote(fragment) not in heading_anchors(destination):
                errors.append(
                    f"{source.relative_to(ROOT)}: missing heading {target}#{fragment}"
                )
manifest = json.loads((ROOT / "research/provenance.json").read_text(encoding="utf-8"))
for item in manifest["files"]:
    if not (ROOT / item["path"]).is_file():
        errors.append(f"Missing imported source: {item['path']}")
for item in manifest.get("retired_files", []):
    if (ROOT / item["path"]).exists():
        errors.append(f"Retired run script remains in current checkout: {item['path']}")
    for replacement in item["replacements"]:
        if not (ROOT / replacement).is_file():
            errors.append(f"Missing maintained replacement: {replacement}")
for item in manifest.get("artifacts", []):
    path = ROOT / item["path"]
    if not path.is_file():
        errors.append(f"Missing research artifact: {item['path']}")
    elif (
        path.stat().st_size != item["bytes"]
        or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]
    ):
        errors.append(
            f"Research artifact differs from its frozen identity: {item['path']}"
        )
integration = json.loads(
    (ROOT / "research/integration-map.json").read_text(encoding="utf-8")
)
for category in integration["categories"].values():
    for target in category["maintained_targets"]:
        if not (ROOT / target).is_file():
            errors.append(f"Missing research integration target: {target}")
for entry in integration["entries"]:
    if entry["category"] not in integration["categories"]:
        errors.append(f"Unknown research category: {entry['source']}")
if errors:
    raise SystemExit("\n".join(errors))
print(
    f"{len(docs)} maintained Markdown files, {len(manifest['files'])} imported paths and "
    f"{len(manifest.get('artifacts', []))} frozen research artifacts checked; "
    f"{anchor_checks} local heading links checked"
)
