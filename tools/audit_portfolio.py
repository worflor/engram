"""Account for original Engram research sources without importing or running them.

Usage: python tools/audit_portfolio.py --portfolio ../worflor.github.io/rag_tests
The retained inventory identifies mechanisms and nonportable host experiments;
it is deliberately not a claim that every historical entry point still runs.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def classify(relative):
    path = relative.lower()
    if "crystallography" in path or "seance/" in path:
        return (
            "host-model-audit",
            ["engram_metrics.py", "tests/python/test_network.py"],
            "Centered variance, persistence and identity-shuffle controls integrated; transformer execution remains a consumer experiment.",
        )
    if (
        "/shaders/" in path
        or "/gpu/" in path
        or "/mass/" in path
        or any(
            s in path
            for s in [
                "hetero",
                "gpu_",
                "mass_embed",
                "bench",
                "stress",
                "profile",
                "fast_embed",
            ]
        )
    ):
        return (
            "acceleration-and-batching",
            [
                "engram_acceleration.py",
                "engram_training.py",
                "examples/gpu_training.py",
            ],
            "Coalesced fitting, stable two-pass RMS, bounded ordered batches and CPU-authoritative updates integrated; historical kernel variants are research comparisons.",
        )
    if any(
        s in path
        for s in [
            "source",
            "genesis",
            "ingest",
            "alexandria",
            "curriculum",
            "finalize",
            "refinement",
            "well_naming",
        ]
    ):
        return (
            "training-and-coordinates",
            ["engram_training.py", "engram_network.py", "examples/train_bytes.py"],
            "Reusable streaming training, byte PMI/SVD, composition refinement, checkpoint resume, consolidation and numeric well descriptions integrated; corpus schedules and exact old runs are not library APIs.",
        )
    if any(
        s in path
        for s in ["mouth", "model_size", "prompt", "logos", "inner_voice", "real_air"]
    ):
        return (
            "host-language-experiment",
            ["engram_retrieval.py", "engram_embeddings.py", "engram_network.py"],
            "Embedding, relationship and memory interfaces integrated; local language-model prompting and semantic claims require separate host evaluation.",
        )
    if any(
        s in path
        for s in [
            "dream",
            "loop",
            "mirror",
            "partner",
            "offspring",
            "mutual",
            "symbiosis",
            "generation",
            "cpr",
            "replication",
        ]
    ):
        return (
            "generation-and-feedback",
            [
                "engram_codec.py",
                "engram_network.py",
                "examples/feedback.py",
                "examples/well_network.py",
            ],
            "Parameter-history dreaming, per-well continuation, graph walks, lazy meta fitting and reproducible bounded feedback integrated; generation is evaluated separately from reconstruction.",
        )
    return (
        "core-geometry-and-replay",
        [
            "engram_codec.py",
            "engram_analysis.py",
            "engram_wire.py",
            "engram_memory_io.py",
            "tests/python/test_reference.py",
            "tests/python/test_network.py",
        ],
        "Shared coordinate, K/G/RMS, graph, spectrum, replay/import and numerical contracts have maintained regression coverage; exploratory conclusions are not promoted automatically.",
    )


def build(portfolio):
    entries = []
    categories = {}
    for path in sorted(portfolio.rglob("*")):
        relative = path.relative_to(portfolio).as_posix()
        parts = relative.split("/")
        if (
            not path.is_file()
            or any(
                p in {"cache", "__pycache__", "results", ".venv", "node_modules"}
                for p in parts
            )
            or path.suffix not in {".py", ".wgsl", ".md", ".rs", ".dart"}
        ):
            continue
        data = path.read_bytes()
        text = data.decode("utf-8", errors="replace")
        if not (
            parts[0] == "engram"
            or "engram" in path.stem.lower()
            or "engram" in text.lower()
        ):
            continue
        category, targets, disposition = classify(relative)
        categories[category] = {
            "maintained_targets": targets,
            "disposition": disposition,
        }
        entry = {
            "source": relative,
            "sha256": hashlib.sha256(data).hexdigest(),
            "category": category,
        }
        if relative in {
            "engram/alexandria.py",
            "engram/genesis.py",
            "engram/genesis_chain.py",
        }:
            entry["run_procedure_retired_from_checkout"] = True
        entries.append(entry)
    return {
        "schema": "engram-research-integration-1",
        "scope": "Original local portfolio rag_tests text sources; no corpora, weights, logs or checkpoints.",
        "method": "Content/path inventory and mechanism-level integration. Categories route review; they do not certify historical scripts.",
        "categories": categories,
        "entries": entries,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio", type=Path, required=True)
    args = parser.parse_args()
    if not args.portfolio.is_dir():
        raise SystemExit("Portfolio source directory does not exist")
    result = build(args.portfolio)
    (ROOT / "research/integration-map.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Accounted for {len(result['entries'])} original research text sources")
