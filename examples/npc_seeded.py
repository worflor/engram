"""Inspect Alexandria as a read-only dynamics prior with a supplied matching vocabulary.

python examples/npc_seeded.py --vocabulary path/to/glove300.bin
The vocabulary is external; this program does not download or bundle weights.
"""

import argparse
from pathlib import Path
import numpy as np
from engram_embeddings import EmbeddingTable
from engram_training import smooth_trajectory
from engram_npc import DynamicsPrior


def run(vocabulary):
    representation = "manifold-glv1-glove300-window10"
    table = EmbeddingTable.from_glv(Path(vocabulary).read_bytes())
    tokens = (
        "blue stone walk water red grass open road tree sky move path " * 4
    ).split()
    trajectory = smooth_trajectory(table.project(tokens), window=10)
    prior = DynamicsPrior.load(
        Path(__file__).resolve().parents[1] / "research/artifacts/alexandria.engram",
        representation=representation,
    )
    for vector in trajectory:
        profile = prior.observe(vector, representation=representation)
    assert profile and all(np.isfinite(row["distance"]) for row in profile)
    for row in profile[:5]:
        print(row)
    print(
        "These are dynamics affinities in the supplied vocabulary coordinates; action consequences come from environment experience."
    )
    return profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", type=Path, required=True)
    run(parser.parse_args().vocabulary)
