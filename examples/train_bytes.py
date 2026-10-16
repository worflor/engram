"""Alexandria-style reusable training on a tiny authored corpus, entirely offline."""

import tempfile
from pathlib import Path
import numpy as np
from engram_network import WellNetwork
from engram_training import (
    LookupSource,
    Trainer,
    TrainingItem,
    learn_byte_embedding,
    refine_byte_embedding,
)

DOCUMENTS = [
    "def rotate(vector, angle): return vector * complex(cos(angle), sin(angle))\n",
    "for frame in frames: network.learn(frame); checkpoint.save()\n",
    "The river moves around the rocks and carries the fallen leaves downstream. ",
    "The wind turns slowly over the hill while the evening light fades. ",
]


def run():
    corpus = [text * 8 for text in DOCUMENTS]
    matrix = learn_byte_embedding(corpus, dim=16)
    source = LookupSource.bytes(matrix, window=4)
    network = WellNetwork(16, novelty_radius=0.12)
    items = [TrainingItem(f"document-{i}", text) for i, text in enumerate(corpus)]
    events = list(Trainer(network, source).fit(items, workers=2, batch_size=2))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.engram"
        network.save(path)
        restored = WellNetwork.load(path)
        Trainer(restored, source)  # checks the coordinate identity before resuming
        np.testing.assert_array_equal(
            network.memory._dream_Ks, restored.memory._dream_Ks
        )
        print(
            f"Learned {matrix.shape} byte coordinates, trained {len(events)} documents, "
            f"restored {len(restored.describe())} wells"
        )
    proposed = refine_byte_embedding(network.memory, matrix)
    # Coordinates changed: fresh sufficient statistics, not old wells in a new basis.
    refined_source = LookupSource.bytes(proposed, window=4)
    refined = WellNetwork(16, novelty_radius=0.12)
    list(Trainer(refined, refined_source).fit(items))
    print(
        f"Refinement coordinate RMS change: {np.sqrt(np.mean((proposed - matrix) ** 2)):.6f}; "
        f"retrained {len(refined.describe())} wells (no accuracy/convergence claim)"
    )
    return network


if __name__ == "__main__":
    run()
