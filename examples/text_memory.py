"""Authored text -> shared-coordinate observations -> named wells -> disposable snapshot."""
from pathlib import Path
import tempfile
import numpy as np
from engram_codec import ByteHistogram, Engram, EngramFile

documents = {
    "garden": ("Leaves gather rain. Roots reach through the soil. Seeds open in the warmth. " * 48),
    "workshop": ("Measure the board, sharpen the blade, tighten the bolts and check the joints. " * 48),
}
embedding = ByteHistogram(chunk_size=512)
codec = Engram(dim=256)
with tempfile.TemporaryDirectory() as directory:
    path = str(Path(directory)/"example.engram")
    memory = EngramFile.create(path, name="authored-text", dim=256)
    for label, text in documents.items():
        trajectory = embedding.text_to_trajectory(text)
        measurement = memory.measure(codec, trajectory)
        packet = measurement["packet"]  # Encoded using the memory's reference pairing.
        routed = memory.absorb(packet, source=label)
        print(f"{label}: source label {routed}; drift {measurement['drift']:.2f}; texture {measurement['texture']:.2f}")
    before = {name: value.copy() for name, value in memory.wells().items()}
    memory.save()
    restored = EngramFile.load(path)
    assert set(before) == set(restored.wells())
    for name, centroid in before.items():
        np.testing.assert_array_equal(centroid, restored.wells()[name])
    print(f"Restored {len(before)} wells. Labels were supplied; no semantic accuracy is inferred.")
