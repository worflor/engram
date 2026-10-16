"""Encode, reconstruct and persist a small statistical memory, entirely offline."""
import tempfile
from pathlib import Path
import numpy as np
from engram_codec import Engram, EngramFile, from_wire, to_wire

samples = np.sin(np.arange(128)[:, None] * .07 + np.arange(4)).astype(np.float32)
codec = Engram(dim=4)
packet = codec.encode(samples)
wire = to_wire(packet)
reconstructed = codec.decode(from_wire(wire))
with tempfile.TemporaryDirectory() as directory:
    path = str(Path(directory) / "example.engram")
    brain = EngramFile.create(path, name="example", dim=4)
    brain.absorb(packet, source="oscillation")
    brain.save()
    loaded = EngramFile.load(path)
    assert loaded.name == "example"
print(f"{packet.length} samples, {len(packet.blocks)} blocks, {len(wire)} wire bytes, "
      f"RMS error {np.sqrt(np.mean((samples-reconstructed)**2)):.6f}")
