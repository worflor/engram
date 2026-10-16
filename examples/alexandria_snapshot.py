"""Inspect the frozen research artifact and generate from its recent history.

No retraining, corpus access or language model. The original bytes stay untouched.
"""

from pathlib import Path
import hashlib
import numpy as np
from engram_codec import Engram, EngramFile

path = Path(__file__).resolve().parents[1] / "research/artifacts/alexandria.engram"
digest = hashlib.sha256(path.read_bytes()).hexdigest()
memory = EngramFile.load(path)
assert memory.dim == 300 and len(memory.wells()) == 225
assert np.isfinite(memory.fingerprint()).all()
dream = Engram(memory.dim).dream(
    memory.dream_packets, n_blocks=2, samples_per_block=8, rng=np.random.default_rng(42)
)
assert dream is not None and dream.shape[1] == 300 and np.isfinite(dream).all()
assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
print(
    f"Alexandria: {len(memory.wells())} wells, {memory._total_absorbed:,} recorded absorptions, "
    f"{len(memory._dream_Ks)} recent histories"
)
print(
    f"Generated history-driven trajectory: {dream.shape}; finite RMS={np.sqrt(np.mean(dream.astype(float) ** 2)):.6f}"
)
print("These are generated coordinates, not reconstructed source passages or words.")
