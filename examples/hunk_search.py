"""Manifold-style relationships: geometric motion score plus explicit identifiers."""

import numpy as np
from engram_retrieval import SignatureIndex, code_tokens
from well_network import signal

rng = np.random.default_rng(4)
index = SignatureIndex(4)
for family, identifier in enumerate(["renderFrame", "readCheckpoint", "rotateVector"]):
    index.add(identifier, signal(family, rng), identifiers=code_tokens(identifier))
query = signal(2, np.random.default_rng(99))
for hit in index.search(
    query, identifiers=code_tokens("rotateVector3"), lexical_weight=0.2
):
    print(
        f"{hit.key}: score={hit.score:.4f}, geometry={hit.geometry:.4f}, "
        f"identifier overlap={hit.identifier_overlap:.4f}, distance={hit.distance:.4f}"
    )
assert index.search(query)[0].key == "rotateVector"
# Real code retrieval replaces these authored signal trajectories with token
# embeddings in a shared basis. This example isolates the relationship mechanism.
