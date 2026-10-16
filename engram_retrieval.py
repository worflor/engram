"""Inspectably scored hunk/document relationships, inspired by Manifold.

Retrieval retains document IDs and signatures; memory wells alone cannot recover
document content. The host owns tokenization, embeddings, evidence and answers.
"""

from dataclasses import dataclass
import re
import numpy as np
from engram_analysis import Signature


def code_tokens(text):
    """ASCII identifier splitting: camelCase, HTTPServer, snake_case, digits."""
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [token.lower() for token in re.findall(r"[A-Za-z]{2,48}", text)]


@dataclass(frozen=True)
class SearchHit:
    key: str
    score: float
    geometry: float
    identifier_overlap: float
    distance: float


class SignatureIndex:
    def __init__(self, dim, *, pairing=None):
        from engram_codec import Engram, _validated_trajectory

        Engram(dim)
        _, self.pairing = _validated_trajectory(
            np.zeros((0, dim)), dim, np.arange(dim) if pairing is None else pairing
        )
        self.dim, self._entries = dim, {}

    def add(self, key, trajectory, *, identifiers=()):
        if not isinstance(key, str) or not key:
            raise ValueError("nonempty document key required")
        x = np.asarray(trajectory)
        if x.ndim != 2 or x.shape[1] != self.dim or len(x) < 3:
            raise ValueError("index needs same-dimension AR(2) trajectories")
        signature = Signature.from_trajectory(x, pairing=self.pairing)
        tokens = frozenset(identifiers)
        if not all(isinstance(t, str) for t in tokens):
            raise ValueError("string identifiers required")
        self._entries[key] = (signature, tokens)

    def search(self, trajectory, *, identifiers=(), lexical_weight=0.0, limit=5):
        x = np.asarray(trajectory)
        if (
            x.ndim != 2
            or x.shape[1] != self.dim
            or len(x) < 3
            or not np.isfinite(lexical_weight)
            or not 0 <= lexical_weight <= 1
            or not isinstance(limit, int)
            or limit < 1
        ):
            raise ValueError("invalid query, blend or result count")
        query = Signature.from_trajectory(x, pairing=self.pairing)
        tokens = frozenset(identifiers)
        if not all(isinstance(t, str) for t in tokens):
            raise ValueError("string identifiers required")
        hits = []
        for key, (signature, stored) in self._entries.items():
            comparison = query.compare(signature)
            overlap = (
                len(tokens & stored) / len(tokens | stored) if tokens | stored else 0.0
            )
            geometry = comparison["joint_cosine"]
            score = (1 - lexical_weight) * geometry + lexical_weight * overlap
            hits.append(
                SearchHit(key, score, geometry, overlap, comparison["kg_rms_distance"])
            )
        return sorted(hits, key=lambda hit: (-hit.score, hit.distance, hit.key))[:limit]
