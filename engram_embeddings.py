"""Optional token-vector input adapter, including Manifold's GLV1 representation."""

from dataclasses import dataclass
import numpy as np
from engram_wire import Reader, MAX_DIM, MAX_VALUES


@dataclass
class EmbeddingTable:
    tokens: list[str]
    vectors: np.ndarray

    def __post_init__(self):
        self.vectors = np.array(self.vectors, dtype=np.float32, copy=True)
        if (
            self.vectors.ndim != 2
            or self.vectors.shape[0] != len(self.tokens)
            or self.vectors.size > MAX_VALUES
        ):
            raise ValueError("invalid embedding matrix")
        d = self.vectors.shape[1]
        if (
            not 2 <= d <= MAX_DIM
            or d % 2
            or not np.isfinite(self.vectors).all()
            or len(set(self.tokens)) != len(self.tokens)
        ):
            raise ValueError("invalid embedding dimensions, values or duplicate tokens")
        self._index = {token: i for i, token in enumerate(self.tokens)}

    @classmethod
    def from_glv(cls, data):
        r = Reader(data)
        if bytes(r.take(4)) != b"GLV1" or r.scalar("<I") != 1:
            raise ValueError("unsupported embedding vocabulary format")
        count, d, scale = r.unpack("<IIf")
        if (
            not 2 <= d <= MAX_DIM
            or d % 2
            or count * d > MAX_VALUES
            or not np.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError("invalid embedding vocabulary header")
        tokens = [bytes(r.take(r.scalar("<B"))).decode("utf-8") for _ in range(count)]
        vectors = r.array(count * d, "<i2").reshape(count, d).astype(np.float32) * (
            scale / 32767.0
        )
        r.finish()
        return cls(tokens, vectors)

    def project(self, tokens):
        """Return matched rows in input order; unknown tokens are omitted explicitly."""
        rows = [self._index[token] for token in tokens if token in self._index]
        return self.vectors[rows].copy()

    def signature(self, tokens, pairing, window=10):
        from engram_analysis import Signature

        if not isinstance(window, int) or window < 1:
            raise ValueError("window must be a positive integer")
        trajectory = self.project(tokens)
        if len(trajectory) < 2:
            raise ValueError("at least two vocabulary matches are required")
        # Leave at least two time samples for WDYM's explicit AR(1) path.
        width = min(window, len(trajectory) - 1)
        sums = np.vstack(
            [
                np.zeros((1, trajectory.shape[1])),
                np.cumsum(trajectory, dtype=np.float64, axis=0),
            ]
        )
        smooth = ((sums[width:] - sums[:-width]) / width).astype(np.float32)
        return Signature.from_trajectory(smooth, pairing=pairing)
