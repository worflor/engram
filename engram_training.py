"""Reusable streaming training, learned byte coordinates and ordered CPU batches.

These are the portable mechanisms behind the portfolio's Alexandria/genesis
experiments. No corpus locations, downloaders, model endpoints or old run scripts.
Changing coordinates creates a NEW model: never mix embeddings in an old well.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from itertools import islice
import numpy as np
from engram_codec import _real_array

from engram_wire import MAX_DIM, MAX_VALUES


def smooth_trajectory(trajectory, window=1):
    x = _real_array(trajectory, dtype=np.float32)
    if (
        x.ndim != 2
        or not np.isfinite(x).all()
        or x.size > MAX_VALUES
        or not isinstance(window, int)
        or window < 1
    ):
        raise ValueError(
            "finite bounded trajectory and positive integer window required"
        )
    if len(x) < window:
        return np.empty((0, x.shape[1]), np.float32)
    sums = np.vstack(
        [np.zeros((1, x.shape[1])), np.cumsum(x, axis=0, dtype=np.float64)]
    )
    return ((sums[window:] - sums[:-window]) / window).astype(np.float32)


class LookupSource:
    """Validated vocabulary-index source; caller owns tokenizer and OOV policy."""

    def __init__(self, matrix, to_indices, *, name="lookup", window=1):
        self.matrix = _real_array(matrix, dtype=np.float32).copy()
        if (
            self.matrix.ndim != 2
            or not len(self.matrix)
            or not 2 <= self.matrix.shape[1] <= MAX_DIM
            or self.matrix.shape[1] % 2
            or self.matrix.size > MAX_VALUES
            or not np.isfinite(self.matrix).all()
            or not callable(to_indices)
            or not isinstance(window, int)
            or window < 1
        ):
            raise ValueError("invalid lookup source")
        self.matrix.flags.writeable = False
        self.to_indices, self.name, self.window = to_indices, str(name), window

    @property
    def dim(self):
        return self.matrix.shape[1]

    @property
    def identity(self):
        # Includes shape/window; a changed smoothing convention is a changed input space.
        h = hashlib.sha256(self.matrix.astype("<f4").tobytes())
        h.update(f"{self.matrix.shape}:{self.window}:{self.name}".encode())
        return h.hexdigest()

    def project(self, payload):
        ids = np.asarray(self.to_indices(payload))
        if (
            ids.ndim != 1
            or (ids.size and not np.issubdtype(ids.dtype, np.integer))
            or ids.size * self.dim > MAX_VALUES
            or np.any(ids < 0)
            or np.any(ids >= len(self.matrix))
        ):
            raise ValueError(
                "tokenizer must return bounded in-vocabulary integer indices"
            )
        return smooth_trajectory(self.matrix[ids.astype(np.intp)], self.window)

    @classmethod
    def bytes(cls, matrix, *, window=1):
        if np.asarray(matrix).shape[0] != 256:
            raise ValueError("byte vocabulary needs 256 rows")
        return cls(
            matrix,
            lambda b: np.frombuffer(
                b.encode("utf-8") if isinstance(b, str) else b, dtype=np.uint8
            ),
            name="utf8-bytes",
            window=window,
        )


def learn_byte_embedding(documents, *, dim=32, window=4, min_count=1):
    """Symmetric byte co-occurrence -> positive PMI -> truncated SVD.

    Accumulates counts per document, with no invented adjacency at boundaries.
    SVD signs are oriented for repeatability; degenerate subspaces can still
    rotate across linear algebra implementations. Unseen bytes stay zero.
    """
    if (
        not isinstance(dim, int)
        or not 2 <= dim <= 256
        or dim % 2
        or not isinstance(window, int)
        or window < 1
        or not isinstance(min_count, int)
        or min_count < 1
    ):
        raise ValueError("even dimension 2..256 and positive window/count required")
    cooc = np.zeros((256, 256), np.float64)
    for document in documents:
        data = document.encode("utf-8") if isinstance(document, str) else document
        arr = np.frombuffer(data, dtype=np.uint8)
        for offset in range(1, min(window + 1, len(arr))):
            np.add.at(cooc, (arr[:-offset], arr[offset:]), 1)
            np.add.at(cooc, (arr[offset:], arr[:-offset]), 1)
    cooc[cooc < min_count] = 0
    total = cooc.sum()
    if total == 0:
        raise ValueError("no co-occurrence evidence")
    denominator = cooc.sum(axis=1)[:, None] * cooc.sum(axis=0)[None, :]
    pmi = np.zeros_like(cooc)
    present = cooc > 0
    pmi[present] = np.maximum(0, np.log(cooc[present] * total / denominator[present]))
    u, singular, _ = np.linalg.svd(pmi, full_matrices=False)
    vectors = u[:, :dim] * np.sqrt(singular[:dim])
    for column in vectors.T:
        if column[np.argmax(abs(column))] < 0:
            column *= -1
    vectors[cooc.sum(axis=1) == 0] = 0
    return vectors.astype(np.float32)


def refine_byte_embedding(memory, previous, *, prior_strength=10.0):
    """Experimental well/composition feedback. Return new coordinates, never mutate.

    This is a representation proposal, not a convergence or accuracy guarantee.
    Unseen bytes retain their prior. Retrain a fresh network before evaluation.
    """
    previous = _real_array(previous)
    if (
        previous.shape != (256, memory.dim)
        or not np.isfinite(previous).all()
        or not np.isfinite(prior_strength)
        or prior_strength <= 0
    ):
        raise ValueError("matching byte matrix and positive finite prior required")
    target = np.zeros_like(previous)
    mass = np.zeros(256)
    for name, histogram in memory._well_sum_raw.items():
        if name not in memory._well_sum_K:
            continue
        k = memory._well_sum_K[name] / memory._well_count[name]
        # Convert parameter pairs back through the SAME coordinate permutation.
        coordinates = np.column_stack([k.real, k.imag]).ravel()
        original = np.empty_like(coordinates)
        original[memory._reference_pairing] = coordinates
        target += histogram[:, None] * original
        mass += histogram
    if not mass.any():
        raise ValueError("no accumulated byte-composition evidence")
    return (
        (target + prior_strength * previous) / (mass[:, None] + prior_strength)
    ).astype(np.float32)


@dataclass(frozen=True)
class TrainingItem:
    key: str
    payload: object
    well: str | None = None
    sequence_start: bool = True


@dataclass(frozen=True)
class TrainingEvent:
    key: str
    well: str | None
    rows: int
    skipped: bool


class Trainer:
    """Bounded batches; parallel feature extraction, serial ordered state updates.

    Short items are reported as skipped. Invalid items raise before ANY update
    in their batch. Explicit episode boundaries prevent cross-document edges.
    Callbacks run after commit; checkpoint/restart is at committed batch/item
    boundaries, with corpus offsets owned by the caller (not guessed from IDs).
    """

    def __init__(self, network, source=None, *, source_id=None):
        self.network, self.source = network, source
        if source is not None and source.dim != network.dim:
            raise ValueError("source and network dimensions differ")
        identity = (
            source.identity
            if source is not None
            else (source_id or "caller-trajectories")
        )
        saved = network.memory.metadata.get("training_source")
        if saved is not None and saved != identity:
            raise ValueError(
                "input representation differs from checkpoint; train a new network"
            )
        network.memory.metadata["training_source"] = identity

    def fit(self, items, *, batch_size=32, workers=1, on_item=None, fitter=None):
        if (
            not isinstance(batch_size, int)
            or batch_size < 1
            or not isinstance(workers, int)
            or workers < 1
        ):
            raise ValueError("positive integer batch size and worker count required")

        def prepare(item):
            if not isinstance(item, TrainingItem) or not isinstance(item.key, str):
                raise ValueError("TrainingItem with a string key required")
            if item.well is not None and (
                not isinstance(item.well, str)
                or not item.well
                or len(item.well.encode("utf-8")) > 65535
            ):
                raise ValueError("invalid well label")
            trajectory = (
                self.source.project(item.payload)
                if self.source is not None
                else _real_array(item.payload, dtype=np.float32)
            )
            from engram_codec import _validated_trajectory

            trajectory, _ = _validated_trajectory(trajectory, self.network.dim)
            signature = (
                self.network.signature(trajectory)
                if len(trajectory) >= 3 and fitter is None
                else None
            )
            raw = None
            if self.source is not None and self.source.name == "utf8-bytes":
                data = (
                    item.payload.encode("utf-8")
                    if isinstance(item.payload, str)
                    else item.payload
                )
                raw = np.bincount(np.frombuffer(data, np.uint8), minlength=256).astype(
                    float
                )
            return item, trajectory, signature, raw

        iterator = iter(items)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            while batch := list(islice(iterator, batch_size)):
                prepared = list(executor.map(prepare, batch))
                if fitter is not None:
                    eligible = [x for _, x, _, _ in prepared if len(x) >= 3]
                    results = list(
                        fitter.fit(
                            eligible, dim=self.network.dim, pairing=self.network.pairing
                        )
                    )
                    if len(results) != len(eligible):
                        raise ValueError(
                            "fitter returned a different number of signatures"
                        )
                    signatures = iter(results)
                    prepared = [
                        (item, x, next(signatures) if len(x) >= 3 else None, raw)
                        for item, x, _, raw in prepared
                    ]
                    for _, _, signature, _ in prepared:
                        if signature is not None:
                            self.network._check_signature(signature)
                for item, trajectory, signature, raw in prepared:
                    if item.sequence_start:
                        self.network.begin_sequence()
                    well = (
                        None
                        if signature is None
                        else self.network.learn_signature(signature, well=item.well)
                    )
                    if raw is not None and well is not None:
                        store = self.network.memory._well_sum_raw
                        store[well] = store.get(well, np.zeros(256)) + raw
                    event = TrainingEvent(
                        item.key, well, len(trajectory), signature is None
                    )
                    if on_item is not None:
                        on_item(event)
                    yield event
