"""Online geometric learning over Engram wells; no autodiff or external model.

Nodes learn sufficient statistics of AR(2) signatures. Directed edges learn
sequence transitions. Distance, novelty radius and activation temperature are
explicit host choices in the coordinate system of the input trajectories.
"""

from dataclasses import dataclass
from pathlib import Path
import math
import os
import tempfile
import numpy as np

from engram_analysis import Signature
from engram_codec import EngramFile, _validated_trajectory, _to_complex, _from_complex
from engram_codec import apply_pairing, undo_pairing, eigenvalues_all


@dataclass(frozen=True)
class Activation:
    well: str
    distance: float
    weight: float  # normalized radial activation, not a calibrated probability


@dataclass(frozen=True)
class Dream:
    trajectory: np.ndarray
    itinerary: tuple[str, ...]
    original_radius: float
    stabilized: bool
    mixture: tuple[tuple[str, float], ...] = ()
    mixture_radius: float | None = None


class WellNetwork:
    """A growing radial network whose nodes are complete K/G/RMS oscillators.

    Each observation contributes ONE whole-trajectory signature. This differs
    deliberately from codec absorption, which weights predictive blocks. The
    novelty radius uses sqrt(mean(|dK|² + |dG|² + residual_weight*dRMS²)).
    Labels may explicitly select nodes; otherwise nodes grow without labels.
    """

    def __init__(
        self,
        dim,
        *,
        pairing=None,
        novelty_radius=0.15,
        temperature=0.1,
        residual_weight=1.0,
        memory=None,
    ):
        from engram_codec import Engram

        Engram(dim)
        for name, value in [
            ("novelty_radius", novelty_radius),
            ("temperature", temperature),
            ("residual_weight", residual_weight),
        ]:
            if (
                not math.isfinite(value)
                or value < 0
                or (name == "temperature" and value == 0)
            ):
                raise ValueError(f"invalid {name}")
        _, basis = _validated_trajectory(
            np.zeros((3, dim)), dim, np.arange(dim) if pairing is None else pairing
        )
        self.memory = memory if memory is not None else EngramFile.create("", dim=dim)
        if self.memory.dim != dim:
            raise ValueError("memory dimension differs")
        known = self.memory._reference_pairing
        if known is not None and not np.array_equal(known, basis):
            raise ValueError("memory basis differs")
        self.pairing = basis.copy()
        self.memory._reference_pairing = basis.copy()
        self.novelty_radius = float(novelty_radius)
        self.temperature = float(temperature)
        self.residual_weight = float(residual_weight)
        self._store_config()

    def _store_config(self):
        self.memory.metadata["well_network"] = {
            "version": 1,
            "observation": "whole-trajectory-ar2",
            "novelty_radius": self.novelty_radius,
            "temperature": self.temperature,
            "residual_weight": self.residual_weight,
        }

    @property
    def dim(self):
        return self.memory.dim

    def signature(self, trajectory):
        x = np.asarray(trajectory)
        if x.ndim != 2 or len(x) < 3 or x.shape[1] != self.dim:
            raise ValueError(
                "network observations need at least three rows in its dimension"
            )
        return Signature.from_trajectory(x, pairing=self.pairing)

    def _check_signature(self, signature):
        if (
            signature.order != 2
            or signature.k.shape != (self.dim // 2,)
            or not np.array_equal(signature.pairing, self.pairing)
            or signature.g.shape != signature.k.shape
            or signature.pair_rms.shape != signature.k.shape
            or not all(
                np.isfinite(a).all()
                for a in [signature.k, signature.g, signature.pair_rms]
            )
            or np.any(signature.pair_rms < 0)
        ):
            raise ValueError("finite AR(2) signature in the network basis required")

    def activate_signature(self, signature):
        self._check_signature(signature)
        rows = []
        for name, count in self.memory._well_count.items():
            # Older artifacts can be inspected but must not masquerade as full nodes.
            if (
                name not in self.memory._well_sum_G
                or name not in self.memory._well_sum_S
                or any(
                    field.startswith("per-well G")
                    for field in getattr(self.memory, "import_info", {}).get(
                        "unavailable", []
                    )
                )
            ):
                raise ValueError(
                    "network routing needs K/G/RMS; imported memory lacks statistics"
                )
            k = self.memory._well_sum_K[name] / count
            g = self.memory._well_sum_G[name] / count
            s = self.memory._well_sum_S[name] / count
            distance = float(
                np.sqrt(
                    np.mean(
                        abs(signature.k - k) ** 2
                        + abs(signature.g - g) ** 2
                        + self.residual_weight * (signature.pair_rms - s) ** 2
                    )
                )
            )
            rows.append((name, distance))
        if not rows:
            return []
        rows.sort(key=lambda row: (row[1], row[0]))
        distances = np.array([d for _, d in rows])
        # Subtract the nearest squared distance before exponentiation.
        with np.errstate(over="ignore", invalid="ignore"):
            logits = (
                -((distances - distances[0]) / self.temperature)
                * ((distances + distances[0]) / self.temperature)
                / 2
            )
        logits[0] = 0
        weights = np.exp(logits)
        weights /= weights.sum()
        return [Activation(n, d, float(w)) for (n, d), w in zip(rows, weights)]

    def activate(self, trajectory):
        return self.activate_signature(self.signature(trajectory))

    def begin_sequence(self):
        """Break temporal edges between independent documents/episodes."""
        self.memory._last_well = None

    def learn_signature(self, signature, *, well=None):
        self._check_signature(signature)
        if well is not None and (
            not isinstance(well, str) or not well or len(well.encode("utf-8")) > 65535
        ):
            raise ValueError("well must be a nonempty UTF-8 name")
        activations = self.activate_signature(signature)
        if well is None:
            if activations and activations[0].distance <= self.novelty_radius:
                well = activations[0].well
            else:
                index = self.memory._next_well_id
                while f"well_{index}" in self.memory._well_count:
                    index += 1
                well = f"well_{index}"
                self.memory._next_well_id = index + 1
        self.memory.absorb_flat(
            signature.k[None], signature.g[None], signature.pair_rms[None], source=well
        )
        return well

    def learn(self, trajectory, *, well=None):
        return self.learn_signature(self.signature(trajectory), well=well)

    def transition_weights(self, well):
        if well not in self.memory._well_count:
            raise KeyError(well)
        edges = self.memory._well_transitions.get(well, {})
        total = sum(edges.values())
        return {n: count / total for n, count in sorted(edges.items())} if total else {}

    def oscillator(self, well):
        if well not in self.memory._well_count:
            raise KeyError(well)
        if well not in self.memory._well_sum_G or well not in self.memory._well_sum_S:
            raise ValueError("well lacks G/RMS; no full oscillator is available")
        if any(
            field.startswith("per-well G")
            for field in getattr(self.memory, "import_info", {}).get("unavailable", [])
        ):
            raise ValueError("imported well lacks measured G/RMS statistics")
        count = self.memory._well_count[well]
        return tuple(
            store[well].copy() / count
            for store in [
                self.memory._well_sum_K,
                self.memory._well_sum_G,
                self.memory._well_sum_S,
            ]
        )

    def describe(self):
        """Numeric physics/topology labels; no inferred semantic names."""
        result = []
        for name, count in sorted(self.memory._well_count.items()):
            k, g, s = self.oscillator(name)
            a, b = eigenvalues_all(k, g)
            dominant = np.where(abs(a) >= abs(b), a, b)
            result.append(
                {
                    "well": name,
                    "count": count,
                    "radius": float(np.max(abs(dominant))),
                    "phase_over_pi": float(np.median(abs(np.angle(dominant))) / np.pi),
                    "residual_rms": float(np.sqrt(np.mean(s * s))),
                    "transitions": self.transition_weights(name),
                }
            )
        return result

    def consolidate(self, radius):
        """Explicitly merge nearby nodes, preserving mass and remapping edges.

        Greedy, deterministic centroid linkage in the same K/G/RMS metric.
        Returns original -> final names. It changes the model: compare held-out
        performance before accepting a consolidated checkpoint.
        """
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("finite nonnegative merge radius required")
        m = self.memory
        mapping = {name: name for name in m._well_count}
        while True:
            names = sorted(m._well_count)
            candidate = None
            for i, a in enumerate(names):
                ka, ga, sa = self.oscillator(a)
                for b in names[i + 1 :]:
                    kb, gb, sb = self.oscillator(b)
                    distance = float(
                        np.sqrt(
                            np.mean(
                                abs(ka - kb) ** 2
                                + abs(ga - gb) ** 2
                                + self.residual_weight * (sa - sb) ** 2
                            )
                        )
                    )
                    row = (distance, a, b)
                    if distance <= radius and (candidate is None or row < candidate):
                        candidate = row
            if candidate is None:
                break
            _, a, b = candidate
            for store in [m._well_sum_K, m._well_sum_G, m._well_sum_S, m._well_sum_raw]:
                if b in store:
                    store[a] = store.get(a, np.zeros_like(store[b])) + store.pop(b)
            m._well_count[a] += m._well_count.pop(b)
            for store in [
                m._well_anchor_K,
                m._well_anchor_G,
                m._well_anchor_S,
                m._well_meta_K,
                m._well_meta_G,
            ]:
                store.pop(b, None)
                if store is m._well_meta_K or store is m._well_meta_G:
                    store.pop(a, None)
            # Relative chronological order is not recoverable from two per-well
            # buffers. Drop merged local history rather than fabricate an order.
            m._well_dream_Ks.pop(a, None)
            m._well_dream_Ks.pop(b, None)
            mapping = {
                original: a if final == b else final
                for original, final in mapping.items()
            }
            edges = {}
            for source, targets in m._well_transitions.items():
                source = a if source == b else source
                for target, count in targets.items():
                    target = a if target == b else target
                    row = edges.setdefault(source, {})
                    row[target] = row.get(target, 0) + count
            m._well_transitions = edges
            if getattr(m, "_last_well", None) == b:
                m._last_well = a
        m.annotations.pop("well_labels", None)
        m._invalidate_caches()
        return mapping

    def dream(
        self,
        context,
        *,
        well=None,
        steps=32,
        noise=0.0,
        rng=None,
        max_radius=None,
        walk_every=None,
        blend=None,
        blend_radius=1.0,
    ):
        """Continue from the last TWO observed positions, in original coordinates.

        Optional graph walking changes wells every walk_every samples, carrying
        state across switches. Noise has complex RMS = noise * learned pair RMS.
        Radius projection is explicit and reported; even unit-radius roots can
        have polynomial transients. Generated observations are never auto-learned.
        """
        from engram_wire import MAX_VALUES

        if blend is not None and (well is not None or walk_every is not None):
            raise ValueError(
                "a frozen Schur blend cannot also select/walk individual wells"
            )
        if blend is None and blend_radius != 1.0:
            raise ValueError("blend_radius requires an explicit frozen blend")

        if (
            not isinstance(steps, int)
            or steps < 1
            or steps * self.dim > MAX_VALUES
            or not math.isfinite(noise)
            or noise < 0
            or (
                max_radius is not None
                and (not math.isfinite(max_radius) or not 0 < max_radius <= 1)
            )
            or (
                walk_every is not None
                and (not isinstance(walk_every, int) or walk_every < 1)
            )
        ):
            raise ValueError("invalid dream length, noise, radius or walk interval")
        x, _ = _validated_trajectory(context, self.dim, self.pairing)
        if len(x) < 2:
            raise ValueError("dream needs two observed seed positions")
        if well is None and blend is None:
            activations = self.activate(x)
            if not activations:
                raise ValueError("train at least one well before dreaming")
            well = activations[0].well
        generator = np.random.default_rng() if rng is None else rng
        z = _to_complex(apply_pairing(x[-2:], self.pairing))
        older, previous = z
        out = np.empty((steps, self.dim // 2), complex)
        itinerary, largest, changed = [], 0.0, False
        for i in range(steps):
            if i == 0 or (walk_every is not None and i % walk_every == 0):
                if i:
                    edges = self.transition_weights(well)
                    if edges:
                        names = list(edges)
                        well = str(generator.choice(names, p=list(edges.values())))
                k, g, s = (
                    self.blend(blend, radius=blend_radius)
                    if blend is not None
                    else self.oscillator(well)
                )
                a, b = eigenvalues_all(k, g)
                radius = float(max(np.max(abs(a)), np.max(abs(b))))
                largest = max(largest, radius)
                if max_radius is not None and radius > max_radius:
                    a *= np.minimum(1, max_radius / np.maximum(abs(a), 1e-300))
                    b *= np.minimum(1, max_radius / np.maximum(abs(b), 1e-300))
                    k, g = a + b, a * b
                    changed = True
                if blend is None:
                    itinerary.append(well)
            with np.errstate(over="ignore", invalid="ignore"):
                value = k * previous - g * older
                if noise:
                    value += (
                        (
                            generator.normal(size=len(k))
                            + 1j * generator.normal(size=len(k))
                        )
                        * s
                        * noise
                        / np.sqrt(2)
                    )
            if (
                not np.isfinite(value).all()
                or np.max(abs(value)) > np.finfo(np.float32).max
            ):
                raise FloatingPointError("dream exceeded finite float32 range")
            out[i] = value
            older, previous = previous, value
        trajectory = undo_pairing(_from_complex(out), self.pairing).astype(np.float32)
        mixture = ()
        if blend is not None:
            names = sorted(name for name, weight in blend.items() if weight > 0)
            weights = np.asarray([blend[name] for name in names], float)
            weights /= weights.max()
            weights /= weights.sum()
            mixture = tuple(
                (name, float(weight)) for name, weight in zip(names, weights)
            )
        return Dream(
            trajectory,
            tuple(itinerary),
            largest,
            changed,
            mixture,
            float(blend_radius) if blend is not None else None,
        )

    def blend(self, weights, *, radius=1.0, margin=1e-8, dtype=np.complex128):
        """Explicit Schur generation view over named learned wells.

        Returns K/G and the weighted residual-power RMS. Observed coefficients,
        counts, transitions and routing remain unchanged. Strict interior wells
        only relative to the requested disk; radius>1 permits growing modes.
        Frozen root bounds do not authorize arbitrary graph switching.
        """
        from engram_stability import blend_schur

        if not isinstance(weights, dict) or not weights:
            raise ValueError("supply a nonempty well-name to weight mapping")
        names = sorted(weights)
        values = np.asarray([weights[n] for n in names], float)
        if (
            not np.isfinite(values).all()
            or np.any(values < 0)
            or not np.any(values > 0)
        ):
            raise ValueError("finite nonnegative nonzero blend weights required")
        active = values > 0
        rows = [self.oscillator(name) for name, used in zip(names, active) if used]
        values = values[active]
        k, g, s = (np.stack([row[i] for row in rows]) for i in range(3))
        output_k, output_g = blend_schur(
            k, g, values, radius=radius, margin=margin, dtype=dtype
        )
        values = values / values.max()
        values /= values.sum()
        return output_k, output_g, np.sqrt(values @ (s * s))

    def save(self, path):
        """Atomically replace a checkpoint; retain complete reference memory state."""
        self._store_config()
        path = Path(path)
        handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".engram")
        os.close(handle)
        try:
            self.memory.save(temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return str(path)

    @classmethod
    def load(cls, path):
        memory = EngramFile.load(path)
        config = memory.metadata.get("well_network", {})
        if (
            config.get("version") != 1
            or config.get("observation") != "whole-trajectory-ar2"
        ):
            raise ValueError("checkpoint is not a version-1 well network")
        return cls(
            memory.dim,
            pairing=memory._reference_pairing,
            memory=memory,
            **{
                k: config[k]
                for k in ["novelty_radius", "temperature", "residual_weight"]
            },
        )
