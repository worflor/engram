"""Environment-independent experience memory for observation/action NPCs.

Hosts supply witnesses, available action IDs and feedback. Engram supplies
incremental temporal geometry, growing experience wells, response values and
an observed transition model for offline planning. No game API or LLM is needed.
"""

from dataclasses import dataclass, field, asdict
import hashlib
import json
import math
from pathlib import Path
import os
import tempfile
import uuid
import numpy as np

from engram_history import WitnessHistory
from engram_codec import EngramFile, _real_array


@dataclass(frozen=True)
class WitnessSchema:
    channels: tuple[str, ...]
    scales: tuple[float, ...] = ()
    offsets: tuple[float, ...] = ()
    representation: str = "host-witness-v1"

    def __post_init__(self):
        object.__setattr__(self, "channels", tuple(self.channels))
        d = len(self.channels)
        if (
            not d
            or len(set(self.channels)) != d
            or not all(isinstance(c, str) and c for c in self.channels)
        ):
            raise ValueError("distinct named witness channels required")
        scales = tuple(self.scales) if self.scales else (1.0,) * d
        offsets = tuple(self.offsets) if self.offsets else (0.0,) * d
        if (
            len(scales) != d
            or len(offsets) != d
            or not np.isfinite(scales + offsets).all()
            or any(s <= 0 for s in scales)
            or not isinstance(self.representation, str)
            or not self.representation
        ):
            raise ValueError(
                "finite offsets, positive scales and representation identity required"
            )
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "offsets", offsets)

    @property
    def dim(self):
        return len(self.channels) + (len(self.channels) % 2)

    @classmethod
    def calibrate(cls, channels, samples, *, representation, minimum_scales=None):
        """Freeze training-stream means/population deviations as witness units.

        Uses O(D) storage. Supply training observations only, in a reproducible
        order; never recalibrate accumulated wells in place. Constant channels
        use one host unit unless a positive minimum scale is supplied.
        """
        schema = cls(channels, representation=representation)
        d = len(schema.channels)
        floors = np.zeros(d) if minimum_scales is None else _real_array(minimum_scales)
        if floors.shape != (d,) or not np.isfinite(floors).all() or np.any(floors < 0):
            raise ValueError("finite nonnegative per-channel minimum scales required")
        mean, scatter, count = np.zeros(d), np.zeros(d), 0
        with np.errstate(over="ignore", invalid="ignore"):
            for values in samples:
                row = schema.vector(values)[:d]
                count += 1
                delta = row - mean
                mean += delta / count
                scatter += delta * (row - mean)
        if count < 2:
            raise ValueError("calibration needs at least two training observations")
        scales = np.maximum(np.sqrt(np.maximum(scatter / count, 0)), floors)
        scales = np.where(scales == 0, 1.0, scales)
        return cls(schema.channels, tuple(scales), tuple(mean), representation)

    @property
    def identity(self):
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def vector(self, values):
        x = _real_array(values)
        if x.shape != (len(self.channels),) or not np.isfinite(x).all():
            raise ValueError("witness must match the finite named schema")
        with np.errstate(over="ignore", invalid="ignore"):
            x = (x - np.asarray(self.offsets)) / np.asarray(self.scales)
        if not np.isfinite(x).all() or np.max(abs(x)) > np.finfo(np.float32).max:
            raise ValueError("witness normalization exceeded finite range")
        # An odd-dimensional host has an explicit zero companion coordinate.
        return np.pad(x, (0, self.dim - len(x)))


@dataclass
class ActionExperience:
    value: float = 0.0
    visits: int = 0
    reward_sum: float = 0.0
    outcomes: dict = field(default_factory=dict)


@dataclass
class ExperienceWell:
    center: np.ndarray
    observations: int = 0
    actions: dict = field(default_factory=dict)
    available: set = field(default_factory=set)


@dataclass(frozen=True)
class Decision:
    serial: int
    action: str
    well: str | None
    value: float
    visits: int
    exploratory: bool
    owner: str


class DynamicsPrior:
    """Read-only dynamics seed in an explicitly declared input representation.

    A general seed supplies affinities, not invented action utilities. Its
    vectors must already use the original embedding/preprocessing coordinates.
    A host can feed these readings into its own named witness schema.
    """

    def __init__(self, memory, *, representation):
        if (
            not isinstance(memory, EngramFile)
            or not isinstance(representation, str)
            or not representation
        ):
            raise ValueError(
                "a memory and explicit representation identity are required"
            )
        if memory._reference_pairing is None:
            raise ValueError("a dynamics prior needs its original coordinate basis")
        self.memory = memory.copy()
        self.representation = representation
        self.history = WitnessHistory(memory.dim)

    @classmethod
    def load(cls, path, *, representation):
        return cls(EngramFile.load(path), representation=representation)

    def begin_episode(self):
        self.history = WitnessHistory(self.memory.dim)

    def observe(self, vector, *, representation):
        if representation != self.representation:
            raise ValueError("prior input representation differs")
        x = np.asarray(vector, float)
        if x.shape != (self.memory.dim,) or not np.isfinite(x).all():
            raise ValueError(
                "project into the prior's exact coordinates before observation"
            )
        self.history.push(x[self.memory._reference_pairing])
        signature = self.history.signature(center=False)
        if signature is None:
            return []
        result = []
        for name, count in self.memory._well_count.items():
            k = self.memory._well_sum_K[name] / count
            terms = abs(signature.k - k) ** 2
            fields = "K"
            unavailable = getattr(self.memory, "import_info", {}).get("unavailable", [])
            if name in self.memory._well_sum_G and not any(
                field.startswith("per-well G") for field in unavailable
            ):
                terms += abs(signature.g - self.memory._well_sum_G[name] / count) ** 2
                fields = "K/G"
            result.append(
                {
                    "well": name,
                    "distance": float(np.sqrt(np.mean(terms))),
                    "statistics": fields,
                    "recorded_mass": count,
                }
            )
        return sorted(result, key=lambda row: (row["distance"], row["well"]))


class EngramNPC:
    """A growing memory of experienced dynamics and action consequences.

    There is no built-in action vocabulary, goal, world size or well count.
    Feedback is host-defined. Planning only reuses observed transitions; it
    never turns a generated trajectory into fictitious ground-truth experience.
    """

    SECTION = "npc.json"

    def __init__(
        self,
        schema,
        *,
        novelty_radius=0.12,
        history_decay=1.0,
        history_decays=None,
        learning_rate=0.25,
        discount=0.95,
        trace_decay=0.8,
        exploration=0.4,
        epsilon=0.1,
        dynamics_weight=1.0,
        seed=0,
    ):
        if not isinstance(schema, WitnessSchema):
            raise ValueError("a WitnessSchema is required")
        parameters = locals().copy()
        parameters.pop("self")
        parameters.pop("schema")
        parameters.pop("history_decays")
        for name, value in parameters.items():
            if name == "seed":
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}")
        if (
            not 0 < history_decay <= 1
            or not 0 < learning_rate <= 1
            or not 0 <= discount < 1
            or not 0 <= trace_decay <= 1
            or not 0 <= epsilon <= 1
        ):
            raise ValueError("invalid decay, learning rate, discount or exploration")
        decays = (history_decay,) if history_decays is None else tuple(history_decays)
        if (
            not decays
            or len(set(decays)) != len(decays)
            or any(not math.isfinite(d) or not 0 < d <= 1 for d in decays)
        ):
            raise ValueError("distinct temporal decays in (0,1] required")
        if history_decay != 1.0 and history_decay != decays[0]:
            raise ValueError("history_decay must match the first temporal scale")
        parameters.pop("history_decay")
        self.schema, self.config = (
            schema,
            {k: float(v) for k, v in parameters.items() if k != "seed"},
        )
        self.config["history_decays"] = tuple(float(d) for d in decays)
        self.memory = EngramFile.create("", dim=schema.dim, name="experience")
        self.memory._reference_pairing = np.arange(schema.dim, dtype=np.int32)
        self.wells = {}
        self.histories = tuple(WitnessHistory(schema.dim, d) for d in decays)
        self.history = self.histories[0]
        self.rng = np.random.default_rng(seed)
        self._identity = uuid.uuid4().hex
        self.current_well = None
        self.current_vector = None
        self.pending = None
        self.serial = 0
        self.episodes = 0
        self.observations = 0
        self.updates = 0
        self.planning_updates = 0
        self.traces = {}

    def begin_episode(self):
        if self.pending is not None:
            raise ValueError(
                "finish the pending decision before beginning another episode"
            )
        self.histories = tuple(
            WitnessHistory(self.schema.dim, d) for d in self.config["history_decays"]
        )
        self.history = self.histories[0]
        self.traces.clear()
        self.current_well = None
        self.current_vector = None
        self.memory._last_well = None
        self.episodes += 1

    def observe(self, witness, *, learn=True):
        vector = self.schema.vector(witness)
        signatures = [history.push(vector) for history in self.histories]
        signature = signatures[0]
        self.current_vector = vector.copy()
        self.observations += 1
        if signature is None:
            self.current_well = None
            return None
        dynamic = np.concatenate(
            [
                part
                for fitted in signatures
                for part in [
                    fitted.k.real,
                    fitted.k.imag,
                    fitted.g.real,
                    fitted.g.imag,
                    np.log1p(fitted.pair_rms),
                ]
            ]
        )
        feature = np.concatenate([vector, self.config["dynamics_weight"] * dynamic])
        distances = [
            (float(np.sqrt(np.mean((well.center - feature) ** 2))), name)
            for name, well in self.wells.items()
        ]
        distance, name = min(distances) if distances else (math.inf, None)
        if learn and (name is None or distance > self.config["novelty_radius"]):
            name = f"experience_{len(self.wells)}"
            self.wells[name] = ExperienceWell(feature.copy())
        if learn:
            well = self.wells[name]
            well.observations += 1
            well.center += (feature - well.center) / well.observations
            self.memory.absorb_flat(
                signature.k[None],
                signature.g[None],
                signature.pair_rms[None],
                source=name,
            )
        self.current_well = name
        return name

    @staticmethod
    def _actions(actions):
        if isinstance(actions, (str, bytes)):
            raise ValueError("supply an iterable of action IDs, not one string")
        values = tuple(actions)
        if not values or not all(
            isinstance(a, str) and a and len(a.encode()) <= 65535 for a in values
        ):
            raise ValueError("nonempty distinct action IDs required")
        return tuple(sorted(set(values)))

    def decide(self, actions, *, explore=True):
        if self.current_vector is None:
            raise ValueError("observe before asking for a response")
        if self.pending is not None:
            raise ValueError("a decision is already awaiting feedback")
        names = self._actions(actions)
        well = self.wells.get(self.current_well)
        if well is not None and explore:
            well.available.update(names)
        stats = {
            name: well.actions.get(name, ActionExperience())
            if well
            else ActionExperience()
            for name in names
        }
        untried = [a for a in names if stats[a].visits == 0]
        exploratory = False
        if explore and (
            well is None or untried or self.rng.random() < self.config["epsilon"]
        ):
            candidates = untried or names
            action = str(self.rng.choice(candidates))
            exploratory = True
        else:
            total = sum(s.visits for s in stats.values())
            scores = {
                a: s.value
                + (
                    self.config["exploration"]
                    * math.sqrt(math.log(total + 1) / (s.visits + 1))
                    if explore
                    else 0.0
                )
                for a, s in stats.items()
            }
            best = max(scores.values())
            action = str(
                self.rng.choice([a for a in names if abs(scores[a] - best) <= 1e-12])
            )
        self.serial += 1
        decision = Decision(
            self.serial,
            action,
            self.current_well,
            stats[action].value,
            stats[action].visits,
            exploratory,
            self._identity,
        )
        self.pending = decision
        return decision

    def _maximum(self, well, actions=None):
        if well not in self.wells:
            return 0.0
        node = self.wells[well]
        return max(
            (
                node.actions.get(a, ActionExperience()).value
                for a in (actions or node.available)
            ),
            default=0.0,
        )

    def feedback(
        self,
        decision,
        next_witness,
        reward,
        *,
        terminal=False,
        next_actions=None,
        learn=True,
    ):
        if self.pending is None or decision != self.pending:
            raise ValueError("feedback must name the current unconsumed decision")
        if not math.isfinite(reward):
            raise ValueError("finite feedback required")
        if type(terminal) is not bool or type(learn) is not bool:
            raise ValueError("terminal and learn must be boolean")
        self.schema.vector(next_witness)  # validate before consuming the decision
        names = self._actions(next_actions) if next_actions is not None else None
        self.pending = None
        next_well = self.observe(next_witness, learn=learn)
        if learn and next_well is not None and names is not None:
            self.wells[next_well].available.update(names)
        if learn and decision.well is not None:
            node = self.wells[decision.well]
            stats = node.actions.setdefault(decision.action, ActionExperience())
            target = float(reward) + (
                0.0
                if terminal
                else self.config["discount"] * self._maximum(next_well, names)
            )
            error = target - stats.value
            key = (decision.well, decision.action)
            self.traces[key] = 1.0
            for (state, action), eligibility in list(self.traces.items()):
                old = self.wells[state].actions.setdefault(action, ActionExperience())
                old.value += self.config["learning_rate"] * error * eligibility
                self.traces[(state, action)] *= (
                    self.config["discount"] * self.config["trace_decay"]
                )
                if self.traces[(state, action)] == 0:
                    del self.traces[(state, action)]
            stats.visits += 1
            stats.reward_sum += float(reward)
            outcome = json.dumps([next_well, bool(terminal)], separators=(",", ":"))
            record = stats.outcomes.setdefault(outcome, {"count": 0, "reward_sum": 0.0})
            record["count"] += 1
            record["reward_sum"] += float(reward)
            self.updates += 1
        if terminal:
            self.traces.clear()
            self.memory._last_well = None
        return next_well

    def dream(self, updates=1):
        """Dyna-style value planning from observed state/action outcomes only.

        The host chooses work; no fixed maximum is imposed. Observed visit
        counts and real history remain unchanged by these planning updates.
        """
        if type(updates) is not int or updates < 0:
            raise ValueError("nonnegative planning update count required")
        candidates = [
            (name, action, stats)
            for name, well in self.wells.items()
            for action, stats in well.actions.items()
            if stats.outcomes
        ]
        if not candidates:
            return 0
        for _ in range(updates):
            name, action, stats = candidates[int(self.rng.integers(len(candidates)))]
            total = sum(row["count"] for row in stats.outcomes.values())
            target = 0.0
            for encoded, row in stats.outcomes.items():
                next_well, terminal = json.loads(encoded)
                target += (
                    row["reward_sum"]
                    + (
                        0.0
                        if terminal
                        else row["count"]
                        * self.config["discount"]
                        * self._maximum(next_well)
                    )
                ) / total
            stats.value += self.config["learning_rate"] * (target - stats.value)
            self.planning_updates += 1
        return updates

    def responses(self):
        if self.current_well is None:
            return {}
        return {
            name: {
                "value": stats.value,
                "visits": stats.visits,
                "mean_feedback": stats.reward_sum / stats.visits
                if stats.visits
                else None,
            }
            for name, stats in sorted(self.wells[self.current_well].actions.items())
        }

    def save(self, path):
        if self.pending is not None:
            raise ValueError(
                "checkpoint after feedback, not during an unacknowledged action"
            )
        payload = {
            "version": 2,
            "schema": asdict(self.schema),
            "schema_identity": self.schema.identity,
            "config": self.config,
            "histories": [h.snapshot() for h in self.histories],
            "rng": self.rng.bit_generator.state,
            "current_well": self.current_well,
            "current_vector": None
            if self.current_vector is None
            else self.current_vector.tolist(),
            "serial": self.serial,
            "episodes": self.episodes,
            "observations": self.observations,
            "updates": self.updates,
            "planning_updates": self.planning_updates,
            "traces": [
                [state, action, value] for (state, action), value in self.traces.items()
            ],
            "wells": {
                name: {
                    "center": well.center.tolist(),
                    "observations": well.observations,
                    "available": sorted(well.available),
                    "actions": {a: asdict(s) for a, s in well.actions.items()},
                }
                for name, well in self.wells.items()
            },
        }
        extras = getattr(self.memory, "_extra_sections", {}).copy()
        extras[self.SECTION] = json.dumps(
            payload, allow_nan=False, separators=(",", ":")
        ).encode()
        self.memory._extra_sections = extras
        path = Path(path)
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".engram")
        os.close(fd)
        try:
            self.memory.save(temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return str(path)

    @classmethod
    def load(cls, path, *, schema=None):
        memory = EngramFile.load(path)
        section = getattr(memory, "_extra_sections", {}).get(cls.SECTION)
        if section is None:
            raise ValueError(
                "memory has dynamics but no NPC experience; use an explicit compatible dynamics prior"
            )
        data = json.loads(section)
        saved_schema = WitnessSchema(**data["schema"])
        if (
            data["version"] not in (1, 2)
            or saved_schema.identity != data["schema_identity"]
            or (schema is not None and schema.identity != saved_schema.identity)
        ):
            raise ValueError("NPC schema or experience version differs")
        agent = cls(saved_schema, **data["config"])
        if memory.dim != saved_schema.dim or not np.array_equal(
            memory._reference_pairing, np.arange(memory.dim)
        ):
            raise ValueError("NPC memory has an incompatible coordinate basis")
        agent.memory = memory
        states = [data["history"]] if data["version"] == 1 else data["histories"]
        if len(states) != len(agent.histories):
            raise ValueError("temporal history bank differs from its declared scales")
        agent.histories = tuple(
            WitnessHistory.restore(memory.dim, decay, state)
            for decay, state in zip(agent.config["history_decays"], states)
        )
        agent.history = agent.histories[0]
        if len({h.samples for h in agent.histories}) != 1 or any(
            not all(
                np.array_equal(a, b) for a, b in zip(h.previous, agent.history.previous)
            )
            for h in agent.histories
        ):
            raise ValueError(
                "temporal histories describe different observation streams"
            )
        if set(data["wells"]) != set(memory._well_count):
            raise ValueError("NPC and dynamics well inventories differ")
        for name, row in data["wells"].items():
            center = np.asarray(row["center"], float)
            if (
                center.shape
                != (memory.dim + 5 * (memory.dim // 2) * len(agent.histories),)
                or not np.isfinite(center).all()
                or name not in memory._well_count
                or row["observations"] != memory._well_count[name]
            ):
                raise ValueError("invalid experience well geometry")
            actions = {
                action: ActionExperience(**stats)
                for action, stats in row["actions"].items()
            }
            for action, stats in actions.items():
                cls._actions([action])
                if (
                    type(stats.visits) is not int
                    or stats.visits < 0
                    or not np.isfinite([stats.value, stats.reward_sum]).all()
                ):
                    raise ValueError("invalid action experience")
                for encoded, outcome in stats.outcomes.items():
                    target, terminal = json.loads(encoded)
                    if (
                        (target is not None and target not in data["wells"])
                        or type(terminal) is not bool
                        or type(outcome["count"]) is not int
                        or outcome["count"] <= 0
                        or not math.isfinite(outcome["reward_sum"])
                    ):
                        raise ValueError("invalid experience transition")
                if sum(o["count"] for o in stats.outcomes.values()) != stats.visits:
                    raise ValueError(
                        "transition counts differ from observed action visits"
                    )
            available = row["available"]
            if available:
                cls._actions(available)
            agent.wells[name] = ExperienceWell(
                center, row["observations"], actions, set(available)
            )
        for key in [
            "serial",
            "episodes",
            "observations",
            "updates",
            "planning_updates",
        ]:
            value = data[key]
            if type(value) is not int or value < 0:
                raise ValueError("invalid NPC counters")
            setattr(agent, key, value)
        if agent.updates != sum(
            s.visits for w in agent.wells.values() for s in w.actions.values()
        ):
            raise ValueError("NPC update count differs from observed action visits")
        agent.current_well = data["current_well"]
        if agent.current_well is not None and agent.current_well not in agent.wells:
            raise ValueError("unknown current experience well")
        agent.current_vector = (
            None
            if data["current_vector"] is None
            else np.asarray(data["current_vector"], float)
        )
        if agent.current_vector is not None and (
            agent.current_vector.shape != (memory.dim,)
            or not np.isfinite(agent.current_vector).all()
        ):
            raise ValueError("invalid current witness")
        agent.traces = {(s, a): float(v) for s, a, v in data["traces"]}
        if any(
            s not in agent.wells
            or a not in agent.wells[s].actions
            or not math.isfinite(v)
            or v < 0
            for (s, a), v in agent.traces.items()
        ):
            raise ValueError("invalid eligibility history")
        agent.rng.bit_generator.state = data["rng"]
        return agent
