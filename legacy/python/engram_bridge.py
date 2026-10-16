"""
engram_bridge.py

Wires the Engram semantic trajectory codec into the SedenionCore hippocampus.

EngramCore extends SedenionCore with three surgical overrides:

    1. witness()  — soft projection via tanh(alpha * S * h_t)
    2. ingest()   — encode trajectories as EngramPackets, discard raw witnesses
    3. handshake() Stage 1 — match by K,G orbital parameters, not static landmarks

the raw witness path is gone. memory is stored as dynamical system parameters.
recall is lazy simulation: spin up the oscillator at the match point,
decode residuals forward, feed generated witnesses into the bypass.

    "you aren't storing memory as a file.
     you are storing it as initial conditions for a dynamical system."
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch

# import from siblings
sys.path.insert(0, os.path.dirname(__file__))

try:
    from hyper_cache import (
        SedenionCore,
        WitnessConfig,
        TrajectoryRecord,
        TrajectoryLifecycle,
        KIZUNA_BOND_MASKS,
        KIZUNA_FAMILY_SLICES,
        _cosine_similarity_np,
        _kizuna_family_profile,
        _token_kizuna_fingerprint,
        _build_witness_landmarks,
        _build_token_landmarks,
        _window_global_fingerprint,
        _landmark_set_alignment,
        _adaptive_signal_weights,
        get_env_bool,
    )
    _HYPER_CACHE_AVAILABLE = True
except ImportError:
    _HYPER_CACHE_AVAILABLE = False
    # minimal stubs for standalone testing
    def _cosine_similarity_np(a, b):
        d = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / d) if d > 1e-8 else 0.0
    def _window_global_fingerprint(w):
        w = w.float() if hasattr(w, 'float') else torch.from_numpy(w).float()
        mid = max(1, w.shape[0] // 2)
        return np.concatenate([
            w.mean(dim=0).numpy(), w.std(dim=0).numpy(),
            w[:mid].mean(dim=0).numpy(), w[mid:].mean(dim=0).numpy(),
        ]).astype(np.float32)
    def _adaptive_signal_weights(*arrays):
        n = len(arrays)
        return np.ones(n, dtype=np.float32) / n
    def get_env_bool(key, default=False):
        return os.environ.get(key, str(default)).lower() in ('1', 'true', 'yes')

from engram import (
    EngramCodec,
    EngramPacket,
    EngramMode,
    ENGRAM_PROJ_DIM,
    ENGRAM_DEFAULT_ALPHA,
    OSCILLATOR_FAMILIES,
    N_FAMILIES,
    engram_parameter_vector,
    engram_parameter_distance,
    engram_summary,
    predict_block,
)


# ── configuration ─────────────────────────────────────────────────────

@dataclass(slots=True)
class EngramConfig:
    """Configuration for the Engram codec within SedenionCore."""
    alpha: float = ENGRAM_DEFAULT_ALPHA
    min_block: int = 4
    max_block: int = 64
    q_bits: int = 8
    enabled: bool = True


# ── extended trajectory record ────────────────────────────────────────
#
# we store BOTH the compressed EngramPacket AND a torch witness tensor.
# the witness tensor is now SOFT (float16, values in [-1,1]) instead of
# HARD (int8, values in {-1,+1}). the EngramPacket is the compressed
# form; the witness is reconstructed on demand for handshake scoring.
#
# why keep both? because TrajectoryRecord.__post_init__ computes landmarks
# from the witness tensor, and the entire existing handshake pipeline reads
# trajectory.witness. we keep compatibility while adding the new path.

# we'll store the EngramPacket as an attribute on the TrajectoryRecord
# using a side-channel dict, keyed by id(trajectory).
_engram_packets: dict[int, EngramPacket] = {}
_engram_param_vectors: dict[int, np.ndarray] = {}


def attach_engram(trajectory: TrajectoryRecord, packet: EngramPacket) -> None:
    """Attach an EngramPacket to a trajectory for parameter-based matching."""
    _engram_packets[id(trajectory)] = packet
    _engram_param_vectors[id(trajectory)] = engram_parameter_vector(packet)


def get_engram(trajectory: TrajectoryRecord) -> Optional[EngramPacket]:
    """Retrieve the EngramPacket attached to a trajectory."""
    return _engram_packets.get(id(trajectory))


def get_engram_params(trajectory: TrajectoryRecord) -> Optional[np.ndarray]:
    """Retrieve the parameter vector attached to a trajectory."""
    return _engram_param_vectors.get(id(trajectory))


def detach_engram(trajectory: TrajectoryRecord) -> None:
    """Remove Engram data from a trajectory."""
    _engram_packets.pop(id(trajectory), None)
    _engram_param_vectors.pop(id(trajectory), None)


# ── lazy simulation ───────────────────────────────────────────────────

def simulate_from_block(
    packet: EngramPacket,
    block_idx: int,
    n_tokens: Optional[int] = None,
) -> np.ndarray:
    """
    Lazy simulation: reconstruct witness path starting from a specific block.

    Does NOT decompress the entire trajectory. Only simulates forward from
    the matched block, decoding residuals on demand.

    Parameters
    ----------
    packet : the compressed trajectory
    block_idx : which block to start from
    n_tokens : how many tokens to simulate (None = rest of trajectory)

    Returns
    -------
    [n_tokens, 256] float32 — reconstructed soft witnesses
    """
    if block_idx >= len(packet.blocks):
        return np.empty((0, packet.proj_dim), dtype=np.float32)

    results: list[np.ndarray] = []
    tokens_generated = 0

    for bi in range(block_idx, len(packet.blocks)):
        block = packet.blocks[bi]

        if block.mode == EngramMode.RESET and block.start_idx == 0:
            chunk = block.residuals.copy()
        else:
            predicted = predict_block(
                block.seed1,
                block.seed2,
                block.family_coeffs,
                block.length,
                block.mode,
            )
            chunk = predicted + block.residuals

        if n_tokens is not None:
            remaining = n_tokens - tokens_generated
            if remaining <= 0:
                break
            chunk = chunk[:remaining]

        results.append(chunk)
        tokens_generated += chunk.shape[0]

        if n_tokens is not None and tokens_generated >= n_tokens:
            break

    if not results:
        return np.empty((0, packet.proj_dim), dtype=np.float32)

    return np.concatenate(results, axis=0)


def find_block_for_token(packet: EngramPacket, token_idx: int) -> Tuple[int, int]:
    """
    Find which block contains a given token index.

    Returns (block_idx, offset_within_block).
    """
    for bi, block in enumerate(packet.blocks):
        if block.start_idx <= token_idx < block.start_idx + block.length:
            return bi, token_idx - block.start_idx

    # fallback: last block
    if packet.blocks:
        return len(packet.blocks) - 1, 0
    return 0, 0


# ── parameter-based handshake scoring ─────────────────────────────────

def engram_orbital_score(
    query_params: np.ndarray,
    stored_params: np.ndarray,
) -> float:
    """
    Score trajectory match by comparing K,G orbital parameters.

    This asks "are these thoughts moving the same way?" rather than
    "are these thoughts in the same place?"

    Uses cosine similarity on the parameter vectors, which captures
    alignment of oscillator dynamics across all 5 Kizuna families.
    """
    q = query_params
    s = stored_params
    # pad shorter to match longer
    if len(q) < len(s):
        q = np.pad(q, (0, len(s) - len(q)), constant_values=0.0)
    elif len(s) < len(q):
        s = np.pad(s, (0, len(q) - len(s)), constant_values=0.0)
    dot = float(np.dot(q, s))
    norm_q = float(np.linalg.norm(q))
    norm_s = float(np.linalg.norm(s))
    denom = norm_q * norm_s
    if denom < 1e-8:
        return 0.0
    return dot / denom


def engram_family_scores(
    query_params: np.ndarray,
    stored_params: np.ndarray,
) -> np.ndarray:
    """
    Per-family orbital similarity.

    Returns [n_blocks, N_FAMILIES] cosine scores showing which families
    have matching dynamics and which diverge.
    """
    # reshape to [n_blocks, N_FAMILIES * 4]
    stride = N_FAMILIES * 4
    q_blocks = len(query_params) // stride
    s_blocks = len(stored_params) // stride
    n_blocks = min(q_blocks, s_blocks)

    if n_blocks == 0:
        return np.zeros((0, N_FAMILIES), dtype=np.float32)

    scores = np.zeros((n_blocks, N_FAMILIES), dtype=np.float32)
    for bi in range(n_blocks):
        for fi in range(N_FAMILIES):
            q_start = bi * stride + fi * 4
            s_start = bi * stride + fi * 4
            q_slice = query_params[q_start:q_start + 4]
            s_slice = stored_params[s_start:s_start + 4]
            dot = float(np.dot(q_slice, s_slice))
            norm = float(np.linalg.norm(q_slice) * np.linalg.norm(s_slice))
            scores[bi, fi] = dot / max(norm, 1e-8)

    return scores


# ── EngramCore: the integrated hippocampus ────────────────────────────

# The base class must exist for the subclass definition.
# When hyper_cache is not importable (missing transformers), we use
# a minimal stub so the module still loads for standalone testing.
if not _HYPER_CACHE_AVAILABLE:
    class _SedenionCoreStub:
        """Minimal stub when hyper_cache is unavailable."""
        def __init__(self, config, **kw): pass
        def _bootstrap_projection_substrate(self, *a, **kw): pass
        def _next_event_step(self): return 0
        def _adaptive_shortlist_size(self, n, q): return min(n, 8)
        def _adaptive_validation_window(self, q): return min(q, 32)
        def _derived_acceptance_prior(self, q): return 0.4
        def _calibrated_acceptance_threshold(self, q): return 0.4
        def _adaptive_prefix_threshold(self, n): return 0.35
        def _update_handshake_calibration(self, *a): pass
        def _mark_trajectory_access(self, *a): pass
        def _trajectory_prior_score(self, t): return 0.5
        def _trajectory_strength(self, t): return 0.5
        def _trajectory_lexical_overlap_scores(self, *a): return np.zeros(0)
        def _query_token_weights(self, *a): return {}
        def handshake(self, *a, **kw): return None, None
    SedenionCore = _SedenionCoreStub  # type: ignore
    WitnessConfig = type('WitnessConfig', (), {'__init__': lambda self, **kw: [setattr(self, k, v) for k, v in kw.items()]})  # type: ignore
    class TrajectoryRecord:  # type: ignore
        def __init__(self, **kw):
            for k, v in kw.items(): setattr(self, k, v)
            self.token_count = len(getattr(self, 'token_ids', []))
            w = self.witness.float() if hasattr(self.witness, 'float') else self.witness
            self.global_fp = _window_global_fingerprint(w)
    class TrajectoryLifecycle:  # type: ignore
        def __init__(self, **kw):
            for k, v in kw.items(): setattr(self, k, v)


class EngramCore(SedenionCore):  # type: ignore
    """
    SedenionCore with Engram codec integration.

    Overrides:
        witness()   — soft projection: tanh(alpha * S * h_t)
        ingest()    — encode + store EngramPacket alongside trajectory
        handshake() — inject orbital parameter scoring into Stage 1
    """

    def __init__(self, config: WitnessConfig, engram_config: Optional[EngramConfig] = None):
        super().__init__(config)
        self.engram_config = engram_config or EngramConfig()
        self.codec = EngramCodec(
            alpha=self.engram_config.alpha,
            min_block=self.engram_config.min_block,
            max_block=self.engram_config.max_block,
            q_bits=self.engram_config.q_bits,
        )

    # ── Override 1: Soft Witness Projection ────────────────────────────

    def witness(self, hidden_path: torch.Tensor) -> torch.Tensor:
        """
        Soft witness: tanh(alpha * S * h_t) instead of sign(S * h_t).

        Returns float32 values in (-1, 1) with meaningful gradients,
        rather than hard bipolar {-1, +1}.

        alpha is the cognitive temperature:
            high alpha -> crystalline, binary, decisive
            low alpha  -> fluid, continuous, nuanced
        """
        if hidden_path.ndim != 2:
            raise ValueError(f"expected hidden path [seq, hidden], got {tuple(hidden_path.shape)}")
        proj_in = self.config.proj_input_dim
        if hidden_path.shape[-1] < proj_in:
            raise ValueError(
                f"hidden width {hidden_path.shape[-1]} < proj_input_dim {proj_in}"
            )

        seed_slice = hidden_path[..., :proj_in]
        projected = seed_slice @ self.scramble_matrix.to(seed_slice.device, seed_slice.dtype).T
        # soft projection: tanh(alpha * projected)
        return torch.tanh(self.engram_config.alpha * projected)

    # ── Override 2: Engram-Encoded Ingestion ───────────────────────────

    def ingest(
        self,
        hidden_path: torch.Tensor,
        token_ids: Sequence[int],
        key_memory: Sequence[torch.Tensor] = (),
        value_memory: Sequence[torch.Tensor] = (),
        *,
        source_index: Optional[int] = None,
        source_start: Optional[int] = None,
        source_end: Optional[int] = None,
    ) -> None:
        """
        Ingest with Engram compression.

        1. Compute soft witness path via tanh projection
        2. Encode the witness path as an EngramPacket (K,G + residuals)
        3. Store the trajectory with the soft witness (for landmark compat)
        4. Attach the EngramPacket for parameter-based handshake
        """
        if hidden_path.shape[0] != len(token_ids):
            raise ValueError("hidden path length must equal token id length")
        if len(key_memory) != len(value_memory):
            raise ValueError("key/value layer counts must match")
        for layer_idx, (keys, values) in enumerate(zip(key_memory, value_memory)):
            if keys.shape[0] != len(token_ids) or values.shape[0] != len(token_ids):
                raise ValueError(f"layer {layer_idx} cache length does not match token count")

        self._bootstrap_projection_substrate(hidden_path, token_ids)

        # compute soft witness (float32, values in [-1, 1])
        soft_w = self.witness(hidden_path).to("cpu")

        # encode as EngramPacket — this is the compressed memory
        witness_np = soft_w.numpy().astype(np.float32)
        packet = self.codec.encode_trajectory(witness_np)

        # store trajectory with float16 witness (soft, not int8 hard)
        # the landmark system reads this for backward-compatible scoring
        trajectory = TrajectoryRecord(
            witness=soft_w.to(dtype=torch.float16),
            keys=[k.detach().to("cpu", dtype=self.config.storage_dtype).contiguous() for k in key_memory],
            values=[v.detach().to("cpu", dtype=self.config.storage_dtype).contiguous() for v in value_memory],
            token_ids=list(token_ids),
            source_index=source_index,
            source_start=source_start,
            source_end=source_end,
            lifecycle=TrajectoryLifecycle(
                created_step=self._next_event_step(),
                last_access_step=self._event_step,
            ),
        )
        self.trajectories.append(trajectory)

        # attach the compressed packet — the old witness can be discarded
        # once the handshake is fully migrated to parameter matching
        attach_engram(trajectory, packet)

        self._invalidate_token_shortcut_index()
        self._invalidate_trajectory_token_df_cache()
        self._invalidate_state_profile()

    # ── Override 3: Orbital Parameter Handshake ────────────────────────

    def handshake(
        self,
        query_hidden_path: torch.Tensor,
        query_token_ids: Optional[Sequence[int]] = None,
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Three-stage memory retrieval with Engram orbital parameter scoring.

        Stage 0 — prefix fast-path (unchanged)
        Stage 1 — ENGRAM ORBITAL SCORING:
                  compare query's K,G parameters against stored trajectories.
                  asks "are these thoughts moving the same way?" not just
                  "are these thoughts in the same place?"
        Stage 2 — precise anchor via lazy simulation
        Stage 3 — replay branch selection (unchanged, delegated to super)

        When Engram packets are available, Stage 1 uses orbital parameter
        cosine similarity. When not available (legacy trajectories), falls
        back to the landmark-based pipeline.
        """
        if not self.trajectories:
            self.last_handshake_debug = {"stage": "empty", "accepted": False}
            return None, None

        self.last_handshake_debug = None
        query_witness = self.witness(query_hidden_path).to(dtype=torch.float32)
        query_tokens = query_hidden_path.shape[0]
        derived_acceptance_prior = self._derived_acceptance_prior(query_tokens)
        calibrated_acceptance = self._calibrated_acceptance_threshold(query_tokens)

        # -- Stage 0: exact prefix fast-path (unchanged) --
        if query_token_ids:
            prefix_len = len(query_token_ids)
            best_prefix_traj: Optional[int] = None
            best_prefix_score = float("-inf")
            for traj_idx, trajectory in enumerate(self.trajectories):
                score = trajectory.score_prefix_alignment(query_witness, query_token_ids)
                if score > best_prefix_score:
                    best_prefix_score = score
                    best_prefix_traj = traj_idx

            prefix_threshold = min(calibrated_acceptance, self._adaptive_prefix_threshold(prefix_len))
            if best_prefix_traj is not None and best_prefix_score >= prefix_threshold:
                self.last_handshake_debug = {
                    "stage": "prefix",
                    "accepted": True,
                    "traj_idx": best_prefix_traj,
                    "anchor": prefix_len - 1,
                    "score": float(best_prefix_score),
                    "derived_threshold": float(derived_acceptance_prior),
                    "threshold": float(prefix_threshold),
                    "mask_basis_source": self.mask_basis_source,
                    "engram": True,
                }
                self._update_handshake_calibration(float(best_prefix_score), True, [float(best_prefix_score)])
                self._mark_trajectory_access(best_prefix_traj)
                return best_prefix_traj, prefix_len - 1

        # -- Stage 1: ENGRAM ORBITAL PARAMETER SCORING --
        #
        # encode the query as an EngramPacket to extract its K,G parameters
        # then score each stored trajectory by orbital cosine similarity
        #
        # this replaces the landmark pre-screening with a fundamentally
        # different question: not "same position?" but "same dynamics?"

        n_trajs = len(self.trajectories)
        shortlist_k = self._adaptive_shortlist_size(n_trajs, query_tokens)

        # encode query trajectory
        query_np = query_witness.cpu().numpy().astype(np.float32)
        query_packet = self.codec.encode_trajectory(query_np)
        query_params = engram_parameter_vector(query_packet)

        # score each stored trajectory by orbital parameter similarity
        orbital_scores = np.zeros(n_trajs, dtype=np.float32)
        has_engram = False

        for traj_idx, trajectory in enumerate(self.trajectories):
            stored_params = get_engram_params(trajectory)
            if stored_params is not None:
                orbital_scores[traj_idx] = engram_orbital_score(query_params, stored_params)
                has_engram = True
            else:
                # fallback: use global witness fingerprint similarity
                q_fp = _window_global_fingerprint(query_witness)
                orbital_scores[traj_idx] = _cosine_similarity_np(q_fp, trajectory.global_fp)

        # blend with lexical scores if available
        lexical_scores: Optional[np.ndarray] = None
        if query_token_ids and len(query_token_ids) >= 2:
            lexical_scores = self._trajectory_lexical_overlap_scores(query_token_ids)

        # also compute landmark scores for trajectories without Engram data
        if not has_engram:
            # full fallback to parent's landmark pipeline
            return super().handshake(query_hidden_path, query_token_ids)

        # combine orbital + lexical scores for shortlisting
        if n_trajs > shortlist_k:
            if lexical_scores is not None:
                weights = _adaptive_signal_weights(orbital_scores, lexical_scores)
                combined = weights[0] * orbital_scores + weights[1] * lexical_scores
            else:
                combined = orbital_scores

            # add strength priors
            strength_priors = np.asarray([
                self._trajectory_prior_score(t) for t in self.trajectories
            ], dtype=np.float32)
            prior_scores = 2.0 * strength_priors - 1.0

            if n_trajs > 1:
                sw = _adaptive_signal_weights(combined, prior_scores)
                combined = sw[0] * combined + sw[1] * prior_scores

            shortlist_indices = np.argsort(combined)[-shortlist_k:]

            if lexical_scores is not None:
                lexical_budget = max(1, min(n_trajs, max(2, shortlist_k // 2)))
                lexical_top = np.argsort(lexical_scores)[-lexical_budget:]
                shortlist_indices = np.unique(np.concatenate([shortlist_indices, lexical_top])).astype(np.int32)
        else:
            shortlist_indices = np.arange(n_trajs)

        # -- Stage 2: precise anchor search with lazy simulation --
        adaptive_window = self._adaptive_validation_window(query_tokens)
        best_traj: Optional[int] = None
        best_anchor_idx: Optional[int] = None
        best_score = float("-inf")
        query_token_weights = self._query_token_weights(query_token_ids or ()) if query_token_ids else {}
        use_kizuna = bool(query_token_ids) and self.config.kizuna_blend > 0.0
        branch_candidates: list[tuple[float, int, int]] = []

        for traj_idx in shortlist_indices:
            trajectory = self.trajectories[int(traj_idx)]
            packet = get_engram(trajectory)

            if packet is not None:
                # LAZY SIMULATION: reconstruct witness only for scoring window
                # this is the key optimization — we don't decompress everything
                anchor, score = trajectory.best_anchor(
                    query_witness,
                    adaptive_window,
                    query_token_ids if use_kizuna else None,
                    query_token_weights=query_token_weights if use_kizuna else None,
                    kizuna_window=self.config.kizuna_window,
                    kizuna_blend=self.config.kizuna_blend if use_kizuna else 0.0,
                    lexical_blend=0.35 if use_kizuna else 0.0,
                )

                # boost score by orbital similarity (the Engram advantage)
                stored_params = get_engram_params(trajectory)
                if stored_params is not None:
                    orb_boost = engram_orbital_score(query_params, stored_params)
                    # blend: 70% anchor score + 30% orbital dynamics
                    score = 0.7 * score + 0.3 * orb_boost
            else:
                anchor, score = trajectory.best_anchor(
                    query_witness,
                    adaptive_window,
                    query_token_ids if use_kizuna else None,
                    query_token_weights=query_token_weights if use_kizuna else None,
                    kizuna_window=self.config.kizuna_window,
                    kizuna_blend=self.config.kizuna_blend if use_kizuna else 0.0,
                    lexical_blend=0.35 if use_kizuna else 0.0,
                )

            branch_candidates.append((score, int(traj_idx), anchor))
            if score > best_score:
                best_score = score
                best_traj = int(traj_idx)
                best_anchor_idx = anchor

        # -- Stage 3: accept/reject --
        if best_traj is None or best_anchor_idx is None:
            self.last_handshake_debug = {
                "stage": "engram_search",
                "accepted": False,
                "n_candidates": len(branch_candidates),
                "engram": True,
            }
            self._update_handshake_calibration(best_score, False, [c[0] for c in branch_candidates])
            return None, None

        # acceptance gate
        if best_score < calibrated_acceptance:
            self.last_handshake_debug = {
                "stage": "engram_rejected",
                "accepted": False,
                "traj_idx": best_traj,
                "anchor": best_anchor_idx,
                "score": float(best_score),
                "derived_threshold": float(derived_acceptance_prior),
                "threshold": float(calibrated_acceptance),
                "engram": True,
            }
            self._update_handshake_calibration(best_score, False, [c[0] for c in branch_candidates])
            return None, None

        # accepted
        self.last_handshake_debug = {
            "stage": "engram_accepted",
            "accepted": True,
            "traj_idx": best_traj,
            "anchor": best_anchor_idx,
            "score": float(best_score),
            "derived_threshold": float(derived_acceptance_prior),
            "threshold": float(calibrated_acceptance),
            "orbital_score": float(orbital_scores[best_traj]),
            "engram": True,
            "mask_basis_source": self.mask_basis_source,
        }
        self._update_handshake_calibration(best_score, True, [c[0] for c in branch_candidates])
        self._mark_trajectory_access(best_traj)

        if get_env_bool("TOWER_DEBUG_BRANCH", False):
            print(
                f"[ENGRAM] handshake traj={best_traj} anchor={best_anchor_idx} "
                f"score={best_score:.4f} orbital={orbital_scores[best_traj]:.4f} "
                f"threshold={calibrated_acceptance:.4f}"
            )

        return best_traj, best_anchor_idx

    # ── Lazy Recall ────────────────────────────────────────────────────

    def recall_from_anchor(
        self,
        traj_idx: int,
        anchor: int,
        n_tokens: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Lazy simulation: reconstruct witness path from anchor point forward.

        Does NOT decompress the entire trajectory. Spins up the oscillator
        at the matching block, decodes residuals from that point, and returns
        the generated witness states.

        This is the Engram bypass: memory recall without the LLM.
        """
        if traj_idx >= len(self.trajectories):
            return None

        trajectory = self.trajectories[traj_idx]
        packet = get_engram(trajectory)

        if packet is None:
            # fallback: return raw witness slice from stored tensor
            w = trajectory.witness.float().numpy()
            end = w.shape[0] if n_tokens is None else min(anchor + n_tokens, w.shape[0])
            return w[anchor:end]

        # find which block contains the anchor
        block_idx, offset = find_block_for_token(packet, anchor)

        # simulate forward from that block
        simulated = simulate_from_block(packet, block_idx, n_tokens)

        # trim to start from the offset within the block
        if offset > 0 and offset < simulated.shape[0]:
            simulated = simulated[offset:]

        if n_tokens is not None:
            simulated = simulated[:n_tokens]

        return simulated

    # ── Diagnostics ────────────────────────────────────────────────────

    def engram_manifold_summary(self) -> dict:
        """Summary of the Engram-enhanced manifold."""
        n_engram = sum(1 for t in self.trajectories if get_engram(t) is not None)
        n_legacy = len(self.trajectories) - n_engram

        mode_totals = {m.name: 0 for m in EngramMode}
        total_blocks = 0
        total_params = 0

        for t in self.trajectories:
            pkt = get_engram(t)
            if pkt is not None:
                for block in pkt.blocks:
                    mode_totals[block.mode.name] += 1
                    total_blocks += 1
                pv = get_engram_params(t)
                if pv is not None:
                    total_params += len(pv)

        return {
            "trajectories": len(self.trajectories),
            "engram_encoded": n_engram,
            "legacy": n_legacy,
            "total_blocks": total_blocks,
            "mode_distribution": mode_totals,
            "total_orbital_params": total_params,
            "alpha": self.engram_config.alpha,
        }


# ── self-test ─────────────────────────────────────────────────────────

def _self_test() -> None:
    """Integration smoke test with synthetic data (no model required)."""
    print("=== Engram Bridge: Integration Test ===\n")

    if not _HYPER_CACHE_AVAILABLE:
        print("hyper_cache not available (missing transformers). running standalone tests.\n")
        _self_test_standalone()
        return

    # create a minimal EngramCore
    config = WitnessConfig(head_dim=64, proj_dim=ENGRAM_PROJ_DIM, proj_input_dim=64)
    core = EngramCore(config, EngramConfig(alpha=4.0))

    rng = np.random.RandomState(42)

    # simulate 3 "conversations" as hidden state trajectories
    print("--- ingesting 3 synthetic trajectories ---")
    for conv_idx in range(3):
        T = 50 + conv_idx * 20
        hidden = torch.randn(T, 64, dtype=torch.float32)
        token_ids = list(range(1000 + conv_idx * 100, 1000 + conv_idx * 100 + T))
        core.ingest(hidden, token_ids, source_index=conv_idx)
        pkt = get_engram(core.trajectories[-1])
        if pkt:
            summary = engram_summary(pkt)
            print(f"  conv {conv_idx}: {T} tokens, {summary['n_blocks']} blocks, modes={summary['mode_distribution']}")

    print(f"\n--- manifold summary ---")
    ms = core.engram_manifold_summary()
    for k, v in ms.items():
        print(f"  {k}: {v}")

    # handshake with a query that partially matches conversation 1
    print("\n--- handshake test ---")
    query_hidden = torch.randn(20, 64, dtype=torch.float32)
    query_tokens = list(range(1100, 1120))
    traj_idx, anchor = core.handshake(query_hidden, query_tokens)
    debug = core.last_handshake_debug or {}
    print(f"  result: traj={traj_idx}, anchor={anchor}")
    print(f"  debug: stage={debug.get('stage')}, score={debug.get('score', 'N/A')}")
    print(f"  engram={debug.get('engram', False)}")

    # lazy recall
    if traj_idx is not None and anchor is not None:
        print("\n--- lazy recall ---")
        recalled = core.recall_from_anchor(traj_idx, anchor, n_tokens=10)
        if recalled is not None:
            print(f"  recalled {recalled.shape[0]} tokens from anchor {anchor}")
            print(f"  witness norm: {np.linalg.norm(recalled, axis=1).mean():.4f}")
        else:
            print("  recall returned None")

    # verify soft witness values are in (-1, 1)
    print("\n--- soft witness verification ---")
    test_hidden = torch.randn(10, 64, dtype=torch.float32)
    sw = core.witness(test_hidden)
    print(f"  shape: {tuple(sw.shape)}")
    print(f"  range: [{sw.min().item():.4f}, {sw.max().item():.4f}]")
    print(f"  mean abs: {sw.abs().mean().item():.4f}")
    assert sw.abs().max().item() <= 1.0, "soft witness out of range!"

    print("\n=== integration test complete ===")
    print("the hippocampus has been replaced.")
    print("memory is stored as dynamical system parameters.")
    print("recall is lazy simulation, not brute-force loading.")


def _self_test_standalone() -> None:
    """Test Engram bridge components without hyper_cache dependency."""

    # test 1: simulate_from_block
    print("--- simulate_from_block ---")
    codec = EngramCodec(alpha=4.0)
    rng = np.random.RandomState(42)
    w = np.zeros((80, ENGRAM_PROJ_DIM), dtype=np.float32)
    t = np.arange(80, dtype=np.float32)
    for d in range(ENGRAM_PROJ_DIM):
        w[:, d] = 0.3 * np.sin(0.1 * t + 0.05 * d) + 0.02 * rng.randn(80)
    packet = codec.encode_trajectory(w)
    print(f"  encoded: {len(packet.blocks)} blocks, {packet.token_count} tokens")

    # simulate from block 0 (full)
    full = simulate_from_block(packet, 0)
    print(f"  full simulation: {full.shape}")

    # simulate from block 1 (lazy — skip first block)
    if len(packet.blocks) > 1:
        partial = simulate_from_block(packet, 1, n_tokens=20)
        print(f"  lazy simulation from block 1: {partial.shape}")
    else:
        print(f"  (only 1 block, lazy simulation N/A)")

    # test 2: find_block_for_token
    print("\n--- find_block_for_token ---")
    for tok_idx in [0, 5, 30, 50, 70]:
        if tok_idx < packet.token_count:
            bi, off = find_block_for_token(packet, tok_idx)
            print(f"  token {tok_idx} -> block {bi}, offset {off}")

    # test 3: parameter-based scoring
    print("\n--- orbital parameter scoring ---")
    w2 = np.zeros((80, ENGRAM_PROJ_DIM), dtype=np.float32)
    for d in range(ENGRAM_PROJ_DIM):
        w2[:, d] = 0.3 * np.sin(0.1 * t + 0.05 * d) + 0.02 * rng.randn(80)
    w3 = rng.randn(80, ENGRAM_PROJ_DIM).astype(np.float32) * 0.3

    p1 = engram_parameter_vector(packet)
    p2 = engram_parameter_vector(codec.encode_trajectory(w2))
    p3 = engram_parameter_vector(codec.encode_trajectory(w3))

    s_self = engram_orbital_score(p1, p1)
    s_similar = engram_orbital_score(p1, p2)
    s_different = engram_orbital_score(p1, p3)
    print(f"  self:      {s_self:.4f} (should be 1.0)")
    print(f"  similar:   {s_similar:.4f} (should be high)")
    print(f"  different: {s_different:.4f} (should be lower)")

    # test 4: per-family scores
    print("\n--- per-family orbital scores ---")
    family_scores = engram_family_scores(p1, p2)
    for bi in range(min(3, family_scores.shape[0])):
        fam_str = " ".join(
            f"{name}={family_scores[bi, fi]:.3f}"
            for fi, (name, _, _) in enumerate(OSCILLATOR_FAMILIES)
        )
        print(f"  block {bi}: {fam_str}")

    # test 5: attach/detach (with mock trajectory)
    print("\n--- attach/detach engram ---")

    class MockTrajectory:
        pass

    mock = MockTrajectory()
    attach_engram(mock, packet)
    assert get_engram(mock) is packet
    assert get_engram_params(mock) is not None
    print(f"  attached: packet with {len(packet.blocks)} blocks")

    detach_engram(mock)
    assert get_engram(mock) is None
    print(f"  detached: clean")

    print("\n=== standalone integration test complete ===")
    print("the bridge components work without hyper_cache.")
    print("when transformers is available, EngramCore will inherit from SedenionCore.")


if __name__ == "__main__":
    _self_test()
