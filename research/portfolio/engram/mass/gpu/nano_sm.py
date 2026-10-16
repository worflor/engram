"""
nano_sm.py — minimal MASS state machine for parallel-scan proof of
concept. designed so state fits in a small alphabet (8 states) and
the transition table is small (8 × 256 bytes = 2 KB).

states:
    0 PROSE        — normal prose, accumulate words
    1 LB1          — just saw single '{'
    2 TMPL         — inside '{{...}}', skip
    3 RB1_TMPL     — inside template, just saw single '}'
    4 LBR1         — just saw single '['
    5 LINK         — inside '[[...]]', emit chars
    6 RBR1_LINK    — inside link, just saw single ']'
    7 LT_TAG       — inside '<...>', skip until '>'

this is intentionally simpler than v4 — no depth tracking, no
category handling, no cell-attribute skip. the point is to prove
the scan-based architecture works, not to match v4's full grammar.
"""

from __future__ import annotations

import numpy as np


# state constants
S_PROSE      = 0
S_LB1        = 1
S_TMPL       = 2
S_RB1_TMPL   = 3
S_LBR1       = 4
S_LINK       = 5
S_RBR1_LINK  = 6
S_LT_TAG     = 7
N_STATES = 8

# byte constants
LBRACE = 0x7B
RBRACE = 0x7D
LBRACK = 0x5B
RBRACK = 0x5D
LT     = 0x3C
GT     = 0x3E


def build_transition_table() -> np.ndarray:
    """
    T[state, byte] = next_state.
    encodes the nano state machine. returns shape (8, 256) uint8.
    """
    T = np.empty((N_STATES, 256), dtype=np.uint8)
    # default: stay in same state
    for s in range(N_STATES):
        T[s, :] = s

    # PROSE
    T[S_PROSE, :] = S_PROSE
    T[S_PROSE, LBRACE] = S_LB1
    T[S_PROSE, LBRACK] = S_LBR1
    T[S_PROSE, LT]     = S_LT_TAG

    # LB1: just saw '{', if another '{' → TMPL, else back to PROSE
    # (and if it was another '{', we never really left PROSE for a
    # token purposes; but our scan treats the single '{' as a
    # transient state that resolves in one byte)
    T[S_LB1, :]       = S_PROSE
    T[S_LB1, LBRACE]  = S_TMPL
    # any other char: we were wrong about it being a template, fall
    # back to PROSE. technically we also "lost" the '{' char but
    # that's fine — single '{' is not a word char anyway.
    T[S_LB1, LBRACK]  = S_LBR1
    T[S_LB1, LT]      = S_LT_TAG

    # TMPL
    T[S_TMPL, :]      = S_TMPL
    T[S_TMPL, RBRACE] = S_RB1_TMPL

    # RB1_TMPL
    T[S_RB1_TMPL, :]      = S_TMPL
    T[S_RB1_TMPL, RBRACE] = S_PROSE

    # LBR1
    T[S_LBR1, :]       = S_PROSE
    T[S_LBR1, LBRACK]  = S_LINK
    T[S_LBR1, LBRACE]  = S_LB1
    T[S_LBR1, LT]      = S_LT_TAG

    # LINK
    T[S_LINK, :]       = S_LINK
    T[S_LINK, RBRACK]  = S_RBR1_LINK

    # RBR1_LINK
    T[S_RBR1_LINK, :]       = S_LINK
    T[S_RBR1_LINK, RBRACK]  = S_PROSE

    # LT_TAG
    T[S_LT_TAG, :]     = S_LT_TAG
    T[S_LT_TAG, GT]    = S_PROSE

    return T


def is_emit_state(state: int) -> bool:
    """states from which word-chars contribute to emitted tokens."""
    return state == S_PROSE or state == S_LINK


# ── sequential reference implementation ──────────────────────────────

def run_sequential(text: bytes, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    reference: byte-by-byte through the state machine.
    returns (states[N+1] starting with initial, words: list of (start, len)).
    """
    n = len(text)
    states = np.empty(n + 1, dtype=np.uint8)
    states[0] = S_PROSE
    for i in range(n):
        states[i + 1] = T[states[i], text[i]]

    # derive words: wordchar runs while state is PROSE or LINK
    words = []
    word_start = -1
    for i in range(n):
        s = states[i]
        b = text[i]
        is_wc = (0x61 <= b <= 0x7A) or (0x41 <= b <= 0x5A) or (0x30 <= b <= 0x39)
        in_emit = is_emit_state(s)
        if is_wc and in_emit:
            if word_start < 0:
                word_start = i
        else:
            if word_start >= 0:
                words.append((word_start, i - word_start))
                word_start = -1
    if word_start >= 0:
        words.append((word_start, n - word_start))

    return states, words


# ── parallel-scan implementation (numpy for verification) ────────────

def compose(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    compose two transition functions f, g: (f ∘ g)[s] = f[g[s]].
    input shape: (8,) uint8 each. output shape: (8,) uint8.
    this is the monoid operation we parallel-scan.
    """
    return f[g]


def compose_batch(fns: np.ndarray, gns: np.ndarray) -> np.ndarray:
    """elementwise composition of two arrays of functions.
    fns, gns: shape (N, 8). output: (N, 8)."""
    # (fns ∘ gns)[i, s] = fns[i, gns[i, s]]
    # vectorized: for each i, row_i = fns[i][gns[i]]
    # numpy advanced indexing:
    N = fns.shape[0]
    idx_0 = np.arange(N, dtype=np.int64)[:, None]  # (N, 1)
    idx_1 = gns.astype(np.int64)                   # (N, 8)
    return fns[idx_0, idx_1]


def run_scan(text: bytes, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    parallel scan implementation (numpy). correctness-verifying version
    — not actually parallel, but uses the same ALGEBRA as the gpu
    version will.

    for each byte i, build the per-byte transition function fn[i]:
        fn[i][s] = T[s, byte[i]]
    then prefix-scan by composition. state[i+1] = (fn[i] ∘ fn[i-1] ∘ ...)[S_PROSE].
    """
    n = len(text)
    if n == 0:
        return np.array([S_PROSE], dtype=np.uint8), []

    text_arr = np.frombuffer(text, dtype=np.uint8)
    # per-byte transition function: shape (N, 8)
    fns = T[:, text_arr].T.copy()  # (N, 8)

    # inclusive scan by composition: prefix[i] = fn[i] ∘ ... ∘ fn[0]
    # naive sequential scan for reference (gpu will do log-depth parallel)
    prefix = fns.copy()
    for i in range(1, n):
        prefix[i] = compose(prefix[i], prefix[i - 1])

    # apply initial state to get state at each position (after processing byte i)
    states_after = prefix[:, S_PROSE]  # state right after byte i
    states = np.empty(n + 1, dtype=np.uint8)
    states[0] = S_PROSE
    states[1:] = states_after

    # same emission logic as sequential
    words = []
    word_start = -1
    for i in range(n):
        s = states[i]
        b = text_arr[i]
        is_wc = (0x61 <= b <= 0x7A) or (0x41 <= b <= 0x5A) or (0x30 <= b <= 0x39)
        in_emit = s == S_PROSE or s == S_LINK
        if is_wc and in_emit:
            if word_start < 0:
                word_start = i
        else:
            if word_start >= 0:
                words.append((word_start, i - word_start))
                word_start = -1
    if word_start >= 0:
        words.append((word_start, n - word_start))

    return states, words


# ── logarithmic-depth parallel scan via Hillis-Steele ────────────────

def run_hillis_steele(text: bytes, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Hillis-Steele inclusive parallel scan with function composition.

    at stage d (d = 1, 2, 4, 8, ...):
        for all i in parallel, if i >= d:
            scan[i] = scan[i] ∘ scan[i - d]

    O(n log n) work, O(log n) depth. correct for non-commutative
    monoids (just apply ∘ consistently — later-in-time on the left).
    this is what the WGSL kernel will execute.
    """
    text_arr = np.frombuffer(text, dtype=np.uint8)
    n = text_arr.shape[0]
    if n == 0:
        return np.array([S_PROSE], dtype=np.uint8), []

    # per-byte transition function, shape (N, 8): row i is the
    # transition applied when seeing byte text[i]. fns[i][s] = T[s, text[i]].
    fns = T[:, text_arr].T.copy()  # (N, 8), uint8

    # inclusive scan
    scan = fns.copy()
    d = 1
    while d < n:
        # new = scan[i] ∘ scan[i-d] for i >= d
        # do in two buffers to avoid in-place hazard
        out = scan.copy()
        # for i >= d: out[i] = scan[i][scan[i-d]]
        mask = np.arange(n) >= d
        idxs = np.where(mask)[0]
        for i in idxs:
            out[i] = scan[i][scan[i - d]]
        scan = out
        d *= 2

    # scan[i] applied to S_PROSE → state AFTER byte i
    states = np.empty(n + 1, dtype=np.uint8)
    states[0] = S_PROSE
    states[1:] = scan[:, S_PROSE]

    words = []
    word_start = -1
    for i in range(n):
        s = states[i]
        b = text_arr[i]
        is_wc = (0x61 <= b <= 0x7A) or (0x41 <= b <= 0x5A) or (0x30 <= b <= 0x39)
        in_emit = s == S_PROSE or s == S_LINK
        if is_wc and in_emit:
            if word_start < 0:
                word_start = i
        else:
            if word_start >= 0:
                words.append((word_start, i - word_start))
                word_start = -1
    if word_start >= 0:
        words.append((word_start, n - word_start))

    return states, words


# keep the old name as alias so tests still work
run_blelloch = run_hillis_steele
