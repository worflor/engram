"""
sources.py — universal corpus → engram adapter.

a `Source` bundles everything genesis_chain needs to feed arbitrary
sequential data through the ZOOMIN GPU pipeline:

    matrix   : float32 [V, D]     — the embedding lookup table
    pairing  : int32   [D]        — pair ordering the engram codec uses
    to_indices: payload → int32[T] — tokenizer that produces indices

any sequential-data modality fits this shape. the GPU pipeline sees
a generic sequence of indices into a D-dimensional table, so what the
indices MEAN (words, frames, code tokens, VQ codes) is the source's
business. the engram codec only cares about the trajectory dynamics.

builders provided:
    from_glove_dict(glove)            — GloVe-style dict[word → [D] vec]
    from_fasttext_dict(ft)            — fastText with script-aware tokenizer
    from_matrix(matrix, tokenize)     — user-supplied matrix + tokenizer

each builder returns a Source whose `pairing` is identity by default.
genesis_chain will derive a real pairing from a sample on first run and
cache it onto the engram file as `ef._reference_pairing`; subsequent
runs check consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Optional, Tuple

import numpy as np

# WELL_ESTABLISHED is the engram spec's threshold for a well being
# "mature enough to route to" — derived from MIN_BLOCK * AR_ORDER +
# AR_ORDER = 10. we reuse it as the default Bayesian prior strength
# for the refinement blend: a byte needs that much evidence-mass
# before we fully trust the well-aligned estimate.
try:
    from engram_codec import WELL_ESTABLISHED as _WELL_ESTABLISHED
except ImportError:
    _WELL_ESTABLISHED = 10.0


@dataclass
class Source:
    """source of embedding trajectories for the genesis chain pipeline."""
    matrix: np.ndarray              # [V, D] float32, contiguous
    pairing: np.ndarray             # [D] int32
    to_indices: Callable[[Any], np.ndarray]  # payload → int32[T]
    name: str = ""

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def vocab_size(self) -> int:
        return int(self.matrix.shape[0])

    def __post_init__(self):
        m = self.matrix
        if m.dtype != np.float32 or not m.flags.c_contiguous:
            self.matrix = np.ascontiguousarray(m, dtype=np.float32)
        p = self.pairing
        if p.dtype != np.int32 or not p.flags.c_contiguous:
            self.pairing = np.ascontiguousarray(p, dtype=np.int32)
        if self.pairing.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                f"pairing size {self.pairing.shape[0]} != matrix dim "
                f"{self.matrix.shape[1]}")

    def with_pairing(self, pairing: np.ndarray) -> "Source":
        """return a copy with a new pairing (typically derived from a
        sample on first ingestion)."""
        return replace(self,
                       pairing=np.ascontiguousarray(pairing, dtype=np.int32))


# ── builders ──────────────────────────────────────────────────────────

def from_glove_dict(
    glove: dict[str, np.ndarray],
    *,
    name: str = "glove",
) -> Source:
    """build a Source from a GloVe-style dict[word → float32[D]].

    tokenization: lowercase, `[a-z]+` regex, drops single-char tokens.
    this matches fast_embed.tokenize_simple / MASS behavior on English.
    for non-English corpora use from_fasttext_dict (handles
    ideographic scripts at char-level, alphabetic at word-level).
    """
    import re

    words = sorted(glove.keys())
    V = len(words)
    if V == 0:
        raise ValueError("empty glove dict")
    D = int(next(iter(glove.values())).shape[0])

    matrix = np.empty((V, D), dtype=np.float32)
    for i, w in enumerate(words):
        matrix[i] = glove[w]
    word_to_idx = {w: i for i, w in enumerate(words)}

    # try to use the fused MASS tokenizer if available (140× faster on
    # wikitext); fall back to regex for plain text.
    try:
        from mass_embed import make_mass_idx_fn
        to_indices = make_mass_idx_fn(word_to_idx)
    except Exception:
        _pat = re.compile(r"[a-z]+")

        def to_indices(text: str) -> np.ndarray:
            out: list[int] = []
            for w in _pat.findall(text.lower()):
                if len(w) > 1:
                    idx = word_to_idx.get(w)
                    if idx is not None:
                        out.append(idx)
            return np.asarray(out, dtype=np.int32)

    return Source(
        matrix=matrix,
        pairing=np.arange(D, dtype=np.int32),
        to_indices=to_indices,
        name=name,
    )


def from_fasttext_dict(
    ft: dict[str, np.ndarray],
    *,
    ideographic_per_char: bool = True,
    name: str = "fasttext",
) -> Source:
    """build a Source from a fastText-style dict.

    ideographic_per_char=True (default): scripts that are written
    WITHOUT word separators (Chinese ideographs, Japanese kana +
    kanji, CJK Extension A) emit one token per character. scripts
    that ARE written with word separators — Latin, Hangul, Arabic,
    Cyrillic, etc. — cluster runs of alphanumeric characters into
    word tokens. this matches how fastText vocab files are typically
    prepared: ja/zh dicts store characters, ko/ar/en dicts store
    whole words. the classifier is purely Unicode range-based so
    no language flag is needed.

    ideographic_per_char=False: regex-only word tokenization (ASCII
    `[a-z]+`). use only when all content is Latin.
    """
    import re

    words = sorted(ft.keys())
    if not words:
        raise ValueError("empty fasttext dict")
    D = int(next(iter(ft.values())).shape[0])
    matrix = np.stack([ft[w] for w in words]).astype(np.float32)
    word_to_idx = {w: i for i, w in enumerate(words)}

    if ideographic_per_char:
        # unicode ranges that emit character-level tokens.
        # CJK Unified Ideographs         U+4E00..U+9FFF
        # CJK Unified Ideographs Ext-A   U+3400..U+4DBF
        # CJK Compatibility Ideographs   U+F900..U+FAFF
        # Hiragana + Katakana            U+3040..U+30FF
        def _is_ideographic(ch: str) -> bool:
            c = ord(ch)
            return (
                (0x4E00 <= c <= 0x9FFF)
                or (0x3400 <= c <= 0x4DBF)
                or (0xF900 <= c <= 0xFAFF)
                or (0x3040 <= c <= 0x30FF)
            )

        def to_indices(text: str) -> np.ndarray:
            # three-way classifier per character:
            #   1. ideographic → its own token (no clustering)
            #   2. alphanumeric (Hangul, Arabic, Latin, Cyrillic, digits) →
            #      append to current run or start one
            #   3. anything else (spaces, punct, symbols) → break the run
            tokens: list[str] = []
            in_run = False
            for ch in text:
                if _is_ideographic(ch):
                    tokens.append(ch)
                    in_run = False
                elif ch.isalnum():
                    if in_run:
                        tokens[-1] += ch
                    else:
                        tokens.append(ch)
                        in_run = True
                else:
                    in_run = False
            out = [word_to_idx[t.lower()] for t in tokens
                   if t.lower() in word_to_idx]
            return np.asarray(out, dtype=np.int32)
    else:
        _pat = re.compile(r"[a-z]+")

        def to_indices(text: str) -> np.ndarray:
            out = [word_to_idx[w] for w in _pat.findall(text.lower())
                   if w in word_to_idx and len(w) > 1]
            return np.asarray(out, dtype=np.int32)

    return Source(
        matrix=matrix,
        pairing=np.arange(D, dtype=np.int32),
        to_indices=to_indices,
        name=name,
    )


def from_matrix(
    matrix: np.ndarray,
    tokenize: Callable[[Any], np.ndarray],
    *,
    pairing: Optional[np.ndarray] = None,
    name: str = "custom",
) -> Source:
    """generic: any [V, D] matrix + any tokenizer."""
    if pairing is None:
        pairing = np.arange(matrix.shape[1], dtype=np.int32)
    return Source(
        matrix=matrix,
        pairing=pairing,
        to_indices=tokenize,
        name=name,
    )


# ── byte-level Source ─────────────────────────────────────────────────
#
# universal tokenization: every byte is a token. vocab = 256.
# lossless by construction — no stripping, no OOV, works on text/code/
# binary/Unicode UTF-8. pairs naturally with byte_cooc_svd_embedding()
# for corpus-derived structural embeddings, and with an already-refined
# matrix from refine_byte_embedding_from_wells() for iterative
# self-bootstrap.


def from_bytes(
    matrix: Optional[np.ndarray] = None,
    *,
    D: int = 32,
    init: str = "random",
    seed: int = 42,
    name: str = "bytes",
) -> Source:
    """build a byte-level Source.

    every byte 0..255 is a token. lossless tokenization — no regex, no
    case-folding, no OOV. `matrix` is [256, D]; if not provided it is
    initialized per `init`.

    init options:
      "random": gaussian init (scale 0.5). no corpus structure — will
                produce a random-walk trajectory; useful only as a seed
                for iterative refinement.
      "eye":    identity on the first D (or full one-hot if D=256).
                same drawback as random — no semantic similarity.

    to inject corpus-derived structure, pass matrix= from
    byte_cooc_svd_embedding() or refine_byte_embedding_from_wells().
    """
    if matrix is None:
        rng = np.random.default_rng(seed)
        if init == "random":
            matrix = (rng.standard_normal((256, D)).astype(np.float32)
                      * 0.5)
        elif init == "eye":
            matrix = np.eye(256, D, dtype=np.float32)
        else:
            raise ValueError(f"unknown init: {init!r}")
    else:
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        if matrix.shape[0] != 256:
            raise ValueError(
                f"byte matrix must have 256 rows, got {matrix.shape[0]}")

    D_actual = int(matrix.shape[1])

    def to_indices(payload: Any) -> np.ndarray:
        if isinstance(payload, (bytes, bytearray)):
            b = bytes(payload)
        elif isinstance(payload, str):
            b = payload.encode("utf-8", errors="replace")
        else:
            b = str(payload).encode("utf-8", errors="replace")
        return np.frombuffer(b, dtype=np.uint8).astype(np.int32)

    return Source(
        matrix=matrix,
        pairing=np.arange(D_actual, dtype=np.int32),
        to_indices=to_indices,
        name=name,
    )


def byte_cooc_svd_embedding(
    corpus_sample: bytes,
    *,
    D: int = 32,
    window: int = 4,
    min_count: int = 1,
) -> np.ndarray:
    """corpus-derived byte embedding via PMI + truncated SVD on a
    256×256 co-occurrence matrix.

    takes a raw byte sample of the target corpus, counts how often each
    pair of bytes appears within `window` positions of each other,
    converts to positive pointwise mutual information, and takes the
    top-D left-singular-vector projection as the [256, D] embedding.

    bytes that appear in similar contexts (e.g. matching brackets `{` and
    `}`, or common letter bigrams) end up near each other in the result.
    that's the structural-similarity signal that lets engram's trajectory
    coherence detector do real work — which is why random/eye init
    doesn't give meaningful wells on a raw corpus.

    this is equivalent to a word2vec/fastText skipgram at the byte level
    but uses a closed-form SVD instead of SGD. tiny (256×256 matrix) so
    the whole thing runs in well under a second on MB-scale samples.

    the returned matrix is float32, contiguous, shape (256, D).
    """
    if isinstance(corpus_sample, str):
        corpus_sample = corpus_sample.encode("utf-8", errors="replace")
    arr = np.frombuffer(corpus_sample, dtype=np.uint8)
    if arr.size < 2 * window + 2:
        raise ValueError(
            f"corpus_sample too small ({arr.size} bytes) for window={window}")

    cooc = np.zeros((256, 256), dtype=np.float64)
    for offset in range(1, window + 1):
        a = arr[:-offset]
        b = arr[offset:]
        # vectorized scatter-add: cooc[a[k], b[k]] += 1 for all k
        # and symmetric counterpart.
        np.add.at(cooc, (a, b), 1.0)
        np.add.at(cooc, (b, a), 1.0)

    # floor rare pairs so log is well-defined
    cooc = np.where(cooc < min_count, 0.0, cooc)

    row_sum = cooc.sum(axis=1) + 1e-12
    col_sum = cooc.sum(axis=0) + 1e-12
    grand = cooc.sum() + 1e-12

    # shifted positive PMI: max(log(P(a,b)/(P(a)P(b))), 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(
            np.clip(cooc, 1e-12, None) * grand
            / (row_sum[:, None] * col_sum[None, :])
        )
    pmi = np.where(np.isfinite(pmi), pmi, 0.0)
    pmi = np.maximum(pmi, 0.0)

    # truncated SVD on 256x256 — fast (ms).
    U, S, _Vt = np.linalg.svd(pmi, full_matrices=False)
    emb = (U[:, :D] * np.sqrt(S[:D])).astype(np.float32)
    return np.ascontiguousarray(emb)


def refine_byte_embedding_from_well_state(
    ef: "EngramFile",
    *,
    D: int = 300,
    prev_emb: Optional[np.ndarray] = None,
    prior_strength: Optional[float] = None,
    target_norm: Optional[float] = None,
) -> np.ndarray:
    """refine a [256, D] byte embedding from brain state alone.

    equivalent to refine_byte_embedding_from_wells but doesn't need
    corpus_articles or article_to_well: it reads per-well byte
    histograms directly from ef._well_sum_raw (which hetero_v3's
    unified absorb accumulates for every Source now). that makes the
    bootstrap-multilingual chain a one-liner at the end of ingest:
    feed the output back into a from_bytes Source, absorb the next
    wave of corpora through wells the brain already mapped.

    each byte b gets a vector = weighted mean of well centroids
    where the weight is (byte b's count in the well's absorbed
    articles). bytes that appeared heavily in mature wells lock
    in strongly; bytes unseen get the grand-mean centroid.

    D must be <= ef.dim so the byte matrix is compatible with the
    brain's existing _reference_pairing.
    """
    P = ef.dim // 2
    D_out = D if D <= 2 * P else 2 * P

    well_names = [
        n for n in ef._well_sum_K
        if ef._well_sum_raw.get(n) is not None
    ]
    if not well_names:
        raise ValueError(
            "no wells with byte-histogram mass; run a corpus ingest "
            "first (any Source type; hetero_v3 populates "
            "_well_sum_raw for all of them).")

    well_real = np.zeros((len(well_names), 2 * P), dtype=np.float32)
    for i, name in enumerate(well_names):
        centroid = (ef._well_sum_K[name]
                    / max(int(ef._well_count[name]), 1)).astype(np.complex128)
        well_real[i, :P] = centroid.real.astype(np.float32)
        well_real[i, P:] = centroid.imag.astype(np.float32)

    if D_out < 2 * P:
        rng = np.random.default_rng(0)
        proj = rng.standard_normal((2 * P, D_out)).astype(np.float32)
        proj, _ = np.linalg.qr(proj)
        well_emb = well_real @ proj
    else:
        well_emb = well_real

    well_counts_arr = np.array(
        [int(ef._well_count.get(n, 0)) for n in well_names],
        dtype=np.float64)

    new_mat = np.zeros((256, D_out), dtype=np.float32)
    count = np.zeros(256, dtype=np.float64)
    evidence = np.zeros(256, dtype=np.float64)

    for w_idx, name in enumerate(well_names):
        hist = ef._well_sum_raw[name].astype(np.float64)
        nz = np.nonzero(hist)[0]
        if nz.size == 0:
            continue
        new_mat[nz] += hist[nz, None].astype(np.float32) * well_emb[w_idx]
        count[nz] += hist[nz]
        evidence[nz] += hist[nz] * well_counts_arr[w_idx]

    seen = count > 0
    new_mat[seen] /= count[seen, None].astype(np.float32)
    if not seen.all():
        grand = (well_emb.mean(axis=0) if seen.sum() == 0
                 else new_mat[seen].mean(axis=0))
        new_mat[~seen] = grand

    norms = np.linalg.norm(new_mat, axis=1, keepdims=True) + 1e-12
    new_mat = new_mat / norms
    if target_norm is None:
        target_norm = float(np.linalg.norm(well_emb, axis=1).mean()) or 1.0
    new_mat = new_mat * target_norm

    prior = float(prior_strength if prior_strength is not None
                  else _WELL_ESTABLISHED)
    blend_per_byte = evidence / (evidence + prior)  # [256] in [0,1)
    blend_vec = blend_per_byte[:, None]

    if prev_emb is not None:
        prev = np.asarray(prev_emb, dtype=np.float32)
        reference = (prev if prev.shape == new_mat.shape
                     else np.eye(256, D_out, dtype=np.float32) * target_norm)
    else:
        reference = np.eye(256, D_out, dtype=np.float32) * target_norm

    new_mat = blend_vec * new_mat + (1.0 - blend_vec) * reference
    return np.ascontiguousarray(new_mat, dtype=np.float32)


def refine_byte_embedding_from_wells(
    ef: "EngramFile",
    corpus_articles: Iterable[Tuple[str, Any]],
    article_to_well: dict,
    *,
    D: int = 32,
    blend: Optional[float] = None,
    target_norm: Optional[float] = None,
    prev_emb: Optional[np.ndarray] = None,
    prior_strength: Optional[float] = None,
) -> np.ndarray:
    """recursive self-bootstrap: rebuild the byte embedding from the
    well structure that the pipeline already discovered.

    inputs:
      ef               — EngramFile after a prior ingestion. carries
                         _well_sum_K[name] (complex P-dim) and
                         _well_count[name] (segment count).
      corpus_articles  — iterable of (id, text) that was ingested.
      article_to_well  — dict id → well_name from the prior ingestion
                         (captured via the on_article callback).
      D                — target embedding dim. ef.dim / 2 = P; well
                         centroids are complex P-dim → laid out as 2P
                         real floats; D <= 2P recommended.
      blend            — convex combination between well-derived and
                         identity directions for unseen bytes.

    algorithm (what we're doing and why):
      each well has a complex P-dim centroid = sum_K / count. that
      vector lives in the same space as the pipeline's per-segment K
      output — it *is* the attractor basin. we want bytes that tend to
      cause articles to route to well w to end up near w's centroid in
      embedding space, because that makes engram's trajectory coherent
      ON EXACTLY THE REGIME THE PIPELINE ALREADY FOUND.

      per byte v, we:
        1. sum (well_centroid[w] * count(v in article_of_w)) across
           every article of w, pooling over all wells.
        2. normalize by total count of v in the corpus.
      so byte v's new vector = byte-v-weighted average of the well
      centroids that its containing articles routed to.

      unseen bytes (zero corpus mass) get the grand-mean centroid as a
      reasonable seed so they don't land at origin.

      blend interpolates between the previous embedding and the well-
      aligned vector:
        refined[b] = (1 - blend[b]) * prev_emb[b] + blend[b] * well_aligned[b]

      by default (blend=None) blend is computed PER-BYTE via a Bayesian
      rule from the maturity of the wells that byte appears in:
        blend[b] = evidence[b] / (evidence[b] + prior_strength)
      where evidence[b] is the total segment-count of all wells
      weighted by how often byte b appears in their articles. mature
      wells pull their bytes hard; fresh wells pull softly. bytes
      that never appeared in any article get blend=0 (no update).

      prior_strength defaults to WELL_ESTABLISHED, the same constant
      the brain uses to decide a well is "real enough to route to."
      smaller prior = faster adaptation, more variance. larger prior
      = smoother updates.

      passing blend=<scalar> forces a uniform blend across all bytes
      (legacy behavior). passing None uses the derived per-byte rule.

      when prev_emb is None we fall back to blending with a scaled
      identity — collapses to rank ~ n_wells for small brains. pass
      prev_emb when iterating.

    returns [256, D] float32 contiguous.
    """
    P = ef.dim // 2
    D_out = D if D <= 2 * P else 2 * P

    # well centroids in 2P real (re || im)
    well_names = list(ef._well_sum_K.keys())
    if not well_names:
        raise ValueError("ef has no wells - ingest first")

    well_real = np.zeros((len(well_names), 2 * P), dtype=np.float32)
    for i, name in enumerate(well_names):
        centroid = (ef._well_sum_K[name]
                    / max(int(ef._well_count[name]), 1)).astype(np.complex128)
        well_real[i, :P] = centroid.real.astype(np.float32)
        well_real[i, P:] = centroid.imag.astype(np.float32)
    name_to_idx = {n: i for i, n in enumerate(well_names)}

    # project well_real down to D_out with a random orthogonal proj if
    # D_out < 2P. keeps isometric structure approximately.
    if D_out < 2 * P:
        rng = np.random.default_rng(0)
        proj = rng.standard_normal((2 * P, D_out)).astype(np.float32)
        proj, _ = np.linalg.qr(proj)
        well_emb = well_real @ proj
    else:
        well_emb = well_real

    # per-byte accumulators. evidence[b] is the total well-count mass
    # of articles containing byte b — used for the Bayesian blend.
    # mature wells contribute more evidence per byte occurrence; a
    # byte in a 10k-count well carries more weight than the same
    # byte in a 1-count well.
    new_mat = np.zeros((256, D_out), dtype=np.float32)
    count = np.zeros(256, dtype=np.float64)
    evidence = np.zeros(256, dtype=np.float64)

    well_counts_arr = np.array(
        [int(ef._well_count.get(n, 0)) for n in well_names],
        dtype=np.float64)

    for aid, payload in corpus_articles:
        well_name = article_to_well.get(aid)
        if well_name is None or well_name not in name_to_idx:
            continue
        w_idx = name_to_idx[well_name]
        w_weight = well_counts_arr[w_idx]
        if isinstance(payload, (bytes, bytearray)):
            b = bytes(payload)
        elif isinstance(payload, str):
            b = payload.encode("utf-8", errors="replace")
        else:
            b = str(payload).encode("utf-8", errors="replace")
        if not b:
            continue
        bv = np.frombuffer(b, dtype=np.uint8)
        hist = np.bincount(bv, minlength=256)
        nz = np.nonzero(hist)[0]
        if nz.size == 0:
            continue
        new_mat[nz] += hist[nz, None] * well_emb[w_idx]
        count[nz] += hist[nz]
        evidence[nz] += hist[nz] * w_weight

    # normalize seen bytes; fill unseen with grand mean
    seen = count > 0
    new_mat[seen] /= count[seen, None]
    if not seen.all():
        grand = (well_emb.mean(axis=0) if seen.sum() == 0
                 else new_mat[seen].mean(axis=0))
        new_mat[~seen] = grand

    # L2-normalize to stabilize direction, then scale to target_norm.
    # if target_norm is None, infer from the well centroid distribution
    # so the refined embedding lives at the same energy scale the
    # pipeline already saw. if the caller knows the original embedding's
    # norm scale (e.g. from cooc-SVD), passing it explicitly keeps
    # trajectory energies matched across refinement iterations — which
    # prevents segment.wgsl's var < MACHINE_EPS branch from trivially
    # tripping and dropping articles.
    norms = np.linalg.norm(new_mat, axis=1, keepdims=True) + 1e-12
    new_mat = new_mat / norms
    if target_norm is None:
        target_norm = float(np.linalg.norm(well_emb, axis=1).mean()) or 1.0
    new_mat = new_mat * target_norm

    # derive per-byte blend from evidence, OR use the scalar the
    # caller passed. the adaptive form:
    #     blend[b] = evidence[b] / (evidence[b] + prior_strength)
    # is the Bayesian posterior weight for the well-aligned estimate
    # given evidence[b] observations with prior pseudo-count
    # prior_strength. a byte appearing heavily in mature wells gets
    # blend → 1 (trust the new estimate); an unseen byte gets blend
    # = 0 (no update). default prior_strength is WELL_ESTABLISHED so
    # blend crosses 0.5 exactly at the established-well threshold.
    if blend is None:
        prior = float(prior_strength if prior_strength is not None
                      else _WELL_ESTABLISHED)
        blend_per_byte = evidence / (evidence + prior)  # [256] in [0,1)
        blend_vec = blend_per_byte[:, None]  # broadcast for [256, D]
    else:
        # scalar path: same weight for every byte.
        blend_vec = float(blend)

    # choose the reference we blend toward (what the non-well portion
    # contributes). prev_emb preserves rank across iterations; identity
    # is the legacy fallback that collapses to rank<=n_wells for small
    # brains.
    if prev_emb is not None:
        prev = np.asarray(prev_emb, dtype=np.float32)
        if prev.shape != new_mat.shape:
            reference = np.eye(256, D_out, dtype=np.float32) * target_norm
        else:
            reference = prev
    else:
        reference = np.eye(256, D_out, dtype=np.float32) * target_norm

    new_mat = blend_vec * new_mat + (1.0 - blend_vec) * reference

    return np.ascontiguousarray(new_mat)


