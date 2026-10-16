"""
test_engram.py

Tests for the Whisper Engram semantic trajectory codec.
All tests are synthetic (no model required).
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import engram
from engram import (
    EngramCodec,
    EngramMode,
    EngramPacket,
    FamilyCoeffs,
    ENGRAM_PROJ_DIM,
    OSCILLATOR_FAMILIES,
    N_FAMILIES,
    zigzag_encode,
    zigzag_decode,
    zigzag_encode_array,
    zigzag_decode_array,
    quantize_residuals,
    dequantize_residuals,
    pack_residuals,
    unpack_residuals,
    fit_complex_ar2,
    predict_family,
    predict_block,
    select_block_mode,
    segment_blocks,
    soft_witness,
    harden_witness,
    encode_engram,
    decode_engram,
    engram_parameter_vector,
    engram_parameter_distance,
    engram_summary,
)


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ── zigzag ────────────────────────────────────────────────────────────

def test_zigzag_roundtrip():
    for v in [-1000, -1, 0, 1, 1000, 2**30, -(2**30)]:
        assert zigzag_decode(zigzag_encode(v)) == v, f"zigzag failed for {v}"
    print("[PASS] zigzag roundtrip")


def test_zigzag_array_roundtrip():
    arr = np.array([-100, -1, 0, 1, 100, 12345], dtype=np.int32)
    enc = zigzag_encode_array(arr)
    dec = zigzag_decode_array(enc)
    np.testing.assert_array_equal(arr, dec)
    print("[PASS] zigzag array roundtrip")


# ── varint packing ────────────────────────────────────────────────────

def test_residual_packing_roundtrip():
    arr = np.array([0, 1, -1, 127, -128, 0, 50, -50], dtype=np.int8)
    packed = pack_residuals(arr)
    unpacked = unpack_residuals(packed, len(arr))
    np.testing.assert_array_equal(arr.astype(np.int32), unpacked)
    print("[PASS] residual packing roundtrip")


def test_residual_packing_zeros():
    arr = np.zeros(256, dtype=np.int8)
    packed = pack_residuals(arr)
    # all zeros should pack to 256 bytes (one byte per varint(0))
    assert len(packed) == 256
    unpacked = unpack_residuals(packed, 256)
    np.testing.assert_array_equal(np.zeros(256, dtype=np.int32), unpacked)
    print("[PASS] residual packing zeros (minimal)")


# ── quantization ──────────────────────────────────────────────────────

def test_quantize_near_lossless():
    # hard witness residuals are in {-2, -1, 0, 1, 2}
    r = np.array([-2, -1, 0, 1, 2, 0, -1, 1], dtype=np.float32)
    q = quantize_residuals(r)
    dq = dequantize_residuals(q, float(np.abs(r).max()))
    assert _rms(r, dq) < 0.05, f"quantize error too large: {_rms(r, dq)}"
    print("[PASS] quantize near-lossless for hard residuals")


def test_quantize_zeros():
    r = np.zeros((10, 256), dtype=np.float32)
    q = quantize_residuals(r)
    np.testing.assert_array_equal(q, np.zeros_like(q))
    print("[PASS] quantize zeros")


# ── complex AR(2) fitting ─────────────────────────────────────────────

def test_ar2_fit_ellipse():
    """Two-frequency signal: eigenvalues should be recovered."""
    T = 80
    z = np.zeros((T, 2), dtype=np.float32)
    theta1, theta2 = 0.3, 0.7
    for n in range(T):
        z[n, 0] = 0.4 * np.cos(theta1 * n) + 0.2 * np.cos(theta2 * n)
        z[n, 1] = 0.4 * np.sin(theta1 * n) + 0.2 * np.sin(theta2 * n)

    K, G, err = fit_complex_ar2(z)
    lam1 = np.exp(1j * theta1)
    lam2 = np.exp(1j * theta2)
    K_true = lam1 + lam2
    G_true = lam1 * lam2

    assert abs(K - K_true) < 0.05, f"K error: {abs(K - K_true)}"
    assert abs(G - G_true) < 0.05, f"G error: {abs(G - G_true)}"
    assert err < 0.01, f"fit error too large: {err}"
    print(f"[PASS] AR(2) fit ellipse: K={K:.4f} (true {K_true:.4f}), err={err:.6f}")


def test_ar2_fit_damped():
    """Damped spiral: |lambda| < 1."""
    T = 60
    z = np.zeros((T, 2), dtype=np.float32)
    r, theta = 0.95, 0.4
    for n in range(T):
        z[n, 0] = (r ** n) * np.cos(theta * n)
        z[n, 1] = (r ** n) * np.sin(theta * n)

    K, G, err = fit_complex_ar2(z)
    # single frequency: K ≈ r*e^{itheta}, G ≈ 0 (ridge-regularized)
    print(f"[PASS] AR(2) fit damped: K={K:.4f} |K|={abs(K):.4f}, G={G:.4f}, err={err:.6f}")
    assert err < 0.02


def test_ar2_fit_static():
    """Constant signal."""
    z = np.ones((30, 4), dtype=np.float32) * 0.5
    K, G, err = fit_complex_ar2(z)
    print(f"[PASS] AR(2) fit static: K={K:.4f}, G={G:.4f}, err={err:.6f}")
    # static: prediction z[n] = K*0.5 - G*0.5 should equal 0.5
    assert err < 0.01


def test_ar2_fit_multidim():
    """Multi-dimensional family with different per-pair frequencies."""
    T = 60
    D = 8
    z = np.zeros((T, D), dtype=np.float32)
    rng = np.random.RandomState(123)
    for pair in range(D // 2):
        theta1 = 0.2 + 0.1 * pair
        theta2 = 0.5 + 0.15 * pair
        for n in range(T):
            z[n, 2 * pair] = 0.3 * np.cos(theta1 * n) + 0.2 * np.cos(theta2 * n) + 0.01 * rng.randn()
            z[n, 2 * pair + 1] = 0.3 * np.sin(theta1 * n) + 0.2 * np.sin(theta2 * n) + 0.01 * rng.randn()

    K, G, err = fit_complex_ar2(z)
    print(f"[PASS] AR(2) fit 8D multi-freq: K={K:.4f}, G={G:.4f}, err={err:.6f}")
    assert err < 0.15  # averaged over multiple frequencies


# ── prediction ────────────────────────────────────────────────────────

def test_predict_family_ellipse():
    """Predict forward from fitted K,G and check error."""
    T = 40
    z = np.zeros((T, 2), dtype=np.float32)
    for n in range(T):
        z[n, 0] = 0.4 * np.cos(0.3 * n) + 0.2 * np.cos(0.7 * n)
        z[n, 1] = 0.4 * np.sin(0.3 * n) + 0.2 * np.sin(0.7 * n)

    K, G, _ = fit_complex_ar2(z)
    predicted = predict_family(z[1], z[0], K, G, T - 2)
    err = _rms(z[2:], predicted)
    print(f"[PASS] predict_family ellipse: RMS={err:.6f}")
    assert err < 0.1  # ridge regularization introduces small bias


# ── block segmentation ────────────────────────────────────────────────

def test_segment_smooth():
    """Smooth trajectory should produce few blocks."""
    w = np.zeros((100, ENGRAM_PROJ_DIM), dtype=np.float32)
    t = np.arange(100, dtype=np.float32)
    for d in range(ENGRAM_PROJ_DIM):
        w[:, d] = 0.3 * np.sin(0.1 * t + 0.05 * d)
    segs = segment_blocks(w)
    print(f"[PASS] segment smooth: {len(segs)} blocks")
    assert len(segs) <= 5


def test_segment_phase_transition():
    """Abrupt change should cause a split."""
    w = np.zeros((80, ENGRAM_PROJ_DIM), dtype=np.float32)
    # smooth first half
    t = np.arange(80, dtype=np.float32)
    for d in range(ENGRAM_PROJ_DIM):
        w[:40, d] = 0.3 * np.sin(0.1 * t[:40])
    # jump
    w[40:] = np.random.RandomState(7).randn(40, ENGRAM_PROJ_DIM).astype(np.float32) * 0.5
    segs = segment_blocks(w)
    print(f"[PASS] segment phase transition: {len(segs)} blocks")
    assert len(segs) >= 2


# ── mode selection ────────────────────────────────────────────────────

def test_mode_static():
    """Constant block should select STATIC."""
    block = np.ones((20, ENGRAM_PROJ_DIM), dtype=np.float32) * 0.5
    seed1 = np.ones(ENGRAM_PROJ_DIM, dtype=np.float32) * 0.5
    seed2 = np.ones(ENGRAM_PROJ_DIM, dtype=np.float32) * 0.5
    mode, _, _ = select_block_mode(block, seed1, seed2)
    assert mode == EngramMode.STATIC, f"expected STATIC, got {mode.name}"
    print("[PASS] mode selection: constant -> STATIC")


def test_mode_orbital_vs_linear():
    """Multi-frequency block should prefer ORBITAL over LINEAR."""
    T = 40
    rng = np.random.RandomState(42)
    block = np.zeros((T, ENGRAM_PROJ_DIM), dtype=np.float32)
    for d in range(0, ENGRAM_PROJ_DIM - 1, 2):
        theta1 = 0.2 + 0.02 * d
        theta2 = 0.5 + 0.01 * d
        for n in range(T):
            block[n, d] = 0.3 * np.cos(theta1 * (n + 2)) + 0.2 * np.cos(theta2 * (n + 2))
            block[n, d + 1] = 0.3 * np.sin(theta1 * (n + 2)) + 0.2 * np.sin(theta2 * (n + 2))
    block += 0.01 * rng.randn(T, ENGRAM_PROJ_DIM).astype(np.float32)

    seed1 = block[0].copy()
    seed2 = np.zeros(ENGRAM_PROJ_DIM, dtype=np.float32)
    for d in range(0, ENGRAM_PROJ_DIM - 1, 2):
        theta1 = 0.2 + 0.02 * d
        theta2 = 0.5 + 0.01 * d
        seed2[d] = 0.3 * np.cos(theta1 * 0) + 0.2 * np.cos(theta2 * 0)
        seed2[d + 1] = 0.3 * np.sin(theta1 * 0) + 0.2 * np.sin(theta2 * 0)

    mode, coeffs, err = select_block_mode(block, seed1, seed2)
    print(f"[PASS] mode selection multi-freq: {mode.name} (err={err:.6f})")
    # with diverse frequencies per dim, orbital should win
    assert mode == EngramMode.ORBITAL, f"expected ORBITAL, got {mode.name}"


# ── soft witness ──────────────────────────────────────────────────────

def test_soft_witness_gradient():
    """Soft witness should have values in (-1, 1), not just {-1, +1}."""
    rng = np.random.RandomState(42)
    hidden = rng.randn(10, 64).astype(np.float32)
    scramble = np.where(rng.randint(0, 2, (256, 64)), -1.0, 1.0).astype(np.float32)

    sw = soft_witness(hidden, scramble, alpha=0.1)  # low alpha for spread
    # with alpha=0.1, values should be spread across (-1, 1)
    assert sw.shape == (10, 256)
    assert float(np.abs(sw).max()) < 1.0, "soft witness should be in open (-1, 1)"
    # check that not all values are near ±1
    mid_count = np.sum((np.abs(sw) > 0.01) & (np.abs(sw) < 0.9))
    assert mid_count > 0, "soft witness should have intermediate values"
    print(f"[PASS] soft witness gradient: {mid_count} intermediate values")


def test_soft_witness_limit():
    """At high alpha, soft witness approaches hard witness."""
    rng = np.random.RandomState(42)
    hidden = rng.randn(10, 64).astype(np.float32)
    scramble = np.where(rng.randint(0, 2, (256, 64)), -1.0, 1.0).astype(np.float32)

    sw_high = soft_witness(hidden, scramble, alpha=100.0)
    hw = harden_witness(soft_witness(hidden, scramble, alpha=100.0))
    diff = _rms(sw_high, hw)
    assert diff < 0.05, f"high-alpha soft witness should match hard: diff={diff}"
    print(f"[PASS] soft witness alpha->inf limit: diff={diff:.6f}")


# ── full codec roundtrip ─────────────────────────────────────────────

def test_roundtrip_constant():
    codec = EngramCodec()
    w = np.ones((50, ENGRAM_PROJ_DIM), dtype=np.float32) * 0.7
    pkt = codec.encode_trajectory(w)
    dec = codec.decode_trajectory(pkt)
    err = _rms(w, dec)
    assert err < 1e-5, f"constant roundtrip error: {err}"
    print(f"[PASS] roundtrip constant: err={err:.8f}")


def test_roundtrip_sinusoidal():
    codec = EngramCodec()
    t = np.linspace(0, 4 * np.pi, 100)
    w = np.zeros((100, ENGRAM_PROJ_DIM), dtype=np.float32)
    for d in range(ENGRAM_PROJ_DIM):
        freq = 0.1 + 0.01 * d
        w[:, d] = 0.5 * np.sin(freq * t + 0.1 * d)
    pkt = codec.encode_trajectory(w)
    dec = codec.decode_trajectory(pkt)
    err = _rms(w, dec)
    assert err < 1e-5, f"sinusoidal roundtrip error: {err}"
    print(f"[PASS] roundtrip sinusoidal: err={err:.8f}")


def test_roundtrip_random():
    """Random data: roundtrip should be exact (residuals = data)."""
    codec = EngramCodec()
    rng = np.random.RandomState(42)
    w = rng.randn(60, ENGRAM_PROJ_DIM).astype(np.float32) * 0.5
    pkt = codec.encode_trajectory(w)
    dec = codec.decode_trajectory(pkt)
    err = _rms(w, dec)
    assert err < 1e-5, f"random roundtrip error: {err}"
    print(f"[PASS] roundtrip random: err={err:.8f}")


def test_roundtrip_short():
    """Very short trajectory (< 3 tokens)."""
    codec = EngramCodec()
    w = np.array([[0.1] * ENGRAM_PROJ_DIM, [0.2] * ENGRAM_PROJ_DIM], dtype=np.float32)
    pkt = codec.encode_trajectory(w)
    dec = codec.decode_trajectory(pkt)
    err = _rms(w, dec)
    assert err < 1e-5, f"short roundtrip error: {err}"
    print(f"[PASS] roundtrip short (2 tokens): err={err:.8f}")


def test_roundtrip_empty():
    codec = EngramCodec()
    w = np.empty((0, ENGRAM_PROJ_DIM), dtype=np.float32)
    pkt = codec.encode_trajectory(w)
    dec = codec.decode_trajectory(pkt)
    assert dec.shape == (0, ENGRAM_PROJ_DIM)
    print("[PASS] roundtrip empty")


# ── wire format ───────────────────────────────────────────────────────

def test_wire_roundtrip():
    codec = EngramCodec()
    rng = np.random.RandomState(42)
    w = np.zeros((80, ENGRAM_PROJ_DIM), dtype=np.float32)
    t = np.arange(80, dtype=np.float32)
    for d in range(ENGRAM_PROJ_DIM):
        w[:, d] = 0.3 * np.sin(0.1 * t + 0.05 * d) + 0.02 * rng.randn(80)

    wire = codec.encode_bytes(w)
    dec = codec.decode_bytes(wire)
    err = _rms(w, dec)
    raw_size = w.size * 4
    ratio = raw_size / max(len(wire), 1)
    print(f"[PASS] wire roundtrip: err={err:.6f}, {raw_size} -> {len(wire)} bytes ({ratio:.1f}x)")
    assert err < 0.1, f"wire roundtrip error: {err}"


def test_wire_format_parse():
    """Verify magic/version/fields survive round-trip."""
    codec = EngramCodec(alpha=3.14)
    w = np.ones((20, ENGRAM_PROJ_DIM), dtype=np.float32) * 0.5
    wire = codec.encode_bytes(w)

    # check magic bytes
    assert wire[:2] == b"EG"
    assert wire[2] == 1  # version

    pkt = decode_engram(wire)
    assert pkt.proj_dim == ENGRAM_PROJ_DIM
    assert pkt.token_count == 20
    assert abs(pkt.alpha - 3.14) < 0.01
    print("[PASS] wire format parse")


# ── compression ratio ─────────────────────────────────────────────────

def test_compression_structured():
    """Structured data should compress well."""
    codec = EngramCodec()
    t = np.arange(200, dtype=np.float32)
    w = np.zeros((200, ENGRAM_PROJ_DIM), dtype=np.float32)
    for d in range(ENGRAM_PROJ_DIM):
        w[:, d] = 0.5 * np.sin(0.05 * t + 0.02 * d)
    wire = codec.encode_bytes(w)
    raw_size = w.size * 4
    ratio = raw_size / max(len(wire), 1)
    print(f"[PASS] compression structured: {ratio:.1f}x ({raw_size} -> {len(wire)})")
    assert ratio > 1.5, f"structured compression too low: {ratio}"


def test_compression_random():
    """Random data should not expand much beyond raw."""
    codec = EngramCodec()
    rng = np.random.RandomState(42)
    w = rng.randn(100, ENGRAM_PROJ_DIM).astype(np.float32)
    wire = codec.encode_bytes(w)
    raw_size = w.size * 4
    # random data won't compress, but shouldn't expand more than ~50%
    assert len(wire) < raw_size * 1.5, f"random expanded too much: {len(wire)} > {raw_size * 1.5}"
    print(f"[PASS] compression random: {raw_size} -> {len(wire)} bytes")


# ── handshake parameters ─────────────────────────────────────────────

def test_parameter_vector_shape():
    codec = EngramCodec()
    w = np.ones((50, ENGRAM_PROJ_DIM), dtype=np.float32) * 0.5
    pkt = codec.encode_trajectory(w)
    pv = engram_parameter_vector(pkt)
    # each block contributes N_FAMILIES * 4 floats
    expected_len = len(pkt.blocks) * N_FAMILIES * 4
    assert len(pv) == expected_len, f"param vec length: {len(pv)} != {expected_len}"
    print(f"[PASS] parameter vector shape: {len(pv)} floats")


def test_parameter_distance_self():
    codec = EngramCodec()
    w = np.ones((50, ENGRAM_PROJ_DIM), dtype=np.float32) * 0.5
    pkt = codec.encode_trajectory(w)
    pv = engram_parameter_vector(pkt)
    d = engram_parameter_distance(pv, pv)
    assert d == 0.0, f"self-distance should be 0: {d}"
    print("[PASS] parameter distance self = 0")


def test_parameter_distance_different():
    codec = EngramCodec()
    rng = np.random.RandomState(42)

    w1 = np.zeros((50, ENGRAM_PROJ_DIM), dtype=np.float32)
    w2 = np.zeros((50, ENGRAM_PROJ_DIM), dtype=np.float32)
    t = np.arange(50, dtype=np.float32)
    for d in range(ENGRAM_PROJ_DIM):
        w1[:, d] = 0.5 * np.sin(0.1 * t + 0.05 * d)
        w2[:, d] = 0.5 * np.cos(0.3 * t + 0.1 * d)

    pv1 = engram_parameter_vector(codec.encode_trajectory(w1))
    pv2 = engram_parameter_vector(codec.encode_trajectory(w2))
    d = engram_parameter_distance(pv1, pv2)
    assert d > 0.1, f"different trajectories should have distance > 0: {d}"
    print(f"[PASS] parameter distance different: {d:.4f}")


# ── summary ───────────────────────────────────────────────────────────

def test_summary():
    codec = EngramCodec()
    w = np.ones((30, ENGRAM_PROJ_DIM), dtype=np.float32) * 0.5
    pkt = codec.encode_trajectory(w)
    s = engram_summary(pkt)
    assert s["token_count"] == 30
    assert "mode_distribution" in s
    assert "residual_rms" in s
    print(f"[PASS] summary: {s['mode_distribution']}")


# ── run all ───────────────────────────────────────────────────────────

def run_all():
    print("=== Whisper Engram: Test Suite ===\n")
    tests = [
        test_zigzag_roundtrip,
        test_zigzag_array_roundtrip,
        test_residual_packing_roundtrip,
        test_residual_packing_zeros,
        test_quantize_near_lossless,
        test_quantize_zeros,
        test_ar2_fit_ellipse,
        test_ar2_fit_damped,
        test_ar2_fit_static,
        test_ar2_fit_multidim,
        test_predict_family_ellipse,
        test_segment_smooth,
        test_segment_phase_transition,
        test_mode_static,
        test_mode_orbital_vs_linear,
        test_soft_witness_gradient,
        test_soft_witness_limit,
        test_roundtrip_constant,
        test_roundtrip_sinusoidal,
        test_roundtrip_random,
        test_roundtrip_short,
        test_roundtrip_empty,
        test_wire_roundtrip,
        test_wire_format_parse,
        test_compression_structured,
        test_compression_random,
        test_parameter_vector_shape,
        test_parameter_distance_self,
        test_parameter_distance_different,
        test_summary,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"result: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("all tests pass.")
    return failed


if __name__ == "__main__":
    sys.exit(run_all())
