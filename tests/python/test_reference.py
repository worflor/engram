import copy
import json
from pathlib import Path
import struct
import zipfile
import numpy as np
import pytest
from engram_codec import (
    Engram,
    EngramFile,
    Signature,
    EmbeddingTable,
    StreamEncoder,
    StreamDecoder,
    derive_pairing,
    encode_block,
    decode_block,
    to_wire,
    from_wire,
    dynamics,
    fit_ar1,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"


@pytest.mark.parametrize("length", [3, 7, 11, 13, 19, 31])
def test_micro_partition_reconstructs_partial_tail(length):
    rng = np.random.default_rng(105 + length)
    x = rng.normal(0, 0.3, (length, 8)).astype(np.float32)
    b = encode_block(x, np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32), 4)
    np.testing.assert_allclose(decode_block(b), x, atol=2e-6)


def test_constant_jump_cannot_hold_an_unrelated_seed():
    x = np.full((12, 4), 0.5, dtype=np.float32)
    b = encode_block(x, np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32), 4)
    np.testing.assert_allclose(decode_block(b), x, atol=2e-6)


@pytest.mark.parametrize("dim", [2, 8, 256])
def test_zero_correlation_channels_still_form_a_permutation(dim):
    x = np.zeros((8, dim), dtype=np.float32)
    x[:, 0] = np.arange(8) * 0.1
    pairing = derive_pairing(x)
    np.testing.assert_array_equal(np.sort(pairing), np.arange(dim))
    p = Engram(dim).encode(x)
    np.testing.assert_allclose(Engram(dim).decode(from_wire(to_wire(p))), x, atol=0.003)


def test_full_packet_saves_actual_block_state_and_compact_saves_basis():
    x = np.sin(np.arange(128)[:, None] * 0.07 + np.arange(8)).astype(np.float32)
    basis = np.array([3, 1, 6, 7, 5, 2, 0, 4], dtype=np.int32)
    c = Engram(8)
    p = c.encode(x, pairing=basis)
    q = from_wire(to_wire(p))
    for a, b in zip(p.blocks, q.blocks):
        assert (a.start, a.length, a.micro_size) == (b.start, b.length, b.micro_size)
        np.testing.assert_array_equal(a.seed1, b.seed1)
        np.testing.assert_array_equal(a.seed2, b.seed2)
    assert np.max(np.abs(c.decode(q) - x)) < 0.01
    compact = from_wire(to_wire(p, compact=True))
    np.testing.assert_array_equal(compact.pairing, basis)
    with pytest.raises(ValueError, match="compact"):
        c.decode(compact)


def test_checked_reader_rejects_all_truncated_prefixes_and_trailing_data():
    p = Engram(2).encode(np.arange(32, dtype=np.float32).reshape(16, 2) * 0.01)
    for compact in [False, True]:
        wire = to_wire(p, compact=compact)
        for end in range(len(wire)):
            with pytest.raises(ValueError):
                from_wire(wire[:end])
        with pytest.raises(ValueError, match="trailing"):
            from_wire(wire + b"x")
    with pytest.raises(ValueError):
        from_wire(
            b"EN\x04" + struct.pack("<BHI fIII", 0, 4096, 0xFFFFFFFF, 0.0, 4, 4, 1)
        )


@pytest.mark.parametrize("kind", ["nan", "odd", "shape", "pairing"])
def test_input_contract_rejects_invalid_trajectories(kind):
    with pytest.raises(ValueError):
        if kind == "odd":
            Engram(3)
        elif kind == "nan":
            Engram(2).encode(np.full((4, 2), np.nan))
        elif kind == "shape":
            Engram(2).encode(np.zeros((4, 4)))
        else:
            Engram(2).encode(np.zeros((4, 2)), pairing=np.array([0, 0]))


def test_signatures_match_actual_dart_and_java_kernel_receipts():
    fixture = json.loads((FIXTURES / "source-kernels.json").read_text())
    for case in fixture["cases"]:
        x = np.array(case["samples"], dtype=np.float32).reshape(
            case["length"], case["dim"]
        )
        signature = Signature.from_trajectory(x, pairing=np.arange(case["dim"]))
        for language, result in case["implementations"].items():
            tolerance = 2e-3 if language == "java" else 1e-8
            np.testing.assert_allclose(
                np.column_stack([signature.k.real, signature.k.imag]),
                result["k"],
                atol=tolerance,
                rtol=1e-6,
            )
            np.testing.assert_allclose(
                np.column_stack([signature.g.real, signature.g.imag]),
                result["g"],
                atol=tolerance,
                rtol=1e-6,
            )


def test_spectral_diagnostics_distinguish_repeated_unit_roots():
    assert dynamics(5 / 4, 3 / 8).classification == "decaying"
    assert dynamics(5 / 2, 1).spectral_radius == 2
    linear = dynamics(2, 1)
    assert (
        linear.repeated_root
        and linear.classification == "unit-radius"
        and linear.half_life_samples is None
    )
    assert dynamics(0, 0.25).half_life_samples == 1
    assert dynamics(0, 0.25).oscillation_period_samples == 4
    k, rms = fit_ar1([1, 2])
    assert abs(k - 2 / (1 + 1e-4)) < 1e-12 and rms > 0


def test_signature_comparison_rejects_different_bases():
    x = np.sin(np.arange(32)[:, None] * 0.1 + np.arange(4)).astype(np.float32)
    a = Signature.from_trajectory(x, pairing=np.arange(4))
    assert a.compare(a)["kg_rms_distance"] == 0
    with pytest.raises(ValueError, match="pairing"):
        a.compare(Signature.from_trajectory(x, pairing=[1, 0, 2, 3]))


@pytest.mark.parametrize("length", [1, 2, 3, 17, 97])
def test_stream_flush_retains_seeds_and_every_sample(length):
    x = np.sin(np.arange(length)[:, None] * 0.15 + np.arange(4)).astype(np.float32)
    encoder, decoder = StreamEncoder(4), StreamDecoder()
    blocks = [b for row in x if (b := encoder.push(row)) is not None]
    if (tail := encoder.flush()) is not None:
        blocks.append(tail)
    for b in blocks:
        decoder.push(b)
    assert decoder.trajectory.shape == x.shape
    np.testing.assert_allclose(decoder.trajectory, x, atol=1e-5)


def test_single_packet_flat_and_batch_memory_updates_are_identical(tmp_path):
    rng = np.random.default_rng(718)
    k = rng.normal(size=(12, 2)) + 1j * rng.normal(size=(12, 2))
    g = rng.normal(size=(12, 2)) + 1j * rng.normal(size=(12, 2))
    rms = np.abs(rng.normal(size=(12, 2)))
    a = EngramFile.create(str(tmp_path / "a.engram"), dim=4)
    b = a.copy()
    a.absorb_batch_flat(k, g, rms, [0, 3, 7, 12], source="signal")
    for start, end in [(0, 3), (3, 7), (7, 12)]:
        b.absorb_flat(k[start:end], g[start:end], rms[start:end], source="signal")
    np.testing.assert_array_equal(a._well_sum_G["signal"], b._well_sum_G["signal"])
    np.testing.assert_array_equal(a._well_sum_S["signal"], b._well_sum_S["signal"])
    assert a._well_transitions == b._well_transitions == {"signal": {"signal": 2}}
    assert len(a._well_dream_Ks["signal"]) == 3
    a.save()
    restored = EngramFile.load(str(tmp_path / "a.engram"))
    for memory in [a, restored]:
        memory.absorb_flat(k[:3], g[:3], rms[:3], source="signal")
    assert restored._well_transitions == a._well_transitions
    np.testing.assert_array_equal(
        restored._well_anchor_G["signal"], a._well_anchor_G["signal"]
    )


def test_packet_absorption_retains_g_and_rms_and_uses_one_basis(tmp_path):
    c = Engram(4)
    x = np.sin(np.arange(96)[:, None] * 0.07 + np.arange(4)).astype(np.float32)
    p = c.encode(x, pairing=np.arange(4))
    memory = EngramFile.create(str(tmp_path / "m.engram"), dim=4)
    memory.absorb(p, source="signal")
    assert np.linalg.norm(memory._well_sum_G["signal"]) > 0
    assert len(memory._well_dream_Ks["signal"]) == 1
    q = c.encode(x, pairing=np.array([1, 0, 2, 3]))
    with pytest.raises(ValueError, match="pairing"):
        memory.absorb(q)


def test_seeded_generation_uses_independent_explicit_random_source():
    c = Engram(4)
    t = np.arange(96)[:, None]
    d = np.arange(4)
    packets = [
        c.encode(np.sin(t * f + d).astype(np.float32), pairing=np.arange(4))
        for f in [0.05, 0.06, 0.07, 0.08]
    ]
    a = c.dream(packets, n_blocks=3, samples_per_block=8, rng=np.random.default_rng(81))
    b = c.dream(packets, n_blocks=3, samples_per_block=8, rng=np.random.default_rng(81))
    np.testing.assert_array_equal(a, b)


def test_glv_adapter_and_short_signature():
    vectors = np.array([[100, 200], [300, 400], [500, 600]], dtype="<i2")
    payload = (
        b"GLV1"
        + struct.pack("<IIIf", 1, 3, 2, 6.0)
        + b"\x01a\x01b\x01c"
        + vectors.tobytes()
    )
    table = EmbeddingTable.from_glv(payload)
    rows = table.project(["a", "unknown", "b"])
    assert rows.shape == (2, 2)
    signature = table.signature(["a", "b"], pairing=np.arange(2))
    assert signature.order == 1
    with pytest.raises(ValueError):
        EmbeddingTable.from_glv(payload[:-1])


def test_memory_preserves_unknown_fields_and_sections(tmp_path):
    m = EngramFile.create(str(tmp_path / "m.engram"), dim=4)
    m._source_manifest = {"research-note": {"counter": 3}, "physics": {"future": True}}
    m._extra_sections = {"extra.json": b'{"note":"preserved"}'}
    m.save()
    loaded = EngramFile.load(str(tmp_path / "m.engram"))
    loaded.save(str(tmp_path / "copy.engram"))
    with zipfile.ZipFile(tmp_path / "copy.engram") as z:
        manifest = json.loads(z.read("manifest.json"))
        assert (
            manifest["research-note"] == {"counter": 3}
            and manifest["physics"]["future"] is True
        )
        assert z.read("extra.json") == b'{"note":"preserved"}'


def test_alexandria_frozen_artifact_stays_loadable():
    p = Path(__file__).parents[2] / "research/artifacts/alexandria.engram"
    m = EngramFile.load(p)
    assert m.name == "Alexandria" and m.dim == 300 and len(m.wells()) == 225
    assert "per-well G/S" in m.import_info["unavailable"]


def test_legacy_frozen_packets_remain_readable():
    full = from_wire((FIXTURES / "legacy-full-v3.en").read_bytes())
    compact = from_wire((FIXTURES / "legacy-compact-v2.en").read_bytes())
    assert full.length == compact.length == 64 and full.dim == compact.dim == 4
    x = (np.arange(64)[:, None] * 0.01 + np.arange(4)[None, :] * 0.1).astype(np.float32)
    np.testing.assert_allclose(Engram(4).decode(full), x, atol=0.01)
    assert not compact.reconstructable
    with pytest.raises(ValueError):
        from_wire((FIXTURES / "legacy-full-v3.en").read_bytes()[:-1])


def test_explicit_error_budget_is_checked_after_wire_replay():
    x = np.sin(
        np.arange(128)[:, None] * np.where(np.arange(128)[:, None] < 64, 0.06, 0.23)
        + np.arange(16)
    ).astype(np.float32)
    for budget in [0.0, 0.001, 0.01]:
        codec = Engram(16, max_error=budget)
        decoded = codec.decode(from_wire(to_wire(codec.encode(x))))
        assert np.max(np.abs(decoded.astype(np.float64) - x)) <= budget


def test_snapshot_preserves_full_precision_history_and_empty_basis(tmp_path):
    path = tmp_path / "state.engram"
    memory = EngramFile.create(str(path), dim=4)
    memory.save()
    empty = EngramFile.load(path)
    assert empty._reference_pairing is None
    for f in [0.05, 0.06, 0.07, 0.08]:
        memory.fast_absorb(
            np.sin(np.arange(32)[:, None] * f + np.arange(4)).astype(np.float32),
            source="signal",
        )
    memory.save()
    loaded = EngramFile.load(path)
    np.testing.assert_array_equal(memory._dream_Ks, loaded._dream_Ks)
    np.testing.assert_array_equal(memory._dream_Gs, loaded._dream_Gs)
    np.testing.assert_array_equal(memory._dream_Ss, loaded._dream_Ss)
    assert (
        loaded._last_well == "signal"
        and loaded._well_transitions == memory._well_transitions
    )


def test_merge_retains_initial_rows_state_and_explicit_overlap():
    from engram_codec import merge

    c = Engram(4)
    x = np.sin(np.arange(96)[:, None] * 0.07 + np.arange(4)).astype(np.float32)
    a, b = (
        c.encode(x[:32], pairing=np.arange(4)),
        c.encode(x[32:], pairing=np.arange(4)),
    )
    combined = merge(a, b)
    np.testing.assert_allclose(c.decode(combined), x, atol=1e-6)
    np.testing.assert_allclose(c.decode(from_wire(to_wire(combined))), x, atol=0.01)
    overlap = c.encode(x[30:], pairing=np.arange(4))
    np.testing.assert_allclose(
        c.decode(merge(a, overlap, overlap=2, overlap_tolerance=1e-6)), x, atol=1e-6
    )
    with pytest.raises(ValueError):
        merge(a, b, overlap=2)


def test_eigenvalue_blend_retains_exact_endpoints_and_zero_identity():
    from engram_codec import blend_eigenvalues

    k, g = np.array([0j, 1.5]), np.array([0j, 0.5])
    for t in [0, 0.5, 1]:
        a, b = blend_eigenvalues(k, g, k, g, t)
        np.testing.assert_array_equal(a, k)
        np.testing.assert_array_equal(b, g)
    with pytest.raises(ValueError):
        blend_eigenvalues(k, g, k, g, 2)


def test_legacy_compact_basis_must_be_supplied_before_reuse():
    packet = from_wire((FIXTURES / "legacy-compact-v2.en").read_bytes())
    assert not packet.basis_known
    with pytest.raises(ValueError, match="basis"):
        to_wire(packet, compact=True)
    attached = packet.with_pairing(np.arange(4))
    assert from_wire(to_wire(attached, compact=True)).basis_known


def test_memory_history_packets_generate_without_claiming_reconstruction(tmp_path):
    memory = EngramFile.create(str(tmp_path / "m.engram"), dim=4)
    assert memory.fingerprint().shape == (22,)
    for f in [0.05, 0.06, 0.07, 0.08]:
        memory.fast_absorb(
            np.sin(np.arange(32)[:, None] * f + np.arange(4)).astype(np.float32),
            source="signal",
        )
    packets = memory.dream_packets
    assert not packets[0].reconstructable and memory.fingerprint().shape == (22,)
    generated = Engram(4).dream(
        packets, n_blocks=2, samples_per_block=4, rng=np.random.default_rng(7)
    )
    assert generated.shape == (8, 4) and np.isfinite(generated).all()


def test_spectral_population_counts_do_not_overlap():
    from engram_codec import spectral_summary

    result = spectral_summary(
        np.array([1.9, 2.0, 2.1]), np.array([0.95**2, 1.0, 1.05**2])
    )
    assert result["n_decaying"] + result["n_sustained"] + result["n_divergent"] == 6
    assert spectral_summary([], [])["mag_mean"] == 0


def test_native_and_manifold_memories_import_without_inventing_missing_fields(tmp_path):
    k = np.array([1.25 + 0.1j, 0.5 - 0.1j], dtype="<c16")
    name = b"fixture"
    name_field = struct.pack("<H", len(name)) + name
    enbr = (
        b"ENBR\x01"
        + struct.pack("<IIfII", 4, 2, 0.005, 3, 1)
        + name_field
        + b"\x01"
        + np.arange(4, dtype="<i4").tobytes()
    )
    enbr += (
        struct.pack("<I", 1)
        + name_field
        + struct.pack("<I", 3)
        + k.tobytes()
        + struct.pack("<I", 0)
    )
    memory = EngramFile.load(enbr)
    np.testing.assert_array_equal(memory._well_sum_K["fixture"], k)
    assert memory.import_info["format"] == "ENBR"
    memory.save(str(tmp_path / "imported.engram"))
    np.testing.assert_array_equal(
        EngramFile.load(tmp_path / "imported.engram")._well_sum_K["fixture"], k
    )
    manifest = json.dumps(
        {
            "format": "engram",
            "version": 4,
            "identity": {"name": "Alexandria-fixture"},
            "physics": {"dim": 4},
            "state": {"n_wells": 1, "total_absorbed": 3},
        }
    ).encode()
    endb = (
        b"ENDB"
        + struct.pack("<IIIII", 2, 4, 2, 1, 9)
        + np.arange(4, dtype="<i2").tobytes()
    )
    endb += (
        name_field
        + struct.pack("<I", 3)
        + k.tobytes()
        + struct.pack("<I", 2048)
        + bytes(2048)
        + struct.pack("<I", len(manifest))
        + manifest
    )
    imported = EngramFile.load(endb)
    assert (
        imported.name == "Alexandria-fixture"
        and imported.import_info["format"] == "ENDB"
    )
    np.testing.assert_array_equal(imported._well_sum_K["fixture"], k)
    for payload in [enbr, endb]:
        for end in range(len(payload)):
            with pytest.raises(ValueError):
                EngramFile.load(payload[:end])


def test_ambiguous_v1_requires_explicit_producer_and_large_samples_can_fall_back():
    wire = bytearray((FIXTURES / "legacy-full-v3.en").read_bytes())
    wire[2] = 1
    with pytest.raises(ValueError, match="ambiguous"):
        from_wire(wire)
    assert from_wire(wire, legacy_v1="python").length == 64
    extreme = np.finfo(np.float32).max
    x = np.array([[extreme, 1], [-extreme, -1]] * 8, dtype=np.float32)
    with np.errstate(all="ignore"):
        c = Engram(2, max_error=0)
        p = c.encode(x)
        np.testing.assert_array_equal(c.decode(from_wire(to_wire(p))), x)
