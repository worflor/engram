from pathlib import Path
import zipfile
import numpy as np
import pytest
from engram_codec import (ByteHistogram, Engram, EngramFile, Mode, apply_pairing,
                          undo_pairing, fit_pair, to_wire, from_wire)


@pytest.mark.parametrize("length", [0, 1, 2, 3, 4, 32, 129])
def test_short_and_predictive_trajectories(length):
    x = np.stack([np.arange(length) * .01 + d * .1 for d in range(4)], axis=1).astype(np.float32)
    codec = Engram(4)
    packet = codec.encode(x)
    assert packet.length == length
    y = codec.decode(packet)
    np.testing.assert_allclose(y, x, atol=0.002)
    if length:
        restored = from_wire(to_wire(packet))
        np.testing.assert_allclose(codec.decode(restored), x, atol=0.01)


def test_pairing_is_a_reversible_coordinate_change():
    x = np.arange(48, dtype=np.float32).reshape(8, 6)
    pairing = np.array([4, 1, 5, 0, 3, 2], dtype=np.int32)
    np.testing.assert_array_equal(undo_pairing(apply_pairing(x, pairing), pairing), x)


def test_fit_recovers_an_oscillator_recurrence():
    k, g = 1.5 + .1j, .65 - .05j
    z = np.zeros(64, dtype=np.complex128)
    z[:2] = [1, .7 + .2j]
    for i in range(2, len(z)):
        z[i] = k*z[i-1] - g*z[i-2]
    fitted_k, fitted_g, _ = fit_pair(z)
    prediction = fitted_k*z[1:-1] - fitted_g*z[:-2]
    # Ridge regularization deliberately biases the least-squares optimum.
    fitted_rms = np.sqrt(np.mean(np.abs(prediction - z[2:])**2))
    linear_rms = np.sqrt(np.mean(np.abs(2*z[1:-1] - z[:-2] - z[2:])**2))
    assert fitted_rms < linear_rms * .1


def test_full_wire_reconstructs_but_compact_discards_residuals():
    rng = np.random.default_rng(710)
    x = np.sin(np.arange(128)[:, None]*.05 + np.arange(8)) + rng.normal(0, .01, (128, 8))
    x = x.astype(np.float32)
    codec = Engram(8)
    packet = codec.encode(x)
    full = to_wire(packet)
    assert full[:3] == b"EN\x04"
    compact = to_wire(packet, compact=True)
    assert len(compact) < len(full)
    assert compact[:3] == b"EN\x04"
    restored = from_wire(full)
    assert np.sqrt(np.mean((codec.decode(restored)-x)**2)) < .2
    small = from_wire(compact)
    assert len(small.blocks) == len(packet.blocks)
    for a, b in zip(packet.blocks, small.blocks):
        np.testing.assert_allclose(a.macro_K, b.macro_K, atol=.002)


def test_text_histogram_and_brain_persistence(tmp_path):
    text = (Path(__file__).parents[1] / "fixtures/long_form.txt").read_text(encoding="utf-8")
    trajectory = ByteHistogram().text_to_trajectory(text)
    assert trajectory.shape[0] >= 4
    np.testing.assert_allclose(trajectory.sum(axis=1), 1, atol=1e-6)
    path = str(tmp_path / "test.engram")
    brain = EngramFile.create(path, name="fixture", dim=256,
                              metadata={"purpose": "test"}, annotations={"voice": "fixture"})
    codec = Engram(256)
    packet = codec.encode(trajectory)
    well = brain.absorb(packet, source="fixture")
    assert well == "fixture"
    before = brain._well_sum_K[well].copy()
    count = brain._well_count[well]
    brain.save()
    assert zipfile.is_zipfile(path)
    loaded = EngramFile.load(path)
    assert loaded.name == "fixture"
    assert loaded.metadata["purpose"] == "test"
    assert loaded.annotations == {"voice": "fixture"}
    assert loaded._well_count[well] == count
    np.testing.assert_array_equal(loaded._well_sum_K[well], before)
    measurement = loaded.measure(codec, trajectory)
    assert np.isfinite(measurement["drift"])
    assert np.isfinite(measurement["texture"])


def test_odd_dimensions_are_rejected():
    with pytest.raises(ValueError, match="even"):
        Engram(3)
