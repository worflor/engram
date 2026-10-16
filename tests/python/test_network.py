"""Portable integration regressions distilled from the portfolio research paths."""

import sys
from pathlib import Path
import numpy as np
import pytest

from engram_codec import (
    Engram,
    EngramFile,
    Loop,
    PartnerLoop,
    EchoWorld,
    MirrorWorld,
    ByteHistogram,
    fit_all,
)
from engram_network import WellNetwork
from engram_training import (
    LookupSource,
    Trainer,
    TrainingItem,
    learn_byte_embedding,
    refine_byte_embedding,
)
from engram_retrieval import SignatureIndex, code_tokens
from engram_metrics import forecast_metrics, permutation_test

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from well_network import signal, experiment


def observations(n=12):
    return [signal(i % 3, np.random.default_rng(i)) for i in range(n)]


def test_unlabeled_network_held_out_and_dream_baseline():
    network, _, report = experiment()
    assert len(network.describe()) == 3
    assert report["accuracy"] > 0.95
    assert report["mean_forecast_rmse"] < report["mean_persistence_rmse"] * 0.15
    assert report["time_shuffled_accuracy"] < 0.5
    assert sum(network.memory._well_count.values()) == 120
    for node in network.describe():
        assert sum(node["transitions"].values()) == pytest.approx(1)


def test_overbroad_novelty_threshold_exposes_cluster_collapse():
    network, _, report = experiment(episodes=4, novelty_radius=2)
    assert len(network.describe()) == 1
    assert report["accuracy"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("seed", [2, 19, 123])
def test_noise_and_new_phases(seed):
    _, _, report = experiment(seed=seed, episodes=8, noise=0.02)
    assert report["accuracy"] >= 0.9
    assert report["mean_forecast_rmse"] < report["mean_persistence_rmse"]


def test_distance_and_unknown_activation():
    network = WellNetwork(4)
    x = observations(1)[0]
    assert network.activate(x) == []
    with pytest.raises(ValueError):
        network.dream(x)
    name = network.learn(x)
    a = network.activate(x)[0]
    assert a.well == name and a.distance == 0 and a.weight == 1
    assert network.learn(x) == name
    assert network.memory._well_count[name] == 2


def test_checkpoint_resume_preserves_next_learning_and_graph(tmp_path):
    network = WellNetwork(4, novelty_radius=0.09, temperature=0.03, residual_weight=0.5)
    items = observations()
    for x in items[:6]:
        network.learn(x)
    path = tmp_path / "network.engram"
    network.save(path)
    restored = WellNetwork.load(path)
    for x in items[6:]:
        assert restored.learn(x) == network.learn(x)
    assert restored.describe() == network.describe()
    np.testing.assert_array_equal(restored.memory._dream_Ks, network.memory._dream_Ks)
    for n in network.memory._well_count:
        np.testing.assert_array_equal(
            restored.memory.per_well_dream_predict(n),
            network.memory.per_well_dream_predict(n),
        )
    np.testing.assert_array_equal(
        restored.dream(items[-1], steps=30).trajectory,
        network.dream(items[-1], steps=30).trajectory,
    )


def test_sequence_boundaries_and_ordered_parallel_batches():
    items = [
        TrainingItem(str(i), x, sequence_start=i % 3 == 0)
        for i, x in enumerate(observations())
    ]
    a, b = WellNetwork(4), WellNetwork(4)
    ea = list(Trainer(a).fit(items, batch_size=1))
    eb = list(Trainer(b).fit(iter(items), batch_size=5, workers=3))
    assert ea == eb and a.describe() == b.describe()
    assert sum(sum(row.values()) for row in a.memory._well_transitions.values()) == 8


def test_failed_batch_does_not_partially_learn():
    n = WellNetwork(4)
    items = [
        TrainingItem("good", observations(1)[0]),
        TrainingItem("bad", np.full((4, 4), np.nan)),
    ]
    with pytest.raises(ValueError):
        list(Trainer(n).fit(items))
    assert n.memory._total_absorbed == 0
    assert not n.describe()


def test_short_observations_are_reported():
    network = WellNetwork(4)
    events = list(Trainer(network).fit([TrainingItem("short", np.zeros((2, 4)))]))
    assert events[0].skipped and events[0].well is None
    assert network.memory._total_absorbed == 0


def test_lazy_meta_fit_invalidated_and_crystallized():
    network = WellNetwork(4)
    for x in observations(6):
        network.learn(x, well="same")
    memory = network.memory
    meta = memory.per_well_meta("same")
    k, g, _, _ = fit_all(np.stack(memory._well_dream_Ks["same"]))
    np.testing.assert_array_equal(meta[0], k)
    np.testing.assert_array_equal(meta[1], g)
    expected = (
        k * memory._well_dream_Ks["same"][-1] - g * memory._well_dream_Ks["same"][-2]
    )
    np.testing.assert_allclose(memory.per_well_dream_predict("same"), expected)
    network.learn(observations(1)[0], well="same")
    assert not memory._well_meta_K
    memory.crystallize_dream(Engram(4))
    assert "same" in memory._well_meta_K
    with pytest.raises(ValueError):
        memory.crystallize_dream(Engram(2))


def test_well_recurrence_seeds_and_coordinate_permutation():
    n = WellNetwork(4, pairing=np.array([2, 0, 3, 1]))
    x = observations(1)[0]
    well = n.learn(x)
    k, g, _ = n.oscillator(well)
    z = x[-2:, n.pairing].astype(float)
    z = z[:, ::2] + 1j * z[:, 1::2]
    expected = k * z[-1] - g * z[-2]
    first = n.dream(x, steps=1).trajectory[0, n.pairing]
    np.testing.assert_allclose(first[::2] + 1j * first[1::2], expected, rtol=1e-6)
    assert n.memory._total_absorbed == 1  # dreaming is not training


def test_dream_graph_walk_repeatability_and_noise():
    n = WellNetwork(4)
    x = observations(1)[0]
    for name in ["a", "b", "c", "a", "b", "c", "a"]:
        n.learn(x, well=name)
    dream = n.dream(
        x, well="a", steps=12, walk_every=4, noise=1, rng=np.random.default_rng(4)
    )
    repeated = n.dream(
        x, well="a", steps=12, walk_every=4, noise=1, rng=np.random.default_rng(4)
    )
    assert dream.itinerary == ("a", "b", "c")
    np.testing.assert_array_equal(dream.trajectory, repeated.trajectory)
    assert not np.array_equal(
        dream.trajectory, n.dream(x, well="a", steps=12).trajectory
    )


def test_unstable_projection_reported_and_overflow_not_hidden():
    n = WellNetwork(2)
    x = np.ones((5, 2), np.float32)
    n.learn(x, well="unstable")
    n.memory._well_sum_K["unstable"][:] = 3
    n.memory._well_sum_G["unstable"][:] = 0
    result = n.dream(x, well="unstable", steps=50, max_radius=0.99)
    assert result.stabilized and result.original_radius == 3
    assert np.isfinite(result.trajectory).all()
    with pytest.raises(FloatingPointError):
        n.dream(x, well="unstable", steps=1000)


def test_consolidate_preserves_mass_edges_and_forgets_unordered_history(tmp_path):
    n = WellNetwork(4)
    x = observations(1)[0]
    for label in ["a", "b", "a", "b"]:
        n.learn(x, well=label)
    before = n.memory._well_sum_K["a"] + n.memory._well_sum_K["b"]
    mapping = n.consolidate(0)
    assert mapping == {"a": "a", "b": "a"}
    assert n.memory._well_count == {"a": 4}
    np.testing.assert_array_equal(n.memory._well_sum_K["a"], before)
    assert n.memory._well_transitions == {"a": {"a": 3}}
    assert n.memory.per_well_meta("a") is None
    assert n.memory._last_well == "a"
    n.save(tmp_path / "merged.engram")
    assert WellNetwork.load(tmp_path / "merged.engram").describe() == n.describe()


def test_old_artifact_does_not_invent_full_wells():
    m = EngramFile.create("", dim=4)
    m._well_count["old"] = 1
    m._well_sum_K["old"] = np.ones(2, complex)
    n = WellNetwork(4, memory=m)
    with pytest.raises(ValueError, match="lacks statistics"):
        n.activate(observations(1)[0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 0},
        {"noise": -1},
        {"max_radius": 2},
        {"walk_every": 0},
        {"steps": 100000000},
    ],
)
def test_dream_rejects_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        WellNetwork(4).dream(observations(1)[0], **kwargs)


def test_lookup_source_bounds_smoothing_and_identity(tmp_path):
    matrix = np.arange(32, dtype=np.float32).reshape(8, 4)
    source = LookupSource(matrix, lambda ids: ids, window=2)
    np.testing.assert_array_equal(
        source.project([0, 1, 2]), [[2, 3, 4, 5], [6, 7, 8, 9]]
    )
    assert source.project([]).shape == (0, 4)
    for ids in [[-1], [8], [1.5], [[1]]]:
        with pytest.raises(ValueError):
            source.project(ids)
    n = WellNetwork(4)
    list(Trainer(n, source).fit([TrainingItem("x", [0, 1, 2, 3, 4])]))
    n.save(tmp_path / "checkpoint.engram")
    restored = WellNetwork.load(tmp_path / "checkpoint.engram")
    Trainer(restored, source)
    with pytest.raises(ValueError, match="representation differs"):
        Trainer(restored, LookupSource(matrix + 1, lambda ids: ids, window=2))


def test_byte_embedding_counts_document_boundaries_and_unseen_rows():
    embedding = learn_byte_embedding([b"abababab", b"cdcdcdcd"], dim=4, window=1)
    assert embedding.shape == (256, 4)
    assert not embedding[0].any()
    np.testing.assert_array_equal(
        embedding, learn_byte_embedding([b"abababab", b"cdcdcdcd"], dim=4, window=1)
    )
    # Each length-one document has no adjacency, including at the boundary.
    with pytest.raises(ValueError, match="no co-occurrence"):
        learn_byte_embedding([b"a", b"b"])


def test_byte_training_refinement_prior_and_checkpoint(tmp_path):
    docs = ["the river flows " * 12, "for key in values: print(key)\n" * 12]
    matrix = learn_byte_embedding(docs, dim=8)
    source = LookupSource.bytes(matrix, window=4)
    n = WellNetwork(8)
    list(
        Trainer(n, source).fit(
            [TrainingItem(str(i), text) for i, text in enumerate(docs)]
        )
    )
    assert sum(h.sum() for h in n.memory._well_sum_raw.values()) == sum(
        len(d) for d in docs
    )
    new = refine_byte_embedding(n.memory, matrix)
    np.testing.assert_array_equal(new[0], matrix[0])  # unseen retains prior exactly
    assert np.isfinite(new).all() and not np.array_equal(new, matrix)
    n.save(tmp_path / "byte.engram")
    restored = WellNetwork.load(tmp_path / "byte.engram")
    np.testing.assert_array_equal(new, refine_byte_embedding(restored.memory, matrix))


def test_retrieval_relationships_and_identifier_controls():
    assert code_tokens("HTTPServer2 snake_case readJSONValue") == [
        "http",
        "server",
        "snake",
        "case",
        "read",
        "json",
        "value",
    ]
    index = SignatureIndex(4)
    xs = observations(3)
    for i, x in enumerate(xs):
        index.add(str(i), x, identifiers=[str(i)])
    hits = index.search(signal(2, np.random.default_rng(99)))
    assert hits[0].key == "2"
    lexical = index.search(xs[2], identifiers=["0"], lexical_weight=1)
    assert lexical[0].key == "0" and lexical[0].identifier_overlap == 1
    assert SignatureIndex(4).search(xs[0]) == []


def test_offset_capture_control_and_negative_r2():
    actual = np.array([[100, 100], [101, 99], [99, 101]], float)
    predicted = np.full_like(actual, 100)
    m = forecast_metrics(actual, predicted, last_observation=actual[0])
    assert m["energy_capture"] > 0.999
    assert m["centered_r2"] == 0
    bad = forecast_metrics(actual, predicted + 2, last_observation=actual[0])
    assert bad["centered_r2"] < 0
    assert (
        forecast_metrics(predicted, predicted, last_observation=predicted[0])[
            "centered_r2"
        ]
        is None
    )


def test_permutation_null_and_invalid_constant():
    x = np.arange(30)
    result = permutation_test(x, x, rng=np.random.default_rng(3), repeats=99)
    assert result["p_value"] == 0.01
    assert abs(result["null_mean"]) < 0.1
    with pytest.raises(ValueError):
        permutation_test(x, np.ones(30), rng=np.random.default_rng(3))


@pytest.mark.parametrize("world", [MirrorWorld(), EchoWorld(0)])
def test_feedback_uses_shared_basis_and_bounded_memory(world):
    loop = Loop(Engram(4), world, capacity=3, rng=np.random.default_rng(12))
    for x in observations(3):
        loop.seed(x)
    loop.run(4)
    assert len(loop.pool) == 3
    assert all(np.array_equal(packet.pairing, loop.pairing) for packet in loop.pool)
    assert np.isfinite(loop.prediction_error_trajectory).all()


def test_partner_repeatability_and_invalid_world():
    def run():
        partner = PartnerLoop(Engram(4), rng=np.random.default_rng(3), capacity=2)
        partner.seed(observations(1)[0])
        loop = Loop(Engram(4), partner, rng=np.random.default_rng(4), capacity=2)
        loop.seed(observations(2)[1])
        return loop.run(4), partner

    a, pa = run()
    b, pb = run()
    assert a == b and len(pa.pool) == len(pb.pool) == 2

    class Invalid:
        def respond(self, action):
            return np.full((4, 4), np.nan)

    with pytest.raises(ValueError):
        Loop(Engram(4), Invalid()).step()
    with pytest.raises(ValueError):
        ByteHistogram(0)


def test_text_short_message_never_reuses_previous_composition():

    memory = EngramFile.create("", dim=256)
    text = (
        "A current flows through the winding river. " * 80
        + "for key in table: update(key)\n" * 80
    )
    well = memory.absorb_text(text, source="text")
    assert well == "text"
    before = memory._well_sum_raw[well].copy()
    count = memory._well_count[well]
    assert before.sum() == pytest.approx(count)
    assert memory.absorb_text("short", source="text") == "unknown"
    np.testing.assert_array_equal(memory._well_sum_raw[well], before)
    assert memory._well_count[well] == count
    with pytest.raises(ValueError):
        EngramFile.create("", dim=4).measure_text(text)


def test_batched_measurement_matches_packet_measurement():
    memory = EngramFile.create("", dim=4)
    codec = Engram(4)
    for x in observations(30):
        memory.absorb(codec.encode(x, pairing=np.arange(4)), source="learned")
    x = observations(1)[0]
    measured = memory.measure(codec, x)
    from engram_codec import Mode

    blocks = [
        b for b in measured["packet"].blocks if b.mode in (Mode.CASCADED, Mode.LINEAR)
    ]
    n = len(blocks)
    summed = memory.measure_from_sums(
        np.array([sum(b.macro_K for b in blocks)]),
        np.array([sum(b.macro_G for b in blocks)]),
        np.array([sum(b.pair_rms for b in blocks)]),
        np.array([n]),
    )[0]
    for key in [
        "drift",
        "nearest_well",
        "well_distance",
        "well_profile",
        "well_attention",
        "dream_state",
        "dream_surprise",
    ]:
        assert summed[key] == measured[key]
    with pytest.raises(ValueError):
        memory.measure_from_sums(
            np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), np.array([0])
        )
    assert np.isfinite(memory.fingerprint()).all()


def test_projection_calibration_and_empty_input():
    from engram_codec import TanhProjection

    projection = TanhProjection(np.eye(4, dtype=np.float32))
    assert projection.project(np.empty((0, 4))).shape == (0, 4)
    with pytest.raises(RuntimeError):
        _ = projection.alpha
    x = np.ones((10, 4), np.float32) * 2
    np.testing.assert_allclose(projection.project(x), 0.5, atol=1e-6)
    with pytest.raises(ValueError):
        projection.project(np.ones((3, 2)))
    with pytest.raises(ValueError):
        TanhProjection(np.ones((3, 4)))
    extreme = TanhProjection(np.full((4, 4), 1e38, np.float32))
    assert np.isfinite(extreme.project(np.full((5, 4), 1e38, np.float32))).all()


def test_bad_accelerator_cannot_partially_update_memory():
    class Broken:
        def fit(self, *args, **kwargs):
            return []

    network = WellNetwork(4)
    with pytest.raises(ValueError, match="number of signatures"):
        list(
            Trainer(network).fit(
                [TrainingItem("x", observations(1)[0])], fitter=Broken()
            )
        )
    assert network.memory._total_absorbed == 0
