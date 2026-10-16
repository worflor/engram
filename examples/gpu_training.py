"""Optional real-device fitting with the SAME ordered reference memory updates."""

import json
import numpy as np
from engram_acceleration import GpuFitter
from engram_network import WellNetwork
from engram_training import Trainer, TrainingItem
from well_network import signal


def run():
    gpu = GpuFitter()
    trajectories = [signal(i % 3, np.random.default_rng(i)) for i in range(24)]
    # Include zero, constant and minimum-length cases.
    trajectories += [np.zeros((3, 4), np.float32), np.ones((8, 4), np.float32)]
    cpu = WellNetwork(4)
    signatures = gpu.fit(trajectories, dim=4, pairing=cpu.pairing)
    errors = []
    for x, accelerated in zip(trajectories, signatures):
        reference = cpu.signature(x)
        errors.append(
            max(
                float(
                    np.max(abs(getattr(reference, field) - getattr(accelerated, field)))
                )
                for field in ["k", "g", "pair_rms"]
            )
        )
    # Float32 normal equations are less accurate than the float64 reference,
    # especially near collinear modes. This is a measured finite-case tolerance.
    assert max(errors) < 0.01, max(errors)
    items = [
        TrainingItem(str(i), x, sequence_start=i % 6 == 0)
        for i, x in enumerate(trajectories[:24])
    ]
    reference, accelerated = WellNetwork(4), WellNetwork(4)
    a = list(Trainer(reference).fit(items))
    b = list(Trainer(accelerated).fit(items, fitter=gpu))
    assert a == b
    assert reference.memory._well_transitions == accelerated.memory._well_transitions
    for key in reference.memory._well_sum_K:
        np.testing.assert_allclose(
            reference.memory._well_sum_K[key],
            accelerated.memory._well_sum_K[key],
            atol=0.01,
        )
    dimension_errors = {}
    for dim in [2, 8, 256]:
        rng = np.random.default_rng(dim)
        batch = [
            rng.normal(0, 0.1, (length, dim)).astype(np.float32)
            for length in [3, 29, 65]
        ]
        pairing = rng.permutation(dim).astype(np.int32)
        network = WellNetwork(dim, pairing=pairing)
        fitted = gpu.fit(batch, dim=dim, pairing=pairing)
        delta = max(
            network.signature(x).compare(s)["kg_rms_distance"]
            for x, s in zip(batch, fitted)
        )
        assert delta < 0.01, (dim, delta)
        dimension_errors[str(dim)] = delta
    assert gpu.fit([], dim=4) == []
    for invalid in [np.zeros((2, 4)), np.full((3, 4), np.nan)]:
        try:
            gpu.fit([invalid], dim=4)
        except ValueError:
            pass
        else:
            raise AssertionError("GPU accepted invalid input")
    print(
        json.dumps(
            {
                "adapter": gpu.adapter_info,
                "trajectories": len(trajectories),
                "maximum_coefficient_or_rms_error": max(errors),
                "ordered_training_assignments": len(a),
                "additional_dimension_pairing_cases": 9,
                "dimension_kg_rms_errors": dimension_errors,
            },
            indent=2,
        )
    )
    return gpu, max(errors)


if __name__ == "__main__":
    run()
