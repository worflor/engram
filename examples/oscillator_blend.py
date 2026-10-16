"""A checked generation view; measured K/G means remain unchanged."""

import json
import numpy as np
from engram_analysis import Signature
from engram_network import WellNetwork


if __name__ == "__main__":
    root = 0.995 + 0.01j
    network = WellNetwork(2)
    for name, value in [("left", root), ("right", root.conjugate())]:
        network.learn_signature(
            Signature(
                np.array([2 * value]),
                np.array([value**2]),
                np.zeros(1),
                np.arange(2),
                20,
                2,
            ),
            well=name,
        )
    counts = network.memory._well_count.copy()
    k, g, _ = network.blend({"left": 1, "right": 1})
    mean_k = 2 * root.real
    mean_g = (root**2).real
    generated = network.dream(
        [[0, 0], [1, 0]], blend={"left": 1, "right": 1}, steps=128
    )
    bounded = network.dream(
        [[0, 0], [1, 0]], blend={"left": 1, "right": 1}, blend_radius=0.999, steps=32
    )
    assert counts == network.memory._well_count
    print(
        json.dumps(
            {
                "ordinary_mean_radius": float(max(abs(np.roots([1, -mean_k, mean_g])))),
                "reflection_mean_radius": float(max(abs(np.roots([1, -k[0], g[0]])))),
                "peak_generated_coordinate": float(np.max(abs(generated.trajectory))),
                "requested_root_radius": bounded.mixture_radius,
                "bounded_generation_radius": bounded.original_radius,
                "learned_well_counts_unchanged": counts == network.memory._well_count,
            },
            indent=2,
        )
    )
    print("Strict frozen roots do not bound transient gain or arbitrary switching.")
