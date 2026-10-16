"""Forecast translated motion with an explicit, unpenalized affine origin."""

import numpy as np

from engram_affine import AffineOscillator
from engram_codec import fit_all, predict_all


def demonstration():
    t = np.arange(128)
    z = np.exp(0.17j * t) + 0.35 * np.exp(-0.11j * t + 0.4j)
    motion = np.column_stack([z.real, z.imag])
    offset = np.array([12.0, -8.0])
    observed = motion[:96] + offset
    expected = motion[96:] + offset
    fitted = AffineOscillator.fit(observed)
    predicted = fitted.forecast(32)
    complex_context = observed[:, :1] + 1j * observed[:, 1:]
    k, g, _, _ = fit_all(complex_context)
    raw = predict_all(complex_context[-1], complex_context[-2], k, g, 32)[:, 0]
    homogeneous = np.column_stack([raw.real, raw.imag])
    report = {
        "observed_samples": 96,
        "future_samples": 32,
        "homogeneous_rmse": float(np.sqrt(np.mean((homogeneous - expected) ** 2))),
        "affine_rmse": float(np.sqrt(np.mean((predicted - expected) ** 2))),
        "translation_defect_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        predicted
                        - offset
                        - AffineOscillator.fit(motion[:96]).forecast(32)
                    )
                    ** 2
                )
            )
        ),
    }
    return report, motion, offset, homogeneous, predicted


if __name__ == "__main__":
    import json

    report, *_ = demonstration()
    print(json.dumps(report, indent=2))
    print(
        "The origin is fitted explicitly; arbitrary changes of motion or unstable roots remain separate problems."
    )
