"""A small parameter-history generation experiment, entirely offline."""
import numpy as np
from engram_codec import Engram

codec = Engram(dim=4)
pairing = np.arange(4, dtype=np.int32)
t = np.arange(96)[:, None]
coordinates = np.arange(4)[None, :]
packets = [codec.encode(np.sin(t*frequency+coordinates).astype(np.float32), pairing=pairing)
           for frequency in [.05, .06, .07, .08]]
generated = codec.dream(packets, n_blocks=3, samples_per_block=8, rng=np.random.default_rng(20261015))
if generated is None:
    raise SystemExit("This input did not produce enough usable parameter history.")
assert generated.shape == (24, 4)
assert np.isfinite(generated).all()
print(f"Generated {generated.shape[0]} rows in {generated.shape[1]} coordinates; "
      f"RMS amplitude {np.sqrt(np.mean(generated**2)):.6f}")
print("A finite synthesized motion is an observation, not a forecasting accuracy result.")
