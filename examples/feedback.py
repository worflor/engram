"""Repeatable perception/dream/action loop with a finite retained packet pool."""

import numpy as np
from engram_codec import Engram, Loop, TrajectoryFeed
from well_network import signal

feed = [signal(i % 3, np.random.default_rng(i)) for i in range(8)]
loop = Loop(Engram(4), TrajectoryFeed(feed), rng=np.random.default_rng(42), capacity=4)
loop.seed(feed[0])
for record in loop.run(8):
    print(
        f"step={record['step']} reconstruction surprise={record['surprise']:.3f}% "
        f"dream/observation RMS={record['prediction_error']:.4f} packets={record['pool_size']}"
    )
assert len(loop.pool) <= 4
