"""bench_async_completion.py — test whether wgpu-py's async completion
primitives give us lower-latency readback than map_sync.

we stage a realistic workload (kernel writes ~600 KB, we read back) and
compare four strategies:

  A. classic:  submit → map_sync → read_mapped → unmap
  B. async:    submit → map_async.sync_wait → read_mapped → unmap
  C. on_work:  submit → queue.on_submitted_work_done_async → map → read
  D. poll:     submit → device._poll(True) (spin) → map_sync

the grimoire hypothesis: map_sync spins a driver loop that is slower
to notice GPU completion than the background _poller thread. if the
async path uses the _poller, it should trigger completion faster.

run with: `python bench_async_completion.py`
"""

from __future__ import annotations

import time
import numpy as np
import wgpu
from wgpu.utils.device import get_default_device


SHADER = """
@group(0) @binding(0) var<storage, read_write> out : array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    // compute workload scales with outer launch size; for the "big"
    // scale we want ~5-10 ms of GPU work to mimic production
    // gather+features timing. iters inside the loop is small so
    // completion latency vs kernel time can be studied separately.
    var x : f32 = f32(i) * 0.001;
    for (var k : u32 = 0u; k < 256u; k = k + 1u) {
        x = x * 0.9999 + sin(x) * 0.01;
    }
    out[i] = x;
}
"""


def setup(device, n_floats):
    module = device.create_shader_module(code=SHADER)
    layout = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.storage}},
    ])
    pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[layout]),
        compute={"module": module, "entry_point": "main"},
    )
    out_buf = device.create_buffer(
        size=n_floats * 4,
        usage=(wgpu.BufferUsage.STORAGE
               | wgpu.BufferUsage.COPY_SRC),
    )
    rb_buf = device.create_buffer(
        size=n_floats * 4,
        usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST,
    )
    bg = device.create_bind_group(
        layout=layout,
        entries=[
            {"binding": 0, "resource": {"buffer": out_buf, "offset": 0,
                                         "size": n_floats * 4}},
        ],
    )
    return pipeline, out_buf, rb_buf, bg


def submit_and_dispatch(device, pipeline, bg, out_buf, rb_buf, n_floats):
    enc = device.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipeline)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups((n_floats + 63) // 64, 1, 1)
    cp.end()
    enc.copy_buffer_to_buffer(out_buf, 0, rb_buf, 0, n_floats * 4)
    device.queue.submit([enc.finish()])


def bench_classic(device, pipeline, out_buf, rb_buf, bg, n_floats, iters):
    t0 = time.perf_counter()
    for _ in range(iters):
        submit_and_dispatch(device, pipeline, bg, out_buf, rb_buf, n_floats)
        rb_buf.map_sync(wgpu.MapMode.READ, size=n_floats * 4)
        _ = np.frombuffer(rb_buf.read_mapped(size=n_floats * 4),
                          dtype=np.float32).copy()
        rb_buf.unmap()
    return (time.perf_counter() - t0) / iters


def bench_map_async(device, pipeline, out_buf, rb_buf, bg, n_floats, iters):
    # map_async returns a GPUPromise; .sync_wait on it should drive the poller
    t0 = time.perf_counter()
    for _ in range(iters):
        submit_and_dispatch(device, pipeline, bg, out_buf, rb_buf, n_floats)
        promise = rb_buf.map_async(wgpu.MapMode.READ, size=n_floats * 4)
        promise.sync_wait()
        _ = np.frombuffer(rb_buf.read_mapped(size=n_floats * 4),
                          dtype=np.float32).copy()
        rb_buf.unmap()
    return (time.perf_counter() - t0) / iters


def bench_on_work(device, pipeline, out_buf, rb_buf, bg, n_floats, iters):
    # wait for queue work done first, then map_sync should be instant
    t0 = time.perf_counter()
    for _ in range(iters):
        submit_and_dispatch(device, pipeline, bg, out_buf, rb_buf, n_floats)
        work_promise = device.queue.on_submitted_work_done_async()
        work_promise.sync_wait()
        rb_buf.map_sync(wgpu.MapMode.READ, size=n_floats * 4)
        _ = np.frombuffer(rb_buf.read_mapped(size=n_floats * 4),
                          dtype=np.float32).copy()
        rb_buf.unmap()
    return (time.perf_counter() - t0) / iters


def bench_poll(device, pipeline, out_buf, rb_buf, bg, n_floats, iters):
    # use device._poll_wait to drive completion, then map
    t0 = time.perf_counter()
    for _ in range(iters):
        submit_and_dispatch(device, pipeline, bg, out_buf, rb_buf, n_floats)
        device._poll_wait()
        rb_buf.map_sync(wgpu.MapMode.READ, size=n_floats * 4)
        _ = np.frombuffer(rb_buf.read_mapped(size=n_floats * 4),
                          dtype=np.float32).copy()
        rb_buf.unmap()
    return (time.perf_counter() - t0) / iters


def main():
    device = get_default_device()
    # two scales: small (~600 KB, quick kernel) and large (~6 MB,
    # kernel doing ~1000× more work — closer to production gather+features
    # where feat_wait is 67-118 ms).
    import sys
    scale = sys.argv[1] if len(sys.argv) > 1 else "small"
    if scale == "big":
        n_floats = 1_500_000
        iters = 20
    else:
        n_floats = 150_000
        iters = 30

    pipeline, out_buf, rb_buf, bg = setup(device, n_floats)

    # warmup
    for _ in range(5):
        submit_and_dispatch(device, pipeline, bg, out_buf, rb_buf, n_floats)
        rb_buf.map_sync(wgpu.MapMode.READ, size=n_floats * 4)
        rb_buf.unmap()

    print(f"n_floats={n_floats} (~{n_floats*4/1024:.0f} KB), iters={iters}")
    print()

    tests = [
        ("classic  map_sync", bench_classic),
        ("map_async sync_wait", bench_map_async),
        ("on_submitted_work_done_async", bench_on_work),
        ("device._poll_wait + map_sync", bench_poll),
    ]

    results = []
    for name, fn in tests:
        # interleave so none gets priority
        samples = []
        for _ in range(3):
            samples.append(fn(device, pipeline, out_buf, rb_buf, bg,
                              n_floats, iters))
        best = min(samples)
        median = sorted(samples)[len(samples)//2]
        results.append((name, best, median))
        print(f"  {name:40s}  best={best*1000:.3f} ms   median={median*1000:.3f} ms")

    print()
    baseline = results[0][1]
    print(f"speedup vs classic map_sync (best-of-3):")
    for name, best, _ in results:
        print(f"  {name:40s}  {baseline/best:.3f}×")


if __name__ == "__main__":
    main()
