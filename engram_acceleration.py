"""Optional WebGPU batch fitting; CPU owns learning order and memory semantics.

The portfolio's coalesced, two-pass residual kernel is retained as a narrow
accelerator. It does not route wells or maintain a divergent second brain.
Float32 fitting is approximate; near-singular/extreme inputs should use the
float64 reference. No silent CPU fallback or performance claim.
"""

from importlib.resources import files
import numpy as np
from engram_analysis import Signature
from engram_codec import Engram, _validated_trajectory
from engram_wire import MAX_VALUES


class GpuFitter:
    def __init__(self):
        try:
            import wgpu
        except ImportError as exc:
            raise RuntimeError(
                'Install the optional "gpu" dependency to use GpuFitter'
            ) from exc
        self.wgpu = wgpu
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError("No WebGPU adapter available")
        self.device = adapter.request_device_sync()
        self.adapter_info = dict(adapter.info)
        module = self.device.create_shader_module(
            code=files("engram_kernels")
            .joinpath("fit.wgsl")
            .read_text(encoding="utf-8")
        )
        self.pipeline = self.device.create_compute_pipeline(
            layout="auto", compute={"module": module, "entry_point": "main"}
        )

    def fit(self, trajectories, *, dim, pairing=None):
        """Return ordered AR(2) signatures for a bounded batch; inputs need >=3 rows."""
        Engram(dim)
        basis = np.arange(dim) if pairing is None else pairing
        arrays, segments, offset = [], [], 0
        for i, trajectory in enumerate(trajectories):
            x, basis = _validated_trajectory(trajectory, dim, basis)
            if len(x) < 3 or offset + x.size > MAX_VALUES:
                raise ValueError("GPU fitting needs >=3 rows and a bounded total batch")
            arrays.append(np.ascontiguousarray(x[:, basis]))
            segments.append((offset, len(x), dim, i * (dim // 2)))
            offset += x.size
        if not arrays:
            return []
        limits = self.device.limits
        count, pairs = len(arrays), dim // 2
        if count > limits["max-compute-workgroups-per-dimension"]:
            raise ValueError("batch exceeds GPU dispatch limits")
        buffers = []
        try:
            storage = self.wgpu.BufferUsage.STORAGE
            for data in [
                np.concatenate([a.ravel() for a in arrays]),
                np.array(segments, dtype=np.uint32),
            ]:
                if data.nbytes > limits["max-storage-buffer-binding-size"]:
                    raise ValueError("batch exceeds GPU storage limits")
                buffers.append(
                    self.device.create_buffer_with_data(data=data, usage=storage)
                )
            for width in [8, 8, 4]:
                size = count * pairs * width
                if size > limits["max-storage-buffer-binding-size"]:
                    raise ValueError("output exceeds GPU storage limits")
                buffers.append(
                    self.device.create_buffer(
                        size=size, usage=storage | self.wgpu.BufferUsage.COPY_SRC
                    )
                )
            bindings = self.device.create_bind_group(
                layout=self.pipeline.get_bind_group_layout(0),
                entries=[
                    {"binding": i, "resource": {"buffer": buffer}}
                    for i, buffer in enumerate(buffers)
                ],
            )
            encoder = self.device.create_command_encoder()
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(self.pipeline)
            compute.set_bind_group(0, bindings)
            compute.dispatch_workgroups(count, (pairs + 63) // 64, 1)
            compute.end()
            self.device.queue.submit([encoder.finish()])
            outputs = [
                np.frombuffer(self.device.queue.read_buffer(buffer), dtype=dtype)
                .copy()
                .reshape(count, pairs)
                for buffer, dtype in zip(
                    buffers[2:], [np.complex64, np.complex64, np.float32]
                )
            ]
            if not all(np.isfinite(a).all() for a in outputs):
                raise FloatingPointError(
                    "GPU fitting exceeded finite range; use float64 reference"
                )
            results = []
            for i, x in enumerate(arrays):
                values = [
                    np.array(outputs[0][i], dtype=complex),
                    np.array(outputs[1][i], dtype=complex),
                    np.array(outputs[2][i], dtype=float),
                    np.array(basis, dtype=np.int32),
                ]
                for value in values:
                    value.flags.writeable = False
                results.append(Signature(*values, len(x), 2))
            return results
        finally:
            for buffer in buffers:
                buffer.destroy()
