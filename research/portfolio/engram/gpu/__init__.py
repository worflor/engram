"""
gpu acceleration layer for the engram codec.

wgpu / WGSL based — vendor-agnostic, future-proof. runs on whatever
backend the wgpu-native crate picks for the host: Vulkan on Linux,
Vulkan or DX12 on Windows, Metal on macOS.

the hot kernel is fit_pair.wgsl (batched complex AR(2) normal equations).
one workgroup per (segment, pair) tuple; threads cooperate on the time-
axis reduction.
"""

from .engine import GpuEngine, gpu_available

__all__ = ["GpuEngine", "gpu_available"]
