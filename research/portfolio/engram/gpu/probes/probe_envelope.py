"""
probe_envelope.py — what this iGPU can safely handle.

probes the ADAPTER LIMITS without running any long compute. prints the
safe operating envelope: max buffer, max dispatch, shared memory budget,
feature list. no kernel dispatches — this is pure introspection so it
cannot hang the machine.

running this as the FIRST step of any new pipeline surfaces hardware
quirks before we write code that assumes specific limits.
"""

from __future__ import annotations

import wgpu
from wgpu.utils.device import get_default_device


def run():
    d = get_default_device()
    a = d.adapter

    print("=" * 60)
    print("  ADAPTER INFO")
    print("=" * 60)
    info = dict(a.info)
    for k in ["device", "vendor", "backend_type", "adapter_type",
              "description"]:
        if k in info:
            print(f"  {k}: {info[k]}")

    print()
    print("  FEATURES")
    for f in sorted(a.features):
        print(f"    {f}")

    print()
    print("  LIMITS (selected — what matters for our pipeline)")
    lim = dict(a.limits)
    safe_buffer_mb = None
    for key in sorted(lim):
        if any(w in key.lower() for w in
               ["buffer", "workgroup", "compute", "storage",
                "bind_groups", "dispatch", "invocation"]):
            v = lim[key]
            marker = ""
            if "buffer_binding_size" in key:
                safe_buffer_mb = v // (1024 * 1024)
                marker = f"  ({safe_buffer_mb} MiB max per binding)"
            if "workgroup_storage_size" in key:
                marker = f"  ({v // 1024} KiB shared memory per workgroup)"
            if "workgroups_per_dimension" in key:
                marker = f"  (max per-axis dispatch)"
            if "invocations_per_workgroup" in key:
                marker = f"  (max threads per workgroup)"
            print(f"    {key}: {v}{marker}")

    print()
    print("=" * 60)
    print("  SAFETY BUDGET (my rules for this hardware)")
    print("=" * 60)
    # conservative: half of max is always safe
    if safe_buffer_mb:
        print(f"  max single buffer:     {safe_buffer_mb} MiB (hardware max)")
        print(f"  soft cap (use this):   {safe_buffer_mb // 2} MiB")
    print(f"  max total GPU memory:  2048 MiB (assume 4GB iGPU, leave 2GB for OS)")
    print(f"  max dispatch per dim:  65535")
    print(f"  max workgroup size:    256 (baseline guarantee)")
    print(f"  max shared memory:     16384 B (baseline guarantee)")
    print(f"  kernel time budget:    1 second (TDR is ~2s — stay under)")
    print()
    print("  design rules:")
    print("    1. single thread owns device/queue (no concurrent submit)")
    print("    2. every WGSL loop has compile-time OR data-capped bound")
    print("    3. pre-validate buffer-binding sizes before dispatch")
    print("    4. bounded queue.Queue with explicit maxsize")
    print("    5. timeout on all blocking ops (queue.get, map_sync)")
    print("    6. if anything errors, destroy device & remake fresh")


if __name__ == "__main__":
    run()
