"""Check packet interoperability on declared synthetic cases, not whole-port equivalence."""

import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engram_codec import Engram, EngramFile, Signature, from_wire, to_wire


def probe(request):
    result = subprocess.run(
        [str(ROOT / "target/debug/examples/probe")],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        cwd=ROOT,
    )
    result.check_returncode()
    return json.loads(result.stdout)


def main():
    subprocess.run(
        ["cargo", "build", "--locked", "--example", "probe"], cwd=ROOT, check=True
    )
    observations = []
    for name, length, dim in [
        ("ramp", 64, 4),
        ("orbit", 128, 8),
        ("short", 2, 4),
        ("one", 1, 4),
        ("three", 3, 4),
        ("four", 4, 4),
        ("static", 16, 4),
        ("permuted", 64, 4),
    ]:
        t = np.arange(length)[:, None]
        d = np.arange(dim)[None, :]
        x = (t * 0.01 + d * 0.1 if name != "orbit" else np.sin(t * 0.07 + d)).astype(
            np.float32
        )
        if name == "static":
            x[:] = 0.25
        pairing = (
            np.array([3, 1, 2, 0], dtype=np.int32)
            if name == "permuted"
            else np.arange(dim, dtype=np.int32)
        )
        rust = probe(
            {"dim": dim, "samples": x.ravel().tolist(), "pairing": pairing.tolist()}
        )
        codec = Engram(dim)
        python = codec.encode(x, pairing=pairing)
        assert [(b.start, b.length, b.mode.name.lower()) for b in python.blocks] == [
            (b["start"], b["length"], b["mode"].lower()) for b in rust["blocks"]
        ]
        rust_in_python = codec.decode(from_wire(bytes(rust["full"])))
        python_in_rust = np.array(
            probe({"dim": dim, "wire": list(to_wire(python))})["decoded"]
        ).reshape(x.shape)
        # Full wire from either encoder must decode equivalently in the other runtime.
        rust_native = np.array(rust["wire_decoded"]).reshape(x.shape)
        python_native = codec.decode(from_wire(to_wire(python)))
        np.testing.assert_allclose(rust_in_python, rust_native, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(python_in_rust, python_native, atol=1e-4, rtol=1e-4)
        compact = from_wire(bytes(rust["compact"]))
        assert compact.length == length and compact.dim == dim
        python_compact = from_wire(to_wire(python, compact=True))
        compact_in_rust = probe(
            {"dim": dim, "compact": True, "wire": list(to_wire(python, compact=True))}
        )
        assert compact_in_rust["length"] == length and len(
            compact_in_rust["blocks"]
        ) == len(python.blocks)
        for a, b in zip(compact_in_rust["blocks"], python_compact.blocks):
            assert a["mode"] == int(b.mode)
            if a["mode"] != 3:  # Raw mode has no oscillator-coefficient payload.
                np.testing.assert_allclose(
                    np.array(a["macro_k"]),
                    np.column_stack([b.macro_K.real, b.macro_K.imag]),
                    atol=1e-4,
                )
            if a["pair_rms"] is not None:
                np.testing.assert_allclose(a["pair_rms"], b.pair_rms, atol=1e-3)
        observations.append(
            {
                "case": name,
                "length": length,
                "dim": dim,
                "decoder_disagreement": max(
                    float(np.max(np.abs(rust_in_python - rust_native))),
                    float(np.max(np.abs(python_in_rust - python_native))),
                ),
                "rust_to_python_max_error": float(np.max(np.abs(rust_in_python - x))),
                "python_to_rust_max_error": float(np.max(np.abs(python_in_rust - x))),
            }
        )
    fixtures = json.loads((ROOT / "tests/fixtures/source-kernels.json").read_text())
    for case in fixtures["cases"]:
        dim = case["dim"]
        basis = list(range(dim))
        native = probe(
            {"analyze": True, "dim": dim, "samples": case["samples"], "pairing": basis}
        )
        sig = Signature.from_trajectory(
            np.array(case["samples"], dtype=np.float32).reshape(case["length"], dim),
            pairing=basis,
        )
        np.testing.assert_allclose(
            native["k"], np.column_stack([sig.k.real, sig.k.imag]), atol=1e-7
        )
        np.testing.assert_allclose(
            native["g"], np.column_stack([sig.g.real, sig.g.imag]), atol=1e-7
        )
    samples = np.sin(np.arange(64)[:, None] * 0.07 + np.arange(4)).astype(np.float32)
    native = probe(
        {
            "memory": True,
            "dim": 4,
            "samples": samples.ravel().tolist(),
            "pairing": [0, 1, 2, 3],
        }
    )
    imported = EngramFile.load(bytes(native["bytes"]))
    assert (
        imported._total_absorbed == native["absorbed"]
        and len(imported.wells()) == native["wells"]
    )
    print(
        json.dumps(
            {
                "packet_interoperability": observations,
                "signature_cases": len(fixtures["cases"]),
                "native_memory_import": "passed",
                "atol": 1e-4,
                "rtol": 1e-4,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
