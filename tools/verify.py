"""One offline acceptance path for local development and CI."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    print("+ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-generated", action="store_true")
    parser.add_argument(
        "--gpu", action="store_true", help="Run real-device optional batch fitting"
    )
    args = parser.parse_args()
    run("cargo", "test", "--locked", "-q")
    run(sys.executable, "-m", "pytest", "-q", "-W", "error", "tests/python")
    run(sys.executable, "-m", "pytest", "-q", "-W", "error", "tests/research")
    run("cargo", "run", "--locked", "--quiet", "--example", "trajectory")
    run(sys.executable, "examples/trajectory.py")
    run("cargo", "run", "--locked", "--quiet", "--example", "motion")
    run(sys.executable, "examples/text_memory.py")
    run(sys.executable, "examples/continuation.py")
    run(sys.executable, "examples/affine_motion.py")
    run(sys.executable, "examples/temporal_witness.py")
    run(sys.executable, "examples/oscillator_blend.py")
    run(sys.executable, "examples/relational_wells.py")
    for example in [
        "well_network",
        "train_bytes",
        "hunk_search",
        "code_retrieval",
        "alexandria_snapshot",
        "feedback",
    ]:
        run(sys.executable, f"examples/{example}.py")
    if args.gpu:
        run(sys.executable, "examples/gpu_training.py")
    run(sys.executable,'examples/npc_experience.py','--training','120','--planning','5')
    run(sys.executable, "tools/parity.py")
    run(sys.executable, "tools/check_repository.py")
    run(
        sys.executable,
        "tools/build_instrument.py",
        *(["--check"] if args.check_generated else []),
    )
    run(
        sys.executable,
        "tools/build_network_demo.py",
        *(["--check"] if args.check_generated else []),
    )
    print(
        "Offline acceptance passed. Optional full-corpus/model/Java campaigns are separate; --gpu selects the maintained device check."
    )
