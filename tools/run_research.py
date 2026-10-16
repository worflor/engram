"""Run an explicitly selected, inspected research script with its original import topology."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT/'research/portfolio'
if len(sys.argv) < 2:
    raise SystemExit('Usage: python tools/run_research.py engram/<script.py> [arguments]')
target = (RESEARCH/sys.argv[1]).resolve()
if not target.is_relative_to(RESEARCH.resolve()) or target.suffix != '.py' or not target.is_file():
    raise SystemExit('Select an existing Python script under research/portfolio')
sys.path[:0] = [str(ROOT), str(RESEARCH), str(RESEARCH/'engram'),
                str(RESEARCH/'engram/experiments'), str(target.parent)]
sys.argv = [str(target), *sys.argv[2:]]
runpy.run_path(str(target), run_name='__main__')
