"""Generate a self-contained explorer from real Python and Rust codec output."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = [p.name for p in sorted(ROOT.glob('engram_*.py'))] + ['examples/probe.rs', 'tools/build_instrument.py',
                'instrument/template.html'] + sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/'src').glob('*.rs'))
sys.path.insert(0, str(ROOT))
from engram_codec import Engram, from_wire, to_wire
sys.path.insert(0, str(ROOT / 'tools'))
from parity import probe


def build():
    subprocess.run(['cargo', 'build', '--locked', '--example', 'probe'], cwd=ROOT, check=True)
    records = []
    for dim in (4, 8):
        t = np.arange(128)[:, None]
        d = np.arange(dim)[None, :]
        rng = np.random.default_rng(20261015 + dim)
        signals = {
            'Orbit': np.sin(t*.07+d),
            'Damped orbit': np.exp(-t*.012)*np.sin(t*.12+d),
            'Ramp': t*.01+d*.1,
            'Phase change': np.sin(t*np.where(t<64, .06, .23)+d),
            'Noise': rng.normal(0, .25, (128, dim)),
        }
        for name, signal in signals.items():
            x = signal.astype(np.float32)
            pairing = np.arange(dim, dtype=np.int32)
            rust = probe({'dim': dim, 'samples': x.ravel().tolist(), 'pairing': pairing.tolist()})
            codec = Engram(dim)
            packet = codec.encode(x, pairing=pairing)
            full, compact = to_wire(packet), to_wire(packet, compact=True)
            blocks = [{'start': b.start, 'length': b.length, 'mode': b.mode.name.title(),
                       'macro_k': [[float(z.real), float(z.imag)] for z in b.macro_K],
                       'macro_g': [[float(z.real), float(z.imag)] for z in b.macro_G]} for b in packet.blocks]
            for backend, decoded, wire, small, bs in (
                ('Python', codec.decode(from_wire(full)), list(full), list(compact), blocks),
                ('Rust', np.array(rust['wire_decoded']).reshape(x.shape), rust['full'], rust['compact'], rust['blocks']),
            ):
                records.append({'name': name, 'backend': backend, 'dim': dim, 'samples': x.tolist(),
                                'decoded': decoded.tolist(), 'full': wire, 'compact': small, 'blocks': bs,
                                'rms': float(np.sqrt(np.mean((decoded-x)**2))),
                                'max_error': float(np.max(np.abs(decoded-x)))})
    evidence = {'schema': 'engram-instrument-1', 'seed': 20261015, 'records': records,
                'sources': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                            for p in SOURCE_PATHS}}
    data = json.dumps(evidence, separators=(',', ':'), allow_nan=False)
    html = (ROOT/'instrument/template.html').read_text(encoding='utf-8').replace('/*__DATA__*/', data)
    return html


def check():
    # Floating measurements can differ by libm/platform. Check source identity
    # without imposing bitwise equality between independently measured platforms.
    html = (ROOT/'instrument/engram.html').read_text(encoding='utf-8')
    data = json.loads(html.split('const DATA=', 1)[1].split(';\nconst $=', 1)[0])
    for path, digest in data['sources'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest:
            raise SystemExit(f'Instrument is stale for {path}: rebuild it')
    if set(data['sources']) != set(SOURCE_PATHS) or len(data['records']) != 20:
        raise SystemExit('Instrument metadata is incomplete')
    print('Instrument source fingerprints and 20-case dataset verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        check()
    else:
        html = build()
        path = ROOT/'instrument/engram.html'
        path.write_text(html, encoding='utf-8', newline='\n')
        print(f'Built offline instrument ({len(html):,} characters, 20 measured cases)')
