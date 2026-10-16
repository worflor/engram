"""Render a small README figure using actual Python full-wire reconstructions."""
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engram_codec import Engram, from_wire, to_wire

def build():
    t = np.arange(128)[:, None]
    d = np.arange(4)[None, :]
    signals = [('Orbit', np.sin(t*.07+d)),
               ('Damped orbit', np.exp(-t*.012)*np.sin(t*.12+d)),
               ('Phase change', np.sin(t*np.where(t<64,.06,.23)+d))]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="360" viewBox="0 0 1200 360" role="img" aria-label="Actual Engram source and full-wire reconstructed trajectories">',
           '<rect width="1200" height="360" fill="#203c36"/>',
           '<text x="32" y="44" fill="#f1f0e8" font-family="Georgia,serif" font-size="26">engram / the shape that remains</text>',
           '<text x="32" y="73" fill="#b8cabc" font-family="monospace" font-size="12">ACTUAL PYTHON FULL-PACKET RECONSTRUCTION · 4 DIMENSIONS · 128 SAMPLES</text>']
    for i,(name,signal) in enumerate(signals):
        x=signal.astype(np.float32)
        c=Engram(4)
        p=c.encode(x,pairing=np.arange(4,dtype=np.int32))
        y=c.decode(from_wire(to_wire(p)))
        lo,hi=min(x[:,0].min(),y[:,0].min())-.1,max(x[:,0].max(),y[:,0].max())+.1
        left=32+i*395
        svg.append(f'<text x="{left}" y="115" fill="#f1f0e8" font-family="Georgia,serif" font-size="20">{name}</text>')
        for values,color in ((x[:,0],'#b8cabc'),(y[:,0],'#c9a6df')):
            points=' '.join(f'{left+n/127*350:.2f},{285-(v-lo)/(hi-lo)*145:.2f}' for n,v in enumerate(values))
            svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        rms=float(np.sqrt(np.mean((x-y)**2)))
        svg.append(f'<text x="{left}" y="320" fill="#b8cabc" font-family="monospace" font-size="12">RMS {rms:.5f} · source / reconstruction</text>')
    svg.append('</svg>')
    path=ROOT/'docs/media/trajectories.svg'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(svg)+'\n',encoding='utf-8',newline='\n')
    print('Built README illustration from actual decoded samples')

if __name__ == '__main__':
    build()
