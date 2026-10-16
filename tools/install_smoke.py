"""Install one built wheel in an isolated environment, outside source imports."""

from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
wheels = list((ROOT / "dist").glob("woflo_engram-*.whl"))
if len(wheels) != 1:
    raise SystemExit("Build exactly one intended wheel in dist before install smoke")
environment = ROOT / ".smoke/python"
subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
subprocess.run(
    [
        str(python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--force-reinstall",
        str(wheels[0]),
    ],
    check=True,
)
code = """
from pathlib import Path
import numpy as np
import engram_codec
import engram_wire, engram_analysis, engram_memory_io, engram_embeddings
import engram_network, engram_training, engram_retrieval, engram_metrics
import engram_corpus, engram_acceleration
import engram_npc
from engram_affine import AffineOscillator
from engram_history import WitnessHistory
from engram_stability import blend_schur
from engram_relations import ContextRelations
from importlib.resources import files
assert files('engram_kernels').joinpath('fit.wgsl').is_file()
from importlib.metadata import version
assert version('woflo-engram') == engram_codec.__version__
assert '.smoke' in str(Path(engram_codec.__file__).resolve()), engram_codec.__file__
affine = AffineOscillator.fit(np.tile([8., -3.], (10, 1)))
np.testing.assert_allclose(affine.forecast(3), np.tile([8., -3.], (3, 1)))
left, right = WitnessHistory(2), WitnessHistory(2)
for row in [[0,0],[1,0]]: left.push(row)
for row in [[0,0],[2,0]]: right.push(row)
assert left.concat(right).samples == 4
blend_schur([[.2],[.3]],[[.1],[.2]],[1,1])
relations = ContextRelations(('cue','outcome'), representation='installed-smoke-v1')
relations.push('known',[-1.,-1.])
relations.push('known',[1.,1.])
restored_relations = ContextRelations.from_dict(relations.to_dict())
assert restored_relations.moments('known').correlation[0,1] == 1
assert restored_relations.moments('unknown') is None
c = engram_codec.Engram(2)
x = np.arange(32, dtype=np.float32).reshape(16,2) * .01
p = engram_codec.from_wire(engram_codec.to_wire(c.encode(x)))
assert p.length == 16
assert np.max(np.abs(c.decode(p)-x)) < .02
signature = engram_codec.Signature.from_trajectory(x,pairing=np.arange(2))
assert signature.compare(signature)['kg_rms_distance']==0
import tempfile
with tempfile.TemporaryDirectory() as directory:
    path = str(Path(directory)/'memory.engram')
    m = engram_codec.EngramFile.create(path,dim=2)
    m.fast_absorb(x,source='smoke')
    m.save()
    loaded = engram_codec.EngramFile.load(path)
    np.testing.assert_array_equal(m._dream_Ks,loaded._dream_Ks)
    network = engram_network.WellNetwork(2)
    network.learn(x)
    checkpoint = Path(directory)/'network.engram'
    network.save(checkpoint)
    restored = engram_network.WellNetwork.load(checkpoint)
    assert restored.dream(x,steps=4).trajectory.shape == (4,2)
    npc = engram_npc.EngramNPC(engram_npc.WitnessSchema(('x','y')))
    for row in x[:4]:
        npc.observe(row)
    decision = npc.decide(['host-response'])
    npc.feedback(decision,x[4],1.0,terminal=True)
    state = Path(directory)/'npc.engram'
    npc.save(state)
    loaded_npc = engram_npc.EngramNPC.load(state)
    assert loaded_npc.updates == 1
    assert loaded_npc.memory._well_count == npc.memory._well_count
print('Installed wheel imported and reconstructed outside the source tree')
"""
env = dict(os.environ)
env.pop("PYTHONPATH", None)
subprocess.run([str(python), "-I", "-c", code], cwd=environment, env=env, check=True)
