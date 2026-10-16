"""Replay retained native observations and verify the real-game evidence chain."""
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree
import numpy as np
from engram_npc import WitnessHistory

ROOT=Path(__file__).resolve().parents[2]


def test_retained_minecraft_receipts_and_native_history_control():
    receipt=json.loads((ROOT/'research/npc/minecraft.json').read_text(encoding='utf-8'))
    first,second=receipt['first'],receipt['seeded_fresh_world']
    assert first['checkpoint_sha256']==second['initial_checkpoint_sha256']
    assert first['learned']['success_rate']==second['learned']['success_rate']==1.
    assert first['without_fitted_history']['success_rate']==.5
    assert first['before_environment_change']['success_rate']==0.
    assert first['after_environment_change']['success_rate']==1.
    for key,name,run in [('first','minecraft-first.xml',first),('second','minecraft-seeded.xml',second)]:
        data=(ROOT/'research/npc'/name).read_bytes()
        assert hashlib.sha256(data).hexdigest()==receipt['junit'][key]['sha256']
        junit=ElementTree.fromstring(data)
        assert len(list(junit.iter('testcase')))==run['junit_cases']==1
        assert not list(junit.iter('failure')) and not list(junit.iter('error'))
        for source,digest in run['sources'].items():
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest,source
    pairs=receipt['identical_current_witness_control']['pairs']
    assert pairs[0]['junction_witness']==pairs[1]['junction_witness']
    assert pairs[0]['earlier_cue']==-pairs[1]['earlier_cue']
    fitted=[]
    for pair in pairs:
        history=WitnessHistory(8)
        for witness in pair['native_observations']:signature=history.push(witness)
        k=np.array([[z.real,z.imag] for z in signature.k])
        np.testing.assert_allclose(k,pair['K'],atol=1e-10)
        fitted.append(signature)
    assert fitted[0].compare(fitted[1])['kg_rms_distance']>.05
