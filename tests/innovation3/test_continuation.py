"""Migration must reject drift and keep the 904-episode global cursor."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "projects/AlgEngine/scripts/diffusiondrive"))
from innovation3.continuation import load_import, inherited_contract, validate_legacy_checkpoint
from innovation3.ddp_session_boundary import assignment


def saved(tmp_path):
    raw=tmp_path/'rank.pt';raw.write_bytes(b'checkpoint')
    digest=hashlib.sha256(raw.read_bytes()).hexdigest()
    rounds=[[dict(checkpoint=dict(path=str(raw),sha256=digest)) for _ in range(8)] for _ in range(113)]
    parent=dict(completed=904,rounds=rounds,contract=dict(world_size=8,episodes=904,
                online_update_timing='after_rollout8',source_sha256={'old':'hash'}))
    p=tmp_path/'parent.json';p.write_text(json.dumps(parent))
    manifest=dict(schema=1,purpose='B_p6_to_p8_explicit_continuation',
        new_source_sha256={'new':'hash'},parent_progress=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()),
        checkpoints=[dict(path=str(raw),sha256=digest)] + [dict(path=str(tmp_path/('rank%d.pt'%i)),sha256=digest) for i in range(1,8)])
    f=tmp_path/'import.json';f.write_text(json.dumps(manifest));ref=dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    return ref,parent,raw


def test_rejects_code_change_and_modified_parent(tmp_path):
    ref,parent,_=saved(tmp_path)
    with pytest.raises(ValueError,match='target source'):load_import(ref,{'new':'changed'})
    (tmp_path/'parent.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum'):load_import(ref)


def test_rejects_rewritten_prefix(tmp_path):
    ref,parent,_=saved(tmp_path)
    state=dict(lineage=dict(reference=ref,inherited_rounds=113),rounds=copy.deepcopy(parent['rounds']))
    assert inherited_contract(state,dict(source_sha256={'new':'hash'}))==113
    state['rounds'][0][0]['checkpoint']['sha256']='bad'
    with pytest.raises(ValueError,match='prefix'):inherited_contract(state,dict(source_sha256={'new':'hash'}))


def test_only_whitelisted_p6_full_state_can_cross_source_boundary(tmp_path):
    ref,parent,raw=saved(tmp_path)
    payload=dict(source_sha256={'old':'hash'},learner=dict(attempts=904,version=904))
    validate_legacy_checkpoint(payload,raw,ref,{'new':'hash'})
    payload['learner']['attempts']=0
    with pytest.raises(ValueError,match='cursor'):validate_legacy_checkpoint(payload,raw,ref,{'new':'hash'})
    payload['learner']['attempts']=904
    raw.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='approved'):validate_legacy_checkpoint(payload,raw,ref,{'new':'hash'})


def test_global_cursor_does_not_restart_and_reaches_exactly_eight_visits():
    from collections import Counter
    for seed in range(3):
        spec=dict(schedule='seeded_cyclic_global_episode_v2',seed=seed,world_size=8,
                  rows=[dict(scene_id='scene%d'%i) for i in range(150)])
        sequence=[assignment(spec,r,k)['scene_id'] for r in range(150) for k in range(8)]
        assert sequence[904:912] != sequence[:8]
        assert set(Counter(sequence).values())=={8}
        assert len(sequence[904:])==296
