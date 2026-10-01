import importlib.util
from pathlib import Path
import sys
import pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/online'))
from gpu_preflight import validate


def good():
    return dict(device_count=8, cuda_available=True, cuda_visible_devices='0,1,2,3,4,5,6,7',
                device_names=['NVIDIA H100 80GB HBM3']*8, learner_atol=5e-5,
                fixed_learning_atol=5e-5, code_root=str(ROOT),
                module_paths={'learner':str(ROOT/'projects/AlgEngine/scripts/diffusiondrive/innovation3/learner.py')})


def test_valid_h100_configuration():
    assert validate(good()) == []


@pytest.mark.parametrize('kind', ['count', 'model', 'cuda', 'atol', 'source'])
def test_failures_name_the_rejected_check(kind):
    info=good()
    if kind=='count':info.update(device_count=1,device_names=info['device_names'][:1]);expected='GPU_COUNT'
    elif kind=='model':info['device_names']=['NVIDIA H800']*8;expected='GPU_MODEL'
    elif kind=='cuda':info['cuda_available']=False;expected='CUDA_UNAVAILABLE'
    elif kind=='atol':info['learner_atol']=1e-5;expected='LOGIT_TOLERANCE'
    else:info['module_paths']['learner']='/tmp/old_research/learner.py';expected='IMPORT_SOURCE'
    assert any(e.startswith(expected) for e in validate(info))


def test_resolves_equivalent_path_aliases(tmp_path):
    alias=tmp_path/'alias';alias.symlink_to(ROOT,target_is_directory=True)
    info=good();info['code_root']=str(alias)
    assert validate(info)==[]
