from pathlib import Path
import sys
root=Path('/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine')
hot=root/'research/selector_v3_online_diagnosis_repair_20260930/hotfix_logit_tolerance_20261001'
code=root/'research/selector_v3_online_diagnosis_20260930/code'
sys.path[:0]=[str(hot/'overlay'),str(code/'projects/AlgEngine/scripts/diffusiondrive'),str(code/'tests/innovation3')]
import innovation3.learner, innovation3.fixed_learning
assert Path(innovation3.learner.__file__).is_relative_to(hot)
import pytest
raise SystemExit(pytest.main(['-q', str(code/'tests/innovation3/test_fixed_learning.py')]))
