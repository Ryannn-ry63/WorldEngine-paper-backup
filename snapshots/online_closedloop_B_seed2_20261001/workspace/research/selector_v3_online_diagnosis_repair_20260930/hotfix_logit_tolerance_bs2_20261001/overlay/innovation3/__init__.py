"""Hotfix overlay: only learner numerical-audit modules are overridden."""
from pathlib import Path
_original = Path('/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_20260930/code/projects/AlgEngine/scripts/diffusiondrive/innovation3')
if _original.is_dir():
    __path__.append(str(_original))
