"""Audit the owned scene pool before creating a formal online manifest.

This is a CPU/file-system gate.  It intentionally fails closed: a preliminary
owned-scene inventory or an engineering manifest can never be promoted to a
formal training manifest by this command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def origin_log(value: str) -> str:
    value = re.sub(r"-[0-9a-f]{16,}$", "", value)
    value = re.sub(r"_[0-9]+_[0-9]+$", "", value)
    if not re.fullmatch(r"\d{4}(?:\.\d{2}){5}_veh-\d+", value):
        raise ValueError(f"unsupported origin log: {value!r}")
    return value


def manifest_logs(path: Path) -> set[str]:
    data = json.loads(path.read_text())
    logs = set()
    for row in data.get("files", []):
        relative = row.get("relative", "")
        if relative:
            logs.add(origin_log(relative.split("/", 1)[0]))
    return logs


def audit(args):
    inventory = json.loads(args.inventory.read_text())
    acceptance = json.loads(args.acceptance.read_text())
    backup = json.loads(args.backup.read_text())
    c_manifest = json.loads(args.c_manifest.read_text())
    exposed_manifest = json.loads(args.exposed_manifest.read_text())

    reasons = []
    checks = {}
    checks["migration_acceptance"] = acceptance.get("status") == "PASS_NEW_INSTANCE_DATA_ACCEPTANCE"
    checks["owned_backup"] = backup.get("status") == "PASS_134_SCENE_OWNED_SUBSET_BACKUP"
    checks["inventory_is_preliminary"] = inventory.get("formal_ready") is False
    if not checks["migration_acceptance"]:
        reasons.append("new-instance migration receipt is not PASS_NEW_INSTANCE_DATA_ACCEPTANCE")
    if not checks["owned_backup"]:
        reasons.append("owned asset backup receipt is not PASS_134_SCENE_OWNED_SUBSET_BACKUP")

    c_excluded = {origin_log(x) for x in c_manifest.get("excluded_logs", [])}
    exposed = {origin_log(x.get("origin_log", "")) for x in exposed_manifest.get("scenes", [])}
    # Engineering manifests are evidence of exposure only; they cannot supply
    # formal candidates.
    checks["engineering_manifest_not_formal"] = exposed_manifest.get("formal_ready") is False
    if not checks["engineering_manifest_not_formal"]:
        reasons.append("exposed-scene manifest is unexpectedly formal_ready")

    candidates, seen_logs = [], set()
    counts = {"inventory": 0, "preliminary_rejected": 0, "missing_files": 0,
              "c_excluded_origin": 0, "exposed_origin": 0, "duplicate_origin": 0}
    for row in inventory.get("scenes", []):
        counts["inventory"] += 1
        if row.get("eligible_after_preliminary_C_screen") is not True:
            counts["preliminary_rejected"] += 1
            continue
        paths = [row.get("owned_pickle"), row.get("owned_background"), row.get("owned_config")]
        if not all(isinstance(p, str) and Path(p).is_file() for p in paths):
            counts["missing_files"] += 1
            continue
        logs = {origin_log(x) for x in row.get("origin_logs", [])}
        if logs & c_excluded:
            counts["c_excluded_origin"] += 1
            continue
        if len(logs) != 1 or next(iter(logs)) in seen_logs:
            counts["duplicate_origin"] += 1
            continue
        log = next(iter(logs)); seen_logs.add(log)
        prior_exposure = bool(logs & exposed)
        if prior_exposure:
            counts["exposed_origin"] += 1
        candidates.append({
            "scene_id": row["scene_id"], "origin_log": log,
            "scene_path": row["owned_pickle"], "asset_path": row["owned_background"],
            "config_path": row["owned_config"], "eligible": True,
            "audit_record_id": row["scene_id"],
            "prior_engineering_exposure": prior_exposure,
            "scene_sha256": sha256(Path(row["owned_pickle"])),
            "asset_sha256": sha256(Path(row["owned_background"])),
        })

    checks["origin_log_unique"] = len(candidates) == len(seen_logs)
    checks["candidate_pool_minimum"] = len(candidates) >= args.min_origin_logs
    if not checks["candidate_pool_minimum"]:
        reasons.append(f"only {len(candidates)} origin logs remain; need at least {args.min_origin_logs}")
    checks["train_split_minimum"] = int(len(candidates) * 0.8) >= args.min_train_scenes
    if not checks["train_split_minimum"]:
        reasons.append(f"80% train split has fewer than {args.min_train_scenes} scenes")

    status = "PASS_FORMAL_SCENE_AUDIT" if not reasons else "BLOCKED_FORMAL_SCENE_AUDIT"
    receipt = {
        "schema": 1, "audit_version": "formal-scene-pool-v1",
        "qualification_scope": "owned_c_and_engineering_exposure_only",
        "status": status, "formal_ready": status == "PASS_FORMAL_SCENE_AUDIT",
        "checks": checks, "reasons": reasons, "counts": counts,
        "excluded_origin_logs": sorted(c_excluded),
        "exposed_origin_logs": sorted(exposed),
        "scenes": candidates,
        "inputs": {k: {"path": str(p.resolve()), "sha256": sha256(p)} for k, p in {
            "inventory": args.inventory, "acceptance": args.acceptance,
            "backup": args.backup, "c_manifest": args.c_manifest,
            "exposed_manifest": args.exposed_manifest}.items()},
        "policy": {"exclusion": "origin_log_level", "split": "origin_log_80_10_10_v1",
                   "min_origin_logs": args.min_origin_logs,
                   "min_train_scenes": args.min_train_scenes},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"report already exists: {args.output}")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "output": str(args.output),
                      "candidate_origin_logs": len(candidates), "reasons": reasons}, sort_keys=True))
    return 0 if not reasons else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--acceptance", type=Path, required=True)
    p.add_argument("--backup", type=Path, required=True)
    p.add_argument("--c-manifest", type=Path, required=True)
    p.add_argument("--exposed-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-origin-logs", type=int, default=20)
    p.add_argument("--min-train-scenes", type=int, default=8)
    args = p.parse_args()
    raise SystemExit(audit(args))


if __name__ == "__main__":
    main()
