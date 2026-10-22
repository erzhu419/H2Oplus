#!/usr/bin/env python3
"""Seal scheduler completion and exact v20 remote/local synchronization evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
RESULTS = REPRO / "h2oplus_v20_full_control_results"
PARTITION = RESULTS / "evaluation_partition_v1/single_jtl110gpu"
EVIDENCE = RESULTS / "evidence/evaluation_v1"
REMOTE = Path(
    "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/"
    "h2oplus_v20_full_control_evaluation_v1/remote_results_v1/single_jtl110gpu"
)
SUBMISSION = REPRO / "h2oplus_v20_full_control_evaluation_submission_v1.json"
SUBMISSION_SHA256 = "801e601c00976590aca84fecd3957ac28d27ec1623e9f93bb33bce7a7d0c4372"
TASK_ID = "t83955"
EXPECTED_TOKEN = "1787705802108374398:365:140143939756032"
EXPECTED_SIGNATURE = (
    "H2Oplus/v20/development/full-control-eval/"
    "009ba0c8e572/single/jtl110gpu/glibc227"
)
EXPECTED_ENV = {
    "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
    "CC": (
        "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/"
        "x86_64-conda-linux-gnu-gcc"
    ),
    "CPATH": (
        "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/"
        "v7_formal_v2_training_v7r1/workspace/reproducibility/toolchain/"
        "libxcrypt-4.4.36-hd590300_1/include"
    ),
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
}
BLOCKS = tuple(f"v20-full-control-development-{index:02d}" for index in range(1, 4))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved.relative_to(ROOT)),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def atomic_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(text)
    os.replace(partial, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def inventory(root: Path) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    rows = []
    for path in sorted(item for item in resolved.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(resolved).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    blob = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode()
    return {
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "aggregate_sha256": hashlib.sha256(blob).hexdigest(),
    }


def queue_task(queue_path: Path) -> dict[str, Any]:
    data = json.loads(queue_path.read_text())
    matches = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("id") == TASK_ID and value.get("signature") == EXPECTED_SIGNATURE:
                matches.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(data)
    if len(matches) != 1:
        raise ValueError(f"expected one complete scheduler record for {TASK_ID}")
    return matches[0]


def validate_partition(root: Path) -> list[dict[str, Any]]:
    host = json.loads((root / "host_execution.json").read_text())
    summary = json.loads((root / "host_summary.json").read_text())
    results = sorted(root.rglob("sumo_seed*.json"))
    manifests = sorted(root.glob("*/matrix_manifest.json"))
    if (
        host.get("schema_version")
        != "transit-h2oplus-v20-full-control-host-execution-v1"
        or host.get("status") != "completed"
        or host.get("completed_results") != 405
        or host.get("expected_results") != 405
        or host.get("formal_held_out_outcomes_read") is not False
        or host.get("training") is not False
        or host.get("workers") != 12
        or len(host.get("completed_blocks", [])) != 3
        or summary.get("schema_version")
        != "transit-h2oplus-v20-full-control-host-summary-v1"
        or summary.get("status") != "completed"
        or summary.get("physical_evaluation_results") != 405
        or summary.get("formal_held_out_outcomes_read") is not False
        or summary.get("training") is not False
        or len(summary.get("matrix_manifests", [])) != 3
        or len(results) != 405
        or len(manifests) != 3
    ):
        raise ValueError("v20 synchronized partition contract failed")
    records = []
    for path in manifests:
        payload = json.loads(path.read_text())
        partition = payload.get("matrix_partition", {})
        block = path.parent.name
        if (
            block not in BLOCKS
            or payload.get("schema_version") != "transit-evaluation-matrix-v7"
            or len(payload.get("jobs", [])) != 135
            or len(payload.get("source_sha256", {})) != 57
            or partition.get("full_job_count") != 135
            or partition.get("selected_job_count") != 135
            or partition.get("shard_count") != 1
            or partition.get("shard_index") != 0
        ):
            raise ValueError(f"v20 matrix contract failed: {path}")
        records.append({"block": block, **descriptor(path), "jobs": 135})
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm"
    )
    parser.add_argument(
        "--queue", type=Path, default=Path.home() / ".claude/scheduler/queue.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if sha256(SUBMISSION) != SUBMISSION_SHA256:
        raise ValueError("v20 submission identity drifted")
    submission = json.loads(SUBMISSION.read_text())
    submitted = submission.get("task", {})
    if (
        submission.get("schema_version")
        != "transit-h2oplus-v20-full-control-evaluation-submission-v1"
        or submission.get("status") != "submitted"
        or submission.get("study_stage") != "development"
        or submission.get("formal_held_out_outcomes_read") is not False
        or submission.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
        or submission.get("total_remote_sumo_episodes") != 405
        or submission.get("task_count") != 1
        or submitted.get("task_id") != TASK_ID
        or submitted.get("signature") != EXPECTED_SIGNATURE
        or submitted.get("expected_jobs") != 405
        or Path(submitted.get("remote_result_dir", "")) != REMOTE
    ):
        raise ValueError("v20 submission contract failed")

    task = queue_task(args.queue.expanduser().resolve(strict=True))
    env = task.get("extra_env", {})
    if (
        task.get("status") != "done"
        or task.get("node") != "jtl110gpu"
        or task.get("require_node") != "jtl110gpu"
        or task.get("cpu_declared_cores") != 12
        or task.get("ram_mb") != 32768
        or task.get("est_vram_mb") != 0
        or task.get("exit_status_token") != EXPECTED_TOKEN
        or Path(task.get("result_dir", "")) != REMOTE
        or task.get("result_sync_error") is not None
        or not task.get("result_synced_at")
        or any(env.get(key) != value for key, value in EXPECTED_ENV.items())
        or "CUDA_VISIBLE_DEVICES" in env
    ):
        raise ValueError("v20 scheduler record or environment drifted")

    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    sys.path.insert(0, str(scheduler_root / "skill"))
    import scheduler  # type: ignore

    status_path = str(task["exit_status_path"])
    log_path = str(task["log_path"])
    remote_code = r'''import hashlib,json,os,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
rows=[]
for p in sorted(x for x in root.rglob('*') if x.is_file()):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 rows.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
blob=''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in rows).encode()
print(json.dumps({'file_count':len(rows),'total_bytes':sum(x['bytes'] for x in rows),'aggregate_sha256':hashlib.sha256(blob).hexdigest()},sort_keys=True))
names={'run_h2oplus_v20_full_control_evaluation_dispatch.py','run_evaluation_matrix_v20_full_control.py','evaluate_transit_v20_full_control.py'}
active=[]
for raw in os.listdir('/proc'):
 if not raw.isdigit(): continue
 try: argv=[x.decode('utf-8','replace') for x in open('/proc/'+raw+'/cmdline','rb').read().split(b'\0') if x]
 except OSError: continue
 scripts={os.path.basename(x) for x in argv if x.endswith('.py')}
 if scripts & names: active.append({'pid':int(raw),'scripts':sorted(scripts & names)})
print(json.dumps({'active_scientific_processes':active},sort_keys=True))'''
    command = (
        f"cat {shlex.quote(status_path)}; printf '\n__LOG__\n'; "
        f"cat {shlex.quote(log_path)}; printf '\n__INVENTORY__\n'; "
        f"/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python -c "
        f"{shlex.quote(remote_code)} {shlex.quote(str(REMOTE))}"
    )
    rc, out, err = scheduler.run_on("jtl110gpu", command, timeout=120, check=False)
    if rc != 0 or "\n__LOG__\n" not in out or "\n__INVENTORY__\n" not in out:
        raise RuntimeError(f"v20 final remote evidence failed: {err or out}")
    exit_text, remainder = out.split("\n__LOG__\n", 1)
    log_text, inventory_text = remainder.split("\n__INVENTORY__\n", 1)
    inventory_lines = [line for line in inventory_text.splitlines() if line.strip()]
    if len(inventory_lines) != 2:
        raise ValueError("v20 remote inventory response malformed")
    remote_inventory = json.loads(inventory_lines[0])
    process_evidence = json.loads(inventory_lines[1])
    fields = exit_text.strip().split("\t")
    if (
        len(fields) != 3
        or fields[0] != "0"
        or fields[2] != EXPECTED_TOKEN
        or fields[2] != task.get("exit_status_token")
        or int(fields[1]) < int(task.get("started_at", 0))
        or int(fields[1]) > int(task.get("finished_at", 0))
        or log_text.count("v20_block_started") != 3
        or log_text.count("v20_block_completed") != 3
        or log_text.count("\nDONE\n") != 1
        or "v20_dispatch_completed" not in log_text
        or any(marker in log_text.lower() for marker in ("traceback", "exception", "v20_dispatch_failed"))
        or process_evidence.get("active_scientific_processes") != []
    ):
        raise ValueError("v20 token-bound backend closure failed")

    manifests = validate_partition(PARTITION)
    local_inventory = inventory(PARTITION)
    if local_inventory != remote_inventory:
        raise ValueError("v20 remote/local file inventory mismatch")

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    backend_path = EVIDENCE / f"backend_exit_{TASK_ID}.status"
    log_evidence_path = EVIDENCE / f"scheduler_{TASK_ID}.log"
    source_sync_path = EVIDENCE / "source_sync_v1.json"
    scheduler_path = EVIDENCE / "scheduler_evidence_v1.json"
    atomic_text(backend_path, exit_text.strip() + "\n")
    atomic_text(log_evidence_path, log_text)
    source_sync = {
        "schema_version": "transit-h2oplus-v20-full-control-source-sync-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "task_id": TASK_ID,
        "remote_partition": str(REMOTE),
        "local_partition": str(PARTITION.resolve(strict=True)),
        "remote_inventory": remote_inventory,
        "local_inventory": local_inventory,
        "exact_remote_local_match": True,
        "physical_result_count": 405,
        "matrix_manifests": manifests,
        "formal_held_out_outcomes_read": False,
    }
    atomic_json(source_sync_path, source_sync)
    scheduler_evidence = {
        "schema_version": "transit-h2oplus-v20-full-control-scheduler-evidence-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "authorized_scientific_task": TASK_ID,
        "task": {
            "task_id": TASK_ID,
            "status": task.get("status"),
            "node": task.get("node"),
            "cpu_cores": task.get("cpu_declared_cores"),
            "ram_mb": task.get("ram_mb"),
            "vram_mb": task.get("est_vram_mb"),
            "signature": task.get("signature"),
            "started_at": task.get("started_at"),
            "finished_at": task.get("finished_at"),
            "peak_ram_mb": task.get("peak_ram_mb"),
            "result_synced_at": task.get("result_synced_at"),
        },
        "token_bound_backend_exit0": True,
        "backend_token": EXPECTED_TOKEN,
        "standalone_done_verified": True,
        "active_scientific_processes_after_completion": [],
        "backend_exit": descriptor(backend_path),
        "scheduler_log": descriptor(log_evidence_path),
        "source_sync": descriptor(source_sync_path),
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_authorized_or_submitted": False,
    }
    atomic_json(scheduler_path, scheduler_evidence)
    print(
        json.dumps(
            {
                "scheduler_evidence": descriptor(scheduler_path),
                "source_sync": descriptor(source_sync_path),
                "inventory": local_inventory,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
