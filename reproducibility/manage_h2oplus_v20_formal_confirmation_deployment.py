#!/usr/bin/env python3
"""Build, verify and synchronize the thin v20 formal deployment overlay."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
PARENT_WORKSPACE = REPRO / "deployments/h2oplus_v20_full_control_evaluation_v1/workspace"
OVERLAY_ROOT = REPRO / "deployments/h2oplus_v20_formal_confirmation_evaluation_v1/overlay"
REMOTE_WORKSPACE = REPRO / "deployments/h2oplus_v20_formal_confirmation_evaluation_v1/workspace"
PARENT_MANIFEST_SHA256 = (
    "009ba0c8e5725936fa984e6361e77d175801cc514e81af531c6db0a621137206"
)
PARENT_AGGREGATE_SHA256 = (
    "f504b4b23139931724ca874dae3a2db018a41f9ab83090f7e63046ee4f26960c"
)
EXECUTION_AUTHORIZATION_SHA256 = (
    "c9e2cdfe10074e5ddded7f756c682b9c5a63d3e5fc6fc52c87498289407226d5"
)
HASHES = {
    "execution_authorization": EXECUTION_AUTHORIZATION_SHA256,
    "preparation_authorization": "7a7b79f029a5437a80a31637c1bc057571849623b9b04a92a67bcf66586882c9",
    "rationale": "72e6d2961846e458721f121ae75eded862bed0556c929dc38015a3b2d9103a4d",
    "registration": "c33111cd72b741a801804cd065f42629094d45c0fe03a5b88f9d95dba62149ce",
    "protocol": "10aacced63f4c8a1b7f6f55024f46dfdebbcae6737c6bbafb79d227b2ad44664",
    "methods": "b4bcad09e13cb50c94d5541f2a52340ce99f15b281b5ddef57be38e792d052cd",
    "environment": "50b831923b76a81c773cf7dc19dcc2e10c729ea8fc7b3be6982fc01b7c3cc141",
    "design_audit": "f0f7f68d8f1273f49e55b1444e5c74282b6aeaf68efaa2f8a0aa0f4261408757",
    "design_verification": "16d33c2b12f280ed8d1595afca91b52043a8a87a844aa7a4cda0227e574f6e29",
    "source_freeze": "70f387eeb232d5021f33b12ce5e653d0dea500b905827d7c8d525fb738e23c25",
    "minimal_sync_policy": "20b313b7b8cc13a0a8df7c2a137a4b0713c621a33ef8e2e314fb69d03de1a531",
    "parent_v20_analysis": "f7de8ca7ea82098a89383f1fe0e9e234f8c1acab1c030d66494100744cd9455b",
    "parent_v20_audit": "2e516ab3ea0cff10dc88607d6fb8adf3e3888a822e32a14f84a86953a466f583",
    "parent_v20_closeout": "dbd1da5442219f404f90da6fbc415b0755f7dddd41d28b9b581039d359882132",
    "parent_v20_verification": "08b25f8b229a9fefbfec198f8198aff5db4f9f2680f16d9bf5521985cdfe7f7e",
    "dataset": "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899",
    "runtime_lock": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
}
REPO_FILES = (
    "SimpleSAC/evaluate_transit_v20_formal_confirmation.py",
    "reproducibility/run_evaluation_matrix_v20_formal_confirmation.py",
    "reproducibility/analyze_h2oplus_v20_formal_confirmation.py",
    "reproducibility/prepare_h2oplus_v20_formal_confirmation.py",
    "reproducibility/audit_h2oplus_v20_formal_confirmation_design.py",
    "reproducibility/verify_h2oplus_v20_formal_confirmation_design.py",
    "reproducibility/build_h2oplus_v20_formal_confirmation_source_freeze.py",
    "reproducibility/sync_h2oplus_v20_formal_confirmation_results.py",
    "reproducibility/run_h2oplus_v20_formal_confirmation_dispatch.py",
    "reproducibility/manage_h2oplus_v20_formal_confirmation_deployment.py",
    "reproducibility/h2oplus_v20_formal_confirmation_execution_authorization_v1.json",
    "reproducibility/h2oplus_v20_formal_confirmation_preparation_authorization_v1.json",
    "reproducibility/h2oplus_v20_formal_confirmation_rationale_v1.json",
    "reproducibility/h2oplus_v20_formal_confirmation_registration_v1.json",
    "reproducibility/h2oplus_v20_formal_confirmation_protocol_v1.json",
    "reproducibility/h2oplus_v20_formal_confirmation_evaluation_methods_v1.json",
    "reproducibility/h2oplus_v20_formal_confirmation_environment_v1.json",
    "reproducibility/h2oplus_v20_formal_confirmation_design_audit_v1.json",
    "reproducibility/h2oplus_v20_formal_confirmation_design_verification_v1.json",
    "reproducibility/h2oplus_v20_formal_confirmation_source_freeze_v1.json",
    "reproducibility/h2oplus_minimal_remote_sync_policy_v1.json",
    "reproducibility/h2oplus_v20_full_control_results/full_control_analysis_v1.json",
    "reproducibility/h2oplus_v20_full_control_results/evaluation_audit_v1.json",
    "reproducibility/h2oplus_v20_full_control_results/evaluation_closeout_v1.json",
    "reproducibility/h2oplus_v20_full_control_results/evaluation_closeout_verification_v1.json",
    "reproducibility/h2oplus_v20_full_control_results/evaluation_file_manifest_verification_v1.json",
)
SCENARIOS = "bus_h2o/formal_confirmation_scenarios_v20/v1"
WORKSPACE_FILES = {
    "SUMO_ruiguang/online_control/intersection_delay/a_sorted_busline_edge.xml": (
        ROOT.parent
        / "SUMO_ruiguang/online_control/intersection_delay/a_sorted_busline_edge.xml"
    )
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path, root: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def aggregate(entries: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in sorted(entries, key=lambda item: item["path"]):
        digest.update(
            (
                row["path"]
                + "\0"
                + str(row["size_bytes"])
                + "\0"
                + row["sha256"]
                + "\n"
            ).encode()
        )
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def validate_bound_inputs() -> dict:
    paths = {
        "execution_authorization": REPRO / "h2oplus_v20_formal_confirmation_execution_authorization_v1.json",
        "preparation_authorization": REPRO / "h2oplus_v20_formal_confirmation_preparation_authorization_v1.json",
        "rationale": REPRO / "h2oplus_v20_formal_confirmation_rationale_v1.json",
        "registration": REPRO / "h2oplus_v20_formal_confirmation_registration_v1.json",
        "protocol": REPRO / "h2oplus_v20_formal_confirmation_protocol_v1.json",
        "methods": REPRO / "h2oplus_v20_formal_confirmation_evaluation_methods_v1.json",
        "environment": REPRO / "h2oplus_v20_formal_confirmation_environment_v1.json",
        "design_audit": REPRO / "h2oplus_v20_formal_confirmation_design_audit_v1.json",
        "design_verification": REPRO / "h2oplus_v20_formal_confirmation_design_verification_v1.json",
        "source_freeze": REPRO / "h2oplus_v20_formal_confirmation_source_freeze_v1.json",
        "minimal_sync_policy": REPRO / "h2oplus_minimal_remote_sync_policy_v1.json",
        "parent_v20_analysis": REPRO / "h2oplus_v20_full_control_results/full_control_analysis_v1.json",
        "parent_v20_audit": REPRO / "h2oplus_v20_full_control_results/evaluation_audit_v1.json",
        "parent_v20_closeout": REPRO / "h2oplus_v20_full_control_results/evaluation_closeout_v1.json",
        "parent_v20_verification": REPRO / "h2oplus_v20_full_control_results/evaluation_closeout_verification_v1.json",
        "dataset": ROOT / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "runtime_lock": REPRO / "h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
    }
    for name, path in paths.items():
        observed = sha256(path.resolve(strict=True))
        if observed != HASHES[name]:
            raise ValueError(f"formal deployment input drifted: {name}: {observed}")
    authorization = json.loads(paths["execution_authorization"].read_text())
    scope = authorization.get("authorization_scope", {})
    closeout = json.loads(paths["parent_v20_closeout"].read_text())
    verification = json.loads(paths["parent_v20_verification"].read_text())
    freeze = json.loads(paths["source_freeze"].read_text())
    if (
        scope.get("formal_confirmation_submission_authorized") is not True
        or scope.get("formal_held_out_outcome_access_authorized") is not True
        or scope.get("remote_sumo_episode_cap") != 1350
        or closeout.get("formal_confirmation_eligible") is not True
        or verification.get("formal_confirmation_eligible") is not True
        or freeze.get("formal_confirmation_submission_authorized") is not False
        or freeze.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("formal deployment authorization boundary drifted")
    return {"paths": paths, "source_freeze": freeze}


def build(args: argparse.Namespace) -> int:
    checked = validate_bound_inputs()
    parent = args.parent_workspace.expanduser().resolve(strict=True)
    parent_manifest_path = parent / "deployment_manifest.json"
    if sha256(parent_manifest_path) != PARENT_MANIFEST_SHA256:
        raise ValueError("v20 parent deployment manifest drifted")
    parent_manifest = json.loads(parent_manifest_path.read_text())
    if (
        parent_manifest.get("aggregate_sha256") != PARENT_AGGREGATE_SHA256
        or parent_manifest.get("entry_count") != 1015
    ):
        raise ValueError("v20 parent deployment scope drifted")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace formal overlay: {output}")
    try:
        repo = output / "H2Oplus"
        for relative in REPO_FILES:
            source = (ROOT / relative).resolve(strict=True)
            destination = repo / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        for relative, raw_source in WORKSPACE_FILES.items():
            source = raw_source.resolve(strict=True)
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        scenario_source = (ROOT / SCENARIOS).resolve(strict=True)
        scenario_destination = repo / SCENARIOS
        shutil.copytree(scenario_source, scenario_destination, copy_function=os.link)
        overlay_entries = [
            descriptor(path, output)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        ]
        effective = {row["path"]: row for row in parent_manifest["entries"]}
        effective.update({row["path"]: row for row in overlay_entries})
        entries = [effective[path] for path in sorted(effective)]
        freeze = checked["source_freeze"]
        source_map = {row["path"]: row["sha256"] for row in freeze["matrix_sources"]}
        source_map.update(
            {row["path"]: row["sha256"] for row in freeze["operational_sources"]}
        )
        for relative, digest in source_map.items():
            effective_path = (
                relative
                if relative.startswith("SUMO_ruiguang/")
                else "H2Oplus/" + relative
            )
            if effective.get(effective_path, {}).get("sha256") != digest:
                raise ValueError(f"formal frozen source missing from deployment: {relative}")
        paths = checked["paths"]

        def bound(name: str) -> dict:
            return {
                "path": "H2Oplus/" + paths[name].relative_to(ROOT).as_posix(),
                "sha256": HASHES[name],
            }

        manifest = {
            "schema_version": "transit-h2oplus-v20-formal-confirmation-deployment-manifest-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "study_stage": "formal-confirmation",
            "matrix_type": "endpoint-only-formal-confirmation",
            "entry_count": len(entries),
            "total_file_bytes": sum(row["size_bytes"] for row in entries),
            "aggregate_sha256": aggregate(entries),
            "overlay_entry_count": len(overlay_entries),
            "overlay_total_file_bytes": sum(row["size_bytes"] for row in overlay_entries),
            "overlay_aggregate_sha256": aggregate(overlay_entries),
            "physical_behavior_count": 45,
            "formal_environment_block_count": 10,
            "formal_environment_cell_count": 30,
            "jobs_per_block": 135,
            "expected_remote_sumo_episodes": 1350,
            "new_training_runs": 0,
            "formal_confirmation_submission_authorized": True,
            "formal_held_out_outcome_access_authorized": True,
            "formal_held_out_outcomes_read": False,
            "sumo_runtime_signature": parent_manifest["sumo_runtime_signature"],
            "parent_workspace_reused_by_remote_hardlinks": True,
            "checkpoint_optimizer_or_replay_transferred_in_overlay": False,
            "minimal_result_sync_required": True,
            "parent_v20_deployment": {
                "path": str(parent),
                "manifest_sha256": PARENT_MANIFEST_SHA256,
                "aggregate_sha256": PARENT_AGGREGATE_SHA256,
                "entry_count": 1015,
            },
            "execution_authorization": bound("execution_authorization"),
            "preparation_authorization": bound("preparation_authorization"),
            "rationale": bound("rationale"),
            "registration": bound("registration"),
            "protocol": bound("protocol"),
            "methods_manifest": bound("methods"),
            "environment": bound("environment"),
            "design_audit": bound("design_audit"),
            "design_verification": bound("design_verification"),
            "source_freeze": bound("source_freeze"),
            "minimal_sync_policy": bound("minimal_sync_policy"),
            "parent_v20_analysis": bound("parent_v20_analysis"),
            "parent_v20_audit": bound("parent_v20_audit"),
            "parent_v20_closeout": bound("parent_v20_closeout"),
            "parent_v20_verification": bound("parent_v20_verification"),
            "dataset": {
                "path": "H2Oplus/" + paths["dataset"].relative_to(ROOT).as_posix(),
                "sha256": HASHES["dataset"],
                "inherited_from_parent": True,
            },
            "runtime_lock": bound("runtime_lock"),
            "matrix_source_sha256": {
                row["path"]: row["sha256"] for row in freeze["matrix_sources"]
            },
            "operational_source_sha256": {
                row["path"]: row["sha256"] for row in freeze["operational_sources"]
            },
            "overlay_entries": overlay_entries,
            "entries": entries,
        }
        manifest_path = output / "deployment_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "output": str(output),
                    "manifest_sha256": sha256(manifest_path),
                    "aggregate_sha256": manifest["aggregate_sha256"],
                    "entry_count": manifest["entry_count"],
                    "overlay_entry_count": manifest["overlay_entry_count"],
                    "overlay_total_file_bytes": manifest["overlay_total_file_bytes"],
                },
                sort_keys=True,
            )
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return 0


def load_manifest(path: Path, expected_sha: str, expected_aggregate: str, expected_count: int) -> dict:
    if sha256(path) != expected_sha:
        raise ValueError("formal deployment manifest identity drifted")
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema_version")
        != "transit-h2oplus-v20-formal-confirmation-deployment-manifest-v1"
        or manifest.get("aggregate_sha256") != expected_aggregate
        or manifest.get("entry_count") != expected_count
        or aggregate(manifest.get("entries", [])) != expected_aggregate
        or manifest.get("expected_remote_sumo_episodes") != 1350
        or manifest.get("formal_confirmation_submission_authorized") is not True
        or manifest.get("formal_held_out_outcome_access_authorized") is not True
    ):
        raise ValueError("formal deployment manifest scope drifted")
    return manifest


def verify_effective(args: argparse.Namespace) -> int:
    workspace = args.workspace.expanduser().resolve(strict=True)
    manifest_path = workspace / "deployment_manifest.json"
    manifest = load_manifest(
        manifest_path,
        args.expected_manifest_sha256,
        args.expected_aggregate_sha256,
        args.expected_entry_count,
    )
    expected = {row["path"] for row in manifest["entries"]}
    actual = {
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != expected:
        raise ValueError(
            "formal effective deployment membership drifted: "
            f"missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}"
        )
    for row in manifest["entries"]:
        path = workspace / row["path"]
        if path.stat().st_size != row["size_bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"formal effective deployment file drifted: {row['path']}")
    report = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-effective-deployment-verification-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "manifest_sha256": args.expected_manifest_sha256,
        "aggregate_sha256": args.expected_aggregate_sha256,
        "entry_count": args.expected_entry_count,
        "formal_held_out_outcomes_read": False,
        "sumo_executed": False,
        "passed": True,
    }
    print(json.dumps(report, sort_keys=True))
    return 0


def verify_local(args: argparse.Namespace) -> int:
    validate_bound_inputs()
    overlay = args.overlay.expanduser().resolve(strict=True)
    manifest = load_manifest(
        overlay / "deployment_manifest.json",
        args.expected_manifest_sha256,
        args.expected_aggregate_sha256,
        args.expected_entry_count,
    )
    overlay_entries = manifest["overlay_entries"]
    actual = {
        path.relative_to(overlay).as_posix()
        for path in overlay.rglob("*")
        if path.is_file() and path.name != "deployment_manifest.json"
    }
    if actual != {row["path"] for row in overlay_entries}:
        raise ValueError("formal local overlay membership drifted")
    for row in overlay_entries:
        path = overlay / row["path"]
        if path.stat().st_size != row["size_bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"formal local overlay file drifted: {row['path']}")
    parent_manifest = args.parent_workspace.expanduser().resolve(strict=True) / "deployment_manifest.json"
    if sha256(parent_manifest) != PARENT_MANIFEST_SHA256:
        raise ValueError("formal local parent deployment drifted")
    report = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-local-deployment-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "overlay": str(overlay),
        "manifest_sha256": args.expected_manifest_sha256,
        "aggregate_sha256": args.expected_aggregate_sha256,
        "entry_count": args.expected_entry_count,
        "overlay_entry_count": len(overlay_entries),
        "overlay_total_file_bytes": sum(row["size_bytes"] for row in overlay_entries),
        "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "checkpoint_optimizer_or_replay_copied_into_overlay": False,
        "formal_held_out_outcomes_read": False,
        "local_sumo_executed": False,
    }
    atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0


def sync(args: argparse.Namespace) -> int:
    overlay = args.overlay.expanduser().resolve(strict=True)
    manifest = load_manifest(
        overlay / "deployment_manifest.json",
        args.expected_manifest_sha256,
        args.expected_aggregate_sha256,
        args.expected_entry_count,
    )
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    sys.path.insert(0, str(scheduler_root))
    scheduler = importlib.import_module("skill.scheduler")
    target = scheduler._ssh_target_for_node(args.node)
    shell = scheduler._ssh_rsync_shell_for_node(args.node)
    parent = args.parent_workspace.expanduser().resolve(strict=True)
    remote = args.remote_workspace.expanduser()
    clone = (
        f"set -eu; test \"$(sha256sum {shlex.quote(str(parent / 'deployment_manifest.json'))} "
        f"| awk '{{print $1}}')\" = {shlex.quote(PARENT_MANIFEST_SHA256)}; "
        f"if test -e {shlex.quote(str(remote))}; then "
        f"test -f {shlex.quote(str(remote / 'deployment_manifest.json'))}; "
        f"else mkdir -p {shlex.quote(str(remote.parent))}; "
        f"cp -al {shlex.quote(str(parent))} {shlex.quote(str(remote))}; "
        f"rm {shlex.quote(str(remote / 'deployment_manifest.json'))}; fi"
    )
    rc, stdout, stderr = scheduler.run_on(args.node, clone, timeout=120, check=False)
    if rc != 0:
        raise RuntimeError(f"remote v20-to-formal hardlink clone failed: {stderr or stdout}")
    command = [
        "rsync",
        "-a",
        "--checksum",
        "--partial",
        "--delay-updates",
        "--omit-dir-times",
        "-e",
        shell,
        str(overlay) + "/",
        f"{target}:{remote}/",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"formal overlay rsync failed: {completed.stderr.strip()}")
    python = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
    manager = remote / "H2Oplus/reproducibility/manage_h2oplus_v20_formal_confirmation_deployment.py"
    verify = shlex.join(
        [
            python,
            str(manager),
            "verify-effective",
            "--workspace",
            str(remote),
            "--expected-manifest-sha256",
            args.expected_manifest_sha256,
            "--expected-aggregate-sha256",
            args.expected_aggregate_sha256,
            "--expected-entry-count",
            str(args.expected_entry_count),
        ]
    )
    env = (
        "export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 "
        "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcurl.so.4 "
        "CC=/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc "
        "CPATH=/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/"
        "v7_formal_v2_training_v7r1/workspace/reproducibility/toolchain/"
        "libxcrypt-4.4.36-hd590300_1/include "
        "CUDA_VISIBLE_DEVICES='' "
        "H2OPLUS_V19_COMPLETION_SAFETY_PROTOCOL_SHA256="
        "fa5f629588d35b1435e28bd39931f49d9851811ea3e308d62672f9eacf1f7184 "
        "H2OPLUS_V20_FORMAL_CONFIRMATION_PROTOCOL_SHA256="
        "10aacced63f4c8a1b7f6f55024f46dfdebbcae6737c6bbafb79d227b2ad44664 "
        f"PYTHONPATH={shlex.quote(str(remote / 'H2Oplus'))}; "
    )
    rc, verify_stdout, verify_stderr = scheduler.run_on(
        args.node, env + verify, timeout=900, check=False
    )
    if rc != 0:
        raise RuntimeError(
            f"remote formal effective verification failed: {verify_stderr or verify_stdout}"
        )
    smoke_commands = [
        [python, str(remote / "H2Oplus/reproducibility/run_h2oplus_v20_formal_confirmation_dispatch.py"), "--help"],
        [python, str(remote / "H2Oplus/reproducibility/run_evaluation_matrix_v20_formal_confirmation.py"), "--help"],
        [
            python,
            "-c",
            (
                "import SimpleSAC.evaluate_transit_v20_formal_confirmation as module; "
                "assert callable(module.main)"
            ),
        ],
    ]
    smoke = " && ".join(shlex.join(command) + " >/dev/null" for command in smoke_commands)
    rc, smoke_stdout, smoke_stderr = scheduler.run_on(
        args.node, env + smoke, timeout=180, check=False
    )
    if rc != 0:
        raise RuntimeError(
            f"remote formal Python 3.8 no-SUMO smoke test failed: {smoke_stderr or smoke_stdout}"
        )
    sync_record = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-deployment-sync-v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "node": args.node,
        "passed": True,
        "transport": "remote-v20-hardlink-clone-plus-thin-overlay-rsync",
        "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "manifest_sha256": args.expected_manifest_sha256,
        "aggregate_sha256": args.expected_aggregate_sha256,
        "entry_count": args.expected_entry_count,
        "overlay_entry_count": manifest["overlay_entry_count"],
        "overlay_total_file_bytes": manifest["overlay_total_file_bytes"],
        "checkpoint_optimizer_or_replay_transferred": False,
        "formal_held_out_outcomes_read": False,
    }
    remote_record = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-jtl110gpu-verification-v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "node": args.node,
        "passed": True,
        "python_version": "3.8.20",
        "manifest_sha256": args.expected_manifest_sha256,
        "aggregate_sha256": args.expected_aggregate_sha256,
        "entry_count": args.expected_entry_count,
        "effective_verification": json.loads(verify_stdout.strip().splitlines()[-1]),
        "no_sumo_smoke_test": True,
        "remote_sumo_executed": False,
        "formal_held_out_outcomes_read": False,
    }
    atomic_json(args.sync_record.expanduser().resolve(), sync_record)
    atomic_json(args.remote_verification_record.expanduser().resolve(), remote_record)
    print(json.dumps({"sync": sync_record, "remote_verification": remote_record}, sort_keys=True))
    return 0


def common_manifest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-aggregate-sha256", required=True)
    parser.add_argument("--expected-entry-count", type=int, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--parent-workspace", type=Path, default=PARENT_WORKSPACE)
    build_parser.add_argument("--output", type=Path, default=OVERLAY_ROOT)
    local_parser = sub.add_parser("verify-local")
    local_parser.add_argument("--overlay", type=Path, default=OVERLAY_ROOT)
    local_parser.add_argument("--parent-workspace", type=Path, default=PARENT_WORKSPACE)
    local_parser.add_argument("--output", type=Path, required=True)
    common_manifest_args(local_parser)
    effective_parser = sub.add_parser("verify-effective")
    effective_parser.add_argument("--workspace", type=Path, required=True)
    common_manifest_args(effective_parser)
    sync_parser = sub.add_parser("sync")
    sync_parser.add_argument("--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm")
    sync_parser.add_argument("--node", choices=("jtl110gpu",), default="jtl110gpu")
    sync_parser.add_argument("--overlay", type=Path, default=OVERLAY_ROOT)
    sync_parser.add_argument("--parent-workspace", type=Path, default=PARENT_WORKSPACE)
    sync_parser.add_argument("--remote-workspace", type=Path, default=REMOTE_WORKSPACE)
    sync_parser.add_argument("--sync-record", type=Path, required=True)
    sync_parser.add_argument("--remote-verification-record", type=Path, required=True)
    common_manifest_args(sync_parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build":
        return build(args)
    if args.command == "verify-local":
        return verify_local(args)
    if args.command == "verify-effective":
        return verify_effective(args)
    return sync(args)


if __name__ == "__main__":
    raise SystemExit(main())
