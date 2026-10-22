#!/usr/bin/env python3
"""Build and verify the path-corrected v20 formal deployment."""

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
V1_DEPLOYMENT = REPRO / "deployments/h2oplus_v20_formal_confirmation_evaluation_v1"
V1_OVERLAY = V1_DEPLOYMENT / "overlay"
V1_REMOTE_WORKSPACE = V1_DEPLOYMENT / "workspace"
V2_DEPLOYMENT = REPRO / "deployments/h2oplus_v20_formal_confirmation_evaluation_v2"
V2_OVERLAY = V2_DEPLOYMENT / "overlay"
V2_REMOTE_WORKSPACE = V2_DEPLOYMENT / "workspace"
PARENT_WORKSPACE = REPRO / "deployments/h2oplus_v20_full_control_evaluation_v1/workspace"
SOURCE_FREEZE = REPRO / "h2oplus_v20_formal_confirmation_source_freeze_v1.json"
CORRECTION = REPRO / "h2oplus_v20_formal_confirmation_deployment_path_correction_v2.json"
INCIDENT_DIR = (
    REPRO
    / "h2oplus_v20_formal_confirmation_results/evaluation_incidents/"
    "v1_deployment_path_mapping"
)
V1_MANIFEST_SHA256 = "91eef065317f1ba8a17fca839f8ce84371555fbb581a515eb474cc8d4bc7cd17"
V1_AGGREGATE_SHA256 = "0bbce7865eaf72f49eb8846054aeb9c6a6e8cc356d763b7e47859dc18a6e203e"
SOURCE_FREEZE_SHA256 = "70f387eeb232d5021f33b12ce5e653d0dea500b905827d7c8d525fb738e23c25"
CORRECTION_SHA256 = "144ce914c4cf915e181ca2f1b67711e46b7a8f6ae9d8ab97a2364bef77bdb66e"
INCIDENT_CLOSEOUT_SHA256 = "4800eb9cdb792df562a1a4e55e636e84e44737ed94bcc6ca10e16f2bc549b783"
INCIDENT_VERIFICATION_SHA256 = "dfe9737ee9440130d6e038db04c8ea6cb272fd11f0cb3c5325b8b32328bdb922"


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


def load_json(path: Path, expected_sha256: str) -> dict:
    resolved = path.expanduser().resolve(strict=True)
    observed = sha256(resolved)
    if observed != expected_sha256:
        raise ValueError(f"evidence drifted: {resolved}: {observed}")
    return json.loads(resolved.read_text())


def source_rows() -> list[dict]:
    freeze = load_json(SOURCE_FREEZE, SOURCE_FREEZE_SHA256)
    rows = freeze.get("matrix_sources", []) + freeze.get("operational_sources", [])
    if len(freeze.get("matrix_sources", [])) != 57:
        raise ValueError("formal matrix source count drifted")
    return rows


def sumo_source_rows() -> list[dict]:
    rows = [row for row in source_rows() if row["path"].startswith("SUMO_ruiguang/")]
    if len(rows) != 13:
        raise ValueError(f"expected 13 frozen SUMO source aliases, observed {len(rows)}")
    return rows


def validate_incident() -> None:
    correction = load_json(CORRECTION, CORRECTION_SHA256)
    closeout = load_json(INCIDENT_DIR / "incident_closeout_v1.json", INCIDENT_CLOSEOUT_SHA256)
    verification = load_json(
        INCIDENT_DIR / "incident_verification_v1.json", INCIDENT_VERIFICATION_SHA256
    )
    if (
        correction.get("scientific_behavior_changed") is not False
        or correction.get("v1_incident_results_reused") is not False
        or correction.get("authorization_basis", {}).get(
            "one_fresh_successor_submission_authorized"
        )
        is not True
        or closeout.get("completed_formal_result_count") != 0
        or closeout.get("remote_sumo_episode_count") != 0
        or closeout.get("formal_held_out_outcomes_read") is not False
        or verification.get("passed") is not True
    ):
        raise ValueError("formal v1 incident or correction boundary drifted")


def load_manifest(path: Path, manifest_sha256: str, aggregate_sha256: str, count: int) -> dict:
    manifest = load_json(path, manifest_sha256)
    entries = manifest.get("entries", [])
    if (
        manifest.get("aggregate_sha256") != aggregate_sha256
        or manifest.get("entry_count") != count
        or len(entries) != count
        or aggregate(entries) != aggregate_sha256
    ):
        raise ValueError("deployment manifest scope drifted")
    return manifest


def build(args: argparse.Namespace) -> int:
    validate_incident()
    v1_manifest = load_manifest(
        V1_OVERLAY / "deployment_manifest.json",
        V1_MANIFEST_SHA256,
        V1_AGGREGATE_SHA256,
        1252,
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace corrected deployment: {output}")
    shutil.copytree(V1_OVERLAY, output, copy_function=os.link)
    (output / "deployment_manifest.json").unlink()
    repo = output / "H2Oplus"
    evidence = {
        "reproducibility/h2oplus_v20_formal_confirmation_deployment_path_correction_v2.json": CORRECTION,
        (
            "reproducibility/h2oplus_v20_formal_confirmation_results/evaluation_incidents/"
            "v1_deployment_path_mapping/incident_closeout_v1.json"
        ): INCIDENT_DIR / "incident_closeout_v1.json",
        (
            "reproducibility/h2oplus_v20_formal_confirmation_results/evaluation_incidents/"
            "v1_deployment_path_mapping/incident_verification_v1.json"
        ): INCIDENT_DIR / "incident_verification_v1.json",
        "reproducibility/manage_h2oplus_v20_formal_confirmation_deployment_v2.py": Path(__file__),
    }
    for relative, source in evidence.items():
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source.resolve(strict=True), destination)
    for row in sumo_source_rows():
        overlay_source = V1_OVERLAY / row["path"]
        source = (
            overlay_source if overlay_source.exists() else PARENT_WORKSPACE / row["path"]
        ).resolve(strict=True)
        if source.stat().st_size != row["bytes"] or sha256(source) != row["sha256"]:
            raise ValueError(f"frozen SUMO alias source drifted: {row['path']}")
        destination = output / "H2Oplus" / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)
    overlay_entries = [
        descriptor(path, output)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    effective = {row["path"]: row for row in v1_manifest["entries"]}
    effective.update({row["path"]: row for row in overlay_entries})
    entries = [effective[path] for path in sorted(effective)]
    manifest = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-deployment-manifest-v2",
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
        "scientific_sources_changed_from_v1": False,
        "scientific_behavior_changed_from_v1": False,
        "v1_incident_results_reused": False,
        "path_alias_count": 13,
        "path_aliases": ["H2Oplus/" + row["path"] for row in sumo_source_rows()],
        "parent_v1_deployment": {
            "path": str(V1_REMOTE_WORKSPACE),
            "manifest_sha256": V1_MANIFEST_SHA256,
            "aggregate_sha256": V1_AGGREGATE_SHA256,
            "entry_count": 1252,
        },
        "deployment_correction": {
            "path": "H2Oplus/reproducibility/"
            "h2oplus_v20_formal_confirmation_deployment_path_correction_v2.json",
            "sha256": CORRECTION_SHA256,
        },
        "v1_incident_closeout_sha256": INCIDENT_CLOSEOUT_SHA256,
        "v1_incident_verification_sha256": INCIDENT_VERIFICATION_SHA256,
        "sumo_runtime_signature": v1_manifest["sumo_runtime_signature"],
        "minimal_result_sync_required": True,
        "checkpoint_optimizer_or_replay_transferred_in_correction": False,
        "matrix_source_sha256": v1_manifest["matrix_source_sha256"],
        "operational_source_sha256": v1_manifest["operational_source_sha256"],
        "entries": entries,
        "overlay_entries": overlay_entries,
    }
    atomic_json(output / "deployment_manifest.json", manifest)
    print(
        json.dumps(
            {
                "manifest_path": str(output / "deployment_manifest.json"),
                "manifest_sha256": sha256(output / "deployment_manifest.json"),
                "aggregate_sha256": manifest["aggregate_sha256"],
                "entry_count": manifest["entry_count"],
                "overlay_entry_count": manifest["overlay_entry_count"],
                "overlay_total_file_bytes": manifest["overlay_total_file_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


def verify_local(args: argparse.Namespace) -> int:
    validate_incident()
    overlay = args.overlay.expanduser().resolve(strict=True)
    manifest_path = overlay / "deployment_manifest.json"
    manifest = load_manifest(
        manifest_path,
        args.expected_manifest_sha256,
        args.expected_aggregate_sha256,
        args.expected_entry_count,
    )
    actual = {
        path.relative_to(overlay).as_posix()
        for path in overlay.rglob("*")
        if path.is_file() and path != manifest_path
    }
    expected = {row["path"] for row in manifest["overlay_entries"]}
    if actual != expected:
        raise ValueError("corrected local overlay membership drifted")
    for row in manifest["overlay_entries"]:
        path = overlay / row["path"]
        if path.stat().st_size != row["size_bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"corrected local overlay file drifted: {row['path']}")
    aliases = {"H2Oplus/" + row["path"]: row for row in sumo_source_rows()}
    if set(manifest.get("path_aliases", [])) != set(aliases):
        raise ValueError("corrected alias membership drifted")
    for relative, row in aliases.items():
        path = overlay / relative
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"corrected alias drifted: {relative}")
    report = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-deployment-local-audit-v2",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "manifest_sha256": args.expected_manifest_sha256,
        "aggregate_sha256": args.expected_aggregate_sha256,
        "entry_count": args.expected_entry_count,
        "overlay_entry_count": len(manifest["overlay_entries"]),
        "path_alias_count": 13,
        "scientific_sources_changed_from_v1": False,
        "v1_incident_results_reused": False,
        "checkpoint_optimizer_or_replay_copied_for_correction": False,
        "formal_held_out_outcomes_read": False,
        "local_sumo_executed": False,
    }
    atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0


def verify_effective(args: argparse.Namespace) -> int:
    workspace = args.workspace.expanduser().resolve(strict=True)
    manifest_path = workspace / "deployment_manifest.json"
    manifest = load_manifest(
        manifest_path,
        args.expected_manifest_sha256,
        args.expected_aggregate_sha256,
        args.expected_entry_count,
    )
    actual = {
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if path.is_file() and path != manifest_path
    }
    expected = {row["path"] for row in manifest["entries"]}
    if actual != expected:
        raise ValueError(
            "corrected effective membership drifted: "
            f"missing={sorted(expected - actual)[:5]} extra={sorted(actual - expected)[:5]}"
        )
    for row in manifest["entries"]:
        path = workspace / row["path"]
        if path.stat().st_size != row["size_bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"corrected effective file drifted: {row['path']}")
    repo = workspace / "H2Oplus"
    for row in source_rows():
        path = repo / row["path"]
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"frozen source-root contract drifted: {row['path']}")
    report = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-effective-deployment-verification-v2",
        "passed": True,
        "manifest_sha256": args.expected_manifest_sha256,
        "aggregate_sha256": args.expected_aggregate_sha256,
        "entry_count": args.expected_entry_count,
        "source_root_file_count": len(source_rows()),
        "path_alias_count": 13,
        "formal_held_out_outcomes_read": False,
        "remote_sumo_executed": False,
    }
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
    remote = args.remote_workspace.expanduser()
    clone = (
        f"set -eu; test \"$(sha256sum {shlex.quote(str(V1_REMOTE_WORKSPACE / 'deployment_manifest.json'))} "
        f"| awk '{{print $1}}')\" = {shlex.quote(V1_MANIFEST_SHA256)}; "
        f"test ! -e {shlex.quote(str(remote))}; "
        f"mkdir -p {shlex.quote(str(remote.parent))}; "
        f"cp -al {shlex.quote(str(V1_REMOTE_WORKSPACE))} {shlex.quote(str(remote))}; "
        f"rm {shlex.quote(str(remote / 'deployment_manifest.json'))}"
    )
    rc, stdout, stderr = scheduler.run_on(args.node, clone, timeout=120, check=False)
    if rc != 0:
        raise RuntimeError(f"remote v1-to-v2 hardlink clone failed: {stderr or stdout}")
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
        raise RuntimeError(f"corrected formal overlay rsync failed: {completed.stderr.strip()}")
    python = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
    manager = remote / "H2Oplus/reproducibility/manage_h2oplus_v20_formal_confirmation_deployment_v2.py"
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
            f"corrected formal effective verification failed: {verify_stderr or verify_stdout}"
        )
    smoke_commands = [
        [python, str(remote / "H2Oplus/reproducibility/run_h2oplus_v20_formal_confirmation_dispatch.py"), "--help"],
        [python, str(remote / "H2Oplus/reproducibility/run_evaluation_matrix_v20_formal_confirmation.py"), "--help"],
        [
            python,
            "-c",
            "import SimpleSAC.evaluate_transit_v20_formal_confirmation as module; assert callable(module.main)",
        ],
    ]
    smoke = " && ".join(shlex.join(item) + " >/dev/null" for item in smoke_commands)
    rc, smoke_stdout, smoke_stderr = scheduler.run_on(
        args.node, env + smoke, timeout=180, check=False
    )
    if rc != 0:
        raise RuntimeError(
            f"corrected formal Python 3.8 no-SUMO smoke failed: {smoke_stderr or smoke_stdout}"
        )
    sync_record = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-deployment-sync-v2",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "node": args.node,
        "passed": True,
        "transport": "remote-v1-formal-hardlink-clone-plus-path-correction-overlay",
        "parent_v1_manifest_sha256": V1_MANIFEST_SHA256,
        "manifest_sha256": args.expected_manifest_sha256,
        "aggregate_sha256": args.expected_aggregate_sha256,
        "entry_count": args.expected_entry_count,
        "path_alias_count": 13,
        "v1_incident_results_reused": False,
        "checkpoint_optimizer_or_replay_transferred_for_correction": False,
        "formal_held_out_outcomes_read": False,
    }
    remote_record = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-jtl110gpu-verification-v2",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "node": args.node,
        "passed": True,
        "python_version": "3.8.20",
        "manifest_sha256": args.expected_manifest_sha256,
        "aggregate_sha256": args.expected_aggregate_sha256,
        "entry_count": args.expected_entry_count,
        "effective_verification": json.loads(verify_stdout.strip().splitlines()[-1]),
        "no_sumo_smoke_test": True,
        "source_root_contract_verified": True,
        "remote_sumo_executed": False,
        "formal_held_out_outcomes_read": False,
    }
    atomic_json(args.sync_record.expanduser().resolve(), sync_record)
    atomic_json(args.remote_verification_record.expanduser().resolve(), remote_record)
    print(json.dumps({"sync": sync_record, "remote_verification": remote_record}, sort_keys=True))
    return 0


def manifest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-aggregate-sha256", required=True)
    parser.add_argument("--expected-entry-count", type=int, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--output", type=Path, default=V2_OVERLAY)
    local_parser = commands.add_parser("verify-local")
    local_parser.add_argument("--overlay", type=Path, default=V2_OVERLAY)
    local_parser.add_argument(
        "--output",
        type=Path,
        default=REPRO / "h2oplus_v20_formal_confirmation_deployment_audit_v2.json",
    )
    manifest_args(local_parser)
    effective_parser = commands.add_parser("verify-effective")
    effective_parser.add_argument("--workspace", type=Path, required=True)
    manifest_args(effective_parser)
    sync_parser = commands.add_parser("sync")
    sync_parser.add_argument("--overlay", type=Path, default=V2_OVERLAY)
    sync_parser.add_argument("--remote-workspace", type=Path, default=V2_REMOTE_WORKSPACE)
    sync_parser.add_argument("--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm")
    sync_parser.add_argument("--node", default="jtl110gpu")
    sync_parser.add_argument(
        "--sync-record",
        type=Path,
        default=REPRO / "h2oplus_v20_formal_confirmation_deployment_sync_v2.json",
    )
    sync_parser.add_argument(
        "--remote-verification-record",
        type=Path,
        default=REPRO / "h2oplus_v20_formal_confirmation_deployment_jtl110gpu_verification_v2.json",
    )
    manifest_args(sync_parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build":
        return build(args)
    if args.command == "verify-local":
        return verify_local(args)
    if args.command == "verify-effective":
        return verify_effective(args)
    if args.command == "sync":
        return sync(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
