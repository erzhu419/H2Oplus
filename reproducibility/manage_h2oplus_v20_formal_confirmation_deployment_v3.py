#!/usr/bin/env python3
"""Build and preflight the dispatch-compatible v20 formal deployment."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import manage_h2oplus_v20_formal_confirmation_deployment_v2 as v2


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
V1_OVERLAY = REPRO / "deployments/h2oplus_v20_formal_confirmation_evaluation_v1/overlay"
V2_DEPLOYMENT = REPRO / "deployments/h2oplus_v20_formal_confirmation_evaluation_v2"
V2_OVERLAY = V2_DEPLOYMENT / "overlay"
V2_REMOTE_WORKSPACE = V2_DEPLOYMENT / "workspace"
V3_DEPLOYMENT = REPRO / "deployments/h2oplus_v20_formal_confirmation_evaluation_v3"
V3_OVERLAY = V3_DEPLOYMENT / "overlay"
V3_REMOTE_WORKSPACE = V3_DEPLOYMENT / "workspace"
CORRECTION = REPRO / "h2oplus_v20_formal_confirmation_manifest_contract_correction_v3.json"
INCIDENT_V2 = (
    REPRO
    / "h2oplus_v20_formal_confirmation_results/evaluation_incidents/v2_manifest_contract"
)
V1_MANIFEST_SHA256 = "91eef065317f1ba8a17fca839f8ce84371555fbb581a515eb474cc8d4bc7cd17"
V1_AGGREGATE_SHA256 = "0bbce7865eaf72f49eb8846054aeb9c6a6e8cc356d763b7e47859dc18a6e203e"
V2_MANIFEST_SHA256 = "ad327b1a054c90094598d36e5ba905465599868cb1088a3c786d9846963586ca"
V2_AGGREGATE_SHA256 = "35cae5822fba4f5381429f1267ccfa7976c9ead4387a1fb4e008276b5c21f91a"
CORRECTION_SHA256 = "6f8a9cd60334b9402451df307b921be28b82bdcd0bb189beffef8e0e83e3a81c"
INCIDENT_V2_CLOSEOUT_SHA256 = "90b028bdebbfbd61cd0ac1f8bb65fbd77454a1e1cf46448ec0a4a6de307c4e6e"
INCIDENT_V2_VERIFICATION_SHA256 = "c45b07b8b1187165a74bd4ad5f08eb204bc567339966eefc3e0c101727f1d405"
HASHES = {
    "execution_authorization": "c9e2cdfe10074e5ddded7f756c682b9c5a63d3e5fc6fc52c87498289407226d5",
    "preparation_authorization": "7a7b79f029a5437a80a31637c1bc057571849623b9b04a92a67bcf66586882c9",
    "rationale": "72e6d2961846e458721f121ae75eded862bed0556c929dc38015a3b2d9103a4d",
    "registration": "c33111cd72b741a801804cd065f42629094d45c0fe03a5b88f9d95dba62149ce",
    "environment": "50b831923b76a81c773cf7dc19dcc2e10c729ea8fc7b3be6982fc01b7c3cc141",
    "protocol": "10aacced63f4c8a1b7f6f55024f46dfdebbcae6737c6bbafb79d227b2ad44664",
    "design_audit": "f0f7f68d8f1273f49e55b1444e5c74282b6aeaf68efaa2f8a0aa0f4261408757",
    "design_verification": "16d33c2b12f280ed8d1595afca91b52043a8a87a844aa7a4cda0227e574f6e29",
    "parent_v20_analysis": "f7de8ca7ea82098a89383f1fe0e9e234f8c1acab1c030d66494100744cd9455b",
    "parent_v20_audit": "2e516ab3ea0cff10dc88607d6fb8adf3e3888a822e32a14f84a86953a466f583",
    "parent_v20_closeout": "dbd1da5442219f404f90da6fbc415b0755f7dddd41d28b9b581039d359882132",
    "parent_v20_verification": "08b25f8b229a9fefbfec198f8198aff5db4f9f2680f16d9bf5521985cdfe7f7e",
    "data_environment_audit": "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    "data_environment_verification": "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
    "source_freeze": "70f387eeb232d5021f33b12ce5e653d0dea500b905827d7c8d525fb738e23c25",
    "methods": "b4bcad09e13cb50c94d5541f2a52340ce99f15b281b5ddef57be38e792d052cd",
    "dataset": "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899",
    "runtime_lock": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
}


def validate_correction() -> None:
    correction = v2.load_json(CORRECTION, CORRECTION_SHA256)
    closeout = v2.load_json(
        INCIDENT_V2 / "incident_closeout_v2.json", INCIDENT_V2_CLOSEOUT_SHA256
    )
    verification = v2.load_json(
        INCIDENT_V2 / "incident_verification_v2.json",
        INCIDENT_V2_VERIFICATION_SHA256,
    )
    if (
        correction.get("scientific_behavior_changed") is not False
        or correction.get("v1_or_v2_incident_results_reused") is not False
        or correction.get("authorization_basis", {}).get(
            "one_fresh_successor_submission_authorized"
        )
        is not True
        or correction.get("frozen_dispatch_contract", {}).get(
            "remote_verify_inputs_preflight_required_before_submission"
        )
        is not True
        or closeout.get("completed_formal_result_count") != 0
        or closeout.get("remote_sumo_episode_count") != 0
        or verification.get("passed") is not True
    ):
        raise ValueError("v3 correction or v2 incident boundary drifted")


def build(args: argparse.Namespace) -> int:
    validate_correction()
    v1_manifest = v2.load_manifest(
        V1_OVERLAY / "deployment_manifest.json",
        V1_MANIFEST_SHA256,
        V1_AGGREGATE_SHA256,
        1252,
    )
    v2_manifest = v2.load_manifest(
        V2_OVERLAY / "deployment_manifest.json",
        V2_MANIFEST_SHA256,
        V2_AGGREGATE_SHA256,
        1269,
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace v3 deployment: {output}")
    shutil.copytree(V2_OVERLAY, output, copy_function=os.link)
    (output / "deployment_manifest.json").unlink()
    repo = output / "H2Oplus"
    evidence = {
        "reproducibility/h2oplus_v20_formal_confirmation_manifest_contract_correction_v3.json": CORRECTION,
        (
            "reproducibility/h2oplus_v20_formal_confirmation_results/evaluation_incidents/"
            "v2_manifest_contract/incident_closeout_v2.json"
        ): INCIDENT_V2 / "incident_closeout_v2.json",
        (
            "reproducibility/h2oplus_v20_formal_confirmation_results/evaluation_incidents/"
            "v2_manifest_contract/incident_verification_v2.json"
        ): INCIDENT_V2 / "incident_verification_v2.json",
        "reproducibility/manage_h2oplus_v20_formal_confirmation_deployment_v3.py": Path(__file__),
    }
    for relative, source in evidence.items():
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source.resolve(strict=True), destination)
    overlay_entries = [
        v2.descriptor(path, output)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    effective = {row["path"]: row for row in v2_manifest["entries"]}
    effective.update({row["path"]: row for row in overlay_entries})
    entries = [effective[path] for path in sorted(effective)]
    manifest = {
        key: value
        for key, value in v1_manifest.items()
        if key
        not in {
            "created_utc",
            "entry_count",
            "total_file_bytes",
            "aggregate_sha256",
            "overlay_entry_count",
            "overlay_total_file_bytes",
            "overlay_aggregate_sha256",
            "entries",
            "overlay_entries",
        }
    }
    manifest.update(
        {
            "schema_version": "transit-h2oplus-v20-formal-confirmation-deployment-manifest-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(entries),
            "total_file_bytes": sum(row["size_bytes"] for row in entries),
            "aggregate_sha256": v2.aggregate(entries),
            "overlay_entry_count": len(overlay_entries),
            "overlay_total_file_bytes": sum(row["size_bytes"] for row in overlay_entries),
            "overlay_aggregate_sha256": v2.aggregate(overlay_entries),
            "scientific_sources_changed_from_v1": False,
            "scientific_behavior_changed_from_v1": False,
            "v1_or_v2_incident_results_reused": False,
            "path_alias_count": 13,
            "path_aliases": v2_manifest["path_aliases"],
            "parent_v2_deployment": {
                "path": str(V2_REMOTE_WORKSPACE),
                "manifest_sha256": V2_MANIFEST_SHA256,
                "aggregate_sha256": V2_AGGREGATE_SHA256,
                "entry_count": 1269,
            },
            "manifest_contract_correction": {
                "path": "H2Oplus/reproducibility/"
                "h2oplus_v20_formal_confirmation_manifest_contract_correction_v3.json",
                "sha256": CORRECTION_SHA256,
            },
            "v2_incident_closeout_sha256": INCIDENT_V2_CLOSEOUT_SHA256,
            "v2_incident_verification_sha256": INCIDENT_V2_VERIFICATION_SHA256,
            "checkpoint_optimizer_or_replay_transferred_in_correction": False,
            "entries": entries,
            "overlay_entries": overlay_entries,
        }
    )
    for key in ("methods_manifest", "environment", "source_freeze"):
        if not isinstance(manifest.get(key), dict) or not manifest[key].get("sha256"):
            raise ValueError(f"required frozen dispatch binding missing: {key}")
    v2.atomic_json(output / "deployment_manifest.json", manifest)
    print(
        json.dumps(
            {
                "manifest_path": str(output / "deployment_manifest.json"),
                "manifest_sha256": v2.sha256(output / "deployment_manifest.json"),
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
    validate_correction()
    overlay = args.overlay.expanduser().resolve(strict=True)
    manifest_path = overlay / "deployment_manifest.json"
    manifest = v2.load_manifest(
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
        raise ValueError("v3 local overlay membership drifted")
    for row in manifest["overlay_entries"]:
        path = overlay / row["path"]
        if path.stat().st_size != row["size_bytes"] or v2.sha256(path) != row["sha256"]:
            raise ValueError(f"v3 local overlay file drifted: {row['path']}")
    if (
        manifest.get("schema_version")
        != "transit-h2oplus-v20-formal-confirmation-deployment-manifest-v1"
        or manifest.get("methods_manifest", {}).get("sha256") != HASHES["methods"]
        or manifest.get("environment", {}).get("sha256") != HASHES["environment"]
        or manifest.get("source_freeze", {}).get("sha256") != HASHES["source_freeze"]
        or manifest.get("v1_or_v2_incident_results_reused") is not False
    ):
        raise ValueError("v3 dispatch-compatible manifest contract drifted")
    report = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-deployment-local-audit-v3",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "manifest_sha256": args.expected_manifest_sha256,
        "aggregate_sha256": args.expected_aggregate_sha256,
        "entry_count": args.expected_entry_count,
        "overlay_entry_count": len(manifest["overlay_entries"]),
        "path_alias_count": 13,
        "dispatch_manifest_contract_verified": True,
        "scientific_sources_changed_from_v1": False,
        "v1_or_v2_incident_results_reused": False,
        "checkpoint_optimizer_or_replay_copied_for_correction": False,
        "formal_held_out_outcomes_read": False,
        "local_sumo_executed": False,
    }
    v2.atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0


def dispatch_preflight_args(workspace: Path, manifest_sha256: str) -> list[str]:
    repo = workspace / "H2Oplus"
    repro = repo / "reproducibility"
    paths = {
        "execution_authorization": repro / "h2oplus_v20_formal_confirmation_execution_authorization_v1.json",
        "preparation_authorization": repro / "h2oplus_v20_formal_confirmation_preparation_authorization_v1.json",
        "rationale": repro / "h2oplus_v20_formal_confirmation_rationale_v1.json",
        "registration": repro / "h2oplus_v20_formal_confirmation_registration_v1.json",
        "environment": repro / "h2oplus_v20_formal_confirmation_environment_v1.json",
        "protocol": repro / "h2oplus_v20_formal_confirmation_protocol_v1.json",
        "design_audit": repro / "h2oplus_v20_formal_confirmation_design_audit_v1.json",
        "design_verification": repro / "h2oplus_v20_formal_confirmation_design_verification_v1.json",
        "parent_v20_analysis": repro / "h2oplus_v20_full_control_results/full_control_analysis_v1.json",
        "parent_v20_audit": repro / "h2oplus_v20_full_control_results/evaluation_audit_v1.json",
        "parent_v20_closeout": repro / "h2oplus_v20_full_control_results/evaluation_closeout_v1.json",
        "parent_v20_verification": repro / "h2oplus_v20_full_control_results/evaluation_closeout_verification_v1.json",
        "data_environment_audit": repro / "h2oplus_v17_data_environment_convergence_audit_v1.json",
        "data_environment_verification": repro / "h2oplus_v17_data_environment_convergence_verification_v1.json",
        "source_freeze": repro / "h2oplus_v20_formal_confirmation_source_freeze_v1.json",
        "methods_manifest": repro / "h2oplus_v20_formal_confirmation_evaluation_methods_v1.json",
        "dataset": repo / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5",
        "sumo_runtime_lock": repro / "h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
        "deployment_manifest": workspace / "deployment_manifest.json",
        "output_root": workspace / ".preflight_no_results",
        "execution_json": workspace / ".preflight_no_execution.json",
        "sumo_dir": workspace / "SUMO_ruiguang/online_control",
        "edge_xml": workspace
        / "SUMO_ruiguang/online_control/intersection_delay/a_sorted_busline_edge.xml",
    }
    result = []
    for name, path in paths.items():
        result.extend(["--" + name.replace("_", "-"), str(path)])
    expected = dict(HASHES)
    expected["deployment_manifest"] = manifest_sha256
    for name, digest in expected.items():
        result.extend(["--expected-" + name.replace("_", "-") + "-sha256", digest])
    result.extend(["--workers", "12"])
    return result


def sync(args: argparse.Namespace) -> int:
    overlay = args.overlay.expanduser().resolve(strict=True)
    manifest = v2.load_manifest(
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
        f"set -eu; test \"$(sha256sum {shlex.quote(str(V2_REMOTE_WORKSPACE / 'deployment_manifest.json'))} "
        f"| awk '{{print $1}}')\" = {shlex.quote(V2_MANIFEST_SHA256)}; "
        f"test ! -e {shlex.quote(str(remote))}; "
        f"mkdir -p {shlex.quote(str(remote.parent))}; "
        f"cp -al {shlex.quote(str(V2_REMOTE_WORKSPACE))} {shlex.quote(str(remote))}; "
        f"rm {shlex.quote(str(remote / 'deployment_manifest.json'))}"
    )
    rc, stdout, stderr = scheduler.run_on(args.node, clone, timeout=120, check=False)
    if rc != 0:
        raise RuntimeError(f"remote v2-to-v3 hardlink clone failed: {stderr or stdout}")
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
        raise RuntimeError(f"v3 formal overlay rsync failed: {completed.stderr.strip()}")
    python = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
    manager = remote / "H2Oplus/reproducibility/manage_h2oplus_v20_formal_confirmation_deployment_v3.py"
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
        "PATH=/home/erzhu419/anaconda3/envs/LSTM-RL/bin:/usr/bin:/bin "
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
        raise RuntimeError(f"v3 effective verification failed: {verify_stderr or verify_stdout}")
    preflight_code = (
        "import json; "
        "from reproducibility import run_h2oplus_v20_formal_confirmation_dispatch as d; "
        "e=d.verify_inputs(d.parse_args()); "
        "print(json.dumps({'passed':True,'block_count':len(e['blocks']),"
        "'formal_held_out_outcomes_read':False,'remote_sumo_executed':False},sort_keys=True))"
    )
    preflight = shlex.join(
        [python, "-c", preflight_code, *dispatch_preflight_args(remote, args.expected_manifest_sha256)]
    )
    rc, preflight_stdout, preflight_stderr = scheduler.run_on(
        args.node, env + preflight, timeout=300, check=False
    )
    if rc != 0:
        raise RuntimeError(
            f"frozen dispatch verify_inputs preflight failed: {preflight_stderr or preflight_stdout}"
        )
    absence_check = (
        f"test ! -e {shlex.quote(str(remote / '.preflight_no_results'))} && "
        f"test ! -e {shlex.quote(str(remote / '.preflight_no_execution.json'))}"
    )
    rc, absence_stdout, absence_stderr = scheduler.run_on(
        args.node, absence_check, timeout=30, check=False
    )
    if rc != 0:
        raise AssertionError(
            "frozen dispatch preflight unexpectedly created result or execution evidence: "
            f"{absence_stderr or absence_stdout}"
        )
    preflight_report = json.loads(preflight_stdout.strip().splitlines()[-1])
    sync_record = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-deployment-sync-v3",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "node": args.node,
        "passed": True,
        "transport": "remote-v2-formal-hardlink-clone-plus-manifest-contract-overlay",
        "parent_v2_manifest_sha256": V2_MANIFEST_SHA256,
        "manifest_sha256": args.expected_manifest_sha256,
        "aggregate_sha256": args.expected_aggregate_sha256,
        "entry_count": args.expected_entry_count,
        "path_alias_count": 13,
        "v1_or_v2_incident_results_reused": False,
        "checkpoint_optimizer_or_replay_transferred_for_correction": False,
        "formal_held_out_outcomes_read": False,
    }
    remote_record = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-jtl110gpu-verification-v3",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "node": args.node,
        "passed": True,
        "python_version": "3.8.20",
        "manifest_sha256": args.expected_manifest_sha256,
        "aggregate_sha256": args.expected_aggregate_sha256,
        "entry_count": args.expected_entry_count,
        "effective_verification": json.loads(verify_stdout.strip().splitlines()[-1]),
        "frozen_dispatch_verify_inputs_preflight": preflight_report,
        "preflight_result_files_created": False,
        "remote_sumo_executed": False,
        "formal_held_out_outcomes_read": False,
    }
    v2.atomic_json(args.sync_record.expanduser().resolve(), sync_record)
    v2.atomic_json(args.remote_verification_record.expanduser().resolve(), remote_record)
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
    build_parser.add_argument("--output", type=Path, default=V3_OVERLAY)
    local_parser = commands.add_parser("verify-local")
    local_parser.add_argument("--overlay", type=Path, default=V3_OVERLAY)
    local_parser.add_argument(
        "--output",
        type=Path,
        default=REPRO / "h2oplus_v20_formal_confirmation_deployment_audit_v3.json",
    )
    manifest_args(local_parser)
    effective_parser = commands.add_parser("verify-effective")
    effective_parser.add_argument("--workspace", type=Path, required=True)
    manifest_args(effective_parser)
    sync_parser = commands.add_parser("sync")
    sync_parser.add_argument("--overlay", type=Path, default=V3_OVERLAY)
    sync_parser.add_argument("--remote-workspace", type=Path, default=V3_REMOTE_WORKSPACE)
    sync_parser.add_argument("--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm")
    sync_parser.add_argument("--node", default="jtl110gpu")
    sync_parser.add_argument(
        "--sync-record",
        type=Path,
        default=REPRO / "h2oplus_v20_formal_confirmation_deployment_sync_v3.json",
    )
    sync_parser.add_argument(
        "--remote-verification-record",
        type=Path,
        default=REPRO / "h2oplus_v20_formal_confirmation_deployment_jtl110gpu_verification_v3.json",
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
        return v2.verify_effective(args)
    if args.command == "sync":
        return sync(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
