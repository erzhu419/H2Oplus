#!/usr/bin/env python3
"""Audit and seal the H2O+ v16 headway-reserve development evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v16_headway_reserve_results"
PARTITION = RESULTS / "evaluation_partition_v1/single_jtl110gpu"
EVIDENCE = RESULTS / "evidence/evaluation_v1"
AUDIT_PATH = RESULTS / "evaluation_audit_v1.json"
MANIFEST_PATH = RESULTS / "evaluation_file_manifest_v1.json"

EXPECTED_HASHES = {
    "registration": "6858bea266a4f46ec919804e643a5b8c6de4c189dd0517ee5bf123f08b64950a",
    "authorization": "77f2234b90ba510fea089800494c637c12d8a3f24b2f2fdd236238ae04fb19a1",
    "rationale": "48e0445e1d82b4f08fc2068e57d5ec1931c26c5d31db9761f51d5082eadadf25",
    "environment": "bfe97d430921f38fbbec04490ac980e330d2000f6c1ad4d62627c95d3deba5e3",
    "protocol": "978faf1b8ddf51128b5fac570552e8388c3d882e39244c4b6f0ec7615e79a43b",
    "methods": "35047fd2ce616acbccd4592ae333a7ba84a1e4231c1a2505c7768d96b91de06a",
    "source_freeze": "606a15bf7db8b9aa1c4dbefa7b9bcc12bc94ca60e605de5a3e185eefec679577",
    "local_verification": "39243050933c1dc0d0f23220022f7b6bdf9b009cd854cb7113eb8e542c7d331d",
    "deployment_sync": "5fb62c16ba94fc8c45b59b403f073e12af5ae12556a3a3b7eee168a85acd08a5",
    "remote_verification": "16f79ecf1ebdb82f63b826ae879561d3bd8be5d4bdf3fb945adc028d5a7a7ab0",
    "submission": "9a0fce7ddb98fb41dea0b18bbc13955782e9cf585f9287d95b38a195066a562b",
    "deployment": "4637a2dd2b935c00070c4ca514abf2ab24213c9ef80943dcdb2ae781d6d7653c",
    "runtime_lock": "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4",
    "parent_training_closeout": "5d609f76f6d139dc8982a0ce4f8c9e95a4f0e5f4c4d9655b3545fa15270d2ffc",
    "parent_training_verification": "56cf282615364dc1b93bbf118cd1f3101b20fa2398ca87e08b202322c833c15b",
    "parent_training_audit": "51296e4eb91cf785dd484ae726f172d6828b16b1aeaa8af8469a6356b45ea136",
    "parent_training_manifest": "bb1e4bb04e437091c137595399abfae1d4f80294b37a471c39ee53fe3eaae558",
    "v14_analysis": "937a5300988311f80a84dc1220dfee3f0f96f2b6f83691d4e309c38d6bbe6856",
    "v14_evaluation_closeout": "95fef5b3a5206ce14a3da164d9f2948b0dc3b5d61ee1d1b34c52c431d1cd1d32",
    "v14_evaluation_verification": "0e20ccc1e7855e4feb5860e5cb334cd60237f28032a948b7325037ad0aa61dc0",
    "v15_analysis": "d122aa79f6ee2cd5010e241607347921ed9bdc80bbed6ba41c3105e601ac9d9c",
    "v15_evaluation_audit": "a060b49e3155092542681c7d6afef66bcb2b31ab04ffe523273754ae2dcfa7e8",
    "v15_evaluation_closeout": "2e4598c433eb6ffaf18fb9919b608b00d5cdd17f5a53e466ca587d6c8213feab",
    "v15_evaluation_verification": "f5edd5c2ef651172068c9cd595e6cb7710a3ffdd66e545671f5b4a014885e0b6",
    "scheduler_evidence": "39786f5b16f6a8f3b125565ad08510f20fa1ca73a117be699c57a925d63ede65",
    "source_sync": "be6009cd2fade6a1f39475463065c9f3b3d90c89e33e10fffa7edf48a86036eb",
}
CONTROL_PATHS = {
    "registration": REPO / "reproducibility/h2oplus_v16_headway_reserve_registration_v1.json",
    "authorization": REPO / "reproducibility/h2oplus_v16_headway_reserve_authorization_v1.json",
    "rationale": REPO / "reproducibility/h2oplus_v16_headway_reserve_rationale_v1.json",
    "environment": REPO / "reproducibility/h2oplus_v12_staged_release_environment_v2.json",
    "protocol": REPO / "reproducibility/h2oplus_v16_headway_reserve_protocol_v1.json",
    "methods": REPO / "reproducibility/h2oplus_v16_headway_reserve_evaluation_methods_v1.json",
    "source_freeze": REPO / "reproducibility/h2oplus_v16_headway_reserve_evaluation_source_freeze_v1.json",
    "local_verification": REPO / "reproducibility/h2oplus_v16_headway_reserve_evaluation_deployment_audit_v1.json",
    "deployment_sync": REPO / "reproducibility/h2oplus_v16_headway_reserve_evaluation_deployment_sync_v1.json",
    "remote_verification": REPO / "reproducibility/h2oplus_v16_headway_reserve_evaluation_deployment_v1_jtl110gpu_verification.json",
    "submission": REPO / "reproducibility/h2oplus_v16_headway_reserve_evaluation_submission_v1.json",
    "deployment": REPO / "reproducibility/deployments/h2oplus_v16_headway_reserve_evaluation_v1/workspace/deployment_manifest.json",
    "runtime_lock": REPO / "reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json",
    "parent_training_closeout": REPO / "reproducibility/h2oplus_v14_terminal_taper_branch_closeout_v1.json",
    "parent_training_verification": REPO / "reproducibility/h2oplus_v14_terminal_taper_branch_closeout_verification_v1.json",
    "parent_training_audit": REPO / "reproducibility/h2oplus_v14_terminal_taper_results/branch_audit_v1.json",
    "parent_training_manifest": REPO / "reproducibility/h2oplus_v14_terminal_taper_results/branch_file_manifest_v1.json",
    "v14_analysis": REPO / "reproducibility/h2oplus_v14_terminal_taper_results/terminal_taper_analysis_v1.json",
    "v14_evaluation_closeout": REPO / "reproducibility/h2oplus_v14_terminal_taper_results/evaluation_closeout_v1.json",
    "v14_evaluation_verification": REPO / "reproducibility/h2oplus_v14_terminal_taper_results/evaluation_closeout_verification_v1.json",
    "v15_analysis": REPO / "reproducibility/h2oplus_v15_hold_calibration_results/hold_calibration_analysis_v1.json",
    "v15_evaluation_audit": REPO / "reproducibility/h2oplus_v15_hold_calibration_results/evaluation_audit_v1.json",
    "v15_evaluation_closeout": REPO / "reproducibility/h2oplus_v15_hold_calibration_results/evaluation_closeout_v1.json",
    "v15_evaluation_verification": REPO / "reproducibility/h2oplus_v15_hold_calibration_results/evaluation_closeout_verification_v1.json",
    "scheduler_evidence": EVIDENCE / "scheduler_evidence_v1.json",
    "source_sync": EVIDENCE / "source_sync_v1.json",
}

EXPECTED_DEPLOYMENT_AGGREGATE = (
    "1e361c6dce106e9d841d7fabc6cbf37414591c4afba2ee075bafbe28159f2805"
)
EXPECTED_DATASET_SHA256 = (
    "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899"
)
EXPECTED_RUNTIME_SIGNATURE = (
    "264d3c412b8e6c42320995208d2e009eb809b85626047206c4de752c2dae06e7"
)
EXPECTED_BACKEND_TOKEN = "1787571481648149575:1528677:126605756510208"
EXPECTED_V14_PROTOCOL_SHA256 = (
    "ba275fc4c134feadb564b49548699571ca5bf6bfd1aee529759d6ce791df12c5"
)
EXPECTED_V15_PROTOCOL_SHA256 = (
    "6cd856504438ad0e9614b49156100cef12f24201b41562bf223e898f15ef0a8c"
)
EXPECTED_ENV = {
    "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
    "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
    "CPATH": (
        "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/"
        "deployments/v7_formal_v2_training_v7r1/workspace/reproducibility/"
        "toolchain/libxcrypt-4.4.36-hd590300_1/include"
    ),
    "PYTHONDONTWRITEBYTECODE": "1",
}
BLOCKS = tuple(f"v12-staged-release-development-0{index}" for index in range(1, 4))
SCALES = {0.85, 1.0, 1.15}
SEEDS = set(range(6201, 6206))
ALGORITHMS = {
    "h2oplus-v16-headway-reserve-half": 0.5,
    "h2oplus-v16-headway-reserve-three-quarter": 0.75,
    "h2oplus-v16-headway-reserve-full": 1.0,
}
METHOD = re.compile(
    r"^(?P<algorithm>h2oplus-v16-headway-reserve-(?:half|three-quarter|full))"
    r"-seed(?P<seed>62\d{2})-events000025000$"
)
METHOD_BINDING_FIELDS = {
    "actor_tensor_sha256",
    "actual_online_events",
    "algorithm",
    "artifact_lineage",
    "base_hold_scale",
    "checkpoint_sha256",
    "headway_balance_gain",
    "learner_update_steps",
    "name",
    "online_events",
    "parent_artifact_lineage",
    "parent_method",
    "ratio_optimizer_steps",
    "reserve_horizon_fraction",
    "temperature_tensor_sha256",
    "training_admission_sha256",
    "training_checkpoint_sha256",
    "training_run_signature",
    "training_seed",
    "type",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    return False


def relative(path: Path) -> str:
    return str(path.resolve(strict=True).relative_to(REPO))


def file_record(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    return {
        "path": relative(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def records_aggregate(records: Iterable[dict]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["path"]):
        digest.update(record["path"].encode())
        digest.update(b"\0")
        digest.update(str(record["size_bytes"]).encode())
        digest.update(b"\0")
        digest.update(record["sha256"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def suffix_hash(source: dict, suffix: str) -> str | None:
    values = [digest for path, digest in source.items() if str(path).endswith(suffix)]
    return values[0] if len(values) == 1 else None


def resolve_checkpoint(methods_path: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = methods_path.parent / path
    return path.resolve(strict=True)


def validate_controls() -> tuple[dict, set[Path], dict[str, dict], dict[str, str], set[Path]]:
    records: dict[str, dict] = {}
    for name, path in CONTROL_PATHS.items():
        actual = sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"control identity drifted for {name}: {actual}")
        records[name] = file_record(path)

    registration = load_json(CONTROL_PATHS["registration"])
    authorization = load_json(CONTROL_PATHS["authorization"])
    protocol = load_json(CONTROL_PATHS["protocol"])
    methods_payload = load_json(CONTROL_PATHS["methods"])
    source_freeze = load_json(CONTROL_PATHS["source_freeze"])
    submission = load_json(CONTROL_PATHS["submission"])
    deployment = load_json(CONTROL_PATHS["deployment"])
    training_closeout = load_json(CONTROL_PATHS["parent_training_closeout"])
    training_verification = load_json(CONTROL_PATHS["parent_training_verification"])
    v14_closeout = load_json(CONTROL_PATHS["v14_evaluation_closeout"])
    v14_verification = load_json(CONTROL_PATHS["v14_evaluation_verification"])
    v15_analysis = load_json(CONTROL_PATHS["v15_analysis"])
    v15_audit = load_json(CONTROL_PATHS["v15_evaluation_audit"])
    v15_closeout = load_json(CONTROL_PATHS["v15_evaluation_closeout"])
    v15_verification = load_json(CONTROL_PATHS["v15_evaluation_verification"])
    if (
        registration.get("schema_version")
        != "transit-h2oplus-v16-headway-reserve-registration-v1"
        or registration.get("study_stage") != "development"
        or registration.get("new_training_runs") != 0
        or registration.get("prior_v14_and_v15_development_outcomes_used_for_design")
        is not True
        or registration.get("v16_development_evaluation_outcomes_read") is not False
        or registration.get("formal_held_out_outcomes_read") is not False
        or authorization.get("schema_version")
        != "transit-h2oplus-v16-headway-reserve-authorization-v1"
        or authorization.get("formal_held_out_outcomes_read") is not False
        or protocol.get("schema_version")
        != "transit-h2oplus-v16-headway-reserve-protocol-v1"
        or protocol.get("local_training_or_sumo_allowed") is not False
        or protocol.get("formal_held_out_outcomes_read") is not False
        or methods_payload.get("schema_version")
        != "transit-evaluation-methods-v16-headway-reserve-v1"
        or methods_payload.get("physical_method_count") != 15
        or methods_payload.get("logical_method_count") != 15
        or methods_payload.get("expected_new_remote_sumo_episodes") != 135
        or methods_payload.get(
            "prior_v14_and_v15_development_outcomes_used_for_design"
        )
        is not True
        or methods_payload.get("v16_development_evaluation_outcomes_read") is not False
        or methods_payload.get("predesignated_candidate")
        != "h2oplus-v16-headway-reserve-three-quarter"
        or methods_payload.get("formal_held_out_outcomes_read") is not False
        or len(methods_payload.get("methods", [])) != 15
        or len(methods_payload.get("aliases", [])) != 15
        or source_freeze.get("schema_version")
        != "transit-h2oplus-v16-headway-reserve-evaluation-source-freeze-v1"
        or source_freeze.get("matrix_source_count") != 12
        or source_freeze.get("expected_matrix_source_file_count") != 51
        or source_freeze.get("operational_source_count") != 2
        or source_freeze.get(
            "prior_v14_and_v15_development_outcomes_used_for_design"
        )
        is not True
        or source_freeze.get("v16_evaluation_outcomes_read") is not False
        or source_freeze.get("formal_held_out_outcomes_read") is not False
        or submission.get("schema_version")
        != "transit-h2oplus-v16-headway-reserve-evaluation-submission-v1"
        or submission.get("total_remote_sumo_episodes") != 135
        or submission.get("physical_behavior_count") != 15
        or submission.get("task_count") != 1
        or submission.get("task", {}).get("task_id") != "t83704"
        or deployment.get("schema_version")
        != "transit-h2oplus-v16-headway-reserve-evaluation-deployment-manifest-v1"
        or deployment.get("aggregate_sha256") != EXPECTED_DEPLOYMENT_AGGREGATE
        or deployment.get("entry_count") != 712
        or training_closeout.get("artifact_integrity_passed") is not True
        or training_closeout.get("scientific_stability_passed_for_all_15_branches")
        is not True
        or training_closeout.get("training_run_count") != 15
        or training_verification.get("passed") is not True
        or v14_closeout.get("passed") is not True
        or v14_verification.get("passed") is not True
        or v15_analysis.get("formal_confirmation_eligible") is not False
        or v15_audit.get("passed") is not True
        or v15_closeout.get("passed") is not True
        or v15_closeout.get("formal_confirmation_eligible") is not False
        or v15_verification.get("passed") is not True
        or v15_verification.get("formal_confirmation_eligible") is not False
        or any(
            payload.get("formal_held_out_outcomes_read") is not False
            for payload in (
                training_closeout,
                training_verification,
                v14_closeout,
                v14_verification,
                v15_analysis,
                v15_audit,
                v15_closeout,
                v15_verification,
            )
        )
    ):
        raise ValueError("v16 immutable control contract failed")

    methods = {row["name"]: row for row in methods_payload["methods"]}
    if len(methods) != 15:
        raise ValueError("duplicate v16 physical behavior")
    expected_names = {
        f"{algorithm}-seed{seed}-events000025000"
        for algorithm in ALGORITHMS
        for seed in SEEDS
    }
    if set(methods) != expected_names:
        raise ValueError("v16 method grid drifted")
    checkpoint_files: set[Path] = set()
    for name, method in methods.items():
        match = METHOD.fullmatch(name)
        if (
            match is None
            or method.get("algorithm") != match.group("algorithm")
            or method.get("training_seed") != int(match.group("seed"))
            or method.get("hold_scale") != 0.125
            or method.get("base_hold_scale") != 0.125
            or method.get("headway_balance_gain") != 1.0
            or method.get("reserve_horizon_fraction")
            != ALGORITHMS[method["algorithm"]]
            or method.get("artifact_lineage") != "v16-headway-reserve"
            or method.get("parent_artifact_lineage") != "v15-hold-calibration"
            or method.get("v16_protocol_sha256") != EXPECTED_HASHES["protocol"]
            or method.get("v15_protocol_sha256") != EXPECTED_V15_PROTOCOL_SHA256
            or method.get("source_protocol_sha256") != EXPECTED_V15_PROTOCOL_SHA256
            or method.get("predesignated_candidate")
            is not (
                method["algorithm"]
                == "h2oplus-v16-headway-reserve-three-quarter"
            )
        ):
            raise ValueError(f"v16 method provenance failed: {name}")
        checkpoint = resolve_checkpoint(CONTROL_PATHS["methods"], method["checkpoint"])
        if sha256(checkpoint) != method.get("checkpoint_sha256"):
            raise ValueError(f"v16 parent checkpoint drifted: {name}")
        checkpoint_files.add(checkpoint)
    aliases = methods_payload["aliases"]
    if (
        {row.get("name") for row in aliases} != set(methods)
        or any(row.get("physical_method") != row.get("name") for row in aliases)
    ):
        raise ValueError("v16 logical alias map drifted")
    if len(checkpoint_files) != 5:
        raise ValueError("v16 must bind five unique sealed parent checkpoints")

    frozen_sources = {
        row["path"]: row["sha256"] for row in source_freeze["matrix_sources"]
    }
    source_files = {REPO / path for path in frozen_sources}
    if (
        len(frozen_sources) != 12
        or any(
            not path.is_file()
            or path.stat().st_size
            != next(row["bytes"] for row in source_freeze["matrix_sources"] if row["path"] == relative(path))
            or sha256(path) != frozen_sources[relative(path)]
            for path in source_files
        )
    ):
        raise ValueError("v16 frozen matrix source identity failed")
    files = set(CONTROL_PATHS.values()) | source_files | checkpoint_files
    return records, files, methods, frozen_sources, checkpoint_files


def validate_scheduler_and_sync() -> tuple[dict, dict, set[Path]]:
    scheduler = load_json(CONTROL_PATHS["scheduler_evidence"])
    source_sync = load_json(CONTROL_PATHS["source_sync"])
    task = scheduler.get("task", {})
    backend = task.get("backend_exit", {})
    backend_path = Path(backend.get("path", "")).resolve(strict=True)
    fields = backend_path.read_text().strip().split("\t")
    required_env = task.get("required_environment", {})
    false_negative = scheduler.get("scheduler_frontend_false_negative", {})
    if (
        scheduler.get("schema_version")
        != "transit-h2oplus-v16-headway-reserve-evaluation-scheduler-evidence-v1"
        or scheduler.get("passed") is not True
        or scheduler.get("token_bound_backend_exit0") is not True
        or task.get("task_id") != "t83704"
        or task.get("node") != "jtl110gpu"
        or task.get("declared_cpu_cores") != 12
        or task.get("submitted_ram_mb") != 32768
        or task.get("declared_vram_mb") != 0
        or task.get("backend_returncode") != 0
        or task.get("backend_token") != EXPECTED_BACKEND_TOKEN
        or task.get("token_matches") is not True
        or task.get("residual_evaluation_process_count") != 0
        or any(required_env.get(name) != value for name, value in EXPECTED_ENV.items())
        or required_env.get("CUDA_VISIBLE_DEVICES", "") != ""
        or false_negative.get("observed_status") != "failed"
        or false_negative.get("reason") != "log only 0B after 10896s (very suspicious)"
        or false_negative.get("scientific_result_status_inferred_from_scheduler_frontend")
        is not False
        or fields != ["0", "1787582340", EXPECTED_BACKEND_TOKEN]
        or sha256(backend_path) != backend.get("sha256")
    ):
        raise ValueError("v16 token-bound scheduler evidence failed")

    partition_files = {path for path in PARTITION.rglob("*") if path.is_file()}
    synced_rows = source_sync.get("partition", {}).get("files", [])
    synced = {row["path"]: row for row in synced_rows}
    local = {str(path.relative_to(PARTITION)): path for path in partition_files}
    if (
        source_sync.get("passed") is not True
        or source_sync.get("checksum_mode")
        != "sha256-per-file-before-and-after-transfer"
        or source_sync.get("local_sumo_executed") is not False
        or source_sync.get("formal_held_out_outcomes_read") is not False
        or source_sync.get("partition", {}).get("expected_results") != 135
        or source_sync.get("partition", {}).get("file_count") != 281
        or len(synced) != 281
        or set(synced) != set(local)
        or any(
            row.get("bytes") != local[name].stat().st_size
            or row.get("sha256") != sha256(local[name])
            for name, row in synced.items()
        )
    ):
        raise ValueError("v16 remote/local per-file synchronization failed")
    return scheduler, source_sync, partition_files | {backend_path}


def validate_parent_training(payload: dict, method: dict) -> None:
    training = payload.get("policy", {}).get("training_metadata", {})
    admission = training.get("training_admission", {})
    intervention = training.get("v14_intervention", {})
    if (
        training.get("training_seed") != method["training_seed"]
        or training.get("run_signature") != method["training_run_signature"]
        or training.get("training_checkpoint_sha256")
        != method["training_checkpoint_sha256"]
        or training.get("study_stage") != "development"
        or training.get("formal_held_out_outcomes_read") is not False
        or training.get("method") != "h2oplus"
        or training.get("stage") != "branch"
        or training.get("phase") != "complete"
        or training.get("v14_protocol", {}).get("sha256")
        != EXPECTED_V14_PROTOCOL_SHA256
        or admission.get("schema_version")
        != "transit-training-admission-v14-terminal-taper-development-v1"
        or admission.get("sha256") != method["training_admission_sha256"]
        or intervention.get("variant") != "terminal-quarter"
        or intervention.get("terminal_actor_lr_multiplier") != 0.25
        or intervention.get("proximal_rate_per_actor_update") != 5e-5
        or intervention.get("registered_post_10k_actor_updates") != 15_000
        or intervention.get("predesignated_candidate") is not True
        or intervention.get("unit_ratio_weights") is not True
        or intervention.get("checkpoint_selection_by_behavior") is not False
    ):
        raise ValueError("v16 result parent-training provenance drifted")


def validate_result(
    job: dict,
    output: Path,
    method: dict,
    source_reference: dict | None,
    frozen_sources: dict[str, str],
) -> dict:
    payload = load_json(output)
    source = payload.get("source_sha256", {})
    job_method = job.get("method", {})
    policy = payload.get("policy", {})
    reserve = policy.get("headway_reserve", {})
    semantics = payload.get("bus_service_semantics_audit", {})
    violations = semantics.get("violations", {})
    ledger = payload.get("ledger_audit", {})
    count_audit = ledger.get("count_source_audit", {})
    passengers = payload.get("passengers", {})
    welfare = payload.get("passenger_welfare", {})
    coverage = welfare.get("chronological_coverage_errors", {})
    service = payload.get("service", {})
    if (
        payload.get("schema_version") != "transit-evaluation-v7"
        or payload.get("evaluation_job", {}).get("signature") != job.get("signature")
        or payload.get("method") != method["name"]
        or int(payload.get("sumo_seed", -1)) != int(job.get("sumo_seed", -2))
        or not finite(payload)
        or source != job.get("source_sha256")
        or len(source) != 51
        or any(suffix_hash(source, suffix) != digest for suffix, digest in frozen_sources.items())
        or any(job_method.get(key) != method.get(key) for key in METHOD_BINDING_FIELDS)
        or job_method.get("training_plan_sha256") != EXPECTED_V15_PROTOCOL_SHA256
        or job.get("dataset_sha256") != EXPECTED_DATASET_SHA256
        or payload.get("dataset", {}).get("sha256") != EXPECTED_DATASET_SHA256
        or job.get("sumo_runtime_signature") != EXPECTED_RUNTIME_SIGNATURE
        or payload.get("runtime", {}).get("sumo_runtime_signature")
        != EXPECTED_RUNTIME_SIGNATURE
        or payload.get("scenario", {}).get("manifest_sha256")
        != job.get("scenario", {}).get("manifest_sha256")
        or policy.get("artifact_lineage") != "v16-headway-reserve"
        or policy.get("parent_artifact_lineage") != "v15-hold-calibration"
        or policy.get("parent_method") != method["parent_method"]
        or policy.get("sha256") != method["checkpoint_sha256"]
        or policy.get("actor_tensor_sha256") != method["actor_tensor_sha256"]
        or policy.get("temperature_tensor_sha256")
        != method["temperature_tensor_sha256"]
        or reserve.get("schema_version")
        != "transit-h2oplus-v16-headway-reserve-v1"
        or reserve.get("base_hold_scale") != method["base_hold_scale"]
        or reserve.get("headway_balance_gain") != method["headway_balance_gain"]
        or reserve.get("reserve_horizon_fraction")
        != method["reserve_horizon_fraction"]
        or reserve.get("variant") != method["variant"]
        or reserve.get("activation_budget_online_events") != 25_000
        or reserve.get("v16_protocol_sha256") != EXPECTED_HASHES["protocol"]
        or reserve.get("predesignated_candidate")
        is not (
            method["algorithm"]
            == "h2oplus-v16-headway-reserve-three-quarter"
        )
        or reserve.get("checkpoint_selection_by_behavior") is not False
        or reserve.get("formal_held_out_outcomes_read") is not False
        or reserve.get("zero_hold_preserved_exactly") is not True
        or reserve.get("transformed_hold_never_exceeds_parent") is not True
        or reserve.get("unit") != "physical_hold_seconds"
        or semantics.get("passed") is not True
        or semantics.get("require_all_departed_trips_completed") is not True
        or not isinstance(violations, dict)
        or any(value not in ([], {}) for value in violations.values())
        or service.get("active_trips_at_horizon") != 0
        or service.get("completed_trips") != service.get("departed_trips")
        or service.get("completion_rate") != 1.0
        or ledger.get("legacy_sampled_trip_histories_used_as_ground_truth") is not False
        or count_audit.get("max_in_vehicle_count_error") != 0
        or count_audit.get("max_waiting_count_error") != 0
        or passengers.get("legacy_sampled_trip_histories_used_for_welfare") is not False
        or passengers.get("active_state_count_balance_error") != 0
        or welfare.get("primary") is not True
        or not coverage
        or any(abs(float(value)) > 1e-9 for value in coverage.values())
        or abs(float(welfare.get("maximum_reward_reconstruction_error", math.inf)))
        > 1e-9
    ):
        raise ValueError(f"v16 result provenance, semantics, or ledger failed: {output}")
    validate_parent_training(payload, method)
    if source_reference is not None and source != source_reference:
        raise ValueError(f"v16 source map differs across results: {output}")
    return source


def validate_partition(
    methods: dict[str, dict], frozen_sources: dict[str, str]
) -> tuple[dict, set[Path]]:
    host = load_json(PARTITION / "host_execution.json")
    summary = load_json(PARTITION / "host_summary.json")
    if (
        host.get("schema_version")
        != "transit-h2oplus-v16-headway-reserve-evaluation-host-execution-v1"
        or host.get("status") != "completed"
        or host.get("completed_results") != 135
        or host.get("workers") != 12
        or host.get("training") is not False
        or host.get("formal_held_out_outcomes_read") is not False
        or len(host.get("completed_blocks", [])) != 3
        or summary.get("status") != "completed"
        or summary.get("physical_evaluation_results") != 135
        or summary.get("training") is not False
        or summary.get("formal_held_out_outcomes_read") is not False
        or len(summary.get("matrix_manifests", [])) != 3
    ):
        raise ValueError("v16 completed host contract failed")
    completed = {row["block"]: row["manifest"] for row in host["completed_blocks"]}
    summarized = {row["block"]: row["manifest"] for row in summary["matrix_manifests"]}
    signatures: set[str] = set()
    outputs: set[Path] = set()
    source_reference: dict | None = None
    cells: set[tuple[str, int, str, float]] = set()
    block_stats: dict[str, dict] = {}
    files = {path for path in PARTITION.rglob("*") if path.is_file()}
    for index, block in enumerate(BLOCKS, start=1):
        manifest_path = PARTITION / block / "matrix_manifest.json"
        manifest = load_json(manifest_path)
        jobs = manifest.get("jobs", [])
        partition = manifest.get("matrix_partition", {})
        source = manifest.get("source_sha256", {})
        manifest_hash = sha256(manifest_path)
        expected_seed = 5100 + index
        if (
            manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or manifest.get("methods_manifest_sha256") != EXPECTED_HASHES["methods"]
            or manifest.get("dataset_sha256") != EXPECTED_DATASET_SHA256
            or manifest.get("sumo_runtime_signature") != EXPECTED_RUNTIME_SIGNATURE
            or len(source) != 51
            or any(suffix_hash(source, suffix) != digest for suffix, digest in frozen_sources.items())
            or len(jobs) != 45
            or partition.get("full_job_count") != 45
            or partition.get("selected_job_count") != 45
            or partition.get("shard_count") != 1
            or partition.get("shard_index") != 0
            or completed.get(block, {}).get("sha256") != manifest_hash
            or summarized.get(block, {}).get("sha256") != manifest_hash
            or completed.get(block, {}).get("source_file_count") != 51
            or summarized.get(block, {}).get("source_file_count") != 51
        ):
            raise ValueError(f"v16 matrix contract failed: {block}")
        block_signatures: set[str] = set()
        for job in jobs:
            name = str(job.get("method", {}).get("name"))
            method = methods.get(name)
            raw_output = Path(job.get("output", "")).expanduser()
            output = raw_output.resolve() if raw_output.is_file() else None
            scale = float(job.get("scenario", {}).get("demand_scale", math.nan))
            signature = str(job.get("signature"))
            if (
                method is None
                or output is None
                or PARTITION.resolve() not in output.parents
                or output in outputs
                or signature in signatures
                or scale not in SCALES
                or job.get("environment_block") != f"sumo-seed-{expected_seed}"
                or int(job.get("sumo_seed", -1)) != expected_seed
            ):
                raise ValueError(f"v16 job/output binding failed: {block} {name}")
            source_reference = validate_result(
                job, output, method, source_reference, frozen_sources
            )
            signatures.add(signature)
            block_signatures.add(signature)
            outputs.add(output)
            cells.add((method["algorithm"], method["training_seed"], block, scale))
        if len(block_signatures) != 45:
            raise ValueError(f"v16 block signature count drifted: {block}")
        block_stats[block] = {
            "matrix_manifest": file_record(manifest_path),
            "result_count": 45,
            "source_file_count": 51,
            "sumo_seed": expected_seed,
            "unique_job_signature_count": 45,
        }
    expected_cells = {
        (algorithm, seed, block, scale)
        for algorithm in ALGORITHMS
        for seed in SEEDS
        for block in BLOCKS
        for scale in SCALES
    }
    result_files = sorted(PARTITION.rglob("sumo_seed*.json"))
    if (
        cells != expected_cells
        or len(signatures) != 135
        or len(outputs) != 135
        or len(result_files) != 135
        or source_reference is None
    ):
        raise ValueError("v16 complete Cartesian result grid failed")
    bad_markers = (
        "Traceback (most recent call last)",
        "Segmentation fault",
        "FATAL:",
        "CUDA out of memory",
    )
    text_files = [path for path in files if path.suffix in {".log", ".out", ".err"}]
    if any(
        marker in path.read_text(errors="replace")
        for path in text_files
        for marker in bad_markers
    ):
        raise ValueError("v16 evaluation text error marker found")
    algorithm_counts = Counter(cell[0] for cell in cells)
    seed_counts = Counter(cell[1] for cell in cells)
    scale_counts = Counter(cell[3] for cell in cells)
    return {
        "blocks": block_stats,
        "algorithm_result_counts": dict(sorted(algorithm_counts.items())),
        "demand_scale_result_counts": {
            str(key): value for key, value in sorted(scale_counts.items())
        },
        "physical_behavior_count": 15,
        "result_count": 135,
        "seed_result_counts": {str(key): value for key, value in sorted(seed_counts.items())},
        "source_file_count": 51,
        "unique_job_signature_count": 135,
    }, files


def build() -> tuple[dict, dict]:
    controls, control_files, methods, frozen_sources, checkpoint_files = validate_controls()
    scheduler, source_sync, synced_files = validate_scheduler_and_sync()
    partition, partition_files = validate_partition(methods, frozen_sources)
    if synced_files - {EVIDENCE / "backend_exit_t83704.status"} != partition_files:
        raise ValueError("v16 audited partition differs from synchronized partition")
    audit = {
        "schema_version": "transit-h2oplus-v16-headway-reserve-evaluation-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_integrity": {
            "all_numeric_values_finite": True,
            "all_service_semantics_passed": True,
            "all_welfare_ledgers_passed": True,
            "matrix_count": 3,
            "physical_behavior_count": 15,
            "result_count": 135,
            "source_file_count": 51,
            "unique_parent_checkpoint_count": len(checkpoint_files),
            "unique_job_signature_count": 135,
            "passed": True,
        },
        "controls": controls,
        "deployment_aggregate_sha256": EXPECTED_DEPLOYMENT_AGGREGATE,
        "partition": partition,
        "scheduler": {
            "task_id": "t83704",
            "token_bound_backend_exit0": True,
            "scheduler_frontend_false_negative": scheduler[
                "scheduler_frontend_false_negative"
            ],
            "scheduler_evidence": file_record(CONTROL_PATHS["scheduler_evidence"]),
        },
        "source_sync": {
            "checksum_mode": source_sync["checksum_mode"],
            "file_count": 281,
            "passed": True,
            "source_sync": file_record(CONTROL_PATHS["source_sync"]),
        },
        "parent_training_scientific_stability": {
            "artifact_integrity_passed": True,
            "registered_runs": 15,
            "scientific_stability_passed": True,
            "new_v16_training_runs": 0,
            "separate_from_v16_evaluation_artifact_integrity": True,
        },
        "development_scientific_results": {
            "artifact_integrity_is_not_scientific_performance": True,
            "analysis_result_not_read_by_artifact_audit": True,
            "prior_v14_and_v15_development_outcomes_used_for_v16_design": True,
            "preregistered_gates_evaluated_in_separate_frozen_analysis": True,
        },
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "supports_formal_held_out_model_effect_claim": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "passed": True,
    }
    atomic_json(AUDIT_PATH, audit)

    tool_files = {
        Path(__file__).resolve(),
        REPO / "reproducibility/verify_h2oplus_v16_headway_reserve_evaluation_manifest.py",
        REPO / "reproducibility/sync_h2oplus_v16_headway_reserve_evaluation_v1.py",
        REPO / "reproducibility/analyze_h2oplus_v16_headway_reserve.py",
        REPO / "reproducibility/run_frozen_h2oplus_v16_headway_reserve_analysis_v1.py",
    }
    files = control_files | partition_files | tool_files | {AUDIT_PATH}
    records = [file_record(path) for path in sorted(files)]
    manifest = {
        "schema_version": "transit-h2oplus-v16-headway-reserve-evaluation-file-manifest-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_integrity": audit["artifact_integrity"],
        "audit": file_record(AUDIT_PATH),
        "file_count": len(records),
        "file_records_aggregate_sha256": records_aggregate(records),
        "files": records,
        "synchronized_partition_file_count": 281,
        "formal_held_out_outcomes_read": False,
        "scientific_analysis": {
            "included_in_artifact_manifest": False,
            "evaluated_separately_in_closeout": True,
            "formal_confirmation_authorized_or_submitted": False,
        },
        "study_stage": "development",
        "supports_formal_held_out_model_effect_claim": False,
        "total_size_bytes": sum(record["size_bytes"] for record in records),
    }
    atomic_json(MANIFEST_PATH, manifest)
    return audit, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=AUDIT_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global AUDIT_PATH, MANIFEST_PATH
    AUDIT_PATH = args.audit.expanduser().resolve()
    MANIFEST_PATH = args.manifest.expanduser().resolve()
    _, manifest = build()
    print(AUDIT_PATH)
    print(sha256(AUDIT_PATH))
    print(MANIFEST_PATH)
    print(sha256(MANIFEST_PATH))
    print(manifest["file_records_aggregate_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
