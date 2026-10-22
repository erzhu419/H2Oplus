#!/usr/bin/env python3
"""Run the unchanged frozen v17 analyzer through the v17-v2 adapter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
ANALYZER = REPO / "reproducibility/analyze_h2oplus_v17_tail_service_reserve.py"
V1_REGISTRATION = REPO / "reproducibility/h2oplus_v17_tail_service_reserve_registration_v1.json"
V1_SOURCE_FREEZE = REPO / "reproducibility/h2oplus_v17_tail_service_reserve_evaluation_source_freeze_v1.json"
V2_REGISTRATION = REPO / "reproducibility/h2oplus_v17_tail_service_reserve_successor_registration_v2.json"
V2_AUTHORIZATION = REPO / "reproducibility/h2oplus_v17_tail_service_reserve_successor_authorization_v2.json"
V2_CORRECTION = REPO / "reproducibility/h2oplus_v17_tail_service_reserve_python38_correction_v2.json"
V2_SOURCE_FREEZE = REPO / "reproducibility/h2oplus_v17_tail_service_reserve_evaluation_source_freeze_v2.json"
EVIDENCE = (
    REPO
    / "reproducibility/h2oplus_v17_tail_service_reserve_results/"
    "analysis_block_label_adapter_evidence_v2.json"
)
EXPECTED_ANALYZER_SHA256 = (
    "66cfff74c8d1b4988b314945aa8ea06d03fe1594db8cc0f76ca3396bfab37f1a"
)
EXPECTED_METHODS_SHA256 = (
    "f2383ced11263bdd78287f7bc48f61be13ef1f181868827f555b9c06e9038ec5"
)
EXPECTED_V1_REGISTRATION_SHA256 = "d5465e677e29971eb1c6e88a926387bacd172de737f669f6fd84426bb7c68364"
EXPECTED_V1_SOURCE_FREEZE_SHA256 = "30e4b23219dc01c96b7eeaf89e4a8564716a887b9e525e8bcf65164ca8026f95"
EXPECTED_V2_REGISTRATION_SHA256 = "41d8399dcf07aedea421fed516407fbeff7fa4dda296652b8c70aa24582c90cf"
EXPECTED_V2_AUTHORIZATION_SHA256 = "4c61471e71c3190c3b450b2805eba0a48712da8c25a9840806b3bd849029ebd4"
EXPECTED_V2_CORRECTION_SHA256 = "1a521809e08618001b19e5d2d7d1077472d819aa648fcaa46e87423e833268ed"
EXPECTED_V2_SOURCE_FREEZE_SHA256 = "5d0e4496183cbd90a2a82191b10255b4e27c70d2c56b41cb2fc651d344a1724e"
REGISTERED_TO_MATRIX_BLOCK = {
    "v12-staged-release-development-01": "sumo-seed-5101",
    "v12-staged-release-development-02": "sumo-seed-5102",
    "v12-staged-release-development-03": "sumo-seed-5103",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_roots(argv: list[str]) -> list[Path]:
    roots: list[Path] = []
    index = 1
    while index < len(argv):
        item = argv[index]
        if item == "--v17-matrix-root":
            if index + 1 >= len(argv):
                raise ValueError("--v17-matrix-root lacks a value")
            roots.append(Path(argv[index + 1]).expanduser().resolve(strict=True))
            index += 2
            continue
        if item.startswith("--v17-matrix-root="):
            roots.append(
                Path(item.split("=", 1)[1]).expanduser().resolve(strict=True)
            )
        index += 1
    return roots


def output_path(argv: list[str]) -> Path:
    for index, item in enumerate(argv[1:], start=1):
        if item == "--output" and index + 1 < len(argv):
            return Path(argv[index + 1]).expanduser().resolve()
        if item.startswith("--output="):
            return Path(item.split("=", 1)[1]).expanduser().resolve()
    raise ValueError("--output is required")


def option_path(argv: list[str], option: str) -> Path:
    for index, item in enumerate(argv[1:], start=1):
        if item == option and index + 1 < len(argv):
            return Path(argv[index + 1]).expanduser().resolve(strict=True)
        if item.startswith(option + "="):
            return Path(item.split("=", 1)[1]).expanduser().resolve(strict=True)
    raise ValueError(f"{option} is required")


def validate_v2_operationalization(argv: list[str], output: Path) -> dict:
    expected = {
        V1_REGISTRATION: EXPECTED_V1_REGISTRATION_SHA256,
        V1_SOURCE_FREEZE: EXPECTED_V1_SOURCE_FREEZE_SHA256,
        V2_REGISTRATION: EXPECTED_V2_REGISTRATION_SHA256,
        V2_AUTHORIZATION: EXPECTED_V2_AUTHORIZATION_SHA256,
        V2_CORRECTION: EXPECTED_V2_CORRECTION_SHA256,
        V2_SOURCE_FREEZE: EXPECTED_V2_SOURCE_FREEZE_SHA256,
    }
    if any(sha256(path) != digest for path, digest in expected.items()):
        raise ValueError("v17-v2 operational control identity drifted")
    v1_freeze = json.loads(V1_SOURCE_FREEZE.read_text())
    v2_freeze = json.loads(V2_SOURCE_FREEZE.read_text())
    correction = json.loads(V2_CORRECTION.read_text())
    registration = json.loads(V2_REGISTRATION.read_text())
    if (
        option_path(argv, "--registration") != V1_REGISTRATION.resolve()
        or option_path(argv, "--source-freeze") != V1_SOURCE_FREEZE.resolve()
        or output.name != "tail_service_reserve_analysis_v2.json"
        or v1_freeze.get("matrix_sources") != v2_freeze.get("matrix_sources")
        or v2_freeze.get("scientific_matrix_sources_unchanged_from_v17_v1")
        is not True
        or v2_freeze.get("frozen_analyzer_unchanged_from_v17_v1") is not True
        or correction.get("scientific_behavior_changed") is not False
        or correction.get("scientific_source_paths_changed") != []
        or correction.get("v1_incident_results_reused") is not False
        or registration.get("v17_v1_scientific_numeric_outcomes_read") is not False
        or registration.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("v17-v2 scientific-equivalence adapter contract failed")
    return {
        "v1_registration": {"path": str(V1_REGISTRATION.relative_to(REPO)), "sha256": sha256(V1_REGISTRATION)},
        "v1_source_freeze": {"path": str(V1_SOURCE_FREEZE.relative_to(REPO)), "sha256": sha256(V1_SOURCE_FREEZE)},
        "v2_registration": {"path": str(V2_REGISTRATION.relative_to(REPO)), "sha256": sha256(V2_REGISTRATION)},
        "v2_authorization": {"path": str(V2_AUTHORIZATION.relative_to(REPO)), "sha256": sha256(V2_AUTHORIZATION)},
        "v2_correction": {"path": str(V2_CORRECTION.relative_to(REPO)), "sha256": sha256(V2_CORRECTION)},
        "v2_source_freeze": {"path": str(V2_SOURCE_FREEZE.relative_to(REPO)), "sha256": sha256(V2_SOURCE_FREEZE)},
        "scientific_matrix_sources_exactly_equal": True,
        "v1_incident_results_reused": False,
    }


def validate_block_labels(roots: list[Path]) -> tuple[tuple[str, ...], list[dict]]:
    if len(roots) != 3 or len(set(roots)) != 3:
        raise ValueError("v17 analysis requires three unique matrix roots")
    observed: dict[str, str] = {}
    evidence: list[dict] = []
    for root in roots:
        registered = root.name
        actual = REGISTERED_TO_MATRIX_BLOCK.get(registered)
        manifest_path = root / "matrix_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        jobs = manifest.get("jobs", [])
        blocks = {str(job.get("environment_block")) for job in jobs}
        seeds = {int(job.get("sumo_seed", -1)) for job in jobs}
        expected_seed = 5100 + int(registered.rsplit("-", 1)[-1]) if actual else -1
        partition = manifest.get("matrix_partition", {})
        if (
            actual is None
            or manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or manifest.get("methods_manifest_sha256") != EXPECTED_METHODS_SHA256
            or len(manifest.get("source_sha256", {})) != 53
            or len(jobs) != 45
            or blocks != {actual}
            or seeds != {expected_seed}
            or partition.get("full_job_count") != 45
            or partition.get("selected_job_count") != 45
            or partition.get("shard_count") != 1
            or partition.get("shard_index") != 0
        ):
            raise ValueError(f"v17 registered/matrix block binding failed: {root}")
        observed[registered] = actual
        evidence.append(
            {
                "registered_block": registered,
                "matrix_block": actual,
                "manifest": str(manifest_path.relative_to(REPO)),
                "manifest_sha256": sha256(manifest_path),
                "jobs": 45,
                "source_file_count": 53,
                "sumo_seed": expected_seed,
            }
        )
    if observed != REGISTERED_TO_MATRIX_BLOCK:
        raise ValueError("v17 registered block coverage is incomplete")
    return tuple(REGISTERED_TO_MATRIX_BLOCK.values()), evidence


def load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "_frozen_h2oplus_v17_analysis", ANALYZER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen v17 analyzer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    actual_hash = sha256(ANALYZER)
    if actual_hash != EXPECTED_ANALYZER_SHA256:
        raise RuntimeError(f"frozen v17 analyzer identity drifted: {actual_hash}")
    if EVIDENCE.exists():
        raise FileExistsError(f"refusing to overwrite adapter evidence: {EVIDENCE}")
    actual_blocks, matrices = validate_block_labels(matrix_roots(sys.argv))
    output = output_path(sys.argv)
    operationalization = validate_v2_operationalization(sys.argv, output)
    module = load_analyzer()
    registered_blocks = tuple(module.BLOCKS)
    if registered_blocks != tuple(REGISTERED_TO_MATRIX_BLOCK):
        raise RuntimeError("frozen v17 analyzer registered block contract drifted")
    parent_registered_blocks = tuple(module.v14.BLOCKS)
    if parent_registered_blocks != tuple(REGISTERED_TO_MATRIX_BLOCK):
        raise RuntimeError("frozen v14 helper registered block contract drifted")

    original_default = json.JSONEncoder.default

    def numpy_scalar_default(encoder: json.JSONEncoder, value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        return original_default(encoder, value)

    module.BLOCKS = actual_blocks
    module.v14.BLOCKS = actual_blocks
    json.JSONEncoder.default = numpy_scalar_default
    try:
        result = int(module.main())
    finally:
        json.JSONEncoder.default = original_default
        if sha256(ANALYZER) != EXPECTED_ANALYZER_SHA256:
            raise RuntimeError("frozen v17 analyzer changed during execution")
    if not output.is_file():
        raise FileNotFoundError(f"frozen analyzer did not create output: {output}")
    atomic_json(
        EVIDENCE,
        {
            "schema_version": (
                "transit-h2oplus-v17-tail-service-reserve-analysis-block-label-"
                "adapter-evidence-v2"
            ),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_analyzer": {
                "path": str(ANALYZER.relative_to(REPO)),
                "sha256": actual_hash,
            },
            "frozen_analyzer_exit_code": result,
            "analysis_output": {
                "path": str(output.relative_to(REPO)),
                "sha256": sha256(output),
                "size_bytes": output.stat().st_size,
            },
            "registered_to_matrix_block_mapping": REGISTERED_TO_MATRIX_BLOCK,
            "matrices": matrices,
            "v17_v2_operationalization": operationalization,
            "adapter_scope": (
                "in-memory v17 analyzer and imported frozen v14 helper "
                "block-label constants only"
            ),
            "v17_v2_operational_identity_only": True,
            "result_files_or_manifests_modified": False,
            "performance_values_or_preregistered_gate_formulas_changed": False,
            "formal_held_out_outcomes_read": False,
            "formal_confirmation_stage_authorized_or_submitted": False,
            "passed": True,
        },
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
