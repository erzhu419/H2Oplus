import json
import pytest
from pathlib import Path

from SimpleSAC import evaluate_transit_v8_fewshot as evaluator
from reproducibility import run_evaluation_matrix_v8_fewshot as matrix
from reproducibility.h2oplus_v8_fewshot_correction import (
    CORRECTION_SCHEMA_V2,
    EVALUATOR_SOURCE,
    MATRIX_SOURCE,
    SOURCE_REASONS,
    validate_evaluation_correction,
)


def test_bind_protocol_environment_sets_verified_hash(monkeypatch):
    protocol_hash = "a" * 64
    monkeypatch.delenv(matrix.PROTOCOL_SHA_ENV, raising=False)

    matrix.bind_protocol_environment(protocol_hash)

    assert matrix.os.environ[matrix.PROTOCOL_SHA_ENV] == protocol_hash


def test_bind_protocol_environment_accepts_matching_hash(monkeypatch):
    protocol_hash = "b" * 64
    monkeypatch.setenv(matrix.PROTOCOL_SHA_ENV, protocol_hash)

    matrix.bind_protocol_environment(protocol_hash)


def test_bind_protocol_environment_rejects_conflict(monkeypatch):
    monkeypatch.setenv(matrix.PROTOCOL_SHA_ENV, "c" * 64)

    with pytest.raises(ValueError, match="protocol environment conflicts"):
        matrix.bind_protocol_environment("d" * 64)


def test_evaluator_source_contract_includes_v8_wrappers(monkeypatch):
    base_source = Path("/tmp/base-evaluator.py")
    monkeypatch.setattr(
        evaluator,
        "_base_evaluation_source_paths",
        lambda _sumo_dir, _edge_xml: (base_source,),
    )

    sources = set(
        evaluator.evaluation_source_paths(Path("/tmp/sumo"), Path("/tmp/edge.xml"))
    )

    assert base_source.resolve() in sources
    assert Path(evaluator.__file__).resolve() in sources
    assert (
        evaluator.ROOT
        / "reproducibility"
        / "run_evaluation_matrix_v8_fewshot.py"
    ).resolve() in sources


def test_sealed_v1_evaluation_correction_still_validates():
    path = evaluator.ROOT / "reproducibility/h2oplus_v8_fewshot_evaluation_correction_v1.json"
    payload = json.loads(path.read_text())

    replacements = validate_evaluation_correction(
        payload, payload["frozen_protocol_sha256"]
    )

    assert set(replacements) == {MATRIX_SOURCE}


def _v2_correction_payload():
    return {
        "schema_version": CORRECTION_SCHEMA_V2,
        "study_stage": "development",
        "scope": "evaluation-provenance-propagation-only",
        "training_or_policy_artifacts_changed": False,
        "formal_held_out_outcomes_read": False,
        "frozen_protocol_sha256": "a" * 64,
        "supersedes_correction_sha256": (
            "1068625af0bd49e87a1c34813be88187d1c3c5af31c77f81be966c2ad141e8c3"
        ),
        "failed_deployment_manifest_sha256": (
            "084f76722ede341cfb3f21ec53c8062afd7a8941399f1e6d61abef6b4c9b0717"
        ),
        "failed_deployment_aggregate_sha256": (
            "f657bded0c964b2258d47a8863f278a7a6b1b122554cec434df3ac57a0c21b75"
        ),
        "invalid_output_policy": (
            "incident-evidence-only; never merge, analyze, select, or reuse"
        ),
        "source_contract": {
            "matrix_source_file_count": 41,
            "result_source_file_count": 39,
            "missing_result_sources": [EVALUATOR_SOURCE, MATRIX_SOURCE],
        },
        "invalid_tasks": [
            {
                "id": task_id,
                "status": "failed",
                "exit_code": 1,
                "result_json_files": 105,
                "result_log_files": 105,
                "matrix_source_file_count": 41,
                "result_source_file_count": 39,
                "backend_exit_file_sha256": "b" * 64,
                "host_execution_sha256": "c" * 64,
                "matrix_manifest_sha256": "d" * 64,
                "archived_partition_aggregate_sha256": "e" * 64,
            }
            for task_id in ("t82400", "t82401")
        ],
        "cancelled_retry_tasks": [
            {"id": task_id, "status": "cancelled", "started": False}
            for task_id in ("t82418", "t82447")
        ],
        "source_replacements": [
            {
                "path": path,
                "registered_sha256": "f" * 64,
                "corrected_sha256": "0" * 64,
                "reason": SOURCE_REASONS[path],
            }
            for path in (MATRIX_SOURCE, EVALUATOR_SOURCE)
        ],
    }


def test_v2_evaluation_correction_validates_exact_incident_contract():
    payload = _v2_correction_payload()

    replacements = validate_evaluation_correction(payload, "a" * 64)

    assert set(replacements) == {MATRIX_SOURCE, EVALUATOR_SOURCE}


def test_v2_evaluation_correction_rejects_missing_wrapper():
    payload = _v2_correction_payload()
    payload["source_contract"]["missing_result_sources"] = [MATRIX_SOURCE]

    with pytest.raises(ValueError, match="incident drifted"):
        validate_evaluation_correction(payload, "a" * 64)
