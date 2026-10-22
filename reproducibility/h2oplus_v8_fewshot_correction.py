"""Fail-closed validation for v8 few-shot evaluation corrections."""

from __future__ import annotations


CORRECTION_SCHEMA_V1 = "transit-h2oplus-v8-fewshot-evaluation-correction-v1"
CORRECTION_SCHEMA_V2 = "transit-h2oplus-v8-fewshot-evaluation-correction-v2"
CORRECTION_SCHEMAS = {CORRECTION_SCHEMA_V1, CORRECTION_SCHEMA_V2}

MATRIX_SOURCE = "reproducibility/run_evaluation_matrix_v8_fewshot.py"
EVALUATOR_SOURCE = "SimpleSAC/evaluate_transit_v8_fewshot.py"

SOURCE_REASONS = {
    MATRIX_SOURCE: (
        "propagate the already-verified frozen protocol hash to evaluator child processes"
    ),
    EVALUATOR_SOURCE: (
        "include both v8 wrapper sources in every evaluator result source map so it "
        "matches the matrix source contract"
    ),
}


def _source_replacements(payload: dict, expected_paths: set[str]) -> dict[str, dict]:
    rows = payload.get("source_replacements")
    if not isinstance(rows, list):
        raise ValueError("evaluation correction source replacements are malformed")
    replacements = {row.get("path"): row for row in rows if isinstance(row, dict)}
    if len(replacements) != len(rows) or set(replacements) != expected_paths:
        raise ValueError("evaluation correction source replacement set drifted")
    for path, row in replacements.items():
        if (
            not isinstance(row.get("registered_sha256"), str)
            or not isinstance(row.get("corrected_sha256"), str)
            or row.get("reason") != SOURCE_REASONS[path]
        ):
            raise ValueError(f"evaluation correction source replacement drifted: {path}")
    return replacements


def validate_evaluation_correction(
    payload: dict, expected_protocol_sha256: str
) -> dict[str, dict]:
    """Validate one sealed correction and return its source replacements by path."""
    schema = payload.get("schema_version")
    if (
        schema not in CORRECTION_SCHEMAS
        or payload.get("study_stage") != "development"
        or payload.get("scope") != "evaluation-provenance-propagation-only"
        or payload.get("training_or_policy_artifacts_changed") is not False
        or payload.get("formal_held_out_outcomes_read") is not False
        or payload.get("frozen_protocol_sha256") != expected_protocol_sha256
    ):
        raise ValueError("few-shot evaluation correction scope drifted")

    if schema == CORRECTION_SCHEMA_V1:
        failed = payload.get("failed_task", {})
        cancelled = payload.get("cancelled_task", {})
        if (
            payload.get("failed_deployment_manifest_sha256")
            != "d5ca1ef143f44dfe7f4c1d01c7279dd05348ddf444999a725f8840987edf8c66"
            or failed.get("id") != "t82395"
            or failed.get("status") != "failed"
            or failed.get("exit_code") != 1
            or cancelled.get("id") != "t82396"
            or cancelled.get("status") != "cancelled"
            or cancelled.get("started") is not False
        ):
            raise ValueError("v1 few-shot evaluation correction incident drifted")
        return _source_replacements(payload, {MATRIX_SOURCE})

    invalid_tasks = payload.get("invalid_tasks")
    cancelled_tasks = payload.get("cancelled_retry_tasks")
    if not isinstance(invalid_tasks, list) or not isinstance(cancelled_tasks, list):
        raise ValueError("v2 few-shot evaluation correction task evidence is malformed")
    invalid_by_id = {row.get("id"): row for row in invalid_tasks if isinstance(row, dict)}
    cancelled_by_id = {
        row.get("id"): row for row in cancelled_tasks if isinstance(row, dict)
    }
    contract = payload.get("source_contract", {})
    if (
        payload.get("supersedes_correction_sha256")
        != "1068625af0bd49e87a1c34813be88187d1c3c5af31c77f81be966c2ad141e8c3"
        or payload.get("failed_deployment_manifest_sha256")
        != "084f76722ede341cfb3f21ec53c8062afd7a8941399f1e6d61abef6b4c9b0717"
        or payload.get("failed_deployment_aggregate_sha256")
        != "f657bded0c964b2258d47a8863f278a7a6b1b122554cec434df3ac57a0c21b75"
        or set(invalid_by_id) != {"t82400", "t82401"}
        or len(invalid_by_id) != len(invalid_tasks)
        or set(cancelled_by_id) != {"t82418", "t82447"}
        or len(cancelled_by_id) != len(cancelled_tasks)
        or contract.get("matrix_source_file_count") != 41
        or contract.get("result_source_file_count") != 39
        or set(contract.get("missing_result_sources", []))
        != {EVALUATOR_SOURCE, MATRIX_SOURCE}
        or payload.get("invalid_output_policy")
        != "incident-evidence-only; never merge, analyze, select, or reuse"
    ):
        raise ValueError("v2 few-shot evaluation correction incident drifted")
    for task_id, row in invalid_by_id.items():
        if (
            row.get("status") != "failed"
            or row.get("exit_code") != 1
            or row.get("result_json_files") != 105
            or row.get("result_log_files") != 105
            or row.get("matrix_source_file_count") != 41
            or row.get("result_source_file_count") != 39
            or not isinstance(row.get("backend_exit_file_sha256"), str)
            or not isinstance(row.get("host_execution_sha256"), str)
            or not isinstance(row.get("matrix_manifest_sha256"), str)
            or not isinstance(row.get("archived_partition_aggregate_sha256"), str)
        ):
            raise ValueError(f"v2 invalid task evidence drifted: {task_id}")
    for task_id, row in cancelled_by_id.items():
        if row.get("status") != "cancelled" or row.get("started") is not False:
            raise ValueError(f"v2 cancelled retry evidence drifted: {task_id}")
    return _source_replacements(payload, {MATRIX_SOURCE, EVALUATOR_SOURCE})
