from __future__ import annotations

import json
from pathlib import Path

import pytest

from reproducibility import build_h2oplus_v10_delayed_evaluation_manifest as builder
from reproducibility import verify_h2oplus_v10_delayed_evaluation_manifest as verifier


def sample_contract() -> tuple[dict, Path, dict, dict]:
    manifest_path = next(
        builder.PARTITIONS.rglob("matrix_manifest.shard-0000-of-0002.json")
    )
    manifest = json.loads(manifest_path.read_text())
    job = manifest["jobs"][0]
    output = Path(job["output"]).resolve(strict=True)
    methods = {
        item["name"]: item
        for item in json.loads(builder.CONTROL_PATHS["methods"].read_text())["methods"]
    }
    source_freeze = json.loads(builder.CONTROL_PATHS["source_freeze"].read_text())
    sources = {item["path"]: item["sha256"] for item in source_freeze["sources"]}
    return job, output, methods[job["method"]["name"]], sources


def test_valid_result_passes_both_v10_auditors() -> None:
    job, output, method, sources = sample_contract()
    expected = json.loads(output.read_text())["source_sha256"]
    assert builder.validate_result(job, output, method, expected, sources) == expected
    assert verifier._validate_result(job, output, method, expected, sources) == expected


def test_heldout_scope_tamper_fails_closed(tmp_path: Path) -> None:
    job, output, method, sources = sample_contract()
    payload = json.loads(output.read_text())
    payload["policy"]["training_metadata"]["formal_held_out_outcomes_read"] = True
    tampered = tmp_path / output.name
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="provenance"):
        builder.validate_result(job, tampered, method, None, sources)
    with pytest.raises(ValueError, match="provenance"):
        verifier._validate_result(job, tampered, method, None, sources)


def test_service_semantics_tamper_fails_closed(tmp_path: Path) -> None:
    job, output, method, sources = sample_contract()
    payload = json.loads(output.read_text())
    payload["bus_service_semantics_audit"]["passed"] = False
    tampered = tmp_path / output.name
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="semantics"):
        builder.validate_result(job, tampered, method, None, sources)
    with pytest.raises(ValueError, match="provenance"):
        verifier._validate_result(job, tampered, method, None, sources)


def test_source_contract_tamper_fails_closed(tmp_path: Path) -> None:
    job, output, method, sources = sample_contract()
    payload = json.loads(output.read_text())
    payload["source_sha256"].pop(next(iter(payload["source_sha256"])))
    tampered = tmp_path / output.name
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="provenance"):
        builder.validate_result(job, tampered, method, None, sources)
    with pytest.raises(ValueError, match="provenance"):
        verifier._validate_result(job, tampered, method, None, sources)


def test_nonfinite_payload_is_rejected() -> None:
    assert builder.finite({"value": float("nan")}) is False
    assert verifier.finite({"value": float("inf")}) is False
