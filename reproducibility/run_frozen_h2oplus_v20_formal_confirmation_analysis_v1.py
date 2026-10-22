#!/usr/bin/env python3
"""Run the frozen v20 formal analyzer with a fail-closed block-label adapter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
RESULTS = REPRO / "h2oplus_v20_formal_confirmation_results"
ANALYZER = REPRO / "analyze_h2oplus_v20_formal_confirmation.py"
REGISTRATION = REPRO / "h2oplus_v20_formal_confirmation_analysis_adapter_registration_v1.json"
EVIDENCE = RESULTS / "analysis_block_label_adapter_evidence_v1.json"

EXPECTED_ANALYZER_SHA256 = "ea69c2fe6b31de622952e2328f749f337ae94cb651a39633bdaec1031f9fdc4b"
EXPECTED_METHODS_SHA256 = "b4bcad09e13cb50c94d5541f2a52340ce99f15b281b5ddef57be38e792d052cd"
EXPECTED_DATASET_SHA256 = "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899"
EXPECTED_RUNTIME_SIGNATURE = "264d3c412b8e6c42320995208d2e009eb809b85626047206c4de752c2dae06e7"
REGISTERED_TO_MATRIX_BLOCK = {
    f"v20-formal-confirmation-{index:02d}": f"sumo-seed-{8100 + index}"
    for index in range(1, 11)
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_roots(argv: list[str]) -> list[Path]:
    for index, item in enumerate(argv[1:], start=1):
        if item == "--matrix-root":
            values = argv[index + 1 : index + 11]
            if len(values) != 10 or any(value.startswith("--") for value in values):
                raise ValueError("--matrix-root requires exactly ten values")
            return [Path(value).expanduser().resolve(strict=True) for value in values]
    raise ValueError("--matrix-root is required")


def output_path(argv: list[str]) -> Path:
    for index, item in enumerate(argv[1:], start=1):
        if item == "--output" and index + 1 < len(argv):
            return Path(argv[index + 1]).expanduser().resolve()
        if item.startswith("--output="):
            return Path(item.split("=", 1)[1]).expanduser().resolve()
    raise ValueError("--output is required")


def load_registration() -> dict:
    payload = json.loads(REGISTRATION.read_text())
    if (
        payload.get("schema_version")
        != "transit-h2oplus-v20-formal-confirmation-analysis-adapter-registration-v1"
        or payload.get("status") != "registered-after-zero-output-analysis-label-incident"
        or payload.get("frozen_analyzer_sha256") != EXPECTED_ANALYZER_SHA256
        or payload.get("adapter_sha256") != sha256(Path(__file__).resolve())
        or payload.get("registered_to_matrix_block") != REGISTERED_TO_MATRIX_BLOCK
        or payload.get("result_files_or_manifests_may_be_modified") is not False
        or payload.get("performance_values_or_gate_formulas_may_change") is not False
        or payload.get("formal_outcome_access_authorized") is not True
    ):
        raise ValueError("formal analysis adapter registration drifted")
    for item in payload.get("bindings", {}).values():
        path = ROOT / item["path"]
        if sha256(path.resolve(strict=True)) != item.get("sha256"):
            raise ValueError(f"analysis adapter binding drifted: {item.get('path')}")
    return payload


def validate_block_labels(roots: list[Path]) -> tuple[tuple[str, ...], list[dict]]:
    if len(roots) != 10 or len(set(roots)) != 10:
        raise ValueError("formal analysis requires ten unique matrix roots")
    observed: dict[str, str] = {}
    evidence = []
    for root in roots:
        registered = root.name
        actual = REGISTERED_TO_MATRIX_BLOCK.get(registered)
        manifest_path = root / "matrix_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        jobs = manifest.get("jobs", [])
        blocks = {str(job.get("environment_block")) for job in jobs}
        seeds = {int(job.get("sumo_seed", -1)) for job in jobs}
        expected_seed = 8100 + int(registered.rsplit("-", 1)[-1]) if actual else -1
        partition = manifest.get("matrix_partition", {})
        if (
            actual is None
            or manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or manifest.get("methods_manifest_sha256") != EXPECTED_METHODS_SHA256
            or manifest.get("dataset_sha256") != EXPECTED_DATASET_SHA256
            or manifest.get("sumo_runtime_signature") != EXPECTED_RUNTIME_SIGNATURE
            or len(manifest.get("source_sha256", {})) != 57
            or len(jobs) != 135
            or blocks != {actual}
            or seeds != {expected_seed}
            or partition.get("full_job_count") != 135
            or partition.get("selected_job_count") != 135
            or partition.get("shard_count") != 1
            or partition.get("shard_index") != 0
        ):
            raise ValueError(f"formal registered/matrix block binding failed: {root}")
        observed[registered] = actual
        evidence.append(
            {
                "registered_block": registered,
                "matrix_block": actual,
                "manifest": str(manifest_path.relative_to(ROOT)),
                "manifest_sha256": sha256(manifest_path),
                "jobs": 135,
                "source_file_count": 57,
                "sumo_seed": expected_seed,
            }
        )
    if observed != REGISTERED_TO_MATRIX_BLOCK:
        raise ValueError("formal registered block coverage is incomplete")
    return tuple(REGISTERED_TO_MATRIX_BLOCK.values()), evidence


def load_analyzer():
    spec = importlib.util.spec_from_file_location("_frozen_h2oplus_v20_formal", ANALYZER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen v20 formal analyzer")
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
    if sha256(ANALYZER) != EXPECTED_ANALYZER_SHA256:
        raise RuntimeError("frozen v20 formal analyzer identity drifted")
    if EVIDENCE.exists():
        raise FileExistsError(f"refusing to overwrite adapter evidence: {EVIDENCE}")
    registration = load_registration()
    roots = matrix_roots(sys.argv)
    actual_blocks, matrices = validate_block_labels(roots)
    output = output_path(sys.argv)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite analysis output: {output}")

    module = load_analyzer()
    registered_blocks = tuple(module.BLOCKS)
    if registered_blocks != tuple(REGISTERED_TO_MATRIX_BLOCK):
        raise RuntimeError("frozen analyzer registered block contract drifted")
    module.BLOCKS = actual_blocks
    try:
        result = int(module.main())
    finally:
        if sha256(ANALYZER) != EXPECTED_ANALYZER_SHA256:
            raise RuntimeError("frozen analyzer changed during execution")
    if result != 0 or not output.is_file():
        raise RuntimeError("frozen analyzer did not complete and create its output")

    atomic_json(
        EVIDENCE,
        {
            "schema_version": (
                "transit-h2oplus-v20-formal-confirmation-analysis-block-label-"
                "adapter-evidence-v1"
            ),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "passed": True,
            "registration": {
                "path": str(REGISTRATION.relative_to(ROOT)),
                "sha256": sha256(REGISTRATION),
            },
            "frozen_analyzer": {
                "path": str(ANALYZER.relative_to(ROOT)),
                "sha256": EXPECTED_ANALYZER_SHA256,
            },
            "frozen_analyzer_exit_code": result,
            "analysis_output": {
                "path": str(output.relative_to(ROOT)),
                "sha256": sha256(output),
                "bytes": output.stat().st_size,
            },
            "registered_to_matrix_block": REGISTERED_TO_MATRIX_BLOCK,
            "matrices": matrices,
            "adapter_scope": "in-memory frozen analyzer BLOCKS constant only",
            "result_files_or_manifests_modified": False,
            "performance_values_or_preregistered_gate_formulas_changed": False,
            "scientific_sources_changed": False,
            "prior_direct_invocation_zero_output_incident_bound": True,
            "formal_held_out_outcome_access_authorized": True,
            "formal_held_out_outcomes_read": True,
            "registration_status": registration["status"],
        },
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
