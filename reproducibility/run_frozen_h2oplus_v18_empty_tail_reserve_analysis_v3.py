#!/usr/bin/env python3
"""Run the unchanged frozen v18 analyzer through the v18-v3 adapter."""

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
RESULTS = REPRO / "h2oplus_v18_empty_tail_reserve_results"
ANALYZER = REPRO / "analyze_h2oplus_v18_empty_tail_reserve.py"
V1_REGISTRATION = REPRO / "h2oplus_v18_empty_tail_reserve_registration_v1.json"
V3_REGISTRATION = REPRO / "h2oplus_v18_empty_tail_reserve_successor_registration_v3.json"
V3_AUTHORIZATION = REPRO / "h2oplus_v18_empty_tail_reserve_successor_authorization_v3.json"
V3_CORRECTION = REPRO / "h2oplus_v18_empty_tail_reserve_namespace_binding_correction_v3.json"
V3_SOURCE_FREEZE = REPRO / "h2oplus_v18_empty_tail_reserve_source_freeze_v3.json"
FILE_MANIFEST = RESULTS / "evaluation_file_manifest_v3.json"
FILE_MANIFEST_VERIFICATION = RESULTS / "evaluation_file_manifest_verification_v3.json"
AUDIT = RESULTS / "evaluation_audit_v3.json"
EVIDENCE = RESULTS / "analysis_namespace_adapter_evidence_v3.json"
EXPECTED = {
    ANALYZER: "ebb90f93e5b43fce3670f465c0a8fe68459dd953b6bd3580594054c305acf1e6",
    V1_REGISTRATION: "45e160f7104a8a494d3231576c15c09ba9f28f184e3a6da4ac6637a34e3135dc",
    V3_REGISTRATION: "d589cf5ac1114522b5a162d0ffc8c2c1fcbdbe8829fca2ecdb88481ff7e02fa7",
    V3_AUTHORIZATION: "9c5872050e0b53d61e22ca3d0ad01bb4da626a88cde51f63d5c71672c536cb74",
    V3_CORRECTION: "f636d7d6e882eca5404c194f47c17789d1b27e985bada74006423735e35076af",
    V3_SOURCE_FREEZE: "d7f4522114edf4bbfd5d8d1a856ee512fc2e5c96c35bb582fc48c984912a434a",
    AUDIT: "053feec4df5d1ffdaaea4e494815a7378fa011956a2bc2703f14a5d3dd1ac64c",
    FILE_MANIFEST: "fab9283929d6bd42f5275d01cf998a7a71610594062b4ff8f3e315a9024cd986",
    FILE_MANIFEST_VERIFICATION: "503784f1d414e637673e1fbe23358c3910f998709fc072fba6b867460dd8786e",
}
REGISTERED_TO_MATRIX_BLOCK = {
    "v18-empty-tail-development-01": ("sumo-seed-7101", 7101),
    "v18-empty-tail-development-02": ("sumo-seed-7102", 7102),
    "v18-empty-tail-development-03": ("sumo-seed-7103", 7103),
}
EXPECTED_METHODS_SHA256 = "f87fec33f993e12fa49edad4264e34acce7ffde5dfa55c571197d424039c24e4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved.relative_to(ROOT)), "sha256": sha256(resolved), "bytes": resolved.stat().st_size}


def option_path(option: str, strict: bool = True) -> Path:
    for index, item in enumerate(sys.argv[1:], start=1):
        if item == option and index + 1 < len(sys.argv):
            return Path(sys.argv[index + 1]).expanduser().resolve(strict=strict)
        if item.startswith(option + "="):
            return Path(item.split("=", 1)[1]).expanduser().resolve(strict=strict)
    raise ValueError(f"{option} is required")


def matrix_roots() -> list[Path]:
    roots = []
    for index, item in enumerate(sys.argv[1:], start=1):
        if item == "--matrix-root" and index + 1 < len(sys.argv):
            roots.append(Path(sys.argv[index + 1]).expanduser().resolve(strict=True))
        elif item.startswith("--matrix-root="):
            roots.append(Path(item.split("=", 1)[1]).expanduser().resolve(strict=True))
    return roots


def load_analyzer():
    spec = importlib.util.spec_from_file_location("_frozen_h2oplus_v18_analysis", ANALYZER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen v18 analyzer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_operationalization(output: Path) -> dict:
    for path, expected in EXPECTED.items():
        if sha256(path.resolve(strict=True)) != expected:
            raise ValueError(f"v18-v3 analysis control drifted: {path}")
    registration = json.loads(V3_REGISTRATION.read_text())
    authorization = json.loads(V3_AUTHORIZATION.read_text())
    correction = json.loads(V3_CORRECTION.read_text())
    freeze = json.loads(V3_SOURCE_FREEZE.read_text())
    manifest_verification = json.loads(FILE_MANIFEST_VERIFICATION.read_text())
    if (
        option_path("--registration") != V1_REGISTRATION.resolve()
        or option_path("--source-freeze") != V3_SOURCE_FREEZE.resolve()
        or output.name != "empty_tail_reserve_analysis_v3.json"
        or registration.get("scientific_design_identity", {}).get("unchanged_from_v18_v1_and_v2") is not True
        or registration.get("v18_v1_or_v2_scientific_numeric_outcomes_read") is not False
        or registration.get("formal_held_out_outcomes_read") is not False
        or authorization.get("formal_held_out_outcomes_read") is not False
        or correction.get("scientific_behavior_changed") is not False
        or correction.get("methods_protocol_scenarios_or_gates_changed") is not False
        or correction.get("training_or_policy_artifacts_changed") is not False
        or correction.get("runtime_changed") is not False
        or correction.get("v18_v1_or_v2_results_reused") is not False
        or freeze.get("frozen_analyzer_unchanged_from_v18_v1") is not True
        or freeze.get("scientific_behavior_unchanged_from_v18_v1_and_v2") is not True
        or freeze.get("methods_protocol_scenarios_and_gates_unchanged_from_v18_v1_and_v2") is not True
        or freeze.get("v18_v1_results_reused") is not False
        or freeze.get("v18_v2_results_reused") is not False
        or freeze.get("analyzer", {}).get("sha256") != EXPECTED[ANALYZER]
        or manifest_verification.get("passed") is not True
        or manifest_verification.get("physical_result_count") != 135
    ):
        raise ValueError("v18-v3 scientific-equivalence adapter contract failed")
    return {
        "v1_registration": descriptor(V1_REGISTRATION),
        "v3_registration": descriptor(V3_REGISTRATION),
        "v3_authorization": descriptor(V3_AUTHORIZATION),
        "v3_correction": descriptor(V3_CORRECTION),
        "v3_source_freeze": descriptor(V3_SOURCE_FREEZE),
        "evaluation_audit": descriptor(AUDIT),
        "file_manifest": descriptor(FILE_MANIFEST),
        "file_manifest_verification": descriptor(FILE_MANIFEST_VERIFICATION),
        "scientific_behavior_or_gate_changed": False,
        "v18_v1_or_v2_results_reused": False,
    }


def validate_block_labels(roots: list[Path]) -> tuple[tuple[str, ...], list[dict]]:
    if len(roots) != 3 or len(set(roots)) != 3:
        raise ValueError("v18 analysis requires three unique matrix roots")
    observed = {}
    evidence = []
    for root in roots:
        registered = root.name
        expected = REGISTERED_TO_MATRIX_BLOCK.get(registered)
        manifest_path = root / "matrix_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        jobs = manifest.get("jobs", [])
        matrix_blocks = {str(job.get("environment_block")) for job in jobs}
        seeds = {int(job.get("sumo_seed", -1)) for job in jobs}
        partition = manifest.get("matrix_partition", {})
        if (
            expected is None
            or manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or manifest.get("methods_manifest_sha256") != EXPECTED_METHODS_SHA256
            or len(manifest.get("source_sha256", {})) != 59
            or len(jobs) != 45
            or matrix_blocks != {expected[0]}
            or seeds != {expected[1]}
            or partition.get("full_job_count") != 45
            or partition.get("selected_job_count") != 45
            or partition.get("shard_count") != 1
            or partition.get("shard_index") != 0
        ):
            raise ValueError(f"v18 registered/matrix block binding failed: {root}")
        observed[registered] = expected[0]
        evidence.append({
            "registered_block": registered,
            "matrix_block": expected[0],
            "sumo_seed": expected[1],
            "manifest": descriptor(manifest_path),
            "jobs": 45,
            "source_file_count": 59,
        })
    if set(observed) != set(REGISTERED_TO_MATRIX_BLOCK):
        raise ValueError("v18 registered block coverage is incomplete")
    return tuple(REGISTERED_TO_MATRIX_BLOCK[name][0] for name in REGISTERED_TO_MATRIX_BLOCK), evidence


def main() -> int:
    output = option_path("--output", strict=False)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v18-v3 analysis: {output}")
    if EVIDENCE.exists():
        raise FileExistsError(f"refusing to overwrite adapter evidence: {EVIDENCE}")
    operationalization = validate_operationalization(output)
    actual_blocks, matrices = validate_block_labels(matrix_roots())
    module = load_analyzer()
    registered_blocks = tuple(module.BLOCKS)
    if registered_blocks != tuple(REGISTERED_TO_MATRIX_BLOCK):
        raise RuntimeError("frozen v18 analyzer registered block contract drifted")
    module.BLOCKS = actual_blocks
    try:
        result = int(module.main())
    finally:
        if sha256(ANALYZER) != EXPECTED[ANALYZER]:
            raise RuntimeError("frozen v18 analyzer changed during execution")
    if result != 0 or not output.is_file():
        raise RuntimeError("frozen v18 analyzer did not produce the v3 output")
    payload = {
        "schema_version": "transit-h2oplus-v18-empty-tail-reserve-analysis-namespace-adapter-evidence-v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "frozen_analyzer": descriptor(ANALYZER),
        "frozen_analyzer_exit_code": result,
        "analysis_output": descriptor(output),
        "registered_to_matrix_block_mapping": {name: value[0] for name, value in REGISTERED_TO_MATRIX_BLOCK.items()},
        "matrices": matrices,
        "v18_v3_operationalization": operationalization,
        "adapter_scope": "in-memory frozen v18 analyzer BLOCKS constant only",
        "result_files_or_manifests_modified": False,
        "performance_values_or_preregistered_gate_formulas_changed": False,
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_eligible": False,
        "formal_confirmation_authorized": False,
    }
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    partial = EVIDENCE.with_suffix(EVIDENCE.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, EVIDENCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
