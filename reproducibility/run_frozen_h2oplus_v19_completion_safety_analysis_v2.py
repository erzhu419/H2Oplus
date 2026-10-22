#!/usr/bin/env python3
"""Run the unchanged frozen v19 analyzer through a fail-closed v2 wrapper."""

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
RESULTS = REPRO / "h2oplus_v19_completion_safety_results"
ANALYZER = REPRO / "analyze_h2oplus_v19_completion_safety.py"
SOURCE_FREEZE_V1 = REPRO / "h2oplus_v19_completion_safety_source_freeze_v1.json"
SOURCE_FREEZE_V2 = REPRO / "h2oplus_v19_completion_safety_source_freeze_v2.json"
CORRECTION = REPRO / "h2oplus_v19_completion_safety_environment_binding_correction_v2.json"
AUDIT = RESULTS / "evaluation_audit_v2.json"
FILE_MANIFEST = RESULTS / "evaluation_file_manifest_v2.json"
FILE_MANIFEST_VERIFICATION = RESULTS / "evaluation_file_manifest_verification_v2.json"
EVIDENCE = RESULTS / "analysis_execution_evidence_v2.json"
EXPECTED = {
    ANALYZER: "9c565719f681fc9f167de5b063ee36baf8f8584a83d0e345c9ea2c3a74b56f7d",
    SOURCE_FREEZE_V1: "9d8619304dd8f433d568012a25777a8f9a774eea0a4ba56cf7c7fed70007be63",
    SOURCE_FREEZE_V2: "6c77e7117b1c7499635e66e05d500bbc77e01e769530c1fe0d04b514c5dc11d7",
    CORRECTION: "2aec76694036ad038d2d15b4197c1444db8587171898ed0abdb7bdc410b4c282",
    AUDIT: "267370a31bed2bc8053842dc28e9223ae76c517ef3877603b3c9c81588b8636b",
    FILE_MANIFEST: "db80e286f39b7d8b1a984e6e1a5ffd6b1b0747e2bd61ac463ab663aa7298b125",
    FILE_MANIFEST_VERIFICATION: "867a110f72010f51e35f55d237ecd1b58a35a8ed6bedc1d326960775d3abb70e",
}
BLOCKS = {
    "v19-completion-safety-development-01": ("sumo-seed-7301", 7301),
    "v19-completion-safety-development-02": ("sumo-seed-7302", 7302),
    "v19-completion-safety-development-03": ("sumo-seed-7303", 7303),
}
EXPECTED_METHODS_SHA256 = "6d63b7dccf261859db7040d52b93482e17c8360d6db53b902de3f3f569729bb8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved.relative_to(ROOT)),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


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
    spec = importlib.util.spec_from_file_location("_frozen_h2oplus_v19_analysis", ANALYZER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen v19 analyzer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_controls(output: Path) -> dict:
    for path, expected in EXPECTED.items():
        if sha256(path.resolve(strict=True)) != expected:
            raise ValueError(f"v19-v2 analysis control drifted: {path}")
    freeze_v1 = json.loads(SOURCE_FREEZE_V1.read_text())
    freeze_v2 = json.loads(SOURCE_FREEZE_V2.read_text())
    correction = json.loads(CORRECTION.read_text())
    audit = json.loads(AUDIT.read_text())
    verification = json.loads(FILE_MANIFEST_VERIFICATION.read_text())
    if (
        option_path("--source-freeze") != SOURCE_FREEZE_V1.resolve()
        or output.name != "completion_safety_analysis_v2.json"
        or freeze_v1.get("schema_version")
        != "transit-h2oplus-v19-completion-safety-source-freeze-v1"
        or freeze_v2.get("schema_version")
        != "transit-h2oplus-v19-completion-safety-source-freeze-v2"
        or freeze_v1.get("analyzer") != freeze_v2.get("analyzer")
        or freeze_v1.get("methods") != freeze_v2.get("methods")
        or freeze_v1.get("protocol") != freeze_v2.get("protocol")
        or freeze_v1.get("matrix_sources") != freeze_v2.get("matrix_sources")
        or freeze_v2.get("analyzer", {}).get("sha256") != EXPECTED[ANALYZER]
        or freeze_v2.get("formal_held_out_outcomes_read") is not False
        or freeze_v2.get("v19_development_evaluation_outcomes_read") is not False
        or correction.get("status") != "frozen-predeployment-zero-result-correction"
        or correction.get("correction_scope", {}).get("change_analysis_or_gates")
        is not False
        or correction.get("correction_scope", {}).get("change_policy_behavior")
        is not False
        or correction.get("execution_boundary", {}).get(
            "completed_scientific_result_count"
        )
        != 0
        or correction.get("execution_boundary", {}).get(
            "scientific_numeric_outcomes_read"
        )
        is not False
        or audit.get("passed") is not True
        or audit.get("artifact_integrity", {}).get("physical_result_count") != 135
        or verification.get("passed") is not True
        or verification.get("physical_result_count") != 135
    ):
        raise ValueError("v19-v2 fail-closed analysis wrapper contract failed")
    return {
        "source_freeze_v1_analyzer_view": descriptor(SOURCE_FREEZE_V1),
        "source_freeze_v2_evaluation_identity": descriptor(SOURCE_FREEZE_V2),
        "environment_binding_correction": descriptor(CORRECTION),
        "evaluation_audit": descriptor(AUDIT),
        "file_manifest": descriptor(FILE_MANIFEST),
        "file_manifest_verification": descriptor(FILE_MANIFEST_VERIFICATION),
        "v1_v2_scientific_matrix_sources_methods_protocol_and_analyzer_equal": True,
        "scientific_behavior_or_gate_changed": False,
        "result_files_or_manifests_modified": False,
    }


def validate_matrices(roots: list[Path]) -> list[dict]:
    if len(roots) != 3 or len(set(roots)) != 3:
        raise ValueError("v19 analysis requires three unique matrix roots")
    evidence = []
    observed = set()
    for root in roots:
        registered = root.name
        expected = BLOCKS.get(registered)
        manifest_path = root / "matrix_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        jobs = manifest.get("jobs", [])
        partition = manifest.get("matrix_partition", {})
        if (
            expected is None
            or manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or manifest.get("methods_manifest_sha256") != EXPECTED_METHODS_SHA256
            or len(manifest.get("source_sha256", {})) != 55
            or len(jobs) != 45
            or {str(job.get("environment_block")) for job in jobs} != {expected[0]}
            or {int(job.get("sumo_seed", -1)) for job in jobs} != {expected[1]}
            or partition.get("full_job_count") != 45
            or partition.get("selected_job_count") != 45
            or partition.get("shard_count") != 1
            or partition.get("shard_index") != 0
        ):
            raise ValueError(f"v19 matrix binding failed: {root}")
        observed.add(registered)
        evidence.append(
            {
                "registered_block": registered,
                "matrix_block": expected[0],
                "sumo_seed": expected[1],
                "manifest": descriptor(manifest_path),
                "jobs": 45,
                "source_file_count": 55,
            }
        )
    if observed != set(BLOCKS):
        raise ValueError("v19 registered block coverage is incomplete")
    return evidence


def main() -> int:
    output = option_path("--output", strict=False)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v19-v2 analysis: {output}")
    if EVIDENCE.exists():
        raise FileExistsError(f"refusing to overwrite analysis evidence: {EVIDENCE}")
    controls = validate_controls(output)
    roots = matrix_roots()
    matrices = validate_matrices(roots)
    module = load_analyzer()
    if tuple(module.BLOCKS) != tuple(BLOCKS):
        raise RuntimeError("frozen v19 analyzer block contract drifted")
    registered_blocks = tuple(module.BLOCKS)
    module.BLOCKS = tuple(BLOCKS[name][0] for name in registered_blocks)
    try:
        result = int(module.main())
    finally:
        module.BLOCKS = registered_blocks
        if sha256(ANALYZER) != EXPECTED[ANALYZER]:
            raise RuntimeError("frozen v19 analyzer changed during execution")
    if result != 0 or not output.is_file():
        raise RuntimeError("frozen v19 analyzer did not produce the v2 output")
    payload = {
        "schema_version": "transit-h2oplus-v19-completion-safety-analysis-execution-evidence-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "frozen_analyzer": descriptor(ANALYZER),
        "frozen_analyzer_exit_code": result,
        "analysis_output": descriptor(output),
        "matrices": matrices,
        "controls": controls,
        "registered_to_matrix_block_mapping": {
            name: BLOCKS[name][0] for name in registered_blocks
        },
        "wrapper_scope": (
            "The unchanged analyzer consumes the superseded v1 schema view only after "
            "the wrapper proves its matrix sources, methods, protocol and analyzer are "
            "identical to the audited v2 evaluation identity. Only the preregistered "
            "development-block labels are mapped in memory to their verified matrix labels."
        ),
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
