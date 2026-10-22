#!/usr/bin/env python3
"""Run frozen v9 analysis with a NumPy-scalar JSON serialization adapter."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from reproducibility import analyze_h2oplus_v9_adaptive as analysis


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reproducibility/h2oplus_v9_adaptive_results"
CORRECTION = RESULTS / "analysis_serialization_correction_v1.json"
EXPECTED_ANALYSIS_SHA256 = (
    "ced53b3a30353dab3d4d66c6503a17851d6f165f95178f9581613a60c7bf0fe8"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(original):
    def convert(encoder, value):
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        return original(encoder, value)

    return convert


def _output_argument() -> Path:
    if "--output" not in sys.argv:
        raise ValueError("frozen v9 analysis requires --output")
    index = sys.argv.index("--output")
    if index + 1 >= len(sys.argv):
        raise ValueError("--output has no value")
    return Path(sys.argv[index + 1]).expanduser().resolve()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    analysis_path = Path(analysis.__file__).resolve()
    if sha256(analysis_path) != EXPECTED_ANALYSIS_SHA256:
        raise ValueError("frozen v9 analysis source identity drifted")
    output = _output_argument()
    original = json.JSONEncoder.default
    json.JSONEncoder.default = _json_default(original)
    try:
        return_code = analysis.main()
    finally:
        json.JSONEncoder.default = original
    if return_code != 0 or not output.is_file():
        raise RuntimeError("frozen v9 analysis did not produce its report")
    report = json.loads(output.read_text())
    if (
        report.get("schema_version") != analysis.REPORT_SCHEMA
        or report.get("formal_held_out_outcomes_read") is not False
        or report.get("supports_formal_held_out_model_effect_claim") is not False
        or report.get("formal_confirmation_stage_authorized_or_submitted") is not False
    ):
        raise ValueError("v9 analysis report scope drifted")
    payload = {
        "schema_version": "transit-h2oplus-v9-adaptive-analysis-serialization-correction-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "incident": {
            "exception": "TypeError",
            "message": "Object of type bool is not JSON serializable",
            "stage": "report serialization after all metrics and gates were computed",
            "usable_report_created_by_failed_attempt": False,
        },
        "correction_scope": {
            "json_serialization_only": True,
            "numpy_bool_to_python_bool": True,
            "numpy_integer_to_python_int": True,
            "numpy_float_to_python_float": True,
            "analysis_inputs_changed": False,
            "analysis_metrics_changed": False,
            "preregistered_gate_logic_changed": False,
            "training_or_evaluation_artifacts_changed": False,
        },
        "frozen_analysis": {
            "path": str(analysis_path.relative_to(REPO)),
            "sha256": sha256(analysis_path),
        },
        "adapter": {
            "path": str(Path(__file__).resolve().relative_to(REPO)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "report": {
            "path": str(output.relative_to(REPO)),
            "sha256": sha256(output),
            "screening_gate_passed": report["development_scientific_result"][
                "screening_gate_passed"
            ],
        },
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "supports_formal_held_out_model_effect_claim": False,
        "passed": True,
    }
    atomic_json(CORRECTION, payload)
    print(CORRECTION)
    print(sha256(CORRECTION))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
