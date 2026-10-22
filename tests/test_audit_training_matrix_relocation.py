from pathlib import Path

import pytest

from reproducibility.audit_training_matrix_v6 import _job_at_matrix_root


def test_relocated_audit_job_preserves_identity_and_rebinds_run_dir(tmp_path: Path):
    original = {
        "name": "rlpd_seed5101",
        "method": "rlpd",
        "seed": 5101,
        "signature": "signed",
        "run_dir": Path("/remote/original/output/rlpd_seed5101"),
    }

    relocated = _job_at_matrix_root(original, tmp_path)

    assert relocated["run_dir"] == tmp_path / "rlpd_seed5101"
    assert relocated["signature"] == "signed"
    assert original["run_dir"] == Path("/remote/original/output/rlpd_seed5101")


def test_relocated_audit_job_uses_canonical_initializer(tmp_path: Path):
    initializer = (
        tmp_path
        / "core"
        / "h2oplus_seed5101"
        / "checkpoints"
        / "offline_initializer.pt"
    )
    initializer.parent.mkdir(parents=True)
    initializer.write_bytes(b"initializer")
    original = {
        "name": "h2oplus-no-ratio_seed5101",
        "seed": 5101,
        "run_dir": Path("/remote/run"),
        "offline_initializer": Path("/remote/initializer.pt"),
        "offline_initializer_dependency": {
            "source_job_name": "h2oplus_seed5101"
        },
    }

    relocated = _job_at_matrix_root(original, tmp_path / "matrix", tmp_path / "core")

    assert relocated["run_dir"] == tmp_path / "matrix" / original["name"]
    assert relocated["offline_initializer"] == initializer
    assert original["offline_initializer"] == Path("/remote/initializer.pt")


def test_relocated_audit_job_rejects_wrong_initializer_source(tmp_path: Path):
    original = {
        "name": "h2oplus-no-ratio_seed5101",
        "seed": 5101,
        "offline_initializer": Path("/remote/initializer.pt"),
        "offline_initializer_dependency": {
            "source_job_name": "h2oplus_seed5102"
        },
    }

    with pytest.raises(ValueError, match="unexpected initializer source"):
        _job_at_matrix_root(original, tmp_path / "matrix", tmp_path / "core")
