import pytest

from reproducibility.build_ablation_merge_manifest_v7 import _signed_job_map


def test_signed_job_map_uses_matrix_derived_signatures():
    matrix = {
        "jobs": [
            {"name": "h2oplus-no-ratio_seed5101", "signature": "signed"}
        ]
    }

    assert _signed_job_map(matrix)["h2oplus-no-ratio_seed5101"][
        "signature"
    ] == "signed"


def test_signed_job_map_rejects_duplicate_jobs():
    matrix = {"jobs": [{"name": "same"}, {"name": "same"}]}

    with pytest.raises(ValueError, match="repeats"):
        _signed_job_map(matrix)
