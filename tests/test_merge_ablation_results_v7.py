import pytest

from reproducibility.merge_ablation_results_v7 import (
    EXPECTED_JOBS,
    dispatch_assignments,
    host_jobs,
    relocated_job,
)


def _execution(*job_names):
    return {
        "status": "completed",
        "gpus": [
            {
                "physical_gpu": "0",
                "status": "completed",
                "stages": [
                    {
                        "name": "ablation",
                        "status": "completed",
                        "workers": [
                            {
                                "name": "worker",
                                "status": "completed",
                                "returncode": 0,
                                "jobs": list(job_names),
                            }
                        ],
                    }
                ],
            }
        ],
    }


def test_host_jobs_requires_unique_successful_assignments():
    assert host_jobs(_execution("a", "b")) == ["a", "b"]
    with pytest.raises(ValueError, match="more than once"):
        host_jobs(_execution("a", "a"))


def test_host_jobs_rejects_nonzero_worker():
    payload = _execution("a")
    payload["gpus"][0]["stages"][0]["workers"][0]["returncode"] = 1
    with pytest.raises(ValueError, match="failed host worker"):
        host_jobs(payload)


def test_dispatch_assignments_preserve_registered_gpu_lanes():
    first = sorted(EXPECTED_JOBS)[:5]
    second = sorted(EXPECTED_JOBS)[5:]
    payload = {
        "status": "frozen-before-formal-outcomes",
        "hosts": [
            {
                "hostname": "jtl110gpu",
                "gpus": [
                    {
                        "physical_gpu": "0",
                        "stages": [{"workers": [{"jobs": first}]}],
                    }
                ],
            },
            {
                "hostname": "jtl110gpu2",
                "gpus": [
                    {
                        "physical_gpu": "1",
                        "stages": [{"workers": [{"jobs": second}]}],
                    }
                ],
            },
        ],
    }

    assert dispatch_assignments(payload) == {
        "jtl110gpu": {"0": first},
        "jtl110gpu2": {"1": second},
    }


def test_dispatch_assignments_reject_duplicate_job():
    names = sorted(EXPECTED_JOBS)
    payload = {
        "status": "frozen-before-formal-outcomes",
        "hosts": [
            {
                "hostname": "jtl110gpu",
                "gpus": [
                    {
                        "physical_gpu": "0",
                        "stages": [{"workers": [{"jobs": names}]}],
                    },
                    {
                        "physical_gpu": "1",
                        "stages": [{"workers": [{"jobs": [names[0]]}]}],
                    },
                ],
            }
        ],
    }

    with pytest.raises(ValueError, match="more than once"):
        dispatch_assignments(payload)


def test_relocated_job_uses_canonical_initializer(tmp_path):
    initializer = (
        tmp_path
        / "core"
        / "h2oplus_seed5101"
        / "checkpoints"
        / "offline_initializer.pt"
    )
    initializer.parent.mkdir(parents=True)
    initializer.write_bytes(b"initializer")
    job = {
        "name": "h2oplus-no-ratio_seed5101",
        "seed": 5101,
        "offline_initializer": tmp_path / "missing" / "offline_initializer.pt",
        "offline_initializer_dependency": {
            "source_job_name": "h2oplus_seed5101"
        },
    }

    relocated = relocated_job(job, tmp_path / "run", tmp_path / "core")

    assert relocated["run_dir"] == tmp_path / "run"
    assert relocated["offline_initializer"] == initializer
    assert job["offline_initializer"] != initializer


def test_relocated_job_rejects_wrong_initializer_source(tmp_path):
    job = {
        "name": "h2oplus-no-ratio_seed5101",
        "seed": 5101,
        "offline_initializer": tmp_path / "offline_initializer.pt",
        "offline_initializer_dependency": {
            "source_job_name": "h2oplus_seed5102"
        },
    }

    with pytest.raises(ValueError, match="unexpected initializer source"):
        relocated_job(job, tmp_path / "run", tmp_path / "core")
