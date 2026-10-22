import pytest

from reproducibility.merge_baseline_results_v7 import host_jobs


def _execution(*job_names):
    return {
        "status": "completed",
        "gpus": [
            {
                "physical_gpu": "0",
                "status": "completed",
                "stages": [
                    {
                        "name": "baseline",
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
