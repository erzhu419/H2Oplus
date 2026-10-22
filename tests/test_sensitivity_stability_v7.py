import json
from pathlib import Path

from reproducibility.audit_sensitivity_stability_v7 import audit_run


def _write_run(root: Path, violation_epochs: set[int]) -> Path:
    run = root / "h2oplus-lambda0_seed5104"
    run.mkdir()
    rows = []
    for epoch in range(1, 191):
        rows.append(
            {
                "epoch": epoch,
                "train": {
                    "train/critic_grad_norm": 100.0,
                    "train/hold_probability": (
                        0.001 if epoch in violation_epochs else 0.5
                    ),
                    "train/positive_action_mean": 0.0,
                },
            }
        )
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    payload = {
        "method": "h2oplus-lambda0",
        "seed": 5104,
        "run_signature": "signed",
        "total_updates": 145000,
    }
    (run / "summary.json").write_text(json.dumps(payload))
    (run / "metadata.json").write_text(json.dumps(payload))
    return run


def test_sensitivity_audit_preserves_each_episode_and_recovery(tmp_path: Path):
    report = audit_run(_write_run(tmp_path, {42, 43, 44, 80}), "signed")

    assert not report["passed"]
    policy = report["policy_nondegeneracy"]
    assert policy["violation_epochs"] == [42, 43, 44, 80]
    assert policy["violation_episodes"] == [
        {
            "first_epoch": 42,
            "last_epoch": 44,
            "epochs": [42, 43, 44],
            "first_recovery_epoch": 45,
            "recovered": True,
        },
        {
            "first_epoch": 80,
            "last_epoch": 80,
            "epochs": [80],
            "first_recovery_epoch": 81,
            "recovered": True,
        },
    ]


def test_sensitivity_audit_accepts_finite_in_bounds_run(tmp_path: Path):
    report = audit_run(_write_run(tmp_path, set()), "signed")

    assert report["passed"]
    assert report["policy_nondegeneracy"]["violation_episodes"] == []
