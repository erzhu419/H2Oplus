import json
from pathlib import Path

from reproducibility.audit_rlpd_long_horizon_stability_v7 import audit_run


def _write_run(root: Path, *, crossing: bool) -> Path:
    run = root / "rlpd_seed5101"
    run.mkdir()
    rows = []
    for epoch in range(1, 191):
        critic = 1_100_000.0 if crossing and epoch == 170 else 100.0
        hold = 0.001 if crossing and epoch == 184 else 0.5
        action = -0.99 if crossing and epoch == 184 else 0.0
        rows.append(
            {
                "epoch": epoch,
                "train": {
                    "train/critic_grad_norm": critic,
                    "train/hold_probability": hold,
                    "train/positive_action_mean": action,
                },
            }
        )
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (run / "summary.json").write_text(
        json.dumps({"seed": 5101, "run_signature": "signed", "total_updates": 1_900_000})
    )
    (run / "metadata.json").write_text(json.dumps({"run_signature": "signed"}))
    return run


def test_stability_audit_records_crossing_and_recovery(tmp_path: Path):
    report = audit_run(_write_run(tmp_path, crossing=True))

    assert not report["passed"]
    assert report["critic_gradient"]["first_limit_crossing"]["epoch"] == 170
    policy = report["policy_nondegeneracy"]
    assert policy["first_crossing_epoch"] == 184
    assert policy["recovered_inside_bounds_after_first_crossing"]
    assert policy["first_recovery_epoch"] == 185


def test_stability_audit_accepts_full_finite_series_inside_bounds(tmp_path: Path):
    report = audit_run(_write_run(tmp_path, crossing=False))

    assert report["passed"]
    assert report["critic_gradient"]["first_limit_crossing"] is None
