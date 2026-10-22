import pytest

from reproducibility.analyze_h2oplus_mechanisms_v7 import (
    EXPECTED_EPOCHS,
    _find_run,
    _validated_methods,
    tail_event_weighted_reward,
)


def test_tail_reward_uses_exact_event_budget():
    rows = [
        {"rollout": {"events": 6, "mean_reward": -10.0}},
        {"rollout": {"events": 7, "mean_reward": -4.0}},
        {"rollout": {"events": 8, "mean_reward": -1.0}},
    ]

    result = tail_event_weighted_reward(rows, event_budget=10)

    assert result["events"] == 10
    assert result["mean_reward_per_event"] == pytest.approx(-1.6)


def test_tail_reward_rejects_underfilled_run():
    with pytest.raises(ValueError, match="only 3"):
        tail_event_weighted_reward(
            [{"rollout": {"events": 3, "mean_reward": -1.0}}],
            event_budget=4,
        )


def test_find_run_rejects_duplicate_synchronized_sources(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "h2oplus_seed5101").mkdir(parents=True)
    (second / "h2oplus_seed5101").mkdir(parents=True)

    with pytest.raises(ValueError, match="exactly one"):
        _find_run([first, second], "h2oplus_seed5101")


def test_stage_local_analysis_requires_unique_h2oplus_reference():
    assert _validated_methods(
        ("h2oplus", "h2oplus-no-ratio", "h2oplus-no-warmstart")
    ) == ("h2oplus", "h2oplus-no-ratio", "h2oplus-no-warmstart")
    with pytest.raises(ValueError, match="requires the h2oplus reference"):
        _validated_methods(("h2oplus-no-ratio",))
    with pytest.raises(ValueError, match="unique"):
        _validated_methods(("h2oplus", "h2oplus"))


def test_source_protocol_uses_its_registered_198_epoch_contract():
    assert EXPECTED_EPOCHS["h2oplus-source"] == 198
    assert EXPECTED_EPOCHS["h2oplus"] == 190
    assert all(
        epochs == 190
        for method, epochs in EXPECTED_EPOCHS.items()
        if method != "h2oplus-source"
    )
