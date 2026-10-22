import pytest

from reproducibility.analyze_collection_matrix_v6 import (
    clustered_paired_summary,
)


def test_v6_behavior_analysis_clusters_correlated_demand_conditions():
    differences = {
        0: (10, -1.0),
        1: (10, -3.0),
        2: (11, -2.0),
        3: (11, -4.0),
        4: (12, -5.0),
        5: (12, -1.0),
    }
    result = clustered_paired_summary(
        differences, seed=3, bootstrap_replicates=200
    )
    assert result["environment_cell_differences"]["n"] == 6
    assert result["realization_block_mean_differences"]["n"] == 3
    assert result["realization_block_mean_differences"]["mean"] == pytest.approx(
        -8.0 / 3.0
    )
    assert result["realization_block_wins"] == 3
    assert result["independent_resampling_units"] == 3
    assert result["demand_conditions_per_unit"] == 2
    assert result["exact_two_sided_sign_flip_pvalue"] == pytest.approx(0.25)


def test_v6_behavior_analysis_rejects_incomplete_realization_blocks():
    with pytest.raises(ValueError, match="unequal demand-condition"):
        clustered_paired_summary(
            {0: (10, -1.0), 1: (10, -2.0), 2: (11, -3.0)},
            bootstrap_replicates=10,
        )
