from pathlib import Path

from reproducibility.verify_ablation_merge_manifest_v7 import (
    file_records,
    records_sha256,
)


def test_file_records_are_deterministic_and_content_bound(tmp_path: Path):
    run = tmp_path / "h2oplus-no-ratio_seed5101"
    run.mkdir()
    artifact = run / "summary.json"
    artifact.write_text("before\n")

    before = file_records(tmp_path, [run.name])
    assert before == file_records(tmp_path, [run.name])

    artifact.write_text("after\n")
    after = file_records(tmp_path, [run.name])
    assert after != before
    assert records_sha256(after) != records_sha256(before)


def test_file_records_preserve_registered_run_order(tmp_path: Path):
    for name in ("b_seed5101", "a_seed5101"):
        run = tmp_path / name
        run.mkdir()
        (run / "artifact").write_text(name)

    records = file_records(tmp_path, ["b_seed5101", "a_seed5101"])

    assert [record["path"] for record in records] == [
        "b_seed5101/artifact",
        "a_seed5101/artifact",
    ]
