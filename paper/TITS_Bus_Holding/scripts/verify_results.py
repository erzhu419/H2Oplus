#!/usr/bin/env python3
"""Check the manuscript's primary comparison using only saved endpoint data.

Usage: python3 scripts/verify_results.py
The script also works when invoked from another directory. It uses only the
Python standard library, reads two packaged files, and changes no files.
It does not rerun simulation or validate the original simulator implementation.

CSV method identifiers are retained unchanged. The JSON's legacy endpoint
label "with" means the proposed 60-second, wider-window rule; "without" means
the matched reference WITHOUT THAT ADDED RULE, still retaining its earlier
20-second rule. Neither CSV group represents a controller without any rule.

The checks reproduce the equal-weight episode-cell averages, then average the
15 paired differences inside each independent environment block before the
prespecified directional sign tests. No new statistical analysis is added.
"""

import csv
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import sys


DATA = Path(__file__).resolve().parents[1] / "data"
METRICS = ("primary", "completion", "unfinished", "waiting", "in_vehicle")
METHOD_PATTERN = re.compile(
    r"h2oplus-v19-completion-safety-(baseline|quarter-load-sixty)"
    r"-seed(\d+)-events000025000"
)
ROLE_BY_VARIANT = {"baseline": "reference", "quarter-load-sixty": "proposed"}
JSON_LABEL_BY_ROLE = {"proposed": "with", "reference": "without"}
EXPECTED_BLOCKS = tuple(range(8101, 8111))
EXPECTED_SEEDS = tuple(range(6201, 6206))
EXPECTED_DEMANDS = (0.75, 1.0, 1.25)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    tolerance = 1e-12 if "completion" in label else 1e-9
    require(
        math.isclose(actual, float(expected), rel_tol=1e-12, abs_tol=tolerance),
        "{} differs: calculated={!r}, recorded={!r}".format(label, actual, expected),
    )


def integer(value, label):
    number = float(value)
    require(math.isfinite(number) and number.is_integer(), label + " is not an integer")
    return int(number)


def averages(rows):
    require(bool(rows), "Cannot average an empty group")
    return {metric: statistics.mean(row[metric] for row in rows) for metric in METRICS}


def compare_metrics(calculated, recorded, label):
    for metric in METRICS:
        close(calculated[metric], recorded[metric], label + "." + metric)


def check_group_summaries(differences, group_field, saved, label):
    values = sorted({row[group_field] for row in differences})
    require(
        {float(key) for key in saved} == {float(value) for value in values},
        label + " has different group identifiers",
    )
    calculated = {}
    for value in values:
        calculated[value] = averages([row for row in differences if row[group_field] == value])
        saved_key = next(key for key in saved if float(key) == float(value))
        compare_metrics(calculated[value], saved[saved_key], label + "[" + str(value) + "]")
    return calculated


def sign_probability(wins, total):
    return sum(math.comb(total, count) for count in range(wins, total + 1)) / 2 ** total


def main():
    with (DATA / "paired_episode_endpoints.csv").open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    with (DATA / "verified_results.json").open(encoding="utf-8") as handle:
        verified = json.load(handle)
    require(len(records) == 300, "Expected exactly 300 primary-comparison endpoint records")
    require(len({row["path"] for row in records}) == 300, "Duplicate endpoint paths")

    paired = {}
    actors_by_seed = {}
    for raw in records:
        match = METHOD_PATTERN.fullmatch(raw["method"])
        require(match is not None, "Unexpected original method label: " + raw["method"])
        role = ROLE_BY_VARIANT[match.group(1)]
        seed = integer(raw["training_seed"], "training_seed")
        require(seed == int(match.group(2)), "Method label and training seed disagree")
        block = integer(raw["block"], "block")
        demand = float(raw["demand_scale"])
        actor = raw["actor"]
        require(re.fullmatch(r"[0-9a-f]{64}", actor) is not None, "Invalid actor hash")
        require(
            seed not in actors_by_seed or actors_by_seed[seed] == actor,
            "Actor tensor changes across scenarios or controllers for seed " + str(seed),
        )
        actors_by_seed[seed] = actor

        row = {metric: float(raw[metric]) for metric in METRICS}
        require(all(math.isfinite(value) for value in row.values()), "Nonfinite endpoint")
        row.update({"actor": actor, "departed": integer(raw["departed"], "departed")})
        require(row["departed"] > 0, "Nonpositive departed-passenger denominator")
        require(0 <= row["completion"] <= 1, "Invalid completion rate")
        require(integer(raw["endpoint"], "endpoint") == 24000, "Different evaluation horizon")
        require(float(raw["ledger_error"]) == 0, "Nonzero recorded ledger reconstruction error")

        waiting_total = integer(raw["wait_total"], "wait_total")
        ride_total = integer(raw["in_vehicle_total"], "in_vehicle_total")
        generalized_total = integer(raw["generalized_total"], "generalized_total")
        require(2 * waiting_total + ride_total == generalized_total, "Total-cost identity fails")
        close(row["waiting"], waiting_total / row["departed"], "episode.waiting")
        close(row["in_vehicle"], ride_total / row["departed"], "episode.in_vehicle")
        close(row["primary"], generalized_total / row["departed"], "episode.primary")
        close(row["primary"], 2 * row["waiting"] + row["in_vehicle"], "episode.2W+V")
        close(row["completion"], 1 - row["unfinished"] / row["departed"], "episode.completion")

        key = (block, seed, demand)
        group = paired.setdefault(key, {})
        require(role not in group, "Duplicate role in pair " + str(key))
        group[role] = row

    require(
        set(paired) == set(itertools.product(EXPECTED_BLOCKS, EXPECTED_SEEDS, EXPECTED_DEMANDS)),
        "The 10-block by 5-seed by 3-demand matrix is incomplete or unexpected",
    )
    require(len(set(actors_by_seed.values())) == 5, "Expected five distinct fixed actor tensors")
    differences = []
    for (block, seed, demand), group in sorted(paired.items()):
        require(set(group) == {"proposed", "reference"}, "Missing controller in a matched pair")
        proposed, reference = group["proposed"], group["reference"]
        require(proposed["actor"] == reference["actor"], "Paired actor hashes differ")
        require(proposed["departed"] == reference["departed"], "Paired passenger counts differ")
        difference = {metric: proposed[metric] - reference[metric] for metric in METRICS}
        difference.update({"block": block, "seed": seed, "demand": demand})
        differences.append(difference)

    mean_differences = averages(differences)
    compare_metrics(mean_differences, verified["means"], "overall difference")
    endpoint_means = {}
    for role, json_label in JSON_LABEL_BY_ROLE.items():
        endpoint_means[role] = averages([group[role] for group in paired.values()])
        compare_metrics(endpoint_means[role], verified["endpoint_means"][json_label], role)

    block_differences = check_group_summaries(
        differences, "block", verified["block_differences"], "block"
    )
    check_group_summaries(differences, "demand", verified["demand_differences"], "demand")
    check_group_summaries(differences, "seed", verified["seed_differences"], "actor seed")
    compare_metrics(averages(list(block_differences.values())), mean_differences, "block average")

    primary_wins = sum(row["primary"] < 0 for row in block_differences.values())
    completion_wins = sum(row["completion"] > -0.003 for row in block_differences.values())
    primary_p = sign_probability(primary_wins, len(block_differences))
    completion_p = sign_probability(completion_wins, len(block_differences))
    require(primary_wins == verified["primary_wins"], "Primary sign-test win count differs")
    require(completion_wins == verified["completion_NI_wins"], "Completion sign-test wins differ")
    close(primary_p, verified["one_sided_p"], "primary sign probability")
    require(primary_p < 0.025 and completion_p < 0.025, "Registered joint criterion does not pass")
    close(
        -100 * mean_differences["primary"] / endpoint_means["reference"]["primary"],
        verified["primary_percent_reduction"], "relative generalized-time reduction",
    )
    close(
        -100 * mean_differences["unfinished"] / endpoint_means["reference"]["unfinished"],
        verified["unfinished_percent_reduction"], "relative unfinished-passenger reduction",
    )
    require(verified["raw_files_read"] == len(records), "Recorded endpoint count differs")
    require(verified["paired_cells"] == len(paired), "Recorded pair count differs")

    print("PASS: 300 endpoint records, 150 matched pairs, 10 independent blocks.")
    print("PASS: fixed actor hashes, matched passenger counts, 24,000-s horizons, and 2W+V accounting.")
    print("PASS: overall, endpoint, demand, block, and actor-seed summaries reproduce verified_results.json.")
    print("Paired means (proposed minus matched reference): Y={:.8f} s/pax; completion={:+.8f} pp; unfinished={:+.2f}.".format(
        mean_differences["primary"], 100 * mean_differences["completion"], mean_differences["unfinished"]
    ))
    print("Exact one-sided sign tests: primary {}/10, p={:.10f}; completion NI {}/10, p={:.10f}.".format(
        primary_wins, primary_p, completion_wins, completion_p
    ))
    print("The matched reference retains its narrower-window 20-s rule. No files or simulations were changed.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        print("VERIFICATION FAILED: " + str(error), file=sys.stderr)
        sys.exit(1)
