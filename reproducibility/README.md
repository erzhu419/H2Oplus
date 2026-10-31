# Reproducibility records

This directory separates immutable experiment evidence from the live training
workspace.  A manifest records the exact Git state, runtime, input hashes and
HDF5 schema used by an experiment.  It does not copy or modify the underlying
datasets.

The publication includes versioned analyses and closeouts together with small
existing `.log`, CSV, JSON, and JSONL records in their original directories.
Look under `h2oplus_v*_results/` for development-round evidence and under
`h2oplus_v20_formal_confirmation_results/` for the separately authorized final
confirmation. Earlier runtime and training records also remain under
`formal_results_v7_v7r1/` and the repository's `experiment_output*` directories.
The 2026-10-31 additions are limited to 2 MiB per text artifact; large raw traces,
replay, checkpoints, deployment copies, and caches are not published. Existing
failure/incident labels and scientific records are preserved unchanged.

The pre-rebuild paper results are marked `legacy-invalid`: they remain useful
for diagnosis, but must not be reused as submission evidence because the v2
replay loader applied the large-headway reward penalty twice.

Create a manifest before launching a run:

```bash
python reproducibility/create_manifest.py \
  --output reproducibility/run_manifest.json \
  --label transit-smdp-v3-smoke \
  --input bus_h2o/datasets_v3/transit_smdp_v3.h5 \
  --input experiment_output/offline_ensemble/offline_ensemble_final.pt
```

Large inputs are hashed by default.  Use `--metadata-only` only for exploratory
runs; submission runs require complete SHA-256 hashes.

## Canonical SUMO evaluation

`run_evaluation_matrix.py` launches one isolated libsumo process per scenario
cell.  With `--workers 0` (the default), it budgets 1.5 GiB per process and uses
at most eight physical cores.  Every job signature binds the policy checkpoint,
canonical dataset, SUMO seed and version, demand scenario, all referenced SUMO
inputs, and all evaluation source files.  An existing result is skipped only
when that complete signature matches; interrupted or stale matrices resume by
rerunning only invalid cells.

```bash
python reproducibility/run_evaluation_matrix.py \
  --checkpoint h2oplus-seed42=PATH/TO/checkpoint.pt \
  --include-zero-hold \
  --sumo-seeds 20260811 20260812 20260813 20260814 20260815 \
  --scenario-manifests \
    bus_h2o/evaluation_scenarios/demand_0p75_seed20260803/manifest.json \
    bus_h2o/evaluation_scenarios/demand_1_seed20260803/manifest.json \
    bus_h2o/evaluation_scenarios/demand_1p25_seed20260803/manifest.json \
  --save-event-logs \
  --output-root experiment_output_v4/evaluation
```

Do not pass `--max-events` for submission runs; it deliberately creates a
partial smoke-test episode.  On the reference Ryzen 9 7945HX system, one full
24,000-second demand-1.0 episode took 431 seconds and 0.88 GiB peak resident
memory.  A four-job smoke matrix improved from 41.3 seconds serial to 14.6
seconds with four workers, while all scientific fields remained bit-identical.
SUMO evaluation is CPU-bound; using a GPU does not accelerate it.

For policy-selection diagnostics, prefer `--max-sim-time-seconds 6000` over
`--max-events`: it gives every policy the same simulated exposure horizon, so
passenger and service metrics remain comparable. Both options are diagnostic
only and are mutually exclusive; omit both for submission runs.

## Formal v7 evaluation partitions

The v7 evaluator is SUMO/CPU-bound. Run policy inference on `cpu`; launching
many batch-one policy processes on one CUDA device adds process contention and
does not accelerate SUMO. A formal matrix can be split deterministically across
hosts without changing a job signature:

```bash
python reproducibility/run_evaluation_matrix_v4.py \
  --mdp-version v7 \
  --methods-manifest PATH/evaluation_methods_v7_formal.json \
  --environment-design reproducibility/evaluation_environment_design_v7_formal_v1.json \
  --dataset bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5 \
  --output-root experiment_output_v7/formal_evaluation \
  --device cpu \
  --workers 11 \
  --shard-count 2 \
  --shard-index 0
```

Run the same command with `--shard-index 1` on the second host. Assignment is
round-robin over sorted, content-derived job keys; absolute host paths do not
affect the assignment, and partition sizes differ by at most one job. Each
partition writes a distinct `matrix_manifest.shard-NNNN-of-NNNN.json` and may
write into the same logical output root.

`--workers 0` selects the minimum of the partition size, physical cores minus
one, and available RAM divided by 1.5 GiB. Under `scheduleurm`, pass an explicit
worker count equal to the task's declared CPU allocation because host-level
auto-detection cannot see the scheduler's per-task CPU quota. The current two
Xeon Silver 4310 servers expose 12 physical cores each, so 11 workers is the
standalone upper bound; lower it while other CPU work is reserved.

After every shard has been synchronized to the canonical local output root,
merge and validate them:

```bash
python reproducibility/merge_evaluation_matrix_v7.py \
  --manifest \
    experiment_output_v7/formal_evaluation/matrix_manifest.shard-0000-of-0002.json \
    experiment_output_v7/formal_evaluation/matrix_manifest.shard-0001-of-0002.json \
  --output-root experiment_output_v7/formal_evaluation
```

The merger refuses missing or duplicated shards, wrong deterministic
assignments, duplicate method/environment cells, unequal environment
coverage, source/runtime drift, and incomplete result artifacts. It can also
combine separately partitioned local-H2O+ and server-baseline method manifests,
provided both execution classes used the same dataset, SUMO version, evaluation
design, and exact evaluator source state. Because source-hash keys are absolute
paths, all hosts must run the evaluator from the same canonical deployment path.
The resulting `matrix_manifest.json` is the only manifest passed to
`analyze_evaluation_matrix_v4.py`.
