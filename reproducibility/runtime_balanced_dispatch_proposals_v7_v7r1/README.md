# Runtime-balanced dispatch proposals

Status: analysis-only proposals. They are not registered, deployed, queued, or
valid replacements for the frozen v7r1 dispatches.

The completed baseline dispatch exposed a static lane imbalance. Observed
sequential launcher times total about 85.6 GPU-hours across four physical GPUs.
The original critical lane on `jtl110gpu` was about 26.3 hours, while its second
lane finished after about 17.6 hours.

The proposals preserve the signed plan, all job names and signatures, and the
one-process-per-physical-GPU contract. They only change the execution partition.

| Proposal | Original estimate | Balanced estimate | Reduction |
| --- | ---: | ---: | ---: |
| Baselines, host-preserving | 26.40 h | 22.12 h | 16.2% |
| Baselines, cross-host | 26.40 h | 21.49 h | 18.6% |
| H2O+ ablation, cross-host | 3.98 h | 3.98 h | 0.0% |
| H2O+ sensitivity, cross-host | 9.32 h | 7.86 h | 15.6% |

The ablation plan is already balanced by its 3/3/2/2 fast/slow-host split. The
20-job sensitivity plan benefits from assigning six jobs to each faster-host
GPU and four jobs to each slower-host GPU.

Epochs within one online RL run are not independent shards: replay state,
optimizer state, simulator state, and registered event counts are sequential.
Shard only independent method/seed/configuration jobs unless a separately
validated distributed trainer is introduced.
