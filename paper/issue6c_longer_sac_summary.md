# Issue 6c — Pure-online SAC at 5x budget (1000 epochs)

## Reviewer challenge
Round-2 reviewer: *"sim-core is cheap; the pure-online SAC failure
(-1654 +/- 329 K reward at 200 epochs, one diverged seed) may be a budget
artifact, not a fundamental result. Re-run at much longer horizon."*

## Hypothesis
Pure-online SAC in sim-core is **fundamentally** capped (not budget-bound)
because (i) the bus-control reward is dense but action effect on travel time
is delayed and noisy, (ii) without an offline-pretrained critic the early
random policy collects very low-value transitions and the ensemble Q
under-estimates collapses to the no-intervention baseline. If true, 5x more
epochs should plateau near the same level (no monotonic improvement past ~200).
If false (budget-bound), we should see eventual recovery toward a non-trivial
return by 1000 epochs.

## Exact command (per seed)

Conda env `LSTM-RL`, sim-core only, fresh ensemble Q init, no offline mixing,
no IS reweighting, no Cal-QL floor, no KL penalty, no contrastive/dynamics
discriminator.

```
WANDB_MODE=disabled \
/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python -u \
  /home/erzhu419/mine_code/sumo-rl/H2Oplus/SimpleSAC/h2o+_bus_main.py \
  --device=cuda --use_ensemble_q --ensemble_size=5 --ensemble_ckpt='' \
  --save_model=True --batch_size=2048 --n_train_step_per_epoch=100 \
  --n_rollout_events_per_epoch=100 --eval_period=10 --eval_n_trajs=1 \
  --warmup_episodes=5 --checkpoint_period=50 --buffer_ratio=1.5 \
  --pretrain_steps=0 --nouse_snapshot_reset --nouse_jtt \
  --nouse_sumo_online --disable_is_weighting --nouse_cal_ql \
  --nouse_contrastive_disc --nouse_dynamics_disc \
  --kl_coeff=0 --warmup_collect_epochs=0 \
  --n_epochs=1000 --seed=${SEED} \
  --current_time=pure_online_1000ep_s${SEED} \
  --name_str=pure_online_1000ep_s${SEED}
```

Differences vs H2O+ baseline (`run_full_experiments_H2O+.sh` C1 `sim_baseline`):
- `--ensemble_ckpt=''` (no offline pretrained Q ensemble; fresh init)
- `--n_epochs=1000` (vs 200)
- `--nouse_contrastive_disc --nouse_dynamics_disc` (no discriminator at all)

Output dir per seed: `experiment_output/h2oplus_bus_seed{SEED}_pure_online_1000ep_s{SEED}/`

## Submitted tasks (scheduler)

| task id | seed | signature                                      | node    |
|---------|------|------------------------------------------------|---------|
| t0099   | 42   | `H2Oplus/pure_online_sac_1000ep_seed42`        | local:GPU0 (4060) |
| t0100   | 123  | `H2Oplus/pure_online_sac_1000ep_seed123`       | local:GPU0 (4060) |
| t0101   | 789  | `H2Oplus/pure_online_sac_1000ep_seed789`       | local:GPU0 (4060) |

Resource declaration: 600 MB VRAM, 3000 MB RAM, 2 CPU per process. All three packed
onto local 4060 (4060 was idle: 0/8188 MB used at dispatch time).

## Expected wall time

200 epochs took ~18 min for the prior pure-online seed789 run. Linear extrapolation:
1000 epochs ~ **90 min/seed**. Three seeds run concurrently on the same GPU; expect
**~2-2.5 h wall** with mild slowdown from contention. Set a budget of 3 h before
checking.

## Eval after training

Use `SimpleSAC/eval_with_metrics.py` against held-out OD seeds, e.g.

```
conda run -n LSTM-RL python SimpleSAC/eval_with_metrics.py \
  --ckpt experiment_output/h2oplus_bus_seed${S}_pure_online_1000ep_s${S}/checkpoint_epoch1000.pt \
  --sumo_seed 7 --od_scale 1.0
```

Compare 200-epoch checkpoint reward (already in paper) vs 1000-epoch checkpoint
reward, plus full learning curve from `train_steps.csv`.

## Drop-in paper paragraph (placeholder)

> **Is the pure-online failure budget-bound?** A reviewer asked whether the
> reported pure-online SAC failure (-1654 +/- 329 K at 200 sim-core epochs) is
> simply a function of training budget, since sim-core rollouts are cheap. We
> re-ran the same configuration (no offline pretrain, no importance-sampling
> reweighting, no offline-Q floor, sim-core rollout, fresh ensemble Q
> initialisation) at **5x the budget (1000 epochs)** on three seeds (42, 123,
> 789). The pure-online return at 1000 epochs is \tbd{R_1000_mean} +/-
> \tbd{R_1000_std}, compared to \tbd{R_200_mean} +/- \tbd{R_200_std} at 200
> epochs (relative change \tbd{delta_pct}\%). \tbd{seeds_diverged} of the three
> seeds diverged. The learning curve (Fig. \tbd{fig_pure_online_curve}) shows
> \tbd{plateau_or_recovery_description}, consistent with the hypothesis that
> the failure is **\tbd{fundamental_or_budgetary}** rather than a budget
> artifact. H2O+ at the original 200-epoch budget still beats pure-online SAC
> at 1000 epochs by \tbd{h2o_vs_sac_delta}.

## Reconstruction notes

- The "pure online" checkpoint dir in `experiment_output/h2oplus_bus_seed789_26-05-01-00-02-28/`
  carries `name_str=pure_online_sim_s789` in `config.json` even though the dir prefix is
  `h2oplus_bus_seed{N}_<timestamp>` — the script always names dirs by seed+timestamp; the
  `--name_str` flag controls only the wandb run name and config tag.
- We pass `--current_time=pure_online_1000ep_s${SEED}` to make the output dir predictable
  (overrides the default `MMDD_HHMMSS` timestamp), so the scheduler `--ckpt-dir` resume
  detection points at a stable path.
- The legacy `run_full_experiments_H2O+.sh` C1 "sim_baseline" config still passes
  `--ensemble_ckpt $OFFLINE_CKPT`, i.e. it is *not* a true pure-online baseline; it is
  online SAC with an offline-pretrained ensemble. The ablation reported here uses
  `--ensemble_ckpt=''` (empty string) for a fully fresh-init online run, matching the
  prior 200-epoch pure-online experiment in `experiment_output/pure_online_0430_2300/`.
