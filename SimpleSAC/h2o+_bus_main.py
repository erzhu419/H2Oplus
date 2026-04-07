"""
h2o+_bus_main.py
================
Phase 3: Main training script for H2O+ Bus Sim-to-Real.

Usage:
    cd H2Oplus/SimpleSAC
    python h2o+_bus_main.py \\
        --seed 42 \\
        --n_epochs 500 \\
        --device cpu

Loads offline SUMO data from bus_h2o/datasets/*.h5, runs online rollout
on the BusSimEnv (sim_core), and trains H2O+ with z-only discriminator.
"""

import csv
import datetime
import json
import logging
import os
import sys
import pprint
import time

import numpy as np
import torch
import absl.app
import absl.flags
from copy import deepcopy
from tqdm import trange

# ── Path setup ────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_H2O_ROOT = os.path.dirname(_HERE)
_BUS_H2O = os.path.join(_H2O_ROOT, "bus_h2o")
sys.path.insert(0, _HERE)
sys.path.insert(0, _BUS_H2O)

from bus_replay_buffer import BusMixedReplayBuffer
from bus_sampler import BusStepSampler, BusEvalSampler
from h2oplus_bus import H2OPlusBus
from model import (
    EmbeddingLayer,
    BusEmbeddingPolicy,
    BusEmbeddingQFunction,
    BusEmbeddingVFunction,
    BusSamplerPolicy,
)
from utils import (
    Timer,
    WandBLogger,
    define_flags_with_default,
    get_user_flags,
    prefix_metrics,
    set_random_seed,
)
from common.data_utils import ZOnlyDiscriminator, TransitionDiscriminator, set_route_length, build_edge_linear_map
from priority_index import PriorityIndex
from snapshot_store import SnapshotStore

# ── Logging setup ─────────────────────────────────────────────────────
try:
    from viskit.logging import logger, setup_logger
except ImportError:
    logger = None
    setup_logger = None

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


nowTime = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")

FLAGS_DEF = define_flags_with_default(
    current_time=nowTime,
    name_str="h2oplus_bus",
    seed=42,
    device="cpu",
    save_model=True,
    batch_size=2048,

    # ── Data ──────────────────────────────────────────────────────
    dataset_file=os.path.join(_BUS_H2O, "datasets_v2", "merged_all_v2.h5"),
    dataset_dir="",
    dataset_glob="sumo_*.h5",
    sim_env_path=os.path.join(_BUS_H2O, "calibrated_env"),
    edge_xml=os.path.join(_BUS_H2O, "network_data", "a_sorted_busline_edge.xml"),
    line_id="7X",
    use_sumo_online=False,  # Use SUMO instead of SIM for online rollout (debug: zero dynamics gap)
    sumo_dir=os.path.normpath(os.path.join(_BUS_H2O, os.pardir, os.pardir, "SUMO_ruiguang", "online_control")),
    sumo_max_steps=18000,

    # ── Architecture (matches legacy SAC checkpoint) ──────────────
    obs_dim=17,         # 15 obs + 2 last_action
    action_dim=2,       # hold_time + speed_ratio
    hidden_size=48,     # legacy SAC hidden dim
    action_range=1.0,  # policy outputs raw tanh [-1,1]; env mapping done externally

    # ── Buffer ────────────────────────────────────────────────────
    buffer_ratio=2.0,   # reduced from 5.0 to prevent OOM with ensemble+SUMO
    reward_scale=1.0,
    reward_bias=0.0,

    # ── Rollout ───────────────────────────────────────────────────
    max_traj_events=100,
    p_reset=0.5,
    h_rollout=30,
    w_threshold=0.05,
    warmup_episodes=50,
    use_snapshot_reset=True,      # snapshot-based resets enabled (lazy load)
    use_jtt=False,                # JTT disabled for baseline run

    # ── Training loop ─────────────────────────────────────────────
    n_epochs=500,
    pretrain_steps=5000,             # pure offline pretraining gradient steps
    n_rollout_events_per_epoch=100,  # original value
    n_train_step_per_epoch=500,      # original value (~120s/epoch)
    adaptive_train_steps=False,      # disabled for baseline
    eval_period=10,
    eval_n_trajs=2,
    checkpoint_period=50,            # save model every N epochs

    # ── H2O+ ──────────────────────────────────────────────────────
    use_ensemble_q=False,   # Use ensemble Q (RE-SAC style) instead of twin-Q + V
    ensemble_size=5,        # Number of Q-networks in ensemble
    ensemble_ckpt="",       # Path to ensemble offline RL checkpoint for initialization
    adaptive_sim_ratio=False, # Auto-scale sim_ratio by buffer sizes (prevents oversampling)
    disable_is_weighting=False, # Disable discriminator IS weighting (for zero-gap debugging)
    use_cal_ql=False,          # Cal-QL: prevent sim Q-targets from dropping below offline baseline
    use_dynamics_disc=False,  # Use DynamicsDiscriminator instead of TransitionDiscriminator
    dynamics_disc_temp=1.0,   # Temperature for dynamics-based IS weights
    use_contrastive_disc=False,  # Use ContrastiveDynamicsDiscriminator (best for SUMO/SIM)
    h2o=H2OPlusBus.get_default_config(),
    logging=WandBLogger.get_default_config(),

    # ── JTT: Targeted Snapshot Reset ──────────────────────────────
    jtt_warmup_epochs=50,         # Phase 1 → Phase 2 switch point (K)
    jtt_alpha=0.5,                # TD error weight in priority score
    jtt_beta=0.3,                 # Q-disagreement weight
    jtt_gamma=0.2,                # Discriminator drift weight
    jtt_temperature_max=2.0,      # Reset sampling breadth (start of Phase 2)
    jtt_temperature_min=0.3,      # Reset sampling tightness (end of training)
    jtt_ema_decay=0.1,            # Priority score EMA smoothing
)


# ======================================================================
#  CSV step logger: writes every train/pretrain step to a CSV file
# ======================================================================

class StepLogger:
    """Append-mode CSV logger that records every gradient step."""

    # Columns we always write, in order. Extra keys go at the end.
    _BASE_COLS = [
        "phase", "epoch", "step", "global_step",
        "training_phase", "policy_loss",
        "real_qf1_loss", "real_qf2_loss",
        "sim_qf1_loss", "sim_qf2_loss",
        "vf_loss", "vf_pred", "vf_error",
        "disc_loss", "alpha", "alpha_loss",
        "log_pi", "sqrt_IS_ratio",
        "mean_real_rewards", "mean_sim_rewards",
        "exploit_coeff",
        "jtt_updated", "jtt_td_error_mean",
        "jtt_q_disagree_mean", "jtt_disc_drift_mean",
    ]

    def __init__(self, csv_path: str):
        self._path = csv_path
        self._file = open(csv_path, "w", newline="")
        self._writer = csv.DictWriter(
            self._file, fieldnames=self._BASE_COLS, extrasaction="ignore",
        )
        self._writer.writeheader()
        self._file.flush()
        self._global_step = 0

    def log_step(self, phase: str, epoch: int, step: int, metrics: dict):
        self._global_step += 1
        row = dict(metrics)  # copy
        row["phase"] = phase
        row["epoch"] = epoch
        row["step"] = step
        row["global_step"] = self._global_step
        self._writer.writerow(row)
        # Flush every 100 steps for performance (not every step)
        if self._global_step % 100 == 0:
            self._file.flush()

    def flush(self):
        self._file.flush()

    def close(self):
        self._file.flush()
        self._file.close()

    @property
    def global_step(self):
        return self._global_step


# ======================================================================
#  Setup file-based logging (Python logging module)
# ======================================================================

def setup_file_logging(log_dir: str, name: str = "h2oplus"):
    """Create a logger that writes to both console and file."""
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    # File handler — DEBUG and above
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    ))
    log.addHandler(fh)

    # Console handler — INFO and above (less verbose)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(ch)

    return log


def main(argv):
    FLAGS = absl.flags.FLAGS
    variant = get_user_flags(FLAGS, FLAGS_DEF)

    set_random_seed(FLAGS.seed)

    # ── Create log directory ──────────────────────────────────────
    log_dir = os.path.join(
        _H2O_ROOT, "experiment_output",
        f"h2oplus_bus_seed{FLAGS.seed}_{FLAGS.current_time}",
    )
    os.makedirs(log_dir, exist_ok=True)

    log = setup_file_logging(log_dir)
    step_csv = StepLogger(os.path.join(log_dir, "train_steps.csv"))

    log.info("=" * 60)
    log.info("H2O+ Bus Training")
    log.info("=" * 60)
    log.info(f"Log directory: {log_dir}")
    log.info(f"Seed: {FLAGS.seed}, Device: {FLAGS.device}")
    log.info(f"Snapshot reset: {FLAGS.use_snapshot_reset}, JTT: {FLAGS.use_jtt}")

    # Save config
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(variant, f, indent=2, default=str)
    log.info(f"Config saved to {log_dir}/config.json")

    # ── Setup wandb ───────────────────────────────────────────────
    if HAS_WANDB:
        wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
        wandb.run.name = f"h2oplus_bus_seed={FLAGS.seed}_{FLAGS.current_time}"
    else:
        wandb_logger = None

    if setup_logger is not None:
        setup_logger(
            variant=variant,
            exp_id=wandb_logger.experiment_id if wandb_logger else "local",
            seed=FLAGS.seed,
            base_log_dir=FLAGS.logging.output_dir if hasattr(FLAGS.logging, 'output_dir') else "/tmp/h2oplus_bus",
            include_exp_prefix_sub_dir=False,
        )

    # ── Set route length (critical for z feature extraction) ──────
    if os.path.exists(FLAGS.edge_xml):
        edge_map = build_edge_linear_map(FLAGS.edge_xml, FLAGS.line_id)
        route_length = max(edge_map.values()) if edge_map else 13119.0
    else:
        route_length = 13119.0
    set_route_length(route_length)
    log.info(f"Route length set to {route_length:.1f} m")

    # ── Load replay buffer ────────────────────────────────────────
    ds_file = FLAGS.dataset_file if FLAGS.dataset_file else None
    ds_dir = FLAGS.dataset_dir if FLAGS.dataset_dir else None
    log.info(f"Loading offline data: file={ds_file}, dir={ds_dir}")
    replay_buffer = BusMixedReplayBuffer(
        state_dim=FLAGS.obs_dim,
        action_dim=FLAGS.action_dim,
        context_dim=30,
        dataset_file=ds_file,
        dataset_dir=ds_dir,
        dataset_glob=FLAGS.dataset_glob,
        device=FLAGS.device,
        buffer_ratio=FLAGS.buffer_ratio,
        reward_scale=FLAGS.reward_scale,
        reward_bias=FLAGS.reward_bias,
        action_scale=1.0,   # raw tanh [-1,1]: no rescaling needed
        action_bias=0.0,
    )
    log.info(
        f"Loaded {replay_buffer.fixed_dataset_size:,} offline transitions, "
        f"max buffer size {replay_buffer.max_size:,}, "
        f"valid snapshots: {len(replay_buffer._valid_snap_indices):,}"
    )
    r_mean, r_std = replay_buffer.get_reward_stats()
    log.info(f"Offline reward stats: mean={r_mean:.2f}, std={r_std:.2f}")

    # ── Lazy SnapshotStore (if merged file has snap index) ────────
    if FLAGS.use_snapshot_reset and getattr(replay_buffer, '_has_lazy_snap_index', False):
        manifest_path = os.path.join(
            os.path.dirname(FLAGS.dataset_file), "file_manifest.json"
        )
        if os.path.exists(manifest_path):
            with open(manifest_path) as mf:
                file_manifest = json.load(mf)  # [(filename, n_rows), ...]
            snap_store = SnapshotStore(
                archive_dir=os.path.dirname(FLAGS.dataset_file),
                file_manifest=file_manifest,
                cache_size=256,
                snapshot_key="snapshot_T1",  # SIM format (has current_time, trip_id etc.)
            )
            replay_buffer.set_snapshot_store(snap_store)
            log.info(
                f"SnapshotStore created: "
                f"{len(file_manifest)} archive files, LRU cache=256"
            )
        else:
            log.warning(
                f"file_manifest.json not found at {manifest_path}. "
                f"Snapshots will not be available."
            )
    elif not FLAGS.use_snapshot_reset:
        log.info("Snapshot reset DISABLED (use_snapshot_reset=False)")

    # ── Create online env ────────────────────────────────────────
    if FLAGS.use_sumo_online:
        log.info(f"Creating SumoGymEnv (SUMO online rollout, zero dynamics gap)")
        log.info(f"  sumo_dir={FLAGS.sumo_dir}, max_steps={FLAGS.sumo_max_steps}")
        from envs.sumo_gym_env import SumoGymEnv
        sim_env = SumoGymEnv(
            sumo_dir=FLAGS.sumo_dir,
            edge_xml=FLAGS.edge_xml,
            max_steps=FLAGS.sumo_max_steps,
            line_id=FLAGS.line_id,
        )
        log.info("Online env: SUMO (via SumoRLBridge)")
    else:
        log.info(f"Creating MultiLineSimEnv from {FLAGS.sim_env_path}")
        from envs.bus_sim_env import MultiLineSimEnv
        sim_env = MultiLineSimEnv(path=FLAGS.sim_env_path, debug=False)

        from sim_core.bus import LINE_SIGNAL_CONFIG
        log.info("Traffic signal model (per line):")
        for lid, cfg in sorted(LINE_SIGNAL_CONFIG.items()):
            log.info(
                f"  {lid}: {cfg['n_signals']} signals, "
                f"green_frac={cfg['green_frac']:.2f}, cycle={cfg['cycle']}s"
            )
        log.info("Online env: SIM (MultiLineSimEnv)")

    # ── Build EmbeddingLayer (matches legacy SAC architecture) ────
    # Categorical features: line_id, bus_id, station_id, time_period, direction
    # Cardinalities come from the offline data ranges
    cat_cols = ['line_id', 'bus_id', 'station_id', 'time_period', 'direction']
    cat_code_dict = {
        'line_id':     {i: i for i in range(12)},   # 12 lines (checkpoint)
        'bus_id':      {i: i for i in range(389)},  # 389 buses (checkpoint)
        'station_id':  {i: i for i in range(1)},    # dynamic encoding
        'time_period': {i: i for i in range(1)},    # dynamic encoding
        'direction':   {0: 0, 1: 1},
    }
    embedding_template = EmbeddingLayer(
        cat_code_dict, cat_cols, layer_norm=True, dropout=0.05
    )
    num_cont_features = FLAGS.obs_dim - len(cat_cols)  # 17 - 5 = 12
    state_dim = embedding_template.output_dim + num_cont_features
    log.info(
        f"EmbeddingLayer output_dim={embedding_template.output_dim}, "
        f"cont_features={num_cont_features}, internal_state_dim={state_dim}"
    )

    # ── Create networks ──────────────────────────────────────────
    policy = BusEmbeddingPolicy(
        num_inputs=state_dim,
        num_actions=FLAGS.action_dim,
        hidden_size=FLAGS.hidden_size,
        embedding_layer=embedding_template.clone(),
        action_range=FLAGS.action_range,
    )

    if FLAGS.use_contrastive_disc:
        from common.data_utils import ContrastiveDynamicsDiscriminator
        discriminator = ContrastiveDynamicsDiscriminator(
            obs_dim=FLAGS.obs_dim,
            action_dim=FLAGS.action_dim,
            embed_dim=16,
            hidden_dim=128,
        )
        log.info("Using ContrastiveDynamicsDiscriminator (delta-based, action-invariant)")
    elif FLAGS.use_dynamics_disc:
        from common.data_utils import DynamicsDiscriminator
        discriminator = DynamicsDiscriminator(
            obs_dim=FLAGS.obs_dim,
            action_dim=FLAGS.action_dim,
            hidden_dim=128,
            temperature=FLAGS.dynamics_disc_temp,
            n_cat=len(cat_cols),
        )
        log.info(f"Using DynamicsDiscriminator (temp={FLAGS.dynamics_disc_temp})")
    else:
        discriminator = TransitionDiscriminator(
            obs_dim=FLAGS.obs_dim,
            action_dim=FLAGS.action_dim,
            context_dim=30,
            z_effective_dim=20,
            use_spectral_norm=getattr(FLAGS.h2o, 'disc_spectral_norm', False),
        )

    if FLAGS.h2o.target_entropy >= 0.0:
        FLAGS.h2o.target_entropy = -float(FLAGS.action_dim)

    if FLAGS.use_ensemble_q:
        # ── Ensemble Q (RE-SAC style) ────────────────────────────
        from model import BusEnsembleCritic
        from h2oplus_ensemble import H2OPlusEnsemble

        E = FLAGS.ensemble_size
        qf = BusEnsembleCritic(
            state_dim, FLAGS.action_dim, FLAGS.hidden_size, E,
            embedding_template.clone(),
        )
        target_qf = deepcopy(qf)
        for p in target_qf.parameters():
            p.requires_grad_(False)

        # Optionally load from ensemble offline RL checkpoint
        if FLAGS.ensemble_ckpt and os.path.exists(FLAGS.ensemble_ckpt):
            ens_ckpt = torch.load(FLAGS.ensemble_ckpt, map_location=FLAGS.device, weights_only=True)
            if 'policy' in ens_ckpt:
                # train_offline_ensemble.py checkpoint format
                log.info(f"Loading ensemble offline checkpoint: {FLAGS.ensemble_ckpt}")
                # Policy architecture might differ — try loading, skip on mismatch
                try:
                    policy.load_state_dict(ens_ckpt['policy'], strict=False)
                    log.info("  Policy loaded from ensemble checkpoint (partial match OK)")
                except Exception as e:
                    log.warning(f"  Policy load failed: {e}")
                if 'qf' in ens_ckpt:
                    try:
                        qf.load_state_dict(ens_ckpt['qf'])
                        target_qf.load_state_dict(ens_ckpt['qf'])
                        log.info("  Q-function loaded from ensemble checkpoint")
                    except Exception as e:
                        log.warning(f"  Q-function load failed: {e}")

        h2o_config = dict(FLAGS.h2o)
        h2o_config['ensemble_size'] = E
        h2o_config['adaptive_sim_ratio'] = FLAGS.adaptive_sim_ratio
        h2o_config['disable_is_weighting'] = FLAGS.disable_is_weighting
        h2o_config['use_cal_ql'] = FLAGS.use_cal_ql
        h2o_config['beta_bc'] = 0.0 if FLAGS.disable_is_weighting else h2o_config.get('beta_bc', 0.005)
        h2o = H2OPlusEnsemble(
            h2o_config, policy, qf, target_qf, replay_buffer,
            discriminator=discriminator,
        )
        log.info(f"Using H2OPlusEnsemble: E={E}, β={FLAGS.h2o.get('beta', -2.0)}")
    else:
        # ── Original twin-Q + V ──────────────────────────────────
        qf1 = BusEmbeddingQFunction(
            num_inputs=state_dim, num_actions=FLAGS.action_dim,
            hidden_size=FLAGS.hidden_size, embedding_layer=embedding_template.clone(),
        )
        target_qf1 = deepcopy(qf1)
        qf2 = BusEmbeddingQFunction(
            num_inputs=state_dim, num_actions=FLAGS.action_dim,
            hidden_size=FLAGS.hidden_size, embedding_layer=embedding_template.clone(),
        )
        target_qf2 = deepcopy(qf2)
        vf = BusEmbeddingVFunction(
            num_inputs=state_dim, hidden_size=FLAGS.hidden_size,
            embedding_layer=embedding_template.clone(),
        )
        h2o = H2OPlusBus(
            FLAGS.h2o, policy, qf1, qf2, target_qf1, target_qf2, vf,
            replay_buffer, discriminator=discriminator,
        )
        log.info("Using H2OPlusBus (twin-Q + V)")

    h2o.torch_to_device(FLAGS.device)

    sampler_policy = BusSamplerPolicy(policy, FLAGS.device)

    # ── Create samplers ───────────────────────────────────────────
    train_sampler = BusStepSampler(
        env=sim_env,
        replay_buffer=replay_buffer,
        max_traj_events=FLAGS.max_traj_events,
        p_reset=FLAGS.p_reset if FLAGS.use_snapshot_reset else 0.0,
        h_rollout=FLAGS.h_rollout,
        w_threshold=FLAGS.w_threshold,
        warmup_episodes=FLAGS.warmup_episodes,
    )

    eval_sampler = BusEvalSampler(
        env=sim_env,
        max_traj_events=FLAGS.max_traj_events,
    )

    # ── JTT: Create PriorityIndex and inject (optional) ──────────
    priority_index = None
    if FLAGS.use_jtt:
        priority_index = PriorityIndex(
            n_offline=replay_buffer.fixed_dataset_size,
            alpha=FLAGS.jtt_alpha,
            beta=FLAGS.jtt_beta,
            gamma=FLAGS.jtt_gamma,
            ema_decay=FLAGS.jtt_ema_decay,
        )
        h2o.priority_index = priority_index
        train_sampler.priority_index = priority_index
        log.info(
            f"JTT PriorityIndex created: "
            f"n_offline={replay_buffer.fixed_dataset_size}, "
            f"warmup={FLAGS.jtt_warmup_epochs} epochs, "
            f"T=[{FLAGS.jtt_temperature_max}->{FLAGS.jtt_temperature_min}]"
        )
    else:
        log.info("JTT DISABLED (use_jtt=False)")

    # ==================================================================
    #  Pretrain phase: pure offline Q bootstrapping
    # ==================================================================
    if FLAGS.pretrain_steps > 0:
        log.info(f"Pretraining for {FLAGS.pretrain_steps} steps (offline only)...")
        t0 = time.time()
        for pt_step in trange(FLAGS.pretrain_steps, desc="Pretraining"):
            pt_metrics = h2o.train(FLAGS.batch_size, pretrain_steps=FLAGS.pretrain_steps)
            step_csv.log_step("pretrain", -1, pt_step, pt_metrics)
        step_csv.flush()
        log.info(
            f"Pretrain done in {time.time() - t0:.1f}s: "
            f"policy_loss={pt_metrics.get('policy_loss', 'N/A'):.4f}, "
            f"qf1_loss={pt_metrics.get('real_qf1_loss', 'N/A'):.4f}, "
            f"qf2_loss={pt_metrics.get('real_qf2_loss', 'N/A'):.4f}"
        )

    # ==================================================================
    #  DynamicsDiscriminator pretrain: learn SUMO dynamics P(s'|s,a)
    # ==================================================================
    if FLAGS.use_dynamics_disc and hasattr(discriminator, 'train_step'):
        n_disc_pretrain = 5000
        log.info(f"Pretraining DynamicsDiscriminator for {n_disc_pretrain} steps...")
        disc_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
        t0 = time.time()
        for ds_step in range(n_disc_pretrain):
            batch = replay_buffer.sample(FLAGS.batch_size, scope="real")
            dl = discriminator.train_step(
                batch["observations"], batch["actions"],
                batch["next_observations"], disc_opt,
            )
            if (ds_step + 1) % 1000 == 0:
                log.info(f"  DynDisc pretrain step {ds_step+1}: loss={dl:.4f}")
        log.info(f"DynDisc pretrain done in {time.time()-t0:.1f}s, final loss={dl:.4f}")

    # ==================================================================
    #  Main training loop
    # ==================================================================
    log.info(f"Starting training for {FLAGS.n_epochs} epochs")
    log.info(
        f"  rollout_events/epoch={FLAGS.n_rollout_events_per_epoch}, "
        f"train_steps/epoch={FLAGS.n_train_step_per_epoch}, "
        f"eval_period={FLAGS.eval_period}, "
        f"checkpoint_period={FLAGS.checkpoint_period}"
    )
    viskit_metrics = {}

    for epoch in trange(FLAGS.n_epochs, desc="H2O+ Bus Training"):
        metrics = {"epoch": epoch}

        # ── JTT: Temperature annealing (only if enabled) ──────────
        jtt_temp = 0.0
        if FLAGS.use_jtt:
            if epoch >= FLAGS.jtt_warmup_epochs:
                progress = (epoch - FLAGS.jtt_warmup_epochs) / max(
                    FLAGS.n_epochs - FLAGS.jtt_warmup_epochs, 1
                )
                jtt_temp = FLAGS.jtt_temperature_max * max(
                    FLAGS.jtt_temperature_min / FLAGS.jtt_temperature_max,
                    (1.0 - progress) ** 2,
                )
            else:
                jtt_temp = 1e6
            train_sampler.current_temperature = jtt_temp

        # ── A. Online rollout ─────────────────────────────────────
        with Timer() as rollout_timer:
            disc_active = (epoch > FLAGS.warmup_episodes)
            rollout_stats = train_sampler.sample(
                sampler_policy,
                FLAGS.n_rollout_events_per_epoch,
                deterministic=False,
                discriminator=discriminator if disc_active else None,
            )
            metrics.update(prefix_metrics(rollout_stats, "rollout"))
            metrics["discriminator_active"] = disc_active

        log.debug(
            f"[Epoch {epoch}] Rollout: "
            f"events={rollout_stats.get('n_events', 0)}, "
            f"trans={rollout_stats.get('n_transitions', 0)}, "
            f"trunc={rollout_stats.get('n_truncated', 0)}, "
            f"mean_w={rollout_stats.get('mean_w', 0):.4f}, "
            f"disc={disc_active}, "
            f"resets(jtt/uni/fresh)={rollout_stats.get('jtt_resets',0)}/"
            f"{rollout_stats.get('uniform_resets',0)}/{rollout_stats.get('fresh_resets',0)}"
        )

        # ── B. H2O+ training updates ─────────────────────────────
        train_steps_actual = 0
        online_buf = replay_buffer.online_size

        if FLAGS.adaptive_train_steps and online_buf > 0:
            max_useful_steps = max(1, online_buf // FLAGS.batch_size)
            train_steps_this_epoch = min(FLAGS.n_train_step_per_epoch, max_useful_steps)
        else:
            train_steps_this_epoch = FLAGS.n_train_step_per_epoch

        # batch_sim_ratio: adaptive (by buffer size) or fixed warmup ramp
        if FLAGS.adaptive_sim_ratio:
            # Let h2o.train() compute sim_ratio from actual buffer sizes
            sim_ratio = -1  # sentinel: adaptive handles it internally
        elif epoch < FLAGS.warmup_episodes and FLAGS.warmup_episodes > 0:
            warmup_progress = epoch / FLAGS.warmup_episodes
            sim_ratio = 0.1 + 0.4 * warmup_progress  # 0.1 -> 0.5
        else:
            sim_ratio = 0.5
        if sim_ratio >= 0:
            h2o.config.batch_sim_ratio = sim_ratio
        metrics["batch_sim_ratio"] = sim_ratio if sim_ratio >= 0 else -1

        epoch_train_metrics = []  # accumulate for epoch summary

        with Timer() as train_timer:
            for batch_idx in range(train_steps_this_epoch):
                if not replay_buffer.has_online_data():
                    break  # Need at least some sim data
                train_steps_actual += 1
                train_metrics = h2o.train(
                    FLAGS.batch_size, pretrain_steps=0
                )
                # Log every step to CSV
                step_csv.log_step("train", epoch, batch_idx, train_metrics)
                epoch_train_metrics.append(train_metrics)

        step_csv.flush()

        # Summarize epoch training into metrics dict
        if epoch_train_metrics:
            last_m = epoch_train_metrics[-1]
            metrics.update(prefix_metrics(last_m, "h2o"))

            # Also compute epoch averages for key metrics
            def _avg(key):
                vals = [m.get(key) for m in epoch_train_metrics if m.get(key) is not None]
                return float(np.mean(vals)) if vals else None

            metrics["h2o/epoch_avg_policy_loss"] = _avg("policy_loss")
            metrics["h2o/epoch_avg_real_qf1_loss"] = _avg("real_qf1_loss")
            metrics["h2o/epoch_avg_disc_loss"] = _avg("disc_loss")
            metrics["h2o/epoch_avg_sqrt_IS_ratio"] = _avg("sqrt_IS_ratio")
            metrics["h2o/epoch_avg_vf_loss"] = _avg("vf_loss")

        metrics["train_steps_actual"] = train_steps_actual
        metrics["train_steps_target"] = train_steps_this_epoch
        metrics["train_steps_skipped"] = (train_steps_actual == 0)

        log.debug(
            f"[Epoch {epoch}] Train: "
            f"steps={train_steps_actual}/{train_steps_this_epoch}, "
            f"sim_ratio={sim_ratio:.2f}, "
            f"avg_ploss={metrics.get('h2o/epoch_avg_policy_loss', 'N/A')}, "
            f"avg_qloss={metrics.get('h2o/epoch_avg_real_qf1_loss', 'N/A')}, "
            f"avg_dloss={metrics.get('h2o/epoch_avg_disc_loss', 'N/A')}"
        )

        # ── C. Evaluation ─────────────────────────────────────────
        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                trajs = eval_sampler.sample(
                    sampler_policy, FLAGS.eval_n_trajs, deterministic=True
                )
                avg_return = np.mean(
                    [np.sum(t["rewards"]) for t in trajs if t["rewards"]]
                ) if any(t["rewards"] for t in trajs) else 0.0
                avg_length = np.mean(
                    [len(t["rewards"]) for t in trajs if t["rewards"]]
                ) if any(t["rewards"] for t in trajs) else 0.0
                metrics["average_return"] = avg_return
                metrics["average_traj_length"] = avg_length

                # RE-SAC v4: update policy anchor on improvement
                if hasattr(h2o, 'update_anchor_if_improved'):
                    h2o.update_anchor_if_improved(avg_return)

                # Discriminator evaluation
                disc_real_acc = 0.0
                disc_sim_acc = 0.0
                if replay_buffer.has_online_data():
                    disc_real_acc, disc_sim_acc = h2o.discriminator_evaluate()
                    metrics["disc_real_acc"] = disc_real_acc
                    metrics["disc_sim_acc"] = disc_sim_acc

                log.info(
                    f"[Epoch {epoch+1}/{FLAGS.n_epochs}] "
                    f"avg_return={avg_return:.2f}, "
                    f"avg_len={avg_length:.0f}, "
                    f"online_buf={replay_buffer.online_size}, "
                    f"train={train_steps_actual}/{train_steps_this_epoch}, "
                    f"sim_ratio={sim_ratio:.2f}, "
                    f"disc_active={disc_active}, "
                    f"disc_acc(R/S)={disc_real_acc:.3f}/{disc_sim_acc:.3f}, "
                    f"phase={metrics.get('h2o/training_phase', '?')}, "
                    f"IS_ratio={metrics.get('h2o/sqrt_IS_ratio', 'N/A')}, "
                    f"disc_loss={metrics.get('h2o/disc_loss', 'N/A')}, "
                    f"mean_w={metrics.get('rollout/mean_w', 0):.4f}, "
                    f"trunc={metrics.get('rollout/n_truncated', 0)}"
                )

        # ── JTT: Priority index logging (only if enabled) ─────────
        jtt_stats = None
        if FLAGS.use_jtt and priority_index is not None:
            jtt_stats = priority_index.get_stats()
            metrics["jtt/temperature"] = jtt_temp
            metrics["jtt/coverage"] = jtt_stats["coverage"]
            metrics["jtt/mean_priority"] = jtt_stats["mean_priority"]
            metrics["jtt/std_priority"] = jtt_stats["std_priority"]
            metrics["jtt/max_priority"] = jtt_stats["max_priority"]
            metrics["jtt/error_set_size"] = jtt_stats["error_set_size"]
            metrics["jtt/phase"] = 2 if epoch >= FLAGS.jtt_warmup_epochs else 1
            log.debug(
                f"[Epoch {epoch}] JTT: "
                f"temp={jtt_temp:.2f}, "
                f"priority={jtt_stats['mean_priority']:.4f}, "
                f"coverage={jtt_stats['coverage']:.1%}"
            )

        metrics["rollout_time"] = rollout_timer()
        metrics["train_time"] = train_timer()
        metrics["eval_time"] = eval_timer()
        metrics["epoch_time"] = rollout_timer() + train_timer() + eval_timer()
        metrics["online_buffer_size"] = replay_buffer.online_size

        if wandb_logger is not None:
            wandb_logger.log(metrics)

        viskit_metrics.update(metrics)
        if logger is not None:
            logger.record_dict(viskit_metrics)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)

        # ── Checkpoint: save best + periodic ─────────────────────
        save_best = False
        if "average_return" in metrics:
            if not hasattr(main, '_best_eval_return'):
                main._best_eval_return = float('-inf')
            if metrics["average_return"] > main._best_eval_return:
                main._best_eval_return = metrics["average_return"]
                save_best = True

        if save_best:
            best_path = os.path.join(log_dir, "checkpoint_best.pt")
            if hasattr(h2o, 'save_checkpoint'):
                h2o.save_checkpoint(best_path, epoch, variant)
            else:
                torch.save({"epoch": epoch, "policy_state_dict": policy.state_dict(),
                             "variant": variant}, best_path)
            log.info(f"New best checkpoint (ep{epoch+1}, return={metrics['average_return']:.1f}): {best_path}")

        if (epoch + 1) % FLAGS.checkpoint_period == 0:
            ckpt_path = os.path.join(log_dir, f"checkpoint_epoch{epoch+1}.pt")
            if hasattr(h2o, 'save_checkpoint'):
                h2o.save_checkpoint(ckpt_path, epoch, variant)
            else:
                torch.save(
                    {
                        "epoch": epoch,
                        "policy_state_dict": policy.state_dict(),
                        "qf1_state_dict": qf1.state_dict(),
                        "qf2_state_dict": qf2.state_dict(),
                        "target_qf1_state_dict": target_qf1.state_dict(),
                        "target_qf2_state_dict": target_qf2.state_dict(),
                        "vf_state_dict": vf.state_dict(),
                        "discriminator_state_dict": discriminator.state_dict(),
                        "log_alpha": h2o.log_alpha.state_dict() if h2o.log_alpha else None,
                        "variant": variant,
                    },
                ckpt_path,
            )
            log.info(f"Checkpoint saved: {ckpt_path}")

        # ── Release large epoch-local objects immediately ─────────
        del metrics, epoch_train_metrics
        if jtt_stats is not None:
            del jtt_stats
        if 'trajs' in dir():
            del trajs

    # ==================================================================
    #  Save final model
    # ==================================================================
    if FLAGS.save_model:
        final_path = os.path.join(log_dir, f"model_final.pt")
        if hasattr(h2o, 'save_checkpoint'):
            h2o.save_checkpoint(final_path, FLAGS.n_epochs - 1, variant)
        else:
            torch.save(
                {
                    "epoch": FLAGS.n_epochs - 1,
                    "policy_state_dict": policy.state_dict(),
                    "qf1_state_dict": qf1.state_dict(),
                    "qf2_state_dict": qf2.state_dict(),
                    "target_qf1_state_dict": target_qf1.state_dict(),
                    "target_qf2_state_dict": target_qf2.state_dict(),
                    "vf_state_dict": vf.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "log_alpha": h2o.log_alpha.state_dict() if h2o.log_alpha else None,
                    "variant": variant,
            },
            final_path,
        )
        log.info(f"Final model saved to {final_path}")

    step_csv.close()
    log.info(
        f"Training complete!  "
        f"CSV log: {step_csv._path}  ({step_csv.global_step} steps)"
    )
    log.info(f"All outputs in: {log_dir}")


if __name__ == "__main__":
    absl.app.run(main)
