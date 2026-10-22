import datetime
import os
import pprint
import re
import sys
import time
import uuid
from copy import deepcopy
from sre_parse import FLAGS

import absl.app
import absl.flags
import d4rl
import gym
# import robel
import numpy as np
import torch
import wandb
import ipdb
from tqdm import trange

from envs import get_new_density_env, get_new_friction_env, get_new_gravity_env, get_new_thigh_range_env, get_new_foot_shape_env, get_new_foot_stiffness_env, get_new_thigh_size_env, get_new_ellipsoid_limb_env, get_new_box_limb_env, get_new_head_size_env, get_new_torso_length_env, get_new_limb_stiffness_env, get_new_tendon_elasticity_env
from mixed_replay_buffer import MixedReplayBuffer
from model import FullyConnectedQFunction, FullyConnectedNetwork, SamplerPolicy, TanhGaussianPolicy
from sampler import StepSampler, TrajSampler
from h2oplus import H2OPLUS
from utils import (Timer, WandBLogger, define_flags_with_default,
                get_user_flags, prefix_metrics, print_flags,
                set_random_seed)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Network.Dynamics_net import Dynamics
from Network.Weight_net import ConcatDiscriminator, ConcatRatioEstimator
from viskit.logging import logger, setup_logger

nowTime = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')


# ─────────────────────────────────────────────────────────────────────────
# Checkpoint helpers (crash-resilient resume)
# Saves only the SIM portion of the replay buffer because the OFFLINE
# portion is deterministic for a given (seed, data_source, residual_ratio)
# and is re-materialized by MixedReplayBuffer on construction.
# Atomic save: write to .tmp then os.replace.
# ─────────────────────────────────────────────────────────────────────────
def _save_ckpt(path, h2o, replay_buffer, train_sampler, viskit_metrics, epoch):
    ckpt = {
        'epoch': epoch,
        # ── H2OPLUS networks (weights) ──
        'policy': h2o.policy.state_dict(),
        'qf1': h2o.qf1.state_dict(),
        'qf2': h2o.qf2.state_dict(),
        'target_qf1': h2o.target_qf1.state_dict(),
        'target_qf2': h2o.target_qf2.state_dict(),
        'vf': h2o.vf.state_dict(),
        # ── Optimizers ──
        'policy_optimizer': h2o.policy_optimizer.state_dict(),
        'qf_optimizer': h2o.qf_optimizer.state_dict(),
        'vf_optimizer': h2o.vf_optimizer.state_dict(),
        # ── Step counter (drives target-net update period & pretrain gate) ──
        'total_steps': int(h2o._total_steps),
        # ── Mid-episode sim sampler state ──
        'sampler_current_obs': train_sampler._current_observation,
        'sampler_traj_steps': int(train_sampler._traj_steps),
        # ── Logged metrics (for continuity of viskit tabular) ──
        'viskit_metrics': dict(viskit_metrics),
        # ── RNG ──
        'np_rng': np.random.get_state(),
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    if getattr(h2o, 'log_alpha', None) is not None:
        ckpt['log_alpha'] = h2o.log_alpha.state_dict()
        ckpt['alpha_optimizer'] = h2o.alpha_optimizer.state_dict()
    if getattr(h2o, 'd_sa', None) is not None:
        ckpt['d_sa'] = h2o.d_sa.state_dict()
        ckpt['d_sas'] = h2o.d_sas.state_dict()
        ckpt['d_sa_optimizer'] = h2o.d_sa_optimizer.state_dict()
        ckpt['d_sas_optimizer'] = h2o.d_sas_optimizer.state_dict()
    if getattr(h2o, 'dynamics_ratio_estimator', None) is not None:
        ckpt['dynamics_ratio_estimator'] = h2o.dynamics_ratio_estimator.state_dict()
        ckpt['dynamics_ratio_estimator_optimizer'] = h2o.dynamics_ratio_estimator_optimizer.state_dict()

    fd = replay_buffer.fixed_dataset_size
    ptr = replay_buffer.ptr
    size = replay_buffer.size
    # The sim region may have wrapped; save the whole slice [fd:max_size)
    # up to `size - fd` valid entries. Simpler and robust: dump entire region
    # because sim region is small (~50k rows × ~80 bytes ≈ 4 MB).
    sim_end = max(ptr, fd + max(0, size - fd))
    ckpt['buffer'] = {
        'ptr': ptr, 'size': size, 'fixed_dataset_size': fd,
        'state': replay_buffer.state[fd:sim_end].copy(),
        'action': replay_buffer.action[fd:sim_end].copy(),
        'reward': replay_buffer.reward[fd:sim_end].copy(),
        'next_state': replay_buffer.next_state[fd:sim_end].copy(),
        'done': replay_buffer.done[fd:sim_end].copy(),
    }

    tmp = path + '.tmp'
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def _load_ckpt(path, h2o, replay_buffer, train_sampler, viskit_metrics, device):
    ckpt = torch.load(path, map_location=device)

    # Networks
    h2o.policy.load_state_dict(ckpt['policy'])
    h2o.qf1.load_state_dict(ckpt['qf1'])
    h2o.qf2.load_state_dict(ckpt['qf2'])
    h2o.target_qf1.load_state_dict(ckpt['target_qf1'])
    h2o.target_qf2.load_state_dict(ckpt['target_qf2'])
    h2o.vf.load_state_dict(ckpt['vf'])
    # Optimizers
    h2o.policy_optimizer.load_state_dict(ckpt['policy_optimizer'])
    h2o.qf_optimizer.load_state_dict(ckpt['qf_optimizer'])
    h2o.vf_optimizer.load_state_dict(ckpt['vf_optimizer'])
    # Step counter (controls target update period & pretrain gate)
    h2o._total_steps = int(ckpt.get('total_steps', 0))
    # Sim sampler mid-episode state
    if 'sampler_current_obs' in ckpt:
        train_sampler._current_observation = ckpt['sampler_current_obs']
        train_sampler._traj_steps = int(ckpt['sampler_traj_steps'])
    # Viskit metrics (restored in-place)
    if 'viskit_metrics' in ckpt:
        viskit_metrics.update(ckpt['viskit_metrics'])

    if 'log_alpha' in ckpt and getattr(h2o, 'log_alpha', None) is not None:
        h2o.log_alpha.load_state_dict(ckpt['log_alpha'])
        h2o.alpha_optimizer.load_state_dict(ckpt['alpha_optimizer'])
    if 'd_sa' in ckpt and getattr(h2o, 'd_sa', None) is not None:
        h2o.d_sa.load_state_dict(ckpt['d_sa'])
        h2o.d_sas.load_state_dict(ckpt['d_sas'])
        h2o.d_sa_optimizer.load_state_dict(ckpt['d_sa_optimizer'])
        h2o.d_sas_optimizer.load_state_dict(ckpt['d_sas_optimizer'])
    if 'dynamics_ratio_estimator' in ckpt and getattr(h2o, 'dynamics_ratio_estimator', None) is not None:
        h2o.dynamics_ratio_estimator.load_state_dict(ckpt['dynamics_ratio_estimator'])
        h2o.dynamics_ratio_estimator_optimizer.load_state_dict(ckpt['dynamics_ratio_estimator_optimizer'])

    buf = ckpt['buffer']
    fd_saved = buf['fixed_dataset_size']
    if fd_saved != replay_buffer.fixed_dataset_size:
        raise ValueError(
            f"fixed_dataset_size mismatch on ckpt load: saved={fd_saved} "
            f"current={replay_buffer.fixed_dataset_size}. Cannot resume — "
            f"check seed/data_source/residual_ratio matches."
        )
    fd = replay_buffer.fixed_dataset_size
    n = buf['state'].shape[0]
    if n > 0:
        replay_buffer.state[fd:fd + n] = buf['state']
        replay_buffer.action[fd:fd + n] = buf['action']
        replay_buffer.reward[fd:fd + n] = buf['reward']
        replay_buffer.next_state[fd:fd + n] = buf['next_state']
        replay_buffer.done[fd:fd + n] = buf['done']
    replay_buffer.ptr = buf['ptr']
    replay_buffer.size = buf['size']

    np.random.set_state(ckpt['np_rng'])
    torch.set_rng_state(ckpt['torch_rng'])
    if ckpt.get('cuda_rng') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt['cuda_rng'])

    return ckpt['epoch']

FLAGS_DEF = define_flags_with_default(
    current_time=nowTime,
    name_str='',
    env_list='HalfCheetah-v2',
    data_source='medium_replay',
    unreal_dynamics="gravity",
    variety_list="2.0",
    replaybuffer_ratio=10.,
    real_residual_ratio=1.,
    tanh_scale=2,
    dis_dropout=False,
    warmup_steps=0,
    max_traj_length=1000,
    seed=42,
    device='cuda',
    save_model=False,
    # Crash-resilient checkpoint: resumes this run from ckpt_dir/ckpt.pt if
    # present. Empty string disables.
    ckpt_dir='',
    ckpt_every=5,
    batch_size=256,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=1.0,
    joint_noise_std=0.0,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    # dynamics model
    dynamics_model=False,
    model_train_epoch=10000,
    model_lr=3e-4,
    model_dropout=False,

    # train and evaluate policy
    n_epochs=1000,
    bc_epochs=0,
    n_rollout_steps_per_epoch=1000,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,

    h2o=H2OPLUS.get_default_config(),
    logging=WandBLogger.get_default_config()
)


def main(argv):
    FLAGS = absl.flags.FLAGS

    # define logged variables for wandb
    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    wandb.run.name = f"{FLAGS.name_str}_{FLAGS.env_list}_{FLAGS.data_source}_{FLAGS.unreal_dynamics}x{FLAGS.variety_list}_seed={FLAGS.seed}_learnedDynamics={FLAGS.dynamics_model}_{FLAGS.current_time}"

    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=FLAGS.seed,
        base_log_dir=FLAGS.logging.output_dir,
        include_exp_prefix_sub_dir=False
    )

    set_random_seed(FLAGS.seed)

    # different unreal dynamics properties: gravity; density; friction
    for unreal_dynamics in FLAGS.unreal_dynamics.split(";"):
        # different environment: Walker2d-v2, Hopper-v2, HalfCheetah-v2
        for env_name in FLAGS.env_list.split(";"):
            # different varieties: 0.5, 1.5, 2.0, ...
            for variety_degree in FLAGS.variety_list.split(";"):
                variety_degree = float(variety_degree)

                if env_name in ["DKittyWalkFixed-v0", "DKittyWalkRandom-v0", "DKittyWalkRandomDynamics-v0"]:
                    real_env = gym.make(env_name) # DKittyWalkFixed-v0, DKittyWalkRandom-v0, DKittyWalkRandomDynamics-v0
                    sim_env = gym.make("DKittyWalkRandomDynamics-v0")
                else:
                    if env_name == "Humanoid-v2":
                        off_env_name = env_name
                    else:
                        off_env_name = "{}-{}-v2".format(env_name.split("-")[0].lower(), FLAGS.data_source).replace('_',"-")
                    if unreal_dynamics == "gravity":
                        real_env = get_new_gravity_env(1, off_env_name)
                        sim_env = get_new_gravity_env(variety_degree, off_env_name)
                    elif unreal_dynamics == "density":
                        real_env = get_new_density_env(1, off_env_name)
                        sim_env = get_new_density_env(variety_degree, off_env_name)
                    elif unreal_dynamics == "friction":
                        real_env = get_new_friction_env(1, off_env_name)
                        sim_env = get_new_friction_env(variety_degree, off_env_name)
                    elif unreal_dynamics == "broken_thigh":
                        real_env = get_new_thigh_range_env(1, off_env_name)
                        sim_env = get_new_thigh_range_env(variety_degree, off_env_name)
                    elif unreal_dynamics == "ellipsoid_foot":
                        real_env = get_new_gravity_env(1, off_env_name)
                        sim_env =  get_new_foot_shape_env(off_env_name)
                    elif unreal_dynamics == "soft_foot":
                        real_env = get_new_foot_stiffness_env(1, off_env_name)
                        sim_env = get_new_foot_stiffness_env(variety_degree, off_env_name)
                    elif unreal_dynamics == "soft_limb":
                        real_env = get_new_limb_stiffness_env(1, off_env_name)
                        sim_env = get_new_limb_stiffness_env(variety_degree, off_env_name)
                    elif unreal_dynamics == "elastic_tendon":
                        real_env = get_new_tendon_elasticity_env(1, off_env_name)
                        sim_env = get_new_tendon_elasticity_env(variety_degree, off_env_name)
                    elif unreal_dynamics == "thigh_size":
                        real_env = get_new_thigh_size_env(1, off_env_name)
                        sim_env = get_new_thigh_size_env(variety_degree, off_env_name)
                    elif unreal_dynamics == "ellipsoid_limb":
                        real_env = get_new_gravity_env(1, off_env_name)
                        sim_env =  get_new_ellipsoid_limb_env(off_env_name)
                    elif unreal_dynamics == "box_limb":
                        real_env = get_new_gravity_env(1, off_env_name)
                        sim_env =  get_new_box_limb_env(off_env_name)
                    elif unreal_dynamics == "head_size":
                        real_env = get_new_head_size_env(1, off_env_name)
                        sim_env = get_new_head_size_env(variety_degree, off_env_name)
                    elif unreal_dynamics == "torso_length":
                        real_env = get_new_torso_length_env(1, off_env_name)
                        sim_env = get_new_torso_length_env(variety_degree, off_env_name)
                    else:
                        raise RuntimeError("Got erroneous unreal dynamics %s" % unreal_dynamics)
                    
                print("\n-------------Env name: {}, variety: {}, unreal_dynamics: {}-------------".format(env_name, variety_degree, unreal_dynamics))

    # a step sampler for "simulated" training
    train_sampler = StepSampler(sim_env.unwrapped, FLAGS.max_traj_length)
    # a trajectory sampler for "real-world" evaluation
    eval_sampler = TrajSampler(real_env.unwrapped, FLAGS.max_traj_length)

    # replay buffer
    num_state = real_env.observation_space.shape[0]
    num_action = real_env.action_space.shape[0]
    replay_buffer = MixedReplayBuffer(FLAGS.reward_scale, FLAGS.reward_bias, FLAGS.clip_action, num_state, num_action, task=env_name.split("-")[0].lower(), data_source=FLAGS.data_source, device=FLAGS.device, buffer_ratio=FLAGS.replaybuffer_ratio, residual_ratio=FLAGS.real_residual_ratio, max_episode_steps=real_env._max_episode_steps)
    # ipdb.set_trace()

    # Should a dynamics model be learned for s' sampling when estimating u(s,a)?
    if FLAGS.dynamics_model:
        # initialize dynamics model
        dynamics_model = Dynamics(num_state, num_action, 256, dropout=FLAGS.model_dropout, device=FLAGS.device).to(FLAGS.device)
        model_optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=FLAGS.model_lr)
        for n in trange(FLAGS.model_train_epoch):
            real_obs, real_action, real_next_obs = replay_buffer.sample(FLAGS.batch_size, scope="real", type="sas").values()
            minus_logp_pi = dynamics_model.get_loss(real_obs, real_action, real_next_obs - real_obs)
            model_optimizer.zero_grad()
            minus_logp_pi.backward()
            model_optimizer.step()
            if n % 100 == 0:
                metrics = {}
                metrics['model_loss'] = minus_logp_pi.cpu().detach().numpy().item()
                wandb_logger.log(metrics)
        xi_sas = ConcatRatioEstimator(2 * num_state + num_action, 256, 1, FLAGS.device, scale=FLAGS.tanh_scale, dropout=FLAGS.dis_dropout).float().to(FLAGS.device) 
    else:
        dynamics_model = None


    # discirminators
    d_sa = ConcatDiscriminator(num_state + num_action, 256, 2, FLAGS.device, dropout=FLAGS.dis_dropout).float().to(FLAGS.device)
    d_sas = ConcatDiscriminator(2 * num_state + num_action, 256, 2, FLAGS.device, dropout=FLAGS.dis_dropout).float().to(FLAGS.device) 

    # agent
    policy = TanhGaussianPolicy(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.policy_arch,
        log_std_multiplier=FLAGS.policy_log_std_multiplier,
        log_std_offset=FLAGS.policy_log_std_offset,
        orthogonal_init=FLAGS.orthogonal_init,
    )

    qf1 = FullyConnectedQFunction(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf1 = deepcopy(qf1)

    qf2 = FullyConnectedQFunction(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf2 = deepcopy(qf2)
    
    vf = FullyConnectedNetwork(
        eval_sampler.env.observation_space.shape[0], 1,
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )

    if FLAGS.h2o.target_entropy >= 0.0:
        FLAGS.h2o.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    if FLAGS.dynamics_model:
        h2o = H2OPLUS(FLAGS.h2o, policy, qf1, qf2, target_qf1, target_qf2, vf, replay_buffer, dynamics_model=dynamics_model, dynamics_ratio_estimator=xi_sas)
    else:
        h2o = H2OPLUS(FLAGS.h2o, policy, qf1, qf2, target_qf1, target_qf2, vf, replay_buffer, d_sa=d_sa, d_sas=d_sas, dynamics_model=dynamics_model)
    h2o.torch_to_device(FLAGS.device)

    # sampling policy is always the current policy: \pi
    sampler_policy = SamplerPolicy(policy, FLAGS.device)

    viskit_metrics = {}

    # ── Checkpoint resume detection ────────────────────────────────────
    start_epoch = 0
    ckpt_path = None
    if FLAGS.ckpt_dir:
        os.makedirs(FLAGS.ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(FLAGS.ckpt_dir, 'ckpt.pt')
        if os.path.exists(ckpt_path):
            print(f"[ckpt] resuming from {ckpt_path}", flush=True)
            saved_epoch = _load_ckpt(ckpt_path, h2o, replay_buffer,
                                      train_sampler, viskit_metrics, FLAGS.device)
            start_epoch = saved_epoch + 1
            print(f"[ckpt] resumed — next epoch = {start_epoch}/{FLAGS.n_epochs} "
                  f"(total_steps={h2o._total_steps})", flush=True)
        else:
            print(f"[ckpt] fresh start, will save every {FLAGS.ckpt_every} epochs to {ckpt_path}", flush=True)

    # When n_rollout_steps_per_epoch=0 (pure offline-only mode), warm up the
    # sim scope of the replay buffer once with random-policy rollouts so that
    # batch sampling from the sim scope doesn't crash. The offline phase
    # otherwise uses only the D4RL "real" scope.  Skip on resume — buffer is
    # already restored from checkpoint.
    if FLAGS.n_rollout_steps_per_epoch == 0 and start_epoch == 0:
        print("[offline-only] warmup: 1024 sim random-policy steps for sim-scope buffer", flush=True)
        train_sampler.sample(
            sampler_policy, 1024,
            deterministic=False, replay_buffer=replay_buffer,
            joint_noise_std=FLAGS.joint_noise_std,
        )

    # train and evaluate for n_epochs
    for epoch in trange(start_epoch, FLAGS.n_epochs, initial=start_epoch, total=FLAGS.n_epochs):
        metrics = {}

        # TODO rollout from the simulator
        with Timer() as rollout_timer:
            # rollout and append simulated trajectories to the replay buffer
            train_sampler.sample(
                sampler_policy, FLAGS.n_rollout_steps_per_epoch,
                deterministic=False, replay_buffer=replay_buffer, joint_noise_std=FLAGS.joint_noise_std
            )
            metrics['epoch'] = epoch

        # TODO Train from the mixed data
        with Timer() as train_timer:
            for batch_idx in trange(FLAGS.n_train_step_per_epoch):
                # batch = subsample_batch(dataset, FLAGS.batch_size)
                # batch = batch_to_torch(batch, FLAGS.device)
                # metrics.update(prefix_metrics(h2o.train(batch, bc=epoch < FLAGS.bc_epochs), 'h2o'))

                # real_batch_size = int(FLAGS.batch_size * (1 - FLAGS.batch_sim_ratio))
                # sim_batch_size = int(FLAGS.batch_size * FLAGS.batch_sim_ratio)
                # real_batch = replay_buffer.sample(FLAGS.batch_size * (1 - FLAGS.batch_sim_ratio), scope="real")
                # sim_batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.batch_sim_ratio, scope="sim")
                # batch = [real_batch, sim_batch]
                if batch_idx + 1 == FLAGS.n_train_step_per_epoch:
                    metrics.update(
                        prefix_metrics(h2o.train(FLAGS.batch_size, FLAGS.warmup_steps), 'h2o')
                    )
                else:
                    h2o.train(FLAGS.batch_size, FLAGS.warmup_steps)

        # TODO Evaluate in the real world
        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                trajs = eval_sampler.sample(
                    sampler_policy, FLAGS.eval_n_trajs, deterministic=True
                )
                if not FLAGS.dynamics_model:
                    eval_dsa_loss, eval_dsas_loss = h2o.discriminator_evaluate()
                    metrics['eval_dsa_loss'] = eval_dsa_loss
                    metrics['eval_dsas_loss'] = eval_dsas_loss
                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                # metrics['average_normalizd_return'] = np.mean(
                #     [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
                # )
                
                if FLAGS.save_model:
                    save_data = {'h2o': h2o, 'variant': variant, 'epoch': epoch}
                    wandb_logger.save_pickle(save_data, 'model_{}.pkl'.format(epoch))

        metrics['rollout_time'] = rollout_timer()
        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = rollout_timer() + train_timer() + eval_timer()
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

        # ── Periodic checkpoint save ───────────────────────────────────
        if ckpt_path is not None and (epoch + 1) % FLAGS.ckpt_every == 0:
            _save_ckpt(ckpt_path, h2o, replay_buffer, train_sampler, viskit_metrics, epoch)
            print(f"[ckpt] saved at epoch {epoch + 1} "
                  f"(total_steps={h2o._total_steps})", flush=True)

    # Guard against the resume-already-finished case (start_epoch==n_epochs,
    # loop body never ran, so `epoch` is undefined).
    final_epoch = epoch if 'epoch' in dir() else start_epoch - 1

    if FLAGS.save_model:
        save_data = {'h2o': h2o, 'variant': variant, 'epoch': final_epoch}
        wandb_logger.save_pickle(save_data, 'model.pkl')

    # Training finished normally — rename ckpt.pt → ckpt.pt.done so a
    # re-launch with the same ckpt_dir starts fresh instead of resuming
    # a completed run.
    if ckpt_path is not None and os.path.exists(ckpt_path):
        done_path = ckpt_path + '.done'
        os.replace(ckpt_path, done_path)
        print(f"[ckpt] run complete — ckpt moved to {done_path}", flush=True)

if __name__ == '__main__':
    absl.app.run(main)
