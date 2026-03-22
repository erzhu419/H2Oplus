"""
train_sim.py
============
SAC training on MultiLineSimEnv (H2Oplus/bus_h2o).

Architecture mirrors sac_ensemble_SUMO_linear_penalty.py:
- EmbeddingLayer for categorical obs features
- Twin-Q ensemble Critic (VectorizedCritic)
- SAC policy with tanh squashing [0, 60s holding]
- Per-episode reward / Q-value logging for convergence check

Obs layout (15-dim, sim_core/bus.py _prepare_for_action):
  [0]  line_idx        (cat)
  [1]  fleet_id        (cat, physical bus; FIFO-stable across trips)
  [2]  station_id      (cat)
  [3]  time_period     (cat, hour of day)
  [4]  direction       (cat, 0/1)
  [5]  fwd_headway     (cont, s)
  [6]  bwd_headway     (cont, s)
  [7]  waiting_pax     (cont)
  [8]  target_hw=360   (cont, constant)
  [9]  base_stop_dur   (cont, s)
  [10] sim_time        (cont, s)
  [11] gap             (cont, target-fwd_hw)
  [12] co_fwd_hw       (cont)
  [13] co_bwd_hw       (cont)
  [14] seg_speed       (cont, m/s)

Action: scalar ∈ [−∞, ∞] → tanh → [0, 60] (holding time, seconds)

Run:
    cd /home/erzhu419/mine_code/sumo-rl/H2Oplus/bus_h2o
    python train_sim.py --max_episodes 40
"""

import os, sys, time, math, copy, argparse, random, csv
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ────────────────────── Path Setup ──────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ────────────────────── CLI Args ─────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--hidden_dim', type=int, default=48)
parser.add_argument('--ensemble_size', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.80)
parser.add_argument('--soft_tau', type=float, default=1e-2)
parser.add_argument('--weight_reg', type=float, default=0.01)
parser.add_argument('--beta_ood', type=float, default=0.01)
parser.add_argument('--beta_bc', type=float, default=0.001)
parser.add_argument('--beta', type=float, default=-2.0)
parser.add_argument('--maximum_alpha', type=float, default=0.6)
parser.add_argument('--training_freq', type=int, default=10, help='Train every N decisions')
parser.add_argument('--critic_actor_ratio', type=int, default=2)
parser.add_argument('--use_state_norm', action='store_true', default=False)
parser.add_argument('--use_reward_scaling', action='store_true', default=False)
parser.add_argument('--run_name', type=str, default='sim_run')
parser.add_argument('--env_config', type=str, default='calibrated_env')
args = parser.parse_args()

# ────────────────────── Device ──────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ────────────────────── Save Dirs ───────────────────────────────────
SAVE_ROOT = SCRIPT_DIR
RUN = args.run_name
PIC_DIR   = os.path.join(SAVE_ROOT, 'train_outputs', RUN, 'pic')
LOG_DIR   = os.path.join(SAVE_ROOT, 'train_outputs', RUN, 'logs')
MODEL_DIR = os.path.join(SAVE_ROOT, 'train_outputs', RUN, 'model')
for d in (PIC_DIR, LOG_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

# ────────────────────── Env ─────────────────────────────────────────
from envs.bus_sim_env import MultiLineSimEnv
env = MultiLineSimEnv(args.env_config)

# ────────────────────── Feature Spec (mirrors SUMO version) ─────────
# 15-dim obs: indices 0-4 are categorical, 5-14 are continuous
CAT_COLS     = ['line_id', 'fleet_id', 'station_id', 'time_period', 'direction']
N_CAT        = len(CAT_COLS)       # 5
N_CONT       = 15 - N_CAT          # 10
ACTION_DIM   = 1

# Cardinalities (generous upper bounds, clamped in forward)
# line_id maps to index 0..N_lines-1
N_LINES      = len(env.line_map)
MAX_BUSES    = max(le.max_agent_num for le in env.line_map.values())
MAX_STATIONS = max(len(le.stations) for le in env.line_map.values())
MAX_HOUR     = 24

CAT_CODE_DICT = {
    'line_id':     {i: i for i in range(N_LINES + 2)},
    'fleet_id':    {i: i for i in range(MAX_BUSES + 2)},  # physical bus idx
    'station_id':  {i: i for i in range(MAX_STATIONS + 2)},
    'time_period': {i: i for i in range(MAX_HOUR + 2)},
    'direction':   {0: 0, 1: 1},
}
# Map line_id string → int index
LINE_ID_MAP = {lid: i for i, lid in enumerate(sorted(env.line_map.keys()))}
print(f"Lines: {LINE_ID_MAP}")
print(f"N_LINES={N_LINES}, MAX_BUSES={MAX_BUSES}, MAX_STATIONS={MAX_STATIONS}")

# action mapping: tanh[-1,1] -> holt[0,60]
ACTION_SCALE = 30.0
ACTION_BIAS  = 30.0
LOG_SCALE    = math.log(ACTION_SCALE)

# ────────────────────── Normalisation helpers ────────────────────────
class RunningStats:
    def __init__(self, dim):
        self.n   = 0
        self.mu  = np.zeros(dim, dtype=np.float64)
        self.var = np.ones( dim, dtype=np.float64)

    def update(self, x):
        self.n += 1
        delta = x - self.mu
        self.mu  += delta / self.n
        self.var += (x - self.mu) * delta * 0  # not Welford, just simple EMA below

    def normalize(self, x):
        return (x - self.mu) / (np.sqrt(self.var) + 1e-8)


class RewardScaler:
    def __init__(self, gamma):
        self.R   = 0.0
        self.mu  = 0.0
        self.var = 1.0
        self.n   = 0
        self.gamma = gamma

    def scale(self, r):
        self.R = self.gamma * self.R + r
        self.n += 1
        old_mu = self.mu
        self.mu += (self.R - self.mu) / self.n
        self.var += (self.R - old_mu) * (self.R - self.mu)
        std = math.sqrt(self.var / (self.n + 1e-8)) + 1e-8
        return r / std


# ────────────────────── Replay Buffer ───────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=5_000_000):
        self.buf  = []
        self.pos  = 0
        self.cap  = int(capacity)

    def push(self, s, a, r, s2, done):
        if len(self.buf) < self.cap:
            self.buf.append(None)
        self.buf[self.pos] = (s, a, r, s2, done)
        self.pos = (self.pos + 1) % self.cap

    def sample(self, n):
        idx = np.random.randint(0, len(self.buf), n)
        batch = [self.buf[i] for i in idx]
        s, a, r, s2, d = zip(*batch)
        return (np.stack(s), np.stack(a),
                np.array(r, np.float32), np.stack(s2),
                np.array(d, np.float32))

    def __len__(self):
        return len(self.buf)


# ────────────────────── Networks (from ensemble SAC) ─────────────────
def suggest_emb_dim(card):
    if card <= 1: return 1
    return min(32, max(2, int(round(card**0.5)) + 1))


class EmbeddingLayer(nn.Module):
    def __init__(self, cat_code_dict, cat_cols):
        super().__init__()
        self.cat_cols = list(cat_cols)
        self.cards = {}
        mods = {}
        self.dims = {}
        for col in cat_cols:
            card = max(cat_code_dict[col].values()) + 1
            self.cards[col] = card
            dim = suggest_emb_dim(card)
            self.dims[col] = dim
            mods[col] = nn.Embedding(card, dim)
        self.embeddings = nn.ModuleDict(mods)
        self.output_dim = sum(self.dims.values())
        self.ln = nn.LayerNorm(self.output_dim)

    def forward(self, cat):
        if cat.dim() == 1:
            cat = cat.unsqueeze(0)
        parts = []
        for i, col in enumerate(self.cat_cols):
            idx = cat[:, i].long().clamp(0, self.cards[col] - 1)
            parts.append(self.embeddings[col](idx))
        out = torch.cat(parts, dim=1)
        return self.ln(out)


class VectorizedLinear(nn.Module):
    def __init__(self, in_f, out_f, E):
        super().__init__()
        self.w = nn.Parameter(torch.empty(E, in_f, out_f))
        self.b = nn.Parameter(torch.empty(E, 1, out_f))
        for i in range(E):
            nn.init.kaiming_uniform_(self.w[i], a=math.sqrt(5))
        fan, _ = nn.init._calculate_fan_in_and_fan_out(self.w[0])
        bd = 1/math.sqrt(fan) if fan > 0 else 0
        nn.init.uniform_(self.b, -bd, bd)

    def forward(self, x):
        return x @ self.w + self.b


class EnsembleCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden, E, emb):
        super().__init__()
        self.emb = emb
        self.E   = E
        self.net = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden, E),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            VectorizedLinear(hidden, hidden, E),
            nn.ReLU(),
            VectorizedLinear(hidden, hidden, E),
            nn.ReLU(),
            VectorizedLinear(hidden, 1, E),
        )

    def forward(self, state, action):
        cat = state[:, :N_CAT]
        cont = state[:, N_CAT:]
        emb  = self.emb(cat)
        x = torch.cat([emb, cont, action], dim=1)
        x = x.unsqueeze(0).repeat_interleave(self.E, dim=0)
        return self.net(x).squeeze(-1)   # (E, B)


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden, emb):
        super().__init__()
        self.emb = emb
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden, ACTION_DIM)
        self.logstd_head  = nn.Linear(hidden, ACTION_DIM)
        nn.init.uniform_(self.mean_head.weight,   -3e-3, 3e-3)
        nn.init.uniform_(self.logstd_head.weight, -3e-3, 3e-3)

    def _encode(self, state):
        cat  = state[:, :N_CAT]
        cont = state[:, N_CAT:]
        emb  = self.emb(cat)
        return self.net(torch.cat([emb, cont], dim=1))

    def forward(self, state):
        h = self._encode(state)
        return self.mean_head(h), self.logstd_head(h).clamp(-20, 2)

    def evaluate(self, state, eps=1e-6):
        mean, logstd = self.forward(state)
        std  = logstd.exp()
        z    = Normal(0, 1).sample(mean.shape).to(device)
        a0   = torch.tanh(mean + std * z)
        action = ACTION_SCALE * a0 + ACTION_BIAS
        logp   = (Normal(mean, std).log_prob(mean + std * z)
                  - torch.log(1 - a0.pow(2) + eps)
                  - math.log(ACTION_SCALE))
        return action, logp.sum(1)

    @torch.no_grad()
    def get_action(self, state_vec, deterministic=False):
        s = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
        mean, logstd = self.forward(s)
        if deterministic:
            a0 = torch.tanh(mean)
        else:
            z  = Normal(0, 1).sample(mean.shape).to(device)
            a0 = torch.tanh(mean + logstd.exp() * z)
        return (ACTION_SCALE * a0 + ACTION_BIAS).cpu().numpy()[0]


# ────────────────────── SAC Trainer ─────────────────────────────────
class SACTrainer:
    def __init__(self, buf, hid, E):
        emb_template = EmbeddingLayer(CAT_CODE_DICT, CAT_COLS)
        emb_dim = emb_template.output_dim
        state_dim = emb_dim + N_CONT    # embedded cat + continuous

        # networks
        self.q    = EnsembleCritic(state_dim, ACTION_DIM, hid, E,
                                   copy.deepcopy(emb_template)).to(device)
        self.q_t  = copy.deepcopy(self.q)
        self.pi   = PolicyNet(state_dim, hid,
                              copy.deepcopy(emb_template)).to(device)
        for p in self.q_t.parameters():
            p.requires_grad_(False)

        # alpha (entropy temp)
        self.target_entropy = -float(ACTION_DIM) + LOG_SCALE
        self.log_alpha = torch.tensor([math.log(0.1)], dtype=torch.float32,
                                      requires_grad=True, device=device)
        self.alpha = 0.1

        # optimisers
        self.opt_q  = optim.Adam(self.q.parameters(),  lr=3e-4)
        self.opt_pi = optim.Adam(self.pi.parameters(), lr=3e-4)
        self.opt_al = optim.Adam([self.log_alpha],      lr=3e-4)

        self.buf = buf
        self.E   = E

    def update(self, step):
        s, a, r, s2, d = self.buf.sample(args.batch_size)
        S  = torch.FloatTensor(s).to(device)
        A  = torch.FloatTensor(a).to(device)
        R  = torch.FloatTensor(r).unsqueeze(1).to(device)
        S2 = torch.FloatTensor(s2).to(device)
        D  = torch.FloatTensor(d).unsqueeze(1).to(device)

        # normalise reward per batch
        R  = 10.0 * (R - R.mean()) / (R.std() + 1e-6)

        with torch.no_grad():
            a2, lp2 = self.pi.evaluate(S2)
            q_next   = self.q_t(S2, a2)     # (E, B)
            lp2 = lp2.unsqueeze(0).expand(self.E, -1)
            q_target = R.T + (1 - D.T) * args.gamma * (q_next - self.alpha * lp2)

        q_pred = self.q(S, A)               # (E, B)
        q_loss = F.mse_loss(q_pred, q_target.detach().unsqueeze(-1)
                            .expand_as(q_pred.unsqueeze(-1)).squeeze(-1))
        ood_penalty = q_pred.std(0).mean()
        q_loss = q_loss + args.beta_ood * ood_penalty

        self.opt_q.zero_grad(); q_loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt_q.step()

        # actor update (every critic_actor_ratio critic steps)
        if step % args.critic_actor_ratio == 0:
            a_new, lp_new = self.pi.evaluate(S)
            q_new = self.q(S, a_new)         # (E, B)
            pi_loss = -(q_new.mean(0) + args.beta * q_new.std(0) - self.alpha * lp_new).mean()
            bc_loss = F.mse_loss(a_new, A.detach())
            pi_loss = pi_loss + args.beta_bc * bc_loss

            self.opt_pi.zero_grad(); pi_loss.backward()
            self.opt_pi.step()

        # alpha update
        _, lp = self.pi.evaluate(S)
        al_loss = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
        self.opt_al.zero_grad(); al_loss.backward()
        self.opt_al.step()
        self.alpha = min(args.maximum_alpha, self.log_alpha.exp().item())

        # Polyak update
        with torch.no_grad():
            for tp, p in zip(self.q_t.parameters(), self.q.parameters()):
                tp.data.mul_(1 - args.soft_tau).add_(p.data * args.soft_tau)

        return q_pred.mean().item(), ood_penalty.item()

    def save(self, path):
        torch.save({'q': self.q.state_dict(),
                    'pi': self.pi.state_dict(),
                    'log_alpha': self.log_alpha.detach()}, path)


# ────────────────────── Obs processing ──────────────────────────────
def obs_to_vec(obs_list, line_id_int):
    """Convert raw 15-dim obs list + line_id_int → numpy array."""
    v = list(obs_list)
    # obs[0] is already line_idx from sim; overwrite with our canonical int
    v[0] = float(line_id_int)
    return np.array(v, dtype=np.float32)


# ────────────────────── Training loop ───────────────────────────────
def main():
    buf     = ReplayBuffer()
    trainer = SACTrainer(buf, args.hidden_dim, args.ensemble_size)
    reward_scalers = {lid: RewardScaler(args.gamma) for lid in env.line_map}

    # Logging
    ep_rewards = []
    ep_q_vals  = []
    step_total = 0
    train_step = 0

    csv_path = os.path.join(LOG_DIR, 'train_log.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['episode', 'ep_reward', 'ep_q_mean', 'buf_size',
                                'ep_wall_sec', 'ep_decisions'])

    print(f"\nStarting training: {args.max_episodes} episodes, batch={args.batch_size}\n")

    for eps in range(args.max_episodes):
        t_ep_start = time.time()

        # ── Reset env and get initial state ─────────────────────────
        # reset() clears bus_all=[] and reinitialises everything.
        # initialize_state() advances sim until at least one bus fires obs.
        # The returned state_dict is the current per-bus accumulated obs list.
        env.reset()
        state_dict, reward_dict_init, _ = env.initialize_state()

        # Build action_dict from the state structure
        action_dict = {lid: {k: None for k in range(le.max_agent_num)}
                       for lid, le in env.line_map.items()}

        done  = False
        ep_reward = 0.0
        ep_decisions = 0
        ep_q_this   = []

        # pending: (line_id, bus_id) -> (state_vec, action_arr)
        pending = {}

        # ── Seed actions from initial state ──────────────────────────
        for lid, buses in state_dict.items():
            lid_int = LINE_ID_MAP.get(lid, 0)
            for bus_id, obs_list in buses.items():
                if not obs_list:
                    continue
                obs_raw = obs_list[-1]
                sv      = obs_to_vec(obs_raw, lid_int)
                a_np    = trainer.pi.get_action(sv)
                a_val   = float(a_np[0]) if hasattr(a_np, '__len__') else float(a_np)
                action_dict[lid][bus_id] = a_val
                pending[(lid, bus_id)] = (sv, np.array([a_val], dtype=np.float32))

        while not done:
            # ── step env (event-driven: skip idle ticks) ─────────────
            cur_state, reward_dict, done = env.step_to_event(action_dict)
            step_total += 1

            # ── reset actions to None; rebuild from new obs ────────────
            for lid in action_dict:
                for k in action_dict[lid]:
                    action_dict[lid][k] = None

            # ── process each line / bus ────────────────────────────────
            for lid, buses in cur_state.items():
                lid_int = LINE_ID_MAP.get(lid, 0)
                for bus_id, obs_list in buses.items():
                    if not obs_list:
                        continue
                    obs_raw = obs_list[-1]
                    sv_new  = obs_to_vec(obs_raw, lid_int)
                    r_raw   = float(reward_dict.get(lid, {}).get(bus_id, 0.0))

                    key = (lid, bus_id)

                    # ── settle pending transition ──────────────────────
                    if key in pending:
                        sv_old, a_old = pending[key]
                        # station_id (obs[2]) must differ → valid transition
                        if int(sv_old[2]) != int(sv_new[2]):
                            # Real transition: s changed station
                            pending.pop(key)
                            r = (reward_scalers[lid].scale(r_raw)
                                 if args.use_reward_scaling else r_raw)
                            buf.push(sv_old, a_old, r, sv_new, 0.0)
                            ep_reward    += r_raw
                            ep_decisions += 1

                            # ── select new action for the NEW state ───
                            a_np = trainer.pi.get_action(sv_new)
                            a_val = float(a_np[0]) if hasattr(a_np, '__len__') else float(a_np)
                            action_dict[lid][bus_id] = a_val
                            pending[key] = (sv_new, np.array([a_val], dtype=np.float32))
                        # else: same station re-fire (WAITING_ACTION tick) →
                        #   do NOT overwrite pending, do NOT select new action.
                        #   The action from the decision tick is already in
                        #   action_dict and will be consumed by the sim.
                    else:
                        # First time seeing this bus → select action
                        a_np = trainer.pi.get_action(sv_new)
                        a_val = float(a_np[0]) if hasattr(a_np, '__len__') else float(a_np)
                        action_dict[lid][bus_id] = a_val
                        pending[key] = (sv_new, np.array([a_val], dtype=np.float32))

            # ── training update ────────────────────────────────────────
            # Train based on DECISION count (not env ticks) to match SUMO SAC.
            # warmup: wait until buffer has at least batch_size samples.
            if (ep_decisions > 0
                    and ep_decisions % args.training_freq == 0
                    and len(buf) >= args.batch_size
                    and step_total >= 2000):           # warmup: ~half first episode
                q_mean, ood = trainer.update(train_step)
                ep_q_this.append(q_mean)
                train_step += 1

        # ── episode done ──────────────────────────────────────────
        wall_sec = time.time() - t_ep_start
        avg_q    = float(np.mean(ep_q_this)) if ep_q_this else 0.0
        ep_rewards.append(ep_reward)
        ep_q_vals.append(avg_q)

        print(f"Ep {eps+1:3d}/{args.max_episodes}: reward={ep_reward:8.1f}  "
              f"Q={avg_q:8.2f}  decisions={ep_decisions}  "
              f"buf={len(buf):,}  wall={wall_sec:.1f}s")

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([eps+1, ep_reward, avg_q, len(buf),
                                    wall_sec, ep_decisions])

        # ── plot every 5 episodes ──────────────────────────────────
        if (eps + 1) % 5 == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
            ax1.plot(ep_rewards, 'b-o', ms=3)
            ax1.set_title('Episode Reward'); ax1.set_xlabel('Episode')
            ax2.plot(ep_q_vals,  'r-o', ms=3)
            ax2.set_title('Mean Q-Value per Episode'); ax2.set_xlabel('Episode')
            plt.tight_layout()
            plt.savefig(os.path.join(PIC_DIR, f'train_ep{eps+1}.png'), dpi=100)
            plt.close()

        # ── save checkpoint every 10 episodes ────────────────────
        if (eps + 1) % 10 == 0:
            trainer.save(os.path.join(MODEL_DIR, f'checkpoint_ep{eps+1}.pt'))

    trainer.save(os.path.join(MODEL_DIR, 'final.pt'))
    print(f"\nDone. Logs: {LOG_DIR}  Model: {MODEL_DIR}")
    if ep_rewards:
        print(f"Best reward: {max(ep_rewards):.1f} at ep {ep_rewards.index(max(ep_rewards))+1}")


if __name__ == '__main__':
    main()
