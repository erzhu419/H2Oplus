"""Quick evaluation of checkpoint_ep39 using H2O+ modified components.

Tests:
1. Load checkpoint into BusEmbeddingPolicy (action_range=1.0)
2. Run BusSimEnv with original action mapping (hold=[0,60])
3. Report cumulative reward
"""
import os, sys, time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_H2O_ROOT = os.path.dirname(_HERE)
_BUS_H2O = os.path.join(_H2O_ROOT, "bus_h2o")
sys.path.insert(0, _HERE)
sys.path.insert(0, _BUS_H2O)

from model import BusEmbeddingPolicy, EmbeddingLayer, BusSamplerPolicy
from bus_sampler import _map_raw_to_env, _extract_active_buses
from envs.bus_sim_env import BusSimEnv
from common.data_utils import set_route_length, build_edge_linear_map

# ── Config ────────────────────────────────────────────────────────
CKPT = os.path.join(_H2O_ROOT, "collect_policy", "checkpoint_episode_39_policy")
NORM = os.path.join(_H2O_ROOT, "collect_policy", "checkpoint_episode_39_norm")
SIM_PATH = os.path.join(_BUS_H2O, "calibrated_env")
EDGE_XML = os.path.join(_BUS_H2O, "network_data", "a_sorted_busline_edge.xml")
DEVICE = "cpu"

# ── Setup ─────────────────────────────────────────────────────────
# Route length
edge_map = build_edge_linear_map(EDGE_XML, "7X")
route_length = max(edge_map.values()) if edge_map else 13119.0
set_route_length(route_length)

# Load normalizer
from normalization import Normalization, RunningMeanStd
norm_data = torch.load(NORM, map_location=DEVICE, weights_only=False)
num_cat = 5
num_num = 12  # 10 cont + 2 last_action
running_ms = RunningMeanStd(shape=(num_num,))
if isinstance(norm_data, dict):
    running_ms.mean = norm_data.get('mean', running_ms.mean)
    running_ms.var = norm_data.get('var', running_ms.var)
    running_ms.count = norm_data.get('count', running_ms.count)
state_norm = Normalization(num_categorical=num_cat, num_numerical=num_num, running_ms=running_ms)

# Build policy
cat_cols = ['line_id', 'bus_id', 'station_id', 'time_period', 'direction']
cat_code_dict = {
    'line_id':     {i: i for i in range(12)},
    'bus_id':      {i: i for i in range(389)},
    'station_id':  {0: 0},
    'time_period': {0: 0},
    'direction':   {0: 0, 1: 1},
}
emb = EmbeddingLayer(cat_code_dict, cat_cols, layer_norm=True, dropout=0.05)
state_dim = emb.output_dim + 12  # 29 + 12 = 41

policy = BusEmbeddingPolicy(
    num_inputs=state_dim,
    num_actions=2,
    hidden_size=48,
    embedding_layer=emb,
    action_range=1.0,
)
policy.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
policy.eval()
sampler_policy = BusSamplerPolicy(policy, DEVICE)

# ── Run evaluation ────────────────────────────────────────────────
env = BusSimEnv(path=SIM_PATH, debug=False)
print(f"[test] BusSimEnv loaded: {len(env.timetables)} trips, max_agent={env.max_agent_num}")

cumulative_reward = 0.0
n_decisions = 0
last_action = {}  # bus_id -> last raw tanh action

env.reset()
# Initialize state
action_dict = {k: None for k in range(env.max_agent_num)}
for _ in range(10000):
    state, reward, done = env.step(action_dict)
    if done: break
    if any(v for v in state.values()): break

t_start = time.time()
step_count = 0

while not env.done:
    active_buses = _extract_active_buses(env.state)
    
    if not active_buses:
        # Fast-forward
        state, reward, done = env.step(action_dict)
        step_count += 1
        continue
    
    action_dict = {k: None for k in range(env.max_agent_num)}
    
    for bus_id, obs_vec in active_buses:
        reward_val = env.reward.get(bus_id, 0.0)
        if reward_val is not None and reward_val != 0:
            cumulative_reward += reward_val
            n_decisions += 1
        
        # Augment obs with last_action
        prev_a = last_action.get(bus_id, np.zeros(2, dtype=np.float32))
        obs_aug = np.concatenate([obs_vec, prev_a])  # 15 + 2 = 17
        
        # Normalize
        obs_normed = state_norm(obs_aug)
        
        # Policy outputs raw tanh [-1, 1]
        obs_tensor = np.expand_dims(obs_normed, 0)
        action_raw = sampler_policy(obs_tensor, deterministic=True)[0]
        
        # Map to env via original mapping (hold=[0,60])
        hold, speed = _map_raw_to_env(action_raw)
        action_dict[bus_id] = [hold, speed]
        last_action[bus_id] = action_raw.copy()
    
    state, reward, done = env.step(action_dict)
    step_count += 1
    
    if step_count % 2000 == 0:
        elapsed = time.time() - t_start
        print(f"  step={step_count}, t_sim={env.current_time:.0f}s, "
              f"decisions={n_decisions}, reward={cumulative_reward:.0f}, "
              f"wall={elapsed:.1f}s")

elapsed = time.time() - t_start
print(f"\n{'='*60}")
print(f"RESULTS (H2O+ BusSimEnv + Checkpoint ep39)")
print(f"{'='*60}")
print(f"  Cumulative reward:  {cumulative_reward:,.0f}")
print(f"  Total decisions:    {n_decisions}")
print(f"  Sim time:           {env.current_time:.0f}s")
print(f"  Wall time:          {elapsed:.1f}s")
print(f"  Avg reward/decision: {cumulative_reward/max(n_decisions,1):.2f}")

# Also run zero-hold baseline
print(f"\n--- Running zero-hold baseline ---")
env.reset()
action_dict = {k: None for k in range(env.max_agent_num)}
for _ in range(10000):
    state, reward, done = env.step(action_dict)
    if done: break
    if any(v for v in state.values()): break

zero_reward = 0.0
zero_decisions = 0
while not env.done:
    active_buses = _extract_active_buses(env.state)
    action_dict = {k: None for k in range(env.max_agent_num)}
    for bus_id, obs_vec in active_buses:
        r = env.reward.get(bus_id, 0.0)
        if r is not None and r != 0:
            zero_reward += r
            zero_decisions += 1
        action_dict[bus_id] = [0.0, 1.0]  # zero hold, normal speed
    state, reward, done = env.step(action_dict)

print(f"  Zero-hold reward:   {zero_reward:,.0f}")
print(f"  Zero-hold decisions: {zero_decisions}")
print(f"  Improvement ratio:  {cumulative_reward/min(zero_reward,-1):.2f}x")
