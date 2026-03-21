"""Smoke test for calibrated SimpleSim with real 7X/7S network data."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'LSTM-RL-legacy'))

from envs.bus_sim_env import BusSimEnv

CALIB_PATH = os.path.join(os.path.dirname(__file__), 'calibrated_env')
print(f'Loading BusSimEnv from: {CALIB_PATH}')

env = BusSimEnv(path=CALIB_PATH)
print(f'  max_agent_num   = {env.max_agent_num}')
print(f'  num_stations    = {len(env.stations)}')
print(f'  route_segments  = {len(env.routes)}')
print(f'  state_dim       = {env.state_dim}')
print(f'  eff_stations[:4]= {env.effective_station_name[:4]}')
print(f'  eff_periods[:3] = {env.effective_period[:3]}')

actions = {k: 0.0 for k in range(env.max_agent_num)}
env.reset()
n_events = 0
for i in range(100):
    obs, rew, done, info = env.step(actions)
    if any(len(v) > 0 for v in obs.values()):
        n_events += 1
    if done:
        break

snap = info['snapshot']
print(f'\nAfter 100 steps:')
print(f'  events={n_events}')
print(f'  sim_time={snap["sim_time"]:.0f}s')
print(f'  active_buses={len(snap["all_buses"])}')
if snap['all_buses']:
    b0 = snap['all_buses'][0]
    print(f'  bus[0] abs_dist={b0["absolute_distance"]:.0f}m (valid for ~13km route)')
print(f'  stations={len(snap["all_stations"])}')
if len(snap['all_stations']) >= 6:
    s5 = snap['all_stations'][5]
    print(f'  station[5] pos={s5["pos"]:.0f}m name={s5["station_name"]}')

print('\nOK - Calibrated SimpleSim functional!')
