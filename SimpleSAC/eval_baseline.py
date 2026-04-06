"""
eval_baseline.py — Run MultiLineSimEnv with zero-action policy for N episodes.
Reports avg_return using the same eval logic as BusEvalSampler.
"""
import os, sys, numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUS_H2O = os.path.join(os.path.dirname(_HERE), "bus_h2o")
sys.path.insert(0, _HERE)
sys.path.insert(0, _BUS_H2O)

from envs.bus_sim_env import MultiLineSimEnv

SIM_ENV_PATH = os.path.join(_BUS_H2O, "calibrated_env")
MAX_TRAJ_EVENTS = 100   # same as training default
N_EPISODES = 10


def _extract_active_buses(state_dict):
    active = []
    for bus_id, obs_list in state_dict.items():
        if not obs_list:
            continue
        inner = obs_list[-1]
        if isinstance(inner, (list, np.ndarray)):
            vec = inner
            if isinstance(vec, list) and vec and isinstance(vec[0], list):
                vec = vec[-1]
            if vec:
                active.append((bus_id, np.array(vec, dtype=np.float32)))
    return active


def run_episode(env, hold_time=0.0, max_events=MAX_TRAJ_EVENTS):
    """Run one episode identical to BusEvalSampler logic."""
    env.reset()

    # _init_env_state: step_fast until 7X has obs (same as real sampler)
    action_dict = {k: 0.0 for k in range(env.max_agent_num)}
    init_steps = 0
    for _ in range(10000):
        state, reward, done = env.step_fast(action_dict)
        init_steps += 1
        if done:
            print(f"    [WARN] done during init after {init_steps} ticks")
            return []
        if any(v for v in state.values()):
            print(f"    [init] first obs after {init_steps} ticks")
            break

    rewards_list = []
    pending = {}

    for ev_idx in range(max_events):
        active_buses = _extract_active_buses(env.state)
        if not active_buses:
            break

        action_dict = {k: None for k in range(env.max_agent_num)}

        for bus_id, obs_vec in active_buses:
            station_idx = int(obs_vec[2]) if len(obs_vec) > 2 else -1
            reward_val = env.reward.get(bus_id, 0.0)

            if bus_id in pending:
                prev = pending.pop(bus_id)
                if station_idx != prev["station_idx"]:
                    rewards_list.append(reward_val)

            action_dict[bus_id] = hold_time
            pending[bus_id] = {"station_idx": station_idx}

        state, reward, done = env.step_to_event(action_dict)
        if done:
            break

    return rewards_list


def main():
    print(f"Loading MultiLineSimEnv from {SIM_ENV_PATH}")
    env = MultiLineSimEnv(path=SIM_ENV_PATH, debug=False)

    policies = [
        ("zero (hold=0s)", 0.0),
        ("short hold (hold=5s)", 5.0),
        ("medium hold (hold=15s)", 15.0),
        ("long hold (hold=30s)", 30.0),
    ]

    for name, hold in policies:
        returns = []
        n_events_list = []
        for ep in range(N_EPISODES):
            rews = run_episode(env, hold_time=hold)
            ep_return = sum(rews)
            returns.append(ep_return)
            n_events_list.append(len(rews))
            print(f"  [{name}] ep {ep+1}/{N_EPISODES}: "
                  f"n_events={len(rews)}, return={ep_return:.1f}")

        avg = np.mean(returns)
        std = np.std(returns)
        avg_events = np.mean(n_events_list)
        print(f"  >>> {name}: avg_return={avg:.1f} ± {std:.1f}, "
              f"avg_events={avg_events:.1f}\n")


if __name__ == "__main__":
    main()
