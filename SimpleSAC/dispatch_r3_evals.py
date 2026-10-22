#!/usr/bin/env python3
"""
dispatch_r3_evals.py
====================
Long-running supervisor that watches the 18 R3 baseline checkpoint paths and
submits 9-scenario eval tasks to the scheduler the moment each ckpt appears.

Run via scheduler (--cpu 1 --ram-mb 256 --require-node local):
    python dispatch_r3_evals.py

Polls every 60s. Exits when all 18 ckpts have had their evals submitted.
"""
import os, sys, time, subprocess, datetime

EXP_ROOT = '/home/erzhu419/mine_code/sumo-rl/H2Oplus/experiment_output'
SCHEDULER = '/home/erzhu419/.claude/skills/scheduler/scheduler.py'
SCHED_PY = '/home/erzhu419/anaconda3/bin/python'  # scheduler runs in base env
EVAL_CWD = '/home/erzhu419/mine_code/sumo-rl/H2Oplus/SimpleSAC'

# method_tag → ckpt path (relative to EXP_ROOT)
CKPTS = {
    'bc_ep39_s42':  'bc_ep39_seed42/bc_final.pt',
    'bc_ep39_s123': 'bc_ep39_seed123/bc_final.pt',
    'bc_ep39_s789': 'bc_ep39_seed789/bc_final.pt',
    'iql_s42':  'iql_seed42/iql_final.pt',
    'iql_s123': 'iql_seed123/iql_final.pt',
    'iql_s789': 'iql_seed789/iql_final.pt',
    'awac_s42':  'awac_seed42/awac_final.pt',
    'awac_s123': 'awac_seed123/awac_final.pt',
    'awac_s789': 'awac_seed789/awac_final.pt',
    'td3bc_s42':  'td3bc_seed42/td3bc_final.pt',
    'td3bc_s123': 'td3bc_seed123/td3bc_final.pt',
    'td3bc_s789': 'td3bc_seed789/td3bc_final.pt',
    'rlpd_s42':  'h2oplus_bus_seed42_r3_rlpd_s42/checkpoint_best.pt',
    'rlpd_s123': 'h2oplus_bus_seed123_r3_rlpd_s123/checkpoint_best.pt',
    'rlpd_s789': 'h2oplus_bus_seed789_r3_rlpd_s789/checkpoint_best.pt',
    'wsrl_s42':  'h2oplus_bus_seed42_r3_wsrl_s42/checkpoint_best.pt',
    'wsrl_s123': 'h2oplus_bus_seed123_r3_wsrl_s123/checkpoint_best.pt',
    'wsrl_s789': 'h2oplus_bus_seed789_r3_wsrl_s789/checkpoint_best.pt',
}
SUMOS = [1001, 1002, 1003]
ODS = ['0.6', '0.8', '1.0']
POLL_S = 120
MAX_HOURS = 12  # safety cap


def log(msg):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    print(f'[{ts}] [r3-dispatch] {msg}', flush=True)


def submit_evals_for(method):
    """Submit 9 scenario evals for one method tag."""
    n_ok = 0
    for s in SUMOS:
        for od in ODS:
            cmd = [
                SCHED_PY, SCHEDULER, 'submit',
                '--description', f'R3 eval: {method} sumo{s} od{od}',
                '--cwd', EVAL_CWD,
                '--signature', f'H2Oplus/r3_eval_{method}_s{s}_od{od}',
                '--cmd', f'bash {EVAL_CWD}/run_multiseed_eval.sh {method} {s} {od}',
                '--cpu', '2', '--ram-mb', '4000', '--vram', '0',
                '--priority', 'low',
                '--require-node', 'local',
            ]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if r.returncode == 0 and 'submitted' in r.stdout:
                    n_ok += 1
            except subprocess.TimeoutExpired:
                log(f'  submit timeout for {method} s{s} od{od}')
    return n_ok


def main():
    submitted = set()
    t_start = time.time()
    log(f'Watching {len(CKPTS)} ckpt paths; polling every {POLL_S}s')

    while len(submitted) < len(CKPTS):
        elapsed_h = (time.time() - t_start) / 3600.0
        if elapsed_h > MAX_HOURS:
            log(f'MAX_HOURS={MAX_HOURS}h reached; {len(submitted)}/{len(CKPTS)} ckpts had evals submitted; exiting')
            break

        for method, rel in CKPTS.items():
            if method in submitted:
                continue
            full = os.path.join(EXP_ROOT, rel)
            if os.path.exists(full) and os.path.getsize(full) > 1024:  # avoid empty stub
                log(f'CKPT READY: {method} → {rel}')
                n = submit_evals_for(method)
                log(f'  submitted {n}/9 eval tasks for {method}')
                submitted.add(method)

        if len(submitted) < len(CKPTS):
            remaining = sorted(set(CKPTS.keys()) - submitted)
            log(f'still waiting for {len(remaining)} ckpts: {remaining[:5]}{"..." if len(remaining)>5 else ""}')
            time.sleep(POLL_S)

    log(f'DONE. {len(submitted)}/{len(CKPTS)} method tags had eval tasks submitted '
        f'(total {len(submitted)*9} eval tasks queued in scheduler).')
    # One last manual dispatch — the watcher does this every 60s anyway, but a final nudge is cheap.
    try:
        subprocess.run([SCHED_PY, SCHEDULER, 'dispatch'], timeout=60, capture_output=True)
    except Exception:
        pass


if __name__ == '__main__':
    main()
