#!/usr/bin/env python3
"""
dispatch_r3_nosnap_evals.py
============================
Watches the 6 R3 nosnap-baseline checkpoints (RLPD x3 + WSRL x3, all with
--nouse_snapshot_reset for fair comparison against H2O+ Contrastive) and
submits 9-scenario eval tasks to scheduler the moment each ckpt appears.
"""
import os, sys, time, subprocess, datetime

EXP_ROOT = '/home/erzhu419/mine_code/sumo-rl/H2Oplus/experiment_output'
SCHEDULER = '/home/erzhu419/.claude/skills/scheduler/scheduler.py'
SCHED_PY = '/home/erzhu419/anaconda3/bin/python'
EVAL_CWD = '/home/erzhu419/mine_code/sumo-rl/H2Oplus/SimpleSAC'

CKPTS = {
    'rlpd_nosnap_s42':  'h2oplus_bus_seed42_r3_rlpd_nosnap_s42/checkpoint_best.pt',
    'rlpd_nosnap_s123': 'h2oplus_bus_seed123_r3_rlpd_nosnap_s123/checkpoint_best.pt',
    'rlpd_nosnap_s789': 'h2oplus_bus_seed789_r3_rlpd_nosnap_s789/checkpoint_best.pt',
    'wsrl_nosnap_s42':  'h2oplus_bus_seed42_r3_wsrl_nosnap_s42/checkpoint_best.pt',
    'wsrl_nosnap_s123': 'h2oplus_bus_seed123_r3_wsrl_nosnap_s123/checkpoint_best.pt',
    'wsrl_nosnap_s789': 'h2oplus_bus_seed789_r3_wsrl_nosnap_s789/checkpoint_best.pt',
}
SUMOS = [1001, 1002, 1003]
ODS = ['0.6', '0.8', '1.0']
POLL_S = 120
MAX_HOURS = 24


def log(msg):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    print(f'[{ts}] [r3-nosnap] {msg}', flush=True)


def submit_evals_for(method):
    ckpt_path = os.path.join(EXP_ROOT, CKPTS[method])
    n_ok = 0
    for s in SUMOS:
        for od in ODS:
            cmd = [SCHED_PY, SCHEDULER, 'submit',
                '--description', f'R3 eval (nosnap): {method} sumo{s} od{od}',
                '--cwd', EVAL_CWD,
                '--signature', f'H2Oplus/r3_eval_nosnap_{method}_s{s}_od{od}',
                '--cmd', f'bash {EVAL_CWD}/run_multiseed_eval.sh {method} {s} {od}',
                '--cpu', '2', '--ram-mb', '4000', '--vram', '0',
                '--wait-for-file', ckpt_path,
                '--priority', 'low',
                '--require-node', 'local',
                '--allow-cpu-training',
                '--cpu-training-justification',
                'libsumo SUMO eval is CPU-only (single-session libsumo); SUMO not installed on remote nodes',
            ]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if r.returncode == 0 and 'submitted' in r.stdout:
                    n_ok += 1
            except subprocess.TimeoutExpired:
                pass
    return n_ok


def main():
    submitted = set()
    t_start = time.time()
    log(f'Watching {len(CKPTS)} nosnap ckpt paths; polling every {POLL_S}s')
    while len(submitted) < len(CKPTS):
        if (time.time() - t_start) / 3600.0 > MAX_HOURS:
            log(f'MAX_HOURS={MAX_HOURS}h reached; {len(submitted)}/{len(CKPTS)} ckpts had evals submitted; exiting')
            break
        for method, rel in CKPTS.items():
            if method in submitted: continue
            full = os.path.join(EXP_ROOT, rel)
            if os.path.exists(full) and os.path.getsize(full) > 1024:
                log(f'CKPT READY: {method}')
                n = submit_evals_for(method)
                log(f'  submitted {n}/9 eval tasks for {method}')
                submitted.add(method)
        if len(submitted) < len(CKPTS):
            remaining = sorted(set(CKPTS.keys()) - submitted)
            log(f'still waiting for {len(remaining)} ckpts: {remaining}')
            time.sleep(POLL_S)
    log(f'DONE. {len(submitted)}/{len(CKPTS)} method tags had eval tasks submitted '
        f'(total {len(submitted)*9} eval tasks queued).')


if __name__ == '__main__':
    main()
