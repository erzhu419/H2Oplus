#!/usr/bin/env python3
"""Wait for the 2 missing RLPD nosnap ckpts (s42, s123), then submit 9 scenario evals each."""
import os, sys, time, subprocess

EXP_ROOT = '/home/erzhu419/mine_code/sumo-rl/H2Oplus/experiment_output'
SCHEDULER = '/home/erzhu419/.claude/skills/scheduler/scheduler.py'
SCHED_PY = '/home/erzhu419/anaconda3/bin/python'
EVAL_CWD = '/home/erzhu419/mine_code/sumo-rl/H2Oplus/SimpleSAC'
JUST = ('libsumo SUMO eval is CPU-only (single-session libsumo); SUMO not '
        'installed on remote nodes; only nosnap ckpts with completed training are eval-able')

TARGETS = {
    'rlpd_nosnap_s42':  'h2oplus_bus_seed42_r3_rlpd_nosnap_s42/checkpoint_best.pt',
    'rlpd_nosnap_s123': 'h2oplus_bus_seed123_r3_rlpd_nosnap_s123/checkpoint_best.pt',
}
SUMOS = [1001, 1002, 1003]; ODS = ['0.6', '0.8', '1.0']

submitted = set()
t0 = time.time()
while len(submitted) < len(TARGETS):
    if (time.time() - t0) / 3600 > 6:
        print(f'TIMEOUT 6h; submitted {len(submitted)}/{len(TARGETS)}', flush=True); break
    for method, rel in TARGETS.items():
        if method in submitted: continue
        full = os.path.join(EXP_ROOT, rel)
        if os.path.exists(full) and os.path.getsize(full) > 1024:
            print(f'CKPT READY: {method}', flush=True)
            n_ok = 0
            for s in SUMOS:
                for od in ODS:
                    cmd = [SCHED_PY, SCHEDULER, 'submit',
                        '--description', f'R3 nosnap eval (retry): {method} sumo{s} od{od}',
                        '--cwd', EVAL_CWD,
                        '--signature', f'H2Oplus/r3_eval_nosnap_{method}_s{s}_od{od}',
                        '--cmd', f'bash {EVAL_CWD}/run_multiseed_eval.sh {method} {s} {od}',
                        '--cpu','2','--ram-mb','4000','--vram','0',
                        '--wait-for-file', full,
                        '--priority','low','--require-node','local',
                        '--allow-cpu-training','--cpu-training-justification', JUST,
                    ]
                    try:
                        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                        if r.returncode == 0 and 'submitted' in r.stdout: n_ok += 1
                    except subprocess.TimeoutExpired: pass
            print(f'  submitted {n_ok}/9 for {method}', flush=True)
            submitted.add(method)
    if len(submitted) < len(TARGETS): time.sleep(120)
print(f'DONE: {len(submitted)*9} eval tasks queued.', flush=True)
