#!/bin/bash
# Sequential BC training queue.
# Usage: run_bc_queue.sh <data> <seed1> <seed2> ...
set -e
DATA=$1
shift
SEEDS=$@

cd /home/erzhu419/mine_code/sumo-rl/H2Oplus/SimpleSAC

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

LOGDIR=/home/erzhu419/mine_code/sumo-rl/H2Oplus/experiment_output/bc_logs
mkdir -p $LOGDIR

for SEED in $SEEDS; do
  echo "=== $(date) Starting BC data=$DATA seed=$SEED ==="
  conda run -n LSTM-RL python train_bc.py --data $DATA --seed $SEED --n_steps 50000 \
      --device cpu --print_every 5000 \
      > $LOGDIR/${DATA}_seed${SEED}.log 2>&1
  echo "=== $(date) Done BC data=$DATA seed=$SEED ==="
done
echo "QUEUE_DONE_$DATA"
