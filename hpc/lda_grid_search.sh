#!/bin/bash
#BSUB -J lda_grid[1-6]
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 08:00
#BSUB -B
#BSUB -N
#BSUB -oo hpc/logs/%J_%I.out
#BSUB -eo hpc/logs/%J_%I.err

set -eo pipefail
cd "$LS_SUBCWD"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

KS=(5 10 15 20 25 30)
K=${KS[$((LSB_JOBINDEX - 1))]}

python scripts/run_lda.py "$K"  --seed 0 --n-iter 200
