#!/bin/sh
#BSUB -q c02613
#BSUB -J gpujob
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -o batch_output/gpujob_%J.out
#BSUB -e batch_output/gpujob_%J.err
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026
# Run Python script
kernprof -l simulate.py 10
python -m line_profiler -rmt "simulate.py.lprof"