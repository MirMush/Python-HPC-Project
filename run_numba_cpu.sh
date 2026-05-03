#!/bin/sh

#BSUB -q hpc
#BSUB -J numba_cpu_p7

#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"

#BSUB -W 00:30

#BSUB -o output_numba_%J.out
#BSUB -e error_numba_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613 || source activate 02613

SCRIPT_DIR=/zhome/be/e/219226/Documents/HPC_Project

echo "Part 7: Numba JIT CPU"
echo "Running 20 floor plans"
echo "Start: $(date)"

python "${SCRIPT_DIR}/simulate_numba_cpu.py" 20 2>&1 | tee numba_results_${LSB_JOBID}.csv

echo "End: $(date)"
