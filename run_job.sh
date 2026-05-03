#!/bin/sh

#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J cuda_test

#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"

#BSUB -W 00:20

#BSUB -o output_%J.out
#BSUB -e error_%J.err


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613 || source activate 02613
 
SCRIPT_DIR=/zhome/be/e/219226/Documents/HPC_Project

module load cuda/11.8
nvcc -O2 -shared -Xcompiler -fPIC \
     -o ${SCRIPT_DIR}/jacobi_kernel.so \
     ${SCRIPT_DIR}/jacobi_kernel.cu
echo "[INFO] nvcc exit code: $?"
module unload cuda/11.8
 
echo "Part 8: Custom CUDA Kernel"
echo "Running 20 floor plans"
echo "Start: $(date)"
 
nvidia-smi | head -20
 
python "${SCRIPT_DIR}/simulate_cuda.py" 20 2>&1 | tee cuda_results_${LSB_JOBID}.csv
 
echo "End: $(date)"