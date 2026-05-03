#!/bin/sh

#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J cuda_all

#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"

#BSUB -W 01:00

#BSUB -o output_all_%J.out
#BSUB -e error_all_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613 || source activate 02613

SCRIPT_DIR=/zhome/be/e/219226/Documents/HPC_Project

# Compile kernel 
module load cuda/11.8
nvcc -O2 -shared -Xcompiler -fPIC \
     -o ${SCRIPT_DIR}/jacobi_kernel.so \
     ${SCRIPT_DIR}/jacobi_kernel.cu
echo "[INFO] nvcc exit code: $?"
module unload cuda/11.8

echo "=== Full dataset run: all 4571 floorplans ==="
echo "Start: $(date)"

nvidia-smi | head -5

python "${SCRIPT_DIR}/simulate_cuda.py" 4571 \
    > ${SCRIPT_DIR}/all_results.csv \
    2> ${SCRIPT_DIR}/all_results.log

echo "End: $(date)"
cat ${SCRIPT_DIR}/all_results.log