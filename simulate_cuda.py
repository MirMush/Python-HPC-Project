"""
Part 8 – Custom CUDA kernel Jacobi solver
"""

from os.path import join, dirname, abspath
import sys
import time
import traceback
import os
import ctypes
import subprocess
import tempfile
import numpy as np

SCRIPT_DIR = dirname(abspath(__file__))
CU_FILE  = join(SCRIPT_DIR, "jacobi_kernel.cu")
SO_FILE  = join(SCRIPT_DIR, "jacobi_kernel.so")

def compile_kernel():
    print(f"[INFO] Compiling {CU_FILE} -> {SO_FILE}", file=sys.stderr)
    cmd = [
        "nvcc", "-O2", "-shared", "-Xcompiler", "-fPIC",
        "-o", SO_FILE, CU_FILE
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] nvcc failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Compiled OK: {SO_FILE}", file=sys.stderr)

if not os.path.exists(SO_FILE):
    compile_kernel()
else:
    print(f"[INFO] Using cached {SO_FILE}", file=sys.stderr)

# Load the shared library
try:
    lib = ctypes.CDLL(SO_FILE)
except OSError as e:
    print(f"[ERROR] Failed to load {SO_FILE}: {e}", file=sys.stderr)
    compile_kernel()
    lib = ctypes.CDLL(SO_FILE)

# Set argument types for run_jacobi(double*, bool*, int, int, int)
lib.run_jacobi.restype  = None
lib.run_jacobi.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # u_host (in/out)
    ctypes.POINTER(ctypes.c_bool),    # mask_host
    ctypes.c_int,                     # max_iter
    ctypes.c_int,                     # padded_rows (514)
    ctypes.c_int,                     # padded_cols (514)
]
print("[INFO] Kernel library loaded and configured.", file=sys.stderr)

# Data loading

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

# Jacobi via CUDA C kernel

def jacobi_cuda(u_host, interior_mask_host, max_iter):
    """
    Calls the compiled CUDA C kernel via ctypes.
    """
    u = np.ascontiguousarray(u_host, dtype=np.float64)
    mask = np.ascontiguousarray(interior_mask_host, dtype=np.bool_)

    u_ptr    = u.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    mask_ptr = mask.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))

    lib.run_jacobi(u_ptr, mask_ptr, max_iter, 514, 514)

    return u

# Summary statistics

def summary_stats(u, interior_mask):
    u_interior   = u[1:-1, 1:-1][interior_mask]
    mean_temp    = u_interior.mean()
    std_temp     = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp':    mean_temp,
        'std_temp':     std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }

# Main

if __name__ == '__main__':
    print(f"[INFO] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}",
          file=sys.stderr)

    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    N = int(sys.argv[1]) if len(sys.argv) >= 2 else 1
    building_ids = building_ids[:N]

    print(f"[INFO] Loading {N} floor plans...", file=sys.stderr)
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    MAX_ITER = 20_000

    # Warm-up: run 1 iteration on first floor plan to trigger JIT + driver init
    print("[INFO] Warming up kernel (1 iteration)...", file=sys.stderr)
    try:
        jacobi_cuda(all_u0[0], all_interior_mask[0], 1)
        print("[INFO] Warm-up done.", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Warm-up failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Running CUDA Jacobi on {N} floor plans...", file=sys.stderr)
    t_start = time.perf_counter()

    all_u = np.empty_like(all_u0)
    try:
        for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
            all_u[i] = jacobi_cuda(u0, interior_mask, MAX_ITER)
            if (i + 1) % 5 == 0:
                print(f"[INFO] Processed {i+1}/{N} floor plans...", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] CUDA Jacobi failed on floor plan {i}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    t_end = time.perf_counter()
    elapsed = t_end - t_start

    print(f"[TIMING] N={N} | Total: {elapsed:.2f}s | Per floor plan: {elapsed/N:.3f}s",
          file=sys.stderr)
    estimated_total = elapsed / N * 4571
    print(f"[ESTIMATE] All 4571 buildings: ~{estimated_total:.1f}s ({estimated_total/60:.1f} min)",
          file=sys.stderr)

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))