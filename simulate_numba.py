"""
Part 7 – Numba JIT Jacobi solver (CPU)
"""

from os.path import join
import sys
import time

import numpy as np
from numba import njit

# Data loading 

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

# Numba JIT Jacobi kernel

@njit(cache=True)
def jacobi_numba(u, interior_mask, max_iter, atol=1e-6):
    rows, cols = interior_mask.shape   # 512 x 512
    u = u.copy()

    for iteration in range(max_iter):
        delta = 0.0

        for i in range(rows):        # outer: rows  (slow index)
            for j in range(cols):    # inner: cols  (fast index, cache-friendly)
                if not interior_mask[i, j]:
                    continue

                pi = i + 1   # offset into padded grid
                pj = j + 1

                new_val = 0.25 * (
                    u[pi, pj - 1] +   # left
                    u[pi, pj + 1] +   # right
                    u[pi - 1, pj] +   # up
                    u[pi + 1, pj]     # down
                )

                diff = abs(u[pi, pj] - new_val)
                if diff > delta:
                    delta = diff

                u[pi, pj] = new_val

        if delta < atol:
            break

    return u

# Summary statistics

def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
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
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    N = int(sys.argv[1]) if len(sys.argv) >= 2 else 1
    building_ids = building_ids[:N]

    # --- Load floor plans ---------------------------------------------------
    print(f"[INFO] Loading {N} floor plans...", file=sys.stderr)
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    MAX_ITER = 20_000
    ABS_TOL  = 1e-4

    # --- JIT warm-up (compile cost excluded from timing) --------------------
    print("[INFO] Warming up Numba JIT (compiling)...", file=sys.stderr)
    _dummy_u    = np.zeros((514, 514))
    _dummy_mask = np.zeros((512, 512), dtype='bool')
    _dummy_mask[1, 1] = True
    jacobi_numba(_dummy_u, _dummy_mask, 1)
    print("[INFO] JIT warm-up done.", file=sys.stderr)

    # --- Run Jacobi ---------------------------------------------------------
    print(f"[INFO] Running Numba JIT Jacobi on {N} floor plans...", file=sys.stderr)
    t_start = time.perf_counter()

    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi_numba(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u

    t_end = time.perf_counter()
    elapsed = t_end - t_start

    print(f"[TIMING] N={N} | Total: {elapsed:.2f}s | Per floor plan: {elapsed/N:.3f}s",
          file=sys.stderr)

    total_buildings = 4571
    estimated_total = elapsed / N * total_buildings
    print(f"[ESTIMATE] All {total_buildings} buildings: ~{estimated_total:.1f}s "
          f"({estimated_total/60:.1f} min)", file=sys.stderr)

    # --- Print CSV results --------------------------------------------------
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))