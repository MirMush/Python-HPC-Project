from os.path import join
import sys
import cupy as cp
import random
import time
import os

def load_data(load_dir, bid):
    SIZE = 512
    u = cp.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = cp.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = cp.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask
@profile
def jacobi(u, interior_mask, max_iter, atol=1e-6, check_interval=100):
    u = cp.array(u)
    interior_mask = cp.array(interior_mask)
    for i in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_interior = u[1:-1, 1:-1]

        if i % check_interval == 0 or i == max_iter - 1:
            diff = cp.abs(u_interior - u_new)
            diff = cp.where(interior_mask, diff, 0.0)
            delta = diff.max()
            u[1:-1, 1:-1] = cp.where(interior_mask, u_new, u_interior)
            if delta < atol:
                break
        else:
            u[1:-1, 1:-1] = cp.where(interior_mask, u_new, u_interior)
    return u

def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    return {
        'mean_temp': u_interior.mean(),
        'std_temp': u_interior.std(),
        'pct_above_18': cp.sum(u_interior > 18) / u_interior.size * 100,
        'pct_below_15': cp.sum(u_interior < 15) / u_interior.size * 100,
    }

if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load and Randomly Sample 20
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        all_ids = f.read().splitlines()
    
    N = 20
    building_ids = random.sample(all_ids, N)
    print(f"Running simulation for {N} random buildings...")

    # --- START TIMING ---
    cp.cuda.Device(0).synchronize()
    start_time = time.perf_counter()

    all_u0 = cp.empty((N, 514, 514))
    all_interior_mask = cp.empty((N, 512, 512), dtype='bool')
    
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    all_u = cp.empty_like(all_u0)

    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        all_u[i] = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)

    cp.cuda.Device(0).synchronize()
    end_time = time.perf_counter()
    # --- END TIMING ---

    # Print Stats and Save
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('\nbuilding_id, ' + ', '.join(stat_keys))
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(f"{float(stats[k]):.4f}" for k in stat_keys))
        
        cp.save(join(results_dir, f"{bid}_final_u.npy"), u)
        cp.save(join(results_dir, f"{bid}_mask.npy"), interior_mask)

    print(f"\nTotal execution time for {N} buildings: {end_time - start_time:.4f} seconds")