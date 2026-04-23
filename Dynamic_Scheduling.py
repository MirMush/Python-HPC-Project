from os.path import join
from multiprocessing.pool import Pool
import random
import time
import os
import cupy as cp

# Directory containing all floorplan data files
LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20_000
ABS_TOL = 1e-4

def load_data(load_dir, bid):
    SIZE = 512
    u = cp.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = cp.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = cp.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

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
        'mean_temp':    float(u_interior.mean()),
        'std_temp':     float(u_interior.std()),
        'pct_above_18': float(cp.sum(u_interior > 18) / u_interior.size * 100),
        'pct_below_15': float(cp.sum(u_interior < 15) / u_interior.size * 100),
    }

def process_building(bid):
    cp.cuda.Device(0).use()
    u0, interior_mask = load_data(LOAD_DIR, bid)
    u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
    stats = summary_stats(u, interior_mask)
    results_dir = 'results'
    cp.save(join(results_dir, f"{bid}_final_u.npy"), u)
    cp.save(join(results_dir, f"{bid}_mask.npy"), interior_mask)
    return bid, stats

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        all_ids = f.read().splitlines()

    N = 100
    NUM_RUNS = 5

    # Use the same 100 buildings as static for a fair comparison
    random.seed(42) 
    building_ids = random.sample(all_ids, N)
    print(f"Using {N} fixed buildings across all experiments (dynamic scheduling)")

    worker_counts = [1, 2, 4, 8, 16]
    all_times = {}

    for num_workers in worker_counts:
        run_times = []
        print(f"\nWorkers = {num_workers} (dynamic — no chunksize)")

        for run in range(1, NUM_RUNS + 1):
            cp.cuda.Device(0).synchronize()
            start_time = time.perf_counter()

            with Pool(processes=num_workers) as pool:
                
                results = pool.map(process_building, building_ids)

            cp.cuda.Device(0).synchronize()
            elapsed = time.perf_counter() - start_time
            run_times.append(elapsed)
            print(f"  run {run}/{NUM_RUNS}: {elapsed:.2f}s")

        all_times[num_workers] = run_times
        mean_t = sum(run_times) / len(run_times)
        print(f"  → mean: {mean_t:.2f}s")

    print("\n\n=== RESULTS SUMMARY (DYNAMIC) ===")
    print(f"{'Workers':>8}  {'Run1':>7}  {'Run2':>7}  {'Run3':>7}  {'Run4':>7}  {'Run5':>7}  {'Mean':>7}  {'Speedup':>8}")
    t1_mean = sum(all_times[1]) / len(all_times[1])
    for nw in worker_counts:
        times = all_times[nw]
        mean_t = sum(times) / len(times)
        speedup = t1_mean / mean_t
        runs_str = "  ".join(f"{t:>7.2f}" for t in times)
        print(f"{nw:>8}  {runs_str}  {mean_t:>7.2f}  {speedup:>8.3f}x")

    import csv
    with open('benchmark_results_dynamic.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['num_workers', 'run', 'elapsed_s'])
        for nw in worker_counts:
            for run_idx, t in enumerate(all_times[nw], start=1):
                writer.writerow([nw, run_idx, f"{t:.4f}"])
    print("\nSaved to benchmark_results_dynamic.csv")