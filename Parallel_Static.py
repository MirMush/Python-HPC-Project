from os.path import join
from multiprocessing.pool import Pool
import random
import time
import os
import cupy as cp

# Directory containing all floorplan data files
LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20_000  # Maximum number of Jacobi iterations per building
ABS_TOL = 1e-4    # Convergence threshold - stop early if change drops below this

def load_data(load_dir, bid):
    """Load domain and interior mask arrays for a given building ID."""
    SIZE = 512  # Grid resolution (512x512 interior cells)
    
    # Initialise a (514x514) grid of zeros - the +2 adds boundary padding on each side
    u = cp.zeros((SIZE + 2, SIZE + 2))
    
    # Fill the interior (excluding boundary rows/cols) with the building's temperature domain
    u[1:-1, 1:-1] = cp.load(join(load_dir, f"{bid}_domain.npy"))
    
    # Load the boolean mask that marks which cells are interior (True = interior cell)
    interior_mask = cp.load(join(load_dir, f"{bid}_interior.npy"))
    
    return u, interior_mask

def jacobi(u, interior_mask, max_iter, atol=1e-6, check_interval=100):
    """Run Jacobi iteration to solve the steady-state heat equation on the grid."""
    u = cp.array(u)                        # Ensure u is a CuPy GPU array
    interior_mask = cp.array(interior_mask) # Ensure mask is a CuPy GPU array

    for i in range(max_iter):
        # Compute the average of the 4 neighbours for every interior cell (Jacobi update rule)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        
        # Extract the current interior values (before update) for comparison
        u_interior = u[1:-1, 1:-1]

        if i % check_interval == 0 or i == max_iter - 1:
            # Every check_interval steps, compute the max change across interior cells
            diff = cp.abs(u_interior - u_new)
            
            # Zero out the diff at boundary/wall cells so they don't affect convergence check
            diff = cp.where(interior_mask, diff, 0.0)
            
            # Find the largest change across the entire grid
            delta = diff.max()
            
            # Apply the Jacobi update only to interior cells; leave wall cells unchanged
            u[1:-1, 1:-1] = cp.where(interior_mask, u_new, u_interior)
            
            # Stop early if the solution has converged below the tolerance
            if delta < atol:
                break
        else:
            # On non-check steps, just apply the update without computing convergence
            u[1:-1, 1:-1] = cp.where(interior_mask, u_new, u_interior)

    return u

def summary_stats(u, interior_mask):
    """Compute temperature statistics over the interior cells of the solved grid."""
    # Flatten and filter to only interior cell values using the boolean mask
    u_interior = u[1:-1, 1:-1][interior_mask]
    
    return {
        'mean_temp':    float(u_interior.mean()),                                     # Average temperature
        'std_temp':     float(u_interior.std()),                                      # Spread of temperatures
        'pct_above_18': float(cp.sum(u_interior > 18) / u_interior.size * 100),      # % of cells above 18°C
        'pct_below_15': float(cp.sum(u_interior < 15) / u_interior.size * 100),      # % of cells below 15°C
    }

def process_building(bid):
    """
    Worker function executed by each subprocess.
    Loads data, runs Jacobi, computes stats, and saves results for one building.
    Must be a top-level function so Python's multiprocessing can pickle and send it to subprocesses.
    """
    # Initialise this subprocess's CUDA context on GPU 0 (required in every child process)
    cp.cuda.Device(0).use()

    # Load the padded temperature grid and interior mask for this building
    u0, interior_mask = load_data(LOAD_DIR, bid)
    
    # Run the Jacobi solver to reach steady-state temperature distribution
    u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
    
    # Compute summary statistics from the converged solution
    stats = summary_stats(u, interior_mask)

    # Save the final temperature grid and mask to disk for later analysis
    results_dir = 'results'
    cp.save(join(results_dir, f"{bid}_final_u.npy"), u)
    cp.save(join(results_dir, f"{bid}_mask.npy"), interior_mask)

    # Return the building ID and its stats so the main process can collect them
    return bid, stats

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')  # ← add this as the very first line
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        all_ids = f.read().splitlines()

    N = 100
    NUM_RUNS = 5  # Repeat each config 5 times for reliable mean

    # Fix the same 100 buildings for ALL configs — removes data variance as a confound
    random.seed(42) 
    building_ids = random.sample(all_ids, N)
    print(f"Using {N} fixed buildings across all experiments")

    # Powers of 2 sweep — standard for Amdahl analysis
    worker_counts = [1, 2, 4, 8, 16]

    # Store results: {num_workers: [elapsed_run1, elapsed_run2, ...]}
    all_times = {}

    for num_workers in worker_counts:
        chunksize = max(1, N // num_workers)  # Static scheduling: equal slice per worker
        run_times = []
        print(f"\nWorkers = {num_workers}, chunksize = {chunksize}")

        for run in range(1, NUM_RUNS + 1):
            cp.cuda.Device(0).synchronize()   # Flush GPU before timing
            start_time = time.perf_counter()

            with Pool(processes=num_workers) as pool:
                results = pool.map(process_building, building_ids, chunksize=chunksize)

            cp.cuda.Device(0).synchronize()   # Wait for GPU after last task
            elapsed = time.perf_counter() - start_time
            run_times.append(elapsed)
            print(f"  run {run}/{NUM_RUNS}: {elapsed:.2f}s")

        all_times[num_workers] = run_times
        mean_t = sum(run_times) / len(run_times)
        print(f"  → mean: {mean_t:.2f}s")

    # Print summary table
    print("\n\n=== RESULTS SUMMARY ===")
    print(f"{'Workers':>8}  {'Run1':>7}  {'Run2':>7}  {'Run3':>7}  {'Run4':>7}  {'Run5':>7}  {'Mean':>7}  {'Speedup':>8}")
    t1_mean = sum(all_times[1]) / len(all_times[1])  # Serial baseline
    for nw in worker_counts:
        times = all_times[nw]
        mean_t = sum(times) / len(times)
        speedup = t1_mean / mean_t
        runs_str = "  ".join(f"{t:>7.2f}" for t in times)
        print(f"{nw:>8}  {runs_str}  {mean_t:>7.2f}  {speedup:>8.3f}x")

    # Save to CSV for plot_speedup.py
    import csv
    with open('benchmark_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['num_workers', 'run', 'elapsed_s'])
        for nw in worker_counts:
            for run_idx, t in enumerate(all_times[nw], start=1):
                writer.writerow([nw, run_idx, f"{t:.4f}"])
    print("\nSaved to benchmark_results.csv")