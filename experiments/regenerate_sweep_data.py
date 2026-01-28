#!/usr/bin/env python3
"""
Regenerate Sweep Data for Paper Figures
========================================
This script regenerates the JSON data files required for Figures 5 and 6.

Produces:
- results/fig5_highres_sweep.json: High-resolution velocity sweep for Figure 5
- results/fig6_ultrarel_sweep.json: Ultra-relativistic sweep for Figure 6

Estimated runtime: 4-8 hours depending on hardware (uses parallel processing).
Total trajectories: ~120,000

Usage:
    python experiments/regenerate_sweep_data.py          # Run all sweeps
    python experiments/regenerate_sweep_data.py --fig5   # Only Figure 5 data
    python experiments/regenerate_sweep_data.py --fig6   # Only Figure 6 data
    python experiments/regenerate_sweep_data.py --quick  # Quick test (reduced samples)
"""

import numpy as np
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.trajectory_classifier import (
    OrbitProfile, TrajectoryOutcome, ThrustStrategy, TrajectoryResult
)
from experiments.thrust_comparison import (
    SimulationConfig, simulate_single_impulse
)
from experiments.ensemble import clopper_pearson_ci, bca_bootstrap_ci

# Suppress integration warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Output directory
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_single_simulation(params: dict) -> dict:
    """
    Run a single trajectory simulation and return results.
    
    Parameters
    ----------
    params : dict
        Simulation parameters: a, E0, Lz0, v_e, delta_m
        
    Returns
    -------
    dict
        Result with is_escape, is_penrose, Delta_E, eta_cum
    """
    try:
        config = SimulationConfig(
            a=params['a'],
            M=1.0,
            E0=params['E0'],
            Lz0=params['Lz0'],
            r0=15.0,
            v_e=params['v_e'],
            delta_m=params['delta_m'],
            trigger_mode='periapsis',
            tau_max=500.0,
            rtol=1e-9,
            atol=1e-11,
        )
        
        result = simulate_single_impulse(config)
        
        is_escape = result.outcome == TrajectoryOutcome.ESCAPE
        is_penrose = (result.E_ex_min < 0) and is_escape
        Delta_E = result.Delta_E if np.isfinite(result.Delta_E) else 0.0
        eta_cum = result.eta_cum if np.isfinite(result.eta_cum) else 0.0
        
        return {
            'is_escape': is_escape,
            'is_penrose': is_penrose,
            'Delta_E': Delta_E,
            'eta_cum': eta_cum,
        }
    except Exception as e:
        # Count failures as non-escape
        return {
            'is_escape': False,
            'is_penrose': False,
            'Delta_E': 0.0,
            'eta_cum': 0.0,
        }


def run_batch_parallel(param_list: list, n_workers: int = 8) -> list:
    """Run a batch of simulations in parallel."""
    results = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(run_single_simulation, p) for p in param_list]
        
        for future in as_completed(futures):
            results.append(future.result())
    
    return results


def generate_fig5_data(n_samples_per_point: int = 500,
                       n_workers: int = 8,
                       verbose: bool = True) -> dict:
    """
    Generate high-resolution velocity sweep data for Figure 5.
    
    Sweeps exhaust velocity from 0.80c to 0.99c in 0.01c increments
    for different mass fractions (delta_m = 0.1, 0.2, 0.3, 0.4).
    
    Parameters
    ----------
    n_samples_per_point : int
        Number of Monte Carlo samples per (v_e, delta_m) combination
    n_workers : int
        Number of parallel workers
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Results keyed by 'v_e={v}_dm={dm}'
    """
    if verbose:
        print("="*70)
        print("GENERATING FIGURE 5 DATA: High-Resolution Velocity Sweep")
        print("="*70)
    
    # Fixed parameters (sweet spot)
    a = 0.95
    E0_center = 1.20
    Lz0_center = 3.0
    
    # Parameter variations (small perturbations for Monte Carlo)
    E_spread = 0.08
    Lz_spread = 0.3
    
    # Velocity grid: 0.80 to 0.99 in 0.01 increments
    v_e_values = np.arange(0.80, 0.995, 0.01)
    delta_m_values = [0.10, 0.20, 0.30, 0.40]
    
    n_total_simulations = len(v_e_values) * len(delta_m_values) * n_samples_per_point
    
    if verbose:
        print(f"  Velocity range: {v_e_values[0]:.2f}c to {v_e_values[-1]:.2f}c")
        print(f"  Delta_m values: {delta_m_values}")
        print(f"  Samples per point: {n_samples_per_point}")
        print(f"  Total simulations: {n_total_simulations:,}")
        print()
    
    results = {}
    rng = np.random.default_rng(42)
    
    t_start_total = time.time()
    
    for v_e in v_e_values:
        for delta_m in delta_m_values:
            key = f'v_e={v_e:.2f}_dm={delta_m}'
            
            if verbose:
                print(f"  Running v_e={v_e:.2f}c, delta_m={delta_m}...", end=" ", flush=True)
            
            t_start = time.time()
            
            # Generate Monte Carlo samples around sweet spot
            param_list = []
            for _ in range(n_samples_per_point):
                E0 = rng.uniform(E0_center - E_spread, E0_center + E_spread)
                Lz0 = rng.uniform(Lz0_center - Lz_spread, Lz0_center + Lz_spread)
                param_list.append({
                    'a': a,
                    'E0': E0,
                    'Lz0': Lz0,
                    'v_e': v_e,
                    'delta_m': delta_m,
                })
            
            # Run simulations
            batch_results = run_batch_parallel(param_list, n_workers)
            
            # Compute statistics
            n_total = len(batch_results)
            n_escape = sum(1 for r in batch_results if r['is_escape'])
            n_penrose = sum(1 for r in batch_results if r['is_penrose'])
            
            p_penrose = n_penrose / n_total
            ci_penrose = clopper_pearson_ci(n_penrose, n_total, 0.05)
            
            # Efficiency statistics (for escaped trajectories)
            eta_vals = [r['eta_cum'] for r in batch_results if r['is_penrose'] and r['eta_cum'] > 0]
            eta_mean = np.mean(eta_vals) if eta_vals else 0.0
            eta_std = np.std(eta_vals) if len(eta_vals) > 1 else 0.0
            
            t_elapsed = time.time() - t_start
            
            results[key] = {
                'v_e': round(v_e, 2),
                'delta_m': delta_m,
                'n_total': n_total,
                'n_penrose': n_penrose,
                'p_penrose': p_penrose,
                'ci_penrose': list(ci_penrose),
                'eta_mean': eta_mean,
                'eta_std': eta_std,
            }
            
            if verbose:
                print(f"Penrose: {n_penrose}/{n_total} = {100*p_penrose:.1f}% "
                      f"(η={100*eta_mean:.2f}%) [{t_elapsed:.1f}s]")
    
    t_total = time.time() - t_start_total
    
    if verbose:
        print()
        print(f"  Total time: {t_total/60:.1f} minutes")
        print(f"  Simulations/second: {n_total_simulations/t_total:.1f}")
    
    return results


def generate_fig6_data(n_samples_per_point: int = 500,
                       n_workers: int = 8,
                       verbose: bool = True) -> dict:
    """
    Generate ultra-relativistic sweep data for Figure 6.
    
    Sweeps exhaust velocity from 0.90c to 0.99999c (gamma ~ 2.3 to 224)
    for different mass fractions.
    
    Parameters
    ----------
    n_samples_per_point : int
        Number of Monte Carlo samples per (v_e, delta_m) combination
    n_workers : int
        Number of parallel workers
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Results keyed by 'v_e={v}_dm={dm}'
    """
    if verbose:
        print("="*70)
        print("GENERATING FIGURE 6 DATA: Ultra-Relativistic Sweep")
        print("="*70)
    
    # Fixed parameters (sweet spot)
    a = 0.95
    E0_center = 1.20
    Lz0_center = 3.0
    E_spread = 0.08
    Lz_spread = 0.3
    
    # Velocity grid: logarithmic spacing from 0.90c to 0.99999c
    # Covering gamma ~ 2.3 to 224
    v_e_values = [
        0.90, 0.92, 0.94, 0.96, 0.98,        # Moderate gamma
        0.99, 0.992, 0.994, 0.996, 0.998,    # High gamma
        0.999, 0.9992, 0.9994, 0.9996, 0.9998,  # Ultra-gamma
        0.9999, 0.99995, 0.99999              # Extreme
    ]
    delta_m_values = [0.10, 0.20, 0.30, 0.40]
    
    n_total_simulations = len(v_e_values) * len(delta_m_values) * n_samples_per_point
    
    if verbose:
        print(f"  Velocity range: {v_e_values[0]}c to {v_e_values[-1]}c")
        print(f"  Gamma range: {1/np.sqrt(1-v_e_values[0]**2):.1f} to {1/np.sqrt(1-v_e_values[-1]**2):.0f}")
        print(f"  Delta_m values: {delta_m_values}")
        print(f"  Samples per point: {n_samples_per_point}")
        print(f"  Total simulations: {n_total_simulations:,}")
        print()
    
    results = {}
    rng = np.random.default_rng(42)
    
    t_start_total = time.time()
    
    for v_e in v_e_values:
        gamma = 1 / np.sqrt(1 - v_e**2)
        
        for delta_m in delta_m_values:
            key = f'v_e={v_e}_dm={delta_m}'
            
            if verbose:
                print(f"  Running v_e={v_e}c (γ={gamma:.1f}), delta_m={delta_m}...", 
                      end=" ", flush=True)
            
            t_start = time.time()
            
            # Generate Monte Carlo samples
            param_list = []
            for _ in range(n_samples_per_point):
                E0 = rng.uniform(E0_center - E_spread, E0_center + E_spread)
                Lz0 = rng.uniform(Lz0_center - Lz_spread, Lz0_center + Lz_spread)
                param_list.append({
                    'a': a,
                    'E0': E0,
                    'Lz0': Lz0,
                    'v_e': v_e,
                    'delta_m': delta_m,
                })
            
            # Run simulations
            batch_results = run_batch_parallel(param_list, n_workers)
            
            # Compute statistics
            n_total = len(batch_results)
            n_escape = sum(1 for r in batch_results if r['is_escape'])
            n_penrose = sum(1 for r in batch_results if r['is_penrose'])
            
            p_penrose = n_penrose / n_total if n_total > 0 else 0
            ci_penrose = clopper_pearson_ci(n_penrose, n_total, 0.05)
            
            # Efficiency statistics
            eta_vals = [r['eta_cum'] for r in batch_results if r['is_penrose'] and r['eta_cum'] > 0]
            eta_mean = np.mean(eta_vals) if eta_vals else 0.0
            eta_std = np.std(eta_vals) if len(eta_vals) > 1 else 0.0
            
            t_elapsed = time.time() - t_start
            
            results[key] = {
                'v_e': v_e,
                'gamma': gamma,
                'delta_m': delta_m,
                'n_total': n_total,
                'n_penrose': n_penrose,
                'p_penrose': p_penrose,
                'ci_penrose': list(ci_penrose),
                'eta_mean': eta_mean,
                'eta_std': eta_std,
            }
            
            if verbose:
                print(f"Penrose: {n_penrose}/{n_total} = {100*p_penrose:.1f}% "
                      f"(η={100*eta_mean:.2f}%) [{t_elapsed:.1f}s]")
    
    t_total = time.time() - t_start_total
    
    if verbose:
        print()
        print(f"  Total time: {t_total/60:.1f} minutes")
        print(f"  Simulations/second: {n_total_simulations/t_total:.1f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate sweep data for paper figures 5 and 6"
    )
    parser.add_argument('--fig5', action='store_true',
                        help='Generate only Figure 5 data')
    parser.add_argument('--fig6', action='store_true',
                        help='Generate only Figure 6 data')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (50 samples per point)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    
    args = parser.parse_args()
    
    # If neither --fig5 nor --fig6 specified, do both
    do_fig5 = args.fig5 or (not args.fig5 and not args.fig6)
    do_fig6 = args.fig6 or (not args.fig5 and not args.fig6)
    
    # Number of workers
    n_workers = args.workers or multiprocessing.cpu_count()
    
    # Samples per point
    n_samples = 50 if args.quick else 500
    
    print("="*70)
    print("REGENERATE SWEEP DATA FOR PAPER FIGURES")
    print("="*70)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Workers: {n_workers}")
    print(f"  Samples per point: {n_samples}")
    print(f"  Mode: {'quick test' if args.quick else 'full'}")
    print()
    
    if do_fig5:
        print()
        fig5_data = generate_fig5_data(n_samples_per_point=n_samples, 
                                        n_workers=n_workers)
        
        # Save to JSON
        output_file = RESULTS_DIR / "fig5_highres_sweep.json"
        with open(output_file, 'w') as f:
            json.dump(fig5_data, f, indent=2)
        print(f"\n  Saved to {output_file}")
    
    if do_fig6:
        print()
        fig6_data = generate_fig6_data(n_samples_per_point=n_samples,
                                        n_workers=n_workers)
        
        # Save to JSON
        output_file = RESULTS_DIR / "fig6_ultrarel_sweep.json"
        with open(output_file, 'w') as f:
            json.dump(fig6_data, f, indent=2)
        print(f"\n  Saved to {output_file}")
    
    print()
    print("="*70)
    print("COMPLETE")
    print("="*70)
    print()
    print("Next step: python experiments/generate_prd_figures.py")


if __name__ == "__main__":
    main()
