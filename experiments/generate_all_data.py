#!/usr/bin/env python3
"""
Generate All Paper Data
=======================
Produces ALL data required by generate_prd_figures.py using real simulations.
No fallbacks, no fabricated data. Every data point comes from actual physics.

Outputs:
  results/fig2_ensemble_data.json   -- Broad + focused sweep for Figure 2 / Tables I, III
  results/fig5_highres_sweep.json   -- Velocity sweep for Figure 5 / Table V
  results/fig6_ultrarel_sweep.json  -- Ultra-relativistic sweep for Figure 6

Usage:
  python experiments/generate_all_data.py                # All data
  python experiments/generate_all_data.py --fig2         # Only Figure 2 data
  python experiments/generate_all_data.py --fig5         # Only Figure 5 data
  python experiments/generate_all_data.py --fig6         # Only Figure 6 data
  python experiments/generate_all_data.py --quick        # Reduced samples for testing
  python experiments/generate_all_data.py --workers 8    # Parallel workers
"""

import numpy as np
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from experiments.trajectory_classifier import TrajectoryOutcome, is_penrose_success
from experiments.thrust_comparison import SimulationConfig, simulate_single_impulse
from experiments.ensemble import clopper_pearson_ci

warnings.filterwarnings('ignore', category=RuntimeWarning)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_single_sim(params: dict) -> dict:
    """Run one simulation. Returns dict with is_escape, is_penrose, Delta_E, eta_cum."""
    try:
        config = SimulationConfig(
            a=params['a'], M=1.0,
            E0=params['E0'], Lz0=params['Lz0'], r0=15.0,
            v_e=params['v_e'],
            delta_m_fraction=params['delta_m'],
            tau_max=500.0, rtol=1e-9, atol=1e-11,
        )
        result = simulate_single_impulse(config)
        is_escape = result.outcome == TrajectoryOutcome.ESCAPE
        is_penrose = is_penrose_success(result)
        Delta_E = result.Delta_E if (result.Delta_E is not None and np.isfinite(result.Delta_E)) else float('nan')
        eta_cum = result.eta_cumulative if (result.eta_cumulative is not None and np.isfinite(result.eta_cumulative)) else float('nan')
        return {
            'is_escape': is_escape, 'is_penrose': is_penrose,
            'Delta_E': Delta_E, 'eta_cum': eta_cum,
        }
    except Exception as e:
        # Conservative convention (paper Sec. IV.D): failures count as non-escape.
        # NaN for numerical fields so they are excluded from averaging.
        import warnings
        warnings.warn(f"Simulation failed ({type(e).__name__}: {e}); counted as capture", RuntimeWarning)
        return {'is_escape': False, 'is_penrose': False, 'Delta_E': float('nan'), 'eta_cum': float('nan')}


def run_batch(param_list: list, n_workers: int = 4) -> list:
    """Run a batch of simulations in parallel."""
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(run_single_sim, p) for p in param_list]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def compute_stats(batch_results: list) -> dict:
    """Compute success rates, CIs, and efficiency stats from batch results."""
    n_total = len(batch_results)
    n_escape = sum(1 for r in batch_results if r['is_escape'])
    n_penrose = sum(1 for r in batch_results if r['is_penrose'])
    p_penrose = n_penrose / n_total if n_total > 0 else 0
    ci = clopper_pearson_ci(n_penrose, n_total, 0.05)
    # eta_cum > 0: for successful Penrose (DeltaE>0, Delta_m>0), eta is
    # always positive; non-positive values are numerical artifacts.
    eta_vals = [r['eta_cum'] for r in batch_results if r['is_penrose'] and np.isfinite(r['eta_cum']) and r['eta_cum'] > 0]
    eta_mean = float(np.mean(eta_vals)) if eta_vals else 0.0
    eta_std = float(np.std(eta_vals)) if len(eta_vals) > 1 else 0.0
    return {
        'n_total': n_total, 'n_escape': n_escape, 'n_penrose': n_penrose,
        'p_escape': n_escape / n_total if n_total > 0 else 0,
        'p_penrose': p_penrose, 'ci_penrose': list(ci),
        'eta_mean': eta_mean, 'eta_std': eta_std,
    }


# =====================================================================
# FIGURE 2 DATA: Broad + Focused sweeps for each spin
# =====================================================================

def generate_fig2_data(n_broad: int = 6400, n_focused: int = 3600,
                       n_workers: int = 4, verbose: bool = True) -> dict:
    """
    Generate Figure 2 data: Extraction-with-escape rate vs spin.

    Broad scan:   E0 in [0.95, 2.0], Lz in [-3.0, 6.0], uniform grid
    Focused scan: E0 in [1.1, 1.4],  Lz in [2.5, 3.8],  uniform grid
    Both use v_e=0.95, delta_m=0.20 (paper Tables I, III).
    """
    if verbose:
        print("=" * 70)
        print("GENERATING FIGURE 2 DATA: Ensemble Statistics")
        print("=" * 70)

    spins = [0.50, 0.70, 0.90, 0.95, 0.99]
    v_e = 0.95
    delta_m = 0.20

    # Determine grid sizes
    n_E_broad = int(np.sqrt(n_broad))
    n_Lz_broad = n_E_broad
    n_E_focused = int(np.sqrt(n_focused))
    n_Lz_focused = n_E_focused

    broad_results = {}
    focused_results = {}
    rng = np.random.default_rng(42)

    for a in spins:
        # --- Broad sweep ---
        if verbose:
            print(f"\n  Broad sweep a/M={a} ({n_E_broad}x{n_Lz_broad} = {n_E_broad*n_Lz_broad} points)...")
        E_vals = np.linspace(0.95, 2.0, n_E_broad)
        Lz_vals = np.linspace(-3.0, 6.0, n_Lz_broad)
        params = []
        for E0 in E_vals:
            for Lz0 in Lz_vals:
                params.append({'a': a, 'E0': E0, 'Lz0': Lz0, 'v_e': v_e, 'delta_m': delta_m})

        t0 = time.time()
        batch = run_batch(params, n_workers)
        stats = compute_stats(batch)
        broad_results[f'a={a}'] = stats
        if verbose:
            print(f"    Done in {time.time()-t0:.0f}s: escape={stats['p_escape']*100:.1f}%, "
                  f"penrose={stats['p_penrose']*100:.2f}% [{stats['ci_penrose'][0]*100:.2f}%, {stats['ci_penrose'][1]*100:.2f}%]")

        # --- Focused sweep ---
        if verbose:
            print(f"  Focused sweep a/M={a} ({n_E_focused}x{n_Lz_focused} = {n_E_focused*n_Lz_focused} points)...")
        E_vals = np.linspace(1.1, 1.4, n_E_focused)
        Lz_vals = np.linspace(2.5, 3.8, n_Lz_focused)
        params = []
        for E0 in E_vals:
            for Lz0 in Lz_vals:
                params.append({'a': a, 'E0': E0, 'Lz0': Lz0, 'v_e': v_e, 'delta_m': delta_m})

        t0 = time.time()
        batch = run_batch(params, n_workers)
        stats = compute_stats(batch)
        focused_results[f'a={a}'] = stats
        if verbose:
            print(f"    Done in {time.time()-t0:.0f}s: escape={stats['p_escape']*100:.1f}%, "
                  f"penrose={stats['p_penrose']*100:.2f}% [{stats['ci_penrose'][0]*100:.2f}%, {stats['ci_penrose'][1]*100:.2f}%]")

    data = {
        'broad': broad_results,
        'focused': focused_results,
        'metadata': {
            'generated': datetime.now().isoformat(),
            'n_broad_per_spin': n_E_broad * n_Lz_broad,
            'n_focused_per_spin': n_E_focused * n_Lz_focused,
            'broad_E_range': [0.95, 2.0], 'broad_Lz_range': [-3.0, 6.0],
            'focused_E_range': [1.1, 1.4], 'focused_Lz_range': [2.5, 3.8],
            'v_e': v_e, 'delta_m': delta_m,
            'spins': spins,
        }
    }

    out_path = RESULTS_DIR / "fig2_ensemble_data.json"
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    if verbose:
        print(f"\n  Saved to {out_path}")

    return data


# =====================================================================
# FIGURE 5 DATA: High-resolution velocity sweep
# =====================================================================

def generate_fig5_data(n_samples: int = 500, n_workers: int = 4,
                       verbose: bool = True) -> dict:
    """
    Generate Figure 5 data: Extraction-with-escape rate vs exhaust velocity.

    Sweet spot: Gaussian(E0=1.22, sigma=0.03) x Gaussian(Lz=3.05, sigma=0.08)
    Velocity: 0.80c to 0.99c in 0.01c steps
    Mass fractions: 0.10, 0.20, 0.30, 0.40
    """
    if verbose:
        print("=" * 70)
        print("GENERATING FIGURE 5 DATA: High-Resolution Velocity Sweep")
        print("=" * 70)

    a = 0.95
    E0_center, Lz0_center = 1.22, 3.05
    sigma_E, sigma_Lz = 0.03, 0.08
    v_e_values = np.arange(0.80, 0.995, 0.01)
    delta_m_values = [0.10, 0.20, 0.30, 0.40]
    n_total_sims = len(v_e_values) * len(delta_m_values) * n_samples

    if verbose:
        print(f"  Sweet spot: E0={E0_center}, Lz0={Lz0_center}")
        print(f"  Gaussian: sigma_E={sigma_E}, sigma_Lz={sigma_Lz}")
        print(f"  Velocities: {v_e_values[0]:.2f}c to {v_e_values[-1]:.2f}c ({len(v_e_values)} values)")
        print(f"  Samples per point: {n_samples}")
        print(f"  Total simulations: {n_total_sims:,}")

    results = {}
    rng = np.random.default_rng(42)
    t_start = time.time()

    for v_e in v_e_values:
        for dm in delta_m_values:
            key = f'v_e={v_e:.2f}_dm={dm}'
            if verbose:
                print(f"  v_e={v_e:.2f}c, dm={dm}...", end=" ", flush=True)
            t0 = time.time()
            params = []
            for _ in range(n_samples):
                params.append({
                    'a': a,
                    'E0': rng.normal(E0_center, sigma_E),
                    'Lz0': rng.normal(Lz0_center, sigma_Lz),
                    'v_e': v_e, 'delta_m': dm,
                })
            batch = run_batch(params, n_workers)
            stats = compute_stats(batch)
            stats['v_e'] = round(float(v_e), 2)
            stats['delta_m'] = dm
            results[key] = stats
            if verbose:
                print(f"{stats['p_penrose']*100:.1f}% ({time.time()-t0:.0f}s)")

    out_path = RESULTS_DIR / "fig5_highres_sweep.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"\n  Total time: {time.time()-t_start:.0f}s")
        print(f"  Saved to {out_path}")
    return results


# =====================================================================
# FIGURE 6 DATA: Ultra-relativistic saturation
# =====================================================================

def generate_fig6_data(n_samples: int = 500, n_workers: int = 4,
                       verbose: bool = True) -> dict:
    """
    Generate Figure 6 data: Efficiency saturation at ultra-relativistic velocities.

    Same sweet-spot Gaussian as Figure 5.
    Velocity: logarithmic spacing from 0.90c to 0.99999c.
    """
    if verbose:
        print("=" * 70)
        print("GENERATING FIGURE 6 DATA: Ultra-Relativistic Sweep")
        print("=" * 70)

    a = 0.95
    E0_center, Lz0_center = 1.22, 3.05
    sigma_E, sigma_Lz = 0.03, 0.08
    v_e_values = [
        0.90, 0.92, 0.94, 0.96, 0.98,
        0.99, 0.992, 0.994, 0.996, 0.998,
        0.999, 0.9992, 0.9994, 0.9996, 0.9998,
        0.9999, 0.99995, 0.99999,
    ]
    delta_m_values = [0.10, 0.20, 0.30, 0.40]
    n_total_sims = len(v_e_values) * len(delta_m_values) * n_samples

    if verbose:
        print(f"  Velocity range: {v_e_values[0]}c to {v_e_values[-1]}c ({len(v_e_values)} values)")
        print(f"  Gamma range: {1/np.sqrt(1-v_e_values[0]**2):.1f} to {1/np.sqrt(1-v_e_values[-1]**2):.0f}")
        print(f"  Samples per point: {n_samples}")
        print(f"  Total simulations: {n_total_sims:,}")

    results = {}
    rng = np.random.default_rng(42)
    t_start = time.time()

    for v_e in v_e_values:
        gamma = 1 / np.sqrt(1 - v_e**2)
        for dm in delta_m_values:
            key = f'v_e={v_e}_dm={dm}'
            if verbose:
                print(f"  v_e={v_e}c (γ={gamma:.1f}), dm={dm}...", end=" ", flush=True)
            t0 = time.time()
            params = []
            for _ in range(n_samples):
                params.append({
                    'a': a,
                    'E0': rng.normal(E0_center, sigma_E),
                    'Lz0': rng.normal(Lz0_center, sigma_Lz),
                    'v_e': v_e, 'delta_m': dm,
                })
            batch = run_batch(params, n_workers)
            stats = compute_stats(batch)
            stats['v_e'] = v_e
            stats['gamma'] = gamma
            stats['delta_m'] = dm
            results[key] = stats
            if verbose:
                print(f"{stats['p_penrose']*100:.1f}% eta={stats['eta_mean']*100:.1f}% ({time.time()-t0:.0f}s)")

    out_path = RESULTS_DIR / "fig6_ultrarel_sweep.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"\n  Total time: {time.time()-t_start:.0f}s")
        print(f"  Saved to {out_path}")
    return results


# =====================================================================
# MAIN
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate all paper data')
    parser.add_argument('--fig2', action='store_true', help='Only Figure 2 data')
    parser.add_argument('--fig5', action='store_true', help='Only Figure 5 data')
    parser.add_argument('--fig6', action='store_true', help='Only Figure 6 data')
    parser.add_argument('--quick', action='store_true', help='Quick mode (reduced samples)')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers')
    args = parser.parse_args()

    run_all = not (args.fig2 or args.fig5 or args.fig6)
    n_workers = args.workers

    if args.quick:
        n_broad, n_focused, n_sweep = 400, 225, 50
    else:
        n_broad, n_focused, n_sweep = 6400, 3600, 500

    if run_all or args.fig2:
        generate_fig2_data(n_broad=n_broad, n_focused=n_focused, n_workers=n_workers)
    if run_all or args.fig5:
        generate_fig5_data(n_samples=n_sweep, n_workers=n_workers)
    if run_all or args.fig6:
        generate_fig6_data(n_samples=n_sweep, n_workers=n_workers)

    print("\nDone! All data files are in results/")
