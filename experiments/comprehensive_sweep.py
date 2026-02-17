"""
Comprehensive Parameter Sweep for Penrose Process
==================================================
In-depth statistical analysis to establish robust rarity claims.

This script performs exhaustive parameter sweeps across:
1. Black hole spin: a/M = 0.5, 0.7, 0.9, 0.95, 0.99
2. Initial energy: E0 = 0.95 to 2.0
3. Angular momentum: Lz0 = -3.0 to 6.0 (retrograde to strongly prograde)
4. Exhaust velocity: v_e = 0.8, 0.9, 0.95, 0.98
5. Mass fraction: delta_m = 0.1, 0.2, 0.3, 0.4

Statistical outputs:
- Clopper-Pearson exact CIs for success rates
- BCa bootstrap CIs for efficiency metrics
- Sweet spot identification
- Efficiency vs spin correlation
"""

import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.trajectory_classifier import (
    OrbitProfile, TrajectoryOutcome, ThrustStrategy, TrajectoryResult
)
from experiments.thrust_comparison import (
    SimulationConfig, simulate_single_impulse, simulate_geodesic
)
from experiments.ensemble import (
    EnsembleConfig, EnsembleResult, run_ensemble,
    clopper_pearson_ci, bca_bootstrap_ci,
    latin_hypercube_sample, uniform_random_sample
)


# Suppress integration warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# SWEEP CONFIGURATION
# =============================================================================

@dataclass
class ComprehensiveSweepConfig:
    """Configuration for comprehensive parameter sweep."""
    
    # Output directory
    output_dir: str = "results/comprehensive_sweep"
    
    # Black hole spins to test
    spins: Tuple[float, ...] = (0.5, 0.7, 0.9, 0.95, 0.99)
    
    # Phase space grid - WIDE coverage
    E_range_broad: Tuple[float, float] = (0.95, 2.0)
    Lz_range_broad: Tuple[float, float] = (-3.0, 6.0)
    n_E_broad: int = 80
    n_Lz_broad: int = 80
    
    # Phase space grid - FOCUSED (sweet spot region)
    E_range_focused: Tuple[float, float] = (1.10, 1.40)
    Lz_range_focused: Tuple[float, float] = (2.5, 3.8)
    n_E_focused: int = 60
    n_Lz_focused: int = 60
    
    # Thrust parameter variations
    v_e_values: Tuple[float, ...] = (0.80, 0.90, 0.95, 0.98)
    delta_m_values: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
    
    # Monte Carlo ensemble size per configuration
    n_samples_per_config: int = 1000
    
    # Statistical parameters
    n_bootstrap: int = 10000
    alpha: float = 0.05  # 95% CI
    
    # Parallelization
    n_workers: int = 8
    
    # Random seed for reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Compute derived quantities."""
        self.n_broad_total = self.n_E_broad * self.n_Lz_broad
        self.n_focused_total = self.n_E_focused * self.n_Lz_focused


# =============================================================================
# SINGLE CONFIGURATION RUNNER
# =============================================================================

def run_single_config(params: Dict) -> Dict:
    """
    Run simulation for a single (E0, Lz0, a, v_e, delta_m) configuration.
    
    Returns a dictionary with results.
    """
    a = params['a']
    E0 = params['E0']
    Lz0 = params['Lz0']
    v_e = params['v_e']
    delta_m = params['delta_m']
    
    config = SimulationConfig(
        a=a,
        E0=E0,
        Lz0=Lz0,
        v_e=v_e,
        delta_m_fraction=delta_m,
        r0=15.0,
        m0=1.0,
        tau_max=800.0,
        escape_radius=50.0,
    )
    
    result = simulate_single_impulse(config)
    
    return {
        'a': a,
        'E0': E0,
        'Lz0': Lz0,
        'v_e': v_e,
        'delta_m': delta_m,
        'outcome': result.outcome.name,
        'is_escape': result.outcome == TrajectoryOutcome.ESCAPE,
        'is_penrose': (result.outcome == TrajectoryOutcome.ESCAPE and 
                       result.E_ex_mean is not None and 
                       result.E_ex_mean < 0),
        'Delta_E': result.Delta_E if result.Delta_E is not None else np.nan,
        'E_ex': result.E_ex_mean if result.E_ex_mean is not None else np.nan,
        'eta_cumulative': result.eta_cumulative if result.eta_cumulative is not None else np.nan,
        'r_min': result.r_min if result.r_min is not None else np.nan,
        'penrose_fraction': result.penrose_fraction if result.penrose_fraction is not None else 0.0,
    }


def run_batch_parallel(param_list: List[Dict], n_workers: int = 8) -> List[Dict]:
    """
    Run batch of configurations in parallel.
    """
    results = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_single_config, p): i 
                   for i, p in enumerate(param_list)}
        
        for future in as_completed(futures):
            try:
                result = future.result(timeout=60)
                results.append(result)
            except Exception as e:
                # Record failure
                idx = futures[future]
                results.append({
                    **param_list[idx],
                    'outcome': 'EXCEPTION',
                    'is_escape': False,
                    'is_penrose': False,
                    'error': str(e),
                })
    
    return results


# =============================================================================
# PHASE 1: BROAD PARAMETER SPACE SWEEP
# =============================================================================

def run_broad_sweep(config: ComprehensiveSweepConfig, verbose: bool = True) -> Dict:
    """
    Phase 1: Broad sweep over full (E, Lz) space for each spin.
    
    Goal: Establish baseline success rates and identify viable regions.
    """
    if verbose:
        print("\n" + "="*80)
        print("PHASE 1: BROAD PARAMETER SPACE SWEEP")
        print("="*80)
        print(f"  Spins: {config.spins}")
        print(f"  E range: {config.E_range_broad}")
        print(f"  Lz range: {config.Lz_range_broad}")
        print(f"  Grid: {config.n_E_broad} x {config.n_Lz_broad} = {config.n_broad_total} points per spin")
        print(f"  Using default v_e=0.95, delta_m=0.20")
    
    results = {}
    
    E_vals = np.linspace(config.E_range_broad[0], config.E_range_broad[1], config.n_E_broad)
    Lz_vals = np.linspace(config.Lz_range_broad[0], config.Lz_range_broad[1], config.n_Lz_broad)
    
    for a in config.spins:
        if verbose:
            print(f"\n  Processing a/M = {a}...")
        
        t_start = time.time()
        
        # Build parameter list
        param_list = []
        for E0 in E_vals:
            for Lz0 in Lz_vals:
                param_list.append({
                    'a': a,
                    'E0': E0,
                    'Lz0': Lz0,
                    'v_e': 0.95,  # Default
                    'delta_m': 0.20,  # Default
                })
        
        # Run in parallel
        batch_results = run_batch_parallel(param_list, config.n_workers)
        
        # Compute statistics
        n_total = len(batch_results)
        n_escape = sum(1 for r in batch_results if r['is_escape'])
        n_penrose = sum(1 for r in batch_results if r['is_penrose'])
        
        # Clopper-Pearson CIs
        p_escape = n_escape / n_total
        ci_escape = clopper_pearson_ci(n_escape, n_total, config.alpha)
        
        p_penrose = n_penrose / n_total
        ci_penrose = clopper_pearson_ci(n_penrose, n_total, config.alpha)
        
        # Energy statistics for escaped trajectories
        escaped_results = [r for r in batch_results if r['is_escape'] and np.isfinite(r['Delta_E'])]
        
        if len(escaped_results) > 0:
            Delta_E_vals = np.array([r['Delta_E'] for r in escaped_results])
            Delta_E_mean = np.mean(Delta_E_vals)
            Delta_E_std = np.std(Delta_E_vals)
            
            if len(escaped_results) >= 10:
                ci_Delta_E = bca_bootstrap_ci(Delta_E_vals, np.mean, 
                                               config.n_bootstrap, config.alpha, config.seed)
            else:
                ci_Delta_E = (Delta_E_mean - 2*Delta_E_std, Delta_E_mean + 2*Delta_E_std)
        else:
            Delta_E_mean = np.nan
            Delta_E_std = np.nan
            ci_Delta_E = (np.nan, np.nan)
        
        t_elapsed = time.time() - t_start
        
        results[f'a={a}'] = {
            'spin': a,
            'n_total': n_total,
            'n_escape': n_escape,
            'n_penrose': n_penrose,
            'p_escape': p_escape,
            'ci_escape': ci_escape,
            'p_penrose': p_penrose,
            'ci_penrose': ci_penrose,
            'Delta_E_mean': Delta_E_mean,
            'Delta_E_std': Delta_E_std,
            'ci_Delta_E': ci_Delta_E,
            'duration_s': t_elapsed,
            'raw_results': batch_results,
        }
        
        if verbose:
            print(f"    Escape: {n_escape}/{n_total} = {100*p_escape:.2f}% "
                  f"[{100*ci_escape[0]:.2f}%, {100*ci_escape[1]:.2f}%]")
            print(f"    Penrose: {n_penrose}/{n_total} = {100*p_penrose:.2f}% "
                  f"[{100*ci_penrose[0]:.2f}%, {100*ci_penrose[1]:.2f}%]")
            print(f"    DeltaE mean: {Delta_E_mean:+.4f} +/- {Delta_E_std:.4f}")
            print(f"    Time: {t_elapsed:.1f}s")
    
    return results


# =============================================================================
# PHASE 2: FOCUSED SWEET SPOT SWEEP
# =============================================================================

def run_focused_sweep(config: ComprehensiveSweepConfig, verbose: bool = True) -> Dict:
    """
    Phase 2: High-resolution sweep in the identified sweet spot region.
    
    Goal: Precise success rate estimates in viable region.
    """
    if verbose:
        print("\n" + "="*80)
        print("PHASE 2: FOCUSED SWEET SPOT SWEEP")
        print("="*80)
        print(f"  E range: {config.E_range_focused}")
        print(f"  Lz range: {config.Lz_range_focused}")
        print(f"  Grid: {config.n_E_focused} x {config.n_Lz_focused} = {config.n_focused_total} points per spin")
    
    results = {}
    
    E_vals = np.linspace(config.E_range_focused[0], config.E_range_focused[1], config.n_E_focused)
    Lz_vals = np.linspace(config.Lz_range_focused[0], config.Lz_range_focused[1], config.n_Lz_focused)
    
    for a in config.spins:
        if verbose:
            print(f"\n  Processing a/M = {a}...")
        
        t_start = time.time()
        
        # Build parameter list
        param_list = []
        for E0 in E_vals:
            for Lz0 in Lz_vals:
                param_list.append({
                    'a': a,
                    'E0': E0,
                    'Lz0': Lz0,
                    'v_e': 0.95,
                    'delta_m': 0.20,
                })
        
        batch_results = run_batch_parallel(param_list, config.n_workers)
        
        # Statistics
        n_total = len(batch_results)
        n_escape = sum(1 for r in batch_results if r['is_escape'])
        n_penrose = sum(1 for r in batch_results if r['is_penrose'])
        
        p_escape = n_escape / n_total
        ci_escape = clopper_pearson_ci(n_escape, n_total, config.alpha)
        
        p_penrose = n_penrose / n_total
        ci_penrose = clopper_pearson_ci(n_penrose, n_total, config.alpha)
        
        t_elapsed = time.time() - t_start
        
        results[f'a={a}'] = {
            'spin': a,
            'n_total': n_total,
            'n_escape': n_escape,
            'n_penrose': n_penrose,
            'p_escape': p_escape,
            'ci_escape': ci_escape,
            'p_penrose': p_penrose,
            'ci_penrose': ci_penrose,
            'duration_s': t_elapsed,
            'raw_results': batch_results,
        }
        
        if verbose:
            print(f"    Escape: {n_escape}/{n_total} = {100*p_escape:.2f}% "
                  f"[{100*ci_escape[0]:.2f}%, {100*ci_escape[1]:.2f}%]")
            print(f"    Penrose: {n_penrose}/{n_total} = {100*p_penrose:.2f}% "
                  f"[{100*ci_penrose[0]:.2f}%, {100*ci_penrose[1]:.2f}%]")
            print(f"    Time: {t_elapsed:.1f}s")
    
    return results


# =============================================================================
# PHASE 3: THRUST PARAMETER VARIATION
# =============================================================================

def run_thrust_parameter_sweep(config: ComprehensiveSweepConfig, 
                                verbose: bool = True) -> Dict:
    """
    Phase 3: Vary v_e and delta_m at fixed sweet spot (E0, Lz0).
    
    Goal: Quantify sensitivity to thrust parameters.
    """
    if verbose:
        print("\n" + "="*80)
        print("PHASE 3: THRUST PARAMETER VARIATION")
        print("="*80)
        print(f"  v_e values: {config.v_e_values}")
        print(f"  delta_m values: {config.delta_m_values}")
    
    results = {}
    
    # Use sweet spot initial conditions
    E0_sweet = 1.22
    Lz0_sweet = 3.05
    
    # Focus on high spin
    test_spins = [0.95, 0.99]
    
    for a in test_spins:
        if verbose:
            print(f"\n  Processing a/M = {a} at sweet spot E0={E0_sweet}, Lz0={Lz0_sweet}...")
        
        spin_results = {}
        
        for v_e in config.v_e_values:
            for delta_m in config.delta_m_values:
                t_start = time.time()
                
                # Run Monte Carlo ensemble
                param_list = []
                rng = np.random.default_rng(config.seed)
                
                for _ in range(config.n_samples_per_config):
                    # Small perturbations around sweet spot
                    E0 = E0_sweet + rng.normal(0, 0.02)
                    Lz0 = Lz0_sweet + rng.normal(0, 0.05)
                    param_list.append({
                        'a': a,
                        'E0': E0,
                        'Lz0': Lz0,
                        'v_e': v_e,
                        'delta_m': delta_m,
                    })
                
                batch_results = run_batch_parallel(param_list, config.n_workers)
                
                n_total = len(batch_results)
                n_escape = sum(1 for r in batch_results if r['is_escape'])
                n_penrose = sum(1 for r in batch_results if r['is_penrose'])
                
                p_penrose = n_penrose / n_total
                ci_penrose = clopper_pearson_ci(n_penrose, n_total, config.alpha)
                
                # Efficiency for Penrose successes
                penrose_results = [r for r in batch_results 
                                   if r['is_penrose'] and np.isfinite(r['eta_cumulative'])]
                
                if len(penrose_results) > 0:
                    eta_vals = np.array([r['eta_cumulative'] for r in penrose_results])
                    eta_mean = np.mean(eta_vals)
                else:
                    eta_mean = 0.0
                
                t_elapsed = time.time() - t_start
                
                key = f'v_e={v_e}_dm={delta_m}'
                spin_results[key] = {
                    'v_e': v_e,
                    'delta_m': delta_m,
                    'n_total': n_total,
                    'n_penrose': n_penrose,
                    'p_penrose': p_penrose,
                    'ci_penrose': ci_penrose,
                    'eta_mean': eta_mean,
                    'duration_s': t_elapsed,
                }
                
                if verbose:
                    print(f"    v_e={v_e:.2f}, deltam={delta_m:.2f}: "
                          f"Penrose {100*p_penrose:.1f}% [{100*ci_penrose[0]:.1f}%, {100*ci_penrose[1]:.1f}%], "
                          f"eta={100*eta_mean:.1f}%")
        
        results[f'a={a}'] = spin_results
    
    return results


# =============================================================================
# PHASE 4: MONTE CARLO ENSEMBLE ANALYSIS
# =============================================================================

def run_monte_carlo_ensemble(config: ComprehensiveSweepConfig,
                              verbose: bool = True) -> Dict:
    """
    Phase 4: Large-scale Monte Carlo for robust statistics.
    
    Goal: Achieve narrow confidence intervals through large sample sizes.
    """
    if verbose:
        print("\n" + "="*80)
        print("PHASE 4: MONTE CARLO ENSEMBLE ANALYSIS")
        print("="*80)
        print(f"  Samples per configuration: {config.n_samples_per_config}")
    
    results = {}
    
    # Test both broad and focused regions
    regions = [
        ('broad', config.E_range_broad, config.Lz_range_broad),
        ('focused', config.E_range_focused, config.Lz_range_focused),
    ]
    
    for a in [0.9, 0.95, 0.99]:
        if verbose:
            print(f"\n  Processing a/M = {a}...")
        
        spin_results = {}
        
        for region_name, E_range, Lz_range in regions:
            t_start = time.time()
            
            # Latin Hypercube Sampling for efficient coverage
            bounds = {
                'E0': E_range,
                'Lz0': Lz_range,
            }
            
            samples = latin_hypercube_sample(
                config.n_samples_per_config, 
                bounds, 
                seed=config.seed
            )
            
            param_list = []
            for s in samples:
                param_list.append({
                    'a': a,
                    'E0': s['E0'],
                    'Lz0': s['Lz0'],
                    'v_e': 0.95,
                    'delta_m': 0.20,
                })
            
            batch_results = run_batch_parallel(param_list, config.n_workers)
            
            n_total = len(batch_results)
            n_escape = sum(1 for r in batch_results if r['is_escape'])
            n_penrose = sum(1 for r in batch_results if r['is_penrose'])
            
            p_escape = n_escape / n_total
            ci_escape = clopper_pearson_ci(n_escape, n_total, config.alpha)
            
            p_penrose = n_penrose / n_total
            ci_penrose = clopper_pearson_ci(n_penrose, n_total, config.alpha)
            
            # Energy statistics
            penrose_results = [r for r in batch_results 
                               if r['is_penrose'] and np.isfinite(r['Delta_E'])]
            
            if len(penrose_results) >= 10:
                Delta_E_vals = np.array([r['Delta_E'] for r in penrose_results])
                Delta_E_mean = np.mean(Delta_E_vals)
                ci_Delta_E = bca_bootstrap_ci(Delta_E_vals, np.mean, 
                                               config.n_bootstrap, config.alpha, config.seed)
                
                eta_vals = np.array([r['eta_cumulative'] for r in penrose_results 
                                     if np.isfinite(r['eta_cumulative'])])
                if len(eta_vals) > 0:
                    eta_mean = np.mean(eta_vals)
                    ci_eta = bca_bootstrap_ci(eta_vals, np.mean, 
                                               config.n_bootstrap, config.alpha, config.seed)
                else:
                    eta_mean = 0.0
                    ci_eta = (0.0, 0.0)
            else:
                Delta_E_mean = np.nan
                ci_Delta_E = (np.nan, np.nan)
                eta_mean = 0.0
                ci_eta = (0.0, 0.0)
            
            t_elapsed = time.time() - t_start
            
            spin_results[region_name] = {
                'E_range': E_range,
                'Lz_range': Lz_range,
                'n_total': n_total,
                'n_escape': n_escape,
                'n_penrose': n_penrose,
                'p_escape': p_escape,
                'ci_escape': ci_escape,
                'p_penrose': p_penrose,
                'ci_penrose': ci_penrose,
                'Delta_E_mean': Delta_E_mean,
                'ci_Delta_E': ci_Delta_E,
                'eta_mean': eta_mean,
                'ci_eta': ci_eta,
                'duration_s': t_elapsed,
            }
            
            if verbose:
                print(f"    {region_name}: Escape {100*p_escape:.2f}%, "
                      f"Penrose {100*p_penrose:.2f}% [{100*ci_penrose[0]:.2f}%, {100*ci_penrose[1]:.2f}%]")
        
        results[f'a={a}'] = spin_results
    
    return results


# =============================================================================
# SUMMARY REPORT GENERATION
# =============================================================================

def generate_comprehensive_report(all_results: Dict, config: ComprehensiveSweepConfig,
                                    output_dir: Path) -> str:
    """Generate comprehensive summary report."""
    
    lines = [
        "="*80,
        "COMPREHENSIVE PENROSE PROCESS PARAMETER SWEEP",
        "="*80,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "CONFIGURATION:",
        f"  Spins tested: {config.spins}",
        f"  Broad E range: {config.E_range_broad}",
        f"  Broad Lz range: {config.Lz_range_broad}",
        f"  Broad grid: {config.n_E_broad} x {config.n_Lz_broad}",
        f"  Focused E range: {config.E_range_focused}",
        f"  Focused Lz range: {config.Lz_range_focused}",
        f"  Focused grid: {config.n_E_focused} x {config.n_Lz_focused}",
        f"  v_e values: {config.v_e_values}",
        f"  delta_m values: {config.delta_m_values}",
        f"  MC samples per config: {config.n_samples_per_config}",
        f"  Bootstrap samples: {config.n_bootstrap}",
        f"  Confidence level: {100*(1-config.alpha):.0f}%",
        "",
    ]
    
    # Phase 1 Results
    if 'broad' in all_results:
        lines.append("="*80)
        lines.append("PHASE 1: BROAD PARAMETER SWEEP RESULTS")
        lines.append("="*80)
        
        for key, data in all_results['broad'].items():
            a = data['spin']
            lines.append(f"\na/M = {a}:")
            lines.append(f"  Total grid points: {data['n_total']}")
            lines.append(f"  Escape rate: {100*data['p_escape']:.2f}% "
                        f"[{100*data['ci_escape'][0]:.2f}%, {100*data['ci_escape'][1]:.2f}%]")
            lines.append(f"  Penrose rate: {100*data['p_penrose']:.3f}% "
                        f"[{100*data['ci_penrose'][0]:.3f}%, {100*data['ci_penrose'][1]:.3f}%]")
            if np.isfinite(data['Delta_E_mean']):
                lines.append(f"  Mean DeltaE: {data['Delta_E_mean']:+.4f} +/- {data['Delta_E_std']:.4f}")
    
    # Phase 2 Results
    if 'focused' in all_results:
        lines.append("")
        lines.append("="*80)
        lines.append("PHASE 2: FOCUSED SWEET SPOT RESULTS")
        lines.append("="*80)
        
        for key, data in all_results['focused'].items():
            a = data['spin']
            lines.append(f"\na/M = {a}:")
            lines.append(f"  Total grid points: {data['n_total']}")
            lines.append(f"  Escape rate: {100*data['p_escape']:.2f}% "
                        f"[{100*data['ci_escape'][0]:.2f}%, {100*data['ci_escape'][1]:.2f}%]")
            lines.append(f"  Penrose rate: {100*data['p_penrose']:.2f}% "
                        f"[{100*data['ci_penrose'][0]:.2f}%, {100*data['ci_penrose'][1]:.2f}%]")
    
    # Phase 3 Results
    if 'thrust_params' in all_results:
        lines.append("")
        lines.append("="*80)
        lines.append("PHASE 3: THRUST PARAMETER SENSITIVITY")
        lines.append("="*80)
        
        for spin_key, spin_data in all_results['thrust_params'].items():
            lines.append(f"\n{spin_key}:")
            for param_key, data in spin_data.items():
                lines.append(f"  {param_key}: Penrose {100*data['p_penrose']:.1f}%, "
                            f"eta={100*data['eta_mean']:.1f}%")
    
    # Phase 4 Results
    if 'monte_carlo' in all_results:
        lines.append("")
        lines.append("="*80)
        lines.append("PHASE 4: MONTE CARLO ENSEMBLE STATISTICS")
        lines.append("="*80)
        
        for spin_key, spin_data in all_results['monte_carlo'].items():
            lines.append(f"\n{spin_key}:")
            for region, data in spin_data.items():
                lines.append(f"  {region} region:")
                lines.append(f"    N = {data['n_total']}, Penrose = {data['n_penrose']}")
                lines.append(f"    Rate: {100*data['p_penrose']:.3f}% "
                            f"[{100*data['ci_penrose'][0]:.3f}%, {100*data['ci_penrose'][1]:.3f}%]")
                if np.isfinite(data['Delta_E_mean']):
                    lines.append(f"    DeltaE: {data['Delta_E_mean']:+.4f} "
                                f"[{data['ci_Delta_E'][0]:+.4f}, {data['ci_Delta_E'][1]:+.4f}]")
                lines.append(f"    eta: {100*data['eta_mean']:.2f}% "
                            f"[{100*data['ci_eta'][0]:.2f}%, {100*data['ci_eta'][1]:.2f}%]")
    
    # Key findings summary
    lines.append("")
    lines.append("="*80)
    lines.append("KEY FINDINGS SUMMARY")
    lines.append("="*80)
    
    # Extract key statistics for summary
    if 'monte_carlo' in all_results:
        mc = all_results['monte_carlo']
        
        # Broad region rates
        broad_rates = []
        focused_rates = []
        
        for spin_key, spin_data in mc.items():
            if 'broad' in spin_data:
                broad_rates.append((spin_key, spin_data['broad']['p_penrose'], 
                                   spin_data['broad']['ci_penrose']))
            if 'focused' in spin_data:
                focused_rates.append((spin_key, spin_data['focused']['p_penrose'],
                                     spin_data['focused']['ci_penrose']))
        
        lines.append("\nBroad region success rates:")
        for spin, rate, ci in broad_rates:
            lines.append(f"  {spin}: {100*rate:.3f}% [{100*ci[0]:.3f}%, {100*ci[1]:.3f}%]")
        
        lines.append("\nSweet spot region success rates:")
        for spin, rate, ci in focused_rates:
            lines.append(f"  {spin}: {100*rate:.2f}% [{100*ci[0]:.2f}%, {100*ci[1]:.2f}%]")
    
    lines.append("")
    lines.append("="*80)
    
    report = "\n".join(lines)
    
    # Save report
    report_path = output_dir / "comprehensive_sweep_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    return report


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_comprehensive_sweep(config: Optional[ComprehensiveSweepConfig] = None,
                             run_phases: Tuple[int, ...] = (1, 2, 3, 4),
                             verbose: bool = True) -> Dict:
    """
    Run the full comprehensive parameter sweep.
    
    Parameters
    ----------
    config : ComprehensiveSweepConfig, optional
        Sweep configuration (uses defaults if None)
    run_phases : tuple
        Which phases to run (1=broad, 2=focused, 3=thrust, 4=monte carlo)
    verbose : bool
        Print progress
    
    Returns
    -------
    dict
        All results from all phases
    """
    if config is None:
        config = ComprehensiveSweepConfig()
    
    output_dir = Path(config.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = output_dir / f"sweep_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("\n" + "="*80)
        print("COMPREHENSIVE PENROSE PROCESS PARAMETER SWEEP")
        print("="*80)
        print(f"Output directory: {output_dir}")
        print(f"Running phases: {run_phases}")
        print(f"Workers: {config.n_workers}")
    
    all_results = {}
    t_total_start = time.time()
    
    # Phase 1: Broad sweep
    if 1 in run_phases:
        all_results['broad'] = run_broad_sweep(config, verbose)
    
    # Phase 2: Focused sweep
    if 2 in run_phases:
        all_results['focused'] = run_focused_sweep(config, verbose)
    
    # Phase 3: Thrust parameters
    if 3 in run_phases:
        all_results['thrust_params'] = run_thrust_parameter_sweep(config, verbose)
    
    # Phase 4: Monte Carlo
    if 4 in run_phases:
        all_results['monte_carlo'] = run_monte_carlo_ensemble(config, verbose)
    
    t_total = time.time() - t_total_start
    
    if verbose:
        print("\n" + "="*80)
        print(f"TOTAL RUNTIME: {t_total/60:.1f} minutes")
        print("="*80)
    
    # Generate report
    report = generate_comprehensive_report(all_results, config, output_dir)
    if verbose:
        print("\n" + report)
    
    # Save config
    config_path = output_dir / "sweep_config.json"
    config_dict = {
        'spins': list(config.spins),
        'E_range_broad': list(config.E_range_broad),
        'Lz_range_broad': list(config.Lz_range_broad),
        'n_E_broad': config.n_E_broad,
        'n_Lz_broad': config.n_Lz_broad,
        'E_range_focused': list(config.E_range_focused),
        'Lz_range_focused': list(config.Lz_range_focused),
        'n_E_focused': config.n_E_focused,
        'n_Lz_focused': config.n_Lz_focused,
        'v_e_values': list(config.v_e_values),
        'delta_m_values': list(config.delta_m_values),
        'n_samples_per_config': config.n_samples_per_config,
        'n_bootstrap': config.n_bootstrap,
        'alpha': config.alpha,
        'seed': config.seed,
        'timestamp': timestamp,
        'total_runtime_s': t_total,
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Save raw results (summary only, not full raw_results to save space)
    summary_results = {}
    for phase, phase_data in all_results.items():
        summary_results[phase] = {}
        for key, data in phase_data.items():
            if isinstance(data, dict):
                # Remove raw_results to save space
                summary_results[phase][key] = {k: v for k, v in data.items() 
                                                if k != 'raw_results'}
            else:
                summary_results[phase][key] = data
    
    results_path = output_dir / "sweep_results.json"
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(x) for x in obj]
        return obj
    
    with open(results_path, 'w') as f:
        json.dump(convert_numpy(summary_results), f, indent=2)
    
    if verbose:
        print(f"\nResults saved to: {output_dir}")
    
    return all_results


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Penrose Process Parameter Sweep')
    parser.add_argument('--quick', action='store_true', 
                        help='Quick test run with reduced grid sizes')
    parser.add_argument('--full', action='store_true',
                        help='Full production run with large samples')
    parser.add_argument('--phases', type=int, nargs='+', default=[1, 2, 3, 4],
                        help='Which phases to run (1-4)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Configure based on mode
    if args.quick:
        config = ComprehensiveSweepConfig(
            spins=(0.95,),
            n_E_broad=20,
            n_Lz_broad=20,
            n_E_focused=15,
            n_Lz_focused=15,
            n_samples_per_config=200,
            n_bootstrap=2000,
        )
    elif args.full:
        config = ComprehensiveSweepConfig(
            spins=(0.5, 0.7, 0.9, 0.95, 0.99),
            n_E_broad=80,
            n_Lz_broad=80,
            n_E_focused=60,
            n_Lz_focused=60,
            n_samples_per_config=5000,
            n_bootstrap=20000,
        )
    else:
        config = ComprehensiveSweepConfig()
    
    # Override workers and output if specified
    if args.workers:
        config.n_workers = args.workers
    else:
        config.n_workers = max(1, multiprocessing.cpu_count() - 2)
    
    if args.output:
        config.output_dir = args.output
    
    # Run sweep
    results = run_comprehensive_sweep(config, tuple(args.phases), verbose=True)
