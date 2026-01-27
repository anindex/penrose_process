"""
Penrose Process Trajectory Profile Study
=========================================
Main runner script for the trajectory profile experiment pipeline.

This script orchestrates:
1. Parameter space exploration
2. Strategy comparison
3. Statistical ensemble analysis
4. Result export and visualization

Usage:
    python run_trajectory_study.py [--quick] [--full] [--spins 0.95 0.99]

Author: An T. Le
"""

import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kerr_utils import setup_prd_style

from experiments.trajectory_classifier import (
    OrbitProfile, TrajectoryOutcome, ThrustStrategy,
    classify_orbit, compute_key_radii_vs_spin, SPIN_VALUES
)
from experiments.parameter_sweep import (
    ParameterGrid, SweepResult, run_parameter_sweep, 
    quick_orbit_sweep, analyze_extraction_zones
)
from experiments.phase_space import (
    plot_orbit_profile_map, plot_periapsis_depth_map,
    plot_spin_comparison, plot_effective_potential_family,
    create_trajectory_analysis_figure
)
from experiments.thrust_comparison import (
    SimulationConfig, compare_strategies, scan_trigger_radii,
    simulate_single_impulse, simulate_geodesic
)
from experiments.ensemble import (
    EnsembleConfig, EnsembleResult, run_ensemble,
    find_sweet_spot, compute_sensitivity, check_convergence
)
from experiments.benchmark import (
    generate_run_id, sweep_to_csv, ensemble_to_csv,
    generate_summary_report, RunMetadata
)


# =============================================================================
# STUDY CONFIGURATIONS
# =============================================================================

# Note: The Penrose process success region is quite narrow in (E, Lz) space.
# Success requires orbits that:
# 1. Are marginally unbound (E slightly > 1)
# 2. Have prograde angular momentum (Lz ~ 3.0 for a=0.95)
# 3. Penetrate ergosphere but don't immediately capture
# Expect ~5-10% escape probability even with tuned parameters.

# Quick study: focused on known sweet spot for demonstration
QUICK_CONFIG = {
    'spins': [0.95],
    'E_range': (1.15, 1.35),
    'Lz_range': (2.9, 3.3),
    'n_E': 10,
    'n_Lz': 10,
    'n_ensemble': 100,  # More samples to hit sweet spot
    'n_workers': 1,
}

# Standard study: balanced coverage
# Focus on E ~ 1.0-1.5 (marginally unbound) and Lz ~ 2.5-4.0 (prograde flyby)
STANDARD_CONFIG = {
    'spins': [0.9, 0.95, 0.99],
    'E_range': (1.05, 1.5),
    'Lz_range': (2.5, 4.0),
    'n_E': 30,
    'n_Lz': 30,
    'n_ensemble': 200,
    'n_workers': 4,
}

# Full study: comprehensive analysis
# E in [1.0, 1.8] captures both marginally bound and unbound orbits
# Lz in [2.0, 5.0] covers prograde flyby regime
FULL_CONFIG = {
    'spins': [0.7, 0.9, 0.95, 0.99],
    'E_range': (1.0, 1.8),
    'Lz_range': (2.0, 5.0),
    'n_E': 50,
    'n_Lz': 50,
    'n_ensemble': 1000,
    'n_workers': 8,
}


# =============================================================================
# STUDY PHASES
# =============================================================================

def phase1_orbit_classification(config: dict, output_dir: Path, verbose: bool = True):
    """
    Phase 1: Classify orbit types across parameter space.
    
    Creates (E, Lz) maps for each spin value showing orbit profiles.
    """
    if verbose:
        print("\n" + "="*70)
        print("PHASE 1: ORBIT CLASSIFICATION")
        print("="*70)
    
    setup_prd_style()
    
    results = {}
    
    for a in config['spins']:
        if verbose:
            print(f"\n  Processing a = {a}...")
        
        # Quick orbit sweep
        orbit_data = quick_orbit_sweep(
            spins=[a],
            E_range=config['E_range'],
            Lz_range=config['Lz_range'],
            n_E=config['n_E'],
            n_Lz=config['n_Lz']
        )
        
        results[f'a={a}'] = orbit_data[f'a={a}']
        
        # Generate visualization
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        plot_orbit_profile_map(
            a=a, 
            E_range=config['E_range'],
            Lz_range=config['Lz_range'],
            n_E=config['n_E'],
            n_Lz=config['n_Lz'],
            ax=ax1
        )
        
        plot_periapsis_depth_map(
            a=a,
            E_range=config['E_range'],
            Lz_range=config['Lz_range'],
            n_E=config['n_E'],
            n_Lz=config['n_Lz'],
            ax=ax2
        )
        
        fig.suptitle(f'Orbit Classification: a/M = {a}', fontsize=12)
        plt.tight_layout()
        
        fig_path = output_dir / f'orbit_classification_a{a}.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"    Saved: {fig_path}")
    
    # Spin comparison figure
    if len(config['spins']) > 1:
        fig = plot_spin_comparison(
            spins=config['spins'],
            E_range=config['E_range'],
            Lz_range=config['Lz_range'],
            n_E=config['n_E'],
            n_Lz=config['n_Lz']
        )
        fig_path = output_dir / 'spin_comparison.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        if verbose:
            print(f"\n    Saved: {fig_path}")
    
    return results


def phase2_strategy_comparison(config: dict, output_dir: Path, verbose: bool = True):
    """
    Phase 2: Compare thrust strategies for selected configurations.
    """
    if verbose:
        print("\n" + "="*70)
        print("PHASE 2: THRUST STRATEGY COMPARISON")
        print("="*70)
    
    results = {}
    
    # Test configurations (known good parameters)
    test_configs = [
        {'a': 0.95, 'E0': 1.2, 'Lz0': 3.0, 'v_e': 0.95},
        {'a': 0.95, 'E0': 1.25, 'Lz0': 3.1, 'v_e': 0.95},
    ]
    
    for i, params in enumerate(test_configs):
        if params['a'] not in config['spins']:
            continue
        
        if verbose:
            print(f"\n  Config {i+1}: E={params['E0']}, Lz={params['Lz0']}, a={params['a']}")
        
        sim_config = SimulationConfig(
            a=params['a'],
            E0=params['E0'],
            Lz0=params['Lz0'],
            v_e=params['v_e'],
        )
        
        # Compare strategies
        comparison = compare_strategies(
            sim_config,
            strategies=[ThrustStrategy.NONE, ThrustStrategy.SINGLE_IMPULSE],
            verbose=verbose
        )
        
        results[f'config_{i}'] = comparison
        
        # Trigger radius scan
        if verbose:
            print(f"    Scanning trigger radii...")
        
        scan = scan_trigger_radii(sim_config, n_points=20)
        
        if scan['optimal_radius']:
            if verbose:
                print(f"    Optimal trigger: r = {scan['optimal_radius']:.4f}M")
                print(f"    Best DeltaE: {scan['optimal_Delta_E']:+.4f}")
        
        results[f'scan_{i}'] = scan
    
    return results


def phase3_ensemble_analysis(config: dict, output_dir: Path, verbose: bool = True):
    """
    Phase 3: Monte Carlo ensemble analysis.
    """
    if verbose:
        print("\n" + "="*70)
        print("PHASE 3: ENSEMBLE ANALYSIS")
        print("="*70)
    
    results = {}
    
    for a in config['spins']:
        if verbose:
            print(f"\n  Running ensemble for a = {a}...")
        
        base_config = SimulationConfig(
            a=a,
            E0=1.2,
            Lz0=3.0,
            v_e=0.95,
        )
        
        ensemble_config = EnsembleConfig(
            base_config=base_config,
            n_samples=config['n_ensemble'],
            sampling_method='lhs',
            E_bounds=config['E_range'],
            Lz_bounds=config['Lz_range'],
            strategy=ThrustStrategy.SINGLE_IMPULSE,
            seed=42,
            n_workers=config['n_workers'],
        )
        
        ensemble_result = run_ensemble(ensemble_config, verbose=False)
        
        if verbose:
            print(f"    Escape probability: {100*ensemble_result.escape_probability:.1f}%")
            print(f"    Penrose probability: {100*ensemble_result.penrose_probability:.1f}%")
            print(f"    Mean DeltaE: {ensemble_result.Delta_E_mean:+.4f}")
        
        results[f'a={a}'] = ensemble_result
        
        # Export
        run_id = ensemble_to_csv(
            ensemble_result, 
            str(output_dir / f'ensemble_a{a}'),
            run_id=f'ensemble_a{a}'
        )
        
        # Find sweet spot
        sweet_spot = find_sweet_spot(ensemble_result, n_top=5)
        if sweet_spot and verbose:
            print(f"    Sweet spot: E={sweet_spot[0]['E0']:.3f}, Lz={sweet_spot[0]['Lz0']:.3f}")
        
        # Sensitivity analysis
        sensitivity = compute_sensitivity(ensemble_result)
        if sensitivity and verbose:
            print(f"    Sensitivity: E0 corr={sensitivity.get('E0', 0):.3f}, "
                  f"Lz0 corr={sensitivity.get('Lz0', 0):.3f}")
    
    return results


def phase4_summary(all_results: dict, output_dir: Path, verbose: bool = True):
    """
    Phase 4: Generate summary and final report.
    """
    if verbose:
        print("\n" + "="*70)
        print("PHASE 4: SUMMARY AND REPORT")
        print("="*70)
    
    # Collect all trajectory results
    all_trajectories = []
    
    if 'ensemble' in all_results:
        for key, ensemble in all_results['ensemble'].items():
            if hasattr(ensemble, 'results'):
                all_trajectories.extend(ensemble.results)
    
    if all_trajectories:
        # Generate summary report
        report = generate_summary_report(
            all_trajectories,
            str(output_dir / 'summary_report.txt')
        )
        
        if verbose:
            print(report)
    
    # Save metadata
    metadata = RunMetadata(
        run_id=generate_run_id('trajectory_study'),
        timestamp=datetime.now().isoformat(),
        description='Trajectory profile study',
        n_samples=len(all_trajectories),
    )
    metadata.save(str(output_dir / 'study_metadata.json'))
    
    if verbose:
        print(f"\n  Results saved to: {output_dir}")


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_trajectory_study(mode: str = 'standard',
                          output_dir: Optional[str] = None,
                          spins: Optional[list] = None,
                          verbose: bool = True):
    """
    Run the complete trajectory profile study.
    
    Parameters
    ----------
    mode : str
        'quick', 'standard', or 'full'
    output_dir : str, optional
        Output directory. Default: results/trajectory_study_<timestamp>
    spins : list, optional
        Override spin values to study
    verbose : bool
        Print progress messages
    """
    # Select configuration
    if mode == 'quick':
        config = QUICK_CONFIG.copy()
    elif mode == 'full':
        config = FULL_CONFIG.copy()
    else:
        config = STANDARD_CONFIG.copy()
    
    # Override spins if provided
    if spins is not None:
        config['spins'] = spins
    
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/trajectory_study_{timestamp}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_path / 'study_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    if verbose:
        print("="*70)
        print("PENROSE PROCESS TRAJECTORY PROFILE STUDY")
        print("="*70)
        print(f"Mode: {mode}")
        print(f"Spins: {config['spins']}")
        print(f"E range: {config['E_range']}")
        print(f"Lz range: {config['Lz_range']}")
        print(f"Output: {output_path}")
    
    all_results = {}
    
    # Phase 1: Orbit classification
    all_results['classification'] = phase1_orbit_classification(
        config, output_path, verbose
    )
    
    # Phase 2: Strategy comparison
    all_results['strategies'] = phase2_strategy_comparison(
        config, output_path, verbose
    )
    
    # Phase 3: Ensemble analysis
    all_results['ensemble'] = phase3_ensemble_analysis(
        config, output_path, verbose
    )
    
    # Phase 4: Summary
    phase4_summary(all_results, output_path, verbose)
    
    if verbose:
        print("\n" + "="*70)
        print("STUDY COMPLETE")
        print("="*70)
    
    return all_results, output_path


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run Penrose process trajectory profile study'
    )
    parser.add_argument(
        '--mode', choices=['quick', 'standard', 'full'],
        default='quick',
        help='Study mode (quick/standard/full)'
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Output directory'
    )
    parser.add_argument(
        '--spins', type=float, nargs='+', default=None,
        help='Spin values to study'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress output'
    )
    
    args = parser.parse_args()
    
    run_trajectory_study(
        mode=args.mode,
        output_dir=args.output,
        spins=args.spins,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
