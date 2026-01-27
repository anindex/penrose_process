"""
Penrose Process Experiment Pipeline
====================================
Systematic study of trajectory profiles, parameter dependencies, and
extraction efficiency in the Penrose process.

Modules:
--------
- trajectory_classifier: Orbit classification and profile analysis
- parameter_sweep: Systematic parameter space exploration
- phase_space: Phase-space visualization tools
- thrust_comparison: Compare single vs continuous thrust strategies
- ensemble: Monte Carlo ensemble runs
- benchmark: Data export and reproducibility tools

Usage:
------
Quick start with command line:
    python experiments/run_trajectory_study.py --mode quick

Or import modules directly:
    from experiments import classify_orbit, OrbitProfile
    from experiments.phase_space import plot_orbit_profile_map
    from experiments.thrust_comparison import compare_strategies

Further Considerations (implemented):
-------------------------------------
1. Spin dependence: SPIN_VALUES = [0.7, 0.9, 0.95, 0.99] as primary set
2. Non-equatorial: Code structure allows extension (Phase 2 deferred)
3. Output formats: Both scripts (batch runs) and notebooks (exploration)
"""

# Trajectory classification
from .trajectory_classifier import (
    OrbitProfile,
    TrajectoryOutcome,
    ThrustStrategy,
    OrbitProperties,
    TrajectoryResult,
    classify_orbit,
    classify_trajectory_outcome,
    compute_orbit_properties,
    compute_effective_potential,
    find_turning_points,
    compute_key_radii_vs_spin,
    SPIN_VALUES,
)

# Parameter sweeps
from .parameter_sweep import (
    ParameterGrid,
    SweepResult,
    run_parameter_sweep,
    quick_orbit_sweep,
    analyze_extraction_zones,
    DEFAULT_SPINS,
)

# Phase space visualization
from .phase_space import (
    plot_effective_potential,
    plot_orbit_profile_map,
    plot_periapsis_depth_map,
    plot_spin_comparison,
    plot_phase_portrait,
    plot_trajectory_xy,
    create_trajectory_analysis_figure,
)

# Thrust comparison
from .thrust_comparison import (
    SimulationConfig,
    ComparisonResult,
    compare_strategies,
    scan_trigger_radii,
    simulate_geodesic,
    simulate_single_impulse,
    simulate_burst,
)

# Ensemble analysis
from .ensemble import (
    EnsembleConfig,
    EnsembleResult,
    run_ensemble,
    find_sweet_spot,
    compute_sensitivity,
    check_convergence,
    latin_hypercube_sample,
)

# Benchmark export
from .benchmark import (
    RunMetadata,
    generate_run_id,
    results_to_csv,
    sweep_to_csv,
    ensemble_to_csv,
    generate_summary_report,
    load_benchmark_run,
)

# Trajectory visualization
from .trajectory_visualization import (
    animate_single_thrust,
    animate_continuous_thrust,
    create_both_animations,
)

__all__ = [
    # Classification
    'OrbitProfile',
    'TrajectoryOutcome',
    'ThrustStrategy',
    'OrbitProperties',
    'TrajectoryResult',
    'classify_orbit',
    'classify_trajectory_outcome',
    'compute_orbit_properties',
    'compute_effective_potential',
    'find_turning_points',
    'compute_key_radii_vs_spin',
    'SPIN_VALUES',
    # Sweeps
    'ParameterGrid',
    'SweepResult',
    'run_parameter_sweep',
    'quick_orbit_sweep',
    'analyze_extraction_zones',
    'DEFAULT_SPINS',
    # Visualization
    'plot_effective_potential',
    'plot_orbit_profile_map',
    'plot_periapsis_depth_map',
    'plot_spin_comparison',
    'plot_phase_portrait',
    'plot_trajectory_xy',
    'create_trajectory_analysis_figure',
    # Thrust
    'SimulationConfig',
    'ComparisonResult',
    'compare_strategies',
    'scan_trigger_radii',
    'simulate_geodesic',
    'simulate_single_impulse',
    'simulate_burst',
    # Ensemble
    'EnsembleConfig',
    'EnsembleResult',
    'run_ensemble',
    'find_sweet_spot',
    'compute_sensitivity',
    'check_convergence',
    'latin_hypercube_sample',
    # Export
    'RunMetadata',
    'generate_run_id',
    'results_to_csv',
    'sweep_to_csv',
    'ensemble_to_csv',
    'generate_summary_report',
    'load_benchmark_run',
    # Visualization
    'animate_single_thrust',
    'animate_continuous_thrust',
    'create_both_animations',
]
