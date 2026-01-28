"""
Parameter Sweep Infrastructure
==============================
Systematic exploration of parameter space for Penrose process trajectories.

This module provides tools for:
1. Grid-based parameter sweeps (E, Lz, a, v_e)
2. Parallel execution of trajectory simulations
3. Result aggregation and analysis

Further considerations implemented:
- Spin dependence: a  in  {0.7, 0.9, 0.95, 0.99} as primary set
- Modular design to allow non-equatorial extension (Phase 2)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import time
import json
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kerr_utils import (
    horizon_radius, ergosphere_radius, isco_radius,
    compute_extraction_limit_radius
)
from experiments.trajectory_classifier import (
    OrbitProfile, TrajectoryOutcome, ThrustStrategy,
    classify_orbit, OrbitProperties, TrajectoryResult
)


# =============================================================================
# DEFAULT PARAMETER RANGES
# =============================================================================

# Spin values (further consideration: focus on high-spin regime)
DEFAULT_SPINS = [0.7, 0.9, 0.95, 0.99]

# Energy range (E > 1 for unbound flyby orbits)
DEFAULT_ENERGY_RANGE = (1.05, 2.0)
DEFAULT_ENERGY_STEPS = 20

# Angular momentum range (positive = prograde)
DEFAULT_LZ_RANGE = (2.0, 5.0)
DEFAULT_LZ_STEPS = 20

# Exhaust velocity range
DEFAULT_VE_RANGE = (0.8, 0.98)
DEFAULT_VE_STEPS = 5


# =============================================================================
# PARAMETER GRID CONFIGURATION
# =============================================================================

@dataclass
class ParameterGrid:
    """
    Configuration for parameter space exploration.
    
    Supports both grid-based and custom sampling strategies.
    """
    # Primary parameters
    spins: List[float] = field(default_factory=lambda: DEFAULT_SPINS.copy())
    
    # Orbital parameters
    E_range: Tuple[float, float] = DEFAULT_ENERGY_RANGE
    E_steps: int = DEFAULT_ENERGY_STEPS
    
    Lz_range: Tuple[float, float] = DEFAULT_LZ_RANGE
    Lz_steps: int = DEFAULT_LZ_STEPS
    
    # Thrust parameters
    v_e_range: Tuple[float, float] = DEFAULT_VE_RANGE
    v_e_steps: int = DEFAULT_VE_STEPS
    
    # Fixed parameters
    M: float = 1.0
    r0: float = 10.0  # Initial radius
    m0: float = 1.0   # Initial mass
    
    # Filtering
    filter_plunge: bool = True      # Skip plunge orbits
    filter_forbidden: bool = True   # Skip forbidden configurations
    require_ergosphere: bool = False  # Only orbits with periapsis in ergosphere
    
    # Output
    output_dir: str = "results"
    
    def __post_init__(self):
        """Validate and setup."""
        assert self.E_range[0] < self.E_range[1], "Invalid E range"
        assert self.Lz_range[0] < self.Lz_range[1], "Invalid Lz range"
        assert all(0 < a < 1 for a in self.spins), "Spins must be in (0, 1)"
    
    @property
    def E_values(self) -> np.ndarray:
        return np.linspace(self.E_range[0], self.E_range[1], self.E_steps)
    
    @property
    def Lz_values(self) -> np.ndarray:
        return np.linspace(self.Lz_range[0], self.Lz_range[1], self.Lz_steps)
    
    @property
    def v_e_values(self) -> np.ndarray:
        return np.linspace(self.v_e_range[0], self.v_e_range[1], self.v_e_steps)
    
    def total_combinations(self, include_v_e: bool = False) -> int:
        """Total number of parameter combinations."""
        n = len(self.spins) * self.E_steps * self.Lz_steps
        if include_v_e:
            n *= self.v_e_steps
        return n
    
    def generate_combinations(self, include_v_e: bool = False) -> List[Dict[str, float]]:
        """Generate all parameter combinations."""
        combos = []
        
        if include_v_e:
            for a, E, Lz, v_e in product(self.spins, self.E_values, 
                                          self.Lz_values, self.v_e_values):
                combos.append({
                    'a': a, 'E0': E, 'Lz0': Lz, 'v_e': v_e,
                    'M': self.M, 'r0': self.r0, 'm0': self.m0
                })
        else:
            for a, E, Lz in product(self.spins, self.E_values, self.Lz_values):
                combos.append({
                    'a': a, 'E0': E, 'Lz0': Lz,
                    'M': self.M, 'r0': self.r0, 'm0': self.m0
                })
        
        return combos
    
    def filter_valid_orbits(self, combinations: List[Dict]) -> List[Dict]:
        """Filter out invalid orbit configurations."""
        valid = []
        for params in combinations:
            props = classify_orbit(
                params['E0'], params['Lz0'], 
                params['a'], params['M']
            )
            
            # Apply filters
            if self.filter_plunge and props.profile == OrbitProfile.PLUNGE:
                continue
            if self.filter_forbidden and props.profile == OrbitProfile.FORBIDDEN:
                continue
            if self.require_ergosphere and not props.in_ergosphere:
                continue
            
            # Add orbit properties to params
            params['orbit_profile'] = props.profile.name
            params['r_periapsis'] = props.r_periapsis
            params['in_extraction_zone'] = props.in_extraction_zone
            
            valid.append(params)
        
        return valid


# =============================================================================
# SWEEP RESULT CONTAINER
# =============================================================================

@dataclass
class SweepResult:
    """Container for parameter sweep results."""
    grid: ParameterGrid
    results: List[TrajectoryResult] = field(default_factory=list)
    
    # Timing
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # Statistics
    n_total: int = 0
    n_completed: int = 0
    n_failed: int = 0
    
    # Summary (computed)
    n_escape: int = 0
    n_capture: int = 0
    n_penrose: int = 0  # Genuine Penrose extraction (E_ex < 0 + escape)
    
    def compute_statistics(self):
        """Compute summary statistics from results."""
        self.n_completed = len(self.results)
        self.n_escape = sum(1 for r in self.results if r.outcome == TrajectoryOutcome.ESCAPE)
        self.n_capture = sum(1 for r in self.results if r.outcome == TrajectoryOutcome.CAPTURE)
        self.n_penrose = sum(1 for r in self.results 
                            if r.outcome == TrajectoryOutcome.ESCAPE and r.penrose_fraction > 0.5)
    
    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def to_dataframe(self):
        """Convert results to pandas DataFrame for analysis."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for DataFrame export")
        
        records = []
        for r in self.results:
            record = {
                **r.initial_conditions,
                'outcome': r.outcome.name,
                'r_final': r.r_final,
                'E_final': r.E_final,
                'm_final': r.m_final,
                'Delta_E': r.Delta_E,
                'Delta_m': r.Delta_m,
                'eta_cumulative': r.eta_cumulative,
                'eta_exhaust': r.eta_exhaust,
                'E_ex_mean': r.E_ex_mean,
                'E_ex_min': r.E_ex_min,
                'penrose_fraction': r.penrose_fraction,
                'r_min': r.r_min,
                'r_max': r.r_max,
                'orbit_profile': r.orbit_profile.name,
                'strategy': r.strategy.name,
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def save(self, filepath: str):
        """Save results to JSON file."""
        data = {
            'grid': {
                'spins': self.grid.spins,
                'E_range': list(self.grid.E_range),
                'E_steps': self.grid.E_steps,
                'Lz_range': list(self.grid.Lz_range),
                'Lz_steps': self.grid.Lz_steps,
                'v_e_range': list(self.grid.v_e_range),
                'v_e_steps': self.grid.v_e_steps,
            },
            'statistics': {
                'n_total': self.n_total,
                'n_completed': self.n_completed,
                'n_failed': self.n_failed,
                'n_escape': self.n_escape,
                'n_capture': self.n_capture,
                'n_penrose': self.n_penrose,
                'duration_seconds': self.duration,
            },
            'results': [
                {
                    'initial_conditions': r.initial_conditions,
                    'outcome': r.outcome.name,
                    'Delta_E': r.Delta_E,
                    'eta_cumulative': r.eta_cumulative,
                    'penrose_fraction': r.penrose_fraction,
                    'r_min': r.r_min,
                    'orbit_profile': r.orbit_profile.name,
                    'strategy': r.strategy.name,
                }
                for r in self.results
            ]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# SWEEP EXECUTION
# =============================================================================

def run_single_simulation(params: Dict, 
                          strategy: ThrustStrategy = ThrustStrategy.SINGLE_IMPULSE,
                          simulation_func: Optional[Callable] = None) -> TrajectoryResult:
    """
    Run a single trajectory simulation with given parameters.
    
    Parameters
    ----------
    params : dict
        Dictionary with keys: a, E0, Lz0, v_e, M, r0, m0
    strategy : ThrustStrategy
        Thrust strategy to use
    simulation_func : callable, optional
        Custom simulation function. If None, uses geodesic classification only.
        
    Returns
    -------
    TrajectoryResult
        Complete result of the simulation
    """
    result = TrajectoryResult(
        initial_conditions=params.copy(),
        strategy=strategy
    )
    
    # Classify orbit
    props = classify_orbit(
        params['E0'], params['Lz0'], 
        params['a'], params['M']
    )
    result.orbit_profile = props.profile
    
    if props.r_periapsis:
        result.r_min = props.r_periapsis
    
    # If custom simulation function provided, run it
    if simulation_func is not None:
        try:
            sim_result = simulation_func(params, strategy)
            # Merge simulation results
            for key, value in sim_result.items():
                if hasattr(result, key):
                    setattr(result, key, value)
        except Exception as e:
            result.outcome = TrajectoryOutcome.INTEGRATION_FAILURE
            print(f"Simulation failed for {params}: {e}")
    else:
        # Without thrust simulation, estimate outcome from orbit classification
        if props.profile == OrbitProfile.PLUNGE:
            result.outcome = TrajectoryOutcome.CAPTURE
        elif props.profile == OrbitProfile.FORBIDDEN:
            result.outcome = TrajectoryOutcome.INTEGRATION_FAILURE
        elif props.profile in [OrbitProfile.FLYBY_DEEP_ERGOSPHERE, 
                               OrbitProfile.FLYBY_SHALLOW_ERGOSPHERE,
                               OrbitProfile.FLYBY_OUTSIDE_ERGOSPHERE]:
            result.outcome = TrajectoryOutcome.ESCAPE
        else:
            result.outcome = TrajectoryOutcome.BOUND
    
    return result


def run_parameter_sweep(grid: ParameterGrid,
                        strategy: ThrustStrategy = ThrustStrategy.SINGLE_IMPULSE,
                        simulation_func: Optional[Callable] = None,
                        n_workers: int = 1,
                        progress_callback: Optional[Callable] = None,
                        include_v_e: bool = False) -> SweepResult:
    """
    Run parameter sweep over the specified grid.
    
    Parameters
    ----------
    grid : ParameterGrid
        Parameter space configuration
    strategy : ThrustStrategy
        Thrust strategy to apply
    simulation_func : callable, optional
        Custom simulation function for each point
    n_workers : int
        Number of parallel workers (1 = sequential)
    progress_callback : callable, optional
        Called with (n_completed, n_total) for progress updates
    include_v_e : bool
        Whether to include exhaust velocity in sweep
        
    Returns
    -------
    SweepResult
        Complete sweep results
    """
    sweep_result = SweepResult(grid=grid)
    sweep_result.start_time = time.time()
    
    # Generate and filter parameter combinations
    all_combos = grid.generate_combinations(include_v_e=include_v_e)
    valid_combos = grid.filter_valid_orbits(all_combos)
    
    sweep_result.n_total = len(valid_combos)
    
    print(f"Parameter sweep: {len(all_combos)} total -> {len(valid_combos)} valid combinations")
    print(f"Strategy: {strategy.name}")
    
    if n_workers == 1:
        # Sequential execution
        for i, params in enumerate(valid_combos):
            result = run_single_simulation(params, strategy, simulation_func)
            sweep_result.results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(valid_combos))
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(run_single_simulation, params, strategy, simulation_func): params
                for params in valid_combos
            }
            
            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    sweep_result.results.append(result)
                except Exception as e:
                    sweep_result.n_failed += 1
                    print(f"Worker failed: {e}")
                
                if progress_callback:
                    progress_callback(i + 1, len(valid_combos))
    
    sweep_result.end_time = time.time()
    sweep_result.compute_statistics()
    
    return sweep_result


# =============================================================================
# QUICK ORBIT CLASSIFICATION SWEEP (no thrust simulation)
# =============================================================================

def quick_orbit_sweep(spins: Optional[List[float]] = None,
                      E_range: Tuple[float, float] = (1.0, 2.5),
                      Lz_range: Tuple[float, float] = (1.0, 6.0),
                      n_E: int = 50,
                      n_Lz: int = 50) -> Dict[str, np.ndarray]:
    """
    Quick sweep to classify orbit types across (E, Lz) space.
    
    Returns arrays suitable for heatmap visualization.
    """
    if spins is None:
        spins = DEFAULT_SPINS
    
    E_vals = np.linspace(E_range[0], E_range[1], n_E)
    Lz_vals = np.linspace(Lz_range[0], Lz_range[1], n_Lz)
    
    results = {}
    
    for a in spins:
        # Create classification grid
        profile_grid = np.zeros((n_E, n_Lz), dtype=int)
        r_peri_grid = np.full((n_E, n_Lz), np.nan)
        
        for i, E in enumerate(E_vals):
            for j, Lz in enumerate(Lz_vals):
                props = classify_orbit(E, Lz, a, 1.0)
                profile_grid[i, j] = props.profile.value
                if props.r_periapsis is not None:
                    r_peri_grid[i, j] = props.r_periapsis
        
        results[f'a={a}'] = {
            'E': E_vals,
            'Lz': Lz_vals,
            'profile': profile_grid,
            'r_periapsis': r_peri_grid,
        }
    
    return results


# =============================================================================
# EXTRACTION ZONE ANALYSIS
# =============================================================================

def analyze_extraction_zones(spins: Optional[List[float]] = None,
                              v_e_values: Optional[List[float]] = None,
                              E_range: Tuple[float, float] = (1.0, 2.0),
                              Lz_range: Tuple[float, float] = (2.0, 5.0),
                              n_samples: int = 20) -> Dict[str, Any]:
    """
    Analyze how extraction zones vary with spin and exhaust velocity.
    
    Returns extraction limit radii and periapsis requirements.
    """
    if spins is None:
        spins = DEFAULT_SPINS
    if v_e_values is None:
        v_e_values = [0.8, 0.9, 0.95, 0.98]
    
    results = {
        'spins': spins,
        'v_e_values': v_e_values,
        'extraction_limits': {},
        'viable_configs': {},
    }
    
    for a in spins:
        r_plus = horizon_radius(a)
        r_erg = ergosphere_radius(np.pi/2, a)
        
        for v_e in v_e_values:
            key = f'a={a}_ve={v_e}'
            
            # Find viable (E, Lz) configurations with deep ergosphere periapsis
            viable = []
            E_vals = np.linspace(E_range[0], E_range[1], n_samples)
            Lz_vals = np.linspace(Lz_range[0], Lz_range[1], n_samples)
            
            for E in E_vals:
                for Lz in Lz_vals:
                    props = classify_orbit(E, Lz, a)
                    if props.in_extraction_zone:
                        # Compute extraction limit for this configuration
                        R_ex = compute_extraction_limit_radius(E, Lz, 1.0, v_e, a)
                        if R_ex is not None:
                            viable.append({
                                'E': E, 'Lz': Lz,
                                'r_periapsis': props.r_periapsis,
                                'R_extraction_limit': R_ex,
                            })
            
            results['viable_configs'][key] = viable
    
    return results


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("PARAMETER SWEEP INFRASTRUCTURE - Quick Test")
    print("="*70)
    
    # Test 1: Quick orbit sweep
    print("\n1. Quick orbit classification sweep...")
    orbit_results = quick_orbit_sweep(
        spins=[0.95],
        E_range=(1.0, 2.0),
        Lz_range=(2.0, 5.0),
        n_E=20,
        n_Lz=20
    )
    
    key = 'a=0.95'
    profiles = orbit_results[key]['profile']
    
    # Count profile types
    from collections import Counter
    profile_counts = Counter(profiles.flatten())
    
    print(f"  Spin a = 0.95:")
    for profile_val, count in profile_counts.items():
        profile_name = OrbitProfile(profile_val).name
        print(f"    {profile_name}: {count} configurations")
    
    # Test 2: Parameter grid
    print("\n2. Parameter grid configuration...")
    grid = ParameterGrid(
        spins=[0.95],
        E_range=(1.1, 1.5),
        E_steps=5,
        Lz_range=(2.5, 3.5),
        Lz_steps=5,
        require_ergosphere=True
    )
    
    combos = grid.generate_combinations()
    valid = grid.filter_valid_orbits(combos)
    
    print(f"  Total combinations: {len(combos)}")
    print(f"  Valid (after filtering): {len(valid)}")
    
    if valid:
        print(f"  Sample configuration: {valid[0]}")
    
    # Test 3: Run sweep (classification only)
    print("\n3. Running classification sweep...")
    sweep_result = run_parameter_sweep(
        grid, 
        strategy=ThrustStrategy.NONE,
        n_workers=1
    )
    
    print(f"  Completed: {sweep_result.n_completed}")
    print(f"  Escape: {sweep_result.n_escape}")
    print(f"  Capture: {sweep_result.n_capture}")
    print(f"  Duration: {sweep_result.duration:.2f}s")
    
    # Test 4: Extraction zone analysis
    print("\n4. Extraction zone analysis...")
    extraction = analyze_extraction_zones(
        spins=[0.95],
        v_e_values=[0.95],
        n_samples=10
    )
    
    key = 'a=0.95_ve=0.95'
    if key in extraction['viable_configs']:
        configs = extraction['viable_configs'][key]
        print(f"  Viable configs with E_ex < 0 possible: {len(configs)}")
        if configs:
            print(f"  Example: E={configs[0]['E']:.2f}, Lz={configs[0]['Lz']:.2f}, "
                  f"r_peri={configs[0]['r_periapsis']:.4f}")
    
    print("\n" + "="*70)
    print("Parameter sweep infrastructure ready!")
    print("="*70)
