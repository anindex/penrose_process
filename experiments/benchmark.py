"""
Benchmark Data Export Module
=============================
Structured data export for reproducibility and comparison.

Supports:
- CSV export for simple analysis
- HDF5 export for large datasets with compression
- JSON metadata for configuration tracking
- Standardized naming conventions
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import json
import csv

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.trajectory_classifier import (
    OrbitProfile, TrajectoryOutcome, ThrustStrategy, TrajectoryResult
)
from experiments.parameter_sweep import SweepResult, ParameterGrid
from experiments.ensemble import EnsembleResult, EnsembleConfig


# =============================================================================
# NAMING CONVENTIONS
# =============================================================================

def generate_run_id(prefix: str = "penrose") -> str:
    """Generate unique run identifier."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def generate_filename(run_id: str, 
                       data_type: str,
                       extension: str = "csv") -> str:
    """Generate standardized filename."""
    return f"{run_id}_{data_type}.{extension}"


# =============================================================================
# METADATA TRACKING
# =============================================================================

@dataclass
class RunMetadata:
    """Metadata for a benchmark run."""
    run_id: str
    timestamp: str
    description: str = ""
    
    # Code version
    version: str = "1.0.0"
    git_commit: Optional[str] = None
    
    # Configuration summary
    n_samples: int = 0
    strategies: List[str] = None
    spin_values: List[float] = None
    E_range: List[float] = None
    Lz_range: List[float] = None
    
    # Timing
    duration_seconds: float = 0.0
    
    # Summary statistics
    n_escape: int = 0
    n_capture: int = 0
    n_penrose: int = 0
    
    def __post_init__(self):
        if self.strategies is None:
            self.strategies = []
        if self.spin_values is None:
            self.spin_values = []
        if self.E_range is None:
            self.E_range = []
        if self.Lz_range is None:
            self.Lz_range = []
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'RunMetadata':
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


# =============================================================================
# CSV EXPORT
# =============================================================================

def results_to_csv(results: List[TrajectoryResult], 
                    filepath: str,
                    include_trajectory: bool = False):
    """
    Export trajectory results to CSV.
    
    Parameters
    ----------
    results : list of TrajectoryResult
        Results to export
    filepath : str
        Output file path
    include_trajectory : bool
        If True, include sampled trajectory points (makes file much larger)
    """
    if not results:
        print("No results to export")
        return
    
    # Define columns
    columns = [
        'run_idx',
        # Initial conditions
        'a', 'E0', 'Lz0', 'r0', 'm0', 'v_e',
        # Orbit classification
        'orbit_profile', 'strategy',
        # Outcome
        'outcome', 'r_final', 'E_final', 'm_final',
        # Metrics
        'Delta_E', 'Delta_m', 'eta_cumulative', 'eta_exhaust',
        # Penrose diagnostics
        'E_ex_mean', 'E_ex_min', 'n_negative_E_ex', 'n_total_E_ex', 'penrose_fraction',
        # Trajectory stats
        'r_min', 'r_max', 'tau_total', 'n_steps',
    ]
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for idx, r in enumerate(results):
            row = {
                'run_idx': idx,
                # Initial conditions
                'a': r.initial_conditions.get('a', ''),
                'E0': r.initial_conditions.get('E0', ''),
                'Lz0': r.initial_conditions.get('Lz0', ''),
                'r0': r.initial_conditions.get('r0', ''),
                'm0': r.initial_conditions.get('m0', ''),
                'v_e': r.initial_conditions.get('v_e', ''),
                # Classification
                'orbit_profile': r.orbit_profile.name if r.orbit_profile else '',
                'strategy': r.strategy.name if r.strategy else '',
                # Outcome
                'outcome': r.outcome.name,
                'r_final': r.r_final,
                'E_final': r.E_final,
                'm_final': r.m_final,
                # Metrics
                'Delta_E': r.Delta_E,
                'Delta_m': r.Delta_m,
                'eta_cumulative': r.eta_cumulative,
                'eta_exhaust': r.eta_exhaust,
                # Penrose
                'E_ex_mean': r.E_ex_mean,
                'E_ex_min': r.E_ex_min,
                'n_negative_E_ex': r.n_negative_E_ex,
                'n_total_E_ex': r.n_total_E_ex,
                'penrose_fraction': r.penrose_fraction,
                # Trajectory stats
                'r_min': r.r_min,
                'r_max': r.r_max,
                'tau_total': r.tau_total,
                'n_steps': r.n_steps,
            }
            writer.writerow(row)
    
    print(f"Exported {len(results)} results to {filepath}")


def sweep_to_csv(sweep: SweepResult, output_dir: str, run_id: Optional[str] = None):
    """Export parameter sweep results to CSV with metadata."""
    if run_id is None:
        run_id = generate_run_id("sweep")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export results
    results_file = output_path / generate_filename(run_id, "results", "csv")
    results_to_csv(sweep.results, str(results_file))
    
    # Export metadata
    metadata = RunMetadata(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        description="Parameter sweep",
        n_samples=sweep.n_total,
        spin_values=sweep.grid.spins,
        E_range=list(sweep.grid.E_range),
        Lz_range=list(sweep.grid.Lz_range),
        duration_seconds=sweep.duration,
        n_escape=sweep.n_escape,
        n_capture=sweep.n_capture,
        n_penrose=sweep.n_penrose,
    )
    
    metadata_file = output_path / generate_filename(run_id, "metadata", "json")
    metadata.save(str(metadata_file))
    
    print(f"Sweep exported to {output_path}")
    return run_id


def ensemble_to_csv(ensemble: EnsembleResult, output_dir: str, 
                     run_id: Optional[str] = None):
    """Export ensemble results to CSV with metadata."""
    if run_id is None:
        run_id = generate_run_id("ensemble")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export results
    results_file = output_path / generate_filename(run_id, "results", "csv")
    results_to_csv(ensemble.results, str(results_file))
    
    # Export metadata
    metadata = RunMetadata(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        description=f"Ensemble ({ensemble.config.sampling_method})",
        n_samples=ensemble.n_total,
        strategies=[ensemble.config.strategy.name],
        E_range=list(ensemble.config.E_bounds),
        Lz_range=list(ensemble.config.Lz_bounds),
        duration_seconds=ensemble.duration,
        n_escape=ensemble.n_escape,
        n_capture=ensemble.n_capture,
        n_penrose=ensemble.n_penrose,
    )
    
    metadata_file = output_path / generate_filename(run_id, "metadata", "json")
    metadata.save(str(metadata_file))
    
    # Export summary statistics
    stats = {
        'escape_probability': ensemble.escape_probability,
        'penrose_probability': ensemble.penrose_probability,
        'Delta_E_mean': ensemble.Delta_E_mean,
        'Delta_E_std': ensemble.Delta_E_std,
        'Delta_E_median': ensemble.Delta_E_median,
        'eta_cum_mean': ensemble.eta_cum_mean,
        'eta_cum_std': ensemble.eta_cum_std,
        'E_ex_mean': ensemble.E_ex_mean,
        'penrose_fraction_mean': ensemble.penrose_fraction_mean,
    }
    
    stats_file = output_path / generate_filename(run_id, "statistics", "json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Ensemble exported to {output_path}")
    return run_id


# =============================================================================
# HDF5 EXPORT (for large datasets)
# =============================================================================

def results_to_hdf5(results: List[TrajectoryResult],
                     filepath: str,
                     include_trajectories: bool = True,
                     compression: str = 'gzip'):
    """
    Export results to HDF5 format with optional trajectory data.
    
    HDF5 is better for large datasets with trajectory arrays.
    """
    try:
        import h5py
    except ImportError:
        print("h5py required for HDF5 export. Install with: pip install h5py")
        return
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        # Metadata
        f.attrs['n_results'] = len(results)
        f.attrs['created'] = datetime.now().isoformat()
        
        # Create datasets for scalar quantities
        n = len(results)
        
        # Initial conditions
        ic_grp = f.create_group('initial_conditions')
        ic_grp.create_dataset('E0', data=[r.initial_conditions.get('E0', np.nan) for r in results])
        ic_grp.create_dataset('Lz0', data=[r.initial_conditions.get('Lz0', np.nan) for r in results])
        ic_grp.create_dataset('a', data=[r.initial_conditions.get('a', np.nan) for r in results])
        
        # Outcomes
        out_grp = f.create_group('outcomes')
        out_grp.create_dataset('outcome_code', data=[r.outcome.value for r in results])
        out_grp.create_dataset('r_final', data=[r.r_final for r in results])
        out_grp.create_dataset('E_final', data=[r.E_final for r in results])
        out_grp.create_dataset('m_final', data=[r.m_final for r in results])
        
        # Metrics
        metrics_grp = f.create_group('metrics')
        metrics_grp.create_dataset('Delta_E', data=[r.Delta_E for r in results])
        metrics_grp.create_dataset('Delta_m', data=[r.Delta_m for r in results])
        metrics_grp.create_dataset('eta_cumulative', data=[r.eta_cumulative for r in results])
        metrics_grp.create_dataset('penrose_fraction', data=[r.penrose_fraction for r in results])
        metrics_grp.create_dataset('r_min', data=[r.r_min for r in results])
        
        # Trajectories (variable length arrays)
        if include_trajectories:
            traj_grp = f.create_group('trajectories')
            
            for i, r in enumerate(results):
                if r.trajectory_data is not None:
                    run_grp = traj_grp.create_group(f'run_{i:06d}')
                    for key, arr in r.trajectory_data.items():
                        if isinstance(arr, np.ndarray):
                            run_grp.create_dataset(
                                key, data=arr, compression=compression
                            )
    
    print(f"Exported {len(results)} results to {filepath}")


# =============================================================================
# LOADING FUNCTIONS
# =============================================================================

def load_results_csv(filepath: str) -> List[Dict]:
    """Load results from CSV file."""
    results = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in row:
                try:
                    if '.' in row[key]:
                        row[key] = float(row[key])
                    elif row[key].isdigit():
                        row[key] = int(row[key])
                except (ValueError, AttributeError):
                    pass
            results.append(row)
    return results


def load_benchmark_run(output_dir: str, run_id: str) -> Dict[str, Any]:
    """Load complete benchmark run (results + metadata)."""
    output_path = Path(output_dir)
    
    # Load metadata
    metadata_file = output_path / generate_filename(run_id, "metadata", "json")
    if metadata_file.exists():
        metadata = RunMetadata.load(str(metadata_file))
    else:
        metadata = None
    
    # Load results
    results_file = output_path / generate_filename(run_id, "results", "csv")
    if results_file.exists():
        results = load_results_csv(str(results_file))
    else:
        results = []
    
    # Load statistics if available
    stats_file = output_path / generate_filename(run_id, "statistics", "json")
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            statistics = json.load(f)
    else:
        statistics = {}
    
    return {
        'run_id': run_id,
        'metadata': metadata,
        'results': results,
        'statistics': statistics,
    }


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def compare_runs(run1_dir: str, run1_id: str,
                  run2_dir: str, run2_id: str) -> Dict[str, Any]:
    """Compare two benchmark runs."""
    run1 = load_benchmark_run(run1_dir, run1_id)
    run2 = load_benchmark_run(run2_dir, run2_id)
    
    comparison = {
        'run1_id': run1_id,
        'run2_id': run2_id,
    }
    
    # Compare sample sizes
    comparison['n_samples'] = {
        'run1': run1['metadata'].n_samples if run1['metadata'] else len(run1['results']),
        'run2': run2['metadata'].n_samples if run2['metadata'] else len(run2['results']),
    }
    
    # Compare outcomes
    for run_key, run_data in [('run1', run1), ('run2', run2)]:
        results = run_data['results']
        comparison[f'{run_key}_escape_rate'] = sum(
            1 for r in results if r.get('outcome') == 'ESCAPE'
        ) / len(results) if results else 0
    
    # Compare statistics if available
    for stat_key in ['Delta_E_mean', 'eta_cum_mean', 'penrose_probability']:
        comparison[stat_key] = {
            'run1': run1['statistics'].get(stat_key),
            'run2': run2['statistics'].get(stat_key),
        }
    
    return comparison


# =============================================================================
# SUMMARY REPORTS
# =============================================================================

def generate_summary_report(results: List[TrajectoryResult],
                             output_file: Optional[str] = None) -> str:
    """Generate human-readable summary report."""
    n_total = len(results)
    
    if n_total == 0:
        return "No results to summarize"
    
    # Count outcomes
    outcomes = {}
    for r in results:
        key = r.outcome.name
        outcomes[key] = outcomes.get(key, 0) + 1
    
    # Statistics for escaped trajectories
    escaped = [r for r in results if r.outcome == TrajectoryOutcome.ESCAPE]
    
    lines = [
        "="*70,
        "BENCHMARK SUMMARY REPORT",
        "="*70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Total samples: {n_total}",
        "",
        "OUTCOME DISTRIBUTION:",
        "-"*40,
    ]
    
    for outcome, count in sorted(outcomes.items()):
        pct = 100 * count / n_total
        lines.append(f"  {outcome:<25} {count:5d} ({pct:5.1f}%)")
    
    if escaped:
        Delta_E_vals = [r.Delta_E for r in escaped]
        eta_vals = [r.eta_cumulative for r in escaped if r.eta_cumulative > 0]
        penrose_fracs = [r.penrose_fraction for r in escaped]
        
        lines.extend([
            "",
            "EXTRACTION STATISTICS (escaped only):",
            "-"*40,
            f"  Energy gain DeltaE:",
            f"    Mean: {np.mean(Delta_E_vals):+.4f}",
            f"    Std:  {np.std(Delta_E_vals):.4f}",
            f"    Min:  {np.min(Delta_E_vals):+.4f}",
            f"    Max:  {np.max(Delta_E_vals):+.4f}",
        ])
        
        if eta_vals:
            lines.extend([
                "",
                f"  Cumulative efficiency eta_cum:",
                f"    Mean: {100*np.mean(eta_vals):.2f}%",
                f"    Std:  {100*np.std(eta_vals):.2f}%",
            ])
        
        n_penrose = sum(1 for f in penrose_fracs if f > 0.5)
        lines.extend([
            "",
            f"  Genuine Penrose extraction (>50% E_ex < 0):",
            f"    Count: {n_penrose}/{len(escaped)} ({100*n_penrose/len(escaped):.1f}%)",
        ])
    
    lines.extend([
        "",
        "="*70,
    ])
    
    report = "\n".join(lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_file}")
    
    return report


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("BENCHMARK DATA EXPORT - Quick Test")
    print("="*70)
    
    # Create mock results
    mock_results = []
    for i in range(5):
        r = TrajectoryResult(
            initial_conditions={'a': 0.95, 'E0': 1.2 + 0.05*i, 'Lz0': 3.0},
            outcome=TrajectoryOutcome.ESCAPE if i % 2 == 0 else TrajectoryOutcome.CAPTURE,
            r_final=50.0 if i % 2 == 0 else 1.5,
            E_final=1.25,
            m_final=0.8,
            Delta_E=0.05,
            Delta_m=0.2,
            eta_cumulative=0.25,
            r_min=1.6,
            orbit_profile=OrbitProfile.FLYBY_DEEP_ERGOSPHERE,
            strategy=ThrustStrategy.SINGLE_IMPULSE,
        )
        mock_results.append(r)
    
    # Test 1: Generate run ID
    run_id = generate_run_id("test")
    print(f"\n1. Generated run ID: {run_id}")
    
    # Test 2: Export to CSV
    print("\n2. Testing CSV export...")
    output_dir = "/tmp/penrose_benchmark"
    csv_file = f"{output_dir}/test_results.csv"
    results_to_csv(mock_results, csv_file)
    
    # Test 3: Load CSV
    print("\n3. Testing CSV load...")
    loaded = load_results_csv(csv_file)
    print(f"   Loaded {len(loaded)} records")
    
    # Test 4: Generate summary report
    print("\n4. Generating summary report...")
    report = generate_summary_report(mock_results)
    print(report)
    
    # Test 5: Metadata
    print("\n5. Testing metadata...")
    metadata = RunMetadata(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        n_samples=5,
        spin_values=[0.95],
        E_range=[1.2, 1.4],
    )
    metadata_file = f"{output_dir}/test_metadata.json"
    metadata.save(metadata_file)
    loaded_meta = RunMetadata.load(metadata_file)
    print(f"   Saved and loaded metadata: {loaded_meta.run_id}")
    
    print("\n" + "="*70)
    print("Benchmark export module ready!")
    print("="*70)
