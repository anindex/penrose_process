"""
Statistical Ensemble Module
============================
Monte Carlo ensemble runs over initial condition distributions.

Provides robust statistics for:
- Penrose extraction probability given parameter uncertainties
- Sensitivity analysis to initial conditions
- Identification of "sweet spot" regions in parameter space

Further considerations:
- Latin Hypercube Sampling for efficient parameter space coverage
- Parallel execution support
- Convergence diagnostics
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import json
from pathlib import Path

from experiments.trajectory_classifier import (
    OrbitProfile, TrajectoryOutcome, ThrustStrategy, TrajectoryResult
)
from experiments.thrust_comparison import (
    SimulationConfig, simulate_geodesic, simulate_single_impulse, simulate_burst
)


# =============================================================================
# SAMPLING STRATEGIES
# =============================================================================

def latin_hypercube_sample(n_samples: int, 
                            bounds: Dict[str, Tuple[float, float]],
                            seed: Optional[int] = None) -> List[Dict[str, float]]:
    """
    Generate Latin Hypercube samples for parameter space.
    
    LHS provides better coverage than random sampling with fewer samples.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    bounds : dict
        Parameter bounds as {name: (min, max)}
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    list of dict
        List of parameter dictionaries
    """
    rng = np.random.default_rng(seed)
    
    n_params = len(bounds)
    param_names = list(bounds.keys())
    
    # Create Latin Hypercube
    samples = np.zeros((n_samples, n_params))
    
    for j in range(n_params):
        # Divide range into n_samples equal strata
        perm = rng.permutation(n_samples)
        samples[:, j] = (perm + rng.random(n_samples)) / n_samples
    
    # Scale to actual bounds
    result = []
    for i in range(n_samples):
        params = {}
        for j, name in enumerate(param_names):
            low, high = bounds[name]
            params[name] = low + samples[i, j] * (high - low)
        result.append(params)
    
    return result


def uniform_random_sample(n_samples: int,
                           bounds: Dict[str, Tuple[float, float]],
                           seed: Optional[int] = None) -> List[Dict[str, float]]:
    """
    Generate uniform random samples for parameter space.
    """
    rng = np.random.default_rng(seed)
    
    result = []
    for _ in range(n_samples):
        params = {}
        for name, (low, high) in bounds.items():
            params[name] = rng.uniform(low, high)
        result.append(params)
    
    return result


def gaussian_perturbation_sample(n_samples: int,
                                   center: Dict[str, float],
                                   std: Dict[str, float],
                                   bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                                   seed: Optional[int] = None) -> List[Dict[str, float]]:
    """
    Generate Gaussian-perturbed samples around a center point.
    
    Useful for sensitivity analysis around a known good configuration.
    """
    rng = np.random.default_rng(seed)
    
    result = []
    for _ in range(n_samples):
        params = {}
        for name, value in center.items():
            sigma = std.get(name, 0.0)
            sample = rng.normal(value, sigma)
            
            # Apply bounds if provided
            if bounds and name in bounds:
                low, high = bounds[name]
                sample = np.clip(sample, low, high)
            
            params[name] = sample
        result.append(params)
    
    return result


# =============================================================================
# STATISTICAL CONFIDENCE INTERVALS
# =============================================================================

def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Compute Clopper-Pearson exact confidence interval for binomial proportion.
    
    This is the standard method for small counts, providing conservative
    coverage guarantees.
    
    Parameters
    ----------
    k : int
        Number of successes
    n : int
        Number of trials
    alpha : float
        Significance level (default 0.05 for 95% CI)
        
    Returns
    -------
    tuple
        (lower, upper) confidence interval bounds
        
    References
    ----------
    Clopper, C. J. and Pearson, E. S. (1934). Biometrika 26, 404-413.
    """
    from scipy import stats
    
    if n == 0:
        return (0.0, 1.0)
    
    if k == 0:
        lower = 0.0
    else:
        lower = stats.beta.ppf(alpha / 2, k, n - k + 1)
    
    if k == n:
        upper = 1.0
    else:
        upper = stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    
    return (lower, upper)


def bca_bootstrap_ci(data: np.ndarray, 
                      stat_func: callable = np.mean,
                      n_bootstrap: int = 10000,
                      alpha: float = 0.05,
                      seed: Optional[int] = None) -> Tuple[float, float]:
    """
    Compute BCa (bias-corrected accelerated) bootstrap confidence interval.
    
    BCa bootstrap provides more accurate intervals than percentile bootstrap,
    especially for small samples and skewed distributions.
    
    Parameters
    ----------
    data : array-like
        Sample data
    stat_func : callable
        Statistic function (default: np.mean)
    n_bootstrap : int
        Number of bootstrap resamples
    alpha : float
        Significance level (default 0.05 for 95% CI)
    seed : int, optional
        Random seed
        
    Returns
    -------
    tuple
        (lower, upper) confidence interval bounds
        
    References
    ----------
    Efron, B. (1987). J. Am. Stat. Assoc. 82, 171-185.
    """
    try:
        from scipy import stats
    except ImportError:
        raise ImportError("scipy required for BCa bootstrap")
    
    data = np.asarray(data)
    n = len(data)
    
    if n < 2:
        return (data[0] if n == 1 else np.nan, data[0] if n == 1 else np.nan)
    
    rng = np.random.default_rng(seed)
    
    # Original statistic
    theta_hat = stat_func(data)
    
    # Bootstrap distribution
    theta_boot = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        theta_boot[i] = stat_func(data[indices])
    
    # Bias correction factor (z0)
    prop_below = np.mean(theta_boot < theta_hat)
    prop_below = np.clip(prop_below, 1e-10, 1 - 1e-10)  # Avoid ppf(0) = -inf or ppf(1) = +inf
    z0 = stats.norm.ppf(prop_below)
    
    # Acceleration factor (a) via jackknife
    theta_jack = np.zeros(n)
    for i in range(n):
        theta_jack[i] = stat_func(np.delete(data, i))
    theta_jack_mean = np.mean(theta_jack)
    
    num = np.sum((theta_jack_mean - theta_jack) ** 3)
    denom = 6 * (np.sum((theta_jack_mean - theta_jack) ** 2) ** 1.5)
    
    if denom == 0:
        a = 0.0
    else:
        a = num / denom
    
    # Adjusted quantiles
    z_alpha = stats.norm.ppf(alpha / 2)
    z_1_alpha = stats.norm.ppf(1 - alpha / 2)
    
    # BCa quantile transformation
    def bca_quantile(z):
        return stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    
    q_lower = bca_quantile(z_alpha)
    q_upper = bca_quantile(z_1_alpha)
    
    # Clip to valid range
    q_lower = np.clip(q_lower, 0.001, 0.999)
    q_upper = np.clip(q_upper, 0.001, 0.999)
    
    # Get CI from bootstrap distribution
    lower = np.percentile(theta_boot, 100 * q_lower)
    upper = np.percentile(theta_boot, 100 * q_upper)
    
    return (lower, upper)


def compute_confidence_intervals(ensemble_result: 'EnsembleResult',
                                   alpha: float = 0.05,
                                   n_bootstrap: int = 10000,
                                   seed: int = 42) -> Dict[str, Dict[str, float]]:
    """
    Compute rigorous confidence intervals for ensemble statistics.
    
    Uses Clopper-Pearson for proportions and BCa bootstrap for continuous metrics.
    
    Parameters
    ----------
    ensemble_result : EnsembleResult
        Completed ensemble results
    alpha : float
        Significance level (default 0.05 for 95% CI)
    n_bootstrap : int
        Number of bootstrap resamples
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Nested dictionary with point estimates and CI bounds
    """
    results = ensemble_result.results
    n_total = len(results)
    
    # Count outcomes (excluding integration failures for outcome analysis)
    n_valid = sum(1 for r in results 
                  if r.outcome != TrajectoryOutcome.INTEGRATION_FAILURE)
    n_escape = sum(1 for r in results 
                   if r.outcome == TrajectoryOutcome.ESCAPE)
    n_capture = sum(1 for r in results 
                    if r.outcome == TrajectoryOutcome.CAPTURE)
    n_integration_failure = sum(1 for r in results 
                                 if r.outcome == TrajectoryOutcome.INTEGRATION_FAILURE)
    
    # Genuine Penrose: escape with >50% thrust at E_ex < 0
    escaped = [r for r in results if r.outcome == TrajectoryOutcome.ESCAPE]
    n_genuine_penrose = sum(1 for r in escaped if r.penrose_fraction > 0.5)
    
    ci_results = {}
    
    # Escape probability (Clopper-Pearson)
    # Two versions: excluding and including integration failures
    p_escape_excl = n_escape / n_valid if n_valid > 0 else 0
    ci_escape_excl = clopper_pearson_ci(n_escape, n_valid, alpha)
    
    p_escape_incl = n_escape / n_total if n_total > 0 else 0
    ci_escape_incl = clopper_pearson_ci(n_escape, n_total, alpha)
    
    ci_results['escape_probability'] = {
        'point': p_escape_excl,
        'lower': ci_escape_excl[0],
        'upper': ci_escape_excl[1],
        'n_success': n_escape,
        'n_trials': n_valid,
        'method': 'Clopper-Pearson (excluding integration failures)',
    }
    
    ci_results['escape_probability_robust'] = {
        'point': p_escape_incl,
        'lower': ci_escape_incl[0],
        'upper': ci_escape_incl[1],
        'n_success': n_escape,
        'n_trials': n_total,
        'method': 'Clopper-Pearson (including failures as non-escapes)',
    }
    
    # Genuine Penrose fraction among escapes
    if len(escaped) > 0:
        p_genuine = n_genuine_penrose / len(escaped)
        ci_genuine = clopper_pearson_ci(n_genuine_penrose, len(escaped), alpha)
        
        ci_results['genuine_penrose_fraction'] = {
            'point': p_genuine,
            'lower': ci_genuine[0],
            'upper': ci_genuine[1],
            'n_success': n_genuine_penrose,
            'n_trials': len(escaped),
            'method': 'Clopper-Pearson',
        }
    
    # Energy gain (BCa bootstrap)
    if len(escaped) >= 2:
        Delta_E_vals = np.array([r.Delta_E for r in escaped])
        Delta_E_mean = np.mean(Delta_E_vals)
        ci_Delta_E = bca_bootstrap_ci(Delta_E_vals, np.mean, n_bootstrap, alpha, seed)
        
        ci_results['Delta_E'] = {
            'point': Delta_E_mean,
            'lower': ci_Delta_E[0],
            'upper': ci_Delta_E[1],
            'std': np.std(Delta_E_vals),
            'n_samples': len(escaped),
            'method': 'BCa bootstrap',
        }
    
    # Cumulative efficiency (BCa bootstrap)
    eta_vals = np.array([r.eta_cumulative for r in escaped if r.eta_cumulative > 0])
    if len(eta_vals) >= 2:
        eta_mean = np.mean(eta_vals)
        ci_eta = bca_bootstrap_ci(eta_vals, np.mean, n_bootstrap, alpha, seed)
        
        ci_results['eta_cumulative'] = {
            'point': eta_mean,
            'lower': ci_eta[0],
            'upper': ci_eta[1],
            'std': np.std(eta_vals),
            'n_samples': len(eta_vals),
            'method': 'BCa bootstrap',
        }
    
    # Integration failure rate
    ci_results['integration_failure_rate'] = {
        'point': n_integration_failure / n_total if n_total > 0 else 0,
        'count': n_integration_failure,
        'total': n_total,
    }
    
    return ci_results


def format_confidence_intervals(ci_results: Dict, percent: bool = True) -> str:
    """
    Format confidence interval results as a human-readable string.
    
    Parameters
    ----------
    ci_results : dict
        Output from compute_confidence_intervals
    percent : bool
        If True, display probabilities as percentages
        
    Returns
    -------
    str
        Formatted summary string
    """
    lines = [
        "="*70,
        "STATISTICAL CONFIDENCE INTERVALS (95% CI)",
        "="*70,
    ]
    
    for key, data in ci_results.items():
        if 'point' not in data:
            continue
            
        point = data['point']
        lower = data.get('lower', 0)
        upper = data.get('upper', 1)
        method = data.get('method', 'unknown')
        
        if percent and ('probability' in key or 'fraction' in key or 'eta' in key):
            lines.append(f"\n{key}:")
            lines.append(f"  Point estimate: {100*point:.2f}%")
            lines.append(f"  95% CI: [{100*lower:.2f}%, {100*upper:.2f}%]")
        else:
            lines.append(f"\n{key}:")
            lines.append(f"  Point estimate: {point:+.4f}")
            lines.append(f"  95% CI: [{lower:+.4f}, {upper:+.4f}]")
        
        if 'n_success' in data:
            lines.append(f"  Counts: {data['n_success']}/{data['n_trials']}")
        if 'n_samples' in data:
            lines.append(f"  N samples: {data['n_samples']}")
        lines.append(f"  Method: {method}")
    
    # Integration failure summary
    if 'integration_failure_rate' in ci_results:
        ifr = ci_results['integration_failure_rate']
        lines.append(f"\nIntegration failures: {ifr['count']}/{ifr['total']} "
                    f"({100*ifr['point']:.1f}%)")
    
    lines.append("="*70)
    return "\n".join(lines)


# =============================================================================
# ENSEMBLE CONFIGURATION
# =============================================================================

@dataclass
class EnsembleConfig:
    """Configuration for ensemble runs."""
    # Base configuration
    base_config: SimulationConfig = field(default_factory=SimulationConfig)
    
    # Sampling
    n_samples: int = 100
    sampling_method: str = 'lhs'  # 'lhs', 'random', 'gaussian'
    seed: Optional[int] = 42
    
    # Parameter bounds for sampling
    # E ~ 1.15-1.35: marginally unbound orbits optimal for extraction
    # Lz ~ 2.9-3.3: prograde flyby that penetrates ergosphere (tuned for a~0.95)
    # Note: Success rate is typically 5-10% due to narrow viable region
    E_bounds: Tuple[float, float] = (1.15, 1.35)
    Lz_bounds: Tuple[float, float] = (2.9, 3.3)
    
    # Optional: vary spin and exhaust velocity
    vary_spin: bool = False
    a_bounds: Tuple[float, float] = (0.9, 0.99)
    
    vary_v_e: bool = False
    v_e_bounds: Tuple[float, float] = (0.9, 0.98)
    
    # Gaussian perturbation parameters (for 'gaussian' method)
    E_std: float = 0.05
    Lz_std: float = 0.1
    
    # Strategy to test
    strategy: ThrustStrategy = ThrustStrategy.SINGLE_IMPULSE
    
    # Execution
    n_workers: int = 1
    
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds dictionary."""
        bounds = {
            'E0': self.E_bounds,
            'Lz0': self.Lz_bounds,
        }
        if self.vary_spin:
            bounds['a'] = self.a_bounds
        if self.vary_v_e:
            bounds['v_e'] = self.v_e_bounds
        return bounds
    
    def get_center(self) -> Dict[str, float]:
        """Get center point for Gaussian sampling."""
        return {
            'E0': self.base_config.E0,
            'Lz0': self.base_config.Lz0,
        }
    
    def get_std(self) -> Dict[str, float]:
        """Get standard deviations for Gaussian sampling."""
        return {
            'E0': self.E_std,
            'Lz0': self.Lz_std,
        }


# =============================================================================
# ENSEMBLE RESULTS
# =============================================================================

@dataclass
class EnsembleResult:
    """Container for ensemble run results."""
    config: EnsembleConfig
    results: List[TrajectoryResult] = field(default_factory=list)
    
    # Timing
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # Computed statistics
    n_total: int = 0
    n_escape: int = 0
    n_capture: int = 0
    n_penrose: int = 0
    
    escape_probability: float = 0.0
    penrose_probability: float = 0.0
    
    # Distribution statistics
    Delta_E_mean: float = 0.0
    Delta_E_std: float = 0.0
    Delta_E_median: float = 0.0
    
    eta_cum_mean: float = 0.0
    eta_cum_std: float = 0.0
    
    E_ex_mean: float = 0.0
    penrose_fraction_mean: float = 0.0
    
    def compute_statistics(self):
        """Compute summary statistics from results."""
        self.n_total = len(self.results)
        
        if self.n_total == 0:
            return
        
        # Outcome counts
        self.n_escape = sum(1 for r in self.results 
                           if r.outcome == TrajectoryOutcome.ESCAPE)
        self.n_capture = sum(1 for r in self.results 
                            if r.outcome == TrajectoryOutcome.CAPTURE)
        self.n_penrose = sum(1 for r in self.results 
                            if r.outcome == TrajectoryOutcome.ESCAPE 
                            and r.penrose_fraction > 0.5)
        
        # Probabilities
        self.escape_probability = self.n_escape / self.n_total
        self.penrose_probability = self.n_penrose / self.n_total
        
        # Energy statistics (for escaped trajectories)
        escaped = [r for r in self.results if r.outcome == TrajectoryOutcome.ESCAPE]
        if escaped:
            Delta_E_vals = [r.Delta_E for r in escaped]
            self.Delta_E_mean = np.mean(Delta_E_vals)
            self.Delta_E_std = np.std(Delta_E_vals, ddof=1) if len(Delta_E_vals) > 1 else 0.0
            self.Delta_E_median = np.median(Delta_E_vals)
            
            eta_vals = [r.eta_cumulative for r in escaped if r.eta_cumulative > 0]
            if eta_vals:
                self.eta_cum_mean = np.mean(eta_vals)
                self.eta_cum_std = np.std(eta_vals, ddof=1) if len(eta_vals) > 1 else 0.0
            
            E_ex_vals = [r.E_ex_mean for r in escaped if r.n_total_E_ex > 0]
            if E_ex_vals:
                self.E_ex_mean = np.mean(E_ex_vals)
            
            penrose_fracs = [r.penrose_fraction for r in escaped]
            self.penrose_fraction_mean = np.mean(penrose_fracs)
    
    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "="*70,
            "ENSEMBLE STATISTICS",
            "="*70,
            f"Total samples: {self.n_total}",
            f"Escape probability: {100*self.escape_probability:.1f}%",
            f"Penrose probability: {100*self.penrose_probability:.1f}%",
            "-"*70,
            f"Energy gain (escaped): DeltaE = {self.Delta_E_mean:+.4f} +/- {self.Delta_E_std:.4f}",
            f"Energy gain median: {self.Delta_E_median:+.4f}",
            f"Efficiency eta_cum: {100*self.eta_cum_mean:.2f}% +/- {100*self.eta_cum_std:.2f}%",
            f"Mean E_ex: {self.E_ex_mean:.4f}",
            f"Mean Penrose fraction: {100*self.penrose_fraction_mean:.1f}%",
            "-"*70,
            f"Duration: {self.duration:.1f}s ({self.duration/self.n_total:.3f}s/sample)" if self.n_total > 0 else f"Duration: {self.duration:.1f}s (no samples)",
            "="*70,
        ]
        return "\n".join(lines)
    
    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for DataFrame export")
        
        records = []
        for r in self.results:
            record = {
                **r.initial_conditions,
                'outcome': r.outcome.name,
                'Delta_E': r.Delta_E,
                'Delta_m': r.Delta_m,
                'eta_cumulative': r.eta_cumulative,
                'E_ex_mean': r.E_ex_mean,
                'penrose_fraction': r.penrose_fraction,
                'r_min': r.r_min,
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def save(self, filepath: str):
        """Save results to JSON."""
        data = {
            'config': {
                'n_samples': self.config.n_samples,
                'sampling_method': self.config.sampling_method,
                'seed': self.config.seed,
                'E_bounds': list(self.config.E_bounds),
                'Lz_bounds': list(self.config.Lz_bounds),
                'strategy': self.config.strategy.name,
            },
            'statistics': {
                'n_total': self.n_total,
                'n_escape': self.n_escape,
                'n_capture': self.n_capture,
                'n_penrose': self.n_penrose,
                'escape_probability': self.escape_probability,
                'penrose_probability': self.penrose_probability,
                'Delta_E_mean': self.Delta_E_mean,
                'Delta_E_std': self.Delta_E_std,
                'eta_cum_mean': self.eta_cum_mean,
                'duration': self.duration,
            },
            'results': [
                {
                    'initial_conditions': r.initial_conditions,
                    'outcome': r.outcome.name,
                    'Delta_E': r.Delta_E,
                    'eta_cumulative': r.eta_cumulative,
                    'penrose_fraction': r.penrose_fraction,
                }
                for r in self.results
            ]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# ENSEMBLE RUNNER
# =============================================================================

def _run_single_sample(args: Tuple) -> TrajectoryResult:
    """Worker function for parallel execution."""
    params, base_config, strategy = args
    
    # Create config with sampled parameters
    config = SimulationConfig(
        a=params.get('a', base_config.a),
        M=base_config.M,
        E0=params['E0'],
        Lz0=params['Lz0'],
        r0=base_config.r0,
        m0=base_config.m0,
        v_e=params.get('v_e', base_config.v_e),
        T_max=base_config.T_max,
        delta_m_fraction=base_config.delta_m_fraction,
        tau_max=base_config.tau_max,
        escape_radius=base_config.escape_radius,
    )
    
    # Run simulation based on strategy
    if strategy == ThrustStrategy.NONE:
        result = simulate_geodesic(config)
    elif strategy == ThrustStrategy.SINGLE_IMPULSE:
        result = simulate_single_impulse(config)
    elif strategy == ThrustStrategy.BURST:
        result = simulate_burst(config)
    else:
        result = simulate_single_impulse(config)
    
    return result


def run_ensemble(ensemble_config: EnsembleConfig,
                  progress_callback: Optional[Callable] = None,
                  verbose: bool = True) -> EnsembleResult:
    """
    Run Monte Carlo ensemble of trajectory simulations.
    
    Parameters
    ----------
    ensemble_config : EnsembleConfig
        Configuration for ensemble run
    progress_callback : callable, optional
        Called with (n_completed, n_total) for progress updates
    verbose : bool
        Print progress messages
        
    Returns
    -------
    EnsembleResult
        Complete ensemble results with statistics
    """
    result = EnsembleResult(config=ensemble_config)
    result.start_time = time.time()
    
    # Generate samples
    bounds = ensemble_config.get_bounds()
    
    if ensemble_config.sampling_method == 'lhs':
        samples = latin_hypercube_sample(
            ensemble_config.n_samples, bounds, ensemble_config.seed
        )
    elif ensemble_config.sampling_method == 'random':
        samples = uniform_random_sample(
            ensemble_config.n_samples, bounds, ensemble_config.seed
        )
    elif ensemble_config.sampling_method == 'gaussian':
        center = ensemble_config.get_center()
        std = ensemble_config.get_std()
        samples = gaussian_perturbation_sample(
            ensemble_config.n_samples, center, std, bounds, ensemble_config.seed
        )
    else:
        raise ValueError(f"Unknown sampling method: {ensemble_config.sampling_method}")
    
    if verbose:
        print(f"\nRunning ensemble: {len(samples)} samples, "
              f"strategy: {ensemble_config.strategy.name}")
        print(f"Sampling method: {ensemble_config.sampling_method}")
    
    # Run simulations
    args_list = [(s, ensemble_config.base_config, ensemble_config.strategy) 
                 for s in samples]
    
    if ensemble_config.n_workers == 1:
        # Sequential execution
        for i, args in enumerate(args_list):
            try:
                sim_result = _run_single_sample(args)
                result.results.append(sim_result)
            except Exception as e:
                print(f"Sample {i} failed: {e}")
            
            if progress_callback:
                progress_callback(i + 1, len(samples))
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{len(samples)}")
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=ensemble_config.n_workers) as executor:
            futures = {executor.submit(_run_single_sample, args): i 
                      for i, args in enumerate(args_list)}
            
            for i, future in enumerate(as_completed(futures)):
                try:
                    sim_result = future.result()
                    result.results.append(sim_result)
                except Exception as e:
                    print(f"Worker failed: {e}")
                
                if progress_callback:
                    progress_callback(i + 1, len(samples))
    
    result.end_time = time.time()
    result.compute_statistics()
    
    if verbose:
        print(result.summary())
    
    return result


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def compute_sensitivity(ensemble_result: EnsembleResult,
                         output_var: str = 'Delta_E') -> Dict[str, float]:
    """
    Compute parameter sensitivity from ensemble results.
    
    Uses correlation coefficients to estimate which parameters
    most influence the output.
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas required for sensitivity analysis")
        return {}
    
    df = ensemble_result.to_dataframe()
    
    # Filter to escaped trajectories
    df_escaped = df[df['outcome'] == 'ESCAPE']
    
    if len(df_escaped) < 10:
        print("Not enough escaped trajectories for sensitivity analysis")
        return {}
    
    # Compute correlations
    input_params = ['E0', 'Lz0']
    if 'a' in df.columns:
        input_params.append('a')
    if 'v_e' in df.columns:
        input_params.append('v_e')
    
    sensitivities = {}
    for param in input_params:
        if param in df_escaped.columns:
            corr = df_escaped[param].corr(df_escaped[output_var])
            sensitivities[param] = corr
    
    return sensitivities


def find_sweet_spot(ensemble_result: EnsembleResult,
                     n_top: int = 10) -> List[Dict]:
    """
    Identify parameter combinations with best performance.
    
    Returns top configurations by energy gain.
    """
    escaped = [r for r in ensemble_result.results 
               if r.outcome == TrajectoryOutcome.ESCAPE and r.Delta_E > 0]
    
    if not escaped:
        return []
    
    # Sort by Delta_E
    sorted_results = sorted(escaped, key=lambda r: r.Delta_E, reverse=True)
    
    top_configs = []
    for r in sorted_results[:n_top]:
        config = {
            **r.initial_conditions,
            'Delta_E': r.Delta_E,
            'eta_cumulative': r.eta_cumulative,
            'penrose_fraction': r.penrose_fraction,
        }
        top_configs.append(config)
    
    return top_configs


# =============================================================================
# CONVERGENCE DIAGNOSTICS
# =============================================================================

def check_convergence(ensemble_result: EnsembleResult,
                       metric: str = 'Delta_E',
                       window_size: int = 20) -> Dict[str, Any]:
    """
    Check if ensemble has converged using running statistics.
    
    Computes running mean and checks stability.
    """
    escaped = [r for r in ensemble_result.results 
               if r.outcome == TrajectoryOutcome.ESCAPE]
    
    if len(escaped) < 2 * window_size:
        return {'converged': False, 'reason': 'Insufficient samples'}
    
    values = [getattr(r, metric, 0) for r in escaped]
    
    # Compute running mean
    n = len(values)
    running_mean = np.zeros(n - window_size + 1)
    for i in range(len(running_mean)):
        running_mean[i] = np.mean(values[i:i + window_size])
    
    # Check if mean has stabilized (last 20% variation < 10%)
    last_portion = running_mean[-int(0.2 * len(running_mean)):]
    relative_std = np.std(last_portion) / (np.mean(last_portion) + 1e-10)
    
    converged = relative_std < 0.1
    
    return {
        'converged': converged,
        'running_mean': running_mean,
        'final_mean': running_mean[-1],
        'relative_std': relative_std,
        'n_samples': len(values),
    }


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("STATISTICAL ENSEMBLE - Quick Test")
    print("="*70)
    
    # Test 1: Latin Hypercube Sampling
    print("\n1. Testing LHS sampling...")
    bounds = {'E0': (1.1, 1.5), 'Lz0': (2.5, 3.5)}
    samples = latin_hypercube_sample(10, bounds, seed=42)
    print(f"   Generated {len(samples)} LHS samples")
    print(f"   Sample 0: E0={samples[0]['E0']:.3f}, Lz0={samples[0]['Lz0']:.3f}")
    
    # Test 2: Small ensemble run
    print("\n2. Running small ensemble (20 samples)...")
    
    base_config = SimulationConfig(
        a=0.95,
        E0=1.2,
        Lz0=3.0,
        v_e=0.95,
    )
    
    ensemble_config = EnsembleConfig(
        base_config=base_config,
        n_samples=20,
        sampling_method='lhs',
        E_bounds=(1.15, 1.25),
        Lz_bounds=(2.8, 3.2),
        strategy=ThrustStrategy.SINGLE_IMPULSE,
        seed=42,
    )
    
    result = run_ensemble(ensemble_config, verbose=True)
    
    # Test 3: Find sweet spot
    print("\n3. Finding sweet spot configurations...")
    sweet_spot = find_sweet_spot(result, n_top=3)
    for i, config in enumerate(sweet_spot):
        print(f"   #{i+1}: E0={config['E0']:.3f}, Lz0={config['Lz0']:.3f}, "
              f"DeltaE={config['Delta_E']:+.4f}")
    
    # Test 4: Sensitivity analysis
    print("\n4. Sensitivity analysis...")
    sens = compute_sensitivity(result)
    for param, corr in sens.items():
        print(f"   {param}: correlation = {corr:+.3f}")
    
    print("\n" + "="*70)
    print("Statistical ensemble module ready!")
    print("="*70)
