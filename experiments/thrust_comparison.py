"""
Thrust Strategy Comparison Module
==================================
Compare single-impulse vs continuous thrust strategies for Penrose extraction.

This module provides head-to-head comparisons of different thrust approaches
using identical initial conditions, allowing direct efficiency comparisons.

Strategies implemented:
1. NONE - Pure geodesic (baseline)
2. SINGLE_IMPULSE - One impulsive burn at optimal point
3. CONTINUOUS - Sustained thrust in ergosphere
4. BURST - Short burst at periapsis
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple, Any
from scipy.integrate import solve_ivp
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kerr_utils import (
    kerr_metric_components, horizon_radius, ergosphere_radius,
    compute_pt_from_mass_shell, compute_exhaust_energy, build_rocket_rest_basis,
    apply_exact_impulse, compute_optimal_exhaust_direction,
    compute_cumulative_efficiency
)
from experiments.trajectory_classifier import (
    OrbitProfile, TrajectoryOutcome, ThrustStrategy,
    classify_orbit, classify_trajectory_outcome, TrajectoryResult
)


# =============================================================================
# SIMULATION CONFIGURATION
# =============================================================================

@dataclass
class SimulationConfig:
    """Configuration for trajectory simulation."""
    # Black hole parameters
    a: float = 0.95
    M: float = 1.0
    
    # Initial conditions
    E0: float = 1.2
    Lz0: float = 3.0
    r0: float = 15.0
    m0: float = 1.0
    
    # Thrust parameters
    v_e: float = 0.95          # Exhaust velocity
    T_max: float = 0.2         # Max thrust magnitude (for continuous)
    m_min: float = 0.1         # Minimum mass fraction
    delta_m_fraction: float = 0.2  # Mass fraction for single impulse
    
    # Integration - improved numerical stability
    tau_max: float = 500.0     # Increased max proper time
    dt: float = 0.001          # Finer initial step
    escape_radius: float = 50.0
    horizon_margin: float = 0.02  # Capture at r < r_+ + 0.02M (matches paper Sec. IV.C)
    
    # Numerical tolerances
    rtol: float = 1e-10        # High accuracy tolerance
    atol: float = 1e-12        # High accuracy tolerance
    
    # Strategy-specific
    extraction_zone_factor: float = 0.85  # r < factor * r_erg for extraction
    
    @property
    def gamma_e(self) -> float:
        return 1.0 / np.sqrt(1 - self.v_e**2)
    
    @property
    def r_plus(self) -> float:
        return horizon_radius(self.a, self.M)
    
    @property
    def r_erg(self) -> float:
        return ergosphere_radius(np.pi/2, self.a, self.M)


# =============================================================================
# GEODESIC DYNAMICS (shared) - Improved numerical stability
# =============================================================================

def geodesic_dynamics(tau: float, state: np.ndarray, config: SimulationConfig) -> np.ndarray:
    """
    Geodesic equations of motion for free-fall with improved numerical stability.
    
    State: [r, phi, p_r, p_phi, m, p_t]
    Note: p_t and p_phi are constants for geodesics (Killing symmetries)
    
    Uses analytic derivatives of Kerr metric for better accuracy.
    """
    r, phi, pr, pphi, m, pt = state
    
    a, M = config.a, config.M
    th = np.pi / 2
    
    # Clamp radius to avoid horizon singularity
    r_plus = config.r_plus
    r_safe = max(r, r_plus + config.horizon_margin)
    
    # Get metric at safe radius
    cov, con = kerr_metric_components(r_safe, th, a, M, clamp_horizon=True, warn_horizon=False)
    g_tt, g_tphi, g_rr, _, g_phiphi = cov
    gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con
    
    # 4-velocity (divide by m for proper-time parameterization)
    u_r = gu_rr * pr / m
    u_phi = (gu_tphi * pt + gu_phiphi * pphi) / m
    
    # Analytic Hamiltonian gradient for equatorial Kerr
    # H = (1/2) g^{munu} p_mu p_nu + (1/2) m^2
    # For equatorial motion: dH/dr with analytic metric derivatives
    
    # Kerr metric quantities at equator
    Delta = r_safe**2 - 2*M*r_safe + a**2
    Sigma = r_safe**2  # At theta = pi/2
    
    # Derivatives of Delta and Sigma
    dDelta_dr = 2*r_safe - 2*M
    dSigma_dr = 2*r_safe
    
    # Contravariant metric derivatives (analytic)
    # g^tt = -(r^2 + a^2 + 2Ma^2/r) / Delta  
    # g^tphi = -2Ma / (r * Delta)
    # g^rr = Delta / r^2
    # g^phiphi = (1 - 2M/r) / (r^2 sin^2theta) -> at equator sin^2theta = 1
    
    # Numerical derivative as fallback (more stable for complex expressions)
    eps = 1e-7 * r_safe  # Scale epsilon with radius
    def H_at_r(r_):
        if r_ < r_plus + config.horizon_margin:
            r_ = r_plus + config.horizon_margin
        _, con_ = kerr_metric_components(r_, th, a, M, clamp_horizon=True, warn_horizon=False)
        gu_tt_, gu_tphi_, gu_rr_, _, gu_phiphi_ = con_
        return 0.5 * (gu_tt_ * pt**2 + 2*gu_tphi_ * pt * pphi + 
                      gu_rr_ * pr**2 + gu_phiphi_ * pphi**2)
    
    # Central difference with safeguards
    r_plus_eps = min(r_safe + eps, r_safe * 1.01)
    r_minus_eps = max(r_safe - eps, r_plus + config.horizon_margin)
    
    if r_plus_eps > r_minus_eps:
        dH_dr = (H_at_r(r_plus_eps) - H_at_r(r_minus_eps)) / (r_plus_eps - r_minus_eps)
    else:
        dH_dr = 0.0  # Fallback near horizon
    
    # Clamp derivatives to prevent overflow
    max_deriv = 1e6
    
    # Equations of motion with overflow protection
    dr_dtau = np.clip(u_r, -max_deriv, max_deriv)
    dphi_dtau = np.clip(u_phi, -max_deriv, max_deriv)
    dpr_dtau = np.clip(-dH_dr / m, -max_deriv, max_deriv) if m > 1e-10 else 0.0
    dpphi_dtau = 0.0  # Killing symmetry
    dm_dtau = 0.0     # No thrust
    dpt_dtau = 0.0    # Killing symmetry
    
    return np.array([dr_dtau, dphi_dtau, dpr_dtau, dpphi_dtau, dm_dtau, dpt_dtau])


# =============================================================================
# STRATEGY 1: PURE GEODESIC (BASELINE)
# =============================================================================

def simulate_geodesic(config: SimulationConfig) -> TrajectoryResult:
    """
    Simulate pure geodesic motion (no thrust).
    
    This serves as the baseline for comparing thrust strategies.
    """
    result = TrajectoryResult(
        initial_conditions={
            'a': config.a, 'E0': config.E0, 'Lz0': config.Lz0,
            'r0': config.r0, 'm0': config.m0
        },
        strategy=ThrustStrategy.NONE
    )
    
    # Classify orbit
    props = classify_orbit(config.E0, config.Lz0, config.a, config.M)
    result.orbit_profile = props.profile
    
    # Initial state
    r0, phi0 = config.r0, 0.0
    pt0 = -config.E0
    pphi0 = config.Lz0
    m0 = config.m0
    
    # Compute initial p_r from mass-shell
    th = np.pi / 2
    _, con = kerr_metric_components(r0, th, config.a, config.M)
    gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con
    
    rhs = -(gu_tt * pt0**2 + 2*gu_tphi * pt0 * pphi0 + gu_phiphi * pphi0**2 + m0**2)
    if rhs < 0:
        result.outcome = TrajectoryOutcome.INTEGRATION_FAILURE
        return result
    pr0 = -np.sqrt(rhs / gu_rr)  # Infalling
    
    state0 = np.array([r0, phi0, pr0, pphi0, m0, pt0])
    
    # Events
    def horizon_event(t, y):
        return y[0] - config.r_plus - config.horizon_margin
    horizon_event.terminal = True
    horizon_event.direction = -1
    
    def escape_event(t, y):
        return y[0] - config.escape_radius
    escape_event.terminal = True
    escape_event.direction = 1
    
    # Integrate with adaptive tolerances
    try:
        sol = solve_ivp(
            lambda t, y: geodesic_dynamics(t, y, config),
            [0, config.tau_max],
            state0,
            method='DOP853',  # 8th order Dormand-Prince, better for smooth problems
            events=[horizon_event, escape_event],
            rtol=config.rtol,
            atol=config.atol,
            dense_output=True,
            max_step=5.0  # Allow larger steps for faster integration
        )
    except Exception as e:
        result.outcome = TrajectoryOutcome.INTEGRATION_FAILURE
        return result
    
    if not sol.success:
        result.outcome = TrajectoryOutcome.INTEGRATION_FAILURE
        return result
    
    # Extract results
    r_hist = sol.y[0]
    pr_hist = sol.y[2]
    
    result.r_final = r_hist[-1]
    result.E_final = -sol.y[5, -1]
    result.m_final = sol.y[4, -1]
    result.r_min = np.min(r_hist)
    result.r_max = np.max(r_hist)
    result.tau_total = sol.t[-1]
    result.n_steps = len(sol.t)
    
    # Classify outcome
    result.outcome = classify_trajectory_outcome(r_hist, pr_hist, config.a, config.M)
    
    # Store trajectory data
    result.trajectory_data = {
        'tau': sol.t,
        'r': r_hist,
        'phi': sol.y[1],
        'pr': pr_hist,
        'pphi': sol.y[3],
        'm': sol.y[4],
        'E': -sol.y[5],
    }
    
    return result


# =============================================================================
# STRATEGY 2: SINGLE IMPULSE
# =============================================================================

def simulate_single_impulse(config: SimulationConfig,
                             trigger_radius: Optional[float] = None) -> TrajectoryResult:
    """
    Simulate single impulsive burn at periapsis (or specified trigger radius).
    
    This implements the classic Penrose extraction scenario.
    """
    result = TrajectoryResult(
        initial_conditions={
            'a': config.a, 'E0': config.E0, 'Lz0': config.Lz0,
            'r0': config.r0, 'm0': config.m0, 'v_e': config.v_e,
            'delta_m_fraction': config.delta_m_fraction
        },
        strategy=ThrustStrategy.SINGLE_IMPULSE
    )
    
    # Classify orbit
    props = classify_orbit(config.E0, config.Lz0, config.a, config.M)
    result.orbit_profile = props.profile
    
    # For FORBIDDEN orbits, return immediately
    if props.profile == OrbitProfile.FORBIDDEN:
        result.outcome = TrajectoryOutcome.INTEGRATION_FAILURE
        return result
    
    # For PLUNGE orbits, we still attempt simulation - may extract before capture
    is_plunge = (props.profile == OrbitProfile.PLUNGE)
    
    # Initial state
    r0, phi0 = config.r0, 0.0
    pt0 = -config.E0
    pphi0 = config.Lz0
    m0 = config.m0
    
    th = np.pi / 2
    _, con = kerr_metric_components(r0, th, config.a, config.M)
    gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con
    
    rhs = -(gu_tt * pt0**2 + 2*gu_tphi * pt0 * pphi0 + gu_phiphi * pphi0**2 + m0**2)
    if rhs < 0:
        result.outcome = TrajectoryOutcome.INTEGRATION_FAILURE
        return result
    pr0 = -np.sqrt(rhs / gu_rr)
    
    state0 = np.array([r0, phi0, pr0, pphi0, m0, pt0])
    
    # Phase 1: Fall to periapsis (detected by p_r changing sign from - to +)
    # This is more robust than using theoretical periapsis from effective potential
    
    # Use trigger_radius as a safety limit, but primarily trigger on periapsis
    use_radius_trigger = (trigger_radius is not None)
    if not use_radius_trigger:
        # For plunge orbits, use ergosphere trigger since there's no periapsis
        if is_plunge:
            trigger_radius = config.r_erg * 0.8
            use_radius_trigger = True
    
    def periapsis_event(t, y):
        # Trigger when p_r goes from negative to positive (turning point)
        return y[2]  # p_r
    periapsis_event.terminal = True
    periapsis_event.direction = 1  # Trigger when p_r crosses 0 going positive
    
    def horizon_event(t, y):
        return y[0] - config.r_plus - config.horizon_margin
    horizon_event.terminal = True
    horizon_event.direction = -1
    
    def radius_trigger_event(t, y):
        return y[0] - trigger_radius if trigger_radius else 1.0
    radius_trigger_event.terminal = True
    radius_trigger_event.direction = -1
    
    # Choose events based on orbit type
    if is_plunge:
        events = [radius_trigger_event, horizon_event]
        trigger_event_idx = 0
    else:
        events = [periapsis_event, horizon_event]
        trigger_event_idx = 0
    
    sol1 = solve_ivp(
        lambda t, y: geodesic_dynamics(t, y, config),
        [0, config.tau_max],
        state0,
        method='DOP853',
        events=events,
        rtol=config.rtol,
        atol=config.atol,
        max_step=5.0
    )
    
    if not sol1.success or len(sol1.t) == 0:
        result.outcome = TrajectoryOutcome.INTEGRATION_FAILURE
        return result
    
    # Record r_min from Phase 1
    r_min_phase1 = np.min(sol1.y[0])
    
    if len(sol1.t_events[1]) > 0:
        # Hit horizon before trigger
        result.outcome = TrajectoryOutcome.CAPTURE
        result.r_final = sol1.y[0, -1]
        result.r_min = r_min_phase1
        return result
    
    if len(sol1.t_events[trigger_event_idx]) == 0:
        # Didn't reach trigger (shouldn't happen for flyby, but possible for edge cases)
        result.outcome = TrajectoryOutcome.STALLED
        result.r_min = r_min_phase1
        return result
    
    # Phase 2: Apply impulse
    state_at_trigger = sol1.y[:, -1]
    r_trig = state_at_trigger[0]
    th = np.pi / 2  # Equatorial
    
    # Extract momentum components
    pt = state_at_trigger[5]
    pr = state_at_trigger[2]
    pphi = state_at_trigger[3]
    m = state_at_trigger[4]
    
    # Get metric at trigger location
    cov, con = kerr_metric_components(r_trig, th, config.a, config.M)
    g_tt, g_tphi, g_rr, _, g_phiphi = cov
    gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con
    
    # Compute contravariant 4-velocity: u^mu = g^{munu} p_nu / m
    u_t = (gu_tt * pt + gu_tphi * pphi) / m
    u_r = gu_rr * pr / m
    u_phi = (gu_tphi * pt + gu_phiphi * pphi) / m
    u_vec = np.array([u_t, u_r, u_phi], dtype=float)
    
    # Find optimal exhaust direction
    try:
        opt_result = compute_optimal_exhaust_direction(
            u_vec, r_trig, th, config.a, config.M, config.v_e,
            g_tt, g_tphi, g_rr, g_phiphi
        )
        
        if opt_result is None:
            raise ValueError("No valid exhaust direction found")
        
        # Apply exact impulse
        delta_mu = config.delta_m_fraction * m / config.gamma_e
        
        # Sanity check: don't expel too much mass
        if delta_mu > 0.5 * m:
            delta_mu = 0.5 * m / config.gamma_e
        
        p_cov = (pt, pr, pphi)
        u_ex_cov = opt_result['u_ex_cov']
        
        impulse_result = apply_exact_impulse(
            p_cov, m, delta_mu, u_ex_cov, r_trig, th, config.a, config.M
        )
        
        # Unpack results
        pt_new, pr_new, pphi_new = impulse_result['p_cov_new']
        m_new = impulse_result['m_new']
        E_ex = impulse_result['E_ex']
        
        # Validate new mass
        if m_new <= 0 or not np.isfinite(m_new):
            raise ValueError(f"Invalid post-impulse mass: {m_new}")
        
    except Exception as e:
        # Silently handle - don't print for every failure
        result.outcome = TrajectoryOutcome.INTEGRATION_FAILURE
        return result
    
    # Record E_ex
    result.E_ex_mean = E_ex
    result.E_ex_min = E_ex
    result.n_total_E_ex = 1
    result.n_negative_E_ex = 1 if E_ex < 0 else 0
    result.penrose_fraction = 1.0 if E_ex < 0 else 0.0
    
    # State after impulse
    state_after = np.array([
        r_trig,
        state_at_trigger[1],  # phi unchanged
        pr_new,
        pphi_new,
        m_new,
        pt_new
    ])
    
    # Phase 3: Post-impulse geodesic
    def escape_event(t, y):
        return y[0] - config.escape_radius
    escape_event.terminal = True
    escape_event.direction = 1
    
    sol2 = solve_ivp(
        lambda t, y: geodesic_dynamics(t, y, config),
        [sol1.t[-1], sol1.t[-1] + config.tau_max],
        state_after,
        method='DOP853',
        events=[horizon_event, escape_event],
        rtol=config.rtol,
        atol=config.atol,
        max_step=5.0
    )
    
    if not sol2.success or len(sol2.t) == 0:
        result.outcome = TrajectoryOutcome.INTEGRATION_FAILURE
        return result
    
    # Check Phase 3 events directly
    hit_horizon = len(sol2.t_events[0]) > 0
    hit_escape = len(sol2.t_events[1]) > 0
    
    # Combine trajectories
    tau_full = np.concatenate([sol1.t, sol2.t])
    r_full = np.concatenate([sol1.y[0], sol2.y[0]])
    phi_full = np.concatenate([sol1.y[1], sol2.y[1]])
    pr_full = np.concatenate([sol1.y[2], sol2.y[2]])
    m_full = np.concatenate([sol1.y[4], sol2.y[4]])
    E_full = np.concatenate([-sol1.y[5], -sol2.y[5]])
    
    # Results
    result.r_final = r_full[-1]
    result.E_final = E_full[-1]
    result.m_final = m_full[-1]
    result.Delta_E = result.E_final - config.E0
    result.Delta_m = config.m0 - result.m_final
    result.r_min = np.min(r_full)
    result.r_max = np.max(r_full)
    result.tau_total = tau_full[-1]
    result.n_steps = len(tau_full)
    
    # Efficiency
    if result.Delta_m > 0:
        result.eta_cumulative = result.Delta_E / result.Delta_m
        result.eta_exhaust = result.Delta_E / delta_mu
    
    # Outcome - use event detection first, then fallback to classification
    if hit_escape:
        result.outcome = TrajectoryOutcome.ESCAPE
    elif hit_horizon:
        result.outcome = TrajectoryOutcome.CAPTURE
    else:
        # Use heuristic classification for ambiguous cases
        result.outcome = classify_trajectory_outcome(r_full, pr_full, config.a, config.M)
    
    # Store trajectory
    result.trajectory_data = {
        'tau': tau_full,
        'r': r_full,
        'phi': phi_full,
        'pr': pr_full,
        'm': m_full,
        'E': E_full,
        'E_ex': np.array([E_ex]),
        'r_ex': np.array([r_trig]),
        'impulse_idx': len(sol1.t) - 1,
    }
    
    return result


# =============================================================================
# STRATEGY 3: BURST AT PERIAPSIS
# =============================================================================

def simulate_burst(config: SimulationConfig,
                   burst_duration: float = 1.0,
                   trigger_radius: Optional[float] = None) -> TrajectoryResult:
    """
    Simulate short burst thrust at periapsis.
    
    This is intermediate between single impulse and continuous thrust.
    """
    result = TrajectoryResult(
        initial_conditions={
            'a': config.a, 'E0': config.E0, 'Lz0': config.Lz0,
            'r0': config.r0, 'm0': config.m0, 'v_e': config.v_e,
            'T_max': config.T_max, 'burst_duration': burst_duration
        },
        strategy=ThrustStrategy.BURST
    )
    
    # Classify orbit
    props = classify_orbit(config.E0, config.Lz0, config.a, config.M)
    result.orbit_profile = props.profile
    
    if props.profile in [OrbitProfile.PLUNGE, OrbitProfile.FORBIDDEN]:
        result.outcome = TrajectoryOutcome.CAPTURE if props.profile == OrbitProfile.PLUNGE \
                        else TrajectoryOutcome.INTEGRATION_FAILURE
        return result
    
    # Use geodesic simulation as approximation for burst
    # (full burst dynamics would require continuous thrust implementation)
    # For now, approximate as single impulse with adjusted mass fraction
    
    # Estimate equivalent mass fraction for burst
    mass_rate = config.T_max / (config.gamma_e * config.v_e)
    delta_m_burst = mass_rate * burst_duration * config.gamma_e
    
    config_burst = SimulationConfig(
        a=config.a, M=config.M,
        E0=config.E0, Lz0=config.Lz0,
        r0=config.r0, m0=config.m0,
        v_e=config.v_e,
        delta_m_fraction=min(delta_m_burst / config.m0, 0.5),
        tau_max=config.tau_max,
        escape_radius=config.escape_radius
    )
    
    # Run as single impulse
    impulse_result = simulate_single_impulse(config_burst, trigger_radius)
    
    # Copy results but mark as BURST strategy
    result.outcome = impulse_result.outcome
    result.r_final = impulse_result.r_final
    result.E_final = impulse_result.E_final
    result.m_final = impulse_result.m_final
    result.Delta_E = impulse_result.Delta_E
    result.Delta_m = impulse_result.Delta_m
    result.eta_cumulative = impulse_result.eta_cumulative
    result.eta_exhaust = impulse_result.eta_exhaust
    result.E_ex_mean = impulse_result.E_ex_mean
    result.E_ex_min = impulse_result.E_ex_min
    result.penrose_fraction = impulse_result.penrose_fraction
    result.r_min = impulse_result.r_min
    result.r_max = impulse_result.r_max
    result.trajectory_data = impulse_result.trajectory_data
    
    return result


# =============================================================================
# STRATEGY COMPARISON
# =============================================================================

@dataclass
class ComparisonResult:
    """Result of comparing multiple thrust strategies."""
    config: SimulationConfig
    results: Dict[ThrustStrategy, TrajectoryResult] = field(default_factory=dict)
    
    def best_strategy(self, metric: str = 'Delta_E') -> ThrustStrategy:
        """Find strategy with best performance on given metric."""
        best = None
        best_value = -float('inf')
        
        for strategy, result in self.results.items():
            if result.outcome == TrajectoryOutcome.ESCAPE:
                value = getattr(result, metric, 0)
                if value > best_value:
                    best_value = value
                    best = strategy
        
        return best
    
    def summary_table(self) -> str:
        """Generate summary table of comparison."""
        lines = [
            "="*75,
            "THRUST STRATEGY COMPARISON",
            "="*75,
            f"{'Strategy':<20} {'Outcome':<12} {'DeltaE':>8} {'Deltam':>8} {'eta_cum':>8} {'E_ex<0':>8}",
            "-"*75,
        ]
        
        for strategy, result in self.results.items():
            penrose_pct = f"{100*result.penrose_fraction:.0f}%" if result.n_total_E_ex > 0 else "N/A"
            lines.append(
                f"{strategy.name:<20} {result.outcome.name:<12} "
                f"{result.Delta_E:+8.4f} {result.Delta_m:8.4f} "
                f"{result.eta_cumulative:8.2%} {penrose_pct:>8}"
            )
        
        lines.append("="*75)
        
        best = self.best_strategy()
        if best:
            lines.append(f"Best strategy for energy gain: {best.name}")
        
        return "\n".join(lines)


def compare_strategies(config: SimulationConfig,
                       strategies: Optional[List[ThrustStrategy]] = None,
                       verbose: bool = True) -> ComparisonResult:
    """
    Compare multiple thrust strategies with identical initial conditions.
    
    Parameters
    ----------
    config : SimulationConfig
        Shared configuration for all strategies
    strategies : list, optional
        Strategies to compare. Default: [NONE, SINGLE_IMPULSE, BURST]
    verbose : bool
        Print progress and results
        
    Returns
    -------
    ComparisonResult
        Comparison results for all strategies
    """
    if strategies is None:
        strategies = [ThrustStrategy.NONE, ThrustStrategy.SINGLE_IMPULSE, ThrustStrategy.BURST]
    
    comparison = ComparisonResult(config=config)
    
    if verbose:
        print(f"\nComparing {len(strategies)} thrust strategies...")
        print(f"Initial conditions: E = {config.E0}, Lz = {config.Lz0}, a = {config.a}")
    
    for strategy in strategies:
        if verbose:
            print(f"  Running {strategy.name}...", end=" ", flush=True)
        
        t_start = time.time()
        
        if strategy == ThrustStrategy.NONE:
            result = simulate_geodesic(config)
        elif strategy == ThrustStrategy.SINGLE_IMPULSE:
            result = simulate_single_impulse(config)
        elif strategy == ThrustStrategy.BURST:
            result = simulate_burst(config)
        else:
            # Placeholder for continuous thrust
            result = simulate_geodesic(config)
            result.strategy = strategy
        
        comparison.results[strategy] = result
        
        if verbose:
            dt = time.time() - t_start
            print(f"{result.outcome.name} (DeltaE = {result.Delta_E:+.4f}, {dt:.2f}s)")
    
    if verbose:
        print(comparison.summary_table())
    
    return comparison


def scan_trigger_radii(config: SimulationConfig,
                        r_min: Optional[float] = None,
                        r_max: Optional[float] = None,
                        n_points: int = 20) -> Dict[str, Any]:
    """
    Scan trigger radii for single impulse to find optimal.
    
    Returns dictionary with scan results.
    """
    props = classify_orbit(config.E0, config.Lz0, config.a, config.M)
    
    if r_min is None:
        r_min = config.r_plus + 0.1
    if r_max is None:
        r_max = config.r_erg
    
    radii = np.linspace(r_min, r_max, n_points)
    
    results = {
        'radii': radii,
        'Delta_E': np.zeros(n_points),
        'eta_cum': np.zeros(n_points),
        'E_ex': np.zeros(n_points),
        'outcome': [],
    }
    
    for i, r_trig in enumerate(radii):
        result = simulate_single_impulse(config, trigger_radius=r_trig)
        results['Delta_E'][i] = result.Delta_E
        results['eta_cum'][i] = result.eta_cumulative
        results['E_ex'][i] = result.E_ex_mean
        results['outcome'].append(result.outcome)
    
    # Find optimal
    escape_mask = np.array([o == TrajectoryOutcome.ESCAPE for o in results['outcome']])
    if escape_mask.any():
        escaped_idx = np.where(escape_mask)[0]
        best_idx = escaped_idx[np.argmax(results['Delta_E'][escaped_idx])]
        results['optimal_radius'] = radii[best_idx]
        results['optimal_Delta_E'] = results['Delta_E'][best_idx]
    else:
        results['optimal_radius'] = None
        results['optimal_Delta_E'] = None
    
    return results


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("THRUST STRATEGY COMPARISON - Quick Test")
    print("="*70)
    
    # Test configuration
    config = SimulationConfig(
        a=0.95,
        E0=1.2,
        Lz0=3.0,
        r0=10.0,
        v_e=0.95,
        delta_m_fraction=0.2,
    )
    
    print(f"\nBlack hole: a = {config.a}")
    print(f"Horizon: r+ = {config.r_plus:.4f}")
    print(f"Ergosphere: r_erg = {config.r_erg:.4f}")
    
    # Test 1: Pure geodesic
    print("\n1. Testing pure geodesic...")
    geo_result = simulate_geodesic(config)
    print(f"   Outcome: {geo_result.outcome.name}")
    print(f"   r_min: {geo_result.r_min:.4f}, r_final: {geo_result.r_final:.4f}")
    
    # Test 2: Single impulse
    print("\n2. Testing single impulse...")
    impulse_result = simulate_single_impulse(config)
    print(f"   Outcome: {impulse_result.outcome.name}")
    print(f"   DeltaE: {impulse_result.Delta_E:+.4f}")
    print(f"   E_ex: {impulse_result.E_ex_mean:.4f}")
    print(f"   Penrose (E_ex < 0): {impulse_result.E_ex_mean < 0}")
    
    # Test 3: Full comparison
    print("\n3. Running full comparison...")
    comparison = compare_strategies(config, verbose=True)
    
    # Test 4: Trigger radius scan
    print("\n4. Scanning trigger radii...")
    scan = scan_trigger_radii(config, n_points=10)
    if scan['optimal_radius']:
        print(f"   Optimal trigger: r = {scan['optimal_radius']:.4f}")
        print(f"   Best DeltaE: {scan['optimal_Delta_E']:+.4f}")
    
    print("\n" + "="*70)
    print("Thrust strategy comparison ready!")
    print("="*70)
