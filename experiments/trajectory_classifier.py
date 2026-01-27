"""
Trajectory Classification Module
=================================
Classify orbits and trajectory outcomes for Penrose process analysis.

Orbit profiles are classified based on geodesic structure (no thrust),
while trajectory outcomes incorporate thrust effects.

References:
- Bardeen, Press & Teukolsky (1972), ApJ 178, 347
- Chandrasekhar (1983), The Mathematical Theory of Black Holes
"""

import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from scipy.optimize import brentq

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kerr_utils import (
    kerr_metric_components, horizon_radius, ergosphere_radius,
    compute_pt_from_mass_shell, frame_dragging_omega, isco_radius
)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class OrbitProfile(Enum):
    """
    Classification of geodesic orbit types (no thrust).
    
    Based on effective potential analysis in the equatorial plane.
    """
    FLYBY_DEEP_ERGOSPHERE = auto()    # Periapsis in extraction zone (r < 0.85*r_erg)
    FLYBY_SHALLOW_ERGOSPHERE = auto() # Periapsis in ergosphere but outside extraction zone
    FLYBY_OUTSIDE_ERGOSPHERE = auto() # Periapsis outside ergosphere
    BOUND_STABLE = auto()              # Stable bound orbit (E < 1, above ISCO)
    BOUND_UNSTABLE = auto()            # Unstable bound orbit (between ISCO and horizon)
    PLUNGE = auto()                    # No turning point, falls into horizon
    FORBIDDEN = auto()                 # Classically forbidden configuration
    CIRCULAR = auto()                  # Special case: circular orbit


class TrajectoryOutcome(Enum):
    """
    Final outcome of a trajectory with or without thrust.
    """
    ESCAPE = auto()           # Reached r >> r_erg with outward velocity
    CAPTURE = auto()          # Fell into horizon
    BOUND = auto()            # Remains on bound orbit
    STALLED = auto()          # Stopped (mass depleted, etc.) before outcome
    INTEGRATION_FAILURE = auto()  # Numerical issues


class ThrustStrategy(Enum):
    """
    Thrust strategies for Penrose extraction.
    """
    NONE = auto()              # Pure geodesic (no thrust)
    SINGLE_IMPULSE = auto()    # One impulsive burn at optimal point
    CONTINUOUS = auto()        # Sustained thrust in ergosphere
    BURST = auto()             # Short burst at periapsis
    TWO_PHASE = auto()         # Extraction phase + escape phase


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OrbitProperties:
    """Properties of a geodesic orbit configuration."""
    E0: float                          # Initial Killing energy
    Lz0: float                         # Initial angular momentum
    a: float                           # Black hole spin
    M: float = 1.0                     # Black hole mass (normalized)
    
    # Derived properties (computed)
    profile: OrbitProfile = OrbitProfile.FORBIDDEN
    r_periapsis: Optional[float] = None
    r_apoapsis: Optional[float] = None
    in_ergosphere: bool = False
    in_extraction_zone: bool = False
    is_prograde: bool = True
    is_bound: bool = False
    
    # Key radii for reference
    r_horizon: float = field(init=False)
    r_ergosphere: float = field(init=False)
    r_isco_pro: float = field(init=False)
    r_isco_retro: float = field(init=False)
    
    def __post_init__(self):
        self.r_horizon = horizon_radius(self.a, self.M)
        self.r_ergosphere = ergosphere_radius(np.pi/2, self.a, self.M)
        self.r_isco_pro = isco_radius(self.a, self.M, prograde=True)
        self.r_isco_retro = isco_radius(self.a, self.M, prograde=False)
        self.is_prograde = self.Lz0 >= 0
        self.is_bound = self.E0 < 1.0


@dataclass
class TrajectoryResult:
    """Complete result of a trajectory simulation."""
    # Initial conditions
    initial_conditions: Dict[str, float] = field(default_factory=dict)
    
    # Final state
    outcome: TrajectoryOutcome = TrajectoryOutcome.INTEGRATION_FAILURE
    r_final: float = 0.0
    E_final: float = 0.0
    m_final: float = 0.0
    
    # Penrose extraction metrics
    Delta_E: float = 0.0               # Energy gain
    Delta_m: float = 0.0               # Mass expelled
    eta_cumulative: float = 0.0        # Cumulative efficiency
    eta_exhaust: float = 0.0           # Exhaust-mass efficiency
    
    # Exhaust energy statistics
    E_ex_mean: float = 0.0
    E_ex_min: float = 0.0
    n_negative_E_ex: int = 0
    n_total_E_ex: int = 0
    penrose_fraction: float = 0.0      # Fraction with E_ex < 0
    
    # Trajectory statistics
    r_min: float = 0.0                 # Minimum radius reached
    r_max: float = 0.0                 # Maximum radius reached  
    tau_total: float = 0.0             # Total proper time
    n_steps: int = 0
    
    # Orbit classification
    orbit_profile: OrbitProfile = OrbitProfile.FORBIDDEN
    
    # Strategy used
    strategy: ThrustStrategy = ThrustStrategy.NONE
    
    # Raw data (optional, for detailed analysis)
    trajectory_data: Optional[Dict[str, np.ndarray]] = None


# =============================================================================
# EFFECTIVE POTENTIAL FUNCTIONS
# =============================================================================

def compute_effective_potential(E: float, Lz: float, r: float, 
                                  a: float, M: float = 1.0) -> float:
    """
    Compute effective potential for radial motion at equator.
    
    For timelike geodesics: p_r^2 = -g^rr * V_eff(r)
    
    V_eff < 0 means motion is allowed (p_r^2 > 0)
    V_eff = 0 is a turning point
    V_eff > 0 is classically forbidden
    
    Parameters
    ----------
    E : float
        Killing energy at infinity (-p_t)
    Lz : float
        Angular momentum (p_phi)
    r : float
        Radial coordinate
    a, M : float
        Black hole parameters
        
    Returns
    -------
    float
        V_eff value (motion allowed where V_eff < 0)
    """
    th = np.pi / 2
    _, con = kerr_metric_components(r, th, a, M, clamp_horizon=True, warn_horizon=False)
    gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con
    
    # V_eff from mass-shell: p_r^2 = g^rr * [-(g^tt E^2 - 2 g^tphi E Lz + g^phiphi Lz^2 + 1)]
    V = gu_tt * E**2 - 2*gu_tphi * E * Lz + gu_phiphi * Lz**2 + 1.0
    return V


def find_turning_points(E: float, Lz: float, a: float, M: float = 1.0,
                         r_min: Optional[float] = None, r_max: float = 100.0,
                         n_samples: int = 1000) -> Tuple[List[float], List[float]]:
    """
    Find radial turning points (periapsis and apoapsis).
    
    Turning points occur where V_eff = 0 and changes sign.
    - Periapsis: V_eff goes from negative (allowed) to positive (forbidden)
    - Apoapsis: V_eff goes from positive (forbidden) to negative (allowed)
    
    Returns
    -------
    tuple
        (periapses, apoapses) - lists of radii
    """
    r_plus = horizon_radius(a, M)
    if r_min is None:
        r_min = r_plus + 0.01
    
    radii = np.linspace(r_min, r_max, n_samples)
    V = np.array([compute_effective_potential(E, Lz, r, a, M) for r in radii])
    
    periapses = []
    apoapses = []
    
    for i in range(len(V) - 1):
        # Periapsis: V goes from - to + (allowed -> forbidden as r decreases)
        # At this point, coming inward, the particle bounces back
        if V[i] <= 0 and V[i+1] > 0:
            try:
                r_turn = brentq(
                    lambda r: compute_effective_potential(E, Lz, r, a, M),
                    radii[i], radii[i+1]
                )
                periapses.append(r_turn)
            except (ValueError, RuntimeError):
                pass
        
        # Apoapsis: V goes from + to - (forbidden -> allowed as r increases)
        # At this point, moving outward, the particle turns back
        if V[i] > 0 and V[i+1] <= 0:
            try:
                r_turn = brentq(
                    lambda r: compute_effective_potential(E, Lz, r, a, M),
                    radii[i], radii[i+1]
                )
                apoapses.append(r_turn)
            except (ValueError, RuntimeError):
                pass
    
    return periapses, apoapses


# =============================================================================
# ORBIT CLASSIFICATION
# =============================================================================

def classify_orbit(E: float, Lz: float, a: float, M: float = 1.0,
                   extraction_zone_factor: float = 0.85) -> OrbitProperties:
    """
    Classify a geodesic orbit based on its effective potential.
    
    Parameters
    ----------
    E : float
        Killing energy at infinity
    Lz : float
        Angular momentum
    a, M : float
        Black hole parameters
    extraction_zone_factor : float
        Fraction of ergosphere radius defining "deep" extraction zone.
        Default 0.85 means r < 0.85 * r_erg is considered deep.
        
    Returns
    -------
    OrbitProperties
        Complete orbit classification with properties
    """
    props = OrbitProperties(E0=E, Lz0=Lz, a=a, M=M)
    
    r_plus = props.r_horizon
    r_erg = props.r_ergosphere
    r_extraction = extraction_zone_factor * r_erg
    
    # Find turning points
    periapses, apoapses = find_turning_points(E, Lz, a, M)
    
    # Check if motion is allowed near horizon
    V_near_horizon = compute_effective_potential(E, Lz, r_plus + 0.05, a, M)
    
    # Classification logic
    if not periapses:
        # No periapsis - either plunge or forbidden
        if V_near_horizon < 0:
            props.profile = OrbitProfile.PLUNGE
        else:
            props.profile = OrbitProfile.FORBIDDEN
    else:
        r_peri = periapses[0]  # Innermost periapsis
        props.r_periapsis = r_peri
        
        if apoapses:
            props.r_apoapsis = apoapses[-1]  # Outermost apoapsis
        
        # Check if periapsis is above horizon
        if r_peri <= r_plus:
            props.profile = OrbitProfile.PLUNGE
        elif r_peri < r_extraction:
            props.profile = OrbitProfile.FLYBY_DEEP_ERGOSPHERE
            props.in_ergosphere = True
            props.in_extraction_zone = True
        elif r_peri < r_erg:
            props.profile = OrbitProfile.FLYBY_SHALLOW_ERGOSPHERE
            props.in_ergosphere = True
            props.in_extraction_zone = False
        else:
            props.profile = OrbitProfile.FLYBY_OUTSIDE_ERGOSPHERE
            props.in_ergosphere = False
            
        # Check for bound orbits
        if apoapses and props.is_bound:
            r_isco = props.r_isco_pro if props.is_prograde else props.r_isco_retro
            if r_peri > r_isco:
                props.profile = OrbitProfile.BOUND_STABLE
            else:
                props.profile = OrbitProfile.BOUND_UNSTABLE
    
    return props


def classify_trajectory_outcome(r_history: np.ndarray, 
                                  pr_history: Optional[np.ndarray],
                                  a: float, M: float = 1.0,
                                  escape_radius: float = 50.0,
                                  horizon_margin: float = 0.02) -> TrajectoryOutcome:
    """
    Classify the outcome of an integrated trajectory.
    
    Parameters
    ----------
    r_history : ndarray
        Radial coordinate history
    pr_history : ndarray, optional
        Radial momentum history (for checking outward motion)
    a, M : float
        Black hole parameters
    escape_radius : float
        Radius threshold for definitive escape (default 50M)
    horizon_margin : float
        Margin above horizon for capture detection
        
    Returns
    -------
    TrajectoryOutcome
        Classification of trajectory outcome
    """
    r_plus = horizon_radius(a, M)
    r_final = r_history[-1]
    r_min = np.min(r_history)
    r_max = np.max(r_history)
    
    # Check for capture (reached near horizon)
    if r_final < r_plus + horizon_margin or r_min < r_plus + horizon_margin:
        return TrajectoryOutcome.CAPTURE
    
    # Check for escape (reached escape radius moving outward)
    if r_final > escape_radius:
        if pr_history is not None and len(pr_history) > 0:
            # Confirm outward motion
            if pr_history[-1] > 0:
                return TrajectoryOutcome.ESCAPE
            else:
                return TrajectoryOutcome.BOUND
        else:
            return TrajectoryOutcome.ESCAPE
    
    # Check if trajectory is clearly outward-bound (high r and moving out)
    if pr_history is not None and len(pr_history) > 1:
        if r_final > 0.8 * escape_radius and pr_history[-1] > 0:
            # Close to escape and moving outward - likely will escape
            return TrajectoryOutcome.ESCAPE
    
    # Check for bound orbit (has turned around)
    if r_max > 0.5 * escape_radius and r_final < r_max:
        return TrajectoryOutcome.BOUND
    
    # Check if still in strong field region
    r_erg = ergosphere_radius(np.pi/2, a, M)
    if r_final < 5 * r_erg:
        # Still near BH, integration may need more time
        return TrajectoryOutcome.STALLED
    
    return TrajectoryOutcome.BOUND


# =============================================================================
# ORBIT PROPERTY COMPUTATIONS
# =============================================================================

def compute_orbit_properties(E: float, Lz: float, a: float, M: float = 1.0) -> Dict[str, Any]:
    """
    Compute comprehensive orbit properties for analysis.
    
    Returns a dictionary with all relevant properties for the experiment pipeline.
    """
    props = classify_orbit(E, Lz, a, M)
    
    # Frame-dragging at periapsis (if exists)
    omega_peri = None
    if props.r_periapsis is not None:
        omega_peri = frame_dragging_omega(props.r_periapsis, a, M)
    
    # Compute specific orbital angular momentum
    l = Lz / E if E != 0 else float('inf')
    
    # Characteristic velocity at periapsis (coordinate basis)
    v_peri = None
    if props.r_periapsis is not None:
        th = np.pi / 2
        cov, con = kerr_metric_components(props.r_periapsis, th, a, M)
        gu_tt, gu_tphi, _, _, gu_phiphi = con
        
        # 4-velocity components at periapsis (pr = 0)
        pt = -E
        pphi = Lz
        u_t = gu_tt * pt + gu_tphi * pphi
        u_phi = gu_tphi * pt + gu_phiphi * pphi
        
        # Coordinate angular velocity
        if u_t != 0:
            omega_coord = u_phi / u_t
            v_peri = omega_coord * props.r_periapsis
    
    return {
        'E0': E,
        'Lz0': Lz,
        'a': a,
        'M': M,
        'profile': props.profile.name,
        'r_periapsis': props.r_periapsis,
        'r_apoapsis': props.r_apoapsis,
        'in_ergosphere': props.in_ergosphere,
        'in_extraction_zone': props.in_extraction_zone,
        'is_prograde': props.is_prograde,
        'is_bound': props.is_bound,
        'r_horizon': props.r_horizon,
        'r_ergosphere': props.r_ergosphere,
        'r_isco_pro': props.r_isco_pro,
        'r_isco_retro': props.r_isco_retro,
        'omega_periapsis': omega_peri,
        'specific_ang_mom': l,
        'v_periapsis': v_peri,
    }


# =============================================================================
# SPIN DEPENDENCE UTILITIES
# =============================================================================

# Recommended spin values for systematic study
SPIN_VALUES = [0.7, 0.9, 0.95, 0.99]

def compute_key_radii_vs_spin(spins: Optional[List[float]] = None, 
                               M: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Compute key radii as functions of spin parameter.
    
    Useful for understanding how the extraction zone evolves with spin.
    """
    if spins is None:
        spins = np.linspace(0.1, 0.998, 100)
    
    spins = np.array(spins)
    
    r_plus = np.array([horizon_radius(a, M) for a in spins])
    r_erg = np.array([ergosphere_radius(np.pi/2, a, M) for a in spins])
    r_isco_pro = np.array([isco_radius(a, M, prograde=True) for a in spins])
    r_isco_retro = np.array([isco_radius(a, M, prograde=False) for a in spins])
    
    return {
        'a': spins,
        'r_horizon': r_plus,
        'r_ergosphere': r_erg,
        'r_isco_prograde': r_isco_pro,
        'r_isco_retrograde': r_isco_retro,
        'ergosphere_width': r_erg - r_plus,
    }


# =============================================================================
# QUICK TESTS
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TRAJECTORY CLASSIFIER - Quick Test")
    print("="*70)
    
    # Test with known good parameters
    a = 0.95
    M = 1.0
    
    test_cases = [
        (1.2, 3.0, "Single thrust case"),
        (1.25, 3.1, "Continuous escape case"),
        (0.95, 2.5, "Bound orbit"),
        (1.5, 1.0, "Low angular momentum"),
        (1.1, 4.0, "High angular momentum"),
    ]
    
    print(f"\nBlack hole: a/M = {a}")
    print(f"Horizon: r+ = {horizon_radius(a, M):.4f}M")
    print(f"Ergosphere: r_erg = {ergosphere_radius(np.pi/2, a, M):.4f}M")
    print(f"ISCO (prograde): {isco_radius(a, M, prograde=True):.4f}M")
    print()
    
    for E, Lz, desc in test_cases:
        props = classify_orbit(E, Lz, a, M)
        print(f"{desc}:")
        print(f"  E = {E:.2f}, Lz = {Lz:+.2f}")
        print(f"  Profile: {props.profile.name}")
        if props.r_periapsis:
            print(f"  Periapsis: {props.r_periapsis:.4f}M", end="")
            if props.in_extraction_zone:
                print(" [EXTRACTION ZONE]")
            elif props.in_ergosphere:
                print(" [IN ERGOSPHERE]")
            else:
                print()
        print()
    
    # Spin dependence
    print("\n" + "="*70)
    print("KEY RADII VS SPIN")
    print("="*70)
    
    radii = compute_key_radii_vs_spin(SPIN_VALUES)
    print(f"\n{'a/M':>8} {'r+':>8} {'r_erg':>8} {'ISCO+':>8} {'Width':>8}")
    print("-"*44)
    for i, a_val in enumerate(radii['a']):
        print(f"{a_val:8.2f} {radii['r_horizon'][i]:8.4f} {radii['r_ergosphere'][i]:8.4f} "
              f"{radii['r_isco_prograde'][i]:8.4f} {radii['ergosphere_width'][i]:8.4f}")
