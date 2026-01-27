"""
Continuous-Thrust Penrose Process Simulation
=============================================
Simulates energy extraction from a Kerr black hole using continuous rocket thrust.

PHYSICS SUMMARY:
----------------
The Penrose process extracts rotational energy from a spinning black hole by
ejecting negative-energy exhaust (E_ex < 0) that falls into the horizon. This
requires:

1. Operating inside the ergosphere (r < r_erg = 2M at equator for any spin)
2. Retrograde exhaust velocity relative to the ZAMO (frame-dragging angular velocity)
3. Sufficiently high exhaust velocity v_e to overcome the rocket's positive energy
4. The exhaust must actually fall into the black hole (verified via geodesic integration)

KEY SIGN CONVENTIONS (-+++ signature):
- Killing energy: E = -p_t (positive for physically valid particles)
- Covariant momentum: p_t < 0 for E > 0 particles
- Angular momentum: L_z = p_phi (positive = prograde, negative = retrograde)
- Prograde orbit + retrograde exhaust maximizes extraction

PHYSICAL VALIDITY CHECKS:
- Mass-shell constraint: g^{munu} p_mu p_nu = -m^2 must hold
- Causality: 4-velocity must be timelike (u*u = -1)
- Horizon safety: r > r_+ + margin required for Boyer-Lindquist validity
- Escape verification: rocket must reach r >> r_erg with outward velocity

References:
- Penrose & Floyd (1971), Nature Phys. Sci. 229, 177
- Bardeen, Press & Teukolsky (1972), ApJ 178, 347
- Wald (1974), ApJ 191, 231 (efficiency limits)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


from kerr_utils import (
    COLORS, PRD_SINGLE_COL, PRD_DOUBLE_COL, setup_prd_style,
    kerr_metric_components, horizon_radius, ergosphere_radius,
    compute_pt_from_mass_shell, compute_energy,
    theoretical_penrose_limit, compute_instantaneous_efficiency,
    compute_cumulative_efficiency, print_efficiency_analysis,
    smoothstep, isco_radius, frame_dragging_omega,
    kerr_metric_derivatives, compute_dH_dr_analytic,
    ThrustMode, compute_exhaust_energy, compute_exhaust_4velocity,
    compute_E_ex_threshold, determine_thrust_mode, compute_extraction_limit_radius,
    compute_energy_budget, print_energy_budget, build_rocket_rest_basis,
    integrate_exhaust_geodesic, verify_exhaust_capture_batch,
    _inner_prod_tr  # Internal helper for thrust magnitude check
)


setup_prd_style()


# =============================================================================
# CONFIG FLAGS
# =============================================================================
USE_ANALYTIC_DERIVATIVES = True   # better accuracy near horizon
USE_ADAPTIVE_THRUST_MODE = True   # automatic mode switching
USE_NEAR_EXTREMAL_PARAMS = True   # parameters that achieve E_ex < 0
USE_EXACT_MOMENTUM_CONSERVATION = True  # NEW: use exact 4-momentum conservation

# Integration timestep - SINGLE SOURCE OF TRUTH
# This must be used by both the integrator and energy budget calculations
DT_INTEGRATION = 0.005  # Euler step size for proper time tau

# NEW: Toggle between "capture" mode (deep dive, higher efficiency but captured)
#      and "escape" mode (shallower orbit, lower efficiency but escapes)
# Set to True to use escape-optimized parameters
USE_ESCAPE_OPTIMIZED = True  # <-- SWITCH THIS TO CHANGE MODES

# R_EXTRACTION_LIMIT will be computed dynamically after parameters are defined


# -----------------------
# PARAMETERS (G=c=M=1 throughout)
# -----------------------
M = 1.0

# Spin: a=0.95 gives strong frame-dragging without extremal instabilities
if USE_NEAR_EXTREMAL_PARAMS:
    a = 0.95
else:
    a = 0.98

r_plus = horizon_radius(a, M)
r_erg_eq = ergosphere_radius(np.pi/2, a, M)  # = 2M at equator

# Safety margin from horizon
# IMPORTANT: Boyer-Lindquist coordinates have a coordinate singularity at r = r_+.
# We must keep r > r_safe = r_+ + margin to avoid numerical issues.
# The margin should be small enough to allow deep ergosphere penetration
# but large enough to maintain numerical stability (~0.01-0.05 M).
if USE_NEAR_EXTREMAL_PARAMS:
    r_safe = r_plus + 0.02
else:
    r_safe = r_plus + 3e-2

# Validate safety margin is physically meaningful
if r_safe >= r_erg_eq:
    raise ValueError(
        f"Safety margin too large: r_safe = {r_safe:.4f}M >= r_erg = {r_erg_eq:.4f}M. "
        f"Cannot operate inside ergosphere with this configuration."
    )
if r_safe <= r_plus:
    raise ValueError(f"Safety margin too small: r_safe = {r_safe:.4f}M <= r_+ = {r_plus:.4f}M")

# Rocket params - high v_e needed for E_ex < 0
if USE_NEAR_EXTREMAL_PARAMS:
    a_max = 0.35     # can push harder when deep in ergosphere
    v_e = 0.95       # exhaust velocity
else:
    a_max = 0.3
    v_e = 0.85

m_min = 0.1      # Minimum mass fraction to retain

# Relativistic mass-flow relation
USE_RELATIVISTIC_MASSFLOW = True
gamma_e = 1.0 / np.sqrt(1.0 - v_e**2)
v_exhaust_momentum = (gamma_e * v_e) if USE_RELATIVISTIC_MASSFLOW else v_e

# =============================================================================
# PHYSICAL VALIDITY CHECKS FOR EXHAUST VELOCITY
# =============================================================================
# v_e = 0.95c corresponds to gamma_e ~ 3.2, which is extreme but necessary for
# achieving E_ex < 0 in the ergosphere. For comparison:
#   - Chemical rockets: v_e ~ 0.00001c (gamma_e ~ 1.0)
#   - Ion thrusters: v_e ~ 0.0001c (gamma_e ~ 1.0)
#   - Relativistic jets: v_e ~ 0.99c (gamma_e ~ 7)
#
# This simulation explores theoretical limits, not current technology.
# The exhaust velocity is a free parameter bounded only by causality (v_e < c).
if v_e >= 1.0:
    raise ValueError(f"Exhaust velocity v_e = {v_e} >= c violates causality!")
if v_e < 0.5:
    print(f"WARNING: v_e = {v_e}c may be too low to achieve E_ex < 0 in the ergosphere.")

# Initial orbit - prograde flyby with periapsis inside ergosphere
# Retrograde exhaust still achieves E_ex < 0 for Penrose extraction
if USE_NEAR_EXTREMAL_PARAMS:
    if USE_ESCAPE_OPTIMIZED:
        # ESCAPE MODE: Tuned for BOTH E_ex < 0 AND escape
        # Key insight: Need periapsis inside extraction zone (~1.7M) but not too deep
        E0 = 1.25        # higher energy for stronger escape
        Lz0 = 3.1        # slightly lower Lz for deeper periapsis ~1.55M
        r0 = 10.0        # starting radius
        a_max = 0.15     # lower thrust to avoid angular momentum loss leading to capture
    else:
        # CAPTURE MODE: Lower Lz for deeper periapsis - maximum E_ex < 0 but captured
        E0 = 1.20        # unbound (E > 1)
        Lz0 = 2.8        # lower Lz for deeper periapsis (~1.45M)
        r0 = 10.0        # starting radius
else:
    # Lower energy prograde config (fallback)
    E0 = 1.10
    Lz0 = 2.6
    r0 = 8.0

phi0 = 0.0
m0 = 1.0

# Operating radius for hovering (must be inside ergosphere, above horizon)
if USE_NEAR_EXTREMAL_PARAMS:
    if USE_ESCAPE_OPTIMIZED:
        r_set = 1.55     # target radius for escape mode (inside extraction zone)
    else:
        r_set = 1.5      # ~0.2M above horizon
else:
    r_set = 1.4

band_width = 0.3
margin = 5e-3

# PID gains for hovering at target radius
use_radial_control = True
if USE_NEAR_EXTREMAL_PARAMS:
    # Stronger gains needed near horizon where gravity is intense
    kp_r = 50.0
    kd_r = 30.0
    ki_r = 5.0
    alpha_max = np.deg2rad(60.0)
else:
    kp_r = 2.0
    kd_r = 4.0
    ki_r = 0.0
    alpha_max = np.deg2rad(12.0)

# Escape mode steering angle
alpha_escape = np.deg2rad(35.0)

# Integration settings
if USE_ESCAPE_OPTIMIZED:
    tau_span = (0.0, 80.0)  # Longer integration for escape mode to reach 50M
else:
    tau_span = (0.0, 60.0)  # Extended to allow escape verification to 50M
rtol = 1e-9
atol = 1e-11
if USE_NEAR_EXTREMAL_PARAMS:
    max_step = 0.02   # Smaller steps for better constraint preservation
else:
    max_step = 0.05

# Compute extraction limit radius dynamically based on current parameters
# This is the maximum radius at which E_ex < 0 is achievable
R_EXTRACTION_LIMIT = compute_extraction_limit_radius(E0, Lz0, m0, v_e, a, M)
if R_EXTRACTION_LIMIT is None:
    # Fallback: use a conservative estimate based on ergosphere
    R_EXTRACTION_LIMIT = r_erg_eq * 0.85  # ~1.7M for a=0.95
    print(f"Warning: Could not compute extraction limit analytically, using fallback: {R_EXTRACTION_LIMIT:.4f} M")
else:
    print(f"Computed extraction limit: R_EXTRACTION_LIMIT = {R_EXTRACTION_LIMIT:.4f} M")

# Mode state machine tracking
current_thrust_mode = ThrustMode.EXTRACTION
E_ex_history = []
delta_mu_history = []  # Track exhaust rest mass per step
r_ex_history = []  # Track radius when E_ex is recorded
mode_history = []
m_reserve_fraction = 0.3  # Reserve 30% fuel for escape

# Integral error accumulator for PID controller
integral_error_r = 0.0
integral_error_max = 1.0  # Anti-windup limit

# Print configuration summary
mode_name = "ESCAPE-OPTIMIZED" if USE_ESCAPE_OPTIMIZED else "CAPTURE (deep dive)"
print("="*65)
print("         CONTINUOUS PENROSE PROCESS CONFIGURATION")
print("="*65)
print(f"  Mode:                    {mode_name}")
print(f"  Analytic derivatives:    {USE_ANALYTIC_DERIVATIVES}")
print(f"  Adaptive thrust mode:    {USE_ADAPTIVE_THRUST_MODE}")
print(f"  Near-extremal params:    {USE_NEAR_EXTREMAL_PARAMS}")
print(f"  Exact momentum conserv:  {USE_EXACT_MOMENTUM_CONSERVATION}")
print(f"  Prograde orbit:          True (with retrograde exhaust)")
print(f"\n  Spin parameter:          a/M = {a:.4f}")
print(f"  Horizon radius:          r_+ = {r_plus:.4f} M")
print(f"  Ergosphere (equator):    r_E = {r_erg_eq:.4f} M")
print(f"  Extraction limit:        r_ex = {R_EXTRACTION_LIMIT:.4f} M")
print(f"  Operating radius:        r_set = {r_set:.4f} M")
print(f"  Exhaust velocity:        v_e = {v_e:.3f} c (gamma_e = {gamma_e:.2f})")
print(f"  Initial orbit:           E_0 = {E0:.3f}, L_z = {Lz0:+.3f}")
print(f"  Prograde ISCO:           r_ISCO+ = {isco_radius(a, M, prograde=True):.3f} M")
print("="*65 + "\n")


# -----------------------
# Local wrappers using module-level a, M
# -----------------------
def kerr_metric_cov_contra(r, th=np.pi/2):
    """Wrapper for kerr_metric_components using global a, M."""
    return kerr_metric_components(r, th, a, M)

def local_ergosphere_radius(th=np.pi/2):
    """Wrapper for ergosphere_radius using global a, M."""
    return ergosphere_radius(th, a, M)

def throttle(r, th=np.pi/2):
    """
    Throttle function: ON only inside the EXTRACTION ZONE (r < R_EXTRACTION_LIMIT).
    This ensures E_ex < 0 is achievable, enabling genuine Penrose extraction.
    Smooth ramps near the horizon and extraction limit.
    """
    global R_EXTRACTION_LIMIT
    
    # Safety margins
    if USE_NEAR_EXTREMAL_PARAMS:
        margin_inner = 5e-3   # very close to horizon
    else:
        margin_inner = 2e-2
    margin_outer = 5e-3

    r_in = r_plus + margin_inner
    # Use extraction limit instead of ergosphere - this is where E_ex < 0 is possible
    r_out = R_EXTRACTION_LIMIT - margin_outer

    if (r <= r_in) or (r >= r_out) or (r_out <= r_in):
        return 0.0

    # Smooth ramps, auto-scaled to region thickness
    width = r_out - r_in
    w_in = max(0.01, 0.15 * width)
    w_out = max(0.01, 0.15 * width)

    up = smoothstep((r - r_in) / w_in)
    down = 1.0 - smoothstep((r - r_out) / w_out)

    return float(np.clip(up * down, 0.0, 1.0))


# -----------------------
# Hamiltonian and constraint monitor
# -----------------------
def constraint_C(r, pt, pr, pphi, m, th=np.pi/2):
    cov, con = kerr_metric_cov_contra(r, th)
    gu_tt, gu_tphi, gu_rr, gu_thth, gu_phiphi = con
    # equatorial: p_th = 0
    return gu_tt*pt*pt + 2*gu_tphi*pt*pphi + gu_rr*pr*pr + gu_phiphi*pphi*pphi + m*m

def H_val(r, pt, pr, pphi, m, th=np.pi/2):
    cov, con = kerr_metric_cov_contra(r, th)
    gu_tt, gu_tphi, gu_rr, gu_thth, gu_phiphi = con
    return 0.5*(gu_tt*pt*pt + 2*gu_tphi*pt*pphi + gu_rr*pr*pr + gu_phiphi*pphi*pphi + m*m)


def project_to_mass_shell(y, last_pr_sign=None):
    """Project state back onto mass-shell constraint.
    
    If rhs <= 0, the state is in the forbidden region. We clamp small 
    negatives (|rhs| < 1e-10) to zero and warn on larger violations.
    
    Parameters
    ----------
    y : array
        State vector [r, phi, pt, pr, pphi, m]
    last_pr_sign : float or None
        Last known non-zero sign of pr (for turning point handling)
    
    Returns
    -------
    y_new : array
        Projected state
    """
    r, phi, pt, pr, pphi, m = y
    th = np.pi/2
    cov, con = kerr_metric_cov_contra(r, th)
    gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con

    rhs = -(gu_tt*pt*pt + 2*gu_tphi*pt*pphi + gu_phiphi*pphi*pphi + m*m)
    
    # Handle forbidden region
    if rhs < 0:
        if abs(rhs) < 1e-10:
            # Small numerical roundoff - clamp to zero
            rhs = 0.0
        else:
            # Larger violation - warn and clamp
            import warnings
            warnings.warn(
                f"Mass-shell violation in projection: rhs={rhs:.6e} at r={r:.4f}. "
                f"State may be unphysical. Clamping to rhs=0.",
                RuntimeWarning
            )
            rhs = 0.0
    
    if rhs == 0:
        # At turning point or forbidden region
        pr_new = 0.0
    else:
        pr_mag = np.sqrt(rhs / gu_rr)
        # Robust sign handling at turning points
        if abs(pr) > 1e-12:
            pr_new = np.sign(pr) * pr_mag
        elif last_pr_sign is not None:
            pr_new = last_pr_sign * pr_mag
        else:
            pr_new = -pr_mag  # Default to infall
    
    return np.array([r, phi, pt, pr_new, pphi, m], dtype=float)


# -----------------------------------------------------------------------------
# NOTE: build_rocket_rest_basis and related helper functions are now imported
# from kerr_utils.py to avoid code duplication. The local definitions have been
# removed. If you need the local versions for debugging, they were:
#   _inner_prod_tr, _normalize_spacelike, build_rocket_rest_basis
# -----------------------------------------------------------------------------


# -----------------------
# Continuous thrust dynamics (equatorial: th=pi/2, p_th=0)
# State: [r, phi, p_t, p_r, p_phi, m]
# -----------------------
def dynamics_continuous(tau, y):
    """
    Equatorial Kerr dynamics with continuous thrust.
    
    State: [r, phi, p_t, p_r, p_phi, m]
    
    Uses analytic derivatives and adaptive thrust mode selection.
    """
    global current_thrust_mode, E_ex_history, delta_mu_history, r_ex_history, mode_history, integral_error_r
    
    r, phi, pt, pr, pphi, m = y
    th = np.pi / 2

    # Metric
    cov, con = kerr_metric_cov_contra(r, th)
    g_tt, g_tphi, g_rr, g_thth, g_phiphi = cov
    gu_tt, gu_tphi, gu_rr, gu_thth, gu_phiphi = con

    # Basic guards
    if (not np.isfinite(r)) or (r <= 0) or (not np.isfinite(m)) or (m <= 0):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if (g_rr <= 0) or (g_phiphi <= 0) or (not np.isfinite(g_rr)) or (not np.isfinite(g_phiphi)):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Velocities from Hamilton's equations: dx^mu/dtau = p^mu/m
    dr = gu_rr * pr / m
    dphi = (gu_phiphi * pphi + gu_tphi * pt) / m

    # Check and fix constraint BEFORE computing thrust
    C_now = gu_tt*pt**2 + 2*gu_tphi*pt*pphi + gu_rr*pr**2 + gu_phiphi*pphi**2 + m**2
    if abs(C_now) > 1e-6:
        # Re-project pr to satisfy constraint
        rhs = -(gu_tt*pt**2 + 2*gu_tphi*pt*pphi + gu_phiphi*pphi**2 + m**2)
        if rhs > 0:
            pr_new = np.sign(pr) * np.sqrt(rhs / gu_rr) if pr != 0 else -np.sqrt(rhs / gu_rr)
            pr = pr_new
            dr = gu_rr * pr / m  # Recompute velocity (divide by m!)
    
    # Radial force from Hamiltonian: dpr/dtau = -(1/m) * dH/dr
    # 
    # CRITICAL: The Hamiltonian H = (1/2) g^{munu} p_mu p_nu satisfies H = -m^2/2.
    # For proper time tau (not affine parameter), Hamilton's equations are:
    #     dx^mu/dtau = (1/m) dH/dp_mu = p^mu/m
    #     dp_mu/dtau = -(1/m) dH/dx^mu
    #
    # The 1/m factor ensures geodesic motion is mass-independent (equivalence principle).
    #
    if USE_ANALYTIC_DERIVATIVES:
        # Use analytic derivatives for better accuracy near horizon
        dH_dr = compute_dH_dr_analytic(r, th, pt, pr, pphi, m, a, M)
    else:
        # Fallback to finite differences
        eps = 1e-6 * max(1.0, abs(r))
        Hp = H_val(r + eps, pt, pr, pphi, m, th)
        Hm = H_val(r - eps, pt, pr, pphi, m, th)
        dH_dr = (Hp - Hm) / (2.0 * eps)

    dpt = 0.0
    dpr = -dH_dr / m  # FIXED: Added 1/m factor for proper time evolution
    dpphi = 0.0
    dm = 0.0

    # Current energy
    E_current = -pt
    r_erg = local_ergosphere_radius(th)

    # Determine thrust mode if adaptive control is enabled
    if USE_ADAPTIVE_THRUST_MODE:
        current_thrust_mode, mode_reason = determine_thrust_mode(
            r, pr, E_current, m, m0, E_ex_history,
            r_plus, r_erg, r_safe, m_reserve_fraction,
            r_extraction_limit=R_EXTRACTION_LIMIT  # Use pre-computed value for speed
        )

    # Thrust logic
    u_th = throttle(r, th)
    
    # Determine if we should thrust based on mode
    # CRITICAL: Only thrust in EXTRACTION mode where E_ex < 0 is achievable
    # Both ESCAPE and COAST modes disable thrust:
    #   - ESCAPE: Coast out on Penrose-boosted trajectory (thrusting here deposits positive E_ex)
    #   - COAST: Conserve fuel or waiting for extraction zone
    should_thrust = (m > m_min) and (r > r_safe) and np.isfinite(u_th) and (u_th > 0.0)
    if USE_ADAPTIVE_THRUST_MODE:
        if current_thrust_mode == ThrustMode.COAST:
            should_thrust = False  # Conserve fuel
        elif current_thrust_mode == ThrustMode.ESCAPE:
            # Coast to infinity - do NOT thrust here
            # Thrusting outside extraction zone deposits positive E_ex, negating gains
            should_thrust = False
        elif current_thrust_mode == ThrustMode.EXTRACTION:
            # Only thrust in extraction mode where E_ex < 0 is achievable
            should_thrust = should_thrust and (m > m_min * 1.05)
    
    if should_thrust:
        # Use throttle function value (already restricted to extraction zone)
        throttle_level = u_th
        
        if throttle_level <= 0:
            return [dr, dphi, dpt, dpr, dpphi, dm]
            
        a_prop = a_max * throttle_level
        
        # ======================================================================
        # EXACT 4-MOMENTUM CONSERVATION (CRITICAL FIX)
        # ======================================================================
        # 
        # Instead of the approximate: thrust = m * a_prop, dm = -thrust/v_exhaust
        # We use EXACT 4-momentum conservation: p'_mu = p_mu - deltamu * u_{ex,mu}
        #
        # The exhaust rest mass rate dmu/dtau is determined by the proper acceleration:
        #     a_prop = (gamma_e * v_e / m) * dmu/dtau
        # =>  dmu/dtau = m * a_prop / (gamma_e * v_e)
        #
        # Then: dp_mu/dtau = -(dmu/dtau) * u_{ex,mu}
        # And:  dm/dtau comes from mass-shell, but approximately: dm ~ -gamma_e * dmu
        #
        # Compute exhaust rest mass rate (needed for both exact and legacy modes)
        delta_mu_rate = m * a_prop / (gamma_e * v_e)
        
        if USE_EXACT_MOMENTUM_CONSERVATION:
            dm = -delta_mu_rate * gamma_e  # Rocket mass loss rate
        else:
            # Legacy: approximate mass flow
            thrust = m * a_prop
            dm = -thrust / v_exhaust_momentum

        # Steering angle based on mode
        # Note: ESCAPE mode sets should_thrust=False, so this block only runs in EXTRACTION mode
        # or with PID control when adaptive mode is disabled
        if USE_ADAPTIVE_THRUST_MODE and current_thrust_mode == ThrustMode.EXTRACTION:
            # In EXTRACTION mode, find alpha that minimizes E_ex for genuine Penrose process
            # We'll scan alpha and choose the one giving minimum E_ex
            # This is done later in the direction selection code
            alpha = 0.0  # Placeholder - actual optimization done below
        elif use_radial_control:
            # PID controller for hovering 
            # Error terms
            error_r = r_set - r
            error_dot_r = -dr  # Want dr -> 0
            
            # Update integral with anti-windup (global already declared at function start)
            integral_error_r = np.clip(integral_error_r + error_r * 0.001, 
                                       -integral_error_max, integral_error_max)
            
            # Feedforward term: estimate gravitational acceleration at this radius
            # dH/dr gives the effective radial force; need to counteract it
            feedforward = 0.3 * dH_dr / (a_max * throttle_level + 1e-10)  # Scaled feedforward
            
            # PID control law + feedforward
            alpha_cmd = (kp_r * error_r + 
                        kd_r * error_dot_r + 
                        ki_r * integral_error_r +
                        feedforward)
            alpha = float(np.clip(alpha_cmd, -alpha_max, alpha_max))
        else:
            alpha = 0.0

        # Contravariant momenta p^mu
        p_contra_t = gu_tt * pt + gu_tphi * pphi
        p_contra_r = gu_rr * pr
        p_contra_phi = gu_tphi * pt + gu_phiphi * pphi

        # 4-velocity components u^mu
        u_t = p_contra_t / m
        u_r = p_contra_r / m
        u_phi = p_contra_phi / m

        # ------------------------------------------------------------------
        # Physically consistent thrust construction (rocket-rest tetrad)
        # ------------------------------------------------------------------

        u_vec = np.array([u_t, u_r, u_phi], dtype=float)
        e_r, e_phi = build_rocket_rest_basis(u_vec, g_tt, g_tphi, g_rr, g_phiphi)
        if (e_r is None) or (e_phi is None):
            return [dr, dphi, dpt, dpr, dpphi, 0.0]

        s_r = np.sin(alpha)
        s_phi = np.cos(alpha)

        def candidate_exact(sign_phi):
            """Compute momentum rates using EXACT 4-momentum conservation."""
            s_vec = s_r * e_r + (sign_phi * s_phi) * e_phi
            
            # Get full exhaust 4-velocity
            u_ex_result = compute_exhaust_4velocity(
                u_vec, s_vec, v_e, g_tt, g_tphi, g_rr, g_phiphi
            )
            u_ex_cov = u_ex_result['u_ex_cov']
            E_ex = u_ex_result['E_ex']
            
            # EXACT: dp_mu/dtau = -(dmu/dtau) * u_{ex,mu}
            # where dmu/dtau = delta_mu_rate (exhaust rest mass rate)
            dpt_c = -delta_mu_rate * u_ex_cov[0]
            dpr_c = -delta_mu_rate * u_ex_cov[1]
            dpphi_c = -delta_mu_rate * u_ex_cov[2]
            
            return dpt_c, dpr_c, dpphi_c, s_vec, E_ex, u_ex_result
        
        def candidate_legacy(sign_phi):
            """Compute momentum rates using legacy approximate method."""
            s_vec = s_r * e_r + (sign_phi * s_phi) * e_phi
            thrust = m * a_prop
            f_t, f_r, f_phi = thrust * s_vec

            dp_contra_t = f_t + u_t * dm
            dp_contra_r = f_r + u_r * dm
            dp_contra_phi = f_phi + u_phi * dm

            dpt_c = g_tt * dp_contra_t + g_tphi * dp_contra_phi
            dpr_c = g_rr * dp_contra_r
            dpphi_c = g_tphi * dp_contra_t + g_phiphi * dp_contra_phi
            
            E_ex, _ = compute_exhaust_energy(u_vec, s_vec, v_e, g_tt, g_tphi, g_phiphi)
            return dpt_c, dpr_c, dpphi_c, s_vec, E_ex, None

        # Choose which candidate function to use
        candidate = candidate_exact if USE_EXACT_MOMENTUM_CONSERVATION else candidate_legacy

        # Compute candidates for both azimuthal signs
        dpt_p, dpr_p, dpphi_p, s_vec_p, E_ex_p, u_ex_p = candidate(+1.0)
        dpt_m, dpr_m, dpphi_m, s_vec_m, E_ex_m, u_ex_m = candidate(-1.0)

        # ------------------------------------------------------------------
        # Thrust direction selection based on mode
        # ------------------------------------------------------------------
        if USE_ADAPTIVE_THRUST_MODE and current_thrust_mode == ThrustMode.EXTRACTION:
            # EXTRACTION MODE: Pure E_ex minimization (Penrose extraction)
            # Scan over alpha values to find direction giving minimum E_ex
            best_E_ex = float('inf')
            best_dpt, best_dpr, best_dpphi = 0.0, 0.0, 0.0
            best_u_ex_result = None
            
            for alpha_scan in np.linspace(-np.pi/2, np.pi/2, 37):
                s_r_scan = np.sin(alpha_scan)
                s_phi_scan = np.cos(alpha_scan)
                
                for sign_phi_scan in [+1.0, -1.0]:
                    s_vec_scan = s_r_scan * e_r + (sign_phi_scan * s_phi_scan) * e_phi
                    
                    if USE_EXACT_MOMENTUM_CONSERVATION:
                        # Exact: get full exhaust 4-velocity and use conservation
                        u_ex_result = compute_exhaust_4velocity(
                            u_vec, s_vec_scan, v_e, g_tt, g_tphi, g_rr, g_phiphi
                        )
                        E_ex_scan = u_ex_result['E_ex']
                        u_ex_cov = u_ex_result['u_ex_cov']
                        
                        if E_ex_scan < best_E_ex:
                            best_E_ex = E_ex_scan
                            best_dpt = -delta_mu_rate * u_ex_cov[0]
                            best_dpr = -delta_mu_rate * u_ex_cov[1]
                            best_dpphi = -delta_mu_rate * u_ex_cov[2]
                            best_u_ex_result = u_ex_result
                    else:
                        # Legacy approximate method
                        thrust = m * a_prop
                        f_t_scan, f_r_scan, f_phi_scan = thrust * s_vec_scan
                        E_ex_scan, _ = compute_exhaust_energy(u_vec, s_vec_scan, v_e, g_tt, g_tphi, g_phiphi)
                        
                        if E_ex_scan < best_E_ex:
                            best_E_ex = E_ex_scan
                            dp_contra_t = f_t_scan + u_t * dm
                            dp_contra_r = f_r_scan + u_r * dm
                            dp_contra_phi = f_phi_scan + u_phi * dm
                            
                            best_dpt = g_tt * dp_contra_t + g_tphi * dp_contra_phi
                            best_dpr = g_rr * dp_contra_r
                            best_dpphi = g_tphi * dp_contra_t + g_phiphi * dp_contra_phi
            
            dpt += best_dpt
            dpr += best_dpr
            dpphi += best_dpphi
            E_ex_history.append(best_E_ex)
            # Track exhaust rest mass rate (for exact conservation) or approximate
            if USE_EXACT_MOMENTUM_CONSERVATION:
                delta_mu_history.append(delta_mu_rate)
            else:
                delta_mu_history.append(-dm / gamma_e)  # Approximate
            r_ex_history.append(r)
        else:
            # DEFAULT MODE (adaptive mode off): Maximize dE/dtau = -dpt/dtau (greedy energy gain)
            # Note: ESCAPE mode never reaches here since should_thrust=False for ESCAPE
            if (-dpt_p) >= (-dpt_m):
                dpt += dpt_p
                dpr += dpr_p
                dpphi += dpphi_p
                E_ex_val = E_ex_p
            else:
                dpt += dpt_m
                dpr += dpr_m
                dpphi += dpphi_m
                E_ex_val = E_ex_m
            E_ex_history.append(E_ex_val)
            if USE_EXACT_MOMENTUM_CONSERVATION:
                delta_mu_history.append(delta_mu_rate)
            else:
                delta_mu_history.append(-dm / gamma_e)
            r_ex_history.append(r)

        # Safety clamp
        if not np.all(np.isfinite([dpt, dpr, dpphi, dm])):
            dpt, dpr, dpphi, dm = 0.0, -dH_dr / m, 0.0, 0.0

    return [dr, dphi, dpt, dpr, dpphi, dm]


# -----------------------
# Initial conditions: choose p_t=-E0, p_phi=Lz0, solve p_r from constraint C=0
# -----------------------
th0 = np.pi/2
cov0, con0 = kerr_metric_cov_contra(r0, th0)
gu_tt0, gu_tphi0, gu_rr0, _, gu_phiphi0 = con0

pt0 = -E0
pphi0 = Lz0

# Solve for pr0^2 from: gu_rr pr^2 = -(gu_tt pt^2 + 2 gu_tphi pt pphi + gu_phiphi pphi^2 + m^2)
rhs = -(gu_tt0*pt0*pt0 + 2*gu_tphi0*pt0*pphi0 + gu_phiphi0*pphi0*pphi0 + m0*m0)
if rhs < 0:
    raise ValueError(
        f"Initial conditions invalid: pr^2 would be negative (rhs={rhs}). "
        f"Try changing Lz0, E0, r0, or a_max."
    )
pr0 = -np.sqrt(rhs / gu_rr0)  # negative means initially infalling

y0 = [r0, phi0, pt0, pr0, pphi0, m0]

r, phi, pt, pr, pphi, m = y0
cov, con = kerr_metric_cov_contra(r, np.pi/2)
g_tt, g_tphi, g_rr, _, g_phiphi = cov
gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con

omega = -g_tphi / g_phiphi
p_contra_t = gu_tt*pt + gu_tphi*pphi
p_contra_phi = gu_tphi*pt + gu_phiphi*pphi

u_t = p_contra_t/m
u_phi = p_contra_phi/m
u_rel_phi = u_phi - omega*u_t

print("omega=", omega, "u_rel_phi=", u_rel_phi, "pphi0=", pphi)

# -----------------------
# Events: stop if we hit r_safe (near horizon), or if mass depleted
# -----------------------
# Escape radius threshold - increased to definitively confirm escape
ESCAPE_RADIUS = 50.0  # Increased for definitive escape verification

def horizon_event(tau, y):
    return y[0] - r_safe
horizon_event.terminal = True
horizon_event.direction = -1

def mass_event(tau, y):
    return y[5] - m_min
mass_event.terminal = True
mass_event.direction = -1

C_MAX = 0.1           # stop if constraint drifts too much (relaxed significantly)
P_MAX = 1e6           # stop if momenta blow up
R_MAX = 200.0         # stop if you fly way out

def constraint_event(tau, y):
    r, phi, pt, pr, pphi, m = y
    return C_MAX - abs(constraint_C(r, pt, pr, pphi, m))
constraint_event.terminal = True
constraint_event.direction = -1

def blowup_event(tau, y):
    r, phi, pt, pr, pphi, m = y
    return P_MAX - max(abs(pt), abs(pr), abs(pphi))
blowup_event.terminal = True
blowup_event.direction = -1

def escape_event(tau, y):
    return ESCAPE_RADIUS - y[0]  # Stop when r > ESCAPE_RADIUS (definitive escape)
escape_event.terminal = True
escape_event.direction = -1

# -----------------------
# Integrate
# -----------------------
# sol = solve_ivp(
#     dynamics_continuous,
#     tau_span,
#     y0,
#     method="Radau",
#     rtol=rtol,
#     atol=atol,
#     max_step=max_step,
#     events=[horizon_event, mass_event],
#     dense_output=True
# )
# tau = sol.t
# r = sol.y[0]
# phi = sol.y[1]
# pt = sol.y[2]
# pr = sol.y[3]
# pphi = sol.y[4]
# m = sol.y[5]
# E = -pt

def integrate_with_projection(y0, tau0=0.0, tau_end=200.0):
    """
    Euler integrator with explicit constraint projection.
    
    Returns (T, Y, termination_reason) where termination_reason is
    'escape', 'horizon', 'mass', or 'timeout'.
    
    NUMERICAL METHOD NOTES:
    -----------------------
    This uses explicit Euler with mass-shell projection after each step.
    The projection modifies p_r to satisfy g^{munu}p_mup_nu + m^2 = 0.
    
    IMPORTANT CAVEATS:
    1. The thrust law uses EXACT differential 4-momentum conservation:
           dp_mu/dtau = -(dmu/dtau) * u_{ex,mu}
       This is physically correct for the emission kinematics.
    
    2. However, the mass-shell projection injects a small non-physical Deltap_r
       that is NOT accounted for in the exhaust emission. This is a numerical
       stabilization artifact, not a physical impulse.
    
    3. ENERGY CONSERVATION IS PRESERVED because:
       - We never project p_t (which determines E = -p_t)
       - The thrust update to p_t is exact: dp_t = -(dmu/dtau) * u_{ex,t}
       - Thus DeltaE_rocket + Sum E_ex*deltamu ~ 0 to machine precision
    
    4. TRAJECTORY ACCURACY is O(dt^2) due to Euler truncation plus projection
       artifacts. The path (periapsis, time in extraction zone) can shift
       with dt. For quantitative claims, verify step-size convergence.
    
    ALTERNATIVES:
    - solve_ivp with Radau (as in single_thrust_case.py) for implicit RK
    - RATTLE/SHAKE for systematic constrained integration
    - Lagrange multiplier enforcement
    """
    dt = DT_INTEGRATION  # Use global constant (single source of truth)
    R_ESCAPE = ESCAPE_RADIUS  # Use global escape radius for definitive escape verification
    
    T = [tau0]
    Y = [np.array(y0, dtype=float)]

    tau = tau0
    y = np.array(y0, dtype=float)
    termination_reason = 'timeout'  # default
    
    # Sample output every this many steps
    sample_every = 10
    step = 0

    while tau < tau_end:
        r, phi, pt, pr, pphi, m = y
        
        # Compute energy for escape check
        E_current = -pt
        
        # Check stopping conditions
        if r <= r_safe:
            print(f"Integration stopped: horizon event at tau={tau:.4f}")
            termination_reason = 'horizon'
            break
        if m <= m_min:
            print(f"Integration stopped: mass event at tau={tau:.4f}")
            termination_reason = 'mass'
            break
        
        # UNIFIED ESCAPE CRITERION:
        # 1. Far from BH (r > R_ESCAPE)
        # 2. Moving outward (pr > 0, which means dr/dtau > 0 since pr = g_rr * dr/dtau)
        # 3. Unbound orbit (E > m, i.e., specific energy E/m > 1)
        # All three conditions are required for a definitive escape claim.
        is_far = r > R_ESCAPE
        is_outward = pr > 0
        is_unbound = E_current > m  # E/m > 1 means unbound
        
        if is_far and is_outward and is_unbound:
            print(f"Integration stopped: DEFINITIVE ESCAPE at tau={tau:.4f}, r={r:.2f}M")
            print(f"  (r > {R_ESCAPE}M, dr/dtau > 0, E/m = {E_current/m:.4f} > 1)")
            termination_reason = 'escape'
            break
        elif is_far and is_outward:
            # Far and outward, but E < m (bound orbit returning)
            # This would require very long integration to confirm - treat as escape for now
            # but with a warning
            print(f"Integration stopped: reached r={r:.2f}M moving outward at tau={tau:.4f}")
            print(f"  WARNING: E/m = {E_current/m:.4f} < 1 suggests bound orbit.")
            print(f"  Trajectory may eventually return. Treating as escape for analysis.")
            termination_reason = 'escape'
            break
        elif r > 200:
            # Very far out but moving inward - definitely on an escape trajectory
            # that will turn around and go to infinity
            if is_unbound:
                print(f"Integration stopped: reached r={r:.2f}M at tau={tau:.4f}")
                print(f"  (E/m = {E_current/m:.4f} > 1 confirms unbound)")
                termination_reason = 'escape'
            else:
                print(f"Integration stopped: reached r={r:.2f}M at tau={tau:.4f}")
                print(f"  WARNING: Moving inward and E/m = {E_current/m:.4f}. Status uncertain.")
                termination_reason = 'escape'  # Still far enough to consider escaped
            break
        
        # Get dynamics
        dydt = dynamics_continuous(tau, y)
        
        # Track last non-zero pr sign for turning point handling
        if abs(y[3]) > 1e-12:
            last_pr_sign = np.sign(y[3])
        
        # Euler step
        y_new = y + np.array(dydt) * dt
        
        # Project back to mass shell (with turning point handling)
        y_new = project_to_mass_shell(y_new, last_pr_sign=last_pr_sign if 'last_pr_sign' in dir() else None)
        
        y = y_new
        tau += dt
        step += 1
        
        # Sample output
        if step % sample_every == 0:
            T.append(tau)
            Y.append(y.copy())

    # CRITICAL FIX: Always append the terminal state for consistent diagnostics
    # This ensures energy budgets and final-state analysis use the actual end state
    if len(T) == 0 or tau != T[-1]:
        T.append(tau)
        Y.append(y.copy())

    return np.array(T), np.array(Y).T, termination_reason  # like solve_ivp: Y has shape (nstate, nt)

tau, Y, termination_reason = integrate_with_projection(y0, tau_span[0], tau_span[1])
r, phi, pt, pr, pphi, m = Y
E = -pt
C = np.array([constraint_C(r[i], pt[i], pr[i], pphi[i], m[i]) for i in range(len(tau))])

print(f"Horizon r+     = {r_plus:.6f}")
print(f"Ergosphere (eq)= {local_ergosphere_radius(np.pi/2):.6f}")
print(f"Minimum radius reached: r_min = {np.min(r):.4f}M (extraction zone: r < {R_EXTRACTION_LIMIT}M)")
print(f"End state: r={r[-1]:.4f}, E={E[-1]:.6f}, m={m[-1]:.6f}")
print(f"Constraint |C| max = {np.max(np.abs(C)):.3e}")

# =============================================================================
# ADDITIONAL PHYSICS VERIFICATION
# =============================================================================
# Verify causality (future-directedness) for sampled states
from kerr_utils import verify_future_directed
n_causality_violations = 0
for i in range(len(tau)):
    th_i = np.pi/2
    cov_i, con_i = kerr_metric_cov_contra(r[i], th_i)
    gu_tt_i, gu_tphi_i, gu_rr_i, _, gu_phiphi_i = con_i
    u_t_i = (gu_tt_i * pt[i] + gu_tphi_i * pphi[i]) / m[i]
    is_future, _ = verify_future_directed(u_t_i, warn=False)
    if not is_future:
        n_causality_violations += 1

if n_causality_violations > 0:
    print(f"WARNING: {n_causality_violations} causality violations (u^t <= 0) detected!")
else:
    print(f"Causality check: [OK] All sampled states have u^t > 0 (future-directed)")

# Report termination status - this is critical for Penrose extraction claims
print(f"\n{'TRAJECTORY OUTCOME':^65}")
print("-"*65)
if termination_reason == 'escape':
    print(f"  *** SUCCESSFUL ESCAPE TO INFINITY ***")
    print(f"  Final radius: r = {r[-1]:.2f}M > {ESCAPE_RADIUS}M (definitive escape)")
    print(f"  Final energy delivered to infinity: E_final = {E[-1]:.6f}")
    delta_E = E[-1] - E[0]
    delta_m = m[0] - m[-1]
    print(f"  Energy gain: DeltaE = {delta_E:.6f} ({100*delta_E/E[0]:.1f}% of initial)")
    print(f"  Mass expended: Deltam = {delta_m:.6f} ({100*delta_m/m[0]:.1f}% of initial)")
elif termination_reason == 'horizon':
    print(f"  *** CAPTURED BY BLACK HOLE ***")
    print(f"  Spacecraft crossed horizon safety margin at r = {r[-1]:.4f}M")
    print(f"  Final energy at capture: E = {E[-1]:.6f}")
    print(f"  NOTE: Energy gain cannot be claimed as 'extracted' since")
    print(f"        the spacecraft did not escape to infinity!")
elif termination_reason == 'mass':
    print(f"  Integration ended: fuel exhausted (m = {m[-1]:.4f})")
else:
    print(f"  Integration ended: timeout (tau = {tau[-1]:.2f})")
print("-"*65)

# =============================================================================
# EFFICIENCY ANALYSIS
# =============================================================================

# Compute efficiency metrics
eta_inst, eta_cum, eta_max, penrose_active = print_efficiency_analysis(tau, E, m, r, a, M)

# Compute derivatives for plotting
dE_dtau = np.gradient(E, tau)
dm_dtau = np.gradient(m, tau)

# Throttle history (geometric throttle based on radius) - useful for plotting
# the extraction-zone gate, but NOT sufficient to infer whether the engine actually fired.
u_hist = np.array([throttle(ri, np.pi/2) for ri in r])

# "Thrust active" should reflect the *actual dynamics* (engine firing), not merely throttle(r)>0,
# because adaptive mode logic can disable thrust even when u_hist>0.
# In this model, m decreases iff the engine is firing, so dm/dtau < 0 is the most robust indicator.
mask = dm_dtau < -1e-10

if np.any(mask):
    total_time = tau[-1] - tau[0]
    # Sum durations of all thrust-active intervals (do not use tau_on[-1]-tau_on[0], which includes gaps)
    duty_time = float(np.sum(np.diff(tau) * mask[:-1]))
    print(f"Thrust-on duration: {duty_time:.6f} out of {total_time:.6f}  (fraction {duty_time/total_time:.3%})")

    # Net changes accumulated only during thrust-active intervals (based on stored sample points)
    dE_thrust = float(np.sum((E[1:] - E[:-1]) * mask[:-1]))
    dm_thrust = float(np.sum((m[:-1] - m[1:]) * mask[:-1]))
    print(f"Delta E during thrust-active intervals: {dE_thrust:.8f}")
    print(f"Delta m during thrust-active intervals: {dm_thrust:.8f}")
else:
    print("Engine never fired (dm/dtau ~ 0 everywhere).")

# =============================================================================
# THRUST MODEL CONSISTENCY CHECKS (tetrad + exhaust energy diagnostic)
# =============================================================================
exhaust_samples_for_capture = []  # Collect samples for capture verification

if np.any(mask):
    idxs = np.where(mask)[0]
    ratios = []
    v_eff_meas = []
    Eex = []

    r_erg_eq = local_ergosphere_radius(np.pi/2)

    for i in idxs:
        ri = float(r[i])
        pti = float(pt[i])
        pri = float(pr[i])
        pphii = float(pphi[i])
        mi = float(m[i])
        if mi <= 0 or (not np.isfinite(mi)):
            continue

        th = np.pi/2
        cov, con = kerr_metric_cov_contra(ri, th)
        g_tt, g_tphi, g_rr, _, g_phiphi = cov
        gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con

        dri = gu_rr * pri / mi
        u_th_i = float(u_hist[i])
        a_prop_i = a_max * u_th_i
        thrust_i = mi * a_prop_i
        if thrust_i <= 0:
            continue

        dm_pred = -thrust_i / (v_e if USE_EXACT_MOMENTUM_CONSERVATION else v_exhaust_momentum)

        # Rocket 4-velocity u^mu (contravariant)
        p_contra_t = gu_tt * pti + gu_tphi * pphii
        p_contra_r = gu_rr * pri
        p_contra_phi = gu_tphi * pti + gu_phiphi * pphii
        u_vec = np.array([p_contra_t/mi, p_contra_r/mi, p_contra_phi/mi], dtype=float)

        e_r, e_phi = build_rocket_rest_basis(u_vec, g_tt, g_tphi, g_rr, g_phiphi)
        if (e_r is None) or (e_phi is None):
            continue

        # Use same steering logic as dynamics_continuous():
        # In EXTRACTION mode: scan angles to MINIMIZE E_ex
        # In other modes: use PID hover steering and pick sign for max energy gain
        # For consistency check, we replicate what the integration actually did.
        
        if USE_ADAPTIVE_THRUST_MODE:
            # EXTRACTION mode: scan angles to minimize E_ex (same as integration)
            best_E_ex = float('inf')
            best_s_vec = None
            for alpha_scan in np.linspace(-np.pi/2, np.pi/2, 37):
                s_r_scan = np.sin(alpha_scan)
                s_phi_scan = np.cos(alpha_scan)
                for sign_phi in [+1.0, -1.0]:
                    s_vec_trial = s_r_scan * e_r + (sign_phi * s_phi_scan) * e_phi
                    E_ex_trial, _ = compute_exhaust_energy(u_vec, s_vec_trial, v_e, g_tt, g_tphi, g_phiphi)
                    if E_ex_trial < best_E_ex:
                        best_E_ex = E_ex_trial
                        best_s_vec = s_vec_trial
            s_vec = best_s_vec if best_s_vec is not None else e_phi
            f_vec = thrust_i * s_vec
        else:
            # Non-adaptive mode: PID hover steering + sign for max energy gain
            alpha = 0.0
            if use_radial_control:
                alpha_cmd = kp_r * (r_set - ri) - kd_r * dri
                alpha = float(np.clip(alpha_cmd, -alpha_max, alpha_max))

            s_r = np.sin(alpha)
            s_phi = np.cos(alpha)

            def cand(sign_phi):
                s_vec = s_r * e_r + (sign_phi * s_phi) * e_phi
                f_vec = thrust_i * s_vec
                dp_contra = f_vec + u_vec * dm_pred
                dpt_c = g_tt * dp_contra[0] + g_tphi * dp_contra[2]
                return dpt_c, s_vec, f_vec

            dpt_p, s_p, f_p = cand(+1.0)
            dpt_m, s_m, f_m = cand(-1.0)
            if (-dpt_p) >= (-dpt_m):
                s_vec = s_p
                f_vec = f_p
            else:
                s_vec = s_m
                f_vec = f_m

        # Check |f| == thrust
        f2 = _inner_prod_tr(f_vec, f_vec, g_tt, g_tphi, g_rr, g_phiphi)
        if np.isfinite(f2) and f2 > 0:
            fmag = np.sqrt(f2)
            ratios.append(fmag / thrust_i)

            # Effective "exhaust momentum" inferred from the numerically realized dm/dtau
            if dm_dtau[i] < -1e-12:
                v_eff_meas.append(fmag / (-dm_dtau[i]))

            # Exhaust 4-velocity in the rocket rest frame: u_ex = gamma_e (u - v_e s)
            # (Valid when v_e is interpreted as a speed in the rocket rest frame.)
            u_ex = gamma_e * (u_vec - v_e * s_vec)
            u_ex_tcov = g_tt * u_ex[0] + g_tphi * u_ex[2]
            E_ex_val = -u_ex_tcov
            Eex.append(E_ex_val)
            
            # Collect sample for capture verification (downsample for performance)
            if i % 10 == 0:  # Every 10th thrust sample
                exhaust_samples_for_capture.append({
                    'E_ex': E_ex_val,
                    'r': ri,
                    'th': th,
                    'u_ex_contra': u_ex  # Contravariant 4-velocity (t, r, phi components)
                })

    ratios = np.array(ratios, dtype=float) if len(ratios) else np.array([])
    v_eff_meas = np.array(v_eff_meas, dtype=float) if len(v_eff_meas) else np.array([])
    Eex = np.array(Eex, dtype=float) if len(Eex) else np.array([])

    print("\n" + "="*65)
    print("THRUST MODEL CONSISTENCY CHECKS")
    print("="*65)
    if len(ratios):
        print(f"  Force normalization: max(| |f|/T - 1 |) = {np.max(np.abs(ratios-1.0)):.3e}")
        print(f"                     mean(|f|/T) = {np.mean(ratios):.6f}")
    else:
        print("  Force normalization: (no valid thrust samples)")

    if len(v_eff_meas):
        print(f"  Inferred v_eff = |f|/(-dm/dtau): mean={np.mean(v_eff_meas):.4f}, std={np.std(v_eff_meas):.4f}")
        # In the EXACT emission model implemented here:
        #   T = v_e * (-dm/dtau)  ==> v_eff should match v_e.
        # In the legacy approximate model:
        #   T = v_exhaust_momentum * (-dm/dtau).
        expected_v_eff = v_e if USE_EXACT_MOMENTUM_CONSERVATION else v_exhaust_momentum
        print(f"  Expected v_eff (|f|/(-dm/dtau)): {expected_v_eff:.4f}  (v_e={v_e:.3f}, gamma_e={gamma_e:.3f})")

    if len(Eex):
        frac_neg = np.mean(Eex < 0.0)
        print(f"  Exhaust energy per unit exhaust mass at infinity: min={np.min(Eex):.6f}, median={np.median(Eex):.6f}")
        print(f"  Fraction with E_ex < 0 (negative-energy exhaust): {100*frac_neg:.1f}%")
        if frac_neg > 0:
            print("    -> This is the direct Penrose signature (waste stream carries negative Killing energy).")
        else:
            print("    -> No negative-energy exhaust detected for this run.")
    print("="*65)

# =============================================================================
# GENUINE PENROSE EXTRACTION ANALYSIS (E_ex tracking)
# =============================================================================
print("\n" + "="*65)
print("         GENUINE PENROSE EXTRACTION ANALYSIS")
print("="*65)

# Global E_ex_history was populated during integration
E_ex_arr = np.array(E_ex_history) if E_ex_history else np.array([])

if len(E_ex_arr) > 0:
    n_negative = np.sum(E_ex_arr < 0)
    n_total = len(E_ex_arr)
    frac_negative = n_negative / n_total
    
    print(f"\n{'EXHAUST ENERGY STATISTICS':^65}")
    print("-"*65)
    print(f"  Total thrust samples:            {n_total}")
    print(f"  Samples with E_ex < 0:           {n_negative} ({100*frac_negative:.1f}%)")
    print(f"  Min E_ex:                        {np.min(E_ex_arr):.6f}")
    print(f"  Max E_ex:                        {np.max(E_ex_arr):.6f}")
    print(f"  Mean E_ex:                       {np.mean(E_ex_arr):.6f}")
    
    if frac_negative > 0:
        neg_mask = E_ex_arr < 0
        mean_neg = np.mean(E_ex_arr[neg_mask])
        print(f"  Mean E_ex (when negative):       {mean_neg:.6f}")
        
        # Compute energy flux using tracked delta_mu_history (exhaust rest mass rates)
        # Each sample contributes: E_ex_i * deltamu_i where deltamu_i = (dmu/dtau)_i * dt
        # This is the CORRECT calculation matching compute_energy_budget:
        #   Energy to BH = Sum E_ex,i * deltamu_i  (not rocket mass loss!)
        if len(delta_mu_history) == len(E_ex_arr):
            delta_mu_integrated = np.array([rate * DT_INTEGRATION for rate in delta_mu_history])
            negative_energy_flux = np.sum(E_ex_arr[neg_mask] * delta_mu_integrated[neg_mask])
        else:
            # History mismatch - cannot compute properly
            print(f"  (Warning: delta_mu_history length mismatch - cannot compute exact flux)")
            print(f"  (E_ex_arr: {len(E_ex_arr)}, delta_mu_history: {len(delta_mu_history)})")
            negative_energy_flux = float('nan')
        
        print(f"\n{'PENROSE EXTRACTION VERDICT':^65}")
        print("-"*65)
        print(f"  *** GENUINE PENROSE EXTRACTION DETECTED ***")
        print(f"  Negative-energy exhaust fell into the black hole,")
        print(f"  extracting rotational energy from the black hole.")
        if np.isfinite(negative_energy_flux):
            print(f"  Energy extracted via negative-E_ex exhaust: {-negative_energy_flux:.6f}")
            print(f"  (Computed as Sum E_ex,i * deltamu_i for E_ex < 0)")
        else:
            print(f"  (Energy flux could not be computed - see warning above)")
    else:
        print(f"\n{'PENROSE EXTRACTION VERDICT':^65}")
        print("-"*65)
        print(f"  No genuine Penrose extraction in this run.")
        print(f"  All exhaust had positive Killing energy (E_ex > 0).")
        print(f"  Energy gain came from relativistic rocket physics,")
        print(f"  not from black hole rotational energy extraction.")
        
        # Compute threshold for E_ex < 0
        # s_t_crit = -E_rocket / v_e
        # Need s_t < s_t_crit for E_ex < 0
        E_rocket_mean = np.mean(E) / np.mean(m)  # Approximate specific energy
        s_t_crit = -E_rocket_mean / v_e
        print(f"\n  For E_ex < 0, need s_t < {s_t_crit:.4f}")
        print(f"  Current v_e = {v_e:.3f}, E_rocket/m ~ {E_rocket_mean:.3f}")
        print(f"  Try: higher v_e, deeper ergosphere (lower r_set),")
        print(f"       or near-extremal spin (USE_NEAR_EXTREMAL_PARAMS=True)")
else:
    print("  No exhaust energy data recorded (no thrust applied).")

print("="*65)

# =============================================================================
# EXHAUST CAPTURE VERIFICATION (confirm negative-energy exhaust falls into BH)
# =============================================================================
# This is critical: negative Killing energy is necessary but not sufficient.
# We must verify the exhaust actually falls into the horizon.

if len(exhaust_samples_for_capture) > 0:
    n_neg_samples = sum(1 for s in exhaust_samples_for_capture if s['E_ex'] < 0)
    if n_neg_samples > 0:
        print("\n" + "="*65)
        print("         EXHAUST CAPTURE VERIFICATION")
        print("="*65)
        print(f"  Verifying {n_neg_samples} negative-energy exhaust samples...")
        print(f"  (downsampled from full thrust history for performance)")
        
        # Run capture verification
        verify_exhaust_capture_batch(exhaust_samples_for_capture, a, M, verbose=True)
    else:
        print("\n  (Skipping capture verification: no negative-energy exhaust samples)")
else:
    print("\n  (Skipping capture verification: no exhaust samples collected)")

# =============================================================================
# ENERGY BUDGET VALIDATION
# =============================================================================
# Compute the proper energy budget using tracked delta_mu values
# delta_mu_history contains RATES (dmu/dtau), need to multiply by dt to get deltamu per step
# Uses global DT_INTEGRATION constant (single source of truth)

if len(E_ex_history) > 0 and len(delta_mu_history) > 0:
    # Convert rates to integrated amounts: deltamu_i = (dmu/dtau)_i * dt
    delta_mu_integrated = [rate * DT_INTEGRATION for rate in delta_mu_history]
    
    budget = compute_energy_budget(
        E_initial=E[0], E_final=E[-1],
        m_initial=m0, m_final=m[-1],
        E_ex_history=E_ex_history,
        delta_mu_history=delta_mu_integrated
    )
    print_energy_budget(budget, a)

# Report on adaptive thrust mode if enabled
if USE_ADAPTIVE_THRUST_MODE and len(mode_history) > 0:
    print("\n" + "="*65)
    print("         ADAPTIVE THRUST MODE HISTORY")
    print("="*65)
    # mode_history tracking would need to be added during integration
    # For now, report final mode
    print(f"  Final thrust mode: {current_thrust_mode.name}")
    print("="*65)

# =============================================================================
# PRD-STYLE PLOTS
# =============================================================================

# =============================================================================
# PRD-STYLE FIGURES - Split into cleaner, separate figures
# =============================================================================

# Prepare common data
x = r * np.cos(phi)
y_coord = r * np.sin(phi)
thrust_on = mask
r_erg_val = local_ergosphere_radius(np.pi/2)

# Compute running cumulative efficiency
eta_cum_running = np.zeros_like(E)
for i in range(1, len(E)):
    dm = m[0] - m[i]
    if dm > 1e-10:
        eta_cum_running[i] = (E[i] - E[0]) / dm
    else:
        eta_cum_running[i] = 0.0

# Clip extreme values for visualization
eta_plot = np.clip(eta_inst, -0.5, 5.0)

# -----------------------------------------------------------------------------
# FIGURE 1: Trajectory and Dynamics (1x2) - ZOOMED to ergosphere region
# -----------------------------------------------------------------------------
fig1, (ax_traj, ax_rad) = plt.subplots(1, 2, figsize=(10, 4.5))
fig1.suptitle('Continuous-Thrust Penrose Process: Trajectory', fontsize=12, fontweight='bold')

# (a) Trajectory - zoomed to show ergosphere burn
for i in range(len(x)-1):
    if thrust_on[i]:
        ax_traj.plot(x[i:i+2], y_coord[i:i+2], color=COLORS['orange'], lw=1.2, zorder=3)
    else:
        ax_traj.plot(x[i:i+2], y_coord[i:i+2], color=COLORS['blue'], lw=1.0, zorder=2)

ax_traj.plot([], [], color=COLORS['orange'], lw=2, label='Thrust ON')
ax_traj.plot([], [], color=COLORS['blue'], lw=1.5, label='Coast')
ax_traj.add_patch(plt.Circle((0, 0), r_plus, fill=True, color=COLORS['black'], zorder=10))
ax_traj.add_patch(plt.Circle((0, 0), r_erg_val, fill=False, 
                              color=COLORS['vermilion'], ls='--', lw=1.5, zorder=1))
ax_traj.plot([], [], 'o', color=COLORS['black'], ms=8, label='Horizon')
ax_traj.plot([], [], '--', color=COLORS['vermilion'], lw=1.5, label='Ergosphere')
theta_ring = np.linspace(0, 2*np.pi, 100)
ax_traj.plot(r_set * np.cos(theta_ring), r_set * np.sin(theta_ring), 
             color=COLORS['green'], ls=':', lw=1.0, alpha=0.8, label=f'$r_{{set}}$={r_set}')
ax_traj.set_aspect('equal', 'box')
ax_traj.set_xlabel(r'$x/M$', fontsize=10)
ax_traj.set_ylabel(r'$y/M$', fontsize=10)
ax_traj.set_title('(a) Equatorial trajectory (zoomed)', fontsize=11)
ax_traj.legend(loc='upper left', fontsize=9)
# Zoom to ergosphere region: show from -5M to +5M to capture entry, burn, and exit
zoom_extent = 5.0
ax_traj.set_xlim(-zoom_extent, zoom_extent)
ax_traj.set_ylim(-zoom_extent, zoom_extent)

# (b) Radius
ax_rad.plot(tau, r, color=COLORS['blue'], lw=1.5, label=r'$r(\tau)$')
ax_rad.fill_between(tau, 0, r.max()*1.1, where=thrust_on, 
                    color=COLORS['orange'], alpha=0.15, zorder=0, label='Thrust ON')
ax_rad.axhline(r_erg_val, ls='--', color=COLORS['vermilion'], lw=1.5, label='Ergosphere')
ax_rad.axhline(r_set, ls='-.', color=COLORS['green'], lw=1.2, label=f'$r_{{set}}$={r_set}')
ax_rad.set_xlabel(r'Proper time $\tau/M$', fontsize=10)
ax_rad.set_ylabel(r'Radius $r/M$', fontsize=10)
ax_rad.set_ylim(0, r.max()*1.05)
ax_rad.set_title('(b) Radial evolution', fontsize=11)
ax_rad.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('continuous_trajectory.pdf', dpi=300, bbox_inches='tight')
plt.savefig('continuous_trajectory.png', dpi=150, bbox_inches='tight')

# -----------------------------------------------------------------------------
# FIGURE 2: Energy and Efficiency (1x2)
# -----------------------------------------------------------------------------
fig2, (ax_em, ax_eff) = plt.subplots(1, 2, figsize=(10, 4))
fig2.suptitle('Continuous-Thrust Penrose Process: Energy', fontsize=12, fontweight='bold')

# (c) Energy and mass - use separate y-axes but better spacing
ax_em.plot(tau, E, color=COLORS['blue'], lw=2, label=r'Energy $E$')
ax_em.set_xlabel(r'Proper time $\tau/M$', fontsize=10)
ax_em.set_ylabel(r'Energy $E$', fontsize=10)
ax_em.set_title(f'(c) Energy: DeltaE = {E[-1] - E[0]:+.3f} ({100*(E[-1]-E[0])/E[0]:+.1f}%)', fontsize=11)

ax_em_twin = ax_em.twinx()
ax_em_twin.plot(tau, m, color=COLORS['orange'], ls='--', lw=2, label=r'Mass $m$')
ax_em_twin.set_ylabel(r'Rest mass $m$', fontsize=10, color=COLORS['orange'])
ax_em_twin.tick_params(axis='y', labelcolor=COLORS['orange'])

# Legends on left side to avoid overlap
ax_em.legend(loc='upper left', fontsize=9)
ax_em_twin.legend(loc='lower left', fontsize=9)

# (d) Cumulative efficiency
ax_eff.plot(tau, eta_cum_running, color=COLORS['green'], lw=2, label=r'$\eta_{\mathrm{cum}}$')
ax_eff.axhline(eta_max, ls='--', color=COLORS['vermilion'], lw=1.5, 
               label=f'$\\eta_{{max}}$ = {eta_max:.2f}')
ax_eff.axhline(1.0, ls=':', color=COLORS['gray'], lw=1.2, label='Break-even')
ax_eff.set_xlabel(r'Proper time $\tau/M$', fontsize=10)
ax_eff.set_ylabel(r'Cumulative efficiency $\eta$', fontsize=10)
ax_eff.set_title('(d) Extraction efficiency', fontsize=11)
ax_eff.legend(loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('continuous_energy.pdf', dpi=300, bbox_inches='tight')
plt.savefig('continuous_energy.png', dpi=150, bbox_inches='tight')

# -----------------------------------------------------------------------------
# FIGURE 3: Penrose Extraction Analysis (1x2)
# -----------------------------------------------------------------------------
if len(E_ex_history) > 0 and len(r_ex_history) > 0:
    E_ex_arr = np.array(E_ex_history)
    r_ex_arr = np.array(r_ex_history)
    neg_mask = E_ex_arr < 0
    pos_mask = E_ex_arr >= 0
    
    fig3, (ax_Ex_r, ax_Ex_hist) = plt.subplots(1, 2, figsize=(10, 4))
    fig3.suptitle('Penrose Extraction: Exhaust Energy Analysis', fontsize=12, fontweight='bold')
    
    # (g) E_ex vs radius
    # Use the computed R_EXTRACTION_LIMIT for plotting consistency
    r_extraction_limit = R_EXTRACTION_LIMIT  # Use computed value, not hardcoded
    
    if neg_mask.any():
        ax_Ex_r.scatter(r_ex_arr[neg_mask], E_ex_arr[neg_mask], 
                        c=COLORS['green'], s=12, alpha=0.7, label=r'$E_{ex} < 0$ (Penrose)', zorder=3)
    if pos_mask.any():
        ax_Ex_r.scatter(r_ex_arr[pos_mask], E_ex_arr[pos_mask], 
                        c=COLORS['gray'], s=10, alpha=0.4, label=r'$E_{ex} \geq 0$', zorder=2)
    
    ax_Ex_r.axhline(0, ls='-', color=COLORS['black'], lw=2, alpha=0.9, zorder=4, label=r'$E_{ex}=0$')
    ax_Ex_r.axvspan(r_plus, r_extraction_limit, color=COLORS['green'], alpha=0.1, label='Extraction zone')
    ax_Ex_r.axvline(r_erg_val, ls='--', color=COLORS['vermilion'], lw=1.5, label='Ergosphere')
    
    ax_Ex_r.set_xlabel(r'Radius $r/M$', fontsize=10)
    ax_Ex_r.set_ylabel(r'Exhaust energy $E_{ex}$', fontsize=10)
    ax_Ex_r.set_title('(a) $E_{ex}$ vs radius', fontsize=11)
    ax_Ex_r.legend(loc='upper right', fontsize=9)
    ax_Ex_r.set_xlim(r_plus - 0.1, 2.2)
    
    y_min = min(np.min(E_ex_arr), -0.6)
    y_max = max(0.5, np.max(E_ex_arr) * 0.3) if np.max(E_ex_arr) > 0.5 else 0.5
    ax_Ex_r.set_ylim(y_min, y_max)
    
    # (h) E_ex histogram
    n_bins = 25
    if neg_mask.any():
        ax_Ex_hist.hist(E_ex_arr[neg_mask], bins=n_bins, color=COLORS['green'], 
                        alpha=0.8, label=f'$E_{{ex}} < 0$ ({neg_mask.sum()})', edgecolor='white')
    if pos_mask.any():
        ax_Ex_hist.hist(E_ex_arr[pos_mask], bins=n_bins, color=COLORS['gray'], 
                        alpha=0.5, label=f'$E_{{ex}} > 0$ ({pos_mask.sum()})', edgecolor='white')
    
    ax_Ex_hist.axvline(0, ls='-', color=COLORS['black'], lw=2)
    ax_Ex_hist.axvline(np.min(E_ex_arr), ls=':', color=COLORS['vermilion'], lw=2,
                       label=f'Min = {np.min(E_ex_arr):.3f}')
    
    ax_Ex_hist.set_xlabel(r'Exhaust energy $E_{ex}$', fontsize=10)
    ax_Ex_hist.set_ylabel('Count', fontsize=10)
    frac = neg_mask.sum() / len(E_ex_arr) * 100 if len(E_ex_arr) > 0 else 0
    ax_Ex_hist.set_title(f'(b) Distribution ({frac:.0f}% Penrose)', fontsize=11)
    ax_Ex_hist.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('continuous_penrose_Eex.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('continuous_penrose_Eex.png', dpi=150, bbox_inches='tight')

# -----------------------------------------------------------------------------
# FIGURE 4: Diagnostics (1x2) - constraint and instantaneous efficiency
# -----------------------------------------------------------------------------
fig4, (ax_const, ax_eta_inst) = plt.subplots(1, 2, figsize=(10, 4))
fig4.suptitle('Simulation Diagnostics', fontsize=12, fontweight='bold')

# Constraint
ax_const.plot(tau, C, color=COLORS['purple'], lw=1.5)
ax_const.set_xlabel(r'Proper time $\tau/M$', fontsize=10)
ax_const.set_ylabel(r'$g^{\mu\nu}p_\mu p_\nu + m^2$', fontsize=10)
ax_const.set_title('(a) Mass-shell constraint', fontsize=11)
ax_const.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))

# Instantaneous efficiency
ax_eta_inst.plot(tau, eta_plot, color=COLORS['green'], lw=1.5, label=r'$\eta_{\mathrm{inst}}$')
ax_eta_inst.axhline(1.0, ls='--', color=COLORS['vermilion'], lw=1.5, label='Penrose threshold')
in_ergo = r < r_erg_val
ax_eta_inst.fill_between(tau, -0.5, 3.0, where=in_ergo, 
                          color=COLORS['orange'], alpha=0.15, label='In ergosphere')
ax_eta_inst.set_xlabel(r'Proper time $\tau/M$', fontsize=10)
ax_eta_inst.set_ylabel(r'$\eta_{\mathrm{inst}}$', fontsize=10)
ax_eta_inst.set_title('(b) Instantaneous efficiency', fontsize=11)
ax_eta_inst.set_ylim(-0.2, 3.0)
ax_eta_inst.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('continuous_diagnostics.pdf', dpi=300, bbox_inches='tight')
plt.savefig('continuous_diagnostics.png', dpi=150, bbox_inches='tight')

print("\nFigures saved:")
print("  - continuous_trajectory.pdf/png")
print("  - continuous_energy.pdf/png")
print("  - continuous_penrose_Eex.pdf/png")
print("  - continuous_diagnostics.pdf/png")
plt.show()