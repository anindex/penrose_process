"""
Single-Thrust Penrose Process
=============================
The classic impulsive burn version. Good for sanity-checking the continuous
thrust simulation. The basic idea: fall into the ergosphere, fire once at
the right moment, and use the negative-energy exhaust trick to escape with
more energy than you started with.

One thing that wasn't obvious at first: you need the exhaust to go retrograde
(against the black hole's rotation) to get E_ex < 0. The orbit itself can be
prograde - that actually helps reach deeper into the ergosphere.

PHYSICS SUMMARY:
----------------
This implements the textbook Penrose mechanism:
1. Rocket falls into the ergosphere on a prograde flyby
2. At optimal radius, fires retrograde exhaust with E_ex < 0
3. By 4-momentum conservation, rocket gains energy and escapes

KEY EQUATIONS:
- 4-momentum conservation: p'_mu = p_mu - deltamu * u_{ex,mu}
- Mass-shell constraint: g^{munu} p_mu p_nu = -m^2 determines new mass
- Penrose condition: E_ex = -u_{ex,t} < 0 (negative Killing energy)

Reference: Penrose & Floyd (1971), Nature Phys. Sci. 229, 177
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


from kerr_utils import (
    COLORS, setup_prd_style,
    kerr_metric_components, horizon_radius, ergosphere_radius,
    compute_pt_from_mass_shell, compute_energy,
    theoretical_penrose_limit, compute_cumulative_efficiency,
    compute_exhaust_energy, compute_exhaust_4velocity, build_rocket_rest_basis,
    apply_exact_impulse, compute_optimal_exhaust_direction,
    compute_energy_budget, print_energy_budget,
    integrate_exhaust_geodesic, verify_exhaust_capture_batch
)


setup_prd_style()

# =============================================================================
# PARAMETERS
# =============================================================================
M = 1.0
a = 0.95  # high spin but not extremal - keeps numerics stable
r_plus = horizon_radius(a, M)
r_erg_eq = ergosphere_radius(np.pi/2, a, M)  # = 2M at equator
r_safe = r_plus + 0.02  # safety margin from horizon

# Validate safety margin
if r_safe >= r_erg_eq:
    raise ValueError(f"Safety margin too large: r_safe >= r_erg. Cannot operate in ergosphere.")

# Rocket parameters
T_max = 80.0     # thrust magnitude (geometrized units)
v_e = 0.95       # exhaust velocity - needs to be high for E_ex < 0

# Validate exhaust velocity
if v_e >= 1.0:
    raise ValueError(f"Exhaust velocity v_e = {v_e} >= c violates causality!")
gamma_e = 1.0 / np.sqrt(1.0 - v_e**2)

# Prograde flyby orbit with periapsis inside ergosphere
# Prograde orbit + retrograde exhaust = Penrose extraction
# E0 > 1 means unbound (escapes to infinity after extraction)

# Prograde flyby - periapsis inside ergosphere
E0 = 1.20        # unbound (E > 1), comes from/escapes to infinity
Lz0 = 3.0        # positive = prograde, balanced for extraction + escape
r0 = 10.0        # start moderately far for focused simulation

# =============================================================================
# LOCAL METRIC WRAPPER (uses global a, M)
# =============================================================================
def metric_components(r, th):
    """Wrapper for kerr_metric_components using module-level a, M."""
    cov, con = kerr_metric_components(r, th, a, M)
    g_tt, g_tphi, g_rr, g_thth, g_phiphi = cov
    gu_tt, gu_tphi, gu_rr, gu_thth, gu_phiphi = con
    # Return in format expected by existing code
    return (g_rr, g_phiphi), (gu_tt, gu_rr, gu_thth, gu_phiphi, gu_tphi)

def compute_pt(r, th, pr, pphi, m):
    """Wrapper for compute_pt_from_mass_shell."""
    return compute_pt_from_mass_shell(r, th, pr, pphi, m, a, M)

def local_compute_energy(r, th, pr, pphi, m):
    """Wrapper for compute_energy."""
    return compute_energy(r, th, pr, pphi, m, a, M)

def local_ergosphere_radius(th=np.pi/2):
    """Wrapper for ergosphere_radius."""
    return ergosphere_radius(th, a, M)

# =============================================================================
# GEODESIC MOTION + IMPULSE
# =============================================================================
def dynamics_freefall(tau, state):
    """
    Geodesic motion - no thrust.
    
    State: [r, th, phi, pr, pth, pphi, m, pt]
    
    IMPORTANT: For geodesic motion, p_t and p_phi are constants of motion (Killing vectors).
    We now integrate p_t explicitly with dp_t/dtau = 0 to avoid numerical drift from 
    repeatedly solving the mass-shell constraint.
    """
    r, th, phi, pr, pth, pphi, m, pt = state
    
    (gl_rr, gl_pp), (gu_tt, gu_rr, gu_thth, gu_pp, gu_tp) = metric_components(r, th)
    
    # Hamiltonian gradient via finite differences
    # Note: pt is now taken from state, not recomputed
    eps = 1e-6
    def H_val(r_):
        _, (u_tt, u_rr, u_thth, u_pp, u_tp) = metric_components(r_, th)
        return 0.5*(u_tt*pt**2 + u_rr*pr**2 + u_pp*pphi**2 + 2*u_tp*pt*pphi)
    dH_dr = (H_val(r+eps) - H_val(r-eps))/(2*eps)
    
    # velocities from Hamilton's eqns: dx^mu/dtau = p^mu/m (proper velocity)
    # Dividing by m ensures equivalence principle: coasting is mass-independent
    ur = gu_rr*pr / m
    uth = gu_thth*pth / m
    uphi = (gu_pp*pphi + gu_tp*pt) / m
    
    # dpr/dtau = -(1/m) dH/dr for proper time parameterization
    dpr = -dH_dr / m
    
    # Geodesic: p_t, p_phi, m are constants (Killing + no thrust)
    dpt = 0.0
    dpphi = 0.0
    dm = 0.0

    return [ur, uth, uphi, dpr, 0.0, dpphi, dm, dpt]

def apply_impulse_exact(state, delta_mu, a_spin, M_bh, v_exhaust, minimize_E_ex=True):
    """
    Apply an instantaneous thrust impulse with EXACT 4-momentum conservation.
    
    This is the CRITICAL fix for energy budget: instead of imposing mass loss
    and adding momentum changes separately, we use exact 4-momentum conservation:
    
        p'_mu = p_mu - deltamu * u_{ex,mu}
    
    The new rocket mass m' is then determined from the mass-shell constraint:
    
        m'^2 = -g^{munu} p'_mu p'_nu
    
    Parameters
    ----------
    state : list
        [r, th, phi, pr, pth, pphi, m, pt] - current state (8-element)
    delta_mu : float
        Exhaust rest mass to eject (positive)
    a_spin, M_bh : float
        Black hole parameters
    v_exhaust : float
        Exhaust velocity in rocket rest frame (0 < v_e < 1)
    minimize_E_ex : bool
        If True, scan thrust angles to minimize exhaust energy (Penrose mode)
    
    Returns
    -------
    new_state, E_ex, delta_mu, u_ex_contra
        E_ex < 0 indicates genuine Penrose extraction.
        u_ex_contra is the exhaust 4-velocity (for geodesic verification).
    """
    from kerr_utils import (compute_exhaust_4velocity, apply_exact_impulse, 
                            compute_exhaust_energy)
    
    # Handle both 7-element (legacy) and 8-element (new) state vectors
    if len(state) == 8:
        r, th, phi, pr, pth, pphi, m, pt = state
    else:
        r, th, phi, pr, pth, pphi, m = state
        pt = compute_pt_from_mass_shell(r, th, pr, pphi, m, a_spin, M_bh)
    
    cov, con = kerr_metric_components(r, th, a_spin, M_bh)
    g_tt, g_tphi, g_rr, _, g_phiphi = cov
    gu_tt, gu_tphi, gu_rr, gu_thth, gu_phiphi = con
    
    # Get current covariant momentum
    pt = compute_pt_from_mass_shell(r, th, pr, pphi, m, a_spin, M_bh)
    p_cov = (pt, pr, pphi)
    
    # Compute 4-velocity
    u_t = (gu_tt * pt + gu_tphi * pphi) / m
    u_r = gu_rr * pr / m
    u_phi = (gu_tphi * pt + gu_phiphi * pphi) / m
    u_vec = np.array([u_t, u_r, u_phi], dtype=float)
    
    # Build rocket rest frame basis
    e_r, e_phi = build_rocket_rest_basis(u_vec, g_tt, g_tphi, g_rr, g_phiphi)
    
    if e_r is None or e_phi is None:
        raise ValueError("Failed to build rocket rest frame basis")
    
    # Find optimal exhaust direction
    best_E_ex = float('inf')
    best_u_ex_result = None
    best_s_vec = None
    
    # Also track best option without escape constraint (fallback)
    best_E_ex_any = float('inf')
    best_u_ex_result_any = None
    
    for alpha in np.linspace(-np.pi/2, np.pi/2, 37):
        s_r = np.sin(alpha)
        s_phi_mag = np.cos(alpha)
        
        for sign_phi in [+1.0, -1.0]:
            s_vec = s_r * e_r + (sign_phi * s_phi_mag) * e_phi
            
            # Compute full exhaust 4-velocity
            u_ex_result = compute_exhaust_4velocity(
                u_vec, s_vec, v_exhaust, g_tt, g_tphi, g_rr, g_phiphi
            )
            E_ex_trial = u_ex_result['E_ex']
            
            if minimize_E_ex:
                # For Penrose extraction: minimize E_ex (want E_ex < 0)
                # Also check escape constraints
                u_ex_cov = u_ex_result['u_ex_cov']
                
                # Trial momentum after impulse
                pt_trial = pt - delta_mu * u_ex_cov[0]
                pr_trial = pr - delta_mu * u_ex_cov[1]
                pphi_trial = pphi - delta_mu * u_ex_cov[2]
                
                E_trial = -pt_trial
                
                # Track best with any E (fallback if no escape-valid direction)
                if E_ex_trial < best_E_ex_any:
                    best_E_ex_any = E_ex_trial
                    best_u_ex_result_any = u_ex_result
                
                # CORRECTED: Proper unbound condition is E > m (not E > 1)
                # Estimate post-impulse mass: m_after ~ m - gamma_e * deltamu
                # (The exact value is computed after, but this is a good estimate)
                gamma_exhaust = 1.0 / np.sqrt(1 - v_exhaust**2)
                m_after_estimate = m - gamma_exhaust * delta_mu
                can_escape = E_trial > m_after_estimate
                
                if can_escape and E_ex_trial < best_E_ex:
                    best_E_ex = E_ex_trial
                    best_u_ex_result = u_ex_result
                    best_s_vec = s_vec
            else:
                # Simple mode: just use retrograde
                if sign_phi < 0:  # Retrograde
                    best_E_ex = E_ex_trial
                    best_u_ex_result = u_ex_result
                    best_s_vec = s_vec
                    break
    
    # Fallback: if no direction satisfies E > 1, use best overall
    if best_u_ex_result is None:
        if best_u_ex_result_any is not None:
            best_u_ex_result = best_u_ex_result_any
            best_E_ex = best_E_ex_any
        else:
            raise ValueError("No valid exhaust direction found")
    
    # Apply exact 4-momentum conservation
    result = apply_exact_impulse(
        p_cov, m, delta_mu, tuple(best_u_ex_result['u_ex_cov']), r, th, a_spin, M_bh
    )
    
    pt_new, pr_new, pphi_new = result['p_cov_new']
    m_new = result['m_new']
    
    # Return 8-element state with pt included
    new_state = [r, th, phi, pr_new, pth, pphi_new, m_new, pt_new]
    
    return new_state, best_E_ex, delta_mu, best_u_ex_result['u_ex_contra']


def apply_impulse(state, delta_m_fraction=0.3, minimize_E_ex=False):
    """
    Apply an instantaneous thrust impulse (LEGACY - wrapper around exact version).
    
    For backward compatibility. Now uses exact 4-momentum conservation internally.
    
    Parameters
    ----------
    state : list
        State vector [r, th, phi, pr, pth, pphi, m, pt] (8-element) or
        [r, th, phi, pr, pth, pphi, m] (7-element, legacy)
    delta_m_fraction : float
        Approximate fraction of rocket mass to expel. Note: The actual mass loss
        is determined by exact 4-momentum conservation, so this is used to compute
        an approximate exhaust rest mass: deltamu = (delta_m_fraction * m) / gamma_e.
        This is NOT an exact relation for finite impulses due to recoil effects.
    minimize_E_ex : bool
        If True, scan thrust angles to minimize exhaust energy (Penrose mode).
    
    Returns
    -------
    new_state, E_ex, delta_mu
        E_ex < 0 indicates genuine Penrose extraction.
        delta_mu is the exhaust rest mass.
    """
    # Handle both 7-element and 8-element state
    if len(state) == 8:
        r, th, phi, pr, pth, pphi, m, pt = state
    else:
        r, th, phi, pr, pth, pphi, m = state
    
    # Convert fuel fraction to exhaust rest mass
    # For a rocket: dm_rocket = gamma_e * deltamu (relativistic mass-energy)
    gamma_e = 1.0 / np.sqrt(1.0 - v_e**2)
    dm_rocket = delta_m_fraction * m  # Rocket mass loss
    delta_mu = dm_rocket / gamma_e     # Exhaust rest mass
    
    try:
        new_state, E_ex, delta_mu_out, u_ex_contra = apply_impulse_exact(
            state, delta_mu, a, M, v_e, minimize_E_ex=minimize_E_ex
        )
        return new_state, E_ex, delta_mu_out
    except ValueError as e:
        # Fallback to approximate method if exact fails
        print(f"Warning: Exact impulse failed ({e}), using approximate method")
        return _apply_impulse_approximate(state, delta_m_fraction, minimize_E_ex)


def _apply_impulse_approximate(state, delta_m_fraction=0.3, minimize_E_ex=False):
    """
    Legacy approximate impulse method (kept as fallback).
    """
    # Handle both 7-element and 8-element state
    if len(state) == 8:
        r, th, phi, pr, pth, pphi, m, pt_in = state
    else:
        r, th, phi, pr, pth, pphi, m = state
        pt_in = None
    
    cov, con = kerr_metric_components(r, th, a, M)
    g_tt, g_tphi, g_rr, _, g_phiphi = cov
    gu_tt, gu_tphi, gu_rr, gu_thth, gu_phiphi = con
    
    # Rocket mass loss (imposed)
    dm = delta_m_fraction * m
    m_new = m - dm
    
    # Exhaust rest mass: dm = deltamu * gamma_e (relativistic mass-energy relation)
    gamma_e = 1.0 / np.sqrt(1.0 - v_e**2)
    delta_mu = dm / gamma_e  # Exhaust rest mass
    
    # Relativistic rocket thrust
    v_exhaust_momentum = v_e * gamma_e
    thrust = dm * v_exhaust_momentum
    
    # Get current covariant momentum and 4-velocity
    pt = compute_pt(r, th, pr, pphi, m)
    u_t = (gu_tt * pt + gu_tphi * pphi) / m
    u_r = gu_rr * pr / m
    u_phi = (gu_tphi * pt + gu_phiphi * pphi) / m
    u_vec = np.array([u_t, u_r, u_phi], dtype=float)
    
    # Build rocket rest frame basis
    e_r, e_phi = build_rocket_rest_basis(u_vec, g_tt, g_tphi, g_rr, g_phiphi)
    
    if e_r is None or e_phi is None:
        # Fallback: simple retrograde exhaust
        thrust_sign = -1.0 if Lz0 > 0 else 1.0
        delta_pphi = thrust_sign * thrust / np.sqrt(g_phiphi)
        pphi_new = pphi + delta_pphi
        
        # Compute E_ex for this direction
        s_vec = thrust_sign * np.array([0.0, 0.0, 1.0 / np.sqrt(g_phiphi)])
        E_ex, _ = compute_exhaust_energy(u_vec, s_vec, v_e, g_tt, g_tphi, g_phiphi)
        
        return [r, th, phi, pr, pth, pphi_new, m_new], E_ex, delta_mu
    
    if not minimize_E_ex:
        # Simple mode: retrograde exhaust for prograde momentum gain
        thrust_sign = -1.0 if Lz0 > 0 else 1.0
        s_vec = thrust_sign * e_phi  # Retrograde for Penrose
        
        f_vec = thrust * s_vec
        
        # Momentum changes (contravariant -> covariant)
        dp_contra_t = f_vec[0] + u_t * (-dm)
        dp_contra_r = f_vec[1] + u_r * (-dm)
        dp_contra_phi = f_vec[2] + u_phi * (-dm)
        
        dpt = g_tt * dp_contra_t + g_tphi * dp_contra_phi
        dpr_impulse = g_rr * dp_contra_r
        dpphi = g_tphi * dp_contra_t + g_phiphi * dp_contra_phi
        
        # Compute E_ex
        E_ex, _ = compute_exhaust_energy(u_vec, s_vec, v_e, g_tt, g_tphi, g_phiphi)
        
        pr_new = pr + dpr_impulse
        pphi_new = pphi + dpphi
        
        return [r, th, phi, pr_new, pth, pphi_new, m_new], E_ex, delta_mu
    
    # Optimization mode: scan angles to minimize E_ex
    best_E_ex = float('inf')
    best_dpr, best_dpphi = 0.0, 0.0
    
    for alpha in np.linspace(-np.pi/2, np.pi/2, 37):
        s_r = np.sin(alpha)
        s_phi_mag = np.cos(alpha)
        
        for sign_phi in [+1.0, -1.0]:
            s_vec = s_r * e_r + (sign_phi * s_phi_mag) * e_phi
            
            E_ex_trial, _ = compute_exhaust_energy(u_vec, s_vec, v_e, g_tt, g_tphi, g_phiphi)
            
            # Compute resulting state
            f_vec = thrust * s_vec
            dp_contra_t = f_vec[0] + u_t * (-dm)
            dp_contra_r = f_vec[1] + u_r * (-dm)
            dp_contra_phi = f_vec[2] + u_phi * (-dm)
            
            dpt = g_tt * dp_contra_t + g_tphi * dp_contra_phi
            dpr_impulse = g_rr * dp_contra_r
            dpphi = g_tphi * dp_contra_t + g_phiphi * dp_contra_phi
            
            pt_new = pt + dpt
            E_new = -pt_new  # Energy = -p_t
            pr_trial = pr + dpr_impulse
            
            # Constraints for escape
            can_escape = E_new > 1.0 and pr_trial > 0
            
            if can_escape and E_ex_trial < best_E_ex:
                best_E_ex = E_ex_trial
                best_dpr = dpr_impulse
                best_dpphi = dpphi
    
    pr_new = pr + best_dpr
    pphi_new = pphi + best_dpphi
    
    return [r, th, phi, pr_new, pth, pphi_new, m_new], best_E_ex, delta_mu

# =============================================================================
# SIMULATION
# =============================================================================
print("="*60)
print("SINGLE-THRUST PENROSE PROCESS SIMULATION")
print("="*60)
print(f"Black hole spin: a = {a}")
print(f"Horizon radius: r+ = {r_plus:.4f} M")
print(f"Ergosphere radius (equator): r_ergo = {local_ergosphere_radius():.4f} M")
print(f"BH rotational energy fraction: {100*theoretical_penrose_limit(a):.2f}% (thermodynamic bound)")
orbit_type = "retrograde" if Lz0 < 0 else "prograde"
print(f"Initial orbit: E0 = {E0}, Lz0 = {Lz0} ({orbit_type})")
print(f"Exhaust velocity: v_e = {v_e}c (gamma_e = {1/np.sqrt(1-v_e**2):.2f})")
print()

# Scan trigger radii to find optimal
print("Scanning trigger radii for optimal energy extraction...")
radii_scan = np.linspace(r_plus + 0.05, local_ergosphere_radius() - 0.1, 30)
energy_results = []
mass_results = []
min_r_results = []  # Track minimum radius reached

for r_trig in radii_scan:
    # Initialize state
    _, (gu_tt, gu_rr, _, gu_pp, gu_tp) = metric_components(r0, np.pi/2)
    # Solve for initial pr from mass-shell: g^munu p_mu p_nu = -m^2
    pt0 = -E0  # p_t = -E
    rem = gu_tt*pt0**2 + 2*gu_tp*pt0*Lz0 + gu_pp*Lz0**2 + 1.0
    if rem > 0:  # Forbidden region
        energy_results.append(E0)
        mass_results.append(1.0)
        min_r_results.append(r0)
        continue
    pr0 = -np.sqrt(-rem / gu_rr)  # Ingoing (negative)
    y0 = [r0, np.pi/2, 0.0, pr0, 0.0, Lz0, 1.0, pt0]  # 8-element state includes pt
    
    # Event: stop at trigger radius (crossing inward)
    def trigger_event(t, y): return y[0] - r_trig
    trigger_event.terminal = True
    trigger_event.direction = -1  # Only trigger when r decreasing
    
    def horizon_event(t, y): return y[0] - r_safe
    horizon_event.terminal = True
    
    # Phase 1: Free fall to trigger radius
    sol1 = solve_ivp(dynamics_freefall, [0, 200], y0, method='Radau', 
                     events=[trigger_event, horizon_event], rtol=1e-9)
    
    # Check if we reached trigger or horizon
    if len(sol1.t_events[0]) == 0:  # Didn't reach trigger
        if len(sol1.t_events[1]) > 0:  # Hit horizon
            energy_results.append(0.0)
            mass_results.append(0.0)
            min_r_results.append(np.min(sol1.y[0]))
            continue
        else:  # Bounced off
            E_final = local_compute_energy(sol1.y[0,-1], np.pi/2, sol1.y[3,-1], sol1.y[5,-1], sol1.y[6,-1])
            energy_results.append(E_final)
            mass_results.append(sol1.y[6,-1])
            min_r_results.append(np.min(sol1.y[0]))
            continue
    
    # Phase 2: Apply instantaneous impulse
    state_at_trigger = sol1.y[:, -1]
    state_after_burn, E_ex_scan, _ = apply_impulse(state_at_trigger, delta_m_fraction=0.20, 
                                                   minimize_E_ex=True)
    
    # Phase 3: Continue trajectory after burn
    sol2 = solve_ivp(dynamics_freefall, [sol1.t[-1], sol1.t[-1] + 200], state_after_burn, 
                     method='Radau', events=horizon_event, rtol=1e-9)
    
    min_r = min(np.min(sol1.y[0]), np.min(sol2.y[0]))
    min_r_results.append(min_r)
    
    # Debug first few cases
    if len(energy_results) < 3:
        print(f"  r_trig={r_trig:.3f}: min_r={min_r:.3f}, m_final={sol2.y[6,-1]:.4f}, r_final={sol2.y[0,-1]:.4f}")
    
    if sol2.y[0, -1] < r_plus + 0.1:
        energy_results.append(0.0)  # Fell into BH
        mass_results.append(0.0)
    else:
        E_final = local_compute_energy(sol2.y[0,-1], np.pi/2, sol2.y[3,-1], sol2.y[5,-1], sol2.y[6,-1])
        energy_results.append(E_final)
        mass_results.append(sol2.y[6,-1])

# Debug: show min radius reached
print(f"Minimum radius reached: {min(min_r_results):.4f} M (need < {radii_scan[-1]:.4f} to trigger)")

energy_results = np.array(energy_results)
mass_results = np.array(mass_results)

# Find best result
valid_mask = (energy_results > 0) & (mass_results > 0)
if valid_mask.any():
    best_idx = np.argmax(energy_results[valid_mask])
    best_idx = np.where(valid_mask)[0][best_idx]
    best_E = energy_results[best_idx]
    best_r = radii_scan[best_idx]
    best_m = mass_results[best_idx]
else:
    best_idx = 0
    best_E = E0
    best_r = radii_scan[0]
    best_m = 1.0

print(f"\nOptimal trigger radius: r_trig = {best_r:.4f} M")
print(f"Final energy: E_final = {best_E:.4f}")
print(f"Final mass: m_final = {best_m:.4f}")

# =============================================================================
# RUN OPTIMAL TRAJECTORY (3-phase: infall, impulse, escape)
# =============================================================================
_, (gu_tt, gu_rr, _, gu_pp, gu_tp) = metric_components(r0, np.pi/2)
pt0 = -E0
rem = gu_tt*pt0**2 + 2*gu_tp*pt0*Lz0 + gu_pp*Lz0**2 + 1.0
pr0 = -np.sqrt(-rem / gu_rr)
y0 = [r0, np.pi/2, 0.0, pr0, 0.0, Lz0, 1.0, pt0]  # 8-element state includes pt

# Escape radius threshold - increased to definitively confirm escape
ESCAPE_RADIUS = 50.0  # Increased from 15M to 50M for definitive escape verification

# Event: stop at trigger radius
def trigger_event(t, y): return y[0] - best_r
trigger_event.terminal = True
trigger_event.direction = -1

def horizon_event(t, y): return y[0] - r_safe
horizon_event.terminal = True

def escape_event(t, y): return y[0] - ESCAPE_RADIUS
escape_event.terminal = True
escape_event.direction = 1  # Only when r increasing

# Phase 1: Free fall to trigger
sol1 = solve_ivp(dynamics_freefall, [0, 200], y0, method='Radau', 
                 events=[trigger_event, horizon_event], rtol=1e-9, dense_output=True)

# Phase 2: Apply impulse (use 20% mass fraction for comparable efficiency to continuous case)
state_at_trigger = sol1.y[:, -1]
state_after_burn, E_ex, delta_mu_impulse = apply_impulse(state_at_trigger, delta_m_fraction=0.20,
                                                          minimize_E_ex=True)

# Store exhaust energy for verification
print(f"\nExhaust energy at impulse: E_ex = {E_ex:.6f}")
if E_ex < 0:
    print(f"  [OK] GENUINE PENROSE EXTRACTION: Exhaust has NEGATIVE energy at infinity!")
else:
    print(f"  [!] E_ex > 0: Energy transfer, but not genuine Penrose mechanism")

# Phase 3: Continue after burn - integrate until escape or horizon
# Increased integration time to reach 50M
sol2 = solve_ivp(dynamics_freefall, [sol1.t[-1], sol1.t[-1] + 1000], state_after_burn, 
                 method='Radau', events=[horizon_event, escape_event], rtol=1e-9, dense_output=True)

# Determine trajectory outcome - use higher threshold
escaped = len(sol2.t_events[1]) > 0 or sol2.y[0, -1] > ESCAPE_RADIUS
captured = len(sol2.t_events[0]) > 0 or sol2.y[0, -1] < r_safe + 0.1

# Combine solutions
tau = np.concatenate([sol1.t, sol2.t])
r = np.concatenate([sol1.y[0], sol2.y[0]])
phi = np.concatenate([sol1.y[2], sol2.y[2]])
m = np.concatenate([sol1.y[6], sol2.y[6]])

# Compute energy history directly from pt (which is now in state[7])
# E = -pt by definition of Killing energy
E_hist1 = -sol1.y[7]  # E = -pt for each step
E_hist2 = -sol2.y[7]
E_hist = np.concatenate([E_hist1, E_hist2])

# Index where burn occurs
burn_idx = len(sol1.t) - 1

# =============================================================================
# EFFICIENCY ANALYSIS
# =============================================================================
E_final = E_hist[-1]
m_final = m[-1]
Delta_E = E_final - E0
Delta_m = 1.0 - m_final

if Delta_m > 1e-10:
    eta_cum = Delta_E / Delta_m
else:
    eta_cum = 0.0

# Two distinct limits: BH rotational budget vs Wald single-decay bound
from kerr_utils import wald_single_decay_limit, max_rotational_energy_fraction
eta_rot = max_rotational_energy_fraction(a)
eta_wald = wald_single_decay_limit()

print("\n" + "="*60)
print("EFFICIENCY ANALYSIS")
print("="*60)
print(f"Initial energy: E_0 = {E0:.4f}")
print(f"Final energy:   E_f = {E_final:.4f}")
print(f"Energy change:  DeltaE = {Delta_E:+.4f} ({100*Delta_E/E0:+.2f}%)")
print(f"Mass expelled:  Deltam = {Delta_m:.4f} ({100*Delta_m:.2f}%)")
print(f"Cumulative efficiency: eta_cum = {100*eta_cum:.2f}%")
print(f"Wald single-decay limit:       {100*eta_wald:.2f}% (per-event bound)")
print(f"BH rotational energy budget:   {100*eta_rot:.2f}% (thermodynamic bound)")
if eta_wald > 0:
    print(f"Fraction of Wald limit:    {100*eta_cum/eta_wald:.1f}%")
    if eta_cum > eta_wald:
        print(f"  (Note: Cumulative efficiency can exceed Wald's single-decay limit)")

if Delta_E > 0 and E_ex < 0:
    print("\n*** GENUINE PENROSE EXTRACTION (E_ex < 0, DeltaE > 0) ***")
    print("   Exhaust has negative Killing energy (absorbed by BH)")
    print("   This extracts rotational energy from the black hole.")
    print("   The spacecraft's Killing energy increased by utilizing")
    print("   the ergosphere's negative-energy states for the exhaust.")
    # Compute exhaust-rest-mass efficiency for context
    if delta_mu_impulse > 0:
        eta_rest = Delta_E / delta_mu_impulse
        print(f"   Exhaust-mass efficiency: eta_rest = DeltaE/deltamu = {100*eta_rest:.2f}%")
elif Delta_E > 0:
    print("\n[!] Positive energy gain but E_ex > 0 - NOT genuine Penrose mechanism")
    print("   This is kinetic energy transfer, not ergosphere extraction")
else:
    print("\n[X] No net energy gain - parameters need tuning")

# Report trajectory outcome - critical for claiming extraction "worked"
print("\n" + "-"*60)
print("TRAJECTORY OUTCOME")
print("-"*60)
if escaped:
    print(f"  *** SUCCESSFUL ESCAPE TO INFINITY ***")
    print(f"  Final radius: r = {sol2.y[0,-1]:.2f}M")
    print(f"  Energy delivered to infinity: E = {E_final:.4f}")
    if E_ex < 0:
        print(f"  This is genuine Penrose extraction: negative-energy exhaust")
        print(f"  was deposited into the BH while the craft escaped with E > E_0!")
elif captured:
    print(f"  *** CAPTURED BY BLACK HOLE ***")
    print(f"  Spacecraft fell into horizon at r = {sol2.y[0,-1]:.4f}M")
    print(f"  WARNING: Energy gain cannot be claimed since craft didn't escape!")
else:
    print(f"  Trajectory ended at r = {sol2.y[0,-1]:.4f}M (still integrating)")
print("-"*60)

# =============================================================================
# ENERGY BUDGET VALIDATION
# =============================================================================
# For single impulse: straightforward - one deltamu, one E_ex
budget = compute_energy_budget(
    E_initial=E0, E_final=E_final,
    m_initial=1.0, m_final=m_final,
    E_ex_history=[E_ex],
    delta_mu_history=[delta_mu_impulse]
)
print_energy_budget(budget, a)

# =============================================================================
# PRD-STYLE FIGURES - Split into separate 1x2 figures for clarity
# =============================================================================

# -----------------------------------------------------------------------------
# Figure 1: Trajectory and Radial evolution (1x2)
# -----------------------------------------------------------------------------
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
fig1.suptitle('Single-Thrust Penrose Process: Trajectory', fontsize=12, fontweight='bold')

# Fig (a): Trajectory with thrust annotation
ax = ax1
xc = r * np.cos(phi)
yc = r * np.sin(phi)

# Plot pre-burn (infall) in blue, post-burn (escape) in green
ax.plot(xc[:burn_idx+1], yc[:burn_idx+1], color=COLORS['blue'], lw=1.5, 
        label='Infall', zorder=2)
ax.plot(xc[burn_idx:], yc[burn_idx:], color=COLORS['green'], lw=1.5, 
        label='Escape', zorder=2)

# Add horizon and ergosphere
ax.add_patch(plt.Circle((0,0), r_plus, color='black', zorder=5))
ax.add_patch(plt.Circle((0,0), local_ergosphere_radius(), color=COLORS['vermilion'], 
                         ls='--', fill=False, lw=1.5, zorder=1))
# Manual legend entries for circles
ax.plot([], [], 'o', color='black', ms=8, label='Horizon')
ax.plot([], [], '--', color=COLORS['vermilion'], lw=1.5, label='Ergosphere')

# Mark burn point with star
ax.plot(xc[burn_idx], yc[burn_idx], '*', color=COLORS['orange'], ms=16, 
        mec='black', mew=1.0, label='Impulse', zorder=10)

# Legend outside plot
ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0.02, 0.98))
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.set_xlabel(r'$x/M$', fontsize=10)
ax.set_ylabel(r'$y/M$', fontsize=10)
ax.set_title('(a) Equatorial trajectory', fontsize=11)
ax.set_aspect('equal')

# Fig (b): Radius vs proper time
ax = ax2
ax.plot(tau[:burn_idx+1], r[:burn_idx+1], color=COLORS['blue'], lw=1.5, label='Infall')
ax.plot(tau[burn_idx:], r[burn_idx:], color=COLORS['green'], lw=1.5, label='Escape')
ax.axvline(tau[burn_idx], color=COLORS['orange'], ls='-', lw=2.5, alpha=0.8, label='Impulse')
ax.plot(tau[burn_idx], r[burn_idx], '*', color=COLORS['orange'], ms=14, mec='black', zorder=10)

ax.axhline(r_plus, color='black', ls='-', lw=1.0, label=f'Horizon ($r_+$)')
ax.axhline(local_ergosphere_radius(), color=COLORS['vermilion'], ls='--', lw=1.2, label='Ergosphere')
ax.set_xlabel(r'Proper time $\tau/M$', fontsize=10)
ax.set_ylabel(r'Radius $r/M$', fontsize=10)
ax.set_title('(b) Radial evolution', fontsize=11)
ax.legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig('single_thrust_trajectory.pdf', dpi=300, bbox_inches='tight')
plt.savefig('single_thrust_trajectory.png', dpi=150, bbox_inches='tight')

# -----------------------------------------------------------------------------
# Figure 2: Energy and Mass evolution (1x2)
# -----------------------------------------------------------------------------
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 4))
fig2.suptitle('Single-Thrust Penrose Process: Energy Extraction', fontsize=12, fontweight='bold')

# Fig (c): Energy vs proper time
ax = ax3
ax.plot(tau, E_hist, color=COLORS['green'], lw=2, label=f'Energy')
ax.axhline(E0, color=COLORS['gray'], ls=':', lw=1.2, label=f'$E_0$ = {E0:.2f}')
ax.axhline(E_final, color=COLORS['green'], ls='--', lw=1.0, alpha=0.7)
ax.fill_between(tau, E0, E_hist, where=(E_hist > E0), 
                color=COLORS['green'], alpha=0.2)
ax.axvline(tau[burn_idx], color=COLORS['orange'], ls='-', lw=2, alpha=0.5)
ax.set_xlabel(r'Proper time $\tau/M$', fontsize=10)
ax.set_ylabel(r'Energy at infinity $E$', fontsize=10)
ax.set_title(f'(c) Energy gain: DeltaE = {Delta_E:+.3f} ({100*Delta_E/E0:+.1f}%)', fontsize=11)
ax.legend(fontsize=9, loc='upper left')

# Fig (d): Mass vs proper time
ax = ax4
ax.plot(tau, m, color=COLORS['orange'], lw=2, label='Rest mass')
ax.axhline(1.0, color=COLORS['gray'], ls=':', lw=1.2, label=f'$m_0$ = 1.0')
ax.axhline(m_final, color=COLORS['orange'], ls='--', lw=1.0, alpha=0.7)
ax.fill_between(tau, m, 1.0, where=(m < 1.0),
                color=COLORS['orange'], alpha=0.2)
ax.axvline(tau[burn_idx], color=COLORS['orange'], ls='-', lw=2, alpha=0.5)
ax.set_xlabel(r'Proper time $\tau/M$', fontsize=10)
ax.set_ylabel(r'Rest mass $m$', fontsize=10)
ax.set_title(f'(d) Mass expelled: Deltam = {Delta_m:.3f} ({100*Delta_m:.1f}%)', fontsize=11)
ax.legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig('single_thrust_energy.pdf', dpi=300, bbox_inches='tight')
plt.savefig('single_thrust_energy.png', dpi=150, bbox_inches='tight')

print("\nFigures saved: single_thrust_trajectory.pdf/png, single_thrust_energy.pdf/png")
plt.show()

# =============================================================================
# TRIGGER RADIUS SCAN PLOT
# =============================================================================
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(radii_scan, energy_results, 'o-', color=COLORS['blue'], ms=4, label='Final energy')
ax.axhline(E0, color=COLORS['gray'], ls=':', label=f'$E_0$ = {E0}')
ax.axvline(r_plus, color='black', ls='-', lw=0.8, label=f'$r_+$')
ax.axvline(local_ergosphere_radius(), color=COLORS['vermilion'], ls='--', lw=0.8, label='$r_{ergo}$')
ax.axvline(best_r, color=COLORS['orange'], ls=':', lw=1.5, label=f'Optimal $r_{{trig}}$')
ax.set_xlabel(r'Trigger radius $r_{trig}/M$')
ax.set_ylabel(r'Final energy $E_f$')
ax.set_title('Energy vs Trigger Radius')
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig('single_thrust_scan.pdf', dpi=300, bbox_inches='tight')
plt.savefig('single_thrust_scan.png', dpi=150, bbox_inches='tight')
plt.show()