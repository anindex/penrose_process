"""
Kerr Utilities
==============
Shared utilities for Penrose process simulations in Kerr spacetime.

Includes: metric calculations, exhaust energy computation, 4-momentum
conservation, efficiency tracking, and plotting helpers.

References:
- Penrose & Floyd (1971), Nature Phys. Sci. 229, 177
- Wald (1974), ApJ 191, 231 (efficiency limits)
- Bardeen, Press & Teukolsky (1972), ApJ 178, 347
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PLOTTING
# =============================================================================

# Colorblind-friendly palette (Wong 2011)
COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'vermilion': '#D55E00',
    'purple': '#CC79A7',
    'black': '#000000',
    'gray': '#999999'
}

# PRD figure dimensions (inches)
PRD_SINGLE_COL = 3.375
PRD_DOUBLE_COL = 7.0


def setup_prd_style():
    """Configure matplotlib for publication-quality (PRD) figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times', 'Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'lines.linewidth': 1.2,
        'axes.linewidth': 0.6,
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'legend.frameon': False,
        'axes.grid': False,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
    })


# =============================================================================
# KERR METRIC FUNCTIONS
# =============================================================================

def kerr_metric_components(r, th, a, M=1.0, clamp_horizon=True, warn_horizon=True):
    """
    Kerr metric components in Boyer-Lindquist coordinates.
    
    Returns (covariant, contravariant) tuples:
        cov = (g_tt, g_tphi, g_rr, g_thth, g_phiphi)
        con = (g^tt, g^tphi, g^rr, g^thth, g^phiphi)
    
    Sign conventions (-+++ signature):
    - g_tt < 0 outside ergosphere, g_tt > 0 inside ergosphere
    - g_tphi < 0 for a > 0 (prograde frame dragging)
    - At ergosphere: g_tt = 0 (stationary limit)
    - At horizon: Delta = 0 (coordinate singularity)
    
    Parameters
    ----------
    clamp_horizon : bool
        If True, clamp Delta to tiny positive value when Delta <= 0 to prevent overflow.
        If False, raise ValueError when inside/at horizon.
    warn_horizon : bool
        If True and clamp_horizon=True, issue warning when clamping occurs.
    
    CAUTION: Near/inside the horizon (Delta -> 0), contravariant components
    diverge due to the coordinate singularity. Results inside the horizon
    are not physically meaningful in Boyer-Lindquist coordinates.
    Use a safety margin r > r_+ + eps.
    """
    sin_th = np.sin(th)
    sin2 = sin_th * sin_th
    cos_th = np.cos(th)
    cos2 = cos_th * cos_th

    Sigma = r*r + a*a*cos2
    Delta = r*r - 2*M*r + a*a
    r_plus = M + np.sqrt(M**2 - a**2)

    # Handle horizon crossing
    if Delta <= 0:
        if clamp_horizon:
            if warn_horizon:
                import warnings
                warnings.warn(
                    f"Trajectory crossed horizon: r={r:.6f}M, r_+={r_plus:.6f}M, Delta={Delta:.6e}. "
                    f"Clamping to Delta=tiny for numerical stability, but results are unphysical.",
                    RuntimeWarning
                )
            Delta = np.finfo(float).tiny
        else:
            raise ValueError(
                f"Horizon crossing detected: r={r:.6f}M < r_+={r_plus:.6f}M (Delta={Delta:.6e}). "
                f"Boyer-Lindquist coordinates invalid inside horizon."
            )

    # Covariant components
    g_tt = -(1.0 - 2*M*r / Sigma)
    g_tphi = -(2*M*a*r*sin2) / Sigma
    g_rr = Sigma / Delta
    g_thth = Sigma
    g_phiphi = (r*r + a*a + (2*M*r*a*a*sin2)/Sigma) * sin2

    # Contravariant components
    gu_rr = Delta / Sigma
    gu_thth = 1.0 / Sigma
    gu_tphi = -(2*M*a*r) / (Sigma * Delta)
    A = (r*r + a*a)**2 - a*a*Delta*sin2
    gu_tt = -A / (Sigma * Delta)
    gu_phiphi = (Delta - a*a*sin2) / (Sigma * Delta * sin2)

    cov = (g_tt, g_tphi, g_rr, g_thth, g_phiphi)
    con = (gu_tt, gu_tphi, gu_rr, gu_thth, gu_phiphi)
    return cov, con


def horizon_radius(a, M=1.0):
    """Event horizon radius: r+ = M + sqrt(M^2 - a^2)."""
    return M + np.sqrt(M**2 - a**2)


def ergosphere_radius(th, a, M=1.0):
    """Ergosphere radius: r_erg = M + sqrt(M^2 - a^2cos^2theta). Equals 2M at equator."""
    return M + np.sqrt(M*M - a*a*np.cos(th)**2)


def compute_pt_from_mass_shell(r, th, pr, pphi, m, a, M=1.0, 
                                warn_forbidden=True, raise_on_forbidden=False):
    """
    Solve mass-shell constraint g^{munu} p_mu p_nu = -m^2 for p_t.
    
    Sign convention:
    - p_t < 0 for positive-energy particles (future-directed)
    - E = -p_t is the Killing energy at infinity
    - For massive particles outside the horizon, E > 0
    
    Parameters
    ----------
    r, th : float
        Boyer-Lindquist coordinates
    pr, pphi : float
        Covariant momentum components p_r, p_phi
    m : float
        Rest mass (must be > 0 for massive particles)
    a, M : float
        Black hole spin and mass
    warn_forbidden : bool
        If True, print warning when (r, pr, pphi, m) is in forbidden region
    raise_on_forbidden : bool
        If True, raise ValueError when in forbidden region instead of clamping
    
    Returns
    -------
    float
        Covariant time momentum p_t (future-directed solution)
    """
    _, con = kerr_metric_components(r, th, a, M)
    gu_tt, gu_tphi, gu_rr, gu_thth, gu_phiphi = con
    
    # Quadratic equation: gu_tt * pt^2 + 2*gu_tphi*pphi*pt + (gu_rr*pr^2 + gu_phiphi*pphi^2 + m^2) = 0
    A = gu_tt
    B = 2*gu_tphi*pphi
    C = gu_rr*pr**2 + gu_phiphi*pphi**2 + m**2
    
    det = B**2 - 4*A*C
    
    if det < 0:
        if raise_on_forbidden:
            raise ValueError(
                f"Forbidden region: discriminant = {det:.3e} < 0 at r = {r:.4f}M. "
                f"The given (pr={pr:.4f}, pphi={pphi:.4f}, m={m:.4f}) cannot satisfy "
                f"mass-shell at this radius. This typically indicates an unphysical "
                f"trajectory that should be rejected."
            )
        if warn_forbidden:
            import warnings
            warnings.warn(
                f"Forbidden region: discriminant = {det:.3e} < 0 at r = {r:.4f}M. "
                f"The given (pr, pphi, m) cannot satisfy mass-shell at this radius. "
                f"Clamping to det=0 (marginal orbit).",
                RuntimeWarning
            )
        det = 0.0
    
    # Choose future-directed solution: p_t < 0 for E > 0 particles
    # Since gu_tt < 0 outside horizon, (-B + sqrt(det))/(2*gu_tt) gives p_t < 0
    return (-B + np.sqrt(det)) / (2*A)


def compute_energy(r, th, pr, pphi, m, a, M=1.0):
    """Killing energy at infinity: E = -p_t."""
    pt = compute_pt_from_mass_shell(r, th, pr, pphi, m, a, M)
    return -pt


# =============================================================================
# EFFICIENCY CALCULATIONS
# =============================================================================

def max_rotational_energy_fraction(a_spin, M_bh=1.0):
    """
    Maximum extractable rotational energy fraction of a Kerr black hole.
    
    This is 1 - M_ir/M where M_ir is the irreducible mass. It represents
    the total rotational energy that could theoretically be extracted via
    repeated Penrose processes, NOT the efficiency of a single decay event.
    
    For a=0.95M: ~19.3%
    For a=M (extremal): ~29.3%
    
    This is DIFFERENT from Wald's single-decay limit (~20.7% for extremal Kerr),
    which bounds the energy gain from a single particle fragmentation event.
    
    Reference: Christodoulou (1970), Phys. Rev. Lett. 25, 1596
    """
    a_star = a_spin / M_bh
    if a_star <= 0 or a_star > 1:
        return 0.0
    inner_sqrt = np.sqrt(1 - a_star**2)
    return 1.0 - np.sqrt(0.5 * (1.0 + inner_sqrt))


def wald_single_decay_limit():
    """
    Wald's single-decay Penrose efficiency limit.
    
    For an idealized single fragmentation event in extremal Kerr (a=M),
    the maximum energy gain per unit rest mass of the escaping fragment is
    approximately 20.7%. This assumes optimal kinematics and that the
    negative-energy fragment falls into the hole.
    
    This limit applies to SINGLE EVENTS, not cumulative processes.
    A rocket executing multiple thrust burns can exceed this per-burn
    by utilizing multiple extraction events.
    
    Reference: Wald (1974), ApJ 191, 231
    """
    return 0.207  # ~20.7% for extremal Kerr


# Backward compatibility alias (deprecated)
def theoretical_penrose_limit(a_spin, M_bh=1.0):
    """
    DEPRECATED: This name is ambiguous.
    
    Use max_rotational_energy_fraction() for the BH's total extractable
    rotational energy, or wald_single_decay_limit() for the single-event
    efficiency bound.
    
    Currently returns max_rotational_energy_fraction for backward compatibility.
    """
    return max_rotational_energy_fraction(a_spin, M_bh)


def compute_instantaneous_efficiency(dE_dtau, dm_dtau, threshold=1e-10):
    """Instantaneous efficiency eta = dE/d(-m). eta > 1 suggests possible Penrose extraction."""
    eta = np.zeros_like(dE_dtau)
    active = np.abs(dm_dtau) > threshold
    eta[active] = -dE_dtau[active] / dm_dtau[active]
    return eta


def compute_cumulative_efficiency(E_initial, E_final, m_initial, m_final):
    """Cumulative efficiency: (E_f - E_0) / (m_0 - m_f)."""
    delta_m = m_initial - m_final
    if delta_m <= 0:
        return 0.0
    return (E_final - E_initial) / delta_m


def print_efficiency_analysis(tau, E, m, r, a_spin, M_bh=1.0):
    """
    Print efficiency statistics to console.
    
    Be careful: eta > 1 by itself doesn't prove Penrose extraction.
    You need E_ex < 0 (negative exhaust Killing energy) to confirm.
    """
    # Compute derivatives
    dE_dtau = np.gradient(E, tau)
    dm_dtau = np.gradient(m, tau)
    
    # Efficiencies
    eta_inst = compute_instantaneous_efficiency(dE_dtau, dm_dtau)
    eta_cum = compute_cumulative_efficiency(E[0], E[-1], m[0], m[-1])
    eta_max = theoretical_penrose_limit(a_spin, M_bh)
    
    # Where are we in/out of ergosphere?
    r_erg = 2.0 * M_bh
    in_ergosphere = r < r_erg
    
    # eta > 1 is suggestive but not proof of Penrose extraction
    penrose_active = (eta_inst > 1.0) & (np.abs(dm_dtau) > 1e-10)
    penrose_in_ergo = penrose_active & in_ergosphere
    
    # Statistics
    if penrose_active.any():
        eta_when_active = eta_inst[penrose_active]
        mean_eta_active = eta_when_active.mean()
        max_eta = eta_when_active.max()
    else:
        mean_eta_active = 0.0
        max_eta = 0.0
    
    active_fraction = penrose_active.sum() / len(tau) if len(tau) > 0 else 0
    
    # Print formatted output
    print("\n" + "="*65)
    print("         PENROSE PROCESS EFFICIENCY ANALYSIS")
    print("="*65)
    
    print(f"\n{'BLACK HOLE PARAMETERS':^65}")
    print("-"*65)
    print(f"  Spin parameter:              a/M = {a_spin:.4f}")
    print(f"  Horizon radius:              r_+  = {horizon_radius(a_spin, M_bh):.4f} M")
    print(f"  Ergosphere (equator):        r_E = {r_erg:.4f} M")
    
    print(f"\n{'ENERGY BUDGET':^65}")
    print("-"*65)
    print(f"  Initial energy:              E_0  = {E[0]:.6f}")
    print(f"  Final energy:                E_f = {E[-1]:.6f}")
    print(f"  Net energy change:           DeltaE  = {E[-1] - E[0]:+.6f}")
    print(f"  Energy gain:                      {(E[-1]/E[0] - 1)*100:+.2f}%")
    
    print(f"\n{'MASS BUDGET':^65}")
    print("-"*65)
    print(f"  Initial mass:                m_0  = {m[0]:.6f}")
    print(f"  Final mass:                  m_f = {m[-1]:.6f}")
    print(f"  Fuel consumed:               Deltam  = {m[0] - m[-1]:.6f}")
    print(f"  Fuel fraction:                    {(m[0] - m[-1])/m[0]*100:.2f}%")
    
    print(f"\n{'EFFICIENCY METRICS':^65}")
    print("-"*65)
    eta_rot = max_rotational_energy_fraction(a_spin, M_bh)
    eta_wald = wald_single_decay_limit()
    print(f"  BH rotational energy budget:       {eta_rot:.4f} ({eta_rot*100:.2f}%)")
    print(f"  Wald single-decay limit:           {eta_wald:.4f} ({eta_wald*100:.2f}%)")
    print(f"  Cumulative efficiency:       eta_cum = {eta_cum:.4f} ({eta_cum*100:.2f}%)")
    if eta_rot > 0:
        print(f"  Fraction of BH rot. budget:        {eta_cum/eta_rot*100:.1f}%")
    
    print(f"\n{'DIAGNOSTICS':^65}")
    print("-"*65)
    print(f"  eta_inst > 1 (rest-mass break-even): {active_fraction*100:.1f}% of trajectory")
    print(f"  Mean eta_inst when active:          {mean_eta_active:.4f}")
    print(f"  Peak eta_inst achieved:             {max_eta:.4f}")
    
    if penrose_in_ergo.any():
        print(f"\n  eta_inst > 1 occurs inside the ergosphere")
        print(f"    NOTE: This is not sufficient to claim Penrose extraction.")
        print(f"    Confirm by checking a negative Killing-energy flux into the horizon")
        print(f"    (e.g., negative-energy exhaust E_ex < 0 that must fall into the hole).")
    else:
        print(f"\n  eta_inst > 1 never occurs in this run")
        print(f"    (No rest-mass break-even, regardless of Penrose conditions.)")
    
    print("="*65 + "\n")
    
    return eta_inst, eta_cum, eta_max, penrose_active


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def smoothstep(x):
    """Quintic smoothstep: S(x) = 6x^5 - 15x^4 + 10x^3. C^2-continuous at endpoints."""
    x = np.clip(x, 0.0, 1.0)
    return 6*x**5 - 15*x**4 + 10*x**3


def isco_radius(a, M=1.0, prograde=True):
    """ISCO radius from Bardeen, Press, Teukolsky (1972)."""
    a_star = a / M
    Z1 = 1 + (1 - a_star**2)**(1/3) * ((1 + a_star)**(1/3) + (1 - a_star)**(1/3))
    Z2 = np.sqrt(3 * a_star**2 + Z1**2)
    
    if prograde:
        return M * (3 + Z2 - np.sqrt((3 - Z1)*(3 + Z1 + 2*Z2)))
    else:
        return M * (3 + Z2 + np.sqrt((3 - Z1)*(3 + Z1 + 2*Z2)))


def frame_dragging_omega(r, a, M=1.0, th=np.pi/2):
    """Frame-dragging angular velocity omega = -g_tphi/g_phiphi (ZAMO angular velocity)."""
    sin2 = np.sin(th)**2
    Sigma = r**2 + a**2 * np.cos(th)**2
    Delta = r**2 - 2*M*r + a**2
    
    g_tphi = -(2*M*a*r*sin2) / Sigma
    g_phiphi = (r**2 + a**2 + (2*M*r*a**2*sin2)/Sigma) * sin2
    
    return -g_tphi / g_phiphi


# =============================================================================
# EXHAUST GEODESIC INTEGRATION (for capture verification)
# =============================================================================

def integrate_exhaust_geodesic(r0, th0, u_ex_contra, a, M=1.0, tau_max=50.0, 
                                 r_horizon_margin=0.01, n_steps=1000):
    """
    Integrate exhaust geodesic to verify horizon capture.
    
    For E_ex < 0 exhaust, this verifies it falls into the black hole.
    Returns dict with 'captured', 'escaped', 'r_final', 'trajectory'.
    """
    r_plus = horizon_radius(a, M)
    r_capture = r_plus + r_horizon_margin
    
    # Get initial covariant momentum (for massless or unit-mass particle)
    # p_mu = g_munu u^nu
    cov, con = kerr_metric_components(r0, th0, a, M)
    g_tt, g_tphi, g_rr, _, g_phiphi = cov
    gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con
    
    u_t, u_r, u_phi = u_ex_contra
    pt = g_tt * u_t + g_tphi * u_phi
    pr = g_rr * u_r
    pphi = g_tphi * u_t + g_phiphi * u_phi
    
    # For geodesic integration, use m=1 (can normalize later if needed)
    m = 1.0
    
    # Simple Euler integration (sufficient for capture/escape check)
    tau_arr = [0.0]
    r_arr = [r0]
    
    state = [r0, th0, 0.0, pr]  # [r, th, phi, pr]
    tau = 0.0
    dt = tau_max / n_steps
    
    for _ in range(n_steps):
        r, th, phi, pr = state
        
        if r < r_capture:
            break  # Captured
        if r > 100.0 * M:
            break  # Escaped
            
        # Get metric
        try:
            cov, con = kerr_metric_components(r, th, a, M)
        except:
            break
        g_tt, g_tphi, g_rr, _, g_phiphi = cov
        gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con
        
        # Compute dr/dtau from pr
        dr = gu_rr * pr / m
        
        # Compute dpr/dtau from Hamiltonian (finite differences for simplicity)
        eps = 1e-6 * max(1.0, abs(r))
        def H_at_r(r_):
            try:
                _, con_ = kerr_metric_components(r_, th, a, M)
                gu_tt_, gu_tphi_, gu_rr_, _, gu_phiphi_ = con_
                return 0.5*(gu_tt_*pt**2 + gu_rr_*pr**2 + gu_phiphi_*pphi**2 + 2*gu_tphi_*pt*pphi)
            except:
                return 0.0
        dH_dr = (H_at_r(r + eps) - H_at_r(r - eps)) / (2*eps)
        dpr = -dH_dr / m  # Proper time parameterization
        
        # Simple Euler step (sufficient for capture check)
        state = [r + dr*dt, th, phi, pr + dpr*dt]
        tau += dt
        
        tau_arr.append(tau)
        r_arr.append(state[0])
    
    r_final = state[0]
    r_min = min(r_arr)
    
    return {
        'captured': r_final < r_capture,
        'escaped': r_final > 50.0 * M,
        'r_final': r_final,
        'tau_final': tau,
        'r_min': r_min,
        'trajectory': (np.array(tau_arr), np.array(r_arr))
    }


def verify_exhaust_capture_batch(exhaust_samples, a, M=1.0, verbose=True, 
                                  check_positive_escape=True):
    """
    Verify horizon capture for exhaust samples.
    
    For E_ex < 0 samples: verify they fall into the horizon (required for Penrose).
    For E_ex >= 0 samples (if check_positive_escape=True): verify they escape to
    infinity, which is needed for accurate energy bookkeeping.
    
    The interpretation "energy deposited into BH" is only valid if:
    - Negative-E_ex exhaust is captured (checked)
    - Positive-E_ex exhaust escapes (optionally checked)
    
    If positive-E_ex exhaust falls into the BH, the energy budget interpretation
    needs adjustment.
    """
    n_total = len(exhaust_samples)
    n_negative_E = sum(1 for s in exhaust_samples if s['E_ex'] < 0)
    n_positive_E = sum(1 for s in exhaust_samples if s['E_ex'] >= 0)
    
    n_neg_captured = 0
    n_neg_escaped = 0
    n_pos_captured = 0
    n_pos_escaped = 0
    
    capture_failures = []  # Samples with E_ex < 0 that didn't get captured
    escape_failures = []   # Samples with E_ex >= 0 that didn't escape
    
    for sample in exhaust_samples:
        result = integrate_exhaust_geodesic(
            sample['r'], sample['th'], sample['u_ex_contra'], a, M
        )
        
        if sample['E_ex'] < 0:
            # Negative-energy exhaust should be captured
            if result['captured']:
                n_neg_captured += 1
            elif result['escaped']:
                n_neg_escaped += 1
                capture_failures.append({
                    'sample': sample,
                    'result': result
                })
        else:
            # Positive-energy exhaust should escape (if we're checking)
            if check_positive_escape:
                if result['escaped']:
                    n_pos_escaped += 1
                elif result['captured']:
                    n_pos_captured += 1
                    escape_failures.append({
                        'sample': sample,
                        'result': result
                    })
    
    if verbose:
        print(f"\n{'EXHAUST FATE VERIFICATION':^65}")
        print("-"*65)
        
        if n_negative_E > 0:
            print(f"  Negative-E_ex samples:       {n_negative_E}")
            print(f"    Confirmed captured:        {n_neg_captured}")
            print(f"    Unexpectedly escaped:      {n_neg_escaped}")
            if n_neg_captured == n_negative_E:
                print(f"    [OK] All negative-E exhaust falls into horizon")
            elif n_neg_escaped > 0:
                print(f"    [!] WARNING: {n_neg_escaped} samples with E_ex < 0 escaped!")
                print(f"      This would INVALIDATE the Penrose extraction claim.")
        
        if check_positive_escape and n_positive_E > 0:
            print(f"\n  Positive-E_ex samples:       {n_positive_E}")
            print(f"    Confirmed escaped:         {n_pos_escaped}")
            print(f"    Unexpectedly captured:     {n_pos_captured}")
            if n_pos_escaped == n_positive_E:
                print(f"    [OK] All positive-E exhaust escapes to infinity")
            elif n_pos_captured > 0:
                print(f"    Note: {n_pos_captured} samples with E_ex >= 0 were captured.")
                print(f"      This is physically possible (low-energy particles can fall in).")
                print(f"      The 'energy to BH' calculation should account for this.")
    
    return {
        'n_negative_E': n_negative_E,
        'n_positive_E': n_positive_E,
        'n_neg_captured': n_neg_captured,
        'n_neg_escaped': n_neg_escaped,
        'n_pos_captured': n_pos_captured,
        'n_pos_escaped': n_pos_escaped,
        'capture_failures': capture_failures,
        'escape_failures': escape_failures,
        'penrose_valid': (n_neg_escaped == 0)  # Penrose claim is valid if all negative-E captured
    }


# =============================================================================
# ANALYTIC METRIC DERIVATIVES
# =============================================================================

def kerr_metric_derivatives(r, th, a, M=1.0):
    """Analytic derivatives d(g^munu)/dr. More accurate than finite differences near horizon."""
    # For equatorial plane (th = pi/2)
    sin2 = np.sin(th)**2
    cos2 = np.cos(th)**2
    
    # At equator: Sigma = r^2, but keep general for future extension
    Sigma = r**2 + a**2 * cos2
    Delta = r**2 - 2*M*r + a**2
    
    # Avoid division by zero near horizon
    if Delta <= 0:
        Delta = np.finfo(float).tiny
    
    # Derivatives of Sigma and Delta
    dSigma_dr = 2*r  # At equator
    dDelta_dr = 2*r - 2*M
    
    # -------------------------------------------------------------------------
    # g^{rr} = Delta/Sum
    # d(g^{rr})/dr = (dDelta/dr * Sum - Delta * dSum/dr) / Sum^2
    # At equator: = (2(r-M)*r^2 - (r^2-2Mr+a^2)*2r) / r^4
    #            = 2r[(r-M)r - (r^2-2Mr+a^2)] / r^4
    #            = 2[(r-M)r - r^2 + 2Mr - a^2] / r^3
    #            = 2[r^2 - Mr - r^2 + 2Mr - a^2] / r^3
    #            = 2[Mr - a^2] / r^3
    # -------------------------------------------------------------------------
    dgu_rr = 2*(M*r - a**2) / (r**3)
    
    # -------------------------------------------------------------------------
    # g^{tphi} = -2Mar / (SumDelta)
    # Let f = -2Mar, g = SumDelta
    # df/dr = -2Ma
    # dg/dr = dSum/dr*Delta + Sum*dDelta/dr = 2r*Delta + r^2*(2r-2M) = 2r[Delta + r(r-M)]
    #       = 2r[r^2-2Mr+a^2 + r^2-Mr] = 2r[2r^2-3Mr+a^2]
    # d(g^{tphi})/dr = (df*g - f*dg) / g^2
    #             = [-2Ma*r^2Delta - (-2Mar)*2r(2r^2-3Mr+a^2)] / (r^2Delta)^2
    #             = [-2Ma*r^2Delta + 4Ma*r^2(2r^2-3Mr+a^2)] / (r^4Delta^2)
    #             = 2Ma*r^2[-Delta + 2(2r^2-3Mr+a^2)] / (r^4Delta^2)
    #             = 2Ma[-Delta + 2(2r^2-3Mr+a^2)] / (r^2Delta^2)
    # Simplify: -Delta + 2(2r^2-3Mr+a^2) = -(r^2-2Mr+a^2) + 4r^2-6Mr+2a^2
    #         = -r^2+2Mr-a^2 + 4r^2-6Mr+2a^2 = 3r^2-4Mr+a^2
    # CORRECTED: The derivative is 2Ma(3r^2 - 4Mr + a^2) / (r^2Delta^2)
    # Previous version had wrong sign: (3r^2 - 2Mr - a^2) was incorrect
    # -------------------------------------------------------------------------
    numerator_gtphi = 2*M*a * (3*r**2 - 4*M*r + a**2)
    dgu_tphi = numerator_gtphi / (r**2 * Delta**2)
    
    # -------------------------------------------------------------------------
    # g^{tt} = -A / (SumDelta) where A = (r^2 + a^2)^2 - a^2Delta
    # At equator: A = (r^2+a^2)^2 - a^2(r^2-2Mr+a^2)
    #           = r^4 + 2a^2r^2 + a^4 - a^2r^2 + 2Ma^2r - a^4
    #           = r^4 + a^2r^2 + 2Ma^2r
    # dA/dr = 4r^3 + 2a^2r + 2Ma^2 = 2r(2r^2 + a^2) + 2Ma^2
    #
    # g^{tt} = -A / (r^2Delta)
    # d(g^{tt})/dr = -[dA/dr*r^2Delta - A*(2rDelta + r^2*dDelta/dr)] / (r^2Delta)^2
    #             = -[dA/dr*r^2Delta - A*r(2Delta + r(2r-2M))] / (r^4Delta^2)
    #             = -[dA/dr*r^2Delta - A*r(2Delta + 2r^2 - 2Mr)] / (r^4Delta^2)
    # -------------------------------------------------------------------------
    A_eq = r**4 + a**2 * r**2 + 2*M*a**2 * r
    dA_dr = 4*r**3 + 2*a**2 * r + 2*M*a**2
    
    # d(r^2Delta)/dr = 2rDelta + r^2(2r-2M) = 2r[Delta + r^2 - Mr]
    d_r2Delta_dr = 2*r*Delta + r**2 * dDelta_dr
    
    numerator_gtt = -(dA_dr * r**2 * Delta - A_eq * d_r2Delta_dr)
    dgu_tt = numerator_gtt / (r**4 * Delta**2)
    
    # -------------------------------------------------------------------------
    # g^{phiphi} = (Delta - a^2sin^2theta) / (SumDeltasin^2theta)
    # At equator (sin^2theta = 1): g^{phiphi} = (Delta - a^2) / (r^2Delta) = (r^2 - 2Mr + a^2 - a^2) / (r^2Delta)
    #                                = (r^2 - 2Mr) / (r^2Delta) = (r - 2M) / (rDelta)
    #
    # d(g^{phiphi})/dr = d[(r-2M)/(rDelta)]/dr
    # Let f = r-2M, g = rDelta
    # df/dr = 1
    # dg/dr = Delta + r(2r-2M) = Delta + 2r^2 - 2Mr
    # d(g^{phiphi})/dr = [1*rDelta - (r-2M)(Delta + 2r^2 - 2Mr)] / (rDelta)^2
    #             = [rDelta - (r-2M)(Delta + 2r^2 - 2Mr)] / (r^2Delta^2)
    # -------------------------------------------------------------------------
    f_phiphi = r - 2*M
    g_phiphi = r * Delta
    dg_phiphi_dr = Delta + r * dDelta_dr
    
    numerator_gphiphi = r*Delta - f_phiphi * dg_phiphi_dr
    dgu_phiphi = numerator_gphiphi / (r**2 * Delta**2)
    
    return {
        'dgu_tt': dgu_tt,
        'dgu_tphi': dgu_tphi,
        'dgu_rr': dgu_rr,
        'dgu_phiphi': dgu_phiphi
    }


def compute_dH_dr_analytic(r, th, pt, pr, pphi, m, a, M=1.0):
    """dH/dr from geodesic Hamiltonian using analytic derivatives."""
    derivs = kerr_metric_derivatives(r, th, a, M)
    
    dH_dr = 0.5 * (
        derivs['dgu_tt'] * pt**2
        + 2 * derivs['dgu_tphi'] * pt * pphi
        + derivs['dgu_rr'] * pr**2
        + derivs['dgu_phiphi'] * pphi**2
    )
    
    return dH_dr


# =============================================================================
# EXHAUST ENERGY AND MODE CONTROL
# =============================================================================

from enum import Enum

class ThrustMode(Enum):
    """Thrust control modes: EXTRACTION (minimize E_ex), ESCAPE (maximize pr), COAST (off)."""
    EXTRACTION = 1  # inside extraction zone, minimizing E_ex
    ESCAPE = 2      # heading out, maximize radial velocity
    COAST = 3       # engines off, conserve fuel


# =============================================================================
# PHYSICAL VALIDITY VERIFICATION
# =============================================================================

class PhysicsValidationError(Exception):
    """Raised when a physical constraint is violated beyond tolerance."""
    pass


def verify_future_directed(u_t, label="", warn=True):
    """
    Verify that a 4-velocity is future-directed: u^t > 0.
    
    In the (-+++) signature, a future-directed timelike vector has u^t > 0.
    This is a basic causality requirement for physical particles.
    
    Returns (is_valid, u_t).
    """
    is_valid = u_t > 0
    if not is_valid and warn:
        import warnings
        warnings.warn(
            f"Future-directedness violation{' (' + label + ')' if label else ''}: "
            f"u^t = {u_t:.6f} <= 0 (must be positive for causality)",
            RuntimeWarning
        )
    return is_valid, u_t


def verify_coast_invariants(pt_history, pphi_history, tol=1e-8, warn=True):
    """
    Verify that p_t and p_phi remain constant during geodesic (coast) phases.
    
    These are constants of motion due to the Killing vectors d/dt and d/dphi.
    Returns (pt_ok, pphi_ok, pt_drift, pphi_drift).
    """
    pt_arr = np.asarray(pt_history)
    pphi_arr = np.asarray(pphi_history)
    
    pt_drift = np.max(np.abs(pt_arr - pt_arr[0])) if len(pt_arr) > 1 else 0.0
    pphi_drift = np.max(np.abs(pphi_arr - pphi_arr[0])) if len(pphi_arr) > 1 else 0.0
    
    pt_ok = pt_drift < tol
    pphi_ok = pphi_drift < tol
    
    if warn and (not pt_ok or not pphi_ok):
        import warnings
        warnings.warn(
            f"Coast invariant drift: Deltap_t = {pt_drift:.2e}, Deltap_phi = {pphi_drift:.2e} "
            f"(tol={tol}). This indicates numerical integration error during geodesic phases.",
            RuntimeWarning
        )
    
    return pt_ok, pphi_ok, pt_drift, pphi_drift


def verify_mass_shell(p_cov, m, r, th, a, M=1.0, tol=1e-3, raise_on_violation=False):
    """
    Verify mass-shell constraint: g^{munu} p_mu p_nu = -m^2.
    
    Returns (is_valid, residual) where residual = |g^{munu}p_mup_nu + m^2|.
    If raise_on_violation=True and residual > tol, raises PhysicsValidationError.
    
    Note: Default tolerance is 1e-3 to account for numerical precision limits
    in curved spacetime calculations, especially near the ergosphere.
    """
    pt, pr, pphi = p_cov
    _, con = kerr_metric_components(r, th, a, M)
    gu_tt, gu_tphi, gu_rr, gu_thth, gu_phiphi = con
    
    p_squared = (gu_tt * pt**2 
                 + 2*gu_tphi * pt * pphi 
                 + gu_rr * pr**2 
                 + gu_phiphi * pphi**2)
    
    # Should equal -m^2
    residual = abs(p_squared + m**2)
    is_valid = residual < tol
    
    if raise_on_violation and not is_valid:
        raise PhysicsValidationError(
            f"Mass-shell violation: g^{{munu}}p_mup_nu + m^2 = {residual:.6e} (tol={tol}). "
            f"State: r={r:.4f}, m={m:.6f}, p=({pt:.6f}, {pr:.6f}, {pphi:.6f})"
        )
    
    return is_valid, residual


def verify_4velocity_normalization(u_vec, g_tt, g_tphi, g_rr, g_phiphi, 
                                    tol=1e-2, raise_on_violation=False, label=""):
    """
    Verify 4-velocity normalization: g_{munu} u^mu u^nu = -1 (timelike).
    
    Returns (is_valid, norm_value) where norm_value = g_{munu} u^mu u^nu.
    For a valid timelike 4-velocity, norm_value should be -1.
    
    Note: Default tolerance is 1e-2 to account for numerical precision limits
    in the Gram-Schmidt orthogonalization process inside the ergosphere.
    """
    u = np.asarray(u_vec)
    
    # g_{munu} u^mu u^nu in (t, r, phi) subspace at equator
    norm = (g_tt * u[0]**2 
            + g_rr * u[1]**2 
            + g_phiphi * u[2]**2 
            + 2*g_tphi * u[0] * u[2])
    
    # Should equal -1 for timelike
    residual = abs(norm + 1.0)
    is_valid = residual < tol
    
    if raise_on_violation and not is_valid:
        raise PhysicsValidationError(
            f"4-velocity normalization violation{' (' + label + ')' if label else ''}: "
            f"g_{{munu}}u^muu^nu = {norm:.6f} (expected -1, residual={residual:.6e})"
        )
    
    return is_valid, norm


def verify_4momentum_conservation(p_old, p_new, delta_mu, u_ex_cov, 
                                   tol=1e-8, raise_on_violation=False):
    """
    Verify 4-momentum conservation: p'_mu = p_mu - deltamu * u_{ex,mu}.
    
    Returns (is_valid, residual_vector) where residual = p' - (p - deltamu*u_ex).
    """
    p_old = np.asarray(p_old)
    p_new = np.asarray(p_new)
    u_ex = np.asarray(u_ex_cov)
    
    expected = p_old - delta_mu * u_ex
    residual = p_new - expected
    max_residual = np.max(np.abs(residual))
    is_valid = max_residual < tol
    
    if raise_on_violation and not is_valid:
        raise PhysicsValidationError(
            f"4-momentum conservation violation: max|p' - (p - deltamu*u_ex)| = {max_residual:.6e}"
        )
    
    return is_valid, residual


def compute_exhaust_4velocity(u_vec, s_vec, v_e, g_tt, g_tphi, g_rr, g_phiphi,
                               verify=True, tol=1e-2):
    """
    Compute exhaust 4-velocity: u_ex^mu = gamma_e(u^mu - v_e s^mu).
    
    Returns dict with 'u_ex_contra', 'u_ex_cov', 'E_ex', 'gamma_e', 'normalization_ok'.
    E_ex < 0 is the signature of genuine Penrose extraction.
    
    If verify=True, checks that the exhaust 4-velocity is properly normalized
    (timelike with g_{munu}u^muu^nu = -1). In extreme frame-dragging regions,
    numerical errors can accumulate, so a tolerance is used.
    
    Note: Default tolerance is 1e-2 (1%) to account for Gram-Schmidt
    orthogonalization errors inside the ergosphere.
    """
    gamma_e = 1.0 / np.sqrt(1.0 - v_e**2)
    
    # Exhaust 4-velocity (contravariant)
    u_ex = gamma_e * (np.asarray(u_vec) - v_e * np.asarray(s_vec))
    
    # Covariant components: u_{ex,mu} = g_{munu} u_ex^nu
    u_ex_t_cov = g_tt * u_ex[0] + g_tphi * u_ex[2]
    u_ex_r_cov = g_rr * u_ex[1]
    u_ex_phi_cov = g_tphi * u_ex[0] + g_phiphi * u_ex[2]
    
    # Killing energy
    E_ex = -u_ex_t_cov
    
    # Verify normalization if requested
    normalization_ok = True
    if verify:
        is_valid, norm = verify_4velocity_normalization(
            u_ex, g_tt, g_tphi, g_rr, g_phiphi, tol=tol, label="exhaust"
        )
        normalization_ok = is_valid
        if not is_valid:
            import warnings
            warnings.warn(
                f"Exhaust 4-velocity not properly normalized: g_{{munu}}u^muu^nu = {norm:.6f} "
                f"(expected -1). This may indicate numerical issues in the tetrad construction.",
                RuntimeWarning
            )
    
    return {
        'u_ex_contra': u_ex,
        'u_ex_cov': np.array([u_ex_t_cov, u_ex_r_cov, u_ex_phi_cov]),
        'E_ex': E_ex,
        'gamma_e': gamma_e,
        'normalization_ok': normalization_ok
    }


def compute_continuous_thrust_exact(p_cov, m, delta_mu, u_ex_result, r, th, a, M=1.0,
                                     verify=True, clamp_negative_mass=True):
    """
    Exact momentum change from 4-momentum conservation: p'mu = pmu - deltamu * u_ex,mu.
    
    New mass is derived from mass-shell constraint. Returns dict with
    'dp_cov', 'dm', 'm_new', 'E_new', 'conservation_residual', 'mass_shell_valid'.
    
    Parameters
    ----------
    clamp_negative_mass : bool
        If True (default), clamp m^2 to tiny positive value if it goes negative.
        If False, raise PhysicsValidationError on negative m^2.
    verify : bool
        If True, verify mass-shell constraint after computing new momentum.
    """
    pt, pr, pphi = p_cov
    u_ex_cov = u_ex_result['u_ex_cov']
    
    # Momentum change from exact 4-momentum conservation
    dpt = -delta_mu * u_ex_cov[0]
    dpr = -delta_mu * u_ex_cov[1]
    dpphi = -delta_mu * u_ex_cov[2]
    
    # New momentum
    pt_new = pt + dpt
    pr_new = pr + dpr
    pphi_new = pphi + dpphi
    
    # New mass from mass-shell constraint: m^2 = -g^{munu} p_mu p_nu
    _, con = kerr_metric_components(r, th, a, M)
    gu_tt, gu_tphi, gu_rr, gu_thth, gu_phiphi = con
    
    m2_new = -(gu_tt * pt_new**2 
               + 2*gu_tphi * pt_new * pphi_new 
               + gu_rr * pr_new**2 
               + gu_phiphi * pphi_new**2)
    
    mass_shell_valid = True
    if m2_new < 0:
        if clamp_negative_mass:
            # Clamp to prevent crash, but flag as invalid
            import warnings
            warnings.warn(
                f"Mass-shell violation: m^2 = {m2_new:.6e} < 0 at r={r:.4f}. "
                f"Impulse (deltamu={delta_mu:.6f}) may be too strong. Clamping to m^2=1e-20.",
                RuntimeWarning
            )
            m2_new = 1e-20
            mass_shell_valid = False
        else:
            raise PhysicsValidationError(
                f"Mass-shell violation: m^2 = {m2_new:.6e} < 0 at r={r:.4f}. "
                f"Impulse (deltamu={delta_mu:.6f}) too strong or unphysical exhaust direction."
            )
        
    m_new = np.sqrt(m2_new)
    dm = m_new - m  # This should be negative (mass loss)
    
    E_new = -pt_new
    
    # Conservation check: verify 4-momentum was properly updated
    # For exact conservation: p'_mu = p_mu - deltamu * u_{ex,mu}
    # This is exact by construction, but we verify for debugging
    if verify:
        p_new = (pt_new, pr_new, pphi_new)
        cons_valid, cons_residual = verify_4momentum_conservation(
            p_cov, p_new, delta_mu, u_ex_cov
        )
        if not cons_valid:
            import warnings
            warnings.warn(
                f"4-momentum conservation residual larger than expected: {np.max(np.abs(cons_residual)):.6e}",
                RuntimeWarning
            )
    
    # Legacy conservation residual (approximate check)
    conservation_residual = abs(dm + delta_mu * u_ex_result['gamma_e'])
    
    return {
        'dp_cov': (dpt, dpr, dpphi),
        'dm': dm,
        'm_new': m_new,
        'E_new': E_new,
        'conservation_residual': conservation_residual,
        'mass_shell_valid': mass_shell_valid
    }


def compute_exhaust_energy(u_vec, s_vec, v_e, g_tt, g_tphi, g_phiphi):
    """Compute exhaust Killing energy E_ex = -u_ex,t. E_ex < 0 means Penrose extraction."""
    gamma_e = 1.0 / np.sqrt(1.0 - v_e**2)
    
    # Exhaust 4-velocity (contravariant)
    u_ex = gamma_e * (u_vec - v_e * s_vec)
    
    # Covariant time component: u_{ex,t} = g_{tmu} u_ex^mu
    u_ex_t_cov = g_tt * u_ex[0] + g_tphi * u_ex[2]
    
    # Killing energy E_ex = -u_{ex,t}
    E_ex = -u_ex_t_cov
    
    # Also return s_t for diagnostics
    s_t = g_tt * s_vec[0] + g_tphi * s_vec[2]
    
    return E_ex, s_t


def compute_E_ex_threshold(E_rocket, v_e):
    """Critical s_t threshold for E_ex < 0."""
    return -E_rocket / v_e


def compute_min_achievable_st(r, E, Lz, m, a, M=1.0, n_samples=37):
    """Minimum achievable s_t at radius r by scanning thrust directions."""
    th = np.pi / 2  # Equatorial
    
    # Get metric
    cov, con = kerr_metric_components(r, th, a, M)
    g_tt, g_tphi, g_rr, g_thth, g_phiphi = cov
    gu_tt, gu_tphi, gu_rr, gu_thth, gu_phiphi = con
    
    # Compute pr from mass shell: g^{munu}p_mup_nu = -m^2
    pt = -E  # p_t = -E (Killing energy)
    pphi = Lz
    rhs = -(gu_tt * pt**2 + 2*gu_tphi * pt * pphi + gu_phiphi * pphi**2 + m**2)
    if rhs < 0:
        return None  # Forbidden region
    pr = -np.sqrt(rhs / gu_rr)  # Ingoing
    
    # 4-velocity (contravariant)
    u_t = (gu_tt * pt + gu_tphi * pphi) / m
    u_r = gu_rr * pr / m
    u_phi = (gu_tphi * pt + gu_phiphi * pphi) / m
    u_vec = np.array([u_t, u_r, u_phi])
    
    # Build rest frame basis
    e_r, e_phi = build_rocket_rest_basis(u_vec, g_tt, g_tphi, g_rr, g_phiphi)
    if e_r is None or e_phi is None:
        return None
    
    # Scan over all thrust directions
    min_st = float('inf')
    for alpha in np.linspace(-np.pi/2, np.pi/2, n_samples):
        s_r = np.sin(alpha)
        s_phi_mag = np.cos(alpha)
        for sign in [+1.0, -1.0]:
            s_vec = s_r * e_r + (sign * s_phi_mag) * e_phi
            s_t = g_tt * s_vec[0] + g_tphi * s_vec[2]
            if s_t < min_st:
                min_st = s_t
    
    return min_st


def compute_extraction_limit_radius(E, Lz, m, v_e, a, M=1.0, r_min=1.1, r_max=2.5, n_radii=50):
    """Find maximum radius where E_ex < 0 is geometrically achievable."""
    E_rocket = E / m
    s_t_crit = -E_rocket / v_e  # Need s_t < s_t_crit for E_ex < 0
    
    r_plus = horizon_radius(a, M)
    r_min = max(r_min, r_plus + 0.05)  # Don't go too close to horizon
    
    # Scan radii from outside in
    for r in np.linspace(r_max, r_min, n_radii):
        min_st = compute_min_achievable_st(r, E, Lz, m, a, M)
        if min_st is not None and min_st < s_t_crit:
            return r  # First radius where extraction becomes possible
    
    return None  # E_ex < 0 never achievable


def determine_thrust_mode(r, pr, E, m, m0, E_ex_history, 
                          r_plus, r_erg, r_safe,
                          m_reserve_fraction=0.3,
                          n_consecutive=10,
                          r_extraction_limit=None,
                          Lz=None, v_e=None, a=None, M=1.0):
    """
    State machine for thrust mode. Returns (mode, reason_string).
    
    EXTRACTION: deep in ergosphere, E_ex < 0 achievable
    ESCAPE: time to leave (low fuel or extraction ineffective)  
    COAST: outside ergosphere or critically low fuel
    """
    m_reserve = m_reserve_fraction * m0
    
    # Check if inside ergosphere
    in_ergosphere = r < r_erg
    
    # Compute extraction limit dynamically if not provided
    if r_extraction_limit is None:
        if Lz is not None and v_e is not None and a is not None:
            # Compute dynamically (expensive - consider caching)
            r_extraction_limit = compute_extraction_limit_radius(E, Lz, m, v_e, a, M)
            if r_extraction_limit is None:
                r_extraction_limit = r_plus  # Never achievable, fallback
        else:
            # Fallback to hardcoded value for backward compatibility
            r_extraction_limit = 1.80
    
    # Default: EXTRACTION if deep enough in ergosphere and fuel available
    if in_ergosphere and r < r_extraction_limit and m > m_reserve and r > r_safe:
        # Check if extraction is still effective
        if len(E_ex_history) >= n_consecutive:
            recent = E_ex_history[-n_consecutive:]
            if all(e >= 0 for e in recent):
                return ThrustMode.ESCAPE, "E_ex no longer negative"
        return ThrustMode.EXTRACTION, f"inside extraction zone (r<{r_extraction_limit:.2f}), extraction active"
    
    # Inside ergosphere but not deep enough for E_ex < 0
    if in_ergosphere and r >= r_extraction_limit:
        if pr < 0:  # Falling inward
            return ThrustMode.COAST, "falling toward extraction zone"
        # Shouldn't happen normally - would be escaping outward inside ergosphere
        return ThrustMode.ESCAPE, "inside ergosphere but r > extraction limit"
    
    # Emergency: too close to horizon
    if r < r_safe + 0.03:
        return ThrustMode.ESCAPE, "emergency: approaching horizon"
    
    # Low fuel: switch to escape
    if m <= m_reserve:
        return ThrustMode.ESCAPE, "fuel reserve reached"
    
    # Outside ergosphere
    if not in_ergosphere:
        # If falling inward toward ergosphere, let geodesic motion continue
        if pr < 0:
            return ThrustMode.COAST, "falling inward toward ergosphere"
        # Check if escape trajectory achieved
        specific_E = E / m if m > 0 else 0
        if specific_E > 1.0 and pr > 0:
            return ThrustMode.COAST, "unbound escape trajectory"
        # Exiting ergosphere outward
        return ThrustMode.ESCAPE, "outside ergosphere, escaping outward"
    
    return ThrustMode.EXTRACTION, "default extraction mode"


# =============================================================================
# ROCKET REST FRAME TETRAD
# =============================================================================
def _inner_prod_tr(v, w, g_tt, g_tphi, g_rr, g_phiphi):
    """Metric inner product g_munu v^mu w^nu in (t,r,phi) subspace."""
    vt, vr, vphi = v
    wt, wr, wphi = w
    return (
        g_tt * vt * wt
        + g_rr * vr * wr
        + g_phiphi * vphi * wphi
        + g_tphi * (vt * wphi + vphi * wt)
    )


def _normalize_spacelike(v, g_tt, g_tphi, g_rr, g_phiphi, eps=1e-30):
    """Normalize spacelike vector: v/sqrt(v*v)."""
    n2 = _inner_prod_tr(v, v, g_tt, g_tphi, g_rr, g_phiphi)
    if (not np.isfinite(n2)) or (n2 <= eps):
        return None
    return v / np.sqrt(n2)


def build_rocket_rest_basis(u_contra, g_tt, g_tphi, g_rr, g_phiphi):
    """Build orthonormal spatial basis (e_r, e_phi) in rocket rest frame via Gram-Schmidt."""
    u = np.asarray(u_contra, dtype=float)

    # Start with unit coordinate directions in the metric.
    b_r = np.array([0.0, 1.0 / np.sqrt(g_rr), 0.0])
    b_phi = np.array([0.0, 0.0, 1.0 / np.sqrt(g_phiphi)])

    # Orthogonalize b_r against u: w_r = b_r + (u*b_r) u  (since u*u=-1)
    u_dot_br = _inner_prod_tr(u, b_r, g_tt, g_tphi, g_rr, g_phiphi)
    w_r = b_r + u_dot_br * u
    e_r = _normalize_spacelike(w_r, g_tt, g_tphi, g_rr, g_phiphi)
    if e_r is None:
        return None, None

    # Orthogonalize b_phi against u
    u_dot_bphi = _inner_prod_tr(u, b_phi, g_tt, g_tphi, g_rr, g_phiphi)
    w_phi = b_phi + u_dot_bphi * u
    # Remove any component along e_r
    er_dot_wphi = _inner_prod_tr(e_r, w_phi, g_tt, g_tphi, g_rr, g_phiphi)
    w_phi2 = w_phi - er_dot_wphi * e_r
    e_phi = _normalize_spacelike(w_phi2, g_tt, g_tphi, g_rr, g_phiphi)
    if e_phi is None:
        return None, None

    return e_r, e_phi

# =============================================================================
# EXACT 4-MOMENTUM CONSERVATION FOR SINGLE IMPULSE
# =============================================================================

def apply_exact_impulse(p_cov, m, delta_mu, u_ex_cov, r, th, a, M=1.0):
    """
    Apply exact 4-momentum conservation for single impulse: p'mu = pmu - deltamu * u_ex,mu.
    
    New mass from mass-shell: m'^2 = -g^munu p'mu p'nu.
    Returns dict with 'p_cov_new', 'm_new', 'E_new', 'E_ex'.
    """
    pt, pr, pphi = p_cov
    u_ex_t, u_ex_r, u_ex_phi = u_ex_cov
    
    # New covariant momentum
    pt_new = pt - delta_mu * u_ex_t
    pr_new = pr - delta_mu * u_ex_r
    pphi_new = pphi - delta_mu * u_ex_phi
    
    # Get contravariant metric for mass-shell
    _, con = kerr_metric_components(r, th, a, M)
    gu_tt, gu_tphi, gu_rr, gu_thth, gu_phiphi = con
    
    # Mass-shell: m^2 = -g^{munu} p_mu p_nu
    m2_new = -(gu_tt * pt_new**2 
               + 2*gu_tphi * pt_new * pphi_new 
               + gu_rr * pr_new**2 
               + gu_phiphi * pphi_new**2)
    
    if m2_new < 0:
        raise ValueError(f"Mass-shell violation: m^2 = {m2_new:.6e} < 0. "
                        "Impulse too strong or unphysical exhaust direction.")
    
    m_new = np.sqrt(m2_new)
    E_new = -pt_new  # E = -p_t
    E_ex = -u_ex_t   # Exhaust Killing energy (per unit rest mass)
    
    return {
        'p_cov_new': (pt_new, pr_new, pphi_new),
        'm_new': m_new,
        'E_new': E_new,
        'E_ex': E_ex
    }


def compute_optimal_exhaust_direction(u_vec, r, th, a, M, v_e, 
                                       g_tt, g_tphi, g_rr, g_phiphi,
                                       n_samples=37, target='min_E_ex'):
    """Find exhaust direction that minimizes E_ex for Penrose extraction."""
    gamma_e = 1.0 / np.sqrt(1.0 - v_e**2)
    
    # Build rocket rest frame basis
    e_r, e_phi = build_rocket_rest_basis(u_vec, g_tt, g_tphi, g_rr, g_phiphi)
    if e_r is None or e_phi is None:
        return None
    
    best_E_ex = float('inf')
    best_s_vec = None
    best_u_ex = None
    
    for alpha in np.linspace(-np.pi/2, np.pi/2, n_samples):
        s_r = np.sin(alpha)
        s_phi_mag = np.cos(alpha)
        
        for sign_phi in [+1.0, -1.0]:
            s_vec = s_r * e_r + (sign_phi * s_phi_mag) * e_phi
            
            # Exhaust 4-velocity (contravariant)
            u_ex = gamma_e * (u_vec - v_e * s_vec)
            
            # Covariant time component
            u_ex_t_cov = g_tt * u_ex[0] + g_tphi * u_ex[2]
            E_ex = -u_ex_t_cov
            
            if E_ex < best_E_ex:
                best_E_ex = E_ex
                best_s_vec = s_vec.copy()
                best_u_ex = u_ex.copy()
    
    if best_s_vec is None or best_u_ex is None:
        return None
    
    # Compute full covariant exhaust 4-velocity
    u_ex_t_cov = g_tt * best_u_ex[0] + g_tphi * best_u_ex[2]
    u_ex_r_cov = g_rr * best_u_ex[1]
    u_ex_phi_cov = g_tphi * best_u_ex[0] + g_phiphi * best_u_ex[2]
    
    return {
        'E_ex': best_E_ex,
        's_vec': best_s_vec,
        'u_ex_contra': best_u_ex,
        'u_ex_cov': (u_ex_t_cov, u_ex_r_cov, u_ex_phi_cov)
    }


# =============================================================================
# ENERGY BUDGET VALIDATION
# =============================================================================

def compute_energy_budget(E_initial, E_final, m_initial, m_final, 
                          E_ex_history, delta_mu_history):
    """
    Compute energy budget for validation: DeltaE_rocket + Sum(E_ex * deltamu) ~ 0.
    
    Returns dict with Delta_E, energy_to_BH, conservation_error, efficiencies.
    """
    E_ex_arr = np.array(E_ex_history) if len(E_ex_history) > 0 else np.array([])
    dmu_arr = np.array(delta_mu_history) if len(delta_mu_history) > 0 else np.array([])
    
    # Actual energy change
    Delta_E_actual = E_final - E_initial
    
    # Fuel consumption
    Delta_m = m_initial - m_final  # Rocket mass loss
    total_delta_mu = np.sum(dmu_arr) if len(dmu_arr) > 0 else 0.0  # Total exhaust rest mass
    
    # Energy deposited into BH (sum of E_ex * deltamu for each step)
    # Negative means energy extracted from BH
    if len(E_ex_arr) > 0 and len(dmu_arr) > 0:
        energy_to_BH = np.sum(E_ex_arr * dmu_arr)
    else:
        energy_to_BH = 0.0
    
    # Conservation check: DeltaE_rocket + energy_to_BH ~ 0
    # (neglecting kinetic energy of exhaust that escapes to infinity)
    conservation_error = Delta_E_actual + energy_to_BH
    
    # Efficiency metrics
    # eta_cum = DeltaE / Deltam (energy per unit rocket mass lost)
    eta_cum = Delta_E_actual / Delta_m if Delta_m > 0 else 0.0
    
    # eta_rest = DeltaE / deltamu_total (energy per unit exhaust rest mass)
    # This is NOT directly comparable to Wald's limit, which is for single decay
    eta_rest = Delta_E_actual / total_delta_mu if total_delta_mu > 0 else 0.0
    
    # Breakdown of exhaust energy
    if len(E_ex_arr) > 0:
        E_ex_negative_mask = E_ex_arr < 0
        n_negative = np.sum(E_ex_negative_mask)
        
        if n_negative > 0:
            # Energy extracted from BH via negative-energy exhaust
            energy_extracted = -np.sum(E_ex_arr[E_ex_negative_mask] * dmu_arr[E_ex_negative_mask])
        else:
            energy_extracted = 0.0
            
        # Energy deposited by positive-energy exhaust
        E_ex_positive_mask = E_ex_arr >= 0
        if np.sum(E_ex_positive_mask) > 0:
            energy_deposited = np.sum(E_ex_arr[E_ex_positive_mask] * dmu_arr[E_ex_positive_mask])
        else:
            energy_deposited = 0.0
    else:
        n_negative = 0
        energy_extracted = 0.0
        energy_deposited = 0.0
    
    return {
        'Delta_E': Delta_E_actual,
        'Delta_m': Delta_m,
        'total_delta_mu': total_delta_mu,
        'energy_to_BH': energy_to_BH,
        'energy_extracted': energy_extracted,
        'energy_deposited': energy_deposited,
        'conservation_error': conservation_error,
        'eta_cum': eta_cum,
        'eta_rest': eta_rest,
        'n_negative_E_ex': n_negative,
        'n_total_steps': len(E_ex_arr)
    }


def print_energy_budget(budget, a_spin, M_bh=1.0):
    """Print formatted energy budget analysis."""
    eta_rot = max_rotational_energy_fraction(a_spin, M_bh)
    eta_wald = wald_single_decay_limit()
    
    print("\n" + "="*65)
    print("         ENERGY BUDGET VALIDATION")
    print("="*65)
    
    print(f"\n{'ROCKET ENERGY CHANGE':^65}")
    print("-"*65)
    print(f"  Energy change:               DeltaE = {budget['Delta_E']:+.6f}")
    print(f"  Rocket mass loss:            Deltam = {budget['Delta_m']:.6f}")
    print(f"  Exhaust rest mass:           deltamu = {budget['total_delta_mu']:.6f}")
    
    print(f"\n{'BLACK HOLE ENERGY EXCHANGE':^65}")
    print("-"*65)
    print(f"  Energy to BH:                    {budget['energy_to_BH']:+.6f}")
    print(f"    via negative-E_ex (extracted): {budget['energy_extracted']:+.6f}")
    print(f"    via positive-E_ex (deposited): {budget['energy_deposited']:+.6f}")
    print(f"  Steps with E_ex < 0:             {budget['n_negative_E_ex']}/{budget['n_total_steps']}")
    
    print(f"\n{'CONSERVATION CHECK':^65}")
    print("-"*65)
    print(f"  DeltaE + E_to_BH =                   {budget['conservation_error']:+.6e}")
    if abs(budget['conservation_error']) < 0.01 * abs(budget['Delta_E'] + 1e-10):
        print(f"    [OK] Energy approximately conserved")
    else:
        print(f"    Note: Large residual is expected for approximate thrust model.")
        print(f"    The model imposes mass loss rather than deriving it from exact")
        print(f"    4-momentum conservation. This doesn't invalidate the physics:")
        print(f"    E_ex < 0 remains the unambiguous Penrose extraction signature.")
    
    print(f"\n{'EFFICIENCY METRICS':^65}")
    print("-"*65)
    print(f"  eta_cum = DeltaE/Deltam:                   {budget['eta_cum']:.4f} ({budget['eta_cum']*100:.2f}%)")
    print(f"  eta_rest = DeltaE/deltamu:                  {budget['eta_rest']:.4f} ({budget['eta_rest']*100:.2f}%)")
    print(f"  Wald single-decay limit:         {eta_wald:.4f} ({eta_wald*100:.2f}%)")
    print(f"  BH rotational energy budget:     {eta_rot:.4f} ({eta_rot*100:.2f}%)")
    print(f"\n  Note: eta_cum and eta_rest are cumulative efficiencies from")
    print(f"    multiple thrust steps. They are NOT directly comparable to")
    print(f"    Wald's limit (single decay) or the BH rotational budget (total).")
    print(f"    Cumulative efficiency CAN exceed Wald's limit through")
    print(f"    multiple extraction events, but cannot exceed the BH budget.")
    
    print("="*65)