
"""Test continuous thrust with retrograde direction for Penrose extraction."""

import numpy as np
from scipy.integrate import solve_ivp
from kerr_utils import (
    kerr_metric_components, horizon_radius, 
    compute_dH_dr_analytic, compute_exhaust_energy
)

M = 1.0
a = 0.95
r_plus = horizon_radius(a, M)
r_safe = r_plus + 0.02
ESCAPE_RADIUS = 50.0
ve = 0.95
gamma_e = 1.0 / np.sqrt(1 - ve**2)

def metric(r):
    return kerr_metric_components(r, np.pi/2, a, M)

def eom_retrograde(tau, y, T, thrust_zone=(1.3, 1.8)):
    """EOMs with retrograde thrust (against black hole rotation) in extraction zone."""
    r, phi, pt, pr, pphi, m = y
    if r < r_safe or m <= 0:
        return [0, 0, 0, 0, 0, 0]
    
    cov, con = metric(r)
    g_tt, g_tphi, g_rr, _, g_phiphi = cov
    gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con
    
    u_t = (gu_tt * pt + gu_tphi * pphi) / m
    u_r = gu_rr * pr / m
    u_phi = (gu_tphi * pt + gu_phiphi * pphi) / m
    
    dH_dr = compute_dH_dr_analytic(r, np.pi/2, pt, pr, pphi, m, a, M)
    dpr_geo = -dH_dr / m
    
    # Apply retrograde thrust in extraction zone
    r_min, r_max = thrust_zone
    if r_min <= r <= r_max and m > 0.1:
        # Retrograde thrust: s points in -phi direction (against BH rotation)
        # Normalized: g_phiphi * s^phi * s^phi = 1 -> s^phi = -1/sqrt(g_phiphi)
        s_phi = -1.0 / np.sqrt(g_phiphi)  # Retrograde
        s_r = 0.0  # No radial component
        s_t = 0.0  # Purely spatial in this coordinate gauge
        
        # Thrust adds momentum in thrust direction
        # dp/dtau = T * gamma_e * s (where s is unit spacelike)
        dpt = 0.0  # No change in t-momentum from purely spatial thrust
        dpr = dpr_geo + T * gamma_e * s_r  # = dpr_geo
        dpphi = T * gamma_e * s_phi  # Retrograde momentum change
        dm = -T / ve
    else:
        dpt = 0.0
        dpr = dpr_geo
        dpphi = 0.0
        dm = 0.0
    
    return [u_r, u_phi, dpt, dpr, dpphi, dm]

def run_simulation(E0, Lz0, T=0.05, r0=10.0, thrust_zone=(1.3, 1.8)):
    """Run the simulation with given initial conditions."""
    _, con0 = metric(r0)
    gu_tt0, gu_tphi0, gu_rr0, _, gu_phiphi0 = con0
    
    pt0 = -E0
    pphi0 = Lz0
    rhs = -(gu_tt0 * pt0**2 + 2 * gu_tphi0 * pt0 * pphi0 + gu_phiphi0 * pphi0**2 + 1.0)
    if rhs < 0:
        return None
    pr0 = -np.sqrt(rhs / gu_rr0)
    y0 = [r0, 0.0, pt0, pr0, pphi0, 1.0]
    
    def event_cap(t, y):
        return y[0] - r_safe
    event_cap.terminal = True
    event_cap.direction = -1
    
    def event_esc(t, y):
        """Escape: r > ESCAPE_RADIUS, pr > 0, AND E/m > 1 (unbound)."""
        r, _, pt, pr, _, m = y
        if pr <= 0 or m <= 0:
            return -1
        E = -pt
        if E / m <= 1.0:  # Must be unbound
            return -1
        return r - ESCAPE_RADIUS
    event_esc.terminal = True
    event_esc.direction = 1
    
    sol = solve_ivp(
        lambda t, y: eom_retrograde(t, y, T, thrust_zone), 
        [0, 500], y0, 
        method='RK45', 
        events=[event_cap, event_esc],
        rtol=1e-8, atol=1e-10, max_step=0.2
    )
    
    r_traj = sol.y[0]
    m_traj = sol.y[5]
    pt_traj = sol.y[2]
    pphi_traj = sol.y[4]
    
    r_min = np.min(r_traj)
    m_final = m_traj[-1]
    E_final = -pt_traj[-1]
    
    if len(sol.t_events[0]) > 0:
        status = 'CAPTURED'
    elif len(sol.t_events[1]) > 0:
        status = 'ESCAPED'
    else:
        status = 'TIMEOUT'
    
    # Check E_ex at a point in the extraction zone
    idx = np.argmin(r_traj)
    r_sample = r_traj[idx]
    if thrust_zone[0] <= r_sample <= thrust_zone[1]:
        cov, con = metric(r_sample)
        g_tt, g_tphi, g_rr, _, g_phiphi = cov
        gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con
        
        m_sample = m_traj[idx]
        pt_sample = pt_traj[idx]
        pphi_sample = pphi_traj[idx]
        pr_sample = sol.y[3, idx]
        
        u_t = (gu_tt * pt_sample + gu_tphi * pphi_sample) / m_sample
        u_r = gu_rr * pr_sample / m_sample
        u_phi = (gu_tphi * pt_sample + gu_phiphi * pphi_sample) / m_sample
        
        s_phi = -1.0 / np.sqrt(g_phiphi)
        u_vec = np.array([u_t, u_r, u_phi])
        s_vec = np.array([0.0, 0.0, s_phi])
        
        E_ex, _ = compute_exhaust_energy(u_vec, s_vec, ve, g_tt, g_tphi, g_phiphi)
    else:
        E_ex = None
    
    return {
        'status': status,
        'r_min': r_min,
        'm_final': m_final,
        'E_final': E_final,
        'delta_E': E_final - E0,
        'delta_E_pct': 100 * (E_final - E0) / E0,
        'E_ex_at_peri': E_ex
    }

if __name__ == '__main__':
    print("Testing continuous RETROGRADE thrust")
    print("=" * 60)
    print(f"Parameters: a={a}, v_e={ve}, r+={r_plus:.4f}M, r_safe={r_safe:.4f}M")
    print()
    
    # Test with parameters from single_thrust_case
    candidates = [
        (1.20, 3.0, 0.05, (1.35, 1.8)),
        (1.20, 3.0, 0.1, (1.35, 1.8)),
        (1.15, 2.8, 0.05, (1.35, 1.8)),
    ]
    
    for E0, Lz0, T, zone in candidates:
        print(f"E0={E0:.2f}, Lz={Lz0:.1f}, T={T}, zone={zone}")
        result = run_simulation(E0, Lz0, T=T, thrust_zone=zone)
        if result:
            print(f"  Status: {result['status']}")
            print(f"  r_min={result['r_min']:.4f}M, m_final={result['m_final']:.4f}")
            print(f"  E_final={result['E_final']:.4f}, DeltaE={result['delta_E']:.4f} ({result['delta_E_pct']:.2f}%)")
            if result['E_ex_at_peri'] is not None:
                status = "PENROSE" if result['E_ex_at_peri'] < 0 else "positive"
                print(f"  E_ex at periapsis: {result['E_ex_at_peri']:.4f} ({status})")
        else:
            print(f"  Invalid initial conditions")
        print()
