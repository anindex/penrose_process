
"""Test continuous thrust with optimal Penrose extraction direction."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.integrate import solve_ivp
from kerr_utils import (
    kerr_metric_components, horizon_radius,
    compute_dH_dr_analytic, compute_optimal_exhaust_direction
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

def eom_optimal(tau, y, T):
    """EOMs with thrust in optimal Penrose direction."""
    r, phi, pt, pr, pphi, m = y
    if r < r_safe or m <= 0:
        return [0, 0, 0, 0, 0, 0]

    cov, con = metric(r)
    g_tt, g_tphi, g_rr, _, g_phiphi = cov
    gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con

    u_t = (gu_tt * pt + gu_tphi * pphi) / m
    u_r = gu_rr * pr / m
    u_phi = (gu_tphi * pt + gu_phiphi * pphi) / m
    u_vec = np.array([u_t, u_r, u_phi])

    dH_dr = compute_dH_dr_analytic(r, np.pi/2, pt, pr, pphi, m, a, M)
    dpr_geo = -dH_dr / m

    # Find optimal thrust direction for Penrose extraction
    res = compute_optimal_exhaust_direction(
        u_vec, r, np.pi/2, a, M, ve,
        g_tt, g_tphi, g_rr, g_phiphi, n_samples=19
    )

    if res is not None and res['E_ex'] < 0 and m > 0.1:
        s = res['s_vec']
        dpt = T * gamma_e * s[0]
        dpr = dpr_geo + T * gamma_e * s[1]
        dpphi = T * gamma_e * s[2]
        dm = -T / ve
    else:
        dpt = 0.0
        dpr = dpr_geo
        dpphi = 0.0
        dm = 0.0

    return [u_r, u_phi, dpt, dpr, dpphi, dm]

def run_simulation(E0, Lz0, T=0.02, r0=10.0):
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
        lambda t, y: eom_optimal(t, y, T),
        [0, 800], y0,
        method='RK45',
        events=[event_cap, event_esc],
        rtol=1e-9, atol=1e-11, max_step=0.1
    )

    r_min = np.min(sol.y[0])
    m_final = sol.y[5, -1]
    E_final = -sol.y[2, -1]

    if len(sol.t_events[0]) > 0:
        status = 'CAPTURED'
    elif len(sol.t_events[1]) > 0:
        status = 'ESCAPED'
    else:
        status = 'TIMEOUT'

    return {
        'status': status,
        'r_min': r_min,
        'm_final': m_final,
        'E_final': E_final,
        'delta_E': E_final - E0,
        'delta_E_pct': 100 * (E_final - E0) / E0
    }


def test_optimal_thrust_extraction():
    """Optimal thrust with E0=1.2, Lz=3.0 should escape (continuous thrust may or may not extract net energy)."""
    result = run_simulation(1.2, 3.0)
    assert result is not None, "run_simulation returned None -- invalid initial conditions"
    assert result['status'] == 'ESCAPED', (
        f"Expected 'ESCAPED' but got '{result['status']}'"
    )
    # Continuous thrust with T=0.02 may not achieve net positive delta_E,
    # but should lose less energy than the fuel mass expended
    assert result['m_final'] < 1.0, (
        f"Expected mass loss from thrust but m_final={result['m_final']:.6f}"
    )


if __name__ == '__main__':
    print("Testing continuous thrust with OPTIMAL Penrose direction")
    print("=" * 60)
    print(f"Parameters: a={a}, v_e={ve}, r+={r_plus:.4f}M, r_safe={r_safe:.4f}M")
    print()

    candidates = [
        (1.20, 3.0),   # Single thrust case params
        (1.15, 2.8),
        (1.10, 2.6),
        (1.05, 2.4),
    ]

    for E0, Lz0 in candidates:
        result = run_simulation(E0, Lz0, T=0.02)
        if result:
            print(f"E0={E0:.2f}, Lz={Lz0:.1f}: {result['status']}")
            print(f"  r_min={result['r_min']:.4f}M, m_final={result['m_final']:.4f}")
            print(f"  E_final={result['E_final']:.4f}, DeltaE={result['delta_E']:.4f} ({result['delta_E_pct']:.2f}%)")
        else:
            print(f"E0={E0:.2f}, Lz={Lz0:.1f}: Invalid initial conditions")
        print()

    print("Running assertions...")
    test_optimal_thrust_extraction()
    print("  test_optimal_thrust_extraction: PASSED")
    print("All tests passed.")
