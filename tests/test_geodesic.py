"""
Test geodesic motion (no thrust) for flyby orbits.

If the flyby orbit has a genuine turning point, it should escape without any thrust.
This confirms our orbital parameters are correct.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from kerr_utils import (
    kerr_metric_components, horizon_radius, ergosphere_radius,
    kerr_metric_derivatives, compute_dH_dr_analytic
)

M = 1.0
a = 0.95
r_plus = horizon_radius(a, M)
r_erg = 2.0
r_safe = r_plus + 0.02
ESCAPE_RADIUS = 50.0
DT_INTEGRATION = 0.01  # Integration timestep - single source of truth

def kerr_metric_cov_contra(r, th=np.pi/2):
    return kerr_metric_components(r, th, a, M)

def project_to_mass_shell(y, last_pr_sign=None):
    """Project state onto mass-shell with robust handling of forbidden regions."""
    r, phi, pt, pr, pphi, m = y
    if r < r_safe or m <= 0:
        return y
    _, con = kerr_metric_cov_contra(r, np.pi/2)
    gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con
    A = gu_tt
    B = 2*gu_tphi*pphi
    C = gu_rr*pr**2 + gu_phiphi*pphi**2 + m**2
    det = B**2 - 4*A*C
    if det < 0:
        if abs(det) < 1e-10:
            det = 0.0  # Clamp small numerical errors
        else:
            return y  # Unphysical state
    pt_new = (-B + np.sqrt(det)) / (2*A)  # Future-directed root (u^t > 0)
    return np.array([r, phi, pt_new, pr, pphi, m])


def geodesic_dynamics(tau, y):
    """Pure geodesic (no thrust)."""
    r, phi, pt, pr, pphi, m = y

    if r < r_safe:
        return [0, 0, 0, 0, 0, 0]

    th = np.pi/2
    _, con = kerr_metric_cov_contra(r, th)
    gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con

    p_contra_r = gu_rr * pr
    p_contra_phi = gu_tphi * pt + gu_phiphi * pphi

    dr = p_contra_r / m
    dphi = p_contra_phi / m

    # Geodesic equation
    dH_dr = compute_dH_dr_analytic(r, th, pt, pr, pphi, m, a, M)
    dpr = -dH_dr / m

    return [dr, dphi, 0.0, dpr, 0.0, 0.0]


def run_geodesic(E0, Lz0, r0=10.0, verbose=True):
    """Run pure geodesic motion."""
    th0 = np.pi/2
    _, con0 = kerr_metric_cov_contra(r0, th0)
    gu_tt0, gu_tphi0, gu_rr0, _, gu_phiphi0 = con0

    pt0 = -E0
    pphi0 = Lz0

    rhs = -(gu_tt0*pt0**2 + 2*gu_tphi0*pt0*pphi0 + gu_phiphi0*pphi0**2 + 1.0)
    if rhs < 0:
        if verbose:
            print(f"  E0={E0}, Lz={Lz0}: Invalid IC (rhs={rhs:.4f})")
        return None
    pr0 = -np.sqrt(rhs / gu_rr0)  # Infalling

    y0 = np.array([r0, 0.0, pt0, pr0, pphi0, 1.0])

    # Integrate
    dt = DT_INTEGRATION  # Use unified timestep constant
    tau = 0.0
    tau_max = 200.0
    y = y0.copy()

    r_min = r0
    r_history = [r0]
    pr_history = [pr0]

    while tau < tau_max:
        r = y[0]
        pr = y[3]

        r_min = min(r_min, r)
        r_history.append(r)
        pr_history.append(pr)

        if r < r_safe:
            if verbose:
                print(f"  E0={E0:.2f}, Lz={Lz0:.2f}: CAPTURED at r={r:.4f}M (r_min={r_min:.4f}M)")
            return {'status': 'captured', 'r_min': r_min}

        # Strong escape criterion: r > ESCAPE_RADIUS, pr > 0, AND E/m > 1 (unbound)
        E_current = -y[2]
        m = y[5]
        is_unbound = (m > 0) and (E_current / m > 1.0)
        if r > ESCAPE_RADIUS and pr > 0 and is_unbound:
            if verbose:
                print(f"  E0={E0:.2f}, Lz={Lz0:.2f}: ESCAPED! r_min={r_min:.4f}M")
            return {'status': 'escaped', 'r_min': r_min}

        dydt = geodesic_dynamics(tau, y)
        y = y + np.array(dydt) * dt
        y = project_to_mass_shell(y)
        tau += dt

    if verbose:
        print(f"  E0={E0:.2f}, Lz={Lz0:.2f}: TIMEOUT at r={y[0]:.2f}M (r_min={r_min:.4f}M)")
    return {'status': 'timeout', 'r_min': r_min, 'r_final': y[0]}


def test_known_flyby_escapes():
    """The known working orbit (E0=1.2, Lz=3.0) should escape on a pure geodesic."""
    result = run_geodesic(1.2, 3.0, verbose=False)
    assert result is not None, "run_geodesic returned None for E0=1.2, Lz=3.0"
    assert result['status'] == 'escaped', (
        f"Expected 'escaped' but got '{result['status']}' for E0=1.2, Lz=3.0"
    )


def test_flyby_configurations():
    """Various flyby configurations should reach a definite outcome (escape or capture)."""
    configs = [
        (1.2, 3.6),
        (1.2, 3.0),
        (1.3, 3.4),
        (1.4, 4.2),
        (1.5, 4.6),
        (1.6, 4.8),
    ]
    for E0, Lz0 in configs:
        result = run_geodesic(E0, Lz0, verbose=False)
        assert result is not None, (
            f"run_geodesic returned None for E0={E0}, Lz={Lz0}"
        )
        assert result['status'] in ('escaped', 'captured'), (
            f"Expected definite outcome for E0={E0}, Lz={Lz0} "
            f"but got '{result['status']}'"
        )


if __name__ == '__main__':
    print("=" * 70)
    print("GEODESIC (NO THRUST) TEST")
    print("=" * 70)
    print(f"Testing pure geodesic motion for flyby orbits...")
    print()

    # Test the single thrust case orbit
    print("Known working orbit (single thrust case):")
    run_geodesic(1.2, 3.0)
    print()

    # Test various flyby configurations
    print("Testing flyby configurations from orbit analysis:")
    configs = [
        (1.2, 3.6),   # Deepest periapsis
        (1.2, 3.0),   # Single thrust case
        (1.3, 3.4),
        (1.4, 4.2),
        (1.5, 4.6),
        (1.6, 4.8),
    ]

    for E0, Lz0 in configs:
        run_geodesic(E0, Lz0)

    print()
    print("Running assertions...")
    test_known_flyby_escapes()
    print("  test_known_flyby_escapes: PASSED")
    test_flyby_configurations()
    print("  test_flyby_configurations: PASSED")
    print("All tests passed.")
