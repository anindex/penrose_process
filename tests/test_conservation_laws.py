"""
Tests for conservation laws and known analytical solutions in Kerr spacetime.

Covers: ISCO radii, metric components, ergosphere geometry, geodesic
conservation of energy and angular momentum, mass-shell constraint,
4-momentum conservation under impulse, tetrad orthonormality, and
4-velocity normalization.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from kerr_utils import (
    kerr_metric_components, horizon_radius, ergosphere_radius,
    isco_radius, compute_pt_from_mass_shell, compute_dH_dr_analytic,
    verify_mass_shell, verify_4velocity_normalization,
    apply_exact_impulse, build_rocket_rest_basis,
    compute_optimal_exhaust_direction
)


# =========================================================================
# 1. ISCO radius for Schwarzschild (a=0)
# =========================================================================
def test_isco_schwarzschild():
    r_isco = isco_radius(0, 1.0, prograde=True)
    assert abs(r_isco - 6.0) < 1e-10, (
        f"Schwarzschild ISCO should be 6.0M, got {r_isco}"
    )


# =========================================================================
# 2. ISCO radius for near-extremal prograde
# =========================================================================
def test_isco_extremal_prograde():
    r_isco = isco_radius(0.999, 1.0, prograde=True)
    assert abs(r_isco - 1.0) < 0.2, (
        f"Near-extremal prograde ISCO should be close to 1.0M, got {r_isco}"
    )


# =========================================================================
# 3. ISCO radius for near-extremal retrograde
# =========================================================================
def test_isco_extremal_retrograde():
    r_isco = isco_radius(0.999, 1.0, prograde=False)
    assert abs(r_isco - 9.0) < 0.02, (
        f"Near-extremal retrograde ISCO should be close to 9.0M, got {r_isco}"
    )


# =========================================================================
# 4. Schwarzschild metric components
# =========================================================================
def test_schwarzschild_metric():
    a = 0.0
    M = 1.0
    th = np.pi / 2
    for r in [3.0, 6.0, 10.0, 50.0]:
        cov, _ = kerr_metric_components(r, th, a, M)
        g_tt, g_tphi, g_rr, g_thth, g_phiphi = cov
        expected_gtt = -(1 - 2 * M / r)
        expected_grr = 1.0 / (1 - 2 * M / r)
        assert abs(g_tt - expected_gtt) < 1e-12, (
            f"g_tt at r={r}: expected {expected_gtt}, got {g_tt}"
        )
        assert abs(g_rr - expected_grr) < 1e-12, (
            f"g_rr at r={r}: expected {expected_grr}, got {g_rr}"
        )
        assert abs(g_tphi) < 1e-15, (
            f"g_tphi should be 0 for a=0 at r={r}, got {g_tphi}"
        )


# =========================================================================
# 5. Ergosphere radius at the equator equals 2M for any spin
# =========================================================================
def test_ergosphere_equatorial():
    M = 1.0
    th = np.pi / 2
    for a in [0.0, 0.3, 0.5, 0.7, 0.95, 0.999]:
        r_erg = ergosphere_radius(th, a, M)
        assert abs(r_erg - 2.0 * M) < 1e-12, (
            f"Equatorial ergosphere for a={a} should be 2M={2*M}, got {r_erg}"
        )


# =========================================================================
# 6. Ergosphere at poles coincides with the horizon
# =========================================================================
def test_ergosphere_pole():
    M = 1.0
    th = 0.0  # pole
    for a in [0.0, 0.3, 0.5, 0.7, 0.95, 0.999]:
        r_erg = ergosphere_radius(th, a, M)
        r_hor = horizon_radius(a, M)
        assert abs(r_erg - r_hor) < 1e-12, (
            f"Polar ergosphere for a={a} should equal horizon {r_hor}, got {r_erg}"
        )


# =========================================================================
# Helper: set up geodesic initial conditions and integration
# =========================================================================
def _integrate_geodesic(E0=1.2, Lz0=3.0, a=0.95, M=1.0, r0=10.0,
                        dt=0.005, tau_max=100.0):
    """
    Integrate an equatorial geodesic using Euler steps.

    Returns arrays of (tau, r, pt, pr, pphi) at each step.
    """
    th = np.pi / 2
    _, con = kerr_metric_components(r0, th, a, M)
    gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con

    pt0 = -E0
    pphi0 = Lz0
    m = 1.0

    rhs = -(gu_tt * pt0**2 + 2 * gu_tphi * pt0 * pphi0
            + gu_phiphi * pphi0**2 + m**2)
    assert rhs >= 0, f"Invalid initial conditions: rhs={rhs}"
    pr0 = -np.sqrt(rhs / gu_rr)  # infalling

    # State: [r, phi, pt, pr, pphi, m]
    y = np.array([r0, 0.0, pt0, pr0, pphi0, m])

    n_steps = int(tau_max / dt)
    r_plus = horizon_radius(a, M)
    r_safe = r_plus + 0.02

    taus = [0.0]
    pts = [pt0]
    prs = [pr0]
    pphis = [pphi0]
    rs = [r0]

    tau = 0.0
    for _ in range(n_steps):
        r, phi, pt, pr, pphi, m_cur = y

        if r < r_safe:
            break

        # Contravariant momenta
        _, con = kerr_metric_components(r, th, a, M)
        gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con
        dr = gu_rr * pr / m_cur
        dphi = (gu_tphi * pt + gu_phiphi * pphi) / m_cur

        # Geodesic equation for pr
        dH_dr = compute_dH_dr_analytic(r, th, pt, pr, pphi, m_cur, a, M)
        dpr = -dH_dr / m_cur

        # Euler step
        y = np.array([
            r + dr * dt,
            phi + dphi * dt,
            pt,           # constant of motion
            pr + dpr * dt,
            pphi,         # constant of motion
            m_cur         # no mass loss on geodesic
        ])

        tau += dt
        taus.append(tau)
        rs.append(y[0])
        pts.append(y[2])
        prs.append(y[3])
        pphis.append(y[4])

    return (np.array(taus), np.array(rs), np.array(pts),
            np.array(prs), np.array(pphis))


# =========================================================================
# 7. Energy conservation along a geodesic
# =========================================================================
def test_energy_conservation_geodesic():
    taus, rs, pts, prs, pphis = _integrate_geodesic()
    E = -pts  # Killing energy
    E0 = E[0]
    max_dev = np.max(np.abs(E - E0))
    assert max_dev < 1e-6, (
        f"Energy drift along geodesic: max|dE| = {max_dev:.2e} (should be < 1e-6)"
    )


# =========================================================================
# 8. Angular momentum conservation along a geodesic
# =========================================================================
def test_angular_momentum_conservation_geodesic():
    taus, rs, pts, prs, pphis = _integrate_geodesic()
    Lz0 = pphis[0]
    max_dev = np.max(np.abs(pphis - Lz0))
    assert max_dev < 1e-6, (
        f"Lz drift along geodesic: max|dLz| = {max_dev:.2e} (should be < 1e-6)"
    )


# =========================================================================
# 9. Mass-shell constraint along a geodesic
# =========================================================================
def test_mass_shell_along_geodesic():
    # Use a smaller timestep for tighter mass-shell fidelity
    taus, rs, pts, prs, pphis = _integrate_geodesic(dt=0.001, tau_max=100.0)
    a = 0.95
    M = 1.0
    th = np.pi / 2
    m = 1.0
    max_residual = 0.0
    for i in range(len(taus)):
        p_cov = (pts[i], prs[i], pphis[i])
        is_valid, residual = verify_mass_shell(p_cov, m, rs[i], th, a, M, tol=1e-3)
        if residual > max_residual:
            max_residual = residual
    assert max_residual < 1e-3, (
        f"Mass-shell violation along geodesic: max residual = {max_residual:.2e}"
    )


# =========================================================================
# 10. 4-momentum conservation under an exact impulse
# =========================================================================
def test_4momentum_conservation_impulse():
    a = 0.95
    M = 1.0
    r = 5.0
    th = np.pi / 2

    # Build an initial state at a safe radius
    E0 = 1.05
    Lz0 = 3.5
    m = 1.0
    pt0 = -E0
    pphi0 = Lz0

    _, con = kerr_metric_components(r, th, a, M)
    gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con
    rhs = -(gu_tt * pt0**2 + 2 * gu_tphi * pt0 * pphi0
            + gu_phiphi * pphi0**2 + m**2)
    assert rhs >= 0, f"Invalid IC for impulse test: rhs={rhs}"
    pr0 = -np.sqrt(rhs / gu_rr)

    p_cov_old = (pt0, pr0, pphi0)

    # Build 4-velocity (contravariant)
    u_t = (gu_tt * pt0 + gu_tphi * pphi0) / m
    u_r = gu_rr * pr0 / m
    u_phi = (gu_tphi * pt0 + gu_phiphi * pphi0) / m
    u_vec = np.array([u_t, u_r, u_phi])

    # Get metric covariant components
    cov, _ = kerr_metric_components(r, th, a, M)
    g_tt, g_tphi, g_rr, _, g_phiphi = cov

    # Optimal exhaust direction
    v_e = 0.5
    opt = compute_optimal_exhaust_direction(
        u_vec, r, th, a, M, v_e,
        g_tt, g_tphi, g_rr, g_phiphi
    )
    assert opt is not None, "compute_optimal_exhaust_direction returned None"

    u_ex_cov = opt['u_ex_cov']
    delta_mu = 0.05 * m  # small impulse to stay physical

    # Apply impulse
    result = apply_exact_impulse(p_cov_old, m, delta_mu, u_ex_cov, r, th, a, M)
    p_cov_new = result['p_cov_new']

    # Verify: p_new_mu = p_old_mu - delta_mu * u_ex_mu
    for i, label in enumerate(['t', 'r', 'phi']):
        expected = p_cov_old[i] - delta_mu * u_ex_cov[i]
        residual = abs(p_cov_new[i] - expected)
        assert residual < 1e-10, (
            f"4-momentum conservation violated in {label}: "
            f"residual = {residual:.2e}"
        )


# =========================================================================
# 11. Tetrad orthonormality
# =========================================================================
def test_tetrad_orthonormality():
    a = 0.95
    M = 1.0
    r = 5.0
    th = np.pi / 2

    # Build a valid 4-velocity
    E0 = 1.05
    Lz0 = 3.5
    m = 1.0
    pt0 = -E0
    pphi0 = Lz0

    cov, con = kerr_metric_components(r, th, a, M)
    g_tt, g_tphi, g_rr, _, g_phiphi = cov
    gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con

    rhs = -(gu_tt * pt0**2 + 2 * gu_tphi * pt0 * pphi0
            + gu_phiphi * pphi0**2 + m**2)
    assert rhs >= 0, f"Invalid IC for tetrad test: rhs={rhs}"
    pr0 = -np.sqrt(rhs / gu_rr)

    u_t = (gu_tt * pt0 + gu_tphi * pphi0) / m
    u_r = gu_rr * pr0 / m
    u_phi = (gu_tphi * pt0 + gu_phiphi * pphi0) / m
    u_vec = np.array([u_t, u_r, u_phi])

    def inner(v, w):
        """Metric inner product in (t, r, phi) subspace."""
        return (g_tt * v[0] * w[0]
                + g_rr * v[1] * w[1]
                + g_phiphi * v[2] * w[2]
                + g_tphi * (v[0] * w[2] + v[2] * w[0]))

    e_r, e_phi = build_rocket_rest_basis(u_vec, g_tt, g_tphi, g_rr, g_phiphi)
    assert e_r is not None and e_phi is not None, "Tetrad construction failed"

    # e_r . e_r = 1 (spacelike, unit norm)
    er_er = inner(e_r, e_r)
    assert abs(er_er - 1.0) < 1e-10, (
        f"e_r not unit-norm: e_r.e_r = {er_er}"
    )

    # e_phi . e_phi = 1
    ephi_ephi = inner(e_phi, e_phi)
    assert abs(ephi_ephi - 1.0) < 1e-10, (
        f"e_phi not unit-norm: e_phi.e_phi = {ephi_ephi}"
    )

    # e_r . e_phi = 0 (orthogonal)
    er_ephi = inner(e_r, e_phi)
    assert abs(er_ephi) < 1e-10, (
        f"e_r and e_phi not orthogonal: e_r.e_phi = {er_ephi}"
    )

    # e_r . u = 0 (spatial in rest frame)
    er_u = inner(e_r, u_vec)
    assert abs(er_u) < 1e-10, (
        f"e_r not orthogonal to u: e_r.u = {er_u}"
    )

    # e_phi . u = 0
    ephi_u = inner(e_phi, u_vec)
    assert abs(ephi_u) < 1e-10, (
        f"e_phi not orthogonal to u: e_phi.u = {ephi_u}"
    )


# =========================================================================
# 12. 4-velocity normalization
# =========================================================================
def test_4velocity_normalization():
    a = 0.95
    M = 1.0
    r = 8.0
    th = np.pi / 2
    m = 1.0

    pphi = 3.0
    pr = 0.0
    pt = compute_pt_from_mass_shell(r, th, pr, pphi, m, a, M)

    cov, con = kerr_metric_components(r, th, a, M)
    g_tt, g_tphi, g_rr, _, g_phiphi = cov
    gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con

    # Contravariant 4-velocity: u^mu = g^{mu nu} p_nu / m
    u_t = (gu_tt * pt + gu_tphi * pphi) / m
    u_r = gu_rr * pr / m
    u_phi = (gu_tphi * pt + gu_phiphi * pphi) / m
    u_vec = np.array([u_t, u_r, u_phi])

    # g_{mu nu} u^mu u^nu should be -1
    norm = (g_tt * u_t**2
            + g_rr * u_r**2
            + g_phiphi * u_phi**2
            + 2 * g_tphi * u_t * u_phi)
    assert abs(norm + 1.0) < 1e-10, (
        f"4-velocity not normalised: g_munu u^mu u^nu = {norm} (expected -1)"
    )


# =========================================================================
# Main runner
# =========================================================================
if __name__ == '__main__':
    tests = [
        ("test_isco_schwarzschild", test_isco_schwarzschild),
        ("test_isco_extremal_prograde", test_isco_extremal_prograde),
        ("test_isco_extremal_retrograde", test_isco_extremal_retrograde),
        ("test_schwarzschild_metric", test_schwarzschild_metric),
        ("test_ergosphere_equatorial", test_ergosphere_equatorial),
        ("test_ergosphere_pole", test_ergosphere_pole),
        ("test_energy_conservation_geodesic", test_energy_conservation_geodesic),
        ("test_angular_momentum_conservation_geodesic",
         test_angular_momentum_conservation_geodesic),
        ("test_mass_shell_along_geodesic", test_mass_shell_along_geodesic),
        ("test_4momentum_conservation_impulse",
         test_4momentum_conservation_impulse),
        ("test_tetrad_orthonormality", test_tetrad_orthonormality),
        ("test_4velocity_normalization", test_4velocity_normalization),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    print()
    print(f"{passed} passed, {failed} failed out of {len(tests)} tests")
    if failed:
        sys.exit(1)
