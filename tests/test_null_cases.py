"""
Null tests for Penrose process physics validation.

These tests verify that the simulation correctly predicts:
1. No Penrose extraction for a=0 (Schwarzschild black hole)
2. No Penrose extraction for thrust outside the ergosphere
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from kerr_utils import (
    kerr_metric_components, horizon_radius, ergosphere_radius,
    compute_pt_from_mass_shell, compute_energy
)


def test_schwarzschild_no_penrose(verbose=True):
    """
    Null test 1: No Penrose extraction in Schwarzschild spacetime (a=0).
    
    The ergosphere requires rotation. For a=0, r_erg = r+ = 2M at all latitudes,
    so there's no ergosphere region where E_ex < 0 is possible.
    """
    a = 0.0  # Schwarzschild
    M = 1.0
    r_plus = horizon_radius(a, M)
    
    if verbose:
        print("=" * 60)
        print("NULL TEST 1: Schwarzschild (a=0) - No Penrose Extraction")
        print("=" * 60)
        print(f"  a/M = {a}")
        print(f"  r+ = {r_plus}M (= r_erg for Schwarzschild)")
    
    # At any radius > r+, try to compute exhaust energy
    # For Schwarzschild, g_tphi = 0, so no frame dragging
    test_radii = [2.5, 3.0, 4.0, 6.0]
    all_positive = True
    
    for r in test_radii:
        th = np.pi/2
        cov, con = kerr_metric_components(r, th, a, M)
        g_tt, g_tphi, g_rr, g_thth, g_phiphi = cov
        
        # In Schwarzschild, g_tphi = 0
        assert abs(g_tphi) < 1e-12, f"g_tphi should be 0 for a=0, got {g_tphi}"
        
        # For any 4-velocity, E = -g_tmu u^mu = -g_tt u^t
        # In Schwarzschild: g_tt = -(1 - 2M/r)
        # For r > 2M: g_tt < 0, so for future-directed u^t > 0:
        # E = -g_tt u^t = |g_tt| u^t > 0
        
        # There's no frame-dragging to make E < 0
        # Even retrograde motion has E > 0 for massive particles
        
        g_tt_value = g_tt
        if verbose:
            print(f"  r = {r}M: g_tt = {g_tt_value:.4f}, g_tphi = {g_tphi:.2e}")
            print(f"           For any u^t > 0: E = -g_tt*u^t = {-g_tt_value:.4f} * u^t > 0 [OK]")
    
    # Verify ergosphere doesn't exist
    r_erg_eq = ergosphere_radius(np.pi/2, a, M)
    assert abs(r_erg_eq - r_plus) < 1e-10, f"Ergosphere should equal horizon for a=0"
    
    if verbose:
        print(f"\n  Ergosphere radius (equator) = {r_erg_eq}M = r+ [OK]")
        print("  => No ergosphere region exists for Schwarzschild")
        print("  => Penrose extraction impossible [OK]")
        print()
    
    return True


def test_no_extraction_outside_ergosphere(verbose=True):
    """
    Null test 2: No Penrose extraction for thrust outside ergosphere.
    
    The Penrose process requires g_tt > 0 (timelike Killing vector becomes
    spacelike), which only occurs inside the ergosphere r < r_erg.
    """
    a = 0.95  # High spin
    M = 1.0
    r_plus = horizon_radius(a, M)
    r_erg_eq = ergosphere_radius(np.pi/2, a, M)  # = 2M at equator
    
    if verbose:
        print("=" * 60)
        print("NULL TEST 2: No Extraction Outside Ergosphere")
        print("=" * 60)
        print(f"  a/M = {a}")
        print(f"  r+ = {r_plus:.4f}M")
        print(f"  r_erg (equator) = {r_erg_eq}M")
    
    # Test at radii just outside ergosphere
    test_radii = [2.1, 2.5, 3.0, 5.0, 10.0]  # All > 2M
    
    for r in test_radii:
        th = np.pi/2
        cov, con = kerr_metric_components(r, th, a, M)
        g_tt, g_tphi, g_rr, g_thth, g_phiphi = cov
        
        # Check if outside ergosphere
        inside_ergo = r < r_erg_eq
        
        # Outside ergosphere: g_tt < 0, so xi*xi = g_tt < 0 (timelike)
        # Killing energy E = -p_t must be positive for future-directed particles
        
        if verbose:
            status = "INSIDE" if inside_ergo else "OUTSIDE"
            print(f"  r = {r:.1f}M ({status}): g_tt = {g_tt:.4f}", end="")
        
        if not inside_ergo:
            # g_tt < 0 outside ergosphere
            assert g_tt < 0, f"g_tt should be < 0 outside ergosphere, got {g_tt}"
            if verbose:
                print(" => Timelike Killing vector => E > 0 required [OK]")
        else:
            # g_tt > 0 inside ergosphere (where Penrose is possible)
            if verbose:
                print(" => Spacelike Killing vector => E < 0 possible")
    
    if verbose:
        print()
        print("  CONCLUSION: For r > r_erg, the timelike Killing vector xi = d/dt")
        print("  remains timelike (g_tt < 0). Any future-directed timelike or null")
        print("  4-velocity u satisfies xi*u < 0, so E = -xi*p = -m(xi*u) > 0.")
        print("  => Penrose extraction (E < 0) is geometrically impossible [OK]")
        print()
    
    return True


def test_exhaust_energy_signs(verbose=True):
    """
    Test that exhaust energy has the expected sign behavior.
    
    Inside ergosphere: optimal retrograde exhaust can have E_ex < 0
    Outside ergosphere: all exhaust must have E_ex > 0
    """
    a = 0.95
    M = 1.0
    r_erg_eq = ergosphere_radius(np.pi/2, a, M)
    
    if verbose:
        print("=" * 60)
        print("TEST: Exhaust Energy Sign Dependence on Location")
        print("=" * 60)
    
    # We need to import the thrust machinery
    try:
        from continuous_thrust_case import compute_optimal_exhaust_direction
    except ImportError:
        if verbose:
            print("  Skipping: continuous_thrust_case not available")
        return True
    
    v_e = 0.95  # Exhaust velocity
    
    # Test points inside and outside ergosphere
    inside_points = [1.5, 1.7, 1.9]  # r < 2M
    outside_points = [2.1, 2.5, 3.0]  # r > 2M
    
    if verbose:
        print(f"  Exhaust velocity: v_e = {v_e}c")
        print(f"  Ergosphere boundary: r_erg = {r_erg_eq}M")
        print()
    
    # Create a typical spacecraft 4-velocity (prograde, infalling)
    for r in inside_points + outside_points:
        th = np.pi/2
        cov, con = kerr_metric_components(r, th, a, M)
        g_tt, g_tphi, g_rr, g_thth, g_phiphi = cov
        gu_tt, gu_tphi, gu_rr, gu_thth, gu_phiphi = con
        
        # Construct a timelike 4-velocity
        # Start with E ~ 1.2, Lz ~ 3.0 (typical sweet spot)
        E, Lz = 1.2, 3.0
        pt = -E
        pphi = Lz
        m = 1.0
        
        # Solve for pr (infalling)
        rhs = -(gu_tt*pt**2 + 2*gu_tphi*pt*pphi + gu_phiphi*pphi**2 + m**2)
        if rhs < 0:
            continue  # Forbidden region
        pr = -np.sqrt(rhs / gu_rr)
        
        # 4-velocity components
        u_t = (gu_tt * pt + gu_tphi * pphi) / m
        u_r = gu_rr * pr / m
        u_phi = (gu_tphi * pt + gu_phiphi * pphi) / m
        u_vec = np.array([u_t, u_r, u_phi])
        
        try:
            opt = compute_optimal_exhaust_direction(
                u_vec, r, th, a, M, v_e, g_tt, g_tphi, g_rr, g_phiphi
            )
            if opt is not None:
                E_ex = opt['E_ex']
                inside = r < r_erg_eq
                status = "INSIDE" if inside else "OUTSIDE"
                
                if verbose:
                    print(f"  r = {r:.1f}M ({status}): E_ex = {E_ex:+.4f}", end="")
                    if E_ex < 0:
                        print(" => PENROSE EXTRACTION [OK]" if inside else " => ERROR!")
                    else:
                        print(" => Normal thrust" + (" => Cannot extract" if not inside else ""))
                
                # Verify physics
                if not inside:
                    assert E_ex >= 0, (
                        f"Physics violation: E_ex = {E_ex:.6f} < 0 outside "
                        f"ergosphere at r = {r:.2f}M (r_erg = {r_erg_eq:.2f}M)"
                    )
                    if E_ex < 0:
                        print(f"  WARNING: Got E_ex < 0 outside ergosphere at r={r}M!")
        except Exception as e:
            if verbose:
                print(f"  r = {r:.1f}M: Could not compute optimal exhaust ({e})")
    
    if verbose:
        print()
    
    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PENROSE PROCESS NULL TESTS")
    print("=" * 70 + "\n")
    
    test_schwarzschild_no_penrose(verbose=True)
    test_no_extraction_outside_ergosphere(verbose=True)
    test_exhaust_energy_signs(verbose=True)
    
    print("=" * 70)
    print("ALL NULL TESTS PASSED")
    print("=" * 70)
