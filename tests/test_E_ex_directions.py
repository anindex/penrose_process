
"""Test which thrust direction gives E_ex < 0."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from kerr_utils import (
    kerr_metric_components, horizon_radius,
    build_rocket_rest_basis, compute_exhaust_4velocity
)

a = 0.95
M = 1.0
ve = 0.95


def scan_directions(r, E0=1.2, Lz=3.0, verbose=True):
    """Scan thrust directions at radius r and return best E_ex and per-direction results.

    Returns a dict with keys:
        'best_E_ex': float -- minimum E_ex found over all scanned angles
        'best_config': (alpha, sign) -- angle and sign for the best direction
        'pure_directions': dict mapping direction name to E_ex value
        'valid': bool -- whether a valid trajectory exists at this r
    """
    th = np.pi / 2

    # Get metric at this radius
    cov, con = kerr_metric_components(r, th, a, M)
    g_tt, g_tphi, g_rr, _, g_phiphi = cov
    gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con

    # Set up 4-velocity similar to single-thrust case
    pt = -E0
    pphi = Lz
    m = 1.0

    # Get pr from mass-shell at this radius
    rhs = -(gu_tt * pt**2 + 2 * gu_tphi * pt * pphi + gu_phiphi * pphi**2 + m**2)
    if rhs < 0:
        if verbose:
            print(f"r = {r:.2f}M: No valid trajectory")
        return {'valid': False, 'best_E_ex': None, 'best_config': None, 'pure_directions': {}}

    pr = -np.sqrt(rhs / gu_rr)  # inbound

    u_t = (gu_tt * pt + gu_tphi * pphi) / m
    u_r = gu_rr * pr / m
    u_phi = (gu_tphi * pt + gu_phiphi * pphi) / m
    u_vec = np.array([u_t, u_r, u_phi])

    if verbose:
        print(f"\n=== r = {r:.2f}M ===")
        print(f"4-velocity: u = ({u_t:.3f}, {u_r:.3f}, {u_phi:.3f})")

    # Build basis
    e_r, e_phi = build_rocket_rest_basis(u_vec, g_tt, g_tphi, g_rr, g_phiphi)
    if e_r is None or e_phi is None:
        if verbose:
            print("Failed to build basis")
        return {'valid': False, 'best_E_ex': None, 'best_config': None, 'pure_directions': {}}

    # Test pure directions
    pure_directions = {}
    if verbose:
        print("Testing pure directions:")
    for name, s_vec in [('+e_r', e_r), ('-e_r', -e_r), ('+e_phi', e_phi), ('-e_phi', -e_phi)]:
        result = compute_exhaust_4velocity(u_vec, s_vec, ve, g_tt, g_tphi, g_rr, g_phiphi)
        pure_directions[name] = result['E_ex']
        if verbose:
            marker = "[OK]" if result['E_ex'] < 0 else ""
            print(f"  {name:8}: E_ex = {result['E_ex']:+.4f} {marker}")

    # Scan all angles
    best_E_ex = float('inf')
    best_config = None
    for alpha in np.linspace(-np.pi / 2, np.pi / 2, 37):
        for sign in [+1, -1]:
            s_vec = np.sin(alpha) * e_r + (sign * np.cos(alpha)) * e_phi
            result = compute_exhaust_4velocity(u_vec, s_vec, ve, g_tt, g_tphi, g_rr, g_phiphi)
            if result['E_ex'] < best_E_ex:
                best_E_ex = result['E_ex']
                best_config = (alpha, sign)

    if verbose:
        marker = "[OK] E_ex < 0!" if best_E_ex < 0 else ""
        print(f"Best: E_ex = {best_E_ex:+.4f} at alpha={np.degrees(best_config[0]):+.1f}deg, sign={best_config[1]:+.0f} {marker}")

    return {
        'valid': True,
        'best_E_ex': best_E_ex,
        'best_config': best_config,
        'pure_directions': pure_directions,
    }


def test_E_ex_negative_in_ergosphere():
    """For radii inside the ergosphere (r < 2.0M for a=0.95), at least one direction
    should give E_ex < 0, which is the necessary condition for Penrose extraction."""
    ergosphere_radii = [1.71, 1.51, 1.40, 1.35]
    for r in ergosphere_radii:
        assert r < 2.0 * M, f"Test radius r={r} is not inside the ergosphere"
        info = scan_directions(r, verbose=False)
        if not info['valid']:
            # Skip radii where no valid trajectory exists
            continue
        # At least one pure direction or the best scan angle should give E_ex < 0
        has_negative = any(v < 0 for v in info['pure_directions'].values())
        if not has_negative:
            has_negative = info['best_E_ex'] < 0
        assert has_negative, (
            f"At r={r:.2f}M (inside ergosphere), no direction gave E_ex < 0. "
            f"Best E_ex = {info['best_E_ex']:.6f}"
        )


def test_E_ex_best_direction():
    """For deep ergosphere radii (r < 1.5M), the best scanned direction must give E_ex < 0."""
    deep_radii = [1.40, 1.35]
    for r in deep_radii:
        info = scan_directions(r, verbose=False)
        if not info['valid']:
            continue
        assert info['best_E_ex'] < 0, (
            f"At r={r:.2f}M (deep ergosphere), best E_ex = {info['best_E_ex']:.6f} "
            f"should be negative"
        )


if __name__ == '__main__':
    # Run the original exploratory scan with verbose output
    for r in [1.71, 1.51, 1.40, 1.35]:
        scan_directions(r, verbose=True)

    print()
    print("Running assertions...")
    test_E_ex_negative_in_ergosphere()
    print("  test_E_ex_negative_in_ergosphere: PASSED")
    test_E_ex_best_direction()
    print("  test_E_ex_best_direction: PASSED")
    print("All tests passed.")
