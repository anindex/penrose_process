
"""Test which thrust direction gives E_ex < 0."""

import numpy as np
from kerr_utils import (
    kerr_metric_components, horizon_radius,
    build_rocket_rest_basis, compute_exhaust_4velocity
)

a = 0.95
M = 1.0
ve = 0.95

# Test at multiple radii
for r in [1.71, 1.51, 1.40, 1.35]:
    th = np.pi/2
    
    # Get metric at this radius
    cov, con = kerr_metric_components(r, th, a, M)
    g_tt, g_tphi, g_rr, _, g_phiphi = cov
    gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con
    
    # Set up 4-velocity similar to single-thrust case
    E0, Lz = 1.2, 3.0
    pt = -E0
    pphi = Lz
    m = 1.0
    
    # Get pr from mass-shell at this radius
    rhs = -(gu_tt * pt**2 + 2 * gu_tphi * pt * pphi + gu_phiphi * pphi**2 + m**2)
    if rhs < 0:
        print(f"r = {r:.2f}M: No valid trajectory")
        continue
    
    pr = -np.sqrt(rhs / gu_rr)  # inbound
    
    u_t = (gu_tt * pt + gu_tphi * pphi) / m
    u_r = gu_rr * pr / m
    u_phi = (gu_tphi * pt + gu_phiphi * pphi) / m
    u_vec = np.array([u_t, u_r, u_phi])
    
    print(f"\n=== r = {r:.2f}M ===")
    print(f"4-velocity: u = ({u_t:.3f}, {u_r:.3f}, {u_phi:.3f})")
    
    # Build basis
    e_r, e_phi = build_rocket_rest_basis(u_vec, g_tt, g_tphi, g_rr, g_phiphi)
    if e_r is None or e_phi is None:
        print("Failed to build basis")
        continue
    
    # Test different directions
    print("Testing pure directions:")
    for name, s_vec in [('+e_r', e_r), ('-e_r', -e_r), ('+e_phi', e_phi), ('-e_phi', -e_phi)]:
        result = compute_exhaust_4velocity(u_vec, s_vec, ve, g_tt, g_tphi, g_rr, g_phiphi)
        marker = "[OK]" if result['E_ex'] < 0 else ""
        print(f"  {name:8}: E_ex = {result['E_ex']:+.4f} {marker}")
    
    # Scan all angles
    best_E_ex = float('inf')
    best_config = None
    for alpha in np.linspace(-np.pi/2, np.pi/2, 37):
        for sign in [+1, -1]:
            s_vec = np.sin(alpha) * e_r + (sign * np.cos(alpha)) * e_phi
            result = compute_exhaust_4velocity(u_vec, s_vec, ve, g_tt, g_tphi, g_rr, g_phiphi)
            if result['E_ex'] < best_E_ex:
                best_E_ex = result['E_ex']
                best_config = (alpha, sign)
    
    marker = "[OK] E_ex < 0!" if best_E_ex < 0 else ""
    print(f"Best: E_ex = {best_E_ex:+.4f} at alpha={np.degrees(best_config[0]):+.1f}deg, sign={best_config[1]:+.0f} {marker}")
