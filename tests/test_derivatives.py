"""
Unit Tests for Kerr Metric Derivatives
=======================================
Compares analytic derivatives against finite-difference approximations
to validate correctness, especially near the horizon where errors matter most.
"""

import numpy as np
import pytest
from kerr_utils import (
    kerr_metric_components, kerr_metric_derivatives, 
    horizon_radius, ergosphere_radius
)


# Test parameters
M = 1.0
SPINS = [0.5, 0.9, 0.95, 0.99]  # Range of spin parameters
THETA = np.pi / 2  # Equatorial plane (where code operates)
EPSILON = 1e-7  # Finite difference step


def finite_difference_derivatives(r, th, a, M=1.0, eps=EPSILON):
    """
    Compute inverse metric derivatives using central finite differences.
    
    Returns dict with keys: 'dgu_tt', 'dgu_tphi', 'dgu_rr', 'dgu_phiphi'
    """
    # Get metric components at r-eps, r, r+eps
    _, con_minus = kerr_metric_components(r - eps, th, a, M)
    _, con_plus = kerr_metric_components(r + eps, th, a, M)
    
    gu_tt_m, gu_tphi_m, gu_rr_m, _, gu_phiphi_m = con_minus
    gu_tt_p, gu_tphi_p, gu_rr_p, _, gu_phiphi_p = con_plus
    
    return {
        'dgu_tt': (gu_tt_p - gu_tt_m) / (2 * eps),
        'dgu_tphi': (gu_tphi_p - gu_tphi_m) / (2 * eps),
        'dgu_rr': (gu_rr_p - gu_rr_m) / (2 * eps),
        'dgu_phiphi': (gu_phiphi_p - gu_phiphi_m) / (2 * eps),
    }


def get_test_radii(a, M=1.0, n_radii=20):
    """
    Generate test radii spanning from just outside horizon to ergosphere.
    
    This is the critical region for Penrose extraction.
    """
    r_plus = horizon_radius(a, M)
    r_erg = ergosphere_radius(THETA, a, M)
    
    # Start slightly outside horizon (avoid numerical issues at horizon)
    r_min = r_plus + 0.02
    r_max = r_erg + 0.5  # Include region slightly outside ergosphere
    
    return np.linspace(r_min, r_max, n_radii)


class TestMetricDerivatives:
    """Test suite for Kerr metric derivative calculations."""
    
    @pytest.mark.parametrize("a", SPINS)
    def test_dgu_tphi_matches_finite_difference(self, a):
        """
        CRITICAL TEST: Verify dg^{tphi}/dr matches finite difference.
        
        This was the source of a major bug where the analytic formula
        used (3r^2 - 2Mr - a^2) instead of correct (3r^2 - 4Mr + a^2).
        """
        radii = get_test_radii(a)
        max_rel_error = 0.0
        worst_r = None
        
        for r in radii:
            analytic = kerr_metric_derivatives(r, THETA, a, M)
            numeric = finite_difference_derivatives(r, THETA, a, M)
            
            ana_val = analytic['dgu_tphi']
            num_val = numeric['dgu_tphi']
            
            # Relative error (handle near-zero values)
            if abs(num_val) > 1e-10:
                rel_error = abs(ana_val - num_val) / abs(num_val)
            else:
                rel_error = abs(ana_val - num_val)
            
            if rel_error > max_rel_error:
                max_rel_error = rel_error
                worst_r = r
        
        # Should match to better than 0.1% (finite diff has O(eps^2) error)
        assert max_rel_error < 1e-3, (
            f"dgu_tphi mismatch for a={a}: max relative error = {max_rel_error:.2e} "
            f"at r = {worst_r:.4f}M"
        )
    
    @pytest.mark.parametrize("a", SPINS)
    def test_dgu_tt_matches_finite_difference(self, a):
        """Verify dg^{tt}/dr matches finite difference."""
        radii = get_test_radii(a)
        
        for r in radii:
            analytic = kerr_metric_derivatives(r, THETA, a, M)
            numeric = finite_difference_derivatives(r, THETA, a, M)
            
            ana_val = analytic['dgu_tt']
            num_val = numeric['dgu_tt']
            
            if abs(num_val) > 1e-10:
                rel_error = abs(ana_val - num_val) / abs(num_val)
                assert rel_error < 1e-3, (
                    f"dgu_tt mismatch at r={r:.4f}, a={a}: "
                    f"analytic={ana_val:.6e}, numeric={num_val:.6e}"
                )
    
    @pytest.mark.parametrize("a", SPINS)
    def test_dgu_rr_matches_finite_difference(self, a):
        """Verify dg^{rr}/dr matches finite difference."""
        radii = get_test_radii(a)
        
        for r in radii:
            analytic = kerr_metric_derivatives(r, THETA, a, M)
            numeric = finite_difference_derivatives(r, THETA, a, M)
            
            ana_val = analytic['dgu_rr']
            num_val = numeric['dgu_rr']
            
            if abs(num_val) > 1e-10:
                rel_error = abs(ana_val - num_val) / abs(num_val)
                assert rel_error < 1e-3, (
                    f"dgu_rr mismatch at r={r:.4f}, a={a}: "
                    f"analytic={ana_val:.6e}, numeric={num_val:.6e}"
                )
    
    @pytest.mark.parametrize("a", SPINS)
    def test_dgu_phiphi_matches_finite_difference(self, a):
        """Verify dg^{phiphi}/dr matches finite difference."""
        radii = get_test_radii(a)
        
        for r in radii:
            analytic = kerr_metric_derivatives(r, THETA, a, M)
            numeric = finite_difference_derivatives(r, THETA, a, M)
            
            ana_val = analytic['dgu_phiphi']
            num_val = numeric['dgu_phiphi']
            
            if abs(num_val) > 1e-10:
                rel_error = abs(ana_val - num_val) / abs(num_val)
                assert rel_error < 1e-3, (
                    f"dgu_phiphi mismatch at r={r:.4f}, a={a}: "
                    f"analytic={ana_val:.6e}, numeric={num_val:.6e}"
                )
    
    def test_near_horizon_accuracy(self):
        """
        Test derivative accuracy very close to horizon.
        
        This is where the original bug had the largest impact.
        """
        a = 0.95
        r_plus = horizon_radius(a, M)
        
        # Test at progressively closer distances to horizon
        offsets = [0.1, 0.05, 0.02, 0.01]
        
        for dr in offsets:
            r = r_plus + dr
            analytic = kerr_metric_derivatives(r, THETA, a, M)
            numeric = finite_difference_derivatives(r, THETA, a, M, eps=dr/10)
            
            for key in ['dgu_tt', 'dgu_tphi', 'dgu_rr', 'dgu_phiphi']:
                ana_val = analytic[key]
                num_val = numeric[key]
                
                if abs(num_val) > 1e-8:
                    rel_error = abs(ana_val - num_val) / abs(num_val)
                    # Allow 2% tolerance very near horizon (finite diff has O(eps^2) error)
                    # The finite-difference eps scales with dr, so near horizon the error grows
                    assert rel_error < 0.02, (
                        f"{key} mismatch at r=r_+ + {dr}: "
                        f"rel_error = {rel_error:.2e}"
                    )


class TestDerivativeFormulas:
    """Verify the mathematical form of derivative expressions."""
    
    def test_dgu_tphi_formula_structure(self):
        """
        Verify the correct formula structure for dg^{tphi}/dr.
        
        At equator: g^{tphi} = -2Mar / (r^2Delta)
        
        The derivative should be: 2Ma(3r^2 - 4Mr + a^2) / (r^2Delta^2)
        
        NOT the incorrect: 2Ma(3r^2 - 2Mr - a^2) / (r^2Delta^2)
        """
        a = 0.95
        r = 1.5  # Inside ergosphere
        M = 1.0
        
        Delta = r**2 - 2*M*r + a**2
        
        # Correct formula
        correct_numerator = 2*M*a * (3*r**2 - 4*M*r + a**2)
        correct_deriv = correct_numerator / (r**2 * Delta**2)
        
        # Wrong formula (the bug)
        wrong_numerator = 2*M*a * (3*r**2 - 2*M*r - a**2)
        wrong_deriv = wrong_numerator / (r**2 * Delta**2)
        
        # Get what the code computes
        computed = kerr_metric_derivatives(r, np.pi/2, a, M)
        
        # Code should match correct formula
        assert np.isclose(computed['dgu_tphi'], correct_deriv, rtol=1e-10), (
            f"Code uses wrong formula! Got {computed['dgu_tphi']:.6e}, "
            f"expected {correct_deriv:.6e}"
        )
        
        # And should NOT match the wrong formula
        assert not np.isclose(computed['dgu_tphi'], wrong_deriv, rtol=0.1), (
            "Code is still using the incorrect formula!"
        )


def run_visual_comparison():
    """
    Generate visual comparison of analytic vs numeric derivatives.
    
    Run this directly for debugging: python test_derivatives.py
    """
    import matplotlib.pyplot as plt
    
    a = 0.95
    radii = get_test_radii(a, n_radii=50)
    
    ana_tphi = []
    num_tphi = []
    
    for r in radii:
        analytic = kerr_metric_derivatives(r, THETA, a, M)
        numeric = finite_difference_derivatives(r, THETA, a, M)
        ana_tphi.append(analytic['dgu_tphi'])
        num_tphi.append(numeric['dgu_tphi'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot derivatives
    ax1.plot(radii, ana_tphi, 'b-', label='Analytic', lw=2)
    ax1.plot(radii, num_tphi, 'r--', label='Finite Difference', lw=2)
    ax1.axvline(horizon_radius(a), color='k', ls=':', label='Horizon')
    ax1.axvline(ergosphere_radius(THETA, a), color='orange', ls=':', label='Ergosphere')
    ax1.set_xlabel('r/M')
    ax1.set_ylabel(r'$\partial g^{t\phi}/\partial r$')
    ax1.legend()
    ax1.set_title(f'dg^{{tphi}}/dr comparison (a={a})')
    
    # Plot relative error
    rel_errors = [abs(a - n) / abs(n) if abs(n) > 1e-10 else 0 
                  for a, n in zip(ana_tphi, num_tphi)]
    ax2.semilogy(radii, rel_errors, 'g-', lw=2)
    ax2.axhline(1e-3, color='r', ls='--', label='1e-3 threshold')
    ax2.axvline(horizon_radius(a), color='k', ls=':')
    ax2.axvline(ergosphere_radius(THETA, a), color='orange', ls=':')
    ax2.set_xlabel('r/M')
    ax2.set_ylabel('Relative Error')
    ax2.legend()
    ax2.set_title('Relative Error (Analytic vs Finite Diff)')
    
    plt.tight_layout()
    plt.savefig('derivative_validation.png', dpi=150)
    plt.show()
    print("Saved: derivative_validation.png")


if __name__ == '__main__':
    # Run pytest when executed directly
    import sys
    
    if '--visual' in sys.argv:
        run_visual_comparison()
    else:
        pytest.main([__file__, '-v'])
