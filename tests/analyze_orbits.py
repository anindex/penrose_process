"""
Analyze orbital dynamics to find flyby configurations.

The key insight: for escape, the orbit must have a TURNING POINT (periapsis)
where pr = 0 and dr/dtau changes sign. Without thrust, orbits with periapsis
inside ergosphere will naturally escape.

With thrust applied AT periapsis (single impulse), we can boost escape.
With continuous thrust, we need to ensure the orbit doesn't plunge.
"""

import numpy as np
from kerr_utils import kerr_metric_components, horizon_radius, ergosphere_radius

M = 1.0
a = 0.95
r_plus = horizon_radius(a, M)
r_erg = 2.0

def compute_effective_potential_radial(E, Lz, r, a=0.95, M=1.0):
    """
    Compute V_eff for radial motion at equator.
    
    For timelike geodesics: pr^2 = -V_eff(r) where
    V_eff = g^rr * [g^tt*pt^2 + 2*g^tphi*pt*pphi + g^phiphi*pphi^2 + 1]
    
    With pt = -E and pphi = Lz:
    V_eff = g^rr * [g^tt*E^2 - 2*g^tphi*E*Lz + g^phiphi*Lz^2 + 1]
    
    Turning point: V_eff = 0 with dV/dr > 0 (periapsis)
    """
    th = np.pi/2
    _, con = kerr_metric_components(r, th, a, M)
    gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con
    
    # V_eff / g^rr (since g^rr > 0 outside horizon)
    V = gu_tt * E**2 - 2*gu_tphi * E * Lz + gu_phiphi * Lz**2 + 1.0
    return V  # pr^2 = -g^rr * V, so V < 0 means pr^2 > 0 (motion allowed)


def find_periapsis(E, Lz, a=0.95, M=1.0):
    """Find periapsis radius where V_eff = 0 and orbit turns around."""
    from scipy.optimize import brentq
    
    r_min = r_plus + 0.01
    r_max = 20.0
    
    # Sample to find sign changes
    radii = np.linspace(r_min, r_max, 500)
    V = np.array([compute_effective_potential_radial(E, Lz, r, a, M) for r in radii])
    
    # Find where V crosses zero from negative to positive (periapsis)
    periapses = []
    for i in range(len(V)-1):
        if V[i] <= 0 and V[i+1] > 0:  # Moving from allowed to forbidden
            try:
                r_peri = brentq(lambda r: compute_effective_potential_radial(E, Lz, r, a, M), 
                               radii[i], radii[i+1])
                periapses.append(r_peri)
            except:
                pass
    
    return periapses


def check_orbit_type(E, Lz, a=0.95, M=1.0):
    """Classify orbit type."""
    periapses = find_periapsis(E, Lz, a, M)
    
    if not periapses:
        # Check if V_eff < 0 all the way to horizon (plunge orbit)
        V_near_horizon = compute_effective_potential_radial(E, Lz, r_plus + 0.1, a, M)
        if V_near_horizon < 0:
            return 'plunge', None
        else:
            return 'forbidden', None
    
    r_peri = periapses[0]  # Innermost periapsis
    
    # Check if periapsis is outside horizon
    if r_peri > r_plus:
        if r_peri < r_erg:
            return 'flyby_in_ergosphere', r_peri
        else:
            return 'flyby_outside_ergosphere', r_peri
    else:
        return 'plunge', None


print("="*70)
print("ORBIT CLASSIFICATION: Finding Flyby Orbits")
print("="*70)
print(f"Horizon: r+ = {r_plus:.4f}M")
print(f"Ergosphere: r_erg = {r_erg:.4f}M")
print()

# Single thrust case parameters (known to work)
print("Single thrust case (known to work):")
E0, Lz0 = 1.2, 3.0
orbit_type, r_peri = check_orbit_type(E0, Lz0)
print(f"  E0={E0}, Lz={Lz0}: {orbit_type}, r_peri={r_peri}")
print()

# Scan for flyby orbits with periapsis in ergosphere
print("Scanning for flyby orbits with periapsis in ergosphere:")
print("-"*70)
print(f"{'E0':>6} {'Lz':>6} {'Type':>25} {'r_peri':>10} {'In Ergo?':>10}")
print("-"*70)

good_configs = []

for E0 in np.arange(1.1, 2.5, 0.1):
    for Lz in np.arange(2.0, 5.0, 0.2):
        orbit_type, r_peri = check_orbit_type(E0, Lz)
        
        if orbit_type == 'flyby_in_ergosphere':
            in_ergo = r_peri < r_erg
            print(f"{E0:6.2f} {Lz:6.2f} {orbit_type:>25} {r_peri:10.4f} {'YES' if in_ergo else 'NO':>10}")
            good_configs.append((E0, Lz, r_peri))

print()
print("="*70)
print("VIABLE CONFIGURATIONS (flyby with periapsis in ergosphere)")
print("="*70)

if good_configs:
    # Sort by deepest periapsis
    good_configs.sort(key=lambda x: x[2])
    
    print("\nTop 10 by deepest periapsis:")
    for i, (E, L, rp) in enumerate(good_configs[:10]):
        print(f"  {i+1}. E0={E:.2f}, Lz={L:.2f}, r_peri={rp:.4f}M")
    
    print("\n\nFiltering for periapsis INSIDE extraction zone (r < 1.8M):")
    deep = [(E, L, rp) for E, L, rp in good_configs if rp < 1.8]
    for E, L, rp in deep[:10]:
        print(f"  E0={E:.2f}, Lz={L:.2f}, r_peri={rp:.4f}M")
else:
    print("No viable configurations found!")
