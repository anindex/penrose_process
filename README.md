# Penrose Energy Extraction via Rocket Propulsion

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Numerical study of energy extraction from rotating (Kerr) black holes via the Penrose process using rocket propulsion. This repository accompanies the paper *"On the rarity of rocket-driven Penrose extraction in Kerr spacetime"* (coming soon).

## Results

Through **252,000 trajectory simulations** (112,000 main experimental phases + 140,000 spin-threshold characterization), we establish:

| Finding | Value |
|---------|-------|
| Broad-scan success rate ($a/M = 0.95$) | ~1% |
| Sweet-spot success rate ($a/M \geq 0.95$) | 11--14% |
| Peak success rate ($v_e = 0.98c$, $\delta m = 0.4$) | ~88.5% |
| Critical spin threshold | $0.88 < a_{\rm crit}/M \lesssim 0.89$ |
| Single-impulse efficiency | $\eta_{\rm cum} \approx 19\%$ |
| Continuous thrust efficiency | 2--4% |

**Sweet spot parameters:** specific energy $E_0 \approx 1.2$, specific angular momentum $L_z \approx 3.0$, $v_e \gtrsim 0.91c$

---

## Visualizations

### Single Impulse Thrust
A single impulsive burn at periapsis achieves maximum efficiency (~19%) by concentrating all thrust at the point of minimum exhaust energy.

![Single Impulse Penrose Extraction](visualizations/single_penrose.gif)

### Continuous Thrust
Sustained thrust throughout the ergosphere passage demonstrates path-averaging effects that reduce efficiency to 2--6%.

![Continuous Penrose Extraction](visualizations/continuous_penrose.gif)

---

## Physics Overview

### The Penrose Process

Within the **ergosphere** of a Kerr black hole, the stationary Killing vector becomes spacelike, permitting **negative-energy states**. A spacecraft can exploit this by ejecting exhaust with negative Killing energy $E_{\rm ex} < 0$, gaining energy at the expense of the black hole's rotation.

### Propulsion Model

We work in the **test-particle limit** ($m_0 \ll M$), so backreaction and self-force are negligible. The code implements exact 4-momentum conservation:

$$p'_\mu = p_\mu - \delta\mu \, u_{{\rm ex},\mu}$$

where the exhaust 4-velocity is $u_{\rm ex}^\mu = \gamma_e(u^\mu - v_e s^\mu)$ for exhaust speed $v_e$ and spatial direction $s^\mu$ orthogonal to the rocket's 4-velocity.

**Penrose signature:** Energy extraction occurs when exhaust Killing energy $E_{\rm ex} = -\gamma_e(u_t - v_e s_t) < 0$.

---

## Repository Structure

```
penrose_process/
|── continuous_thrust_case.py   # Sustained ergosphere thrust
|── single_thrust_case.py       # Single impulsive burn at periapsis
|── kerr_utils.py               # Kerr metric utilities
|── experiments/
|   |── trajectory_classifier.py    # Orbit classification
|   |── parameter_sweep.py          # Grid-based exploration
|   |── comprehensive_sweep.py      # Full statistical sweeps
|   |── thrust_comparison.py        # Strategy comparison
|   |── ensemble.py                 # Monte Carlo analysis
|   |── run_trajectory_study.py     # CLI runner
|   `── trajectory_visualization.py # Animated visualizations
|── tests/                      # Physics validation tests
`── results/                    # Generated output
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python >= 3.8, numpy, scipy, matplotlib

---

## Usage

### Run simulations

```bash
# Single impulse case
python single_thrust_case.py

# Continuous thrust case
python continuous_thrust_case.py
```

### Parameter studies

```bash
# Quick validation (~2 min)
python experiments/run_trajectory_study.py --mode quick

# Standard study (~10 min)
python experiments/run_trajectory_study.py --mode standard

# Full analysis (~30 min)
python experiments/run_trajectory_study.py --mode full
```

### Generate animations

```bash
python experiments/trajectory_visualization.py --spin 0.95
```

---

## Numerical Configuration

Recommended solver settings for Kerr geodesics:

```python
from scipy.integrate import solve_ivp

solution = solve_ivp(
    geodesic_rhs,
    t_span=(0, tau_max),
    y0=initial_state,
    method='DOP853',      # 8th-order Dormand-Prince
    rtol=1e-10,
    atol=1e-12,
)
```

Initial conditions: $r_0 = 15M$, escape radius: $50M$

---

## References

1. R. Penrose and R. M. Floyd, *Nature Phys. Sci.* **229**, 177 (1971). [doi:10.1038/physci229177a0](https://doi.org/10.1038/physci229177a0)

2. R. M. Wald, *Astrophys. J.* **191**, 231 (1974). [doi:10.1086/152959](https://doi.org/10.1086/152959)

3. S. Chandrasekhar, *The Mathematical Theory of Black Holes* (Oxford University Press, 1983).

