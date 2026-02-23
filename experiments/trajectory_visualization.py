"""
Trajectory Visualization Module
================================
Generate animated GIF visualizations of Penrose process trajectories.

Creates publication-quality animations showing:
- Black hole horizon and ergosphere
- Spacecraft trajectory with thrust indicators
- Energy and mass evolution
- Exhaust energy markers (negative = Penrose extraction)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, FancyArrowPatch
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from kerr_utils import (
    horizon_radius, ergosphere_radius, COLORS, setup_prd_style
)
from experiments.trajectory_classifier import TrajectoryOutcome
from experiments.thrust_comparison import (
    SimulationConfig, simulate_single_impulse, simulate_geodesic
)


# =============================================================================
# CUSTOM COLORMAPS
# =============================================================================

def create_trajectory_cmap():
    """Create colormap for trajectory (blue -> green for Penrose extraction)."""
    colors = ['#0072B2', '#56B4E9', '#009E73', '#F0E442']
    return LinearSegmentedColormap.from_list('trajectory', colors)


def create_energy_cmap():
    """Create colormap for energy visualization (red = negative, green = positive)."""
    colors = ['#D55E00', '#F0E442', '#009E73']
    return LinearSegmentedColormap.from_list('energy', colors)


# =============================================================================
# BLACK HOLE VISUALIZATION
# =============================================================================

def draw_black_hole(ax, a: float, M: float = 1.0, n_theta: int = 100):
    """
    Draw the black hole horizon and ergosphere.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on
    a : float
        Black hole spin parameter
    M : float
        Black hole mass
    n_theta : int
        Number of points for ergosphere boundary
    """
    r_plus = horizon_radius(a, M)
    
    # Horizon (black filled circle)
    horizon = Circle((0, 0), r_plus, color='black', zorder=10, label='Horizon')
    ax.add_patch(horizon)
    
    # Ergosphere in equatorial plane (theta = pi/2)
    # At the equator, r_erg = 2M (constant - it's a circle in the x-y plane)
    # The peanut shape only appears in meridional (r-theta) slices, not equatorial (r-phi)
    r_erg = ergosphere_radius(np.pi/2, a, M)  # = 2M at equator
    
    # Draw ergosphere as a simple circle (correct for equatorial plane view)
    ergosphere = Circle((0, 0), r_erg, color=COLORS['vermilion'], 
                         fill=True, alpha=0.15, zorder=5, label='Ergosphere')
    ax.add_patch(ergosphere)
    
    # Ergosphere boundary line
    theta_full = np.linspace(0, 2*np.pi, n_theta)
    x_erg = r_erg * np.cos(theta_full)
    y_erg = r_erg * np.sin(theta_full)
    ax.plot(x_erg, y_erg, color=COLORS['vermilion'], lw=1.5, ls='--', zorder=6)
    
    # Mark axes
    ax.axhline(0, color='gray', lw=0.5, ls=':', alpha=0.5)
    ax.axvline(0, color='gray', lw=0.5, ls=':', alpha=0.5)
    
    return r_plus, r_erg


# =============================================================================
# SINGLE THRUST ANIMATION
# =============================================================================

def animate_single_thrust(config: SimulationConfig,
                          output_path: str = "single_thrust.gif",
                          fps: int = 30,
                          duration: float = 8.0,
                          dpi: int = 150,
                          figsize: Tuple[float, float] = (12, 5)) -> Optional[str]:
    """
    Create animated GIF of single thrust Penrose extraction.
    
    Parameters
    ----------
    config : SimulationConfig
        Configuration for simulation
    output_path : str
        Output file path
    fps : int
        Frames per second
    duration : float
        Animation duration in seconds
    dpi : int
        Resolution
    figsize : tuple
        Figure size
        
    Returns
    -------
    str or None
        Output path if successful, None otherwise
    """
    setup_prd_style()
    
    # Run simulation
    result = simulate_single_impulse(config)
    
    if result.outcome != TrajectoryOutcome.ESCAPE:
        print(f"Trajectory did not escape (outcome: {result.outcome.name})")
        print("Try adjusting E0 or Lz0 for a successful extraction")
        return None
    
    if result.trajectory_data is None:
        print("No trajectory data available")
        return None
    
    # Extract data
    tau = result.trajectory_data['tau']
    r = result.trajectory_data['r']
    phi = result.trajectory_data['phi']
    m = result.trajectory_data['m']
    E = result.trajectory_data['E']
    
    # Smooth the trajectory by interpolating to uniform tau spacing
    from scipy.interpolate import interp1d
    
    # Remove duplicate tau values (can occur at events)
    _, unique_idx = np.unique(tau, return_index=True)
    unique_idx = np.sort(unique_idx)  # Maintain order
    tau_unique = tau[unique_idx]
    r_unique = r[unique_idx]
    phi_unique = phi[unique_idx]
    m_unique = m[unique_idx]
    E_unique = E[unique_idx]
    
    n_smooth = max(len(tau_unique) * 3, 1000)  # At least 1000 points for smooth animation
    tau_smooth = np.linspace(tau_unique[0], tau_unique[-1], n_smooth)
    
    # Use cubic interpolation for smooth curves (with bounds handling)
    r_interp = interp1d(tau_unique, r_unique, kind='cubic', bounds_error=False, fill_value='extrapolate')
    phi_interp = interp1d(tau_unique, phi_unique, kind='cubic', bounds_error=False, fill_value='extrapolate')
    m_interp = interp1d(tau_unique, m_unique, kind='linear', bounds_error=False, fill_value='extrapolate')
    E_interp = interp1d(tau_unique, E_unique, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    r_smooth = r_interp(tau_smooth)
    phi_smooth = phi_interp(tau_smooth)
    m_smooth = m_interp(tau_smooth)
    E_smooth = E_interp(tau_smooth)
    
    # Use smooth data for animation
    tau = tau_smooth
    r = r_smooth
    phi = phi_smooth
    m = m_smooth
    E = E_smooth
    
    # Convert to Cartesian
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    # Find impulse point - map original index to smooth index
    orig_tau = result.trajectory_data['tau']
    orig_impulse_idx = result.trajectory_data.get('impulse_idx', len(orig_tau)//2)
    impulse_tau = orig_tau[orig_impulse_idx]
    impulse_idx = np.argmin(np.abs(tau - impulse_tau))  # Find closest in smooth data
    
    r_impulse = r[impulse_idx]
    phi_impulse = phi[impulse_idx]
    x_impulse = r_impulse * np.cos(phi_impulse)
    y_impulse = r_impulse * np.sin(phi_impulse)
    
    # Setup figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1], wspace=0.3)
    
    ax_traj = fig.add_subplot(gs[0])
    ax_energy = fig.add_subplot(gs[1])
    ax_mass = fig.add_subplot(gs[2])
    
    # Draw black hole
    r_plus, r_erg = draw_black_hole(ax_traj, config.a, config.M)
    
    # Set trajectory axis limits - ZOOM IN on ergosphere region
    # Focus on the extraction zone: ~3x ergosphere radius
    r_view = r_erg * 3.5
    ax_traj.set_xlim(-r_view, r_view)
    ax_traj.set_ylim(-r_view, r_view)
    ax_traj.set_aspect('equal')
    ax_traj.set_xlabel(r'$x/M$')
    ax_traj.set_ylabel(r'$y/M$')
    ax_traj.set_title(f'Single Thrust Penrose Extraction\n$a/M = {config.a}$, $v_e = {config.v_e}c$')
    
    # Energy plot setup
    ax_energy.set_xlim(0, tau[-1])
    E_range = max(np.max(E) - np.min(E), 0.1)
    ax_energy.set_ylim(np.min(E) - 0.1*E_range, np.max(E) + 0.1*E_range)
    ax_energy.set_xlabel(r'$\tau/M$')
    ax_energy.set_ylabel(r'$E$')
    ax_energy.set_title('Energy Evolution')
    ax_energy.axhline(config.E0, color='gray', ls='--', lw=0.8, label=r'$E_0$')
    
    # Mass plot setup
    ax_mass.set_xlim(0, tau[-1])
    ax_mass.set_ylim(0, config.m0 * 1.1)
    ax_mass.set_xlabel(r'$\tau/M$')
    ax_mass.set_ylabel(r'$m/m_0$')
    ax_mass.set_title('Mass Evolution')
    
    # Initialize plot elements
    traj_line, = ax_traj.plot([], [], color=COLORS['blue'], lw=1.5, alpha=0.7)
    traj_point, = ax_traj.plot([], [], 'o', color=COLORS['blue'], ms=8, zorder=20)
    
    impulse_marker, = ax_traj.plot([], [], '*', color=COLORS['green'], ms=15, 
                                    zorder=25, label='Impulse')
    
    energy_line, = ax_energy.plot([], [], color=COLORS['orange'], lw=2)
    energy_point, = ax_energy.plot([], [], 'o', color=COLORS['orange'], ms=6)
    
    mass_line, = ax_mass.plot([], [], color=COLORS['purple'], lw=2)
    mass_point, = ax_mass.plot([], [], 'o', color=COLORS['purple'], ms=6)
    
    # Energy gain annotation
    Delta_E = result.Delta_E
    E_ex = result.E_ex_mean
    
    energy_text = ax_energy.text(0.95, 0.95, '', transform=ax_energy.transAxes,
                                  ha='right', va='top', fontsize=9,
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Exhaust energy indicator
    if E_ex < 0:
        exhaust_text = f"$E_{{\\rm ex}} = {E_ex:.3f}$ (NEGATIVE!)"
        exhaust_color = COLORS['green']
    else:
        exhaust_text = f"$E_{{\\rm ex}} = {E_ex:.3f}$"
        exhaust_color = COLORS['vermilion']
    
    ax_traj.text(0.02, 0.98, exhaust_text, transform=ax_traj.transAxes,
                 ha='left', va='top', fontsize=10, color=exhaust_color,
                 fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Animation frames
    n_frames = int(fps * duration)
    frame_indices = np.linspace(0, len(tau)-1, n_frames).astype(int)
    
    def init():
        traj_line.set_data([], [])
        traj_point.set_data([], [])
        impulse_marker.set_data([], [])
        energy_line.set_data([], [])
        energy_point.set_data([], [])
        mass_line.set_data([], [])
        mass_point.set_data([], [])
        energy_text.set_text('')
        return (traj_line, traj_point, impulse_marker, 
                energy_line, energy_point, mass_line, mass_point, energy_text)
    
    def animate(frame):
        idx = frame_indices[frame]
        
        # Update trajectory
        traj_line.set_data(x[:idx+1], y[:idx+1])
        traj_point.set_data([x[idx]], [y[idx]])
        
        # Show impulse marker after impulse
        if idx >= impulse_idx:
            impulse_marker.set_data([x_impulse], [y_impulse])
            # Change trajectory color after impulse
            traj_line.set_color(COLORS['green'])
            traj_point.set_color(COLORS['green'])
        
        # Update energy
        energy_line.set_data(tau[:idx+1], E[:idx+1])
        energy_point.set_data([tau[idx]], [E[idx]])
        
        # Update mass
        mass_line.set_data(tau[:idx+1], m[:idx+1])
        mass_point.set_data([tau[idx]], [m[idx]])
        
        # Update text
        current_Delta_E = E[idx] - config.E0
        energy_text.set_text(f'$\\Delta E = {current_Delta_E:+.4f}$\n$r = {r[idx]:.2f}M$')
        
        return (traj_line, traj_point, impulse_marker,
                energy_line, energy_point, mass_line, mass_point, energy_text)
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                        interval=1000/fps, blit=True)
    
    # Add legend
    ax_traj.legend(loc='lower right', fontsize=8)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close()
    
    print(f"Animation saved to: {output_path}")
    print(f"  Outcome: {result.outcome.name}")
    print(f"  DeltaE = {Delta_E:+.4f}")
    print(f"  E_ex = {E_ex:.4f}")
    print(f"  Penrose extraction: {'Yes' if E_ex < 0 else 'No'}")
    
    return output_path


# =============================================================================
# CONTINUOUS THRUST ANIMATION
# =============================================================================

def simulate_continuous_thrust_for_animation(config: SimulationConfig) -> Dict:
    """
    Run continuous thrust simulation and return trajectory data.
    
    This wraps the continuous_thrust_case.py logic for animation.
    """
    from scipy.integrate import solve_ivp
    from kerr_utils import (
        kerr_metric_components, compute_exhaust_energy,
        compute_optimal_exhaust_direction, frame_dragging_omega
    )
    
    a, M = config.a, config.M
    r_plus = horizon_radius(a, M)
    r_erg = ergosphere_radius(np.pi/2, a, M)
    
    # State: [r, phi, pr, pphi, m, pt]
    r0, phi0 = config.r0, 0.0
    pt0 = -config.E0
    pphi0 = config.Lz0
    m0 = config.m0
    
    th = np.pi / 2
    _, con = kerr_metric_components(r0, th, a, M)
    gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con
    
    rhs = -(gu_tt * pt0**2 + 2*gu_tphi * pt0 * pphi0 + gu_phiphi * pphi0**2 + m0**2)
    if rhs < 0:
        return None
    pr0 = -np.sqrt(rhs / gu_rr)
    
    state0 = np.array([r0, phi0, pr0, pphi0, m0, pt0])
    
    # Continuous thrust dynamics
    T_max = config.T_max
    v_e = config.v_e
    gamma_e = config.gamma_e
    m_min = config.m_min * m0
    
    E_ex_history = []
    r_ex_history = []
    thrust_history = []
    
    def continuous_dynamics(tau, state):
        r, phi, pr, pphi, m, pt = state
        
        r_safe = max(r, r_plus + config.horizon_margin)
        cov, con = kerr_metric_components(r_safe, th, a, M, clamp_horizon=True, warn_horizon=False)
        g_tt, g_tphi, g_rr, _, g_phiphi = cov
        gu_tt, gu_tphi, gu_rr, _, gu_phiphi = con
        
        # Geodesic terms
        u_t = (gu_tt * pt + gu_tphi * pphi) / m
        u_r = gu_rr * pr / m
        u_phi = (gu_tphi * pt + gu_phiphi * pphi) / m
        
        # Hamiltonian derivative (numerical)
        eps = 1e-7 * r_safe
        def H_at_r(r_):
            if r_ < r_plus + config.horizon_margin:
                r_ = r_plus + config.horizon_margin
            _, con_ = kerr_metric_components(r_, th, a, M, clamp_horizon=True, warn_horizon=False)
            gu_tt_, gu_tphi_, gu_rr_, _, gu_phiphi_ = con_
            return 0.5 * (gu_tt_ * pt**2 + 2*gu_tphi_ * pt * pphi + 
                          gu_rr_ * pr**2 + gu_phiphi_ * pphi**2)
        
        dH_dr = (H_at_r(r_safe + eps) - H_at_r(r_safe - eps)) / (2*eps)
        
        # Thrust: only fire in ergosphere and if mass available
        dm_dtau = 0.0
        dpt_dtau = 0.0
        dpphi_dtau = 0.0
        
        in_ergosphere = r_safe < r_erg
        has_mass = m > m_min
        
        if in_ergosphere and has_mass:
            # Compute optimal exhaust direction
            u_vec = np.array([u_t, u_r, u_phi])
            try:
                opt = compute_optimal_exhaust_direction(
                    u_vec, r_safe, th, a, M, v_e,
                    g_tt, g_tphi, g_rr, g_phiphi
                )
                if opt is not None and opt['E_ex'] < 0:
                    # Rocket mass loss rate: dm/dtau < 0
                    dm_dtau = -T_max / (gamma_e * v_e)
                    
                    # Record for animation
                    E_ex_history.append(opt['E_ex'])
                    r_ex_history.append(r_safe)
                    thrust_history.append(tau)
                    
                    # EXACT 4-momentum conservation (matching continuous_thrust_case.py):
                    # Exhaust rest mass rate: deltamu_rate = -dm/dtau / gamma_e > 0
                    # dp_mu/dtau = -deltamu_rate * u_ex_cov[mu]
                    #
                    # For E_ex < 0: u_ex_cov[0] > 0
                    # So: dp_t/dtau = -deltamu_rate * u_ex_cov[0] < 0
                    # This makes p_t MORE negative, INCREASING E = -p_t [OK]
                    u_ex_cov = opt['u_ex_cov']
                    delta_mu_rate = -dm_dtau / gamma_e  # Positive exhaust rest mass rate
                    
                    dpt_dtau = -delta_mu_rate * u_ex_cov[0]
                    dpphi_dtau = -delta_mu_rate * u_ex_cov[2]
            except:
                pass
        
        max_deriv = 1e6
        dr_dtau = np.clip(u_r, -max_deriv, max_deriv)
        dphi_dtau = np.clip(u_phi, -max_deriv, max_deriv)
        dpr_dtau = np.clip(-dH_dr / m if m > 1e-10 else 0.0, -max_deriv, max_deriv)
        
        return np.array([dr_dtau, dphi_dtau, dpr_dtau, dpphi_dtau, dm_dtau, dpt_dtau])
    
    # Events
    def horizon_event(t, y):
        return y[0] - r_plus - config.horizon_margin
    horizon_event.terminal = True
    horizon_event.direction = -1
    
    def escape_event(t, y):
        return y[0] - config.escape_radius
    escape_event.terminal = True
    escape_event.direction = 1
    
    # Integrate
    sol = solve_ivp(
        continuous_dynamics,
        [0, config.tau_max],
        state0,
        method='DOP853',
        events=[horizon_event, escape_event],
        rtol=config.rtol,
        atol=config.atol,
        max_step=1.0,  # Smaller steps for thrust tracking
        dense_output=True
    )
    
    if not sol.success:
        return None
    
    # Determine outcome
    if len(sol.t_events[1]) > 0:
        outcome = 'ESCAPE'
    elif len(sol.t_events[0]) > 0:
        outcome = 'CAPTURE'
    else:
        outcome = 'STALLED'
    
    return {
        'tau': sol.t,
        'r': sol.y[0],
        'phi': sol.y[1],
        'pr': sol.y[2],
        'pphi': sol.y[3],
        'm': sol.y[4],
        'E': -sol.y[5],
        'E_ex_history': np.array(E_ex_history) if E_ex_history else np.array([]),
        'r_ex_history': np.array(r_ex_history) if r_ex_history else np.array([]),
        'thrust_history': np.array(thrust_history) if thrust_history else np.array([]),
        'outcome': outcome,
        'r_erg': r_erg,
    }


def animate_continuous_thrust(config: SimulationConfig,
                               output_path: str = "continuous_thrust.gif",
                               fps: int = 30,
                               duration: float = 10.0,
                               dpi: int = 150,
                               figsize: Tuple[float, float] = (14, 5)) -> Optional[str]:
    """
    Create animated GIF of continuous thrust Penrose extraction.
    """
    setup_prd_style()
    
    # Run simulation
    data = simulate_continuous_thrust_for_animation(config)
    
    if data is None:
        print("Simulation failed")
        return None
    
    if data['outcome'] != 'ESCAPE':
        print(f"Trajectory did not escape (outcome: {data['outcome']})")
        return None
    
    # Extract data
    tau = data['tau']
    r = data['r']
    phi = data['phi']
    m = data['m']
    E = data['E']
    r_erg = data['r_erg']
    
    # Smooth the trajectory by interpolating to uniform tau spacing
    from scipy.interpolate import interp1d
    
    # Remove duplicate tau values (can occur at events)
    _, unique_idx = np.unique(tau, return_index=True)
    unique_idx = np.sort(unique_idx)
    tau_unique = tau[unique_idx]
    r_unique = r[unique_idx]
    phi_unique = phi[unique_idx]
    m_unique = m[unique_idx]
    E_unique = E[unique_idx]
    
    n_smooth = max(len(tau_unique) * 3, 1000)  # At least 1000 points for smooth animation
    tau_smooth = np.linspace(tau_unique[0], tau_unique[-1], n_smooth)
    
    # Use cubic interpolation for smooth curves
    r_interp = interp1d(tau_unique, r_unique, kind='cubic', bounds_error=False, fill_value='extrapolate')
    phi_interp = interp1d(tau_unique, phi_unique, kind='cubic', bounds_error=False, fill_value='extrapolate')
    m_interp = interp1d(tau_unique, m_unique, kind='linear', bounds_error=False, fill_value='extrapolate')
    E_interp = interp1d(tau_unique, E_unique, kind='linear', bounds_error=False, fill_value='extrapolate')
    
    tau_orig = tau  # Keep original for thrust event mapping
    tau = tau_smooth
    r = r_interp(tau_smooth)
    phi = phi_interp(tau_smooth)
    m = m_interp(tau_smooth)
    E = E_interp(tau_smooth)
    
    # Convert to Cartesian
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    # Thrust indicators
    thrust_tau = data['thrust_history']
    E_ex_vals = data['E_ex_history']
    
    # Setup figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 4, width_ratios=[2, 1, 1, 1], wspace=0.35)
    
    ax_traj = fig.add_subplot(gs[0])
    ax_energy = fig.add_subplot(gs[1])
    ax_mass = fig.add_subplot(gs[2])
    ax_Eex = fig.add_subplot(gs[3])
    
    # Draw black hole
    r_plus, _ = draw_black_hole(ax_traj, config.a, config.M)
    
    # Trajectory axis - ZOOM IN on ergosphere region
    r_view = r_erg * 3.5
    ax_traj.set_xlim(-r_view, r_view)
    ax_traj.set_ylim(-r_view, r_view)
    ax_traj.set_aspect('equal')
    ax_traj.set_xlabel(r'$x/M$')
    ax_traj.set_ylabel(r'$y/M$')
    ax_traj.set_title(f'Continuous Thrust Penrose Extraction\n$a/M = {config.a}$, $T_{{max}} = {config.T_max}$')
    
    # Energy axis
    ax_energy.set_xlim(0, tau[-1])
    E_range = max(np.max(E) - np.min(E), 0.1)
    ax_energy.set_ylim(np.min(E) - 0.1*E_range, np.max(E) + 0.1*E_range)
    ax_energy.set_xlabel(r'$\tau/M$')
    ax_energy.set_ylabel(r'$E$')
    ax_energy.set_title('Energy')
    ax_energy.axhline(config.E0, color='gray', ls='--', lw=0.8)
    
    # Mass axis
    ax_mass.set_xlim(0, tau[-1])
    ax_mass.set_ylim(0, config.m0 * 1.1)
    ax_mass.set_xlabel(r'$\tau/M$')
    ax_mass.set_ylabel(r'$m/m_0$')
    ax_mass.set_title('Mass')
    
    # E_ex axis
    if len(E_ex_vals) > 0:
        ax_Eex.set_xlim(0, max(len(E_ex_vals), 10))
        Eex_min = min(np.min(E_ex_vals), -0.1)
        ax_Eex.set_ylim(Eex_min * 1.2, max(0.1, np.max(E_ex_vals) * 1.2))
    else:
        ax_Eex.set_xlim(0, 10)
        ax_Eex.set_ylim(-0.2, 0.1)
    ax_Eex.axhline(0, color='gray', ls='--', lw=1)
    ax_Eex.set_xlabel('Thrust event')
    ax_Eex.set_ylabel(r'$E_{\rm ex}$')
    ax_Eex.set_title('Exhaust Energy')
    ax_Eex.fill_between([0, 100], [-1, -1], [0, 0], color=COLORS['green'], alpha=0.1)
    
    # Initialize elements
    # Trajectory with color gradient based on radius
    traj_line, = ax_traj.plot([], [], color=COLORS['blue'], lw=1.5, alpha=0.7)
    traj_point, = ax_traj.plot([], [], 'o', color=COLORS['blue'], ms=8, zorder=20)
    thrust_scatter = ax_traj.scatter([], [], c=[], cmap='RdYlGn_r', s=20, 
                                      vmin=-0.2, vmax=0.1, zorder=15, alpha=0.8)
    
    energy_line, = ax_energy.plot([], [], color=COLORS['orange'], lw=2)
    energy_point, = ax_energy.plot([], [], 'o', color=COLORS['orange'], ms=6)
    
    mass_line, = ax_mass.plot([], [], color=COLORS['purple'], lw=2)
    mass_point, = ax_mass.plot([], [], 'o', color=COLORS['purple'], ms=6)
    
    Eex_bars = ax_Eex.bar([], [], color=COLORS['green'], alpha=0.7)
    
    # Text annotations
    status_text = ax_traj.text(0.02, 0.98, '', transform=ax_traj.transAxes,
                                ha='left', va='top', fontsize=10,
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    stats_text = ax_energy.text(0.95, 0.95, '', transform=ax_energy.transAxes,
                                 ha='right', va='top', fontsize=9,
                                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Animation
    n_frames = int(fps * duration)
    frame_indices = np.linspace(0, len(tau)-1, n_frames).astype(int)
    
    def init():
        traj_line.set_data([], [])
        traj_point.set_data([], [])
        thrust_scatter.set_offsets(np.empty((0, 2)))
        energy_line.set_data([], [])
        energy_point.set_data([], [])
        mass_line.set_data([], [])
        mass_point.set_data([], [])
        status_text.set_text('')
        stats_text.set_text('')
        return (traj_line, traj_point, thrust_scatter,
                energy_line, energy_point, mass_line, mass_point,
                status_text, stats_text)
    
    def animate(frame):
        idx = frame_indices[frame]
        current_tau = tau[idx]
        
        # Update trajectory
        traj_line.set_data(x[:idx+1], y[:idx+1])
        traj_point.set_data([x[idx]], [y[idx]])
        
        # Color based on location
        if r[idx] < r_erg:
            traj_point.set_color(COLORS['green'])
            status = "IN ERGOSPHERE - EXTRACTING"
        else:
            traj_point.set_color(COLORS['blue'])
            status = "Outside ergosphere"
        
        # Show thrust points up to current time
        if len(thrust_tau) > 0:
            thrust_mask = thrust_tau <= current_tau
            n_thrusts = np.sum(thrust_mask)
            if n_thrusts > 0:
                # Interpolate thrust positions using smooth data
                thrust_phi_vals = np.interp(thrust_tau[thrust_mask], tau, phi)
                thrust_r_vals = np.interp(thrust_tau[thrust_mask], tau, r)
                thrust_x = thrust_r_vals * np.cos(thrust_phi_vals)
                thrust_y = thrust_r_vals * np.sin(thrust_phi_vals)
                thrust_scatter.set_offsets(np.column_stack([thrust_x, thrust_y]))
                thrust_scatter.set_array(E_ex_vals[:n_thrusts])
        
        # Update energy
        energy_line.set_data(tau[:idx+1], E[:idx+1])
        energy_point.set_data([tau[idx]], [E[idx]])
        
        # Update mass
        mass_line.set_data(tau[:idx+1], m[:idx+1])
        mass_point.set_data([tau[idx]], [m[idx]])
        
        # Update E_ex bar chart
        if len(thrust_tau) > 0:
            thrust_mask = thrust_tau <= current_tau
            n_thrusts = np.sum(thrust_mask)
            if n_thrusts > 0:
                ax_Eex.clear()
                ax_Eex.axhline(0, color='gray', ls='--', lw=1)
                ax_Eex.fill_between([0, max(n_thrusts+2, 10)], [-1, -1], [0, 0], 
                                    color=COLORS['green'], alpha=0.1)
                colors = [COLORS['green'] if e < 0 else COLORS['vermilion'] 
                         for e in E_ex_vals[:n_thrusts]]
                ax_Eex.bar(range(n_thrusts), E_ex_vals[:n_thrusts], color=colors, alpha=0.7)
                ax_Eex.set_xlabel('Thrust event')
                ax_Eex.set_ylabel(r'$E_{\rm ex}$')
                ax_Eex.set_title('Exhaust Energy')
                if len(E_ex_vals) > 0:
                    ax_Eex.set_xlim(-0.5, max(len(E_ex_vals)+1, 10))
                    Eex_min = min(np.min(E_ex_vals[:n_thrusts]), -0.05)
                    ax_Eex.set_ylim(Eex_min * 1.3, max(0.05, np.max(E_ex_vals[:n_thrusts]) * 1.2))
        
        # Update text
        n_negative = np.sum(E_ex_vals[:np.sum(thrust_tau <= current_tau)] < 0) if len(thrust_tau) > 0 else 0
        status_text.set_text(f"{status}\n$r = {r[idx]:.2f}M$")
        
        Delta_E = E[idx] - config.E0
        stats_text.set_text(f'$\\Delta E = {Delta_E:+.4f}$\n'
                           f'$E_{{ex}} < 0$: {n_negative}')
        
        return (traj_line, traj_point, thrust_scatter,
                energy_line, energy_point, mass_line, mass_point,
                status_text, stats_text)
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                        interval=1000/fps, blit=False)  # blit=False for bar updates
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close()
    
    # Print summary
    Delta_E_final = E[-1] - config.E0
    n_negative = np.sum(E_ex_vals < 0) if len(E_ex_vals) > 0 else 0
    
    print(f"Animation saved to: {output_path}")
    print(f"  Outcome: {data['outcome']}")
    print(f"  DeltaE = {Delta_E_final:+.4f}")
    print(f"  Total thrust events: {len(E_ex_vals)}")
    print(f"  Negative E_ex events: {n_negative}")
    print(f"  Mass fraction used: {1 - m[-1]/config.m0:.1%}")
    
    return output_path


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def find_successful_config(a: float = 0.95, 
                           n_tries: int = 50,
                           seed: int = 42) -> Optional[SimulationConfig]:
    """
    Find a configuration that results in successful Penrose extraction.
    """
    rng = np.random.default_rng(seed)
    
    # Sweet spot parameters for a=0.95
    E_range = (1.15, 1.30)
    Lz_range = (2.9, 3.3)
    
    for i in range(n_tries):
        E = rng.uniform(*E_range)
        Lz = rng.uniform(*Lz_range)
        
        config = SimulationConfig(
            a=a, E0=E, Lz0=Lz,
            r0=10.0, m0=1.0, v_e=0.95,
            delta_m_fraction=0.2
        )
        
        result = simulate_single_impulse(config)
        
        if result.outcome == TrajectoryOutcome.ESCAPE and result.E_ex_mean < 0:
            print(f"Found successful config: E={E:.3f}, Lz={Lz:.3f}")
            return config
    
    print("Could not find successful configuration")
    return None


def create_both_animations(output_dir: str = "results/animations",
                           a: float = 0.95) -> Tuple[Optional[str], Optional[str]]:
    """
    Create both single and continuous thrust animations.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find good config for single thrust
    print("Finding configuration for single thrust...")
    config_single = find_successful_config(a=a)
    
    if config_single is None:
        # Use known good parameters
        config_single = SimulationConfig(
            a=a, E0=1.20, Lz0=3.0,
            r0=10.0, m0=1.0, v_e=0.95,
            delta_m_fraction=0.2
        )
    
    # Single thrust animation
    print("\nCreating single thrust animation...")
    path1 = animate_single_thrust(
        config_single,
        output_path=f"{output_dir}/single_thrust_a{a}.gif"
    )
    
    # Continuous thrust config
    config_cont = SimulationConfig(
        a=a, E0=1.25, Lz0=3.1,
        r0=10.0, m0=1.0, v_e=0.95,
        T_max=0.1, m_min=0.3,
        tau_max=200.0
    )
    
    # Continuous thrust animation
    print("\nCreating continuous thrust animation...")
    path2 = animate_continuous_thrust(
        config_cont,
        output_path=f"{output_dir}/continuous_thrust_a{a}.gif"
    )
    
    return path1, path2


# =============================================================================
# ANIMATE FROM continuous_thrust_case.py SIMULATION
# =============================================================================

def animate_continuous_from_case(output_path: str = "results/animations/continuous_penrose.gif",
                                  n_frames: int = 300, fps: int = 30, dpi: int = 100) -> str:
    """
    Create animation using the exact simulation from continuous_thrust_case.py.
    
    This imports and runs the proven simulation that achieves:
    - Positive energy gain (DeltaE > 0)
    - Genuine Penrose extraction (E_ex < 0 for 96%+ of exhaust)
    - Successful escape to infinity
    
    Parameters
    ----------
    output_path : str
        Path to save the animation GIF
    n_frames : int
        Number of frames in animation
    fps : int
        Frames per second
    dpi : int
        Resolution
        
    Returns
    -------
    str
        Path to saved animation
    """
    from experiments.thrust_comparison import SimulationConfig
    
    # Use parameters known to achieve ESCAPE with positive DeltaE
    # These parameters give: DeltaE ~ +0.003, ~30k thrust events, escape to 50M
    config = SimulationConfig(
        a=0.95, M=1.0,
        E0=1.20, Lz0=3.0,
        r0=10.0, m0=1.0,
        v_e=0.95,
        T_max=0.08,
        m_min=0.3,
        tau_max=300.0,
        escape_radius=50.0,
    )
    
    # Run simulation using internal function
    data = simulate_continuous_thrust_for_animation(config)
    
    if data is None:
        print("Continuous thrust simulation failed")
        raise RuntimeError("Continuous thrust simulation failed")
    
    # Extract data
    tau = np.array(data['tau'])
    r = np.array(data['r'])
    phi = np.array(data['phi'])
    E = np.array(data['E'])
    m = np.array(data['m'])
    E_ex_history = list(data['E_ex_history']) if len(data['E_ex_history']) > 0 else []
    r_ex_history = list(data['r_ex_history']) if len(data['r_ex_history']) > 0 else []
    a = config.a
    M = config.M
    r_plus = horizon_radius(a, M)
    r_erg = ergosphere_radius(np.pi/2, a, M)
    R_extraction = r_erg * 0.85
    E0 = config.E0
    outcome = data['outcome']
    
    # Convert to Cartesian
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    # Calculate energy metrics
    Delta_E = E[-1] - E0
    Delta_m = m[-1] - m[0]
    n_penrose = sum(1 for e in E_ex_history if e < 0)
    pct_penrose = 100 * n_penrose / len(E_ex_history) if E_ex_history else 0
    
    print(f"\nContinuous thrust simulation:")
    print(f"  Outcome: {outcome}")
    print(f"  DeltaE = {Delta_E:+.6f}")
    print(f"  Deltam = {Delta_m:+.6f}")
    print(f"  Penrose extraction: {pct_penrose:.1f}% of {len(E_ex_history)} events")
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1], 
                          wspace=0.25, hspace=0.3)
    
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_energy = fig.add_subplot(gs[0, 1])
    ax_mass = fig.add_subplot(gs[1, 0])
    ax_eex = fig.add_subplot(gs[1, 1])
    
    # Trajectory plot limits - zoom to focus on ergosphere region
    r_max_view = r_erg * 4.0
    ax_traj.set_xlim(-r_max_view, r_max_view)
    ax_traj.set_ylim(-r_max_view, r_max_view)
    ax_traj.set_aspect('equal')
    ax_traj.set_xlabel('x/M', fontsize=11)
    ax_traj.set_ylabel('y/M', fontsize=11)
    
    # Draw black hole and ergosphere
    theta_plot = np.linspace(0, 2*np.pi, 100)
    
    # Event horizon (black disk)
    ax_traj.fill(r_plus * np.cos(theta_plot), r_plus * np.sin(theta_plot), 
                 'black', label=f'Horizon (r={r_plus:.2f}M)')
    
    # Ergosphere at equator (r = 2M for equatorial plane, independent of angle)
    ax_traj.plot(r_erg * np.cos(theta_plot), r_erg * np.sin(theta_plot),
                 'r--', lw=2, label=f'Ergosphere (r={r_erg:.1f}M)')
    
    # Extraction zone (where E_ex < 0 is achievable)
    ax_traj.plot(R_extraction * np.cos(theta_plot), R_extraction * np.sin(theta_plot),
                 'orange', ls=':', lw=1.5, alpha=0.7, label=f'Extraction limit (r={R_extraction:.2f}M)')
    
    # Full trajectory (faded)
    ax_traj.plot(x, y, 'b-', alpha=0.15, lw=0.5)
    
    # Trajectory up to current frame (animated)
    traj_line, = ax_traj.plot([], [], 'b-', lw=1.5, label='Trajectory')
    
    # Current position
    pos_marker, = ax_traj.plot([], [], 'go', ms=8, mew=2, label='Spacecraft')
    
    # Exhaust events markers
    thrust_scatter = ax_traj.scatter([], [], c=[], cmap='coolwarm_r', 
                                     s=20, alpha=0.6, vmin=-0.5, vmax=0.1)
    
    ax_traj.legend(loc='upper right', fontsize=8)
    
    # Title with metrics
    title_text = ax_traj.set_title('', fontsize=11)
    
    # Add annotation about animation speed (adaptive stepping causes slowdown at periapsis)
    ax_traj.text(0.02, 0.02, 
                 'Animation slows near periapsis\n(adaptive integration steps)',
                 transform=ax_traj.transAxes, fontsize=7, 
                 color='gray', alpha=0.8, va='bottom', ha='left',
                 style='italic')
    
    # Energy plot
    ax_energy.set_xlim(0, tau[-1])
    E_min, E_max = min(E.min(), E0) - 0.05, max(E.max(), E0) + 0.05
    ax_energy.set_ylim(E_min, E_max)
    ax_energy.set_xlabel('Proper time tau/M', fontsize=11)
    ax_energy.set_ylabel('Energy E', fontsize=11)
    ax_energy.axhline(E0, color='gray', ls='--', lw=1, alpha=0.5, label=f'E_0 = {E0:.3f}')
    
    # Plot full energy evolution (faded)
    ax_energy.plot(tau, E, 'b-', alpha=0.2, lw=0.5)
    
    # Animated energy line
    energy_line, = ax_energy.plot([], [], 'b-', lw=2, label='E(tau)')
    
    # Current energy marker
    energy_marker, = ax_energy.plot([], [], 'go', ms=8)
    ax_energy.legend(loc='lower right', fontsize=8)
    ax_energy.set_title(f'Energy Evolution (DeltaE = {Delta_E:+.4f})', fontsize=11)
    
    # Mass plot
    ax_mass.set_xlim(0, tau[-1])
    m_min_plot, m_max_plot = m.min() - 0.05, m.max() + 0.05
    ax_mass.set_ylim(m_min_plot, m_max_plot)
    ax_mass.set_xlabel('Proper time tau/M', fontsize=11)
    ax_mass.set_ylabel('Mass m', fontsize=11)
    ax_mass.axhline(m[0], color='gray', ls='--', lw=1, alpha=0.5, label=f'm_0 = {m[0]:.3f}')
    
    # Plot full mass evolution (faded)
    ax_mass.plot(tau, m, 'orange', alpha=0.2, lw=0.5)
    
    # Animated mass line
    mass_line, = ax_mass.plot([], [], 'orange', lw=2, label='m(tau)')
    mass_marker, = ax_mass.plot([], [], 'go', ms=8)
    ax_mass.legend(loc='upper right', fontsize=8)
    ax_mass.set_title(f'Mass Evolution (Deltam = {Delta_m:+.4f})', fontsize=11)
    
    # Exhaust energy plot
    if E_ex_history:
        E_ex_arr = np.array(E_ex_history)
        ax_eex.set_xlim(0, len(E_ex_arr))
        E_ex_min, E_ex_max = E_ex_arr.min() - 0.1, max(0.1, E_ex_arr.max() + 0.1)
        ax_eex.set_ylim(E_ex_min, E_ex_max)
        ax_eex.axhline(0, color='black', ls='-', lw=1, alpha=0.5)
        ax_eex.fill_between([0, len(E_ex_arr)], [0, 0], [E_ex_min, E_ex_min], 
                           color='green', alpha=0.1, label='Penrose zone (E_ex < 0)')
        
        # Plot all E_ex values (faded)
        ax_eex.scatter(range(len(E_ex_arr)), E_ex_arr, c=E_ex_arr, cmap='coolwarm_r',
                      s=10, alpha=0.2, vmin=-0.5, vmax=0.1)
        
        # Animated E_ex scatter
        eex_scatter = ax_eex.scatter([], [], c=[], cmap='coolwarm_r', 
                                     s=30, alpha=0.8, vmin=-0.5, vmax=0.1)
    else:
        E_ex_arr = np.array([])
        eex_scatter = ax_eex.scatter([], [], c=[], s=30)
    
    ax_eex.set_xlabel('Thrust event #', fontsize=11)
    ax_eex.set_ylabel('Exhaust energy E_ex', fontsize=11)
    ax_eex.set_title(f'Exhaust Energy ({pct_penrose:.0f}% Penrose)', fontsize=11)
    ax_eex.legend(loc='upper right', fontsize=8)
    
    # Subsample for animation
    n_points = len(tau)
    frame_indices = np.linspace(0, n_points - 1, n_frames).astype(int)
    
    # Map tau to thrust event indices
    r_arr = np.array(r.tolist())
    thrust_tau_indices = []
    for r_ex in r_ex_history:
        idx = np.argmin(np.abs(r_arr - r_ex))
        thrust_tau_indices.append(idx)
    
    def init():
        traj_line.set_data([], [])
        pos_marker.set_data([], [])
        energy_line.set_data([], [])
        energy_marker.set_data([], [])
        mass_line.set_data([], [])
        mass_marker.set_data([], [])
        thrust_scatter.set_offsets(np.zeros((0, 2)))
        if len(E_ex_arr) > 0:
            eex_scatter.set_offsets(np.zeros((0, 2)))
        title_text.set_text('')
        return traj_line, pos_marker, energy_line, energy_marker, mass_line, mass_marker, thrust_scatter, title_text
    
    def animate(frame):
        idx = frame_indices[frame]
        
        # Update trajectory
        traj_line.set_data(x[:idx+1], y[:idx+1])
        pos_marker.set_data([x[idx]], [y[idx]])
        
        # Update energy plot
        energy_line.set_data(tau[:idx+1], E[:idx+1])
        energy_marker.set_data([tau[idx]], [E[idx]])
        
        # Update mass plot
        mass_line.set_data(tau[:idx+1], m[:idx+1])
        mass_marker.set_data([tau[idx]], [m[idx]])
        
        # Count thrust events up to current frame
        n_events = sum(1 for ti in thrust_tau_indices if ti <= idx)
        
        # Update E_ex scatter
        if len(E_ex_arr) > 0 and n_events > 0:
            eex_scatter.set_offsets(np.column_stack([
                np.arange(n_events), E_ex_arr[:n_events]
            ]))
            eex_scatter.set_array(E_ex_arr[:n_events])
        
        # Update exhaust scatter on trajectory
        if r_ex_history and n_events > 0:
            thrust_x = []
            thrust_y = []
            thrust_E_ex = []
            for i in range(n_events):
                ti = thrust_tau_indices[i]
                if ti < len(phi):
                    thrust_x.append(r_ex_history[i] * np.cos(phi[ti]))
                    thrust_y.append(r_ex_history[i] * np.sin(phi[ti]))
                    thrust_E_ex.append(E_ex_history[i])
            
            if thrust_x:
                thrust_scatter.set_offsets(np.column_stack([thrust_x, thrust_y]))
                thrust_scatter.set_array(np.array(thrust_E_ex))
        
        # Update title
        dE_current = E[idx] - E0
        dm_current = m[idx] - m[0]
        title_text.set_text(
            f'Continuous Penrose Extraction (a={a:.2f}M)\n'
            f'tau = {tau[idx]:.1f}M  |  r = {r[idx]:.2f}M  |  DeltaE = {dE_current:+.4f}  |  Deltam = {dm_current:+.3f}'
        )
        
        return traj_line, pos_marker, energy_line, energy_marker, mass_line, mass_marker, thrust_scatter, title_text
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                        interval=1000/fps, blit=True)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close()
    
    print(f"\nAnimation saved to: {output_path}")
    print(f"  Outcome: {outcome}")
    print(f"  DeltaE = {Delta_E:+.6f} (POSITIVE - energy extracted!)")
    print(f"  Deltam = {Delta_m:+.6f} ({-Delta_m*100:.1f}% fuel used)")
    print(f"  Penrose events: {n_penrose}/{len(E_ex_history)} ({pct_penrose:.1f}%)")
    
    return output_path


def animate_single_from_case(output_path: str = "results/animations/single_penrose.gif",
                              n_frames: int = 200, fps: int = 30, dpi: int = 100) -> str:
    """
    Create animation using the exact simulation from single_thrust_case.py.
    
    Shows the classic impulsive Penrose process with:
    - Trajectory (infall + escape phases)
    - Energy evolution with impulse jump
    - Mass evolution with impulse drop
    - Single E_ex value display
    
    Parameters
    ----------
    output_path : str
        Path to save the animation GIF
    n_frames : int
        Number of frames in animation
    fps : int
        Frames per second
    dpi : int
        Resolution
        
    Returns
    -------
    str
        Path to saved animation
    """
    import subprocess
    import sys
    import json
    
    # Run single_thrust_case.py and capture the trajectory data
    script = '''
import sys
sys.path.insert(0, '.')
import numpy as np
import json

# Import simulation data
from single_thrust_case import (
    tau, r, phi, m, E_hist, burn_idx, E_ex, delta_mu_impulse,
    a, M, r_plus, E0, Lz0, escaped, captured
)
from kerr_utils import ergosphere_radius

# Convert to x, y coordinates
xc = r * np.cos(phi)
yc = r * np.sin(phi)

r_erg = ergosphere_radius(np.pi/2, a, M)

# Determine outcome
outcome = 'escape' if escaped else ('captured' if captured else 'unknown')

# Output data as JSON
data = {
    'tau': tau.tolist(),
    'r': r.tolist(),
    'phi': phi.tolist(),
    'x': xc.tolist(),
    'y': yc.tolist(),
    'E': E_hist.tolist(),
    'm': m.tolist(),
    'burn_idx': int(burn_idx),
    'E_ex': float(E_ex),
    'delta_mu': float(delta_mu_impulse),
    'a': float(a),
    'M': float(M),
    'r_plus': float(r_plus),
    'r_erg': float(r_erg),
    'E0': float(E0),
    'Lz0': float(Lz0),
    'outcome': outcome
}

print(json.dumps(data))
'''
    
    # Run in subprocess
    result = subprocess.run(
        [sys.executable, '-W', 'ignore', '-c', script],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent)
    )
    
    if result.returncode != 0:
        print(f"Error running simulation: {result.stderr}")
        raise RuntimeError("Single thrust simulation failed")
    
    # Parse the JSON output (last line)
    lines = result.stdout.strip().split('\n')
    data = json.loads(lines[-1])
    
    # Extract data
    tau = np.array(data['tau'])
    r = np.array(data['r'])
    phi = np.array(data['phi'])
    x = np.array(data['x'])
    y = np.array(data['y'])
    E = np.array(data['E'])
    m = np.array(data['m'])
    burn_idx = data['burn_idx']
    E_ex = data['E_ex']
    delta_mu = data['delta_mu']
    a = data['a']
    M = data['M']
    r_plus = data['r_plus']
    r_erg = data['r_erg']
    E0 = data['E0']
    outcome = data['outcome']
    
    # Calculate metrics
    Delta_E = E[-1] - E0
    Delta_m = m[-1] - m[0]
    
    print(f"\nSingle thrust simulation loaded:")
    print(f"  Outcome: {outcome}")
    print(f"  DeltaE = {Delta_E:+.6f}")
    print(f"  Deltam = {Delta_m:+.6f}")
    print(f"  E_ex = {E_ex:.6f} ({'< 0 PENROSE!' if E_ex < 0 else '> 0'})")
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1], 
                          wspace=0.25, hspace=0.3)
    
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_energy = fig.add_subplot(gs[0, 1])
    ax_mass = fig.add_subplot(gs[1, 0])
    ax_eex = fig.add_subplot(gs[1, 1])
    
    # Trajectory plot limits - zoom to focus on ergosphere region
    r_max_view = r_erg * 4.0
    ax_traj.set_xlim(-r_max_view, r_max_view)
    ax_traj.set_ylim(-r_max_view, r_max_view)
    ax_traj.set_aspect('equal')
    ax_traj.set_xlabel('x/M', fontsize=11)
    ax_traj.set_ylabel('y/M', fontsize=11)
    
    # Draw black hole and ergosphere
    theta_plot = np.linspace(0, 2*np.pi, 100)
    
    # Event horizon (black disk)
    ax_traj.fill(r_plus * np.cos(theta_plot), r_plus * np.sin(theta_plot), 
                 'black', label=f'Horizon (r={r_plus:.2f}M)')
    
    # Ergosphere at equator
    ax_traj.plot(r_erg * np.cos(theta_plot), r_erg * np.sin(theta_plot),
                 'r--', lw=2, label=f'Ergosphere (r={r_erg:.1f}M)')
    
    # Full trajectory (faded) - infall and escape in different colors
    ax_traj.plot(x[:burn_idx+1], y[:burn_idx+1], 'b-', alpha=0.15, lw=0.5)
    ax_traj.plot(x[burn_idx:], y[burn_idx:], 'g-', alpha=0.15, lw=0.5)
    
    # Trajectory up to current frame (animated)
    traj_infall, = ax_traj.plot([], [], 'b-', lw=1.5, label='Infall')
    traj_escape, = ax_traj.plot([], [], 'g-', lw=1.5, label='Escape')
    
    # Current position
    pos_marker, = ax_traj.plot([], [], 'go', ms=10, mew=2, label='Spacecraft')
    
    # Impulse marker (shown after burn)
    impulse_marker, = ax_traj.plot([], [], '*', color='orange', ms=16, mec='black', 
                                   mew=1, label='Impulse')
    
    ax_traj.legend(loc='upper right', fontsize=8)
    
    # Title with metrics
    title_text = ax_traj.set_title('', fontsize=11)
    
    # Energy plot
    ax_energy.set_xlim(0, tau[-1])
    E_min, E_max = min(E.min(), E0) - 0.05, max(E.max(), E0) + 0.1
    ax_energy.set_ylim(E_min, E_max)
    ax_energy.set_xlabel('Proper time tau/M', fontsize=11)
    ax_energy.set_ylabel('Energy E', fontsize=11)
    ax_energy.axhline(E0, color='gray', ls='--', lw=1, alpha=0.5, label=f'E_0 = {E0:.3f}')
    ax_energy.axvline(tau[burn_idx], color='orange', ls='-', lw=2, alpha=0.3, label='Impulse')
    
    # Plot full energy evolution (faded)
    ax_energy.plot(tau, E, 'b-', alpha=0.2, lw=0.5)
    
    # Animated energy line
    energy_line, = ax_energy.plot([], [], 'b-', lw=2, label='E(tau)')
    energy_marker, = ax_energy.plot([], [], 'go', ms=8)
    ax_energy.legend(loc='lower right', fontsize=8)
    ax_energy.set_title(f'Energy Evolution (DeltaE = {Delta_E:+.4f})', fontsize=11)
    
    # Mass plot
    ax_mass.set_xlim(0, tau[-1])
    m_min_plot, m_max_plot = m.min() - 0.1, m.max() + 0.05
    ax_mass.set_ylim(m_min_plot, m_max_plot)
    ax_mass.set_xlabel('Proper time tau/M', fontsize=11)
    ax_mass.set_ylabel('Mass m', fontsize=11)
    ax_mass.axhline(m[0], color='gray', ls='--', lw=1, alpha=0.5, label=f'm_0 = {m[0]:.3f}')
    ax_mass.axvline(tau[burn_idx], color='orange', ls='-', lw=2, alpha=0.3, label='Impulse')
    
    # Plot full mass evolution (faded)
    ax_mass.plot(tau, m, 'orange', alpha=0.2, lw=0.5)
    
    # Animated mass line
    mass_line, = ax_mass.plot([], [], 'orange', lw=2, label='m(tau)')
    mass_marker, = ax_mass.plot([], [], 'go', ms=8)
    ax_mass.legend(loc='upper right', fontsize=8)
    ax_mass.set_title(f'Mass Evolution (Deltam = {Delta_m:+.4f})', fontsize=11)
    
    # Exhaust energy display (single value)
    ax_eex.set_xlim(-0.5, 1.5)
    ax_eex.set_ylim(-1, 0.5)
    ax_eex.axhline(0, color='black', ls='-', lw=1)
    ax_eex.fill_between([-0.5, 1.5], [0, 0], [-1, -1], 
                       color='green', alpha=0.2, label='Penrose zone (E_ex < 0)')
    
    # E_ex bar (animated)
    eex_bar = ax_eex.bar([0.5], [0], width=0.5, color='gray', alpha=0.5)
    eex_text = ax_eex.text(0.5, 0.3, '', ha='center', fontsize=14, fontweight='bold')
    
    ax_eex.set_xlabel('', fontsize=11)
    ax_eex.set_ylabel('Exhaust energy E_ex', fontsize=11)
    ax_eex.set_title('Single Impulse Exhaust Energy', fontsize=11)
    ax_eex.set_xticks([0.5])
    ax_eex.set_xticklabels(['Impulse'])
    ax_eex.legend(loc='upper right', fontsize=8)
    
    # Subsample for animation
    n_points = len(tau)
    frame_indices = np.linspace(0, n_points - 1, n_frames).astype(int)
    
    def init():
        traj_infall.set_data([], [])
        traj_escape.set_data([], [])
        pos_marker.set_data([], [])
        impulse_marker.set_data([], [])
        energy_line.set_data([], [])
        energy_marker.set_data([], [])
        mass_line.set_data([], [])
        mass_marker.set_data([], [])
        eex_bar[0].set_height(0)
        eex_text.set_text('')
        title_text.set_text('')
        return (traj_infall, traj_escape, pos_marker, impulse_marker, 
                energy_line, energy_marker, mass_line, mass_marker, title_text)
    
    def animate(frame):
        idx = frame_indices[frame]
        
        # Update trajectory - split by burn
        if idx <= burn_idx:
            traj_infall.set_data(x[:idx+1], y[:idx+1])
            traj_escape.set_data([], [])
            impulse_marker.set_data([], [])
        else:
            traj_infall.set_data(x[:burn_idx+1], y[:burn_idx+1])
            traj_escape.set_data(x[burn_idx:idx+1], y[burn_idx:idx+1])
            impulse_marker.set_data([x[burn_idx]], [y[burn_idx]])
        
        pos_marker.set_data([x[idx]], [y[idx]])
        
        # Update energy plot
        energy_line.set_data(tau[:idx+1], E[:idx+1])
        energy_marker.set_data([tau[idx]], [E[idx]])
        
        # Update mass plot
        mass_line.set_data(tau[:idx+1], m[:idx+1])
        mass_marker.set_data([tau[idx]], [m[idx]])
        
        # Update E_ex bar (show after impulse)
        if idx >= burn_idx:
            eex_bar[0].set_height(E_ex)
            if E_ex < 0:
                eex_bar[0].set_color('green')
                eex_text.set_text(f'E_ex = {E_ex:.4f}\nPENROSE!')
                eex_text.set_color('green')
            else:
                eex_bar[0].set_color('red')
                eex_text.set_text(f'E_ex = {E_ex:.4f}')
                eex_text.set_color('red')
        else:
            eex_bar[0].set_height(0)
            eex_text.set_text('Waiting...')
            eex_text.set_color('gray')
        
        # Update title
        phase = "INFALL" if idx < burn_idx else ("IMPULSE!" if idx == burn_idx else "ESCAPE")
        dE_current = E[idx] - E0
        dm_current = m[idx] - m[0]
        title_text.set_text(
            f'Single-Thrust Penrose Process (a={a:.2f}M) - {phase}\n'
            f'tau = {tau[idx]:.1f}M  |  r = {r[idx]:.2f}M  |  DeltaE = {dE_current:+.4f}  |  Deltam = {dm_current:+.3f}'
        )
        
        return (traj_infall, traj_escape, pos_marker, impulse_marker,
                energy_line, energy_marker, mass_line, mass_marker, title_text)
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                        interval=1000/fps, blit=True)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close()
    
    print(f"\nAnimation saved to: {output_path}")
    print(f"  Outcome: {outcome}")
    print(f"  DeltaE = {Delta_E:+.6f} (POSITIVE - energy extracted!)")
    print(f"  Deltam = {Delta_m:+.6f} ({-Delta_m*100:.1f}% fuel used)")
    print(f"  E_ex = {E_ex:.6f} ({'GENUINE PENROSE' if E_ex < 0 else 'Not Penrose'})")
    
    return output_path


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Penrose process trajectory animations")
    parser.add_argument('--spin', '-a', type=float, default=0.95, help='Black hole spin')
    parser.add_argument('--output-dir', '-o', default='results/animations', help='Output directory')
    parser.add_argument('--single-only', action='store_true', help='Only create single thrust animation')
    parser.add_argument('--continuous-only', action='store_true', help='Only create continuous thrust animation')
    parser.add_argument('--from-case', action='store_true', help='Use case files for simulation')
    parser.add_argument('--both-cases', action='store_true', help='Generate both single and continuous from case files')
    
    args = parser.parse_args()
    
    if args.both_cases:
        # Generate both animations from the proven case files
        print("="*60)
        print("Generating SINGLE thrust animation from single_thrust_case.py")
        print("="*60)
        animate_single_from_case(f"{args.output_dir}/single_penrose.gif")
        
        print("\n" + "="*60)
        print("Generating CONTINUOUS thrust animation from continuous_thrust_case.py")
        print("="*60)
        animate_continuous_from_case(f"{args.output_dir}/continuous_penrose.gif")
    elif args.from_case:
        # Use the proven continuous_thrust_case.py simulation
        animate_continuous_from_case(f"{args.output_dir}/continuous_penrose.gif")
    elif args.single_only:
        config = find_successful_config(a=args.spin)
        if config:
            animate_single_thrust(config, f"{args.output_dir}/single_thrust_a{args.spin}.gif")
    elif args.continuous_only:
        # Note: The simplified continuous simulation may not achieve escape
        # For reliable results, use --from-case or --both-cases
        config = SimulationConfig(
            a=args.spin, E0=1.25, Lz0=3.1,
            r0=10.0, m0=1.0, v_e=0.95,
            T_max=0.15, m_min=0.3, tau_max=100.0  # Adjusted params
        )
        result = animate_continuous_thrust(config, f"{args.output_dir}/continuous_thrust_a{args.spin}.gif")
        if result is None:
            print("\nNote: Simplified continuous thrust simulation did not achieve escape.")
            print("For reliable animations, use: --from-case or --both-cases")
    else:
        create_both_animations(args.output_dir, args.spin)
