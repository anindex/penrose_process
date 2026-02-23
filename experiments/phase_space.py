"""
Phase-Space Visualization Module
=================================
Visualize trajectory profiles in various phase-space representations.

Includes:
- (r, p_r) phase portraits
- (E, L_z) outcome maps
- Effective potential landscapes
- Periapsis depth heatmaps
- Extraction zone boundaries

Uses PRD-style publication quality settings.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple, Any

from kerr_utils import (
    COLORS, PRD_SINGLE_COL, PRD_DOUBLE_COL, setup_prd_style,
    kerr_metric_components, horizon_radius, ergosphere_radius, isco_radius
)
from experiments.trajectory_classifier import (
    OrbitProfile, TrajectoryOutcome,
    classify_orbit, compute_effective_potential, find_turning_points
)


# =============================================================================
# COLORMAP DEFINITIONS
# =============================================================================

# Profile colormap: distinct colors for each orbit type
PROFILE_COLORS = {
    OrbitProfile.FLYBY_DEEP_ERGOSPHERE: '#009E73',      # Green - best for Penrose
    OrbitProfile.FLYBY_SHALLOW_ERGOSPHERE: '#56B4E9',   # Light blue
    OrbitProfile.FLYBY_OUTSIDE_ERGOSPHERE: '#0072B2',   # Blue
    OrbitProfile.BOUND_STABLE: '#E69F00',               # Orange
    OrbitProfile.BOUND_UNSTABLE: '#F0E442',             # Yellow
    OrbitProfile.PLUNGE: '#D55E00',                     # Vermilion - dangerous
    OrbitProfile.FORBIDDEN: '#999999',                  # Gray
    OrbitProfile.CIRCULAR: '#CC79A7',                   # Purple
}

OUTCOME_COLORS = {
    TrajectoryOutcome.ESCAPE: '#009E73',          # Green
    TrajectoryOutcome.CAPTURE: '#D55E00',         # Vermilion
    TrajectoryOutcome.BOUND: '#E69F00',           # Orange
    TrajectoryOutcome.STALLED: '#F0E442',         # Yellow
    TrajectoryOutcome.INTEGRATION_FAILURE: '#999999',  # Gray
}


def get_profile_cmap():
    """Create colormap for orbit profile visualization."""
    colors = [PROFILE_COLORS.get(OrbitProfile(i), '#999999') 
              for i in range(1, len(OrbitProfile) + 1)]
    return ListedColormap(colors)


# =============================================================================
# EFFECTIVE POTENTIAL VISUALIZATION
# =============================================================================

def plot_effective_potential(E: float, Lz: float, a: float, M: float = 1.0,
                              r_range: Tuple[float, float] = (1.2, 20.0),
                              n_points: int = 500,
                              ax: Optional[plt.Axes] = None,
                              show_turning_points: bool = True) -> plt.Axes:
    """
    Plot the effective potential for radial motion.
    
    Motion is allowed where V_eff < 0 (shaded region).
    Turning points occur where V_eff = 0.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(PRD_SINGLE_COL * 1.5, 3))
    
    r_plus = horizon_radius(a, M)
    r_erg = ergosphere_radius(np.pi/2, a, M)
    
    r_vals = np.linspace(max(r_range[0], r_plus + 0.01), r_range[1], n_points)
    V_vals = np.array([compute_effective_potential(E, Lz, r, a, M) for r in r_vals])
    
    # Plot potential
    ax.plot(r_vals, V_vals, color=COLORS['blue'], lw=1.5, label=r'$V_{\rm eff}(r)$')
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    
    # Shade allowed region (V < 0)
    ax.fill_between(r_vals, V_vals, 0, where=(V_vals < 0),
                    color=COLORS['blue'], alpha=0.15, label='Allowed')
    
    # Mark key radii
    ax.axvline(r_plus, color='black', ls='-', lw=1.0, label=r'$r_+$')
    ax.axvline(r_erg, color=COLORS['vermilion'], ls='--', lw=1.0, label=r'$r_{\rm erg}$')
    
    # Find and mark turning points
    if show_turning_points:
        periapses, apoapses = find_turning_points(E, Lz, a, M)
        for r_p in periapses:
            ax.plot(r_p, 0, 'o', color=COLORS['green'], ms=8, 
                   label='Periapsis' if r_p == periapses[0] else '', zorder=5)
        for r_a in apoapses:
            ax.plot(r_a, 0, 's', color=COLORS['orange'], ms=7,
                   label='Apoapsis' if r_a == apoapses[0] else '', zorder=5)
    
    ax.set_xlabel(r'$r/M$')
    ax.set_ylabel(r'$V_{\rm eff}$')
    ax.set_title(f'$E = {E:.2f},\\ L_z = {Lz:.2f},\\ a = {a:.2f}$')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(r_range)
    
    return ax


def plot_effective_potential_family(Lz: float, a: float, M: float = 1.0,
                                     E_values: Optional[List[float]] = None,
                                     ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot effective potential for multiple energy values.
    
    Shows how orbit type changes with energy.
    """
    if E_values is None:
        E_values = [0.95, 1.0, 1.1, 1.2, 1.5]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(PRD_SINGLE_COL * 1.5, 3))
    
    r_plus = horizon_radius(a, M)
    r_erg = ergosphere_radius(np.pi/2, a, M)
    
    r_vals = np.linspace(r_plus + 0.01, 15.0, 500)
    
    cmap = plt.cm.viridis
    for i, E in enumerate(E_values):
        V_vals = np.array([compute_effective_potential(E, Lz, r, a, M) for r in r_vals])
        color = cmap(i / (len(E_values) - 1))
        ax.plot(r_vals, V_vals, color=color, lw=1.2, label=f'$E = {E:.2f}$')
    
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.axvline(r_plus, color='black', ls='-', lw=1.0)
    ax.axvline(r_erg, color=COLORS['vermilion'], ls='--', lw=1.0)
    
    ax.set_xlabel(r'$r/M$')
    ax.set_ylabel(r'$V_{\rm eff}$')
    ax.set_title(f'$L_z = {Lz:.2f},\\ a = {a:.2f}$')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(r_plus, 15)
    ax.set_ylim(-0.5, 0.5)
    
    return ax


# =============================================================================
# PHASE PORTRAIT (r, p_r)
# =============================================================================

def plot_phase_portrait(trajectory_data: Dict[str, np.ndarray],
                        a: float, M: float = 1.0,
                        ax: Optional[plt.Axes] = None,
                        color_by: str = 'time',
                        show_horizon: bool = True) -> plt.Axes:
    """
    Plot trajectory in (r, p_r) phase space.
    
    Parameters
    ----------
    trajectory_data : dict
        Must contain 'r', 'pr', and optionally 'tau' or 'E_ex' for coloring
    color_by : str
        'time', 'E_ex', or 'single' (uniform color)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(PRD_SINGLE_COL * 1.5, 3))
    
    r = trajectory_data['r']
    pr = trajectory_data['pr']
    
    r_plus = horizon_radius(a, M)
    r_erg = ergosphere_radius(np.pi/2, a, M)
    
    if color_by == 'time' and 'tau' in trajectory_data:
        tau = trajectory_data['tau']
        scatter = ax.scatter(r, pr, c=tau, cmap='viridis', s=2, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label=r'$\tau/M$')
    elif color_by == 'E_ex' and 'E_ex' in trajectory_data:
        E_ex = trajectory_data['E_ex']
        # Extend E_ex to match trajectory length if needed
        if len(E_ex) < len(r):
            E_ex_full = np.zeros(len(r))
            E_ex_full[:len(E_ex)] = E_ex
            E_ex = E_ex_full
        scatter = ax.scatter(r, pr, c=E_ex, cmap='RdYlGn_r', s=2, alpha=0.7,
                            vmin=-0.5, vmax=0.5)
        plt.colorbar(scatter, ax=ax, label=r'$E_{\rm ex}$')
    else:
        ax.plot(r, pr, color=COLORS['blue'], lw=1.2)
    
    # Mark horizon and ergosphere
    if show_horizon:
        ax.axvline(r_plus, color='black', ls='-', lw=1.5, label=r'$r_+$')
        ax.axvline(r_erg, color=COLORS['vermilion'], ls='--', lw=1.2, label=r'$r_{\rm erg}$')
    
    ax.set_xlabel(r'$r/M$')
    ax.set_ylabel(r'$p_r$')
    ax.legend(fontsize=7)
    
    return ax


def plot_multiple_phase_portraits(trajectories: List[Dict],
                                   a: float, M: float = 1.0,
                                   labels: Optional[List[str]] = None,
                                   ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Overlay multiple trajectories in (r, p_r) space.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(PRD_DOUBLE_COL * 0.5, 4))
    
    if labels is None:
        labels = [f'Traj {i+1}' for i in range(len(trajectories))]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    for traj, label, color in zip(trajectories, labels, colors):
        ax.plot(traj['r'], traj['pr'], color=color, lw=1.0, label=label, alpha=0.8)
    
    r_plus = horizon_radius(a, M)
    r_erg = ergosphere_radius(np.pi/2, a, M)
    ax.axvline(r_plus, color='black', ls='-', lw=1.5)
    ax.axvline(r_erg, color=COLORS['vermilion'], ls='--', lw=1.2)
    
    ax.set_xlabel(r'$r/M$')
    ax.set_ylabel(r'$p_r$')
    ax.legend(fontsize=7, loc='best')
    
    return ax


# =============================================================================
# (E, Lz) PARAMETER SPACE MAPS
# =============================================================================

def plot_orbit_profile_map(a: float, M: float = 1.0,
                            E_range: Tuple[float, float] = (0.9, 2.5),
                            Lz_range: Tuple[float, float] = (0.0, 6.0),
                            n_E: int = 100, n_Lz: int = 100,
                            ax: Optional[plt.Axes] = None,
                            show_contours: bool = True) -> Tuple[plt.Axes, np.ndarray]:
    """
    Create heatmap of orbit profiles in (E, Lz) parameter space.
    
    Color indicates orbit classification.
    """
    setup_prd_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(PRD_DOUBLE_COL * 0.6, 4))
    
    E_vals = np.linspace(E_range[0], E_range[1], n_E)
    Lz_vals = np.linspace(Lz_range[0], Lz_range[1], n_Lz)
    
    profile_grid = np.zeros((n_E, n_Lz))
    r_peri_grid = np.full((n_E, n_Lz), np.nan)
    
    for i, E in enumerate(E_vals):
        for j, Lz in enumerate(Lz_vals):
            props = classify_orbit(E, Lz, a, M)
            profile_grid[i, j] = props.profile.value
            if props.r_periapsis is not None:
                r_peri_grid[i, j] = props.r_periapsis
    
    # Custom colormap
    cmap = get_profile_cmap()
    bounds = np.arange(0.5, len(OrbitProfile) + 1.5)
    norm = BoundaryNorm(bounds, cmap.N)
    
    im = ax.imshow(profile_grid.T, origin='lower', aspect='auto',
                   extent=[E_range[0], E_range[1], Lz_range[0], Lz_range[1]],
                   cmap=cmap, norm=norm)
    
    # Add periapsis contours
    if show_contours:
        r_erg = ergosphere_radius(np.pi/2, a, M)
        r_plus = horizon_radius(a, M)
        
        E_grid, Lz_grid = np.meshgrid(E_vals, Lz_vals)
        
        contour_levels = [r_plus + 0.1, 0.85 * r_erg, r_erg]
        contour_colors = ['black', COLORS['green'], COLORS['vermilion']]
        contour_labels = [r'$r_+ + 0.1$', r'Extraction', r'$r_{\rm erg}$']
        
        for level, color, label in zip(contour_levels, contour_colors, contour_labels):
            cs = ax.contour(E_vals, Lz_vals, r_peri_grid.T, levels=[level],
                           colors=[color], linestyles=['--'], linewidths=[1.0])
    
    # Bound/unbound boundary
    ax.axvline(1.0, color='white', ls=':', lw=1.0, alpha=0.7)
    ax.text(1.02, Lz_range[1] * 0.95, 'unbound ->', color='white', fontsize=7, va='top')
    
    ax.set_xlabel(r'Energy $E$')
    ax.set_ylabel(r'Angular momentum $L_z$')
    ax.set_title(f'Orbit profiles: $a/M = {a}$')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PROFILE_COLORS[OrbitProfile.FLYBY_DEEP_ERGOSPHERE], 
              label='Deep ergo'),
        Patch(facecolor=PROFILE_COLORS[OrbitProfile.FLYBY_SHALLOW_ERGOSPHERE], 
              label='Shallow ergo'),
        Patch(facecolor=PROFILE_COLORS[OrbitProfile.PLUNGE], 
              label='Plunge'),
        Patch(facecolor=PROFILE_COLORS[OrbitProfile.FORBIDDEN], 
              label='Forbidden'),
    ]
    ax.legend(handles=legend_elements, fontsize=6, loc='lower right')
    
    return ax, profile_grid


def plot_periapsis_depth_map(a: float, M: float = 1.0,
                               E_range: Tuple[float, float] = (1.0, 2.0),
                               Lz_range: Tuple[float, float] = (2.0, 5.0),
                               n_E: int = 80, n_Lz: int = 80,
                               ax: Optional[plt.Axes] = None) -> Tuple[plt.Axes, np.ndarray]:
    """
    Heatmap of periapsis depth in (E, Lz) space.
    
    Darker = deeper periapsis (closer to horizon).
    """
    setup_prd_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(PRD_DOUBLE_COL * 0.6, 4))
    
    E_vals = np.linspace(E_range[0], E_range[1], n_E)
    Lz_vals = np.linspace(Lz_range[0], Lz_range[1], n_Lz)
    
    r_peri_grid = np.full((n_E, n_Lz), np.nan)
    
    r_plus = horizon_radius(a, M)
    r_erg = ergosphere_radius(np.pi/2, a, M)
    
    for i, E in enumerate(E_vals):
        for j, Lz in enumerate(Lz_vals):
            props = classify_orbit(E, Lz, a, M)
            if props.r_periapsis is not None and props.r_periapsis > r_plus:
                r_peri_grid[i, j] = props.r_periapsis
    
    im = ax.imshow(r_peri_grid.T, origin='lower', aspect='auto',
                   extent=[E_range[0], E_range[1], Lz_range[0], Lz_range[1]],
                   cmap='viridis_r', vmin=r_plus, vmax=r_erg + 1)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'Periapsis $r_{\rm peri}/M$')
    
    # Mark key contours
    E_grid, Lz_grid = np.meshgrid(E_vals, Lz_vals)
    
    cs1 = ax.contour(E_vals, Lz_vals, r_peri_grid.T, levels=[r_erg],
                    colors=[COLORS['vermilion']], linestyles=['--'], linewidths=[1.5])
    ax.clabel(cs1, fmt=r'$r_{\rm erg}$', fontsize=7)
    
    cs2 = ax.contour(E_vals, Lz_vals, r_peri_grid.T, levels=[0.85 * r_erg],
                    colors=[COLORS['green']], linestyles=['-'], linewidths=[1.5])
    ax.clabel(cs2, fmt='Extract', fontsize=7)
    
    ax.set_xlabel(r'Energy $E$')
    ax.set_ylabel(r'Angular momentum $L_z$')
    ax.set_title(f'Periapsis depth: $a/M = {a}$')
    
    return ax, r_peri_grid


# =============================================================================
# SPIN COMPARISON FIGURE
# =============================================================================

def plot_spin_comparison(spins: Optional[List[float]] = None,
                          E_range: Tuple[float, float] = (1.0, 2.0),
                          Lz_range: Tuple[float, float] = (2.0, 5.0),
                          n_E: int = 50, n_Lz: int = 50,
                          figsize: Tuple[float, float] = None) -> plt.Figure:
    """
    Compare orbit profiles across different spin values.
    
    Creates a multi-panel figure showing how the parameter space
    changes with black hole spin.
    """
    setup_prd_style()
    
    if spins is None:
        spins = [0.7, 0.9, 0.95, 0.99]
    
    n_spins = len(spins)
    if figsize is None:
        figsize = (PRD_DOUBLE_COL, 2.5 * n_spins)
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_spins, 2, width_ratios=[1, 1], hspace=0.3, wspace=0.25)
    
    for i, a in enumerate(spins):
        r_plus = horizon_radius(a)
        r_erg = ergosphere_radius(np.pi/2, a)
        
        # Left: orbit profile map
        ax_left = fig.add_subplot(gs[i, 0])
        plot_orbit_profile_map(a, E_range=E_range, Lz_range=Lz_range,
                               n_E=n_E, n_Lz=n_Lz, ax=ax_left, show_contours=False)
        ax_left.set_title(f'$a/M = {a}$: $r_+ = {r_plus:.3f}$')
        
        # Right: periapsis depth
        ax_right = fig.add_subplot(gs[i, 1])
        plot_periapsis_depth_map(a, E_range=E_range, Lz_range=Lz_range,
                                  n_E=n_E, n_Lz=n_Lz, ax=ax_right)
        ax_right.set_title(f'Ergosphere width: {r_erg - r_plus:.3f}M')
    
    fig.suptitle('Orbit Classification vs Black Hole Spin', fontsize=11, y=1.01)
    
    return fig


# =============================================================================
# TRAJECTORY VISUALIZATION
# =============================================================================

def plot_trajectory_xy(trajectory_data: Dict[str, np.ndarray],
                        a: float, M: float = 1.0,
                        ax: Optional[plt.Axes] = None,
                        zoom_ergosphere: bool = False,
                        show_thrust_regions: bool = True) -> plt.Axes:
    """
    Plot trajectory in equatorial (x, y) plane.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(PRD_SINGLE_COL * 1.5, 4))
    
    r = trajectory_data['r']
    phi = trajectory_data['phi']
    
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    r_plus = horizon_radius(a, M)
    r_erg = ergosphere_radius(np.pi/2, a, M)
    
    # Plot trajectory
    if show_thrust_regions and 'thrust_active' in trajectory_data:
        thrust = trajectory_data['thrust_active']
        # Plot coasting segments
        ax.plot(x[~thrust], y[~thrust], '.', color=COLORS['blue'], ms=1, alpha=0.5)
        # Plot thrusting segments
        ax.plot(x[thrust], y[thrust], '.', color=COLORS['orange'], ms=2, alpha=0.8)
    else:
        ax.plot(x, y, color=COLORS['blue'], lw=1.0)
    
    # Horizon
    horizon = Circle((0, 0), r_plus, color='black', zorder=10)
    ax.add_patch(horizon)
    
    # Ergosphere
    ergo = Circle((0, 0), r_erg, fill=False, 
                  color=COLORS['vermilion'], ls='--', lw=1.5)
    ax.add_patch(ergo)
    
    # Mark start and end
    ax.plot(x[0], y[0], 'o', color=COLORS['green'], ms=8, label='Start', zorder=5)
    ax.plot(x[-1], y[-1], 's', color=COLORS['purple'], ms=8, label='End', zorder=5)
    
    ax.set_xlabel(r'$x/M$')
    ax.set_ylabel(r'$y/M$')
    ax.set_aspect('equal')
    ax.legend(fontsize=7)
    
    if zoom_ergosphere:
        margin = 1.5
        ax.set_xlim(-r_erg - margin, r_erg + margin)
        ax.set_ylim(-r_erg - margin, r_erg + margin)
    
    return ax


# =============================================================================
# COMBINED ANALYSIS FIGURE
# =============================================================================

def create_trajectory_analysis_figure(trajectory_data: Dict[str, np.ndarray],
                                       a: float, M: float = 1.0,
                                       E: float = None, Lz: float = None,
                                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive trajectory analysis figure with multiple panels.
    
    Panels:
    (a) x-y trajectory
    (b) r(tau) with thrust regions
    (c) (r, p_r) phase portrait
    (d) E_ex vs r (if available)
    """
    setup_prd_style()
    
    fig = plt.figure(figsize=(PRD_DOUBLE_COL, 5))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
    
    r = trajectory_data['r']
    
    # (a) Trajectory
    ax_a = fig.add_subplot(gs[0, 0])
    plot_trajectory_xy(trajectory_data, a, M, ax=ax_a, zoom_ergosphere=True)
    ax_a.set_title('(a) Equatorial trajectory')
    
    # (b) r(tau)
    ax_b = fig.add_subplot(gs[0, 1])
    if 'tau' in trajectory_data:
        tau = trajectory_data['tau']
        ax_b.plot(tau, r, color=COLORS['blue'], lw=1.2)
        ax_b.axhline(ergosphere_radius(np.pi/2, a, M), color=COLORS['vermilion'], 
                     ls='--', lw=1.0, label=r'$r_{\rm erg}$')
        ax_b.axhline(horizon_radius(a, M), color='black', ls='-', lw=1.0, label=r'$r_+$')
        ax_b.set_xlabel(r'$\tau/M$')
        ax_b.set_ylabel(r'$r/M$')
        ax_b.legend(fontsize=7)
    ax_b.set_title('(b) Radial evolution')
    
    # (c) Phase portrait
    ax_c = fig.add_subplot(gs[1, 0])
    if 'pr' in trajectory_data:
        plot_phase_portrait(trajectory_data, a, M, ax=ax_c, color_by='single')
    ax_c.set_title('(c) Phase portrait')
    
    # (d) E_ex analysis
    ax_d = fig.add_subplot(gs[1, 1])
    if 'E_ex' in trajectory_data and 'r_ex' in trajectory_data:
        E_ex = trajectory_data['E_ex']
        r_ex = trajectory_data['r_ex']
        colors = [COLORS['green'] if e < 0 else COLORS['vermilion'] for e in E_ex]
        ax_d.scatter(r_ex, E_ex, c=colors, s=10, alpha=0.6)
        ax_d.axhline(0, color='gray', ls='--', lw=0.8)
        ax_d.set_xlabel(r'$r/M$')
        ax_d.set_ylabel(r'$E_{\rm ex}$')
        
        n_neg = np.sum(np.array(E_ex) < 0)
        ax_d.text(0.05, 0.95, f'{n_neg}/{len(E_ex)} with $E_{{\\rm ex}} < 0$',
                  transform=ax_d.transAxes, fontsize=7, va='top')
    ax_d.set_title('(d) Exhaust energy')
    
    # Overall title
    if E is not None and Lz is not None:
        fig.suptitle(f'Trajectory Analysis: $E = {E:.2f},\\ L_z = {Lz:.2f},\\ a = {a:.2f}$',
                     fontsize=11, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    setup_prd_style()
    
    print("="*70)
    print("PHASE-SPACE VISUALIZATION - Quick Test")
    print("="*70)
    
    a = 0.95
    M = 1.0
    
    # Test 1: Effective potential
    print("\n1. Generating effective potential plot...")
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    plot_effective_potential(E=1.2, Lz=3.0, a=a, M=M, ax=ax1)
    plt.tight_layout()
    plt.savefig('test_effective_potential.png', dpi=150)
    print("   Saved: test_effective_potential.png")
    
    # Test 2: Orbit profile map
    print("\n2. Generating orbit profile map...")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    plot_orbit_profile_map(a=a, ax=ax2, n_E=50, n_Lz=50)
    plt.tight_layout()
    plt.savefig('test_orbit_profile_map.png', dpi=150)
    print("   Saved: test_orbit_profile_map.png")
    
    # Test 3: Periapsis depth map
    print("\n3. Generating periapsis depth map...")
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    plot_periapsis_depth_map(a=a, ax=ax3, n_E=50, n_Lz=50)
    plt.tight_layout()
    plt.savefig('test_periapsis_depth.png', dpi=150)
    print("   Saved: test_periapsis_depth.png")
    
    # Test 4: Spin comparison
    print("\n4. Generating spin comparison figure...")
    fig4 = plot_spin_comparison(spins=[0.7, 0.95], n_E=30, n_Lz=30)
    plt.savefig('test_spin_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved: test_spin_comparison.png")
    
    print("\n" + "="*70)
    print("Phase-space visualization ready!")
    print("="*70)
    
    plt.show()
