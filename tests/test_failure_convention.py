"""
Test failure-counting convention: integration failures count as captures.

Verifies that the conservative convention described in the paper (Sec. IV.D,
lines 260, 689) is applied consistently across all modules:
- failures are never counted as escapes or Penrose successes
- NaN efficiency values from failures are excluded from averaging
- is_penrose_success rejects non-ESCAPE outcomes
"""

import math
import numpy as np
from experiments.trajectory_classifier import (
    TrajectoryOutcome, TrajectoryResult, is_penrose_success,
)

def test_penrose_success_requires_escape():
    """Non-ESCAPE outcomes must never be classified as Penrose success."""
    for outcome in TrajectoryOutcome:
        r = TrajectoryResult(outcome=outcome, Delta_E=0.05)
        if outcome == TrajectoryOutcome.ESCAPE:
            assert is_penrose_success(r) is True
        else:
            assert is_penrose_success(r) is False, (
                f"{outcome.name} should not be Penrose success"
            )


def test_penrose_success_requires_positive_delta_e():
    """Escape with zero or negative DeltaE is not Penrose extraction."""
    for dE in [0.0, -0.01, -1.0]:
        r = TrajectoryResult(outcome=TrajectoryOutcome.ESCAPE, Delta_E=dE)
        assert is_penrose_success(r) is False, (
            f"DeltaE={dE} should not be Penrose success"
        )


def test_penrose_success_rejects_nan_delta_e():
    """NaN or None DeltaE must be rejected."""
    r_nan = TrajectoryResult(outcome=TrajectoryOutcome.ESCAPE, Delta_E=float('nan'))
    assert is_penrose_success(r_nan) is False

    r_none = TrajectoryResult(outcome=TrajectoryOutcome.ESCAPE)
    r_none.Delta_E = None
    assert is_penrose_success(r_none) is False


def test_penrose_success_positive_case():
    """Escape with positive DeltaE is Penrose success."""
    r = TrajectoryResult(outcome=TrajectoryOutcome.ESCAPE, Delta_E=0.013)
    assert is_penrose_success(r) is True


def test_exception_handler_returns_nan_efficiency():
    """
    Verify that run_single_simulation exception path returns NaN
    for numerical fields (not 0.0) so they are excluded from averaging.
    """
    from experiments.regenerate_sweep_data import run_single_simulation

    # Missing required key triggers KeyError inside run_single_simulation
    bad_params = {
        'a': 0.95,
        'E0': 1.2,
        # 'Lz0' deliberately omitted — causes KeyError
        'v_e': 0.95,
        'delta_m': 0.2,
    }
    result = run_single_simulation(bad_params)

    assert result['is_escape'] is False, "Failure must not be escape"
    assert result['is_penrose'] is False, "Failure must not be Penrose"
    assert math.isnan(result['Delta_E']), (
        f"Delta_E should be NaN for failures, got {result['Delta_E']}"
    )
    assert math.isnan(result['eta_cum']), (
        f"eta_cum should be NaN for failures, got {result['eta_cum']}"
    )


def test_generate_all_data_exception_returns_nan():
    """Verify generate_all_data exception path returns NaN for failed simulations."""
    from experiments.generate_all_data import run_single_sim

    bad_params = {
        'a': 0.95,
        'E0': 1.2,
        # 'Lz0' deliberately omitted — causes KeyError
        'v_e': 0.95,
        'delta_m': 0.2,
    }
    result = run_single_sim(bad_params)

    assert result['is_escape'] is False, "Failure must not be escape"
    assert result['is_penrose'] is False, "Failure must not be Penrose"
    assert math.isnan(result['Delta_E']), (
        f"Delta_E should be NaN for failures, got {result['Delta_E']}"
    )
    assert math.isnan(result['eta_cum']), (
        f"eta_cum should be NaN for failures, got {result['eta_cum']}"
    )


def test_nan_excluded_from_efficiency_averaging():
    """NaN values must not pollute efficiency means."""
    eta_vals_raw = [0.05, float('nan'), 0.06, float('nan'), 0.07]
    # The filtering pattern used in regenerate_sweep_data.py
    eta_vals = [v for v in eta_vals_raw if np.isfinite(v) and v > 0]
    mean_eta = np.mean(eta_vals)
    assert abs(mean_eta - 0.06) < 1e-10, (
        f"Mean should be 0.06, got {mean_eta}"
    )
    assert len(eta_vals) == 3


if __name__ == '__main__':
    test_penrose_success_requires_escape()
    test_penrose_success_requires_positive_delta_e()
    test_penrose_success_rejects_nan_delta_e()
    test_penrose_success_positive_case()
    test_exception_handler_returns_nan_efficiency()
    test_generate_all_data_exception_returns_nan()
    test_nan_excluded_from_efficiency_averaging()
    print("All failure-convention tests passed.")
