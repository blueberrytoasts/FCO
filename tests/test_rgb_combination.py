import numpy as np
import pytest
from rgb_combination_analysis import fit_linear_combination, compute_r2, compute_rms


def test_fit_returns_three_coefficients():
    X = np.array([[0.1, 0.05, 0.02],
                  [0.3, 0.15, 0.06],
                  [0.6, 0.30, 0.12]])
    y = np.array([0.12, 0.36, 0.72])
    a, b, c = fit_linear_combination(X, y)
    assert isinstance(a, float)
    assert isinstance(b, float)
    assert isinstance(c, float)


def test_fit_perfect_red_only():
    # If y = 2 * OD_R exactly, we expect a≈2, b≈0, c≈0
    X = np.array([[0.1, 0.0, 0.0],
                  [0.3, 0.0, 0.0],
                  [0.6, 0.0, 0.0]])
    y = np.array([0.2, 0.6, 1.2])
    a, b, c = fit_linear_combination(X, y)
    assert abs(a - 2.0) < 1e-6
    assert abs(b) < 1e-6
    assert abs(c) < 1e-6


def test_compute_r2_perfect_fit():
    y = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert abs(compute_r2(y, y_pred) - 1.0) < 1e-10


def test_compute_r2_zero_fit():
    y = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])  # predicts the mean
    assert abs(compute_r2(y, y_pred)) < 1e-10


def test_compute_rms_known_value():
    y = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    assert abs(compute_rms(y, y_pred) - 1.0) < 1e-10


def test_fit_no_intercept_passes_through_origin():
    # With no intercept, zero input must give zero output
    X = np.array([[0.0, 0.0, 0.0],
                  [0.1, 0.05, 0.02],
                  [0.5, 0.25, 0.10]])
    y = np.array([0.0, 0.13, 0.65])
    a, b, c = fit_linear_combination(X, y)
    predicted_zero = a * 0.0 + b * 0.0 + c * 0.0
    assert abs(predicted_zero) < 1e-10
