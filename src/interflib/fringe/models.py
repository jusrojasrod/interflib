"""Interference model and fit-quality metrics for fringe-profile fitting.
"""
from dataclasses import dataclass

import numpy as np


def interference_model(x, A, w, phi, C):
    """Two-beam interference intensity ``A * sin(w*x + phi) + C``."""
    return A * np.sin(w * x + phi) + C


@dataclass
class FitQuality:
    """Goodness-of-fit metrics for a sinusoidal fringe-profile fit."""

    r2: float
    rmse: float
    phase_error: float          # standard uncertainty of the phase, radians
    is_flat_line: bool = False


def fit_quality(x, data, params, pcov) -> FitQuality:
    """Compute goodness-of-fit metrics for one fitted profile.

    Parameters
    ----------
    x : numpy.ndarray
        Sample positions.
    data : numpy.ndarray
        Measured profile values.
    params : sequence
        Fitted ``(A, w, phi, C)``.
    pcov : numpy.ndarray
        Parameter covariance matrix returned by ``curve_fit``; entry [2] is the phase.
    """
    residuals = data - interference_model(x, *params)

    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((data - np.mean(data)) ** 2))

    is_flat_line = ss_tot == 0
    r2 = float("-inf") if is_flat_line else 1 - (ss_res / ss_tot)
    phase_error = float("inf") if np.isinf(pcov).any() else float(np.sqrt(np.diag(pcov))[2])

    return FitQuality(r2=r2, rmse=rmse, phase_error=phase_error, is_flat_line=is_flat_line)
