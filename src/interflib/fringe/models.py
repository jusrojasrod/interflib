"""Interference model and fit-quality metrics for fringe-profile fitting.
"""
from dataclasses import dataclass

import numpy as np


def interference_model(x, A, w, phi, C):
    """Two-beam interference intensity ``A * sin(w*x + phi) + C``."""
    return A * np.sin(w * x + phi) + C


def sine_seed(profile, x, frequency):
    """Seed ``[A, w, phi, C]`` for fitting ``A*sin(w*x + phi) + C`` at a known frequency.

    At a fixed angular frequency ``w`` the model is linear in ``A*cos(phi)``,
    ``A*sin(phi)`` and ``C``, so a linear least-squares solve gives the amplitude, phase
    and offset that best match the profile at that frequency (the three-parameter
    known-frequency sine fit of IEEE Std 1057). Seeding the phase from the data, instead
    of a fixed value, keeps the non-linear fit from trading phase for frequency.
    """
    x = np.asarray(x, dtype=float)
    design = np.column_stack([np.sin(frequency * x), np.cos(frequency * x), np.ones_like(x)])
    (a, b, c), *_ = np.linalg.lstsq(design, np.asarray(profile, dtype=float), rcond=None)
    return [float(np.hypot(a, b)), float(frequency), float(np.arctan2(b, a)), float(c)]


def parabolic_vertex(left, centre, right) -> float:
    """Offset, in samples, of the vertex of the parabola through three equally spaced samples.

    The offset is measured from the centre sample (``|offset| <= 0.5`` when the centre is
    the extreme of the three); it is 0 when the samples are collinear or not finite. Pass
    logarithms to refine a Gaussian-like spectral peak.
    """
    curvature = left - 2 * centre + right
    if not np.isfinite(curvature) or curvature == 0:
        return 0.0
    return float(0.5 * (left - right) / curvature)


def canonical_sinusoid(A, w, phi):
    """Fold ``A*sin(w*x + phi)`` onto the branch ``A >= 0``, ``w > 0``; returns ``(A, w, phi)``.

    The model is invariant under ``(A, phi) -> (-A, phi + pi)`` and under
    ``(w, phi) -> (-w, pi - phi)``, so independent fits can land on different branches.
    A ``pi`` between two fits adds half a fringe to their phase difference, and on the
    ``w < 0`` branch the phase runs backwards along ``x``, turning ``F`` into ``1 - F``.
    On the canonical branch the phase always increases along ``+x``.
    """
    if w < 0:
        w, phi = -w, np.pi - phi
    if A < 0:
        A, phi = -A, phi + np.pi
    return A, w, phi


def signed_fraction(phase_difference, order_sign=1) -> float:
    """Fringe fraction in ``[0, 1)`` from a block-minus-platen phase difference, radians.

    ``order_sign`` relates the measured phase to the interference order, counted so that
    the block face, ``L`` above the platen, lies ``2 L / lambda`` orders above it: ``+1``
    when the phase increases with the order, ``-1`` when it decreases. With the wrong sign
    the fraction comes out as ``1 - F``. An intensity image cannot tell the two apart (the
    cosine is even), so the sign is a property of the set-up, determined on the bench.
    """
    if order_sign not in (1, -1):
        raise ValueError("order_sign must be +1 or -1, got %r" % (order_sign,))
    return float((order_sign * phase_difference / (2 * np.pi)) % 1.0)


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
