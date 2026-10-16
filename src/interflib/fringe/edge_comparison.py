"""Fringe fraction with the block and platen phases compared at a common column.

In a gauge-block interferogram the fringe fraction is the phase difference between the
block face and the platen it is wrung to, ``F = (phi_block - phi_platen) / 2 pi``. That
difference is only free of the instrument when both phases refer to the **same pixel**:
any term that shifts the fringes of both surfaces alike (reference-mirror tilt, air and
vibration piston) then cancels in the subtraction. Block and platen do not overlap,
however, so the phases are measured on either side of the block edge.

Comparing two strips at different columns, as a plain profile fit does, is exact only
for perfectly horizontal fringes. If the fringes are slightly rotated the phase also
changes along ``x`` with a gradient ``g_x`` common to both surfaces, and the strip
separation ``dx`` adds ``g_x * dx`` to ``F``.

This module measures the phase **column by column** instead. Each narrow column band is
fitted with a sinusoid along the rows (``y``), with the coordinate referred to an
evaluation row so the fitted phase is the phase at that row. On each side of the edge a
polynomial (a straight line by default) is fitted to the band phases, unwrapped along
``x``, and both are evaluated at the edge column. There the two phases share a pixel and
every common term cancels by construction, without needing the fringes to be
horizontal. The block-platen wedge does **not** cancel -- it is a real property of the
wrung assembly -- so ``F`` refers to the edge column at the evaluation row.

Image columns are ``x`` and rows ``y``; fringes are expected to be roughly horizontal
(spatial carrier along ``y``). The platen is assumed to lie left of the edge and the
block right of it; mirror the image for the opposite layout.

The sinusoidal model and the linear least-squares seed (a regional synchronous
detection at one frequency) follow Servin, Quiroga & Padilla, *Fringe Pattern Analysis
for Optical Metrology*, Wiley-VCH, 2014, ch. 2. Extrapolating the platen phase to the
point of the block where the fraction is read is the treatment discussed for phase-step
gauge-block interferometry by Decker, Schoedel & Boensch, *Metrologia* 41, L11-L17
(2004), doi:10.1088/0026-1394/41/3/L03.

This module has no I/O side effects: it takes and returns arrays.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from .models import fit_quality, interference_model


@dataclass
class ColumnPhases:
    """Phase at the evaluation row of a set of column bands.

    Attributes
    ----------
    x : numpy.ndarray
        Centre column of each band, px.
    phase : numpy.ndarray
        Fitted phase at the evaluation row, radians in ``[0, 2 pi)`` (NaN if the fit
        failed).
    phase_unwrapped : numpy.ndarray
        ``phase`` unwrapped along ``x`` over the converged bands.
    period : numpy.ndarray
        Fitted fringe period along ``y``, px per fringe.
    r2 : numpy.ndarray
        Coefficient of determination of each band fit.
    phase_error : numpy.ndarray
        Standard uncertainty of each fitted phase, radians.
    """

    x: np.ndarray
    phase: np.ndarray
    phase_unwrapped: np.ndarray
    period: np.ndarray
    r2: np.ndarray
    phase_error: np.ndarray


@dataclass
class EdgeFractionResult:
    """Fringe fraction compared at the edge column.

    Attributes
    ----------
    fraction : float
        ``F`` in ``[0, 1)`` at the edge column and evaluation row.
    phase_platen, phase_block : float
        Extrapolated phases at the edge, radians (unwrapped, so only their difference
        modulo ``2 pi`` is meaningful).
    slope_platen, slope_block : float
        Phase gradients along ``x`` at the edge, fringes per px (``g_p`` and ``g_b``).
    edge_x : float
        Edge column used, px.
    eval_row : float
        Evaluation row used, px.
    platen, block : ColumnPhases
        The band phases of each side, for diagnostics.
    """

    fraction: float
    phase_platen: float
    phase_block: float
    slope_platen: float
    slope_block: float
    edge_x: float
    eval_row: float
    platen: ColumnPhases
    block: ColumnPhases


def detect_edge_column(img, rows=(600, 1600), search=(1000, 1300), smooth=5) -> int:
    """Column of the block edge from the mean horizontal intensity jump.

    Along the edge the fringes of block and platen are out of step, so the mean
    absolute difference between neighbouring columns peaks there.

    Parameters
    ----------
    img : numpy.ndarray
        Interferogram, rows x columns.
    rows : tuple of int
        Row range ``(y0, y1)`` over which the difference is averaged.
    search : tuple of int
        Column range ``(x0, x1)`` searched for the edge.
    smooth : int
        Width of the moving average applied to the difference profile, px.

    Returns
    -------
    int
        Edge column: the boundary between the two columns of the largest jump.
    """
    y0, y1 = rows
    xa, xb = search
    jump = np.abs(np.diff(np.asarray(img, dtype=float)[y0:y1, xa:xb], axis=1)).mean(axis=0)
    jump = np.convolve(jump, np.ones(smooth) / smooth, mode="same")
    return xa + int(np.argmax(jump)) + 1


def edge_line_centre(img, approx, rows=(600, 1600), half=15) -> float:
    """Sub-pixel centre of the dark line that the block edge draws near ``approx``.

    The edge (chamfer and step) images as a dark band some 10 px wide. Its darkest
    column, refined by a parabola through the column means, is a physical feature that
    stays put when the fringes move or tilt, so it registers the edge more reproducibly
    than the column of largest intensity jump, which can fall on either flank.

    Parameters
    ----------
    img : numpy.ndarray
        Interferogram, rows x columns.
    approx : int
        Approximate edge column (for instance from :func:`detect_edge_column`).
    rows : tuple of int
        Row range ``(y0, y1)`` over which the column means are taken.
    half : int
        Half-width of the search window around ``approx``, px.

    Returns
    -------
    float
        Edge column, px.
    """
    y0, y1 = rows
    means = np.convolve(np.asarray(img, dtype=float)[y0:y1].mean(axis=0), np.ones(3) / 3, mode="same")
    seg = means[approx - half:approx + half + 1]
    i = int(np.argmin(seg))
    delta = 0.0
    if 0 < i < len(seg) - 1:
        curvature = seg[i - 1] - 2 * seg[i] + seg[i + 1]
        if curvature > 0:
            delta = 0.5 * (seg[i - 1] - seg[i + 1]) / curvature
    return float(approx - half + i + delta)


def _linear_seed(profile, y, frequency):
    # At a fixed frequency the sinusoid is linear in A*cos(phi), A*sin(phi) and C.
    design = np.column_stack([np.sin(frequency * y), np.cos(frequency * y), np.ones_like(y)])
    (a, b, c), *_ = np.linalg.lstsq(design, profile, rcond=None)
    return [float(np.hypot(a, b)), float(frequency), float(np.arctan2(b, a)), float(c)]


def fit_column_band(profile, y, period_guess):
    """Fit ``A*sin(w*y + phi) + C`` to one column-band profile.

    Parameters
    ----------
    profile : numpy.ndarray
        Intensity of the band, averaged across its columns, as a function of row.
    y : numpy.ndarray
        Row coordinate already referred to the evaluation row, so the fitted ``phi`` is
        the phase at that row.
    period_guess : float
        Starting fringe period along ``y``, px per fringe.

    Returns
    -------
    tuple
        ``(phi, period, r2, phase_error)`` with ``phi`` in ``[0, 2 pi)``; NaNs if the
        fit does not converge. A negative fitted amplitude is folded into the phase so
        that every band lies on the same branch.
    """
    profile = np.asarray(profile, dtype=float)
    try:
        params, cov = curve_fit(interference_model, y, profile,
                                p0=_linear_seed(profile, y, 2 * np.pi / period_guess), maxfev=5000)
    except RuntimeError:
        return np.nan, np.nan, np.nan, np.nan
    amplitude, w, phi, _ = params
    if amplitude < 0:
        phi += np.pi
    quality = fit_quality(y, profile, params, cov)
    return float(phi % (2 * np.pi)), float(2 * np.pi / w), quality.r2, quality.phase_error


def column_phases(img, x_centres, rows, eval_row, period_guess, half_width=2) -> ColumnPhases:
    """Phase at ``eval_row`` of each column band ``x - half_width .. x + half_width``.

    Parameters
    ----------
    img : numpy.ndarray
        Interferogram, rows x columns.
    x_centres : array_like of int
        Centre column of each band.
    rows : tuple of int
        Fit window ``(y0, y1)``, rows.
    eval_row : float
        Row the phase is referred to (normally the centre of the window).
    period_guess : float
        Starting fringe period along ``y``, px per fringe.
    half_width : int
        Half-width of each band; a band spans ``2 * half_width + 1`` columns.

    Returns
    -------
    ColumnPhases
    """
    img = np.asarray(img, dtype=float)
    y0, y1 = rows
    y = np.arange(y0, y1, dtype=float) - eval_row
    x_centres = np.asarray(x_centres, dtype=int)
    if (x_centres.min() - half_width < 0 or x_centres.max() + half_width >= img.shape[1]
            or y0 < 0 or y1 > img.shape[0]):
        raise ValueError("column bands or rows fall outside the image")
    fits = np.array([fit_column_band(img[y0:y1, x - half_width:x + half_width + 1].mean(axis=1),
                                     y, period_guess) for x in x_centres])
    phase = fits[:, 0]
    unwrapped = np.full_like(phase, np.nan)
    ok = np.isfinite(phase)
    unwrapped[ok] = np.unwrap(phase[ok])
    return ColumnPhases(x=x_centres, phase=phase, phase_unwrapped=unwrapped, period=fits[:, 1],
                        r2=fits[:, 2], phase_error=fits[:, 3])


def side_fit(cols: ColumnPhases, x_range, deg=1, r2_min=0.3) -> np.ndarray:
    """Polynomial fit of the unwrapped band phase over ``x_range``.

    Bands with ``r2 < r2_min`` (failed or washed-out fits) are left out.

    Returns
    -------
    numpy.ndarray
        Polynomial coefficients, highest power first (as ``numpy.polyfit``).
    """
    lo, hi = x_range
    sel = ((cols.x >= lo) & (cols.x <= hi) & (cols.r2 >= r2_min) & np.isfinite(cols.phase_unwrapped))
    if sel.sum() <= deg + 1:
        raise ValueError("too few valid column bands in %s" % (x_range,))
    return np.polyfit(cols.x[sel], cols.phase_unwrapped[sel], deg)


def edge_fraction(img, edge_x, rows, eval_row, period_guess, gap=30, span=180, step=5,
                  half_width=2, deg=1, r2_min=0.3) -> EdgeFractionResult:
    """Fringe fraction with block and platen phases compared at the edge column.

    The platen is taken left of the edge and the block right of it. Each side uses
    column bands from ``gap`` to ``span`` px away from the edge (the strip next to the
    edge is left out because the step diffracts, and ``span`` must not reach the far
    edge of the platen), fits the phase along ``x`` with a polynomial of degree ``deg``
    and evaluates it at ``edge_x``.

    Parameters
    ----------
    img : numpy.ndarray
        Interferogram, rows x columns.
    edge_x : float
        Edge column, px (for instance from :func:`edge_line_centre`).
    rows : tuple of int
        Fit window ``(y0, y1)``, rows.
    eval_row : float
        Evaluation row; the fraction refers to it.
    period_guess : float
        Starting fringe period along ``y``, px per fringe.
    gap, span, step : int
        Band positions: from ``gap`` to ``span`` px from the edge, every ``step`` px.
    half_width : int
        Half-width of each band, px.
    deg : int
        Degree of the polynomial fitted on each side.
    r2_min : float
        Bands with a smaller coefficient of determination are left out of the fits.

    Returns
    -------
    EdgeFractionResult
    """
    xs_platen = np.arange(edge_x - span, edge_x - gap + 1, step)
    xs_block = np.arange(edge_x + gap, edge_x + span + 1, step)
    platen = column_phases(img, xs_platen, rows, eval_row, period_guess, half_width)
    block = column_phases(img, xs_block, rows, eval_row, period_guess, half_width)
    cp = side_fit(platen, (edge_x - span, edge_x - gap), deg, r2_min)
    cb = side_fit(block, (edge_x + gap, edge_x + span), deg, r2_min)
    phi_p, phi_b = np.polyval(cp, edge_x), np.polyval(cb, edge_x)
    dp, db = np.polyval(np.polyder(cp), edge_x), np.polyval(np.polyder(cb), edge_x)
    return EdgeFractionResult(fraction=float(((phi_b - phi_p) / (2 * np.pi)) % 1),
                              phase_platen=float(phi_p), phase_block=float(phi_b),
                              slope_platen=float(dp / (2 * np.pi)), slope_block=float(db / (2 * np.pi)),
                              edge_x=float(edge_x), eval_row=float(eval_row), platen=platen, block=block)
