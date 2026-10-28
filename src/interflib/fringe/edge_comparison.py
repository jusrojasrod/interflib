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

The lines cross a gap next to the edge where no band is read, so ``F`` rests on the side
phases being straight up to the edge. :func:`span_sensitivity` repeats the comparison
with other bands; a fraction that moves with them shows that they are not.

Image columns are ``x`` and rows ``y``; fringes are expected to be roughly horizontal
(spatial carrier along ``y``). The platen is assumed to lie left of the edge and the
block right of it; mirror the image for the opposite layout.

Sign convention: every band phase is taken on the branch whose phase increases towards
``+y`` (down the image). The fraction is ``F`` when the interference order also increases
towards ``+y`` (``order_sign=+1``, the default) and ``1 - F`` otherwise, so the set-up's
direction must be declared with ``order_sign=-1`` when it runs the other way; see
:func:`interflib.fringe.models.signed_fraction`.

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

from .models import (canonical_sinusoid, fit_quality, interference_model, parabolic_vertex, signed_fraction,
                     sine_seed)


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
    fraction_error : float
        Standard uncertainty of ``fraction`` from the extrapolation to the edge, fringes:
        the two side polynomials evaluated at ``edge_x``, each with the covariance given
        by the scatter of its band phases, combined in quadrature. It assumes independent
        band phases, so it falls short when the demodulation correlates neighbouring
        bands (the two-dimensional Fourier filter), and it does not include a model error
        of the polynomial (for instance curvature with ``deg=1``).
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
    order_sign : int
        Direction of the interference order used for ``fraction`` (``+1``: increasing
        towards ``+y``).
    """

    fraction: float
    fraction_error: float
    phase_platen: float
    phase_block: float
    slope_platen: float
    slope_block: float
    edge_x: float
    eval_row: float
    platen: ColumnPhases
    block: ColumnPhases
    order_sign: int = 1


def _middle_half(n_rows):
    # Default row range: the middle half of the frame, clear of the top and bottom borders.
    return n_rows // 4, n_rows - n_rows // 4


def detect_edge_column(img, rows=None, search=None, smooth=5) -> int:
    """Column of the block edge from the mean horizontal intensity jump.

    Along the edge the fringes of block and platen are out of step, so the mean
    absolute difference between neighbouring columns peaks there.

    Parameters
    ----------
    img : numpy.ndarray
        Interferogram, rows x columns.
    rows : tuple of int, optional
        Row range ``(y0, y1)`` over which the difference is averaged; the middle half of
        the image if omitted.
    search : tuple of int, optional
        Column range ``(x0, x1)`` searched for the edge; the whole width if omitted.
        Restrict it when other features (the far edge of the platen strip, the beam
        border) produce larger jumps than the block edge.
    smooth : int
        Width of the moving average applied to the difference profile, px.

    Returns
    -------
    int
        Edge column: the boundary between the two columns of the largest jump.
    """
    img = np.asarray(img, dtype=float)
    y0, y1 = _middle_half(img.shape[0]) if rows is None else rows
    xa, xb = (0, img.shape[1]) if search is None else search
    jump = np.abs(np.diff(img[y0:y1, xa:xb], axis=1)).mean(axis=0)
    jump = np.convolve(jump, np.ones(smooth) / smooth, mode="same")
    return xa + int(np.argmax(jump)) + 1


def edge_line_centre(img, approx, rows=None, half=15) -> float:
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
    rows : tuple of int, optional
        Row range ``(y0, y1)`` over which the column means are taken; the middle half of
        the image if omitted.
    half : int
        Half-width of the search window around ``approx``, px.

    Returns
    -------
    float
        Edge column, px.
    """
    img = np.asarray(img, dtype=float)
    y0, y1 = _middle_half(img.shape[0]) if rows is None else rows
    means = np.convolve(img[y0:y1].mean(axis=0), np.ones(3) / 3, mode="same")
    seg = means[approx - half:approx + half + 1]
    i = int(np.argmin(seg))
    delta = parabolic_vertex(*seg[i - 1:i + 2]) if 0 < i < len(seg) - 1 else 0.0
    return float(approx - half + i + delta)


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
        fit does not converge. The fit is folded onto ``A >= 0``, ``w > 0``
        (:func:`interflib.fringe.models.canonical_sinusoid`) so that every band lies on
        the same branch, with the phase increasing towards ``+y``.
    """
    profile = np.asarray(profile, dtype=float)
    try:
        params, cov = curve_fit(interference_model, y, profile,
                                p0=sine_seed(profile, y, 2 * np.pi / period_guess), maxfev=5000)
    except RuntimeError:
        return np.nan, np.nan, np.nan, np.nan
    quality = fit_quality(y, profile, params, cov)
    _, w, phi = canonical_sinusoid(*params[:3])
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
    x, phase = _side_points(cols, x_range, deg, r2_min)
    return np.polyfit(x, phase, deg)


def _side_points(cols: ColumnPhases, x_range, deg, r2_min):
    # Valid bands of one side: inside x_range, converged, and above the r2 threshold.
    lo, hi = x_range
    sel = ((cols.x >= lo) & (cols.x <= hi) & (cols.r2 >= r2_min) & np.isfinite(cols.phase_unwrapped))
    if sel.sum() <= deg + 1:
        raise ValueError("too few valid column bands in %s" % (x_range,))
    return cols.x[sel].astype(float), cols.phase_unwrapped[sel]


def _value_error(cols: ColumnPhases, x_range, deg, r2_min, at) -> float:
    """Standard uncertainty (radians) of the side polynomial evaluated at ``at``.

    The fit is repeated in ``x - at``, so the value at ``at`` is the constant term and
    its variance the last diagonal entry of the covariance, scaled by the residual
    variance of the band phases (at least one degree of freedom is left by
    :func:`_side_points`).
    """
    x, phase = _side_points(cols, x_range, deg, r2_min)
    design = np.vander(x - at, deg + 1)
    coef, *_ = np.linalg.lstsq(design, phase, rcond=None)
    resid = phase - design @ coef
    cov = np.linalg.inv(design.T @ design) * (resid @ resid / (len(x) - (deg + 1)))
    return float(np.sqrt(cov[-1, -1]))


def edge_fraction(img, edge_x, rows, eval_row, period_guess, gap=30, span=180, step=5,
                  half_width=2, deg=1, r2_min=0.3, order_sign=1) -> EdgeFractionResult:
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
    order_sign : {1, -1}
        ``+1`` when the interference order increases towards ``+y``, ``-1`` when it
        decreases (the fraction is then ``1 - F`` of the ``+1`` reading).

    Returns
    -------
    EdgeFractionResult
    """
    xs_platen, xs_block = side_band_centres(edge_x, gap, span, step)
    platen = column_phases(img, xs_platen, rows, eval_row, period_guess, half_width)
    block = column_phases(img, xs_block, rows, eval_row, period_guess, half_width)
    return fraction_at_edge(platen, block, edge_x, eval_row, gap, span, deg, r2_min, order_sign)


def side_band_centres(edge_x, gap, span, step):
    """Band centre columns on the platen (left) and block (right) sides of the edge.

    Returns
    -------
    tuple of numpy.ndarray
        ``(xs_platen, xs_block)``, from ``gap`` to ``span`` px away from ``edge_x`` every
        ``step`` px.
    """
    return (np.arange(edge_x - span, edge_x - gap + 1, step),
            np.arange(edge_x + gap, edge_x + span + 1, step))


def fraction_at_edge(platen: ColumnPhases, block: ColumnPhases, edge_x, eval_row, gap=30, span=180,
                     deg=1, r2_min=0.3, order_sign=1) -> EdgeFractionResult:
    """Extrapolate the column phases of each side to the edge and take their difference.

    This is the comparison step of :func:`edge_fraction`, independent of how the column
    phases were obtained (sinusoidal fit here, Fourier demodulation in
    :mod:`interflib.fringe.fourier`).

    Parameters
    ----------
    platen, block : ColumnPhases
        Column phases left and right of the edge, referred to ``eval_row``.
    edge_x : float
        Edge column, px.
    eval_row : float
        Evaluation row the phases refer to.
    gap, span : int
        Only bands from ``gap`` to ``span`` px away from the edge enter the side fits.
    deg : int
        Degree of the polynomial fitted on each side.
    r2_min : float
        Bands with a smaller coefficient of determination are left out of the fits.
    order_sign : {1, -1}
        Direction of the interference order along ``y``, as in :func:`edge_fraction`.

    Returns
    -------
    EdgeFractionResult
    """
    sides = (platen, (edge_x - span, edge_x - gap)), (block, (edge_x + gap, edge_x + span))
    cp, cb = [side_fit(cols, x_range, deg, r2_min) for cols, x_range in sides]
    phi_p, phi_b = np.polyval(cp, edge_x), np.polyval(cb, edge_x)
    dp, db = np.polyval(np.polyder(cp), edge_x), np.polyval(np.polyder(cb), edge_x)
    u_p, u_b = [_value_error(cols, x_range, deg, r2_min, edge_x) for cols, x_range in sides]
    return EdgeFractionResult(fraction=signed_fraction(phi_b - phi_p, order_sign),
                              fraction_error=float(np.hypot(u_p, u_b) / (2 * np.pi)),
                              phase_platen=float(phi_p), phase_block=float(phi_b),
                              slope_platen=float(dp / (2 * np.pi)), slope_block=float(db / (2 * np.pi)),
                              edge_x=float(edge_x), eval_row=float(eval_row), platen=platen, block=block,
                              order_sign=order_sign)


SPAN_VARIANTS = ((30, 180), (45, 180), (60, 180), (30, 120))   # (gap, span) px; the first is the reading


@dataclass
class SpanSensitivity:
    """Fringe fraction as a function of the bands that enter the side fits.

    Attributes
    ----------
    variants : numpy.ndarray
        ``(gap, span)`` of each variant, px; the first is the reference reading.
    fraction : numpy.ndarray
        Fraction of each variant at ``at_x``, unwrapped next to the first, fringes.
    deviation : float
        Largest absolute difference from the first variant, fringes.
    model_error : float
        Standard uncertainty of the fraction from the choice of bands, fringes: the
        deviation itself. A rectangular distribution of half-width ``deviation`` would
        give ``deviation / sqrt(3)`` (JCGM 100:2008, 4.3.7), but in simulations where the
        image is blurred coherently across the edge the bias of the reading reached 1.2
        times the deviation, which would leave it at about two standard uncertainties.
    at_x : float
        Column where the fractions are evaluated, px.
    """

    variants: np.ndarray
    fraction: np.ndarray
    deviation: float
    model_error: float
    at_x: float


def span_sensitivity(platen: ColumnPhases, block: ColumnPhases, edge_x, eval_row, variants=SPAN_VARIANTS,
                     at_x=None, deg=1, r2_min=0.3, order_sign=1) -> SpanSensitivity:
    """Repeat the comparison at the edge with other bands and see how far the fraction moves.

    The side lines are extrapolated across the ``gap`` next to the edge, where no band is
    read. When the side phases are not straight over the bands (a curvature that differs
    between block and platen, or a phase that bends towards the other surface close to the
    edge), the fraction depends on which bands enter the fits, and ``fraction_error``,
    which only reflects the scatter about the lines, does not show it. Leaving out the
    bands nearest the edge or shortening the span exposes that dependence. The deviation
    also contains part of the band noise, so on a straight field it is of the order of
    ``fraction_error``.

    Parameters
    ----------
    platen, block : ColumnPhases
        Column phases left and right of the edge, as in :func:`fraction_at_edge`. Every
        variant must lie inside the bands they cover; the variants only select among them.
    edge_x, eval_row
        As in :func:`fraction_at_edge`.
    variants : sequence of (int, int)
        ``(gap, span)`` of each variant, px; the first is the reading being assessed.
    at_x : float, optional
        Column where the fraction is used: the edge if omitted. At another column (the
        face centre) each variant extends its own side lines there.
    deg, r2_min, order_sign
        As in :func:`fraction_at_edge`.

    Returns
    -------
    SpanSensitivity
    """
    at = float(edge_x if at_x is None else at_x)
    values = []
    for gap, span in variants:
        res = fraction_at_edge(platen, block, edge_x, eval_row, gap, span, deg, r2_min, order_sign)
        values.append(res.fraction + order_sign * (res.slope_block - res.slope_platen) * (at - edge_x))
    values = np.asarray(values)
    first = values[0] % 1.0
    fractions = first + (values - first + 0.5) % 1.0 - 0.5
    deviation = float(np.max(np.abs(fractions - first)))
    return SpanSensitivity(variants=np.asarray(variants, dtype=int), fraction=fractions, deviation=deviation,
                           model_error=deviation, at_x=at)
