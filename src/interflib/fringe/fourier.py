"""Single-shot Fourier (spatial-carrier) demodulation of gauge-block interferograms.

With the reference mirror tilted, the fringes carry a spatial carrier and one image is
enough to recover the phase: in the Fourier spectrum of ``I = a + b cos(phi)`` the
fringe term splits into two side lobes at ``+-g_y``, where ``g_y = 1 / Lambda`` is the
carrier (the fringe-order gradient along ``y``, ``Lambda`` the fringe period); one lobe is
isolated with a
band-pass filter and transformed back. The angle of the resulting complex (analytic)
signal is the wrapped phase at every pixel. This is the Fourier-transform method of
Takeda, Ina & Kobayashi, *J. Opt. Soc. Am.* 72, 156-160 (1982), in the
synchronous-detection reading of Servin, Quiroga & Padilla, *Fringe Pattern Analysis for
Optical Metrology*, Wiley-VCH, 2014, sec. 4.2.

Here the transform runs **along each column** (axis ``y``), because the fringes are
roughly horizontal (carrier along ``y``) and the block edge is vertical: a column lies
entirely on the block or on the platen, so the edge step does not leak into the
spectrum of any column. Fringes rotated by a few degrees only add a phase gradient from
column to column, which the method follows. The horizontal ends of the block (its top
or bottom face edge, when they are in the field) do cut the columns, so the transform
span ``fft_rows`` must stay clear of them.

A two-dimensional variant, :func:`demodulate_2d`, transforms a whole rectangular region
that lies on one surface (the platen beside the block, or the block face) and filters
the carrier lobe in two dimensions, by synchronous detection: the region is multiplied by
``exp(-i 2 pi (g_x x + g_y y))``, low-pass filtered and shifted back. It limits the
bandwidth along ``x`` as well as along ``y`` and returns a phase map of the region, with
and without the carrier; :func:`unwrap_phase_map` unwraps the carrier-free phase into a
continuous map of the surface, the end product of the method in Takeda et al. (1982) and
in Munoz Maciel et al., "Digital Processing Techniques for Fringe Analysis", in *Optical
Interferometry*, InTech, 2017, sec. 5, doi:10.5772/66474. :func:`fourier_edge_fraction_2d`
demodulates a platen region and a block region separately and compares them at the edge
like the column variant.

The phase is returned in the convention of :mod:`interflib.fringe.edge_comparison`
(the phase ``phi`` of ``A sin(w y + phi) + C`` with ``w > 0``), so the column phases of
the two methods are directly comparable and the edge comparison is shared: the column
phases are extrapolated to the edge and subtracted there by
:func:`interflib.fringe.edge_comparison.fraction_at_edge`. The filter keeps the side lobe
at ``+g_y``, so the phase increases towards ``+y`` whichever way the reference is tilted:
the direction of the interference order is not observable in one image and is declared
with ``order_sign`` (``+1``: order increasing towards ``+y``), as in that module.

This module has no I/O side effects: it takes and returns arrays.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.signal.windows import tukey

from .edge_comparison import (
    ColumnPhases,
    EdgeFractionResult,
    fraction_at_edge,
    side_band_centres,
)
from .unwrap import unwrap_goldstein, unwrap_itoh, unwrap_quality_guided


@dataclass
class ColumnDemodulation:
    """Fourier demodulation of a set of column bands over a span of rows.

    Attributes
    ----------
    y : numpy.ndarray
        Rows of the transform span.
    phase : numpy.ndarray
        Wrapped phase, rows x bands, radians in ``[0, 2 pi)``, in the sine convention
        (see the module docstring); it includes the carrier ``2 pi g_y y``.
    modulation : numpy.ndarray
        Fringe amplitude ``b`` recovered from the side lobe, rows x bands, in the units
        of the image.
    background : numpy.ndarray
        Low-pass background ``a``, rows x bands.
    carrier : float
        Carrier frequency used to centre the filter, fringes per px along ``y``.
    """

    y: np.ndarray
    phase: np.ndarray
    modulation: np.ndarray
    background: np.ndarray
    carrier: float


def _bands(img, x_centres, row_span, half_width):
    img = np.asarray(img, dtype=float)
    y0, y1 = row_span
    x_centres = np.asarray(x_centres, dtype=int)
    if (x_centres.min() - half_width < 0 or x_centres.max() + half_width >= img.shape[1]
            or y0 < 0 or y1 > img.shape[0]):
        raise ValueError("column bands or rows fall outside the image")
    return np.stack([img[y0:y1, x - half_width:x + half_width + 1].mean(axis=1) for x in x_centres],
                    axis=1)


def estimate_carrier(profiles, min_cycles=4.0):
    """Carrier frequency along ``y`` from the mean power spectrum of column profiles.

    Parameters
    ----------
    profiles : numpy.ndarray
        Column profiles, rows x columns (one column per profile).
    min_cycles : float
        Lowest carrier accepted, in fringes over the profile length. Below it the side lobe
        cannot be separated from the background and the method does not apply.

    Returns
    -------
    float
        Carrier, fringes per px, refined to a fraction of a bin by a parabola through the
        logarithm of the three highest spectral samples (zero-padded fourfold).

    Raises
    ------
    ValueError
        If the spectral peak lies below ``min_cycles`` fringes over the profile length.
    """
    profiles = np.asarray(profiles, dtype=float)
    n = profiles.shape[0]
    centred = (profiles - profiles.mean(axis=0)) * np.hanning(n)[:, None]
    n_fft = 4 * int(2 ** np.ceil(np.log2(n)))
    power = (np.abs(np.fft.rfft(centred, n=n_fft, axis=0)) ** 2).mean(axis=1)
    freqs = np.fft.rfftfreq(n_fft)
    # The global peak (zero frequency excluded) is the carrier; searching only above the
    # threshold would pick a side lobe of a carrier that lies below it.
    k = 1 + int(np.argmax(power[1:]))
    if freqs[k] < min_cycles / n or k >= len(power) - 1:
        raise ValueError("no carrier above %.1f fringes over %d rows" % (min_cycles, n))
    la, lb, lc = np.log(power[k - 1:k + 2] + 1e-300)
    delta = 0.5 * (la - lc) / (la - 2 * lb + lc)
    return float((k + delta) * (freqs[1] - freqs[0]))


def demodulate_columns(img, x_centres, fft_rows, carrier=None, bandwidth=0.5, half_width=2,
                       taper=0.0, min_cycles=4.0) -> ColumnDemodulation:
    """Wrapped phase of each column band by Fourier filtering along ``y``.

    Each band (the mean of ``2 * half_width + 1`` columns) is tapered at its ends,
    transformed along ``y``, multiplied by a raised-cosine band-pass that keeps only the
    positive side lobe, ``|f - g_y| < bandwidth * g_y``, and transformed back.

    Parameters
    ----------
    img : numpy.ndarray
        Interferogram, rows x columns.
    x_centres : array_like of int
        Centre column of each band.
    fft_rows : tuple of int
        Row span ``(y0, y1)`` of the transform. It must not cross a horizontal edge of the
        block.
    carrier : float, optional
        Carrier frequency, fringes per px. Estimated from the bands if omitted.
    bandwidth : float
        Half-width of the band-pass relative to the carrier, in ``(0, 1)``. It must keep
        the background (around zero frequency) out of the pass band.
    half_width : int
        Half-width of each band; a band spans ``2 * half_width + 1`` columns.
    taper : float
        Fraction of the span tapered (Tukey window) to soften the ends. The phase near
        the ends is less reliable either way; keep the rows used for the phase well inside.
    min_cycles : float
        Passed to :func:`estimate_carrier`.

    Returns
    -------
    ColumnDemodulation
    """
    if not 0.0 < bandwidth < 1.0:
        raise ValueError("bandwidth must lie in (0, 1)")
    profiles = _bands(img, x_centres, fft_rows, half_width)
    n = profiles.shape[0]
    if carrier is None:
        carrier = estimate_carrier(profiles, min_cycles)
    elif carrier * n < min_cycles:
        raise ValueError("carrier gives fewer than %.1f fringes over %d rows" % (min_cycles, n))
    window = tukey(n, taper)[:, None] if taper > 0 else np.ones((n, 1))
    spectrum = np.fft.fft(profiles * window, axis=0)
    freqs = np.fft.fftfreq(n)[:, None]
    half = bandwidth * carrier
    side = np.where(np.abs(freqs - carrier) < half, np.cos(np.pi * (freqs - carrier) / (2 * half)) ** 2, 0.0)
    low = np.where(np.abs(freqs) < half, np.cos(np.pi * freqs / (2 * half)) ** 2, 0.0)
    analytic = np.fft.ifft(spectrum * side, axis=0)
    background = np.fft.ifft(spectrum * low, axis=0).real
    with np.errstate(invalid="ignore", divide="ignore"):
        background = background / window
        modulation = 2 * np.abs(analytic) / window
    # angle(analytic) is the phase of b*cos(.); the sine convention adds pi/2
    phase = (np.angle(analytic) + np.pi / 2) % (2 * np.pi)
    y = np.arange(fft_rows[0], fft_rows[1], dtype=float)
    return ColumnDemodulation(y=y, phase=phase, modulation=modulation, background=background,
                              carrier=float(carrier))


def fourier_column_phases(img, x_centres, rows, eval_row, fft_rows=None, carrier=None, bandwidth=0.5,
                          half_width=2, deg=1, taper=0.0, min_cycles=4.0) -> ColumnPhases:
    """Phase at ``eval_row`` of each column band, by Fourier demodulation.

    The demodulated phase of each band is unwrapped along ``y``, a polynomial of degree
    ``deg`` is fitted to it over ``rows``, and its value at ``eval_row`` is the band
    phase. With ``deg=1`` this is the phase and period of the best straight line through
    the pixel phases, the counterpart of the sinusoidal fit of
    :func:`interflib.fringe.edge_comparison.column_phases`.

    Parameters
    ----------
    img : numpy.ndarray
        Interferogram, rows x columns.
    x_centres : array_like of int
        Centre column of each band.
    rows : tuple of int
        Rows ``(y0, y1)`` whose pixel phases enter the fit.
    eval_row : float
        Row the phase is referred to.
    fft_rows : tuple of int, optional
        Transform span; ``rows`` itself if omitted. A span wider than ``rows`` keeps the
        ends of the transform away from the fitted rows.
    carrier, bandwidth, taper, min_cycles
        Passed to :func:`demodulate_columns`.
    half_width : int
        Half-width of each band, px.
    deg : int
        Degree of the polynomial fitted along ``y`` to the unwrapped pixel phases.

    Returns
    -------
    ColumnPhases
        ``r2`` is the coefficient of determination of the demodulated fringe model
        ``a + b sin(phase)`` against the band intensity over ``rows``; ``phase_error`` is
        the standard error of the polynomial at ``eval_row``, inflated for the
        correlation that the band-pass introduces between neighbouring pixels (about
        ``1 / (2 * bandwidth * carrier)`` px). Both are diagnostics, not uncertainties.
    """
    fft_rows = rows if fft_rows is None else fft_rows
    if not (fft_rows[0] <= rows[0] and rows[1] <= fft_rows[1]):
        raise ValueError("rows must lie inside fft_rows")
    demod = demodulate_columns(img, x_centres, fft_rows, carrier, bandwidth, half_width, taper, min_cycles)
    sel = slice(rows[0] - fft_rows[0], rows[1] - fft_rows[0])
    return _phases_at_row(demod.phase[sel], demod.background[sel], demod.modulation[sel],
                          _bands(img, x_centres, rows, half_width), demod.y[sel], eval_row, deg,
                          1.0 / (2 * bandwidth * demod.carrier), x_centres)


def _phases_at_row(band_phase, background, modulation, profiles, y, eval_row, deg, corr, x_centres):
    # Shared by the column and the two-dimensional demodulation: unwrap each band phase
    # along y, fit a polynomial over the rows given and read it at eval_row.
    y = np.asarray(y, dtype=float) - eval_row
    unwrapped = np.unwrap(band_phase, axis=0)
    n_bands = unwrapped.shape[1]
    phase = np.empty(n_bands)
    period = np.empty(n_bands)
    r2 = np.empty(n_bands)
    phase_error = np.empty(n_bands)
    design = np.vander(y, deg + 1)
    for j in range(n_bands):
        coef, *_ = np.linalg.lstsq(design, unwrapped[:, j], rcond=None)
        resid = unwrapped[:, j] - design @ coef
        dof = max(len(y) - (deg + 1), 1)
        cov = np.linalg.inv(design.T @ design) * (resid @ resid / dof) * max(corr, 1.0)
        phase[j] = coef[-1] % (2 * np.pi)
        phase_error[j] = np.sqrt(cov[-1, -1])
        slope = coef[-2] if deg >= 1 else np.nan
        period[j] = 2 * np.pi / slope if slope else np.nan
        model = background[:, j] + modulation[:, j] * np.sin(band_phase[:, j])
        ss_res = np.sum((profiles[:, j] - model) ** 2)
        ss_tot = np.sum((profiles[:, j] - profiles[:, j].mean()) ** 2)
        r2[j] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    unwrapped_x = np.unwrap(phase)
    return ColumnPhases(x=np.asarray(x_centres, dtype=int), phase=phase, phase_unwrapped=unwrapped_x,
                        period=period, r2=r2, phase_error=phase_error)


def fourier_edge_fraction(img, edge_x, rows, eval_row, fft_rows=None, carrier=None, bandwidth=0.5,
                          gap=30, span=180, step=5, half_width=2, deg_y=1, deg=1, r2_min=0.3,
                          taper=0.0, min_cycles=4.0, order_sign=1) -> EdgeFractionResult:
    """Fringe fraction at the edge column with Fourier-demodulated column phases.

    The same comparison as :func:`interflib.fringe.edge_comparison.edge_fraction` (platen
    left of the edge, block right of it, each side extrapolated to ``edge_x`` with a
    polynomial of degree ``deg``), with the column phases from
    :func:`fourier_column_phases` instead of a sinusoidal fit.

    Parameters
    ----------
    img : numpy.ndarray
        Interferogram, rows x columns.
    edge_x : float
        Edge column, px.
    rows : tuple of int
        Rows whose pixel phases enter the fit along ``y``.
    eval_row : float
        Evaluation row; the fraction refers to it.
    fft_rows, carrier, bandwidth, taper, min_cycles
        Passed to :func:`fourier_column_phases`. One carrier is shared by both sides; it
        is estimated from the platen bands if omitted.
    gap, span, step : int
        Band positions: from ``gap`` to ``span`` px from the edge, every ``step`` px.
    half_width : int
        Half-width of each band, px.
    deg_y : int
        Degree of the fit along ``y`` of each band.
    deg : int
        Degree of the fit along ``x`` of each side.
    r2_min : float
        Bands with a smaller coefficient of determination are left out of the side fits.
    order_sign : {1, -1}
        ``+1`` when the interference order increases towards ``+y``, ``-1`` when it
        decreases (the fraction is then ``1 - F`` of the ``+1`` reading).

    Returns
    -------
    EdgeFractionResult
    """
    fft_rows = rows if fft_rows is None else fft_rows
    xs_platen, xs_block = side_band_centres(edge_x, gap, span, step)
    if carrier is None:
        carrier = estimate_carrier(_bands(img, xs_platen, fft_rows, half_width), min_cycles)
    kwargs = dict(fft_rows=fft_rows, carrier=carrier, bandwidth=bandwidth, half_width=half_width,
                  deg=deg_y, taper=taper, min_cycles=min_cycles)
    platen = fourier_column_phases(img, xs_platen, rows, eval_row, **kwargs)
    block = fourier_column_phases(img, xs_block, rows, eval_row, **kwargs)
    return fraction_at_edge(platen, block, edge_x, eval_row, gap, span, deg, r2_min, order_sign)


# --------------------------------------------------------------------------------------
# Two-dimensional demodulation
# --------------------------------------------------------------------------------------
@dataclass
class PhaseMap:
    """Two-dimensional Fourier demodulation of a rectangular region.

    Attributes
    ----------
    x, y : numpy.ndarray
        Columns and rows of the region.
    phase : numpy.ndarray
        Wrapped phase, rows x columns, radians in ``[0, 2 pi)``, in the sine convention;
        it includes the carrier.
    object_phase : numpy.ndarray
        The same phase with the carrier removed, wrapped to ``[0, 2 pi)``: what the
        surface adds to the tilt of the reference. ``phase = object_phase + 2 pi (g_x (x -
        x[0]) + g_y (y - y[0]))`` modulo ``2 pi``, with the carrier referred to the first
        row and column of the region.
    modulation : numpy.ndarray
        Fringe amplitude ``b``, rows x columns.
    background : numpy.ndarray
        Low-pass background ``a``, rows x columns.
    carrier : tuple of float
        ``(g_x, g_y)``, fringes per px, with ``g_y > 0``.
    analytic : numpy.ndarray
        Complex analytic signal ``(b / 2) exp(i phi)`` of the cosine convention, kept so
        that bands can be averaged as complex numbers.
    """

    x: np.ndarray
    y: np.ndarray
    phase: np.ndarray
    object_phase: np.ndarray
    modulation: np.ndarray
    background: np.ndarray
    carrier: tuple
    analytic: np.ndarray


@dataclass
class UnwrappedMap:
    """Continuous phase of a two-dimensional phase map.

    Attributes
    ----------
    x, y : numpy.ndarray
        Columns and rows of the region.
    object_phase : numpy.ndarray
        Unwrapped phase with the carrier removed, radians (``nan`` outside ``mask``). It
        is defined up to a constant: unwrapping fixes differences, not the absolute value.
    phase : numpy.ndarray
        ``object_phase`` plus the carrier ramp, the continuous phase of the fringes.
    mask : numpy.ndarray
        Pixels that took part in the unwrapping.
    method : str
        Unwrapping method used.
    """

    x: np.ndarray
    y: np.ndarray
    object_phase: np.ndarray
    phase: np.ndarray
    mask: np.ndarray
    method: str


def _region(img, rows, cols):
    img = np.asarray(img, dtype=float)
    (y0, y1), (x0, x1) = rows, cols
    if y0 < 0 or x0 < 0 or y1 > img.shape[0] or x1 > img.shape[1] or y1 <= y0 or x1 <= x0:
        raise ValueError("region falls outside the image")
    return img[y0:y1, x0:x1]


def _parabolic_offset(values):
    la, lb, lc = np.log(np.asarray(values) + 1e-300)
    return 0.5 * (la - lc) / (la - 2 * lb + lc)


def estimate_carrier_2d(region, min_cycles=4.0):
    """Carrier ``(g_x, g_y)`` of a region from its two-dimensional power spectrum.

    The global spectral peak outside the zero-frequency bins is taken, on the side with
    ``g_y > 0`` (the two lobes are symmetric), and refined to a fraction of a bin along
    each axis by a parabola through the logarithm of the neighbouring samples (zero-padded
    fourfold).

    Parameters
    ----------
    region : numpy.ndarray
        Interferogram region, rows x columns, with fringes roughly horizontal.
    min_cycles : float
        Lowest carrier accepted along ``y``, in fringes over the region height.

    Returns
    -------
    tuple of float
        ``(g_x, g_y)``, fringes per px.

    Raises
    ------
    ValueError
        If the peak gives fewer than ``min_cycles`` fringes along ``y``.
    """
    region = np.asarray(region, dtype=float)
    ny, nx = region.shape
    centred = (region - region.mean()) * np.outer(np.hanning(ny), np.hanning(nx))
    py, px = 4 * int(2 ** np.ceil(np.log2(ny))), 4 * int(2 ** np.ceil(np.log2(nx)))
    power = np.abs(np.fft.fft2(centred, s=(py, px))) ** 2
    fy, fx = np.fft.fftfreq(py), np.fft.fftfreq(px)
    dc = (np.abs(fy)[:, None] < 1.5 / ny) & (np.abs(fx)[None, :] < 1.5 / nx)
    ky, kx = np.unravel_index(np.argmax(np.where(~dc & (fy[:, None] > 0), power, -np.inf)), power.shape)
    g_y = fy[ky] + fy[1] * _parabolic_offset(power[[(ky - 1) % py, ky, (ky + 1) % py], kx])
    g_x = fx[kx] + fx[1] * _parabolic_offset(power[ky, [(kx - 1) % px, kx, (kx + 1) % px]])
    if g_y * ny < min_cycles:
        raise ValueError("no carrier above %.1f fringes over %d rows" % (min_cycles, ny))
    return float(g_x), float(g_y)


def demodulate_2d(img, rows, cols, carrier=None, bandwidth=0.5, taper=0.0, min_cycles=4.0) -> PhaseMap:
    """Wrapped phase of a rectangular region by two-dimensional Fourier filtering.

    The region is (optionally) tapered, transformed in two dimensions, multiplied by a
    radial raised-cosine pass band of radius ``bandwidth * |g|`` centred on the carrier
    lobe ``g = (g_x, g_y)``, and transformed back. Unlike :func:`demodulate_columns`, the
    pass band follows the lobe wherever the fringe tilt places it and limits the
    bandwidth along ``x`` as well as along ``y``. The region must lie entirely on one
    surface: a block edge inside it would spread over the whole spectrum.

    Parameters
    ----------
    img : numpy.ndarray
        Interferogram, rows x columns.
    rows, cols : tuple of int
        Region ``(y0, y1)`` and ``(x0, x1)``.
    carrier : tuple of float, optional
        ``(g_x, g_y)``, fringes per px. Estimated from the region if omitted.
    bandwidth : float
        Radius of the pass band relative to ``|g|``, in ``(0, 1)``.
    taper : float
        Fraction of each side tapered with a Tukey window.
    min_cycles : float
        Passed to :func:`estimate_carrier_2d`.

    Returns
    -------
    PhaseMap
    """
    if not 0.0 < bandwidth < 1.0:
        raise ValueError("bandwidth must lie in (0, 1)")
    region = _region(img, rows, cols)
    ny, nx = region.shape
    if carrier is None:
        carrier = estimate_carrier_2d(region, min_cycles)
    g_x, g_y = carrier
    if g_y * ny < min_cycles:
        raise ValueError("carrier gives fewer than %.1f fringes over %d rows" % (min_cycles, ny))
    window = np.outer(tukey(ny, taper), tukey(nx, taper)) if taper > 0 else np.ones((ny, nx))
    fy, fx = np.fft.fftfreq(ny)[:, None], np.fft.fftfreq(nx)[None, :]
    radius = bandwidth * np.hypot(g_x, g_y)
    r_low = np.hypot(fx, fy)
    low = np.where(r_low < radius, np.cos(np.pi * r_low / (2 * radius)) ** 2, 0.0)
    # Synchronous detection: shift the carrier lobe to zero frequency in the space domain
    # before filtering. The shifted fringe term varies slowly and is nearly continuous across
    # the ends of the region, so the circular transform does not leak it; filtering the lobe
    # where it lies would spread the phase jump between opposite ends (the carrier phase is
    # not periodic in the region) over the columns next to the block edge.
    yy, xx = np.arange(ny)[:, None], np.arange(nx)[None, :]
    reference = np.exp(-2j * np.pi * (g_x * xx + g_y * yy))
    baseband = np.fft.ifft2(np.fft.fft2(region * window * reference) * low)
    analytic = baseband * np.conj(reference)
    background = np.fft.ifft2(np.fft.fft2(region * window) * low).real
    with np.errstate(invalid="ignore", divide="ignore"):
        analytic = analytic / window
        background = background / window
    # angle(analytic) is the phase of b*cos(.); the sine convention adds pi/2
    phase = (np.angle(analytic) + np.pi / 2) % (2 * np.pi)
    # Before shifting the carrier back, the baseband angle is the phase of the surface alone
    # (carrier removed, as in Takeda et al. 1982); same sine convention as `phase`.
    object_phase = (np.angle(baseband) + np.pi / 2) % (2 * np.pi)
    return PhaseMap(x=np.arange(*cols), y=np.arange(*rows), phase=phase, object_phase=object_phase,
                    modulation=2 * np.abs(analytic), background=background, carrier=(float(g_x), float(g_y)),
                    analytic=analytic)


def modulation_mask(modulation, min_fraction=0.3, reference=None):
    """Pixels whose fringe modulation reaches a fraction of a reference level.

    The modulation ``b`` (the envelope of the demodulated fringes) is low where there are
    no fringes to speak of: outside the illuminated field, beyond the far end of the
    platen strip, on the dark line of the block edge, under dust. Thresholding it gives a
    validity mask (the pupil) without looking at the raw intensity, which a bright but
    fringe-free area would pass.

    Parameters
    ----------
    modulation : array_like
        Modulation map, for instance ``PhaseMap.modulation``.
    min_fraction : float
        Threshold as a fraction of ``reference``.
    reference : float, optional
        Reference modulation; the median of the finite values if omitted.

    Returns
    -------
    numpy.ndarray of bool
    """
    modulation = np.asarray(modulation, dtype=float)
    ref = np.nanmedian(modulation) if reference is None else float(reference)
    with np.errstate(invalid="ignore"):
        return np.isfinite(modulation) & (modulation >= min_fraction * ref)


def unwrap_phase_map(phase_map: PhaseMap, method="quality", mask=None, min_modulation=None) -> UnwrappedMap:
    """Continuous phase of a region: unwrap its carrier-free phase in two dimensions.

    The carrier is removed before unwrapping, so the unwrapper only has to follow the
    phase the surface adds (few or no wraps for a flat face); the carrier ramp, known
    exactly, is added back to give the continuous fringe phase. Unwrapping uses
    :mod:`interflib.fringe.unwrap`; with ``method="quality"`` the modulation ``b`` is the
    quality map, so the integration goes through the best-modulated pixels first.

    Two limits come from the demodulation, not from the unwrapping. A linear term of the
    surface cannot be told apart from the carrier, so ``object_phase`` keeps only what
    departs from a plane fitted with the carrier estimate. And a curved surface makes the
    carrier-free phase non-periodic in the region: its jump between opposite ends leaks
    into the pass band and bends the phase near the ends (in a test with 1.2 fringes of
    Gaussian relief, some 0.03 to 0.08 rad within 60 to 150 px of the ends); a wider
    ``bandwidth`` or reading the phase farther inside reduces it.

    Parameters
    ----------
    phase_map : PhaseMap
        Output of :func:`demodulate_2d`.
    method : {"quality", "goldstein", "itoh"}
        Unwrapping algorithm. ``"itoh"`` takes no mask.
    mask : array_like of bool, optional
        Valid pixels of the region (the pupil): for instance, excluding the dark ends of a
        platen strip. Pixels outside it are ``nan`` in the result.
    min_modulation : float, optional
        If given, pixels with modulation below this fraction of the median modulation are
        also left out.

    Returns
    -------
    UnwrappedMap
    """
    shape = phase_map.object_phase.shape
    valid = np.ones(shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if valid.shape != shape:
        raise ValueError("mask shape %s does not match the region %s" % (valid.shape, shape))
    if min_modulation is not None:
        valid = valid & modulation_mask(phase_map.modulation, min_modulation)
    valid = valid & np.isfinite(phase_map.object_phase)
    wrapped = np.angle(np.exp(1j * phase_map.object_phase))
    if method == "quality":
        obj = unwrap_quality_guided(wrapped, quality=phase_map.modulation, mask=valid)
    elif method == "goldstein":
        obj = unwrap_goldstein(wrapped, mask=valid)
    elif method == "itoh":
        if not valid.all():
            raise ValueError("the Itoh unwrapper takes no mask; use 'quality' or 'goldstein'")
        obj = unwrap_itoh(wrapped)
    else:
        raise ValueError("unknown unwrapping method %r" % (method,))
    g_x, g_y = phase_map.carrier
    yy = (phase_map.y - phase_map.y[0])[:, None]
    xx = (phase_map.x - phase_map.x[0])[None, :]
    obj = np.where(valid, obj, np.nan)
    return UnwrappedMap(x=phase_map.x, y=phase_map.y, object_phase=obj,
                        phase=obj + 2 * np.pi * (g_x * xx + g_y * yy), mask=valid, method=method)


def map_column_phases(phase_map: PhaseMap, img, x_centres, rows, eval_row, half_width=2, deg=1,
                      bandwidth=0.5) -> ColumnPhases:
    """Phase at ``eval_row`` of column bands taken from a two-dimensional phase map.

    Each band averages the analytic signal of its ``2 * half_width + 1`` columns as
    complex numbers; from there on it is the same fit along ``y`` as
    :func:`fourier_column_phases`.

    Parameters
    ----------
    phase_map : PhaseMap
        Output of :func:`demodulate_2d`; the bands and ``rows`` must lie inside it.
    img : numpy.ndarray
        The interferogram the map was computed from (for the ``r2`` diagnostic).
    x_centres : array_like of int
        Centre column of each band.
    rows : tuple of int
        Rows whose pixel phases enter the fit.
    eval_row : float
        Row the phase is referred to.
    half_width, deg : int
        Band half-width and degree of the fit along ``y``.
    bandwidth : float
        The ``bandwidth`` used for the map (sets the correlation length of ``phase_error``).

    Returns
    -------
    ColumnPhases
    """
    x_centres = np.asarray(x_centres, dtype=int)
    x0, y0 = int(phase_map.x[0]), int(phase_map.y[0])
    if (x_centres.min() - half_width < x0 or x_centres.max() + half_width > phase_map.x[-1]
            or rows[0] < y0 or rows[1] > phase_map.y[-1] + 1):
        raise ValueError("bands or rows fall outside the phase map")
    rsel = slice(rows[0] - y0, rows[1] - y0)
    bands = [slice(x - half_width - x0, x + half_width + 1 - x0) for x in x_centres]
    analytic = np.stack([phase_map.analytic[rsel, b].mean(axis=1) for b in bands], axis=1)
    background = np.stack([phase_map.background[rsel, b].mean(axis=1) for b in bands], axis=1)
    band_phase = (np.angle(analytic) + np.pi / 2) % (2 * np.pi)
    corr = 1.0 / (2 * bandwidth * np.hypot(*phase_map.carrier))
    return _phases_at_row(band_phase, background, 2 * np.abs(analytic), _bands(img, x_centres, rows, half_width),
                          phase_map.y[rsel], eval_row, deg, corr, x_centres)


def fourier_edge_fraction_2d(img, edge_x, rows, eval_row, fft_rows=None, carrier=None, bandwidth=0.5,
                             gap=30, span=180, step=5, half_width=2, deg_y=1, deg=1, r2_min=0.3,
                             taper=0.0, min_cycles=4.0, inner=12, outer=20, order_sign=1) -> EdgeFractionResult:
    """Fringe fraction at the edge column with two-dimensional Fourier demodulation.

    A platen region left of the edge and a block region right of it are demodulated
    separately with :func:`demodulate_2d`, so the edge lies in neither, each with its own
    carrier unless one is given. The column phases of the bands from ``gap`` to ``span``
    px from the edge are then read from each map and compared at the edge exactly as in
    :func:`fourier_edge_fraction`.

    The regions reach beyond the bands, from ``inner`` px off the edge to ``span + outer``
    px (clipped to the image), because the phase near the ends of a region is degraded by
    spectral leakage of the background and of the conjugate lobe. With real images the
    result depends on these margins when the bands come close to the ends. Both regions
    must stay on their surface: ``inner`` clear of the dark line of the edge and well
    below ``gap``, and ``span + outer`` short of the far edge of the platen strip (or of
    the block face).

    Parameters
    ----------
    img : numpy.ndarray
        Interferogram, rows x columns.
    edge_x : float
        Edge column, px.
    rows : tuple of int
        Rows whose pixel phases enter the fit along ``y``.
    eval_row : float
        Evaluation row; the fraction refers to it.
    fft_rows : tuple of int, optional
        Rows of the two regions; ``rows`` if omitted.
    carrier, bandwidth, taper, min_cycles
        Passed to :func:`demodulate_2d`.
    gap, span, step, half_width, deg_y, deg, r2_min, order_sign
        As in :func:`fourier_edge_fraction`.
    inner, outer : int
        Region margins: each region runs from ``inner`` px off the edge to ``span + outer``
        px away from it.

    Returns
    -------
    EdgeFractionResult
        With ``fraction_error`` set to NaN: the two-dimensional filter correlates
        neighbouring bands along ``x``, the side fits absorb that correlated error and the
        residual scatter underestimates it (by about a factor of ten in synthetic tests).
        Use repeated frames or :func:`interflib.fringe.diagnostics.bandwidth_sensitivity`.
    """
    fft_rows = rows if fft_rows is None else fft_rows
    if not (fft_rows[0] <= rows[0] and rows[1] <= fft_rows[1]):
        raise ValueError("rows must lie inside fft_rows")
    if not 0 <= inner < gap - half_width:
        raise ValueError("inner must lie between 0 and gap - half_width")
    width = np.asarray(img).shape[1]
    xs_platen, xs_block = side_band_centres(edge_x, gap, span, step)
    xs_platen, xs_block = xs_platen.astype(int), xs_block.astype(int)
    cols_platen = (max(0, int(np.floor(edge_x - span - outer)) - half_width), int(np.floor(edge_x - inner)))
    cols_block = (int(np.ceil(edge_x + inner)), min(width, int(np.ceil(edge_x + span + outer)) + half_width + 1))
    # Each region gets its own carrier unless one is given: the synchronous detection of
    # demodulate_2d needs the carrier of the surface it demodulates, and the block-platen
    # wedge makes the two differ.
    maps = [demodulate_2d(img, fft_rows, c, carrier, bandwidth, taper, min_cycles) for c in (cols_platen, cols_block)]
    platen, block = [map_column_phases(m, img, xs, rows, eval_row, half_width, deg_y, bandwidth)
                     for m, xs in zip(maps, (xs_platen, xs_block))]
    result = fraction_at_edge(platen, block, edge_x, eval_row, gap, span, deg, r2_min, order_sign)
    return replace(result, fraction_error=float("nan"))
