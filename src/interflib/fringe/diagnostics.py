"""Diagnostics of a gauge-block interferogram built on the Fourier demodulation.

Three tools that help to judge a take before (or besides) measuring the fringe fraction:

- :func:`field_overview` demodulates the whole field at once, platen, edge and block
  together, and returns the same maps as :func:`interflib.fringe.fourier.demodulate_2d`:
  the fringes without background (real part of the filtered signal), their envelope (the
  modulation ``b``) and the carrier-free phase, where block and platen show as two phase
  surfaces with a step at the edge. Because the edge lies inside the transform, these maps
  are for looking, not for measuring: near the edge the phase is smeared over the
  resolution of the filter.
- :func:`edge_from_modulation` locates the block edge by the dip of the modulation across
  it, column by column (see also :func:`interflib.fringe.fourier.modulation_mask` for
  validity masks from the same envelope).
- :func:`bandwidth_sensitivity` repeats the edge-column fraction for a range of filter
  widths. On real images there is no true value to compare with, but a fraction that
  moves with the filter reveals a systematic effect of the demodulation (edge leakage,
  curvature, non-uniform illumination); the spread is a measure of it.

This module has no I/O side effects: it takes and returns arrays.
"""
from dataclasses import dataclass

import numpy as np

from .fourier import (
    PhaseMap,
    demodulate_2d,
    demodulate_columns,
    estimate_carrier_2d,
    fourier_edge_fraction,
    fourier_edge_fraction_2d,
)


def field_overview(img, rows, cols, carrier=None, carrier_cols=None, bandwidth=0.5, taper=0.0,
                   min_cycles=4.0) -> PhaseMap:
    """Two-dimensional demodulation of the whole field, for inspection.

    Parameters
    ----------
    img : numpy.ndarray
        Interferogram, rows x columns.
    rows, cols : tuple of int
        Field to demodulate; it may contain the block edge.
    carrier : tuple of float, optional
        ``(g_x, g_y)``. If omitted, it is estimated on ``carrier_cols`` (for instance a
        platen strip, so that the carrier-free phase of the platen is nearly flat and the
        block stands out by its fraction and wedge), or on the whole field.
    carrier_cols : tuple of int, optional
        Columns, within ``rows``, used to estimate the carrier.
    bandwidth, taper, min_cycles
        Passed to :func:`interflib.fringe.fourier.demodulate_2d`.

    Returns
    -------
    PhaseMap
        Not suited for measuring near the edge (see the module docstring).
    """
    if carrier is None:
        c0, c1 = carrier_cols if carrier_cols is not None else cols
        carrier = estimate_carrier_2d(np.asarray(img, dtype=float)[rows[0]:rows[1], c0:c1], min_cycles)
    return demodulate_2d(img, rows, cols, carrier, bandwidth, taper, min_cycles)


@dataclass
class ModulationEdge:
    """Block edge located by the dip of the fringe modulation.

    Attributes
    ----------
    edge_x : float
        Sub-pixel column of the modulation minimum, px.
    x : numpy.ndarray
        Columns searched.
    profile : numpy.ndarray
        Mean modulation of each column over the rows used.
    contrast : float
        Depth of the dip, ``1 - min / median`` of the profile: close to 0 when there is no
        clear dip (no edge in the search range, or one the fringes do not mark).
    """

    edge_x: float
    x: np.ndarray
    profile: np.ndarray
    contrast: float


def edge_from_modulation(img, rows, search, fft_rows=None, carrier=None, bandwidth=0.5, half_width=2,
                         min_cycles=4.0) -> ModulationEdge:
    """Column of the block edge from the minimum of the fringe modulation.

    Two effects make the column-wise modulation of
    :func:`interflib.fringe.fourier.demodulate_columns` dip at the edge: in real images the
    chamfer and the step scatter the light and draw a dark line; and a column band that
    straddles the edge averages the fringes of block and platen, which are out of step by
    the fraction ``F``, so their sum shrinks by ``|cos(pi F)|``. The second needs bands
    wider than one column (``half_width >= 1``) and fails for ``F`` near 0; the first
    does not depend on ``F``. Being column-wise, the profile keeps the resolution along
    ``x`` of the bands. The minimum is refined by a parabola through the three lowest
    columns; check ``contrast``: a shallow dip means no edge was found.

    Parameters
    ----------
    img : numpy.ndarray
        Interferogram, rows x columns.
    rows : tuple of int
        Rows whose modulation is averaged.
    search : tuple of int
        Column range ``(x0, x1)`` searched.
    fft_rows : tuple of int, optional
        Transform span; ``rows`` if omitted.
    carrier, bandwidth, min_cycles
        Passed to :func:`interflib.fringe.fourier.demodulate_columns`.
    half_width : int
        Half-width of each column band; at least 1 for the straddling effect.

    Returns
    -------
    ModulationEdge
    """
    fft_rows = rows if fft_rows is None else fft_rows
    xs = np.arange(search[0], search[1])
    demod = demodulate_columns(img, xs, fft_rows, carrier, bandwidth, half_width, 0.0, min_cycles)
    sel = slice(rows[0] - fft_rows[0], rows[1] - fft_rows[0])
    profile = np.nanmean(demod.modulation[sel], axis=0)
    i = int(np.nanargmin(profile))
    delta = 0.0
    if 0 < i < len(profile) - 1:
        a, b, c = profile[i - 1:i + 2]
        curvature = a - 2 * b + c
        if curvature > 0:
            delta = 0.5 * (a - c) / curvature
    contrast = float(1.0 - profile[i] / np.nanmedian(profile))
    return ModulationEdge(edge_x=float(xs[i] + delta), x=xs, profile=profile, contrast=contrast)


@dataclass
class BandwidthSensitivity:
    """Edge-column fraction as a function of the filter width.

    Attributes
    ----------
    bandwidth : numpy.ndarray
        Relative pass-band widths tried.
    fraction : numpy.ndarray
        Fraction obtained with each, in ``[0, 1)``.
    spread : float
        Range of the fractions (maximum minus minimum, circular), in fringes.
    method : str
        ``"2d"`` or ``"columns"``.
    """

    bandwidth: np.ndarray
    fraction: np.ndarray
    spread: float
    method: str


def bandwidth_sensitivity(img, edge_x, rows, eval_row, bandwidths=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
                          method="2d", **kwargs) -> BandwidthSensitivity:
    """Repeat the edge-column fraction for several filter widths.

    Parameters
    ----------
    img : numpy.ndarray
        Interferogram, rows x columns.
    edge_x, rows, eval_row
        As in :func:`interflib.fringe.fourier.fourier_edge_fraction`.
    bandwidths : sequence of float
        Relative pass-band widths, each in ``(0, 1)``.
    method : {"2d", "columns"}
        Two-dimensional (:func:`~interflib.fringe.fourier.fourier_edge_fraction_2d`) or
        column (:func:`~interflib.fringe.fourier.fourier_edge_fraction`) variant.
    **kwargs
        Passed to the edge-fraction function (``fft_rows``, ``gap``, ``span`` ...).

    Returns
    -------
    BandwidthSensitivity
    """
    if method == "2d":
        fn = fourier_edge_fraction_2d
    elif method == "columns":
        fn = fourier_edge_fraction
    else:
        raise ValueError("method must be '2d' or 'columns'")
    bws = np.asarray(bandwidths, dtype=float)
    fractions = np.array([fn(img, edge_x, rows, eval_row, bandwidth=float(b), **kwargs).fraction for b in bws])
    centred = (fractions - fractions[0] + 0.5) % 1.0 - 0.5
    return BandwidthSensitivity(bandwidth=bws, fraction=fractions, spread=float(centred.max() - centred.min()),
                                method=method)
