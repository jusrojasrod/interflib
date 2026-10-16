"""Wedge between the gauge-block face and the platen, from a single interferogram.

If the block face and the platen it is wrung to were exactly parallel, their fringes
would share the same spatial frequency and orientation, and the fringe fraction would
be the same everywhere on the face. A wedge between the two surfaces shows up as a
difference between their phase gradients:

* along ``x`` (columns), as the difference between the phase slopes of block and
  platen at the edge, ``wedge_x = g_b - g_p``, obtained from the column-resolved phase
  of :mod:`interflib.fringe.edge_comparison`;
* along ``y`` (rows), as the difference between their fringe frequencies,
  ``wedge_y = 1/Lambda_b - 1/Lambda_p``, from the band fits of the same comparison.

Both are in fringes per pixel. Any term common to the two surfaces (in particular the
tilt of the reference mirror, which sets the fringe density and orientation) cancels in
the difference, so the wedge is a property of the wrung assembly: it combines the
parallelism of the block faces, the wringing film and the platen flatness, and does not
separate them.

In double pass one fringe is half a wavelength of surface height, so a phase-gradient
difference ``dg`` (fringes/px) is a height slope ``dg * (lambda/2) / p``, with ``p`` the
size of one pixel on the object. With ``lambda`` in nm and ``p`` in mm the slope comes out
in nm/mm, which is numerically a wedge angle in microradians. Without ``p`` the wedge can
only be reported in fringes/px.

The wedge returned is the mean over the fit window and the column bands used, not a
point value: real surfaces are not perfectly flat, so the result depends somewhat on the
region. In particular ``wedge_y`` comes from the fringe frequency fitted in each band,
which needs several fringes in the window; on real gauge-block images it was found to
drift with the window height below about 300 rows and to settle for taller windows, so
a tall window (300 to 500 rows or more) is advisable for the wedge even when the
fraction itself is read with a short one. On ideal synthetic fringes the estimator is
unbiased for any window.

No acceptance threshold is built in: :func:`measure_wedge` takes an optional
``max_angle_urad`` and only then reports whether the wedge is within it.

This module has no I/O side effects: it takes and returns arrays.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .edge_comparison import EdgeFractionResult, edge_fraction


@dataclass
class WedgeMeasurement:
    """Block-platen wedge measured at the edge column and evaluation row.

    Attributes
    ----------
    wedge_x : float
        ``g_b - g_p``, difference of phase gradients along ``x``, fringes per px.
    wedge_y : float
        ``1/Lambda_b - 1/Lambda_p``, difference of fringe frequencies along ``y``,
        fringes per px (median over the valid bands of each side).
    angle_x_urad, angle_y_urad : float or None
        Wedge angle components, microradians (equivalently nm of height per mm on the
        object); ``None`` if no pixel scale was given.
    angle_urad : float or None
        Magnitude ``hypot(angle_x_urad, angle_y_urad)``, microradians.
    within_tolerance : bool or None
        ``angle_urad <= max_angle_urad`` if both are available, else ``None``.
    edge : EdgeFractionResult
        The underlying edge comparison (fraction, slopes and band phases).
    """

    wedge_x: float
    wedge_y: float
    angle_x_urad: Optional[float]
    angle_y_urad: Optional[float]
    angle_urad: Optional[float]
    within_tolerance: Optional[bool]
    edge: EdgeFractionResult

    def height_change_nm(self, length_mm: float) -> float:
        """Height difference produced by the wedge over ``length_mm`` on the object, nm.

        Uses the wedge magnitude, so it requires the pixel scale.
        """
        if self.angle_urad is None:
            raise ValueError("the pixel scale is needed to express the wedge as a height")
        return self.angle_urad * length_mm


def fringe_gradient_to_urad(gradient, wavelength_nm, pixel_mm):
    """Convert a phase-gradient difference in fringes/px to a wedge angle in microradians.

    ``angle = gradient * (wavelength/2) / pixel``; with the wavelength in nm and the pixel
    size on the object in mm, nm/mm equals microradians.
    """
    return gradient * (wavelength_nm / 2.0) / pixel_mm


def _median_frequency(cols, r2_min):
    ok = (cols.r2 >= r2_min) & np.isfinite(cols.period) & (cols.period != 0)
    if not ok.any():
        raise ValueError("no valid column bands to estimate the fringe frequency")
    return float(np.median(1.0 / np.abs(cols.period[ok])))


def measure_wedge(img, edge_x, rows, eval_row, period_guess, wavelength_nm=None, pixel_mm=None,
                  max_angle_urad=None, r2_min=0.3, **edge_kwargs) -> WedgeMeasurement:
    """Measure the block-platen wedge from one interferogram.

    Parameters
    ----------
    img : numpy.ndarray
        Interferogram, rows x columns (platen left of the edge, block right of it).
    edge_x : float
        Edge column, px.
    rows : tuple of int
        Fit window ``(y0, y1)``, rows.
    eval_row : float
        Evaluation row, px.
    period_guess : float
        Starting fringe period along ``y``, px per fringe.
    wavelength_nm : float, optional
        Laser wavelength, nm. Needed, with ``pixel_mm``, to express the wedge as an angle.
    pixel_mm : float, optional
        Size of one pixel on the object, mm.
    max_angle_urad : float, optional
        Acceptance limit for the wedge magnitude, microradians. Without it no
        acceptance verdict is given.
    r2_min : float
        Bands with a smaller coefficient of determination are left out.
    **edge_kwargs
        Passed on to :func:`interflib.fringe.edge_comparison.edge_fraction`
        (``gap``, ``span``, ``step``, ``half_width``, ``deg``).

    Returns
    -------
    WedgeMeasurement
    """
    if (wavelength_nm is None) != (pixel_mm is None):
        raise ValueError("give both wavelength_nm and pixel_mm, or neither")
    edge = edge_fraction(img, edge_x, rows, eval_row, period_guess, r2_min=r2_min, **edge_kwargs)
    wedge_x = edge.slope_block - edge.slope_platen
    wedge_y = _median_frequency(edge.block, r2_min) - _median_frequency(edge.platen, r2_min)

    angle_x = angle_y = angle = None
    if pixel_mm is not None:
        angle_x = float(fringe_gradient_to_urad(wedge_x, wavelength_nm, pixel_mm))
        angle_y = float(fringe_gradient_to_urad(wedge_y, wavelength_nm, pixel_mm))
        angle = float(np.hypot(angle_x, angle_y))
    within = None if (angle is None or max_angle_urad is None) else bool(angle <= max_angle_urad)
    return WedgeMeasurement(wedge_x=float(wedge_x), wedge_y=float(wedge_y), angle_x_urad=angle_x,
                            angle_y_urad=angle_y, angle_urad=angle, within_tolerance=within, edge=edge)
