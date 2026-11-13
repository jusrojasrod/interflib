"""Phase recovery from a phase-shifted interferogram stack (phase stepping / PSI).

A phase-shifting interferometer records a stack of ``N`` frames while the reference
phase is stepped by known amounts ``delta_k``:

    I_k(x, y) = a(x, y) + b(x, y) * cos(phi(x, y) + delta_k),      k = 0 .. N-1

with ``a`` the background (our ``I0``), ``b`` the modulation amplitude (our ``I0 * V``)
and ``phi`` the wrapped optical phase. This module recovers ``phi`` (and ``a``, ``b`` and
the fringe visibility ``V = b / a``) from the stack. It is the temporal counterpart of
the single-shot ``profile_fit`` back-end for the fringe-to-phase stage.

Two families are provided:

* :func:`least_squares_psa` -- the general least-squares phase-shifting algorithm for
  *any* set of known steps (uniform or not, ``N >= 3``). Writing
  ``cos(phi + delta) = cos phi cos delta - sin phi sin delta`` turns each frame into a
  linear equation in the unknowns ``[a, b cos phi, b sin phi]``, solved per pixel in one
  vectorised pseudo-inverse (Bruning et al. 1974; Greivenkamp 1984).
* :func:`three_step`, :func:`four_step`, :func:`hariharan_five_step` -- named fixed-step
  algorithms. The three- and four-step forms are the least-squares solver at the classic
  ``pi/2`` steps; :func:`hariharan_five_step` is the windowed Schwider-Hariharan estimator,
  whose frequency-transfer function has a double zero at the carrier and is therefore
  insensitive to a *linear* miscalibration of the step to first order (Hariharan, Oreb &
  Eiju 1987).

The demodulation is analysed, across all of these, as a linear quadrature filter through
its frequency-transfer function (Servin, Quiroga & Padilla, *Fringe Pattern Analysis for
Optical Metrology*, 2014, ch. 2).

Phase *unwrapping* is deliberately out of scope here: these functions return the wrapped
phase in ``(-pi, pi]``. For the gauge-block fringe fraction the full-field unwrap is not
needed: :func:`edge_fraction_from_phase` reads the block-vs-platen fraction from the
wrapped map with the two phases compared at the same column, the edge, as
:mod:`interflib.fringe.edge_comparison` does for one image. :func:`fraction_from_phase`
compares two strips at different columns instead and is exact only when the phase does
not change along ``x`` other than by the block step.

This module has no I/O side effects: it takes and returns arrays.
"""
from dataclasses import dataclass

import numpy as np

from .edge_comparison import ColumnPhases, EdgeFractionResult, band_centres, compare
from .models import signed_fraction


@dataclass
class PhaseStepResult:
    """Outcome of a phase-stepping demodulation.

    Attributes
    ----------
    phase : numpy.ndarray
        Wrapped optical phase ``phi``, radians in ``(-pi, pi]``, shape of one frame.
    background : numpy.ndarray
        Background ``a`` (our ``I0``), same shape.
    modulation : numpy.ndarray
        Modulation amplitude ``b`` (our ``I0 * V``), same shape.
    visibility : numpy.ndarray
        Fringe visibility ``V = b / a``, same shape.
    steps : numpy.ndarray
        Phase steps ``delta_k`` used, radians.
    algorithm : str
        Name of the estimator that produced this result.
    """

    phase: np.ndarray
    background: np.ndarray
    modulation: np.ndarray
    visibility: np.ndarray
    steps: np.ndarray
    algorithm: str


def _visibility(modulation: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Fringe visibility ``b / a``, guarding a near-zero background."""
    with np.errstate(divide="ignore", invalid="ignore"):
        vis = np.where(background != 0.0, modulation / background, np.inf)
    return vis


def least_squares_psa(frames, steps) -> PhaseStepResult:
    """Recover the wrapped phase from a stack with arbitrary known phase steps.

    Solves, per pixel and in the least-squares sense, the linear system obtained by
    expanding ``I_k = a + b cos(phi + delta_k)`` into
    ``I_k = a + (b cos phi) cos(delta_k) + (b sin phi) (-sin delta_k)``. The design matrix
    ``[1, cos delta_k, -sin delta_k]`` is shared by every pixel, so one pseudo-inverse
    demodulates the whole field (Bruning et al. 1974; Greivenkamp 1984).

    Parameters
    ----------
    frames : array_like
        Interferogram stack of shape ``(N, ...)``; the leading axis indexes the ``N``
        phase-stepped frames and the trailing axes are the spatial layout.
    steps : array_like
        The ``N`` known phase steps ``delta_k``, radians.

    Returns
    -------
    PhaseStepResult

    Raises
    ------
    ValueError
        If fewer than three frames are given, if ``steps`` does not match the stack, or if
        the steps are degenerate (the design matrix is rank-deficient, e.g. all steps
        equal or spaced by a multiple of ``pi``).
    """
    frames = np.asarray(frames, dtype=float)
    steps = np.asarray(steps, dtype=float)

    n_frames = frames.shape[0]
    if n_frames < 3:
        raise ValueError(f"phase stepping needs at least 3 frames, got {n_frames}.")
    if steps.shape != (n_frames,):
        raise ValueError(
            f"steps must have one entry per frame: expected ({n_frames},), "
            f"got {steps.shape}."
        )

    design = np.column_stack([np.ones(n_frames), np.cos(steps), -np.sin(steps)])
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError(
            "degenerate phase steps: the design matrix is rank-deficient (the steps do "
            "not resolve background, cosine and sine independently)."
        )

    spatial_shape = frames.shape[1:]
    flat = frames.reshape(n_frames, -1)          # (N, M)
    coeffs = np.linalg.pinv(design) @ flat       # (3, M): [a, b cos phi, b sin phi]

    background = coeffs[0].reshape(spatial_shape)
    cos_term = coeffs[1].reshape(spatial_shape)
    sin_term = coeffs[2].reshape(spatial_shape)

    phase = np.arctan2(sin_term, cos_term)
    modulation = np.hypot(cos_term, sin_term)
    return PhaseStepResult(
        phase=phase,
        background=background,
        modulation=modulation,
        visibility=_visibility(modulation, background),
        steps=steps,
        algorithm="least_squares",
    )


def three_step(frames) -> PhaseStepResult:
    """Three-step algorithm at steps ``0, pi/2, pi`` (least-squares, exactly determined)."""
    result = least_squares_psa(frames, [0.0, np.pi / 2.0, np.pi])
    result.algorithm = "three_step"
    return result


def four_step(frames) -> PhaseStepResult:
    """Four-step algorithm at steps ``0, pi/2, pi, 3pi/2`` (Wyant's classic estimator)."""
    result = least_squares_psa(frames, [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])
    result.algorithm = "four_step"
    return result


def hariharan_five_step(frames) -> PhaseStepResult:
    """Schwider-Hariharan five-step algorithm at ``pi/2`` steps, robust to detuning.

    With frames at ``delta_k = 0, pi/2, pi, 3pi/2, 2pi`` the phase is

        phi = atan2( 2 (I4 - I2),  I1 - 2 I3 + I5 ),

    which equals ``atan2(4 b sin phi, 4 b cos phi)``. Averaging the redundant fifth frame
    against the first places a double zero of the frequency-transfer function at the
    carrier, so a linear error in the step size cancels to first order -- the property that
    makes this the workhorse five-frame estimator (Hariharan, Oreb & Eiju, *Appl. Opt.*
    26, 2504, 1987). The extra frame is why it tolerates PZT miscalibration that biases the
    three- and four-step forms.

    Parameters
    ----------
    frames : array_like
        Stack of exactly five phase-stepped frames, shape ``(5, ...)``.

    Returns
    -------
    PhaseStepResult

    Raises
    ------
    ValueError
        If the stack does not have exactly five frames.
    """
    frames = np.asarray(frames, dtype=float)
    if frames.shape[0] != 5:
        raise ValueError(
            f"the Hariharan five-step algorithm needs exactly 5 frames, got "
            f"{frames.shape[0]}."
        )
    i1, i2, i3, i4, i5 = frames

    sin_term = 2.0 * (i4 - i2)        # = 4 b sin phi
    cos_term = i1 - 2.0 * i3 + i5     # = 4 b cos phi
    phase = np.arctan2(sin_term, cos_term)
    modulation = np.hypot(sin_term, cos_term) / 4.0
    background = (i1 + 2.0 * i2 + 2.0 * i3 + 2.0 * i4 + i5) / 8.0
    return PhaseStepResult(
        phase=phase,
        background=background,
        modulation=modulation,
        visibility=_visibility(modulation, background),
        steps=np.arange(5) * np.pi / 2.0,
        algorithm="hariharan_5",
    )


def _circular_mean(values, axis=None):
    """Circular mean of angles (radians), robust across the ``+-pi`` wrap."""
    return np.angle(np.mean(np.exp(1j * np.asarray(values, dtype=float)), axis=axis))


def fraction_from_phase(phase, roi_platen, roi_block, order_sign=1) -> float:
    """Fringe fraction between a gauge block and its platen from a wrapped-phase map.

    The demodulated phase carries the common reference tilt plus, over the block, the extra
    ``2 pi F`` of the central-length step. Taking the block-minus-platen phase difference
    row by row cancels the shared tilt (both regions span the same rows), leaving the
    fraction ``F = frac(2 L / lambda)``. This mirrors what ``profile_fit`` extracts from a
    single intensity image, but from an already-demodulated phase map, so a linear phase
    difference replaces the nonlinear sinusoidal fit.

    The two strips sit at different columns, so any phase gradient along ``x`` common to
    both surfaces (fringes not exactly along the rows) adds ``g_x * dx`` to ``F``, with
    ``dx`` the strip separation. :func:`edge_fraction_from_phase` compares the phases at
    the same column and cancels it; this function is kept to reproduce strip analyses.

    Parameters
    ----------
    phase : numpy.ndarray
        Wrapped phase map, radians (e.g. :attr:`PhaseStepResult.phase`).
    roi_platen, roi_block : tuple
        Regions of interest ``(y1, y2, x1, x2)`` over the reference platen and the block.
        Both must span the same rows; the reference tilt is assumed to run along the rows
        (the fringe direction), matching the ``profile_fit`` convention.
    order_sign : {1, -1}
        ``+1`` when the demodulated phase increases with the interference order, ``-1``
        when it decreases (the fraction is then ``1 - F`` of the ``+1`` reading). With
        phase stepping this is set by the direction of the steps relative to the order,
        not by the reference tilt; see :func:`interflib.fringe.models.signed_fraction`.

    Returns
    -------
    float
        Fringe fraction in ``[0, 1)``.

    Raises
    ------
    ValueError
        If the two regions of interest do not span the same number of rows.
    """
    phase = np.asarray(phase, dtype=float)
    yp1, yp2, xp1, xp2 = roi_platen
    yb1, yb2, xb1, xb2 = roi_block
    platen = phase[yp1:yp2, xp1:xp2]
    block = phase[yb1:yb2, xb1:xb2]
    if platen.shape[0] != block.shape[0]:
        raise ValueError(
            "platen and block ROIs must span the same rows so the reference tilt cancels: "
            f"got {platen.shape[0]} vs {block.shape[0]} rows."
        )

    platen_line = _circular_mean(platen, axis=1)          # per-row phase over the platen
    block_line = _circular_mean(block, axis=1)            # per-row phase over the block
    per_row_diff = block_line - platen_line               # tilt cancels row by row
    return signed_fraction(_circular_mean(per_row_diff), order_sign)


def column_phases_from_map(phase, x_centres, rows, eval_row, modulation=None, half_width=2,
                           deg=1) -> ColumnPhases:
    """Phase at ``eval_row`` of column bands taken from a wrapped-phase map.

    Each band averages ``b exp(i phi)`` over its ``2 * half_width + 1`` columns (``b`` the
    modulation, so faint pixels weigh less; uniform if ``modulation`` is None), unwraps the
    band phase along ``y`` and fits a polynomial of degree ``deg`` in ``y - eval_row``; the
    constant term is the phase at ``eval_row``. Phases are returned in the convention of
    :mod:`interflib.fringe.edge_comparison` (the ``phi`` of ``A sin(w y + phi)``), that is
    ``pi / 2`` above the phase of ``a + b cos(phi + delta)``.

    Parameters
    ----------
    phase : numpy.ndarray
        Wrapped phase map, radians (e.g. :attr:`PhaseStepResult.phase`).
    x_centres : array_like of int
        Centre column of each band.
    rows : tuple of int
        Rows ``(y0, y1)`` that enter the fit along ``y``.
    eval_row : float
        Row the phase is referred to.
    modulation : numpy.ndarray, optional
        Modulation ``b`` (e.g. :attr:`PhaseStepResult.modulation`), same shape as ``phase``.
    half_width, deg : int
        Band half-width and degree of the fit along ``y``.

    Returns
    -------
    ~interflib.fringe.edge_comparison.ColumnPhases
        ``r2`` holds the mean resultant length of the fit residuals,
        ``|mean exp(i (Phi - fit))|`` in ``[0, 1]`` (1 for a clean band, near 0 for noise),
        because a phase map has no intensity profile to compare against; it plays the same
        role in the ``r2_min`` selection. ``phase_error`` is the standard uncertainty of the
        constant term from the residual scatter, assuming independent pixels. ``period`` is
        ``2 pi`` over the fitted slope (NaN with ``deg = 0`` or no slope).
    """
    phase = np.asarray(phase, dtype=float)
    weight = np.ones_like(phase) if modulation is None else np.asarray(modulation, dtype=float)
    x_centres = np.asarray(x_centres, dtype=int)
    y0, y1 = rows
    if (x_centres.min() - half_width < 0 or x_centres.max() + half_width >= phase.shape[1]
            or y0 < 0 or y1 > phase.shape[0]):
        raise ValueError("column bands or rows fall outside the phase map")

    signal = weight[y0:y1] * np.exp(1j * phase[y0:y1])
    y = np.arange(y0, y1, dtype=float) - eval_row
    design = np.vander(y, deg + 1)
    pseudo = np.linalg.pinv(design)
    unscaled = np.linalg.inv(design.T @ design)
    dof = max(len(y) - (deg + 1), 1)

    n = len(x_centres)
    out = {key: np.full(n, np.nan) for key in ("phase", "period", "r2", "phase_error")}
    for j, x in enumerate(x_centres):
        band = signal[:, x - half_width:x + half_width + 1].mean(axis=1)
        unwrapped = np.unwrap(np.angle(band) + np.pi / 2)
        coef = pseudo @ unwrapped
        resid = unwrapped - design @ coef
        out["phase"][j] = coef[-1] % (2 * np.pi)
        out["phase_error"][j] = np.sqrt(unscaled[-1, -1] * (resid @ resid) / dof)
        out["r2"][j] = np.abs(np.mean(np.exp(1j * resid)))
        if deg >= 1 and coef[-2] != 0:
            out["period"][j] = 2 * np.pi / coef[-2]
    return ColumnPhases(x=x_centres, phase=out["phase"], phase_unwrapped=np.unwrap(out["phase"]),
                        period=out["period"], r2=out["r2"], phase_error=out["phase_error"])


def edge_fraction_from_phase(phase, edge_x, rows, eval_row, modulation=None, gap=30, span=180, step=5,
                             half_width=2, deg_y=1, deg=1, r2_min=0.3, order_sign=1, at_x=None,
                             centre_half=None) -> EdgeFractionResult:
    """Fringe fraction at the edge column from a wrapped-phase map (phase stepping).

    The same comparison as :func:`interflib.fringe.edge_comparison.edge_fraction`: platen
    left of the edge and block right of it, column bands from ``gap`` to ``span`` px away
    from the edge, each side extrapolated to ``edge_x`` with a polynomial of degree ``deg``
    and the two phases subtracted there, so any phase term common to both surfaces cancels.
    The band phases come from :func:`column_phases_from_map` instead of a sinusoidal fit.

    Parameters
    ----------
    phase : numpy.ndarray
        Wrapped phase map, radians (e.g. :attr:`PhaseStepResult.phase`).
    edge_x : float
        Edge column, px.
    rows : tuple of int
        Rows ``(y0, y1)`` that enter the fit along ``y``.
    eval_row : float
        Evaluation row; the fraction refers to it.
    modulation : numpy.ndarray, optional
        Modulation ``b`` used to weight the pixels of each band.
    gap, span, step : int
        Band positions: from ``gap`` to ``span`` px from the edge, every ``step`` px.
    half_width : int
        Half-width of each band, px.
    deg_y, deg : int
        Degree of the fit along ``y`` of each band and along ``x`` of each side.
    r2_min : float
        Bands with a smaller ``r2`` (see :func:`column_phases_from_map`) are left out.
    order_sign : {1, -1}
        ``+1`` when the demodulated phase increases with the interference order (each step
        ``delta_k`` raises the order), ``-1`` otherwise; see
        :func:`interflib.fringe.models.signed_fraction`.
    at_x, centre_half
        As in :func:`interflib.fringe.edge_comparison.edge_fraction`: with ``at_x`` the
        block is read around that column and only the platen line is extended to it.

    Returns
    -------
    ~interflib.fringe.edge_comparison.EdgeFractionResult
    """
    xs_platen, xs_block = band_centres(edge_x, gap, span, step, at_x, centre_half)
    platen, block = [column_phases_from_map(phase, xs, rows, eval_row, modulation, half_width, deg_y)
                     for xs in (xs_platen, xs_block)]
    return compare(platen, block, edge_x, eval_row, gap, span, deg, r2_min, order_sign, at_x, centre_half)
