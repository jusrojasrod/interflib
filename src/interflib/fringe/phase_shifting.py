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
needed -- :func:`fraction_from_phase` reads the block-vs-platen fraction directly from the
wrapped map -- so unwrapping is left as a separate design decision.

This module has no I/O side effects: it takes and returns arrays.
"""
from dataclasses import dataclass

import numpy as np


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


def fraction_from_phase(phase, roi_platen, roi_block) -> float:
    """Fringe fraction between a gauge block and its platen from a wrapped-phase map.

    The demodulated phase carries the common reference tilt plus, over the block, the extra
    ``2 pi F`` of the central-length step. Taking the block-minus-platen phase difference
    row by row cancels the shared tilt (both regions span the same rows), leaving the
    fraction ``F = frac(2 L / lambda)``. This mirrors what ``profile_fit`` extracts from a
    single intensity image, but from an already-demodulated phase map, so a linear phase
    difference replaces the nonlinear sinusoidal fit.

    Parameters
    ----------
    phase : numpy.ndarray
        Wrapped phase map, radians (e.g. :attr:`PhaseStepResult.phase`).
    roi_platen, roi_block : tuple
        Regions of interest ``(y1, y2, x1, x2)`` over the reference platen and the block.
        Both must span the same rows; the reference tilt is assumed to run along the rows
        (the fringe direction), matching the ``profile_fit`` convention.

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
    fraction = _circular_mean(per_row_diff) / (2.0 * np.pi)
    return float(fraction % 1.0)
