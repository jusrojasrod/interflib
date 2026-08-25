"""Classical multi-wavelength method of exact fractions (Excess Fraction, EF).

This is the standard EF method. A single wavelength only reveals the fractional part
of the interference order, so the length is ambiguous modulo lambda/2. Measuring the
fringe fraction at several wavelengths at once resolves the integer order: only one
set of integer orders is jointly consistent with all wavelengths within the search
window fixed by a known nominal length.

The implementation follows the correction-to-nominal parameterisation of Liu et al.
(2022), which writes the Michelson-Benoit formulation used by Decker et al. (2003) as
a search over the difference ``c = L_real - L'`` between the real and the nominal
length. One wavelength anchors the sub-fringe position: its measured fraction is
reproduced exactly, so the candidate lengths are ``(N1 + eps1) * lambda1 / 2`` for the
integer orders ``N1`` allowed by the nominal tolerance. The remaining wavelengths then
select the integer order whose predicted fractions best match the measured ones.

Contrast with :mod:`interflib.dimensional.mefm`: the monochromatic solver keeps this
same ``c`` search but replaces the extra wavelengths with the uncertainty interval of a
prior calibration to pick the order. Here the disambiguation is done the standard way,
with two or more wavelengths, and no prior uncertainty bound is needed.

References
----------
T. Liu, J. Wu, R. Sato, H. Matsukuma, W. Gao, "Theoretical analysis on gauge block
length measurement by Excess Fraction method", Proc. 8th Int. Conf. on Nanomanufacturing
& 4th AET Symposium (Nanoman-AETS), IEEE, 2022, eqs. (1)-(12).

J. E. Decker et al., "Increasing the range of unambiguity in step-height measurement
with multiple-wavelength interferometry: application to absolute long gauge block
measurement", Appl. Opt. 42(28), 5670-5678 (2003).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def _fractional_part(x):
    """Positive fractional part of ``x`` (matches Liu's INT() = floor for x > 0)."""
    return x - np.floor(x)


def _circular_distance(a, b):
    """Shortest distance between two fringe fractions on the unit circle, in [0, 0.5].

    Fringe fractions live on a circle: 0.99 and 0.01 are 0.02 apart, not 0.98. This
    wraps the difference into [-0.5, 0.5) and returns its magnitude.
    """
    return np.abs(_fractional_part(a - b + 0.5) - 0.5)


@dataclass
class OrderCandidate:
    """One integer-order hypothesis evaluated by the solver."""

    integer_offset: int   # m: order offset from the nominal anchor order (Liu eq. 5)
    length: float         # candidate real length L' + c_j, in nm
    correction: float     # c_j = length - nominal, in nm
    residual: float       # RMS circular fraction residual over the selecting wavelengths


@dataclass
class ExcessFractionSolution:
    """Result of a multi-wavelength excess-fraction solve. Lengths in nanometres."""

    length: float                # resolved absolute length L_real, in nm
    correction: float            # c_J = L_real - nominal, in nm
    integer_orders: list[int]    # resolved integer order N_i, one per wavelength (input order)
    residual: float              # RMS fraction residual of the winning candidate, in [0, 0.5]
    margin: float                # residual gap to the runner-up; small margin => ambiguous
    candidates: list[OrderCandidate] = field(default_factory=list)  # all, ranked by residual


class MultiwavelengthEFSolver:
    """Standard multi-wavelength method of exact fractions."""

    def __init__(self, wavelengths: Sequence[float]):
        """
        Parameters
        ----------
        wavelengths : sequence of float
            Measurement wavelengths **in air**, in nanometres (nm). The first entry is
            the anchor: its measured fraction is reproduced exactly by the result. At
            least two wavelengths are required.
        """
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        if self.wavelengths.size < 2:
            raise ValueError(
                "The excess fraction method requires at least two wavelengths; use "
                "the monochromatic solver (mefm) for the single-wavelength case."
            )
        if np.any(self.wavelengths <= 0):
            raise ValueError("Wavelengths must be positive.")

    def solve(self, nominal_length: float, nominal_tolerance: float,
              fractions: Sequence[float]) -> ExcessFractionSolution:
        """Resolve the integer order and return the absolute length.

        Parameters
        ----------
        nominal_length : float
            Prior (nominal) length L', in nanometres. Typically from a mechanical
            comparison or from a previous calibration of the block.
        nominal_tolerance : float
            Half-width of the prior length error range (the ``+/-`` value), in
            nanometres. The search is restricted to real lengths within
            ``nominal_length +/- nominal_tolerance``.
        fractions : sequence of float
            Measured fringe fractions in [0, 1), one per wavelength, in the same order
            as ``wavelengths``.

        Returns
        -------
        ExcessFractionSolution
            The resolved length and the ranked integer-order candidates.

        Raises
        ------
        ValueError
            If ``fractions`` and ``wavelengths`` differ in length.
        RuntimeError
            If no candidate length falls within the nominal tolerance window.
        """
        fractions = np.asarray(fractions, dtype=float)
        if fractions.shape != self.wavelengths.shape:
            raise ValueError("fractions and wavelengths must have the same length.")

        lambda_anchor = self.wavelengths[0]
        fraction_anchor = fractions[0]

        # Anchor integer order at the nominal length (INT = floor for a positive argument).
        anchor_order_nominal = math.floor(2.0 * nominal_length / lambda_anchor)
        # Integer search half-range implied by the nominal tolerance (Liu eq. 7), padded
        # by one to avoid edge misses. Candidates are filtered by the actual tolerance
        # window below, so the pad never widens the accepted region.
        offset_range = math.floor(2.0 * nominal_tolerance / lambda_anchor) + 1

        selecting = self.wavelengths[1:]
        selecting_fractions = fractions[1:]

        candidates: list[OrderCandidate] = []
        for offset in range(-offset_range, offset_range + 1):
            anchor_order = anchor_order_nominal + offset
            # This length reproduces the anchor fraction exactly (Liu eq. 1).
            length = (anchor_order + fraction_anchor) * lambda_anchor / 2.0
            correction = length - nominal_length
            if abs(correction) > nominal_tolerance:
                continue
            predicted = _fractional_part(2.0 * length / selecting)
            residuals = _circular_distance(predicted, selecting_fractions)
            rms = float(np.sqrt(np.mean(residuals ** 2)))
            candidates.append(OrderCandidate(integer_offset=offset, length=length,
                                             correction=correction, residual=rms))

        if not candidates:
            raise RuntimeError(
                f"No candidate length falls within the nominal tolerance window "
                f"({nominal_length} +/- {nominal_tolerance} nm)."
            )

        candidates.sort(key=lambda c: c.residual)
        best = candidates[0]
        margin = (candidates[1].residual - best.residual
                  if len(candidates) > 1 else math.inf)

        integer_orders = [int(round(2.0 * best.length / lam - frac))
                          for lam, frac in zip(self.wavelengths, fractions)]

        return ExcessFractionSolution(
            length=best.length,
            correction=best.correction,
            integer_orders=integer_orders,
            residual=best.residual,
            margin=margin,
            candidates=candidates,
        )


if __name__ == "__main__":
    # Worked example from Liu et al. (2022), Table I/II: a 1400 um block, nominal error
    # range +/- 1 um, resolved from wavelength pairs sharing the 667.8184 nm anchor.
    solver = MultiwavelengthEFSolver(wavelengths=[667.8184, 587.5649])
    solution = solver.solve(nominal_length=1_400_000.0, nominal_tolerance=1_000.0,
                            fractions=[0.28, 0.85])
    print(f"L_real          : {solution.length / 1000.0:.6f} um "
          f"(Liu Table II: 1399.840861 um)")
    print(f"correction c_J  : {solution.correction:.3f} nm")
    print(f"integer orders  : {solution.integer_orders}")
    print(f"residual/margin : {solution.residual:.2e} / {solution.margin:.2e}")
