"""Brute-force coincidence method of exact fractions (Doiron & Beers / Tilford).

An independent formulation of the standard multi-wavelength excess-fraction method,
complementary to :mod:`interflib.dimensional.multiwavelength`.

Picture each wavelength as a ruler with pitch ``lambda/2`` and an indeterminate zero
(Doiron & Beers 1995, Gage Block Handbook, section 6.8). For a colour ``i`` the block
length is any ``L = (n_i + f_i) * lambda_i / 2`` with integer ``n_i`` (eq. 6.22), so one
colour alone allows a whole comb of candidate lengths. The true length is the one where
the combs of *all* colours coincide, within the fringe-fraction reading uncertainty
(about 1/20 of a fringe). With enough colours only one coincidence survives inside a
window set by a rough prior length.

This differs from the correction-to-nominal solver in two ways that make it a genuine
cross-check:

- It is **symmetric**: no wavelength is privileged. The reported length is a
  variance-weighted mean over every colour's nearest comb line, not the anchor-exact
  length of one colour. On clean data both solvers agree to machine precision; under
  fraction noise they differ slightly, which is informative.
- It makes ambiguity **explicit**: instead of always returning one best length, it
  returns every length where all colours coincide within tolerance. Two colours over a
  wide window typically leave several coincidences (the very ambiguity that longer
  wavelength sets, or tighter fractions, remove).

References
----------
T. Doiron, J. Beers, "The Gage Block Handbook", NIST Monograph 180, 1995, sections 6.5
(single-wavelength computation) and 6.8 (multiple-wavelength coincidence), eqs. 6.22-6.23.
The uncertainty and wavelength-choice analysis of the coincidence method is due to
Tilford (ref. [49] therein).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from interflib._deprecation import renamed_arguments

from ._fractions import circular_distance as _circular_distance
from ._fractions import fractional_part as _fractional_part


@dataclass
class CoincidenceSolution:
    """Result of a coincidence solve. Lengths in nanometres.

    A clean measurement has exactly one coincidence (``is_unique``). Zero means no length
    satisfied every colour within tolerance (data inconsistent or tolerance too tight);
    more than one means the wavelength set and window leave a genuine ambiguity.
    """

    length: float                  # best coincidence, variance-weighted over colours, nm
    residual: float                # worst-colour fringe residual at that length, in [0, 0.5]
    coincidence_lengths: list[float]      # every length where all colours coincide, nm
    is_unique: bool                       # exactly one coincidence found
    integer_orders: list[int]             # resolved order n_i at the reported length
    candidate_lengths_per_colour: list[list[float]] = field(default_factory=list)  # Fig 6.11 combs


class CoincidenceEFSolver:
    """Multi-wavelength excess fractions by brute-force comb coincidence."""

    def __init__(self, wavelengths: Sequence[float], fraction_uncertainty: float = 0.05):
        """
        Parameters
        ----------
        wavelengths : sequence of float
            Measurement wavelengths **in air**, in nanometres (nm). At least two.
        fraction_uncertainty : float, optional
            Per-colour reading uncertainty of the fringe fraction, in fringes. Default
            0.05 = 1/20 fringe, the conservative value of Doiron & Beers (section 6.8).
            Two colours coincide when their comb lines fall within twice this value.
        """
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        if self.wavelengths.size < 2:
            raise ValueError(
                "The excess fraction method requires at least two wavelengths; use "
                "the monochromatic solver (mefm) for the single-wavelength case."
            )
        if np.any(self.wavelengths <= 0):
            raise ValueError("Wavelengths must be positive.")
        if fraction_uncertainty <= 0:
            raise ValueError("fraction_uncertainty must be positive.")
        self.fraction_uncertainty = float(fraction_uncertainty)
        # Two comb lines (each a bar of half-width fraction_uncertainty) overlap when the
        # circular fraction residual between the colours is within their combined width.
        self._coincidence_tolerance = 2.0 * self.fraction_uncertainty

    @renamed_arguments(fractions="fringe_fractions")
    def solve(self, nominal_length: float, nominal_tolerance: float,
              fringe_fractions: Sequence[float]) -> CoincidenceSolution:
        """Find the length(s) where every colour's fringe comb coincides.

        Parameters
        ----------
        nominal_length : float
            Rough prior length, in nanometres, that localises the search window.
        nominal_tolerance : float
            Half-width of the search window (the ``+/-`` value), in nanometres.
        fringe_fractions : sequence of float
            Measured fringe fractions ``F_i`` in [0, 1), one per wavelength, in the same
            order as ``wavelengths``. The former name ``fractions`` is deprecated.

        Returns
        -------
        CoincidenceSolution

        Raises
        ------
        ValueError
            If ``fringe_fractions`` and ``wavelengths`` differ in length.
        RuntimeError
            If the search window contains no candidate length for the reference colour.
        """
        fractions = np.asarray(fringe_fractions, dtype=float)
        if fractions.shape != self.wavelengths.shape:
            raise ValueError("fractions and wavelengths must have the same length.")

        low = nominal_length - nominal_tolerance
        high = nominal_length + nominal_tolerance

        # Enumerate the reference colour's comb inside the window (colour 0 is used only
        # to localise candidates; the coincidence test and the estimate stay symmetric).
        comb0 = self._comb_lengths(0, fractions[0], low, high)
        if comb0.size == 0:
            raise RuntimeError(
                f"No candidate length falls within the search window "
                f"({nominal_length} +/- {nominal_tolerance} nm)."
            )

        tol = self._coincidence_tolerance
        coincidences: list[float] = []
        best_length = None
        best_residual = np.inf
        for candidate in comb0:
            residuals = _circular_distance(
                _fractional_part(2.0 * candidate / self.wavelengths), fractions)
            worst = float(np.max(residuals))
            if worst <= best_residual:            # keep a best-effort answer regardless
                best_residual = worst
                best_length = candidate
            if worst <= tol:
                coincidences.append(self._refine(candidate, fractions))

        # If a coincidence exists, report the one with the smallest worst-colour residual.
        if coincidences:
            refined = min(coincidences, key=lambda L: self._worst_residual(L, fractions))
            length = refined
        else:
            length = self._refine(best_length, fractions)

        residual = self._worst_residual(length, fractions)
        integer_orders = [int(round(2.0 * length / lam - frac))
                          for lam, frac in zip(self.wavelengths, fractions, strict=True)]
        combs = [self._comb_lengths(i, fractions[i], low, high).tolist()
                 for i in range(self.wavelengths.size)]

        return CoincidenceSolution(
            length=length,
            residual=residual,
            coincidence_lengths=sorted(coincidences),
            is_unique=len(coincidences) == 1,
            integer_orders=integer_orders,
            candidate_lengths_per_colour=combs,
        )

    def _comb_lengths(self, index: int, fraction: float, low: float,
                      high: float) -> np.ndarray:
        """Comb lines ``(n + fraction) * lambda/2`` of one colour inside [low, high]."""
        half = self.wavelengths[index] / 2.0
        n_low = int(np.ceil(low / half - fraction))
        n_high = int(np.floor(high / half - fraction))
        orders = np.arange(n_low, n_high + 1)
        return (orders + fraction) * half

    def _refine(self, length: float, fractions: np.ndarray) -> float:
        """Variance-weighted mean of every colour's comb line nearest to ``length``.

        Each colour resolves the length with an uncertainty proportional to its own
        ``lambda/2``, so the inverse-variance weight is ``(2/lambda)^2``. Coarser rulers
        (longer wavelengths) count for less. Symmetric in the colours by construction.
        """
        half = self.wavelengths / 2.0
        nearest_order = np.round(length / half - fractions)
        nearest_line = (nearest_order + fractions) * half
        weight = 1.0 / half ** 2
        return float(np.sum(weight * nearest_line) / np.sum(weight))

    def _worst_residual(self, length: float, fractions: np.ndarray) -> float:
        """Largest circular fringe-fraction residual over all colours at ``length``."""
        residuals = _circular_distance(
            _fractional_part(2.0 * length / self.wavelengths), fractions)
        return float(np.max(residuals))


if __name__ == "__main__":
    # Liu et al. (2022) four-colour data resolves to a single coincidence.
    solver = CoincidenceEFSolver(wavelengths=[667.8184, 587.5649, 501.5702, 447.1501])
    solution = solver.solve(nominal_length=1_400_000.0, nominal_tolerance=1_000.0,
                            fractions=[0.28, 0.85, 0.90, 0.23])
    print(f"length         : {solution.length / 1000.0:.6f} um")
    print(f"unique?        : {solution.is_unique}  "
          f"(coincidences: {[round(L/1000, 4) for L in solution.coincidence_lengths]})")
    print(f"worst residual : {solution.residual:.3f} fringe")
    print(f"integer orders : {solution.integer_orders}")
