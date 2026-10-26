"""Fringe fraction from a Twyman-Green interferogram by sinusoidal profile fitting.

Two regions of interest (the reference platen and the gauge block) are averaged into
1-D intensity profiles, and each is fitted with a sinusoidal interference model. The
fractional fringe order is the phase difference between the block and the platen,
divided by ``2*pi``.

The profiles run along the rows (``y``) and each fit is taken on the branch whose phase
increases towards ``+y``. The fraction is ``F`` when the interference order also increases
towards ``+y`` (``order_sign=+1``, the default) and ``1 - F`` otherwise; see
:func:`interflib.fringe.models.signed_fraction`.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from interflib.fringe.fourier import estimate_carrier
from interflib.fringe.models import (FitQuality, canonical_sinusoid, fit_quality,
                                     interference_model, signed_fraction, sine_seed)


@dataclass
class FringeFractionResult:
    """Outcome of a fringe-fraction measurement."""

    fringe_fraction: float
    platen_phase: float          # radians, in [0, 2*pi)
    block_phase: float           # radians, in [0, 2*pi)
    platen_quality: FitQuality
    block_quality: FitQuality


class FringeFractionMeasurement:
    """Measure the fringe fraction between a gauge block and its reference platen."""

    def __init__(self, image: np.ndarray, roi_platen: tuple, roi_block: tuple,
                 order_sign: int = 1):
        """
        Parameters
        ----------
        image : numpy.ndarray
            Grayscale interferogram.
        roi_platen, roi_block : tuple
            Regions of interest ``(y1, y2, x1, x2)`` over the reference platen and the
            gauge block.
        order_sign : {1, -1}
            ``+1`` when the interference order increases towards ``+y``, ``-1`` when it
            decreases (the fraction is then ``1 - F`` of the ``+1`` reading).
        """
        self.image = image
        self.order_sign = order_sign
        yp1, yp2, xp1, xp2 = roi_platen
        yb1, yb2, xb1, xb2 = roi_block
        self.platen_region = image[yp1:yp2, xp1:xp2]
        self.block_region = image[yb1:yb2, xb1:xb2]

        self.platen_profile = None
        self.block_profile = None
        self.pixel_indices = None

        self.platen_params = None
        self.block_params = None
        self.platen_quality = None
        self.block_quality = None

        self.platen_phase = None
        self.block_phase = None
        self.fringe_fraction = None

    def get_fringe_profiles(self) -> tuple:
        """Average each ROI across the fringe direction into a 1-D profile."""
        self.platen_profile = np.mean(self.platen_region, axis=1)
        self.block_profile = np.mean(self.block_region, axis=1)
        self.pixel_indices = np.arange(len(self.platen_profile))
        return self.platen_profile, self.block_profile, self.pixel_indices

    @staticmethod
    def _initial_parameter_guess(profile, x, frequency: float):
        """Seed ``(A, w, phi, C)`` for the non-linear fit at a given frequency.

        A fixed phase seed is not enough: when the seed frequency is slightly off (the
        block is seeded at the platen frequency, and a wedge between block and platen
        makes their periods differ) and the true phase lies about a quarter period from
        the seed, the optimiser can trade phase for frequency and converge to a
        spurious sinusoid. The seed is the linear known-frequency solve of
        :func:`interflib.fringe.models.sine_seed`.
        """
        return sine_seed(profile, x, frequency)

    @staticmethod
    def _canonicalize(params: np.ndarray) -> np.ndarray:
        """Fold the fitted ``(A, w, phi, C)`` onto ``A >= 0``, ``w > 0``.

        The model ``A*sin(w*x + phi) + C`` is invariant under
        ``(A, phi) -> (-A, phi + pi)`` and ``(w, phi) -> (-w, pi - phi)``, so two
        independent fits can land on different branches: a spurious ``pi`` (half a
        fringe) in their difference, or a phase running backwards (``1 - F``). Folding
        both fits onto one branch (:func:`interflib.fringe.models.canonical_sinusoid`)
        removes both before the fraction is computed.
        """
        A, w, phi = canonical_sinusoid(*params[:3])
        return np.array([A, w, phi, params[3]])

    def fit(self, pixels_between_fringes: float = None, max_iterations: int = 5000):
        """Fit both profiles and store the fit quality.

        The block fit is seeded at the platen's fitted frequency but its frequency stays
        free, so a period difference between block and platen (a wedge) is preserved.

        Parameters
        ----------
        pixels_between_fringes : float, optional
            Starting fringe period along the profile, px. If omitted it is estimated
            from the power spectrum of the platen profile
            (:func:`interflib.fringe.fourier.estimate_carrier`), which needs at least four
            fringes in the profile.
        max_iterations : int
            Maximum number of function evaluations of each non-linear fit.

        Returns ``self`` so calls can be chained.
        """
        if self.platen_profile is None:
            self.get_fringe_profiles()
        if pixels_between_fringes is None:
            pixels_between_fringes = 1.0 / estimate_carrier(self.platen_profile[:, None])

        p0_platen = self._initial_parameter_guess(self.platen_profile, self.pixel_indices,
                                                  2 * np.pi / pixels_between_fringes)
        self.platen_params, cov_platen = curve_fit(
            interference_model, self.pixel_indices, self.platen_profile,
            p0=p0_platen, maxfev=max_iterations,
        )
        self.platen_params = self._canonicalize(self.platen_params)
        self.platen_quality = fit_quality(self.pixel_indices, self.platen_profile,
                                          self.platen_params, cov_platen)

        # Seed the block at the platen's fitted frequency (the block frequency stays free).
        _, frequency, _, _ = self.platen_params
        p0_block = self._initial_parameter_guess(self.block_profile, self.pixel_indices, frequency)
        self.block_params, cov_block = curve_fit(
            interference_model, self.pixel_indices, self.block_profile,
            p0=p0_block, maxfev=max_iterations,
        )
        self.block_params = self._canonicalize(self.block_params)
        self.block_quality = fit_quality(self.pixel_indices, self.block_profile,
                                         self.block_params, cov_block)
        return self

    def calculate_fraction(self) -> float:
        """Fringe fraction ``= order_sign * (block_phase - platen_phase) / (2*pi)``, in ``[0, 1)``."""
        if self.platen_params is None or self.block_params is None:
            raise ValueError("Call fit() before calculate_fraction().")

        self.platen_phase = self.platen_params[2] % (2 * np.pi)
        self.block_phase = self.block_params[2] % (2 * np.pi)
        self.fringe_fraction = signed_fraction(self.block_phase - self.platen_phase, self.order_sign)
        return self.fringe_fraction

    def measure(self, pixels_between_fringes: float = None,
                max_iterations: int = 5000) -> FringeFractionResult:
        """Fit and compute the fraction in one call, returning a result object.

        ``pixels_between_fringes`` is the starting period, estimated if omitted (see :meth:`fit`).
        """
        self.fit(pixels_between_fringes, max_iterations)
        self.calculate_fraction()
        return FringeFractionResult(
            fringe_fraction=self.fringe_fraction, # type: ignore
            platen_phase=self.platen_phase, # type: ignore
            block_phase=self.block_phase,   # type: ignore
            platen_quality=self.platen_quality, # type: ignore
            block_quality=self.block_quality,   # type: ignore
        )

    def plot(self, ax=None):
        """Plot both profiles and their fits. Returns the matplotlib axes."""
        if self.fringe_fraction is None:
            raise ValueError("Call measure() (or fit + calculate_fraction) before plot().")
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        x = self.pixel_indices
        ax.plot(x, self.platen_profile, ".", color="#1f5482", label="Platen (data)")
        ax.plot(x, self.block_profile, ".", color="#c2141b", label="Block (data)")
        ax.plot(x, interference_model(x, *self.platen_params), "-", color="#1f5482", label="Platen fit")
        ax.plot(x, interference_model(x, *self.block_params), "-", color="#c2141b", label="Block fit")
        ax.set_title(f"Fringe fraction = {self.fringe_fraction:.4f}")
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Intensity")
        ax.legend()
        return ax

