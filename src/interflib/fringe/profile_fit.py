"""Fringe fraction from a Twyman-Green interferogram by sinusoidal profile fitting.

Wiki: wiki/conceptos/analisis-de-franjas.md

Two regions of interest (the reference platen and the gauge block) are averaged into
1-D intensity profiles, and each is fitted with a sinusoidal interference model. The
fractional fringe order is the phase difference between the block and the platen,
divided by ``2*pi``.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from interflib.fringe.models import FitQuality, fit_quality, interference_model


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

    def __init__(self, image: np.ndarray, roi_platen: tuple, roi_block: tuple):
        """
        Parameters
        ----------
        image : numpy.ndarray
            Grayscale interferogram.
        roi_platen, roi_block : tuple
            Regions of interest ``(y1, y2, x1, x2)`` over the reference platen and the
            gauge block.
        """
        self.image = image
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
    def _initial_parameter_guess(profile, pixels_between_fringes: float = 175.0):
        guess_amplitude = (np.max(profile) - np.min(profile)) / 2
        guess_offset = np.mean(profile)
        guess_frequency = 2 * np.pi / pixels_between_fringes
        guess_phase = 0.0
        return [guess_amplitude, guess_frequency, guess_phase, guess_offset]

    @staticmethod
    def _canonicalize_amplitude(params: np.ndarray) -> np.ndarray:
        """Force the fitted amplitude ``A >= 0``, absorbing its sign into the phase.

        The model ``A*sin(w*x + phi) + C`` is invariant under
        ``(A, phi) -> (-A, phi + pi)``, so independently fitting two profiles can land
        them on opposite sign branches and inject a spurious ``pi`` (half a fringe)
        into their phase difference. Canonicalizing every fit to ``A >= 0`` puts both
        profiles on the same branch before the fraction is computed.
        """
        A, w, phi, C = params
        if A < 0:
            A, phi = -A, phi + np.pi
        return np.array([A, w, phi, C])

    def fit(self, pixels_between_fringes: float = 175.0, max_iterations: int = 5000):
        """Fit both profiles (sharing the fringe frequency) and store the fit quality.

        Returns ``self`` so calls can be chained.
        """
        if self.platen_profile is None:
            self.get_fringe_profiles()

        p0_platen = self._initial_parameter_guess(self.platen_profile, pixels_between_fringes)
        self.platen_params, cov_platen = curve_fit(
            interference_model, self.pixel_indices, self.platen_profile,
            p0=p0_platen, maxfev=max_iterations,
        )
        self.platen_params = self._canonicalize_amplitude(self.platen_params)
        self.platen_quality = fit_quality(self.pixel_indices, self.platen_profile,
                                          self.platen_params, cov_platen)

        # Fit the block sharing the platen's fringe frequency.
        _, frequency, _, _ = self.platen_params
        p0_block = self._initial_parameter_guess(self.block_profile, pixels_between_fringes)
        p0_block[1] = frequency
        self.block_params, cov_block = curve_fit(
            interference_model, self.pixel_indices, self.block_profile,
            p0=p0_block, maxfev=max_iterations,
        )
        self.block_params = self._canonicalize_amplitude(self.block_params)
        self.block_quality = fit_quality(self.pixel_indices, self.block_profile,
                                         self.block_params, cov_block)
        return self

    def calculate_fraction(self) -> float:
        """Fringe fraction ``= (block_phase - platen_phase) / (2*pi)``, in ``[0, 1)``."""
        if self.platen_params is None or self.block_params is None:
            raise ValueError("Call fit() before calculate_fraction().")

        self.platen_phase = self.platen_params[2] % (2 * np.pi)
        self.block_phase = self.block_params[2] % (2 * np.pi)
        phase_difference = (self.block_phase - self.platen_phase) % (2 * np.pi)
        self.fringe_fraction = phase_difference / (2 * np.pi)
        return self.fringe_fraction

    def measure(self, pixels_between_fringes: float = 175.0,
                max_iterations: int = 5000) -> FringeFractionResult:
        """Fit and compute the fraction in one call, returning a result object."""
        self.fit(pixels_between_fringes, max_iterations)
        self.calculate_fraction()
        return FringeFractionResult(
            fringe_fraction=self.fringe_fraction,
            platen_phase=self.platen_phase,
            block_phase=self.block_phase,
            platen_quality=self.platen_quality,
            block_quality=self.block_quality,
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


if __name__ == "__main__":
    from pathlib import Path
    from interflib.io.images import load_image

    images = Path(__file__).resolve().parents[3] / "simulations" / "interferograms"
    img = load_image(images / "block_100.0_mm-WL_543.5-(simulated).png")

    measurement = FringeFractionMeasurement(img, (700, 1900, 500, 900), (700, 1900, 1000, 1400))
    result = measurement.measure()
    print(f"fringe fraction : {result.fringe_fraction:.4f}")
    print(f"platen fit R^2  : {result.platen_quality.r2:.4f}")
    print(f"block  fit R^2  : {result.block_quality.r2:.4f}")
