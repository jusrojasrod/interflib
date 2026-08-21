"""End-to-end integration test: full pipeline on simulated interferograms.

Loads a simulated Twyman-Green interferogram from ``simulations/interferograms/``,
extracts the fringe fraction (sinusoidal fit) and resolves the absolute length by
excess fraction, then checks the recovered length against the value encoded in the
filename.

The simulator applies the wavelength directly to the optical phase (no air index),
so the wavelength in the filename is the wavelength in air and the fringe fraction is
``frac(2 L / lambda)``.
"""
from pathlib import Path

import numpy as np
import pytest

from interflib.io.images import load_image
from interflib.fringe.profile_fit import FringeFractionMeasurement
from interflib.dimensional.excess_fraction import GaugeBlockLength

IMAGES = Path(__file__).resolve().parents[1] / "simulations" / "interferograms"
ROI_PLATEN = (700, 1900, 500, 900)   # (y1, y2, x1, x2)
ROI_BLOCK = (700, 1900, 1000, 1400)

# name, nominal length (mm), wavelength in air (nm)
CASES = [
    ("block_10.0_mm-WL_632.9905846-(simulated).png", 10.0, 632.9905846),
    # 100 mm previously failed by half a fringe (-158 nm); regression guard.
    ("block_100.0_mm-WL_632.9905846-(simulated).png", 100.0, 632.9905846),
    # non-trivial encoded deviation of +158.2 nm from 100 mm.
    ("block_100.0001582_mm-WL_632.9905846-(simulated).png", 100.0001582, 632.9905846),
]


@pytest.mark.parametrize("name, nominal_mm, wavelength_nm", CASES)
def test_end_to_end_recovers_length(name, nominal_mm, wavelength_nm):
    img = load_image(IMAGES / name)

    fraction = FringeFractionMeasurement(img, ROI_PLATEN, ROI_BLOCK).measure().fringe_fraction

    # Ground-truth fringe fraction encoded by the simulator: frac(2 L / lambda).
    true_order = 2 * nominal_mm * 1e6 / wavelength_nm
    expected_fraction = true_order - np.floor(true_order)
    assert fraction == pytest.approx(expected_fraction, abs=0.02)

    # Simulator uses lambda directly (no air index) -> pass it as lambda_air.
    gb = GaugeBlockLength(lambda_air=wavelength_nm / 1000.0,
                          Lp=nominal_mm, C=11.5e-6, t_block=20.0,
                          obs_fraction=fraction)
    length_mm = gb.block_length()
    assert length_mm == pytest.approx(nominal_mm, abs=2e-5)  # within 20 nm
