"""End-to-end calibration runner for simulated Twyman-Green interferograms.

Runs the full pipeline on a simulated interferogram and reports the absolute
length:  load  ->  fringe fraction (sinusoidal fit)  ->  excess fraction.

The simulated images under ``interflib/simulations/interferograms/`` were generated
with the wavelength applied directly to the optical phase (no air index), so the
wavelength encoded in the filename is treated as the wavelength in air. For a real
measurement, build the length with
``GaugeBlockLength.from_environment(...)`` instead, which converts the vacuum
wavelength to air via ``interflib.dimensional.refraction`` (Birch & Downs).

Usage
-----
    python tools/run_calibration.py
    python tools/run_calibration.py "block_100.0_mm-WL_632.9905846-(simulated).png"
    python tools/run_calibration.py <image> --wavelength 632.99 --plot fit.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from interflib.io.images import load_image
from interflib.fringe.profile_fit import FringeFractionMeasurement
from interflib.dimensional.excess_fraction import GaugeBlockLength

IMAGES = Path(__file__).resolve().parents[1] / "simulations" / "interferograms"
DEFAULT_IMAGE = "block_10.0_mm-WL_632.9905846-(simulated).png"

# ROIs for the shared 2400x2400 simulated geometry: platen band vs. block band.
ROI_PLATEN = (700, 1900, 500, 900)   # (y1, y2, x1, x2)
ROI_BLOCK = (700, 1900, 1000, 1400)

# Linear thermal expansion coefficient of gauge-block steel, 1/K: the value declared by the
# manufacturer (Mahr), as quoted in CENAM calibration certificate CNM-CC-740-606/2023.
STEEL_CTE = 11.5e-6


def parse_filename(name: str) -> tuple[float, float | None]:
    """Return ``(nominal_length_mm, wavelength_nm_or_None)`` parsed from a filename."""
    m_len = re.search(r"block_([0-9.]+)_mm", name)
    if not m_len:
        raise ValueError(f"Cannot parse nominal length from '{name}'.")
    m_wl = re.search(r"WL_([0-9.]+)", name)
    wavelength = float(m_wl.group(1)) if m_wl else None
    return float(m_len.group(1)), wavelength


def run(image_name: str, wavelength_nm: float | None = None,
        plot_path: str | None = None) -> dict:
    """Run the end-to-end pipeline on one simulated interferogram."""
    nominal_mm, wl_from_name = parse_filename(image_name)
    wavelength_nm = wavelength_nm if wavelength_nm is not None else wl_from_name
    if wavelength_nm is None:
        raise ValueError(f"'{image_name}' has no wavelength in its name; pass --wavelength.")

    img = load_image(IMAGES / image_name)

    measurement = FringeFractionMeasurement(img, ROI_PLATEN, ROI_BLOCK)
    result = measurement.measure()
    fraction = result.fringe_fraction

    # Ground-truth fringe fraction encoded by the simulator: frac(2L/lambda).
    true_order = 2 * nominal_mm * 1e6 / wavelength_nm
    expected_fraction = true_order - np.floor(true_order)

    # Simulator uses lambda directly (no air index) -> it is the wavelength in the medium (nm).
    gb = GaugeBlockLength(wavelength=wavelength_nm, nominal_length=nominal_mm * 1e6, cte=STEEL_CTE,
                          temp_c=20.0, fringe_fraction=fraction)
    solution = gb.solve()
    length_mm = solution.length / 1e6
    deviation_nm = solution.length_dev

    print("=" * 64)
    print(f"END-TO-END  |  {image_name}")
    print("=" * 64)
    print(f"nominal length      : {nominal_mm} mm")
    print(f"wavelength (air)    : {wavelength_nm} nm")
    print(f"fit quality R^2     : platen {result.platen_quality.r2:.4f}  "
          f"block {result.block_quality.r2:.4f}")
    print(f"measured fraction   : {fraction:.4f}   (expected {expected_fraction:.4f})")
    print(f"integer order       : {solution.integer_order}")
    print(f"absolute length L20 : {length_mm:.8f} mm")
    print(f"deviation nominal   : {deviation_nm:+.1f} nm")

    if plot_path:
        _save_plot(measurement, length_mm, image_name, Path(plot_path))
        print(f"plot saved          : {plot_path}")

    return {
        "nominal_mm": nominal_mm, "wavelength_nm": wavelength_nm,
        "fraction": fraction, "expected_fraction": expected_fraction,
        "length_mm": length_mm, "deviation_nm": deviation_nm,
    }


def _save_plot(measurement, length_mm, title, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")

    ax = measurement.plot()
    ax.set_title(f"{title}\nfraction = {measurement.fringe_fraction:.4f}  ->  "
                 f"L20 = {length_mm:.6f} mm")
    ax.figure.tight_layout()
    ax.figure.savefig(out, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="End-to-end calibration on a simulated interferogram.",
    )
    ap.add_argument("image", nargs="?", default=DEFAULT_IMAGE,
                    help="filename in simulations/interferograms/")
    ap.add_argument("--wavelength", type=float, default=None,
                    help="wavelength in air (nm); needed if not in the filename")
    ap.add_argument("--plot", metavar="PATH", default=None,
                    help="save a fit plot to PATH")
    args = ap.parse_args()
    run(args.image, args.wavelength, args.plot)


if __name__ == "__main__":
    main()
