"""End-to-end gauge-block measurement: interferogram -> fringe fraction -> length at 20 C.

The chain, with the readers and solvers of the library:

1. the fringe fraction ``F`` at the edge column and evaluation row, with the three
   single-shot readers (sine fit, Fourier by columns, Fourier in two dimensions); one of
   them is the primary and the others are a cross-check;
2. the wavelength in air from the vacuum wavelength and the air temperature, pressure
   and humidity (modified Edlen);
3. the integer order from the nominal length and, when a prior deviation with its
   expanded uncertainty is given, also by MEFM;
4. the length reduced to 20 C.

It prints a summary of the inputs (with their origin), the air correction, the image
(edge, period, visibility, saturation, wedge), the fraction of every method with its
spread and repeatability, the integer order and the length, and the warnings that apply.
``--json`` writes the same summary to a file.

Input is either a **simulated** scene (the default; the summary then also reports the
error against the known truth) or a **real series** (``--series``): a folder of frames
of one burst plus its ``.meta.yaml`` sidecar, from which the conditions are read. Values
missing from the sidecar (the pressure of the 2026-10 campaign, for instance) must be
given on the command line; the script does not invent them.

The F-only contribution ``u(F) lambda / 2`` reported here is **not** the calibration
uncertainty: the air, temperature, wavelength, phase-change and wringing contributions
are outside its scope.

Usage
-----
    python tools/run_calibration.py                         # simulated 2 mm block
    python tools/run_calibration.py --series PATH --pressure 75300 --json out.json
    python tools/run_calibration.py --help
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")   # the camera TIFFs carry private tags

import numpy as np  # noqa: E402

from interflib.dimensional.excess_fraction import GaugeBlockLength  # noqa: E402
from interflib.dimensional.mefm import MonochromaticEFMSolver  # noqa: E402
from interflib.dimensional.refraction import EdlenRefractiveIndex  # noqa: E402
from interflib.fringe.edge_comparison import (detect_edge_column, edge_fraction, edge_line_centre,  # noqa: E402
                                              side_band_centres, side_fit)
from interflib.fringe.fourier import (demodulate_columns, estimate_carrier, fourier_edge_fraction,  # noqa: E402
                                      fourier_edge_fraction_2d)
from interflib.fringe.wedge import measure_wedge  # noqa: E402
from interflib.io.images import load_image  # noqa: E402
from interflib.simulation import PhaseModel, SceneGeometry, SensorModel, synthesize_interferogram  # noqa: E402

# Linear thermal expansion coefficient of gauge-block steel, 1/K: the value declared by the
# manufacturer (Mahr), as quoted in CENAM calibration certificate CNM-CC-740-606/2023.
STEEL_CTE = 11.5e-6

METHODS = ("sine", "fourier", "fourier2d")
HALF_WINDOW = 75             # fit window along y: eval_row +- 75 rows (about 150 rows)
FFT_HALF = 450               # Fourier transform span: eval_row +- 450 rows, clipped to the image
BENCH_EVAL_ROW = 1050        # evaluation row of the INM bench frames (campaign 2026-10)
BENCH_EDGE_ROWS = (600, 1600)  # rows used to locate the edge in bench frames

# Warning thresholds
SATURATION_WARN = 0.01       # fraction of saturated pixels in the analysis region
R2_WARN = 0.9                # median coefficient of determination of the band fits
WAVELENGTH_EFFECT_WARN_NM = 10.0   # length effect of the stated resolution of lambda_0
SIGMA_WARN = 3.0             # disagreement or margin, in standard uncertainties


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------
class Inputs(dict):
    """Input values with their origin: ``inputs[name] = (value, origin)``."""

    def value(self, name):
        return self[name][0]


def _first(*candidates):
    """First ``(value, origin)`` whose value is not None, else ``(None, "missing")``."""
    for value, origin in candidates:
        if value is not None:
            return value, origin
    return None, "missing"


def _decimals(value) -> int:
    text = repr(float(value))
    return len(text.split(".")[1]) if "." in text and "e" not in text else 0


def read_sidecar(folder: Path) -> dict:
    """The ``.meta.yaml`` sidecar next to a series folder (``<folder>.meta.yaml``)."""
    path = folder.with_name(folder.name + ".meta.yaml")
    if not path.exists():
        raise SystemExit(f"no sidecar {path.name} next to the series; the data contract requires one")
    import yaml   # optional dependency: pip install -e ".[tools]"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def collect_inputs(args, sidecar: dict | None) -> Inputs:
    """Merge command line, sidecar and stated assumptions; the CLI wins over the sidecar."""
    side = sidecar or {}
    cond = side.get("condiciones") or {}
    nominal_from_patron = None
    match = re.search(r"(\d+(?:\.\d+)?)\s*mm", str(side.get("patron", "")))
    if match:
        nominal_from_patron = float(match.group(1))
    sim = sidecar is None
    defaults = SIMULATION_DEFAULTS if sim else {}
    inputs = Inputs()
    for name, cli, side_value in (
            ("nominal_mm", args.nominal_mm, nominal_from_patron),
            ("wavelength_vac_nm", args.wavelength_vac, side.get("longitud_onda_nm")),
            ("air_temperature_c", args.temperature, cond.get("temperatura_C")),
            ("pressure_pa", args.pressure, cond.get("presion_Pa")),
            ("humidity_pct", args.humidity, cond.get("humedad_relativa_pct"))):
        inputs[name] = _first((cli, "command line"), (side_value, "sidecar"),
                              (defaults.get(name), "simulation setting"))
    missing = [n for n, (v, _) in inputs.items() if v is None]
    if missing:
        raise SystemExit(f"missing inputs {missing}: not in the sidecar, give them on the command line")
    inputs["block_temperature_c"] = _first((args.block_temp, "command line"),
                                           (inputs.value("air_temperature_c"),
                                            "simulation setting (= air)" if sim else "assumed equal to the air"))
    inputs["cte_per_k"] = _first((args.cte, "command line"), (STEEL_CTE, "steel, manufacturer value"))
    inputs["order_sign"] = _first((args.order_sign, "command line"), (side.get("sentido_orden"), "sidecar"),
                                  (1 if sim else None, "simulation (carrier towards +y)"),
                                  (1, "assumed (+1, library default)"))
    return inputs


# --------------------------------------------------------------------------------------
# Scene: simulated or real
# --------------------------------------------------------------------------------------
SIMULATION_DEFAULTS = {"nominal_mm": 2.0, "wavelength_vac_nm": 543.5163, "air_temperature_c": 20.0,
                       "pressure_pa": 75300.0, "humidity_pct": 50.0}


def simulated_scene(args, inputs: Inputs, wavelength_air: float) -> dict:
    """Frames of a simulated block of known length, at the wavelength in air of the inputs."""
    cte, t_block = inputs.value("cte_per_k"), inputs.value("block_temperature_c")
    true_length = inputs.value("nominal_mm") * 1e6 + args.true_dev_nm               # at 20 C, nm
    geometry = SceneGeometry.cs505mu(scale=args.scale, fov_mm=25.0)
    model = PhaseModel(wavelength_nm=wavelength_air, central_length_nm=true_length * (1 + cte * (t_block - 20.0)),
                       fringes_y=args.fringes_y, fringes_x=1.0)
    sims = [synthesize_interferogram(geometry, model, sensor=SensorModel.cs505mu(), seed=k)
            for k in range(args.frames)]
    mask = sims[0].block_mask
    height, width = mask.shape
    eval_row = args.eval_row if args.eval_row is not None else height // 2
    block_cols = np.flatnonzero(mask[eval_row])
    return {"source": f"simulated CS505MU scene, scale {args.scale}, {args.frames} frames",
            "frames": [s.image.astype(float) for s in sims], "eval_row": eval_row,
            "edge_x": float(block_cols[0]) - 0.5, "edge_origin": "block mask",
            "face_centre_x": float(block_cols.mean()), "pixel_mm": 25.0 / (width - 1),
            "truth": {"fraction": sims[0].fraction_truth, "length_nm": true_length}}


def real_series(args, folder: Path) -> dict:
    """Frames of one burst; the edge is located as in the 2026-10 campaign analysis."""
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff", ".png"))
    if not files:
        raise SystemExit(f"no image files in {folder}")
    frames = [load_image(p).astype(float) for p in files]
    mean = np.mean(frames, axis=0)
    eval_row = args.eval_row if args.eval_row is not None else BENCH_EVAL_ROW
    search, platen_left = None, None
    if args.edge_x is not None:
        edge_x, origin = args.edge_x, "command line"
    else:
        if args.edge_search:
            search = tuple(args.edge_search)
        else:
            search, platen_left = _search_from_platen(mean)
        step = detect_edge_column(mean, rows=BENCH_EDGE_ROWS, search=search)
        edge_x = edge_line_centre(mean, step, rows=BENCH_EDGE_ROWS)
        origin = f"dark edge line, searched in columns {search[0]}-{search[1]}"
    return {"source": f"series {folder.name}, {len(frames)} frames", "frames": frames, "eval_row": eval_row,
            "edge_x": float(edge_x), "edge_origin": origin, "face_centre_x": args.face_centre_x,
            "pixel_mm": args.pixel_mm, "truth": None, "edge_search": search, "platen_left": platen_left}


def _search_from_platen(img):
    # The platen strip starts where the fringe modulation rises (left border of the beam on
    # the platen); the block edge lies 150 to 400 px to its right on the INM bench.
    y0, y1 = BENCH_EDGE_ROWS
    contrast = np.convolve(img[y0:y1].std(axis=0), np.ones(9) / 9, "same")
    left = int(np.flatnonzero(contrast > 0.4 * np.percentile(contrast, 95)).min())
    return (left + 150, left + 400), left


# --------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------
def _read_fraction(method, img, edge_x, rows, eval_row, period, fft_rows, order_sign):
    if method == "sine":
        return edge_fraction(img, edge_x, rows, eval_row, period, order_sign=order_sign)
    if method == "fourier":
        return fourier_edge_fraction(img, edge_x, rows, eval_row, fft_rows=fft_rows, order_sign=order_sign)
    return fourier_edge_fraction_2d(img, edge_x, rows, eval_row, fft_rows=fft_rows, order_sign=order_sign)


def _wrap(values, reference):
    return reference + (np.asarray(values) - reference + 0.5) % 1.0 - 0.5


def measure(args, scene: dict, inputs: Inputs, wavelength_air: float, notes: list, artifacts=None) -> dict:
    frames, eval_row, edge_x = scene["frames"], scene["eval_row"], scene["edge_x"]
    img = np.mean(frames, axis=0)
    height, width = img.shape
    rows = (eval_row - HALF_WINDOW, eval_row + HALF_WINDOW)
    fft_rows = (max(0, eval_row - FFT_HALF), min(height, eval_row + FFT_HALF))
    order_sign = int(inputs.value("order_sign"))
    xe = int(round(edge_x))

    platen_profile = img[fft_rows[0]:fft_rows[1], xe - 100:xe - 40].mean(axis=1)
    period = 1.0 / estimate_carrier(platen_profile[:, None])

    # --- F with every reader on the averaged image
    readings = {}
    for method in METHODS:
        try:
            readings[method] = _read_fraction(method, img, edge_x, rows, eval_row, period, fft_rows, order_sign)
        except (ValueError, RuntimeError) as exc:
            notes.append(f"{method}: the reader failed ({exc})")
    if args.method not in readings:
        raise SystemExit(f"the primary method {args.method!r} failed on this image")
    primary = readings[args.method]
    reference = primary.fraction
    fractions = {m: float(_wrap(r.fraction, reference)) for m, r in readings.items()}
    spread = max(fractions.values()) - min(fractions.values())

    # --- repeatability of the primary method over the frames
    per_frame = []
    if len(frames) > 1:
        for frame in frames:
            try:
                per_frame.append(_read_fraction(args.method, frame, edge_x, rows, eval_row, period, fft_rows,
                                                order_sign).fraction)
            except (ValueError, RuntimeError):
                pass
    per_frame = _wrap(per_frame, reference) if per_frame else np.array([])
    s_frame = float(np.std(per_frame, ddof=1)) if per_frame.size > 1 else float("nan")
    u_mean = s_frame / math.sqrt(per_frame.size) if per_frame.size > 1 else float("nan")
    u_extrap = primary.fraction_error
    parts = [u for u in (u_extrap, u_mean) if np.isfinite(u)]
    u_fraction = float(math.sqrt(sum(u * u for u in parts))) if parts else float("nan")

    # --- F at the face centre: both side lines extended to x_c (exact for straight lines)
    x_c = scene.get("face_centre_x")
    if x_c is not None:
        shift = order_sign * (primary.slope_block - primary.slope_platen) * (x_c - edge_x)
        fraction_used, where = float((primary.fraction + shift) % 1.0), "face centre column"
    else:
        shift, fraction_used, where = None, primary.fraction, "edge column"

    # --- image quality: saturation, visibility per side, band fits, wedge
    max_adu = 2 ** args.bit_depth - 1
    region = (slice(*fft_rows), slice(max(0, xe - 200), min(width, xe + 200)))
    saturated = float(np.mean([np.mean(f[region] >= max_adu) for f in frames]))
    xs_platen, xs_block = side_band_centres(edge_x, 30, 180, 15)
    visibility = {}
    for side, xs in (("platen", xs_platen), ("block", xs_block)):
        demod = demodulate_columns(img, xs.astype(int), fft_rows)
        sel = slice(rows[0] - fft_rows[0], rows[1] - fft_rows[0])
        visibility[side] = float(np.median(demod.modulation[sel] / demod.background[sel]))
    r2 = None
    if "sine" in readings:
        bands = np.concatenate([readings["sine"].platen.r2, readings["sine"].block.r2])
        r2 = float(np.nanmedian(bands))
    try:
        pixel_mm = scene.get("pixel_mm")      # the angle needs the scale; without it, fringes/px only
        wedge = measure_wedge(img, edge_x, rows, eval_row, period, pixel_mm=pixel_mm,
                              wavelength_nm=wavelength_air if pixel_mm else None)
    except (ValueError, RuntimeError) as exc:
        wedge = None
        notes.append(f"wedge: the measurement failed ({exc})")

    if artifacts is not None:
        artifacts.update(image=img, primary=primary, rows=rows, fft_rows=fft_rows, face_centre_x=x_c,
                         edge_search=scene.get("edge_search"), platen_left=scene.get("platen_left"))
    return {
        "image": {"source": scene["source"], "size_px": [height, width], "frames": len(frames),
                  "edge_x_px": edge_x, "edge_origin": scene["edge_origin"], "eval_row_px": eval_row,
                  "fit_rows": list(rows), "fourier_rows": list(fft_rows), "period_px": period,
                  "fringes_in_fit_window": (rows[1] - rows[0]) / period, "saturated_fraction": saturated,
                  "visibility_platen": visibility["platen"], "visibility_block": visibility["block"],
                  "median_band_r2_sine": r2,
                  "wedge_x_fringes_per_px": wedge.wedge_x if wedge else None,
                  "wedge_y_fringes_per_px": wedge.wedge_y if wedge else None,
                  "wedge_x_urad": wedge.angle_x_urad if wedge else None,
                  "wedge_y_urad": wedge.angle_y_urad if wedge else None},
        "fraction": {"order_sign": order_sign, "primary_method": args.method,
                     "methods": {m: {"fraction": fractions[m], "fraction_error": readings[m].fraction_error}
                                 for m in readings},
                     "spread_between_methods": spread,
                     "frames_read": int(per_frame.size), "repeatability_per_frame": s_frame,
                     "u_mean_of_frames": u_mean, "u_extrapolation": u_extrap, "u_fraction": u_fraction,
                     "edge_fraction": primary.fraction, "face_centre_x_px": x_c,
                     "shift_edge_to_centre": shift, "fraction_used": fraction_used, "evaluated_at": where},
    }


# --------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------
def run(args, artifacts=None) -> dict:
    """Measure one block and return the summary (also the content of ``--json``).

    ``artifacts``, if a dict, receives the averaged image and the primary reading for :func:`plot_regions`.
    """
    notes = []
    folder = Path(args.series) if args.series else None
    inputs = collect_inputs(args, read_sidecar(folder) if folder else None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        air = EdlenRefractiveIndex(inputs.value("air_temperature_c"), inputs.value("pressure_pa"),
                                   relative_humidity=inputs.value("humidity_pct"))
        lam0 = inputs.value("wavelength_vac_nm")
        n_air, wavelength_air = air.index_refraction(lam0), air.wavelength_in_air(lam0)
    notes += [f"air index: {w.message}" for w in caught]

    scene = simulated_scene(args, inputs, wavelength_air) if folder is None else real_series(args, folder)
    measured = measure(args, scene, inputs, wavelength_air, notes, artifacts)
    frac = measured["fraction"]

    nominal_nm = inputs.value("nominal_mm") * 1e6
    solution = GaugeBlockLength(wavelength=wavelength_air, nominal_length=nominal_nm, cte=inputs.value("cte_per_k"),
                                temp_c=inputs.value("block_temperature_c"),
                                fringe_fraction=frac["fraction_used"]).solve()
    half = wavelength_air / 2
    u_length_f = frac["u_fraction"] * half
    order = {"nominal_fraction": solution.nominal_fraction, "integer_order": solution.integer_order,
             "quarter_wave_nm": wavelength_air / 4,
             "margin_to_quarter_wave_nm": wavelength_air / 4 - abs(solution.length_dev)}
    if args.prior_dev_nm is not None and args.prior_U_nm is not None:
        try:
            mefm = MonochromaticEFMSolver(wavelength_air).solve(nominal_nm, args.prior_dev_nm, args.prior_U_nm,
                                                                frac["fraction_used"])
            order["mefm_length_dev_nm"] = mefm.length_dev
            order["mefm_acceptance_interval_nm"] = list(mefm.acceptance_interval)
            if abs(mefm.length_dev - solution.length_dev) > 1e-6:
                notes.append(f"MEFM picks another order than the nominal prior: d = {mefm.length_dev:+.1f} nm "
                             f"against {solution.length_dev:+.1f} nm")
        except (ValueError, RuntimeError) as exc:
            notes.append(f"MEFM: {exc}")

    summary = {
        "inputs": {name: {"value": value, "origin": origin} for name, (value, origin) in inputs.items()},
        "air": {"n_minus_1": n_air - 1, "wavelength_air_nm": wavelength_air, "half_wavelength_nm": half},
        **measured,
        "order": order,
        "length": {"length_20c_nm": solution.length, "length_20c_mm": solution.length / 1e6,
                   "deviation_from_nominal_nm": solution.length_dev, "u_from_fraction_nm": u_length_f},
    }
    if scene["truth"]:
        truth = scene["truth"]
        summary["truth"] = {"fraction": truth["fraction"], "length_20c_nm": truth["length_nm"],
                            "fraction_error": float(_wrap(frac["fraction_used"], truth["fraction"]) - truth["fraction"]),
                            "length_error_nm": solution.length - truth["length_nm"]}
    summary["warnings"] = notes + _warnings(summary, inputs)
    return summary


def _warnings(summary, inputs: Inputs) -> list:
    out = []
    for name in ("pressure_pa", "block_temperature_c", "order_sign"):
        value, origin = inputs[name]
        if origin.startswith("assumed"):
            out.append(f"{name} = {value} is {origin}")
    lam0, lam_origin = inputs["wavelength_vac_nm"]
    if lam_origin != "simulation setting":
        resolution = 0.5 * 10.0 ** -_decimals(lam0)
        effect = inputs.value("nominal_mm") * 1e6 * resolution / lam0
        if effect > WAVELENGTH_EFFECT_WARN_NM:
            out.append(f"lambda_0 = {lam0} nm ({lam_origin}) is stated to +-{resolution:g} nm: that alone moves L by "
                       f"up to {effect:.0f} nm (quarter wave {summary['order']['quarter_wave_nm']:.0f} nm); "
                       "the integer order may be wrong")
    img, frac, length = summary["image"], summary["fraction"], summary["length"]
    if img["saturated_fraction"] > SATURATION_WARN:
        out.append(f"{100 * img['saturated_fraction']:.1f} % of the pixels in the analysis region are saturated")
    if img["median_band_r2_sine"] is not None and img["median_band_r2_sine"] < R2_WARN:
        out.append(f"median R^2 of the band fits is {img['median_band_r2_sine']:.2f} (< {R2_WARN})")
    u = frac["u_fraction"]
    if np.isfinite(u) and u > 0 and frac["spread_between_methods"] > SIGMA_WARN * u:
        out.append(f"the methods disagree by {frac['spread_between_methods']:.4f} fringe, more than "
                   f"{SIGMA_WARN:g} u(F) = {SIGMA_WARN * u:.4f}")
    if summary["order"]["margin_to_quarter_wave_nm"] < SIGMA_WARN * max(length["u_from_fraction_nm"], 1.0):
        out.append("the deviation from nominal is close to a quarter wave: the integer order from the nominal "
                   "prior is not reliable")
    out.append("the nominal prior gives the right order only if the true deviation is within a quarter wave "
               f"({summary['order']['quarter_wave_nm']:.0f} nm)")
    if frac["evaluated_at"] == "edge column":
        out.append("F is evaluated at the edge column, not at the face centre where the central length is "
                   "defined; give --face-centre-x to extend both sides to the centre (the block-platen wedge "
                   "makes the two differ)")
    else:
        out.append("F at the face centre extends the side lines beyond the measured bands; u(F) does not include "
                   "that extrapolation")
    if not np.isfinite(frac["u_extrapolation"]):
        out.append(f"the {frac['primary_method']} reader gives no extrapolation uncertainty; u(F) is the "
                   "repeatability only")
    return out


def _fmt(value, spec=".4f"):
    if value is None:
        return "-"
    if isinstance(value, float):
        return "nan" if not np.isfinite(value) else format(value, spec)
    return str(value)


def format_summary(s: dict) -> str:
    """Console report of the summary returned by :func:`run`."""
    line = "=" * 78
    out = [line, "END-TO-END GAUGE-BLOCK MEASUREMENT", line]
    length = s["length"]
    out.append(f"Length at 20 C : {length['length_20c_mm']:.7f} mm   "
               f"(d = {length['deviation_from_nominal_nm']:+.1f} nm from nominal; "
               f"F contribution u = {_fmt(length['u_from_fraction_nm'], '.2f')} nm)")
    out.append("\nInputs")
    units = {"nominal_mm": "mm", "wavelength_vac_nm": "nm", "air_temperature_c": "C", "pressure_pa": "Pa",
             "humidity_pct": "%", "block_temperature_c": "C", "cte_per_k": "1/K", "order_sign": ""}
    for name, item in s["inputs"].items():
        out.append(f"  {name:<22}{_fmt(item['value'], 'g'):>12} {units[name]:<4} [{item['origin']}]")
    air = s["air"]
    out.append(f"\nAir            n - 1 = {air['n_minus_1']:.4e}   lambda_air = {air['wavelength_air_nm']:.5f} nm")
    img = s["image"]
    out.append("\nImage")
    out.append(f"  source        {img['source']} ({img['size_px'][0]} x {img['size_px'][1]} px)")
    out.append(f"  edge          x = {img['edge_x_px']:.2f} px ({img['edge_origin']}); evaluation row {img['eval_row_px']}")
    out.append(f"  fringes       period {img['period_px']:.2f} px; {img['fringes_in_fit_window']:.1f} fringes "
               f"in the fit window {img['fit_rows']}")
    out.append(f"  visibility    platen {img['visibility_platen']:.2f}, block {img['visibility_block']:.2f}; "
               f"saturated {100 * img['saturated_fraction']:.2f} %; median band R^2 {_fmt(img['median_band_r2_sine'], '.3f')}")
    wedge = (f"  wedge         dg_x = {_fmt(img['wedge_x_fringes_per_px'], '.2e')}, "
             f"dg_y = {_fmt(img['wedge_y_fringes_per_px'], '.2e')} fringe/px")
    if img["wedge_x_urad"] is not None:
        wedge += f"  ({_fmt(img['wedge_x_urad'], '.1f')}, {_fmt(img['wedge_y_urad'], '.1f')} urad)"
    out.append(wedge)
    frac = s["fraction"]
    out.append(f"\nFringe fraction (order_sign {frac['order_sign']:+d}; primary: {frac['primary_method']})")
    for method, r in frac["methods"].items():
        mark = "*" if method == frac["primary_method"] else " "
        out.append(f"  {mark} {method:<10} F = {r['fraction']:.4f}   u_e(F) = {_fmt(r['fraction_error'])}")
    out.append(f"  spread between methods {frac['spread_between_methods']:.4f} fringe "
               f"({frac['spread_between_methods'] * s['air']['half_wavelength_nm']:.1f} nm)")
    if frac["frames_read"] > 1:
        out.append(f"  repeatability ({frac['frames_read']} frames) s = {frac['repeatability_per_frame']:.4f}, "
                   f"s/sqrt(n) = {frac['u_mean_of_frames']:.4f} fringe")
    if frac["shift_edge_to_centre"] is not None:
        out.append(f"  edge F = {frac['edge_fraction']:.4f} -> face centre (x = {frac['face_centre_x_px']:.1f} px) "
                   f"F = {frac['fraction_used']:.4f}")
    out.append(f"  F used = {frac['fraction_used']:.4f} at the {frac['evaluated_at']}, u(F) = {_fmt(frac['u_fraction'])}")
    order = s["order"]
    out.append(f"\nInteger order  kappa = {order['integer_order']}, nominal fraction F' = {order['nominal_fraction']:.4f}, "
               f"margin to lambda/4 = {order['margin_to_quarter_wave_nm']:.1f} nm")
    if "mefm_length_dev_nm" in order:
        lo, hi = order["mefm_acceptance_interval_nm"]
        out.append(f"  MEFM          d = {order['mefm_length_dev_nm']:+.1f} nm (prior interval {lo:+.1f} to {hi:+.1f} nm)")
    if "truth" in s:
        t = s["truth"]
        out.append(f"\nTruth          F = {t['fraction']:.4f}, L = {t['length_20c_nm'] / 1e6:.7f} mm;   "
                   f"error F {t['fraction_error']:+.4f}, L {t['length_error_nm']:+.2f} nm")
    out.append("\nWarnings")
    out += [f"  - {w}" for w in s["warnings"]] or ["  none"]
    out.append(line)
    return "\n".join(out)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


PLATEN_COLOUR, BLOCK_COLOUR, EDGE_COLOUR, FOURIER_COLOUR = "#1f5482", "#c2141b", "#e0a100", "#2a9d3c"


def plot_regions(path, summary: dict, data: dict) -> None:
    """Save a figure of the areas the measurement used.

    (a) the whole frame: platen strip border and edge search range (real series), Fourier
    rows and the zoomed box; (b) the zoom around the edge: edge, evaluation row, fit window,
    platen and block bands, the two regions of the 2-D Fourier reader, the columns used for
    the period and the face centre; (c) the band phases of the primary reader at the
    evaluation row, with the straight lines of each side extended to the edge.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    img = data["image"]
    height, width = img.shape
    x_e, y_e = summary["image"]["edge_x_px"], summary["image"]["eval_row_px"]
    rows, fft_rows = data["rows"], data["fft_rows"]
    gap, span, half = 30, 180, 2                      # band layout of the readers (their defaults)
    lo, hi = np.percentile(img, [1, 99.5])
    fig = plt.figure(figsize=(16, 5.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.1, 1.0], wspace=0.28)
    ax_frame, ax_zoom, ax_phase = (fig.add_subplot(grid[0, i]) for i in range(3))

    # (a) whole frame
    ax_frame.imshow(img, cmap="gray", vmin=lo, vmax=hi)
    ax_frame.axhspan(*fft_rows, color=FOURIER_COLOUR, alpha=0.12, label=f"Fourier rows {fft_rows[0]}-{fft_rows[1]}")
    if data.get("edge_search"):
        ax_frame.axvspan(*data["edge_search"], color=EDGE_COLOUR, alpha=0.18, label="edge search range")
    if data.get("platen_left") is not None:
        ax_frame.axvline(data["platen_left"], color=PLATEN_COLOUR, ls=":", lw=1.2,
                         label="left border of the platen strip")
    ax_frame.axvline(x_e, color=EDGE_COLOUR, lw=1.2, label=f"edge x = {x_e:.1f}")
    zx = (max(0, int(x_e) - 270), min(width, int(x_e) + 270))
    zy = (max(0, y_e - 300), min(height, y_e + 300))
    ax_frame.add_patch(Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], fill=False, ec="w", lw=1.2, ls="--"))
    ax_frame.set_title("(a) frame\nwhite box: zoom in (b)")
    ax_frame.legend(fontsize=7, loc="lower left", framealpha=0.85)

    # (b) zoom around the edge
    crop = img[zy[0]:zy[1], zx[0]:zx[1]]
    ax_zoom.imshow(crop, cmap="gray", vmin=lo, vmax=hi, extent=(zx[0], zx[1], zy[1], zy[0]), aspect="auto")
    for x0, x1, colour, name in ((x_e - span - half, x_e - gap + half, PLATEN_COLOUR, "platen bands"),
                                 (x_e + gap - half, x_e + span + half, BLOCK_COLOUR, "block bands")):
        ax_zoom.add_patch(Rectangle((x0, rows[0]), x1 - x0, rows[1] - rows[0], fill=False, ec=colour, lw=1.8,
                                    label=f"{name} ({x0:.0f}-{x1:.0f})"))
    regions_2d = ((math.floor(x_e - span - 20) - half, math.floor(x_e - 12)),
                  (math.ceil(x_e + 12), math.ceil(x_e + span + 20) + half + 1))
    for (x0, x1), colour in zip(regions_2d, (PLATEN_COLOUR, BLOCK_COLOUR), strict=True):
        ax_zoom.add_patch(Rectangle((x0, fft_rows[0]), x1 - x0, fft_rows[1] - fft_rows[0], fill=False, ec=colour,
                                    lw=1.0, ls="--"))
    ax_zoom.plot([], [], color="0.4", ls="--", lw=1.0, label="2-D Fourier regions")
    ax_zoom.add_patch(Rectangle((x_e - 100, fft_rows[0]), 60, fft_rows[1] - fft_rows[0], fill=True,
                                fc=FOURIER_COLOUR, alpha=0.25, ec="none", label="columns for the period"))
    ax_zoom.axvline(x_e, color=EDGE_COLOUR, lw=1.2, label="edge")
    ax_zoom.axhline(y_e, color="w", lw=0.8, ls=":", label=f"evaluation row {y_e}")
    if data.get("face_centre_x") is not None:
        ax_zoom.axvline(data["face_centre_x"], color="w", lw=1.0, ls="-.",
                        label=f"face centre x = {data['face_centre_x']:.1f}")
    ax_zoom.set_xlim(*zx)
    ax_zoom.set_ylim(zy[1], zy[0])
    ax_zoom.set_title(f"(b) areas around the edge\nfit window rows {rows[0]}-{rows[1]}")
    ax_zoom.legend(fontsize=7, loc="lower left", framealpha=0.85)

    # (c) band phases of the primary reader
    primary = data["primary"]
    platen_line = side_fit(primary.platen, (x_e - span, x_e - gap))
    block_line = side_fit(primary.block, (x_e + gap, x_e + span))
    ref = np.polyval(platen_line, x_e)
    shift = 2 * np.pi * np.floor((np.polyval(block_line, x_e) - ref) / (2 * np.pi))
    for cols, line, colour, name, offset in ((primary.platen, platen_line, PLATEN_COLOUR, "platen", 0.0),
                                             (primary.block, block_line, BLOCK_COLOUR, "block", shift)):
        good = np.isfinite(cols.phase_unwrapped) & (cols.r2 >= 0.3)
        ax_phase.plot(cols.x[good], (cols.phase_unwrapped[good] - offset - ref) / (2 * np.pi), "o", ms=3,
                      color=colour, label=f"{name} bands")
        ax_phase.plot(cols.x[~good], (cols.phase_unwrapped[~good] - offset - ref) / (2 * np.pi), "x", ms=4,
                      color=colour, alpha=0.5)
        xs = np.array([min(cols.x.min(), x_e), max(cols.x.max(), x_e)])
        ax_phase.plot(xs, (np.polyval(line, xs) - offset - ref) / (2 * np.pi), "-", color=colour, lw=1)
    ax_phase.axvline(x_e, color=EDGE_COLOUR, lw=1.2)
    frac = summary["fraction"]
    ax_phase.set_title(f"(c) {frac['primary_method']} band phases at row {y_e}\n"
                       f"F at the edge = {frac['edge_fraction']:.4f}")
    ax_phase.set_xlabel("column x (px)")
    ax_phase.set_ylabel("phase relative to the platen at the edge (fringes)")
    ax_phase.legend(fontsize=7)
    for ax in (ax_frame, ax_zoom):
        ax.set_xlabel("column x (px)")
        ax.set_ylabel("row y (px)")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="End-to-end gauge-block measurement with a summary report.")
    ap.add_argument("--series", help="folder of one burst of frames, with its <folder>.meta.yaml sidecar")
    ap.add_argument("--nominal-mm", type=float, help="nominal length, mm (sidecar: patron)")
    ap.add_argument("--wavelength-vac", type=float, help="vacuum wavelength, nm (sidecar: longitud_onda_nm)")
    ap.add_argument("--temperature", type=float, help="air temperature, C")
    ap.add_argument("--pressure", type=float, help="air pressure, Pa")
    ap.add_argument("--humidity", type=float, help="relative humidity, %%")
    ap.add_argument("--block-temp", type=float, help="block temperature, C (default: the air temperature)")
    ap.add_argument("--cte", type=float, help="thermal expansion coefficient, 1/K (default: steel)")
    ap.add_argument("--order-sign", type=int, choices=(1, -1), help="direction of the interference order")
    ap.add_argument("--method", choices=METHODS, default="fourier", help="primary fraction reader")
    ap.add_argument("--eval-row", type=int, help="evaluation row (bench default 1050; simulated: centre)")
    ap.add_argument("--edge-x", type=float, help="edge column, px (default: located on the image)")
    ap.add_argument("--edge-search", type=int, nargs=2, metavar=("X0", "X1"), help="columns searched for the edge")
    ap.add_argument("--face-centre-x", type=float, help="column of the face centre, px, to evaluate F there")
    ap.add_argument("--pixel-mm", type=float, help="pixel size on the object, mm (wedge in microradians)")
    ap.add_argument("--prior-dev-nm", type=float, help="prior deviation from nominal for MEFM, nm")
    ap.add_argument("--prior-U-nm", type=float, help="expanded uncertainty of that prior, nm")
    ap.add_argument("--bit-depth", type=int, default=12, help="sensor bit depth, for saturation (default 12)")
    ap.add_argument("--json", metavar="PATH", help="write the summary to PATH")
    ap.add_argument("--plot", metavar="PATH", help="save a figure of the areas used (frame, edge zoom, band phases)")
    sim = ap.add_argument_group("simulated scene (when --series is not given)")
    sim.add_argument("--true-dev-nm", type=float, default=51.0, help="true deviation from nominal, nm")
    sim.add_argument("--scale", type=float, default=0.5, help="frame scale of the 2448 x 2048 sensor")
    sim.add_argument("--frames", type=int, default=5, help="frames with independent noise")
    sim.add_argument("--fringes-y", type=float, default=30.0, help="reference tilt, fringes along y")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    artifacts = {} if args.plot else None
    summary = run(args, artifacts)
    print(format_summary(summary))
    if args.json:
        Path(args.json).write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"summary written to {args.json}")
    if args.plot:
        plot_regions(args.plot, summary, artifacts)
        print(f"figure written to {args.plot}")
    return summary


if __name__ == "__main__":
    main()
