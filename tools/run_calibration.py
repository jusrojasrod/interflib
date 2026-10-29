"""End-to-end gauge-block measurement: interferogram -> fringe fraction -> length at 20 C.

The chain, with the readers and solvers of the library:

1. the fringe fraction ``F`` with the three single-shot readers (sine fit, Fourier by
   columns, Fourier in two dimensions); one of them is the primary and the others are a
   cross-check. ``F`` is read at the centre of the measuring face, where the central length
   is defined (ISO 3650:1998, clause 3), when ``locate_face_centre`` can place it: from the
   whole outline of the face, from part of it and the face dimensions (``--face-mm``), or
   from a centre given by hand (``--face-centre``). The face dimensions may also come from
   the sidecar (``cara_mm``, ``u_cara_mm``). Otherwise it is read at the edge column
   and the evaluation row, with a warning. Its ``u(F)`` combines the scatter about the side
   lines, the repeatability over the frames, how far ``F`` moves with the bands in the side
   fits (``span_sensitivity``) and the uncertainty of the centre through the wedge;
2. the wavelength in air from the vacuum wavelength and the air temperature, pressure
   and humidity (modified Edlen);
3. the integer order from the nominal length and, when a prior deviation with its
   expanded uncertainty is given, also by MEFM;
4. the length reduced to 20 C;
5. the parity of the reading: ``order_sign`` and the platen side are declared, and a
   wrong one reads ``1 - F``. The summary always gives the length the other parity would
   give, and tests both against the prior when there is one (a check standard or a
   pre-measurement, ``--prior-dev-nm`` and ``--prior-U-nm``).

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
import dataclasses
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
from interflib.dimensional.parity import check_fraction_parity  # noqa: E402
from interflib.dimensional.refraction import EdlenRefractiveIndex  # noqa: E402
from interflib.fringe.edge_comparison import (detect_edge_column, edge_fraction, edge_line_centre,  # noqa: E402
                                              side_band_centres, side_fit, span_sensitivity)
from interflib.fringe.fourier import (demodulate_columns, estimate_carrier, fourier_edge_fraction,  # noqa: E402
                                      fourier_edge_fraction_2d)
from interflib.fringe.geometry import locate_block, locate_face_centre  # noqa: E402
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
BAND_REACH_PX = 202          # the readers use columns up to 202 px from the edge (2-D region: 180 + 20 + 2)
ZONE_COVER_WARN = 0.95       # fraction of the band areas that must lie in the fringe zone
FRINGE_ANGLE_WARN = 14.0     # deg: fringe rotation beyond the tested range of the readers
FRAME_MOTION_WARN = 0.25     # fringe: motion of the fringes between frames that blurs the average
ASPECT_WARN = 0.02           # relative mismatch of the measured face aspect against the declared dimensions
EDGE_ORIGINS = {"phase step and dark line": "phase-step line refined to the dark chamfer line",
                "dark line": "dark chamfer line nearest the platen side (no phase-step line along it)",
                "phase step": "phase-step line, not confirmed by a dark chamfer line"}
PRIOR_K = 2.0                # coverage factor of --prior-U-nm (certificates and pre-measurements give U with k = 2)


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


def _pair(value, name):
    """A sidecar pair of numbers (``[a, b]``), or None when the key is absent."""
    if value is None:
        return None
    try:
        a, b = (float(v) for v in value)
    except (TypeError, ValueError):
        raise SystemExit(f"sidecar {name} must be a pair of numbers [width, length] in mm, not {value!r}") from None
    return a, b


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
    side_from_sidecar = {"izquierda": "left", "derecha": "right", "left": "left", "right": "right"}.get(
        str(side.get("lado_platina", "")).strip().lower())
    inputs["platen_side"] = _first((args.platen_side, "command line"), (side_from_sidecar, "sidecar"),
                                   ("left" if sim else None, "simulation setting"),
                                   ("left", "assumed (left, as in the 2026-10 campaign)"))
    inputs["order_sign"] = _first((args.order_sign, "command line"), (side.get("sentido_orden"), "sidecar"),
                                  (1 if sim else None, "simulation (carrier towards +y)"),
                                  (1, "assumed (+1, library default)"))
    # Measuring face of the block: it places the face centre when the field shows only part of the outline.
    inputs["face_mm"] = _first((tuple(args.face_mm) if args.face_mm else None, "command line"),
                               (_pair(side.get("cara_mm"), "cara_mm"), "sidecar"),
                               (tuple(args.block_size_mm) if sim else None, "simulation setting"))
    inputs["u_face_mm"] = _first((tuple(args.u_face_mm) if args.u_face_mm else None, "command line"),
                                 (_pair(side.get("u_cara_mm"), "u_cara_mm"), "sidecar"),
                                 ((0.0, 0.0), "simulation setting" if sim else "assumed exact"))
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
    geometry = SceneGeometry.cs505mu(scale=args.scale, fov_mm=25.0, block_rotation_deg=args.block_rotation_deg,
                                     block_center_mm=tuple(args.block_centre_mm),
                                     block_width_mm=args.block_size_mm[0], block_height_mm=args.block_size_mm[1])
    model = PhaseModel(wavelength_nm=wavelength_air, central_length_nm=true_length * (1 + cte * (t_block - 20.0)),
                       fringes_y=args.fringes_y, fringes_x=1.0)
    sims = [synthesize_interferogram(geometry, model, sensor=SensorModel.cs505mu(), seed=k)
            for k in range(args.frames)]
    mask = sims[0].block_mask
    height, width = mask.shape
    eval_row = args.eval_row if args.eval_row is not None else height // 2
    pixel_mm = 25.0 / (width - 1)
    offset_x, offset_y = args.block_centre_mm
    centre = ((height - 1) / 2 + offset_y / pixel_mm, (width - 1) / 2 + offset_x / pixel_mm)
    return {"source": f"simulated CS505MU scene, scale {args.scale}, {args.frames} frames",
            "frames": [s.image.astype(float) for s in sims], "eval_row": eval_row,
            "pixel_mm": pixel_mm, "mask": mask,
            "truth": {"fraction": sims[0].fraction_truth, "length_nm": true_length, "centre": centre}}


def real_series(args, folder: Path) -> dict:
    """Frames of one burst; the edge is located as in the 2026-10 campaign analysis."""
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff", ".png"))
    if not files:
        raise SystemExit(f"no image files in {folder}")
    frames = [load_image(p).astype(float) for p in files]
    eval_row = args.eval_row if args.eval_row is not None else BENCH_EVAL_ROW
    return {"source": f"series {folder.name}, {len(frames)} frames", "frames": frames, "eval_row": eval_row,
            "pixel_mm": args.pixel_mm, "mask": None, "truth": None}


def normalise_scene(args, scene: dict, inputs: Inputs, notes: list) -> dict:
    """Find the block and normalise every frame: edge vertical, platen on the left.

    :func:`interflib.fringe.geometry.locate_block` on the mean frame; if it fails (too few
    fringes for the demodulation), or with ``--edge-x``/``--edge-search``, the edge is
    located as in the 2026-10 campaign (dark line, platen strip on the left) after
    mirroring the frames when the platen is on the right.

    When the centre of the face is placed (:func:`_face_centre`), the evaluation row moves
    to it and the edge is taken at that row.
    """
    frames, eval_row = scene["frames"], scene["eval_row"]
    platen_side = inputs.value("platen_side")
    mean = np.mean(frames, axis=0)
    width = mean.shape[1]
    mirror = platen_side == "right"
    geo, info = None, {"platen_side": platen_side}
    if args.edge_x is None and not args.edge_search:
        try:
            geo = locate_block(mean, platen_side=platen_side, eval_row=eval_row)
        except ValueError as exc:
            notes.append(f"general block detection failed ({exc}); the edge is located as in the 2026-10 campaign, "
                         "which assumes the bench layout")
    if geo is not None:
        frames = [geo.normalise(f) for f in frames]
        edge_x, origin = geo.edge_x, EDGE_ORIGINS[geo.edge_source]
        to_norm = geo.to_normalised
        info.update(method="general (Fourier phase step)", angle_deg=geo.angle_deg, rotation_deg=geo.rotation_deg,
                    mirrored=geo.mirrored, support=geo.support, edge_x_phase_px=geo.edge_x_phase,
                    edge_source=geo.edge_source, dark_line_depth=geo.dark_line_depth,
                    refined_by_dark_line=geo.refined_by_dark_line, platen_width_px=geo.platen_width_px,
                    face_width_px=geo.face_width_px, face_rows=list(geo.face_rows),
                    fringe_angle_deg=float(np.degrees(np.arctan2(geo.carrier[0], geo.carrier[1]))) - geo.rotation_deg)
        zone = geo.zone
    else:
        frames = [f[:, ::-1] for f in frames] if mirror else list(frames)
        norm_mean = np.mean(frames, axis=0)
        to_norm = (lambda y, x: (y, width - 1 - x)) if mirror else (lambda y, x: (y, x))
        if args.edge_x is not None:
            edge_x, origin = float(to_norm(eval_row, args.edge_x)[1]), "command line"
        else:
            search = tuple(args.edge_search) if args.edge_search else _search_from_platen(norm_mean)[0]
            step = detect_edge_column(norm_mean, rows=BENCH_EDGE_ROWS, search=search)
            edge_x = edge_line_centre(norm_mean, step, rows=BENCH_EDGE_ROWS)
            origin = f"dark edge line, searched in columns {search[0]}-{search[1]}"
            info["edge_search"] = list(search)
        info.update(method="bench heuristic (platen strip on the left)", mirrored=mirror)
        zone = None
    centre = _face_centre(args, geo, to_norm, eval_row, scene.get("pixel_mm"), inputs, notes)
    height = frames[0].shape[0]
    if centre is not None and not HALF_WINDOW <= centre["y"] < height - HALF_WINDOW:
        notes.append(f"the face centre at row {centre['y']:.0f} leaves no room for the fit window; F is read at the "
                     "evaluation row")
        centre = None
    if centre is not None and centre["source"] != "manual column":
        eval_row = int(round(centre["y"]))
        if geo is not None:
            edge_x = geo.edge_x_at(eval_row)
    truth_edge = truth_centre = None
    if scene.get("mask") is not None:
        mask = geo.normalise(scene["mask"].astype(float)) > 0.5 if geo is not None else \
            (scene["mask"][:, ::-1] if mirror else scene["mask"])
        cols = np.flatnonzero(mask[eval_row])
        if cols.size:
            truth_edge = float(cols[0]) - 0.5
        truth_centre = [float(v) for v in to_norm(*scene["truth"]["centre"])]
    return {**scene, "frames": frames, "eval_row": eval_row, "edge_x": float(edge_x), "edge_origin": origin,
            "face_centre": centre, "geometry": info, "zone": zone, "truth_edge_x": truth_edge,
            "truth_centre": truth_centre}


def _face_centre(args, geo, to_norm, eval_row, pixel_mm, inputs, notes):
    """Centre of the measuring face in the normalised frames, as a dict, or None.

    With ``geo`` (the general detection), :func:`interflib.fringe.geometry.locate_face_centre`
    places it from the outline, from the outline and the face dimensions (``--face-mm`` or
    the sidecar), or from ``--face-centre``.
    With the bench heuristic only a centre given by hand is possible. ``--face-centre-x``
    gives a column only: ``F`` is then extended to it at the evaluation row, as before.
    """
    nan = float("nan")
    u_manual = args.u_face_centre_px
    if args.face_centre is None and args.face_centre_x is not None:
        y, x = to_norm(eval_row, args.face_centre_x)
        return {"y": float(y), "x": float(x), "u_y": nan, "u_x": nan if u_manual is None else float(u_manual),
                "source": "manual column", "scale_px_per_mm": nan, "scale_source": "none", "features": {},
                "aspect": (nan, nan), "notes": ()}
    manual = tuple(args.face_centre) if args.face_centre is not None else None
    if geo is None:
        if manual is None:
            return None
        y, x = to_norm(*manual)
        u = nan if u_manual is None else float(u_manual)
        return {"y": float(y), "x": float(x), "u_y": u, "u_x": u, "source": "manual", "scale_px_per_mm": nan,
                "scale_source": "none", "features": {}, "aspect": (nan, nan), "notes": ()}
    try:
        face = locate_face_centre(geo, face_mm=inputs.value("face_mm"), u_face_mm=inputs.value("u_face_mm"),
                                  pixel_mm=pixel_mm, manual=manual, u_manual_px=u_manual)
    except ValueError as exc:
        notes.append(f"face centre not located: {exc}")
        return None
    return {**dataclasses.asdict(face), "face_mm": inputs.value("face_mm")}


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
    geometry = scene.get("geometry", {})
    face_rows = geometry.get("face_rows")
    if face_rows:
        clipped = (max(fft_rows[0], face_rows[0]), min(fft_rows[1], face_rows[1]))
        if clipped != fft_rows and clipped[1] - clipped[0] >= rows[1] - rows[0]:
            notes.append(f"Fourier rows clipped from {fft_rows[0]}-{fft_rows[1]} to the face, {clipped[0]}-{clipped[1]}")
            fft_rows = clipped
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
    per_frame, platen_phase = [], []
    if len(frames) > 1:
        for frame in frames:
            try:
                reading = _read_fraction(args.method, frame, edge_x, rows, eval_row, period, fft_rows, order_sign)
            except (ValueError, RuntimeError):
                continue
            per_frame.append(reading.fraction)
            platen_phase.append(reading.phase_platen / (2 * np.pi))
    motion = float(np.ptp(_wrap(platen_phase, platen_phase[0]))) if len(platen_phase) > 1 else 0.0
    per_frame = _wrap(per_frame, reference) if per_frame else np.array([])
    s_frame = float(np.std(per_frame, ddof=1)) if per_frame.size > 1 else float("nan")
    u_mean = s_frame / math.sqrt(per_frame.size) if per_frame.size > 1 else float("nan")
    # --- F at the face centre: both side lines extended to x_c (exact for straight lines), at
    # the row of the centre (the evaluation row moved there)
    centre = scene.get("face_centre")
    x_c = centre["x"] if centre else None
    g_x = order_sign * (primary.slope_block - primary.slope_platen)          # fringe/px
    if x_c is not None:
        shift = g_x * (x_c - edge_x)
        fraction_used = float((primary.fraction + shift) % 1.0)
        where = "face centre column" if centre["source"] == "manual column" else "face centre"
    else:
        shift, fraction_used, where = None, primary.fraction, "edge column"

    # --- sensitivity to the bands in the side fits, where F is used: a model error of the lines
    choice = span_sensitivity(primary.platen, primary.block, edge_x, eval_row, at_x=x_c, order_sign=order_sign)
    u_extrap, u_bands = primary.fraction_error, choice.model_error

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

    # --- position of the centre: F changes across the face with the wedge
    u_centre = float("nan")
    if centre:
        g_y = wedge.wedge_y if wedge else float("nan")
        u_centre = float(math.hypot(g_x * centre["u_x"], g_y * centre["u_y"]))
    parts = [u for u in (u_extrap, u_mean, u_bands, u_centre) if np.isfinite(u)]
    u_fraction = float(math.sqrt(sum(u * u for u in parts))) if parts else float("nan")

    # --- geometry checks: band areas inside the fringe zone, fit window on the face
    zone = scene.get("zone")
    if zone is not None:
        for side, (x0, x1) in (("platen", (xe - BAND_REACH_PX, xe - 28)), ("block", (xe + 28, xe + BAND_REACH_PX))):
            area = zone[rows[0]:rows[1], max(0, x0):min(width, x1)]
            geometry[f"{side}_bands_in_zone"] = float(area.mean()) if area.size else 0.0
    if face_rows:
        geometry["fit_window_on_face"] = bool(face_rows[0] <= rows[0] and rows[1] <= face_rows[1])
    if artifacts is not None:
        artifacts.update(image=img, primary=primary, rows=rows, fft_rows=fft_rows, face_centre=centre,
                         edge_search=geometry.get("edge_search"), face_rows=face_rows, geometry=geometry)
    return {
        "image": {"source": scene["source"], "size_px": [height, width], "frames": len(frames),
                  "edge_x_px": edge_x, "edge_origin": scene["edge_origin"], "eval_row_px": eval_row,
                  "geometry": geometry, "face_centre": centre, "fringe_motion_between_frames": motion,
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
                     "u_mean_of_frames": u_mean, "u_extrapolation": u_extrap,
                     "band_variants": {f"{gap}-{span}": float(f) for (gap, span), f in zip(choice.variants, choice.fraction,
                                                                                          strict=True)},
                     "band_deviation": choice.deviation, "u_band_choice": u_bands,
                     "u_centre_position": u_centre, "u_fraction": u_fraction,
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
    scene = normalise_scene(args, scene, inputs, notes)
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

    # --- parity of the reading: F against 1 - F, tested against the prior when there is one
    has_prior = args.prior_dev_nm is not None and args.prior_U_nm is not None
    prior_dev = args.prior_dev_nm if has_prior else 0.0
    prior_u = args.prior_U_nm / PRIOR_K if has_prior else None
    u_optical = math.hypot(u_length_f if np.isfinite(u_length_f) else 0.0, args.u_other_nm or 0.0)
    check = check_fraction_parity(half, solution.length_dev, frac["fraction_used"], prior_length_dev=prior_dev,
                                  prior_u=prior_u, u_length=u_optical)
    parity = {"order_sign": frac["order_sign"], "platen_side": inputs.value("platen_side"),
              "verdict": check.verdict, "fraction": check.fraction, "flipped_fraction": check.flipped_fraction,
              "length_dev_nm": check.length_dev, "flipped_length_dev_nm": check.flipped_length_dev,
              "shift_nm": check.shift, "separation_nm": check.separation,
              "prior_dev_nm": prior_dev, "prior_origin": "--prior-dev-nm" if has_prior else "nominal",
              "prior_u_nm": prior_u, "u_optical_nm": u_optical,
              "z": check.z, "flipped_z": check.flipped_z, "log10_likelihood_ratio": check.log10_likelihood_ratio}

    summary = {
        "inputs": {name: {"value": value, "origin": origin} for name, (value, origin) in inputs.items()},
        "air": {"n_minus_1": n_air - 1, "wavelength_air_nm": wavelength_air, "half_wavelength_nm": half},
        **measured,
        "order": order,
        "parity": parity,
        "length": {"length_20c_nm": solution.length, "length_20c_mm": solution.length / 1e6,
                   "deviation_from_nominal_nm": solution.length_dev, "u_from_fraction_nm": u_length_f},
    }
    if scene["truth"]:
        truth = scene["truth"]
        summary["truth"] = {"fraction": truth["fraction"], "length_20c_nm": truth["length_nm"],
                            "fraction_error": float(_wrap(frac["fraction_used"], truth["fraction"]) - truth["fraction"]),
                            "length_error_nm": solution.length - truth["length_nm"],
                            "edge_x_px": scene.get("truth_edge_x"), "centre_yx_px": scene.get("truth_centre")}
        centre = measured["image"]["face_centre"]
        if centre and scene.get("truth_centre"):
            summary["truth"]["centre_error_yx_px"] = [centre["y"] - scene["truth_centre"][0],
                                                      centre["x"] - scene["truth_centre"][1]]
    summary["warnings"] = notes + _warnings(summary, inputs)
    return summary


def _parity_warning(parity: dict, platen_assumed: bool) -> str | None:
    z, zf = parity["z"], parity["flipped_z"]
    other = f"with the other parity F = {parity['flipped_fraction']:.4f} and d = {parity['flipped_length_dev_nm']:+.1f} nm"
    verdict = parity["verdict"]
    if verdict == "unverified":
        text = (f"the parity of the reading (order_sign and platen side) is not verified: {other} "
                f"({parity['shift_nm']:+.1f} nm); give --prior-dev-nm and --prior-U-nm (a check standard or a "
                "pre-measurement) to test it")
        return text + ("; the --plot figure shows which side was taken as platen" if platen_assumed else "")
    prior = f"the prior d = {parity['prior_dev_nm']:+.1f} nm (u = {parity['prior_u_nm']:.1f} nm)"
    if verdict == "contradicted":
        return (f"WRONG PARITY: {prior} agrees with the flipped reading (z = {zf:+.1f}) and not with the declared "
                f"one (z = {z:+.1f}): order_sign or the platen side is wrong; {other}")
    if verdict == "undecided":
        return (f"{prior} cannot tell the two parities apart (z = {z:+.1f} declared, {zf:+.1f} flipped): a wrong "
                f"parity would move d by {parity['shift_nm']:+.1f} nm")
    if verdict == "inconsistent":
        return (f"{prior} agrees with neither parity (z = {z:+.1f} declared, {zf:+.1f} flipped): check lambda_0, "
                "the pressure, the prior and the point where F is evaluated")
    return None


def _warnings(summary, inputs: Inputs) -> list:
    out = []
    parity_note = _parity_warning(summary["parity"], inputs["platen_side"][1].startswith("assumed"))
    if parity_note:
        out.append(parity_note)
    for name in ("pressure_pa", "block_temperature_c", "order_sign", "platen_side"):
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
    geo = img.get("geometry", {})
    if geo.get("edge_source") == "phase step":
        out.append("the edge is a line of phase steps with no dark chamfer line along it: it is unconfirmed and may "
                   "be a stray line (on the 2026-10 bench frames with 5-20 fringes such lines lay up to 150 px "
                   "from the edge); check it with --plot or give --edge-x")
    elif geo.get("edge_source") == "dark line":
        out.append("no phase-step line ran along a dark chamfer line; the edge is the dark line nearest the platen "
                   "side, which assumes a near-vertical edge")
    if geo.get("rotation_deg"):
        out.append(f"the edge was {geo['angle_deg']:+.2f} deg from vertical; the frames were rotated to read F "
                   "(interpolation)")
    if abs(geo.get("fringe_angle_deg", 0.0)) > FRINGE_ANGLE_WARN:
        out.append(f"fringes rotated {geo['fringe_angle_deg']:+.1f} deg from the rows, beyond the {FRINGE_ANGLE_WARN:g} deg "
                   "tested with the readers")
    for side in ("platen", "face"):
        width = geo.get(f"{side}_width_px")
        if width is not None and width < BAND_REACH_PX:
            out.append(f"the {side} is {width:.0f} px wide at the evaluation row, less than the {BAND_REACH_PX} px the "
                       "readers use: bands fall off it")
    for side in ("platen", "block"):
        cover = geo.get(f"{side}_bands_in_zone")
        if cover is not None and cover < ZONE_COVER_WARN:
            out.append(f"only {100 * cover:.0f} % of the {side} band area has fringes (fringe zone)")
    if geo.get("fit_window_on_face") is False:
        out.append(f"the fit window {img['fit_rows']} is not inside the rows where the face has fringes "
                   f"{geo['face_rows']}: move --eval-row")
    if img.get("fringe_motion_between_frames", 0.0) > FRAME_MOTION_WARN:
        out.append(f"the fringes moved {img['fringe_motion_between_frames']:.2f} fringe between frames: the averaged "
                   "image loses contrast")
    if img["saturated_fraction"] > SATURATION_WARN:
        out.append(f"{100 * img['saturated_fraction']:.1f} % of the pixels in the analysis region are saturated")
    if img["median_band_r2_sine"] is not None and img["median_band_r2_sine"] < R2_WARN:
        out.append(f"median R^2 of the band fits is {img['median_band_r2_sine']:.2f} (< {R2_WARN})")
    u = frac["u_fraction"]
    if np.isfinite(u) and u > 0 and frac["spread_between_methods"] > SIGMA_WARN * u:
        out.append(f"the methods disagree by {frac['spread_between_methods']:.4f} fringe, more than "
                   f"{SIGMA_WARN:g} u(F) = {SIGMA_WARN * u:.4f}")
    noise = next((u for u in (frac["u_extrapolation"], frac["u_mean_of_frames"]) if np.isfinite(u)), float("nan"))
    if np.isfinite(noise) and frac["band_deviation"] > SIGMA_WARN * noise:
        half = summary["air"]["half_wavelength_nm"]
        out.append(f"F moves by up to {frac['band_deviation']:.4f} fringe ({frac['band_deviation'] * half:.1f} nm) "
                   "when the bands nearest the edge are left out or the span is shortened, more than "
                   f"{SIGMA_WARN:g} times the scatter about the side lines: the side phases are not straight up to "
                   f"the edge; u(F) includes it as u_m(F) = {frac['u_band_choice']:.4f}")
    if summary["order"]["margin_to_quarter_wave_nm"] < SIGMA_WARN * max(length["u_from_fraction_nm"], 1.0):
        out.append("the deviation from nominal is close to a quarter wave: the integer order from the nominal "
                   "prior is not reliable")
    out.append("the nominal prior gives the right order only if the true deviation is within a quarter wave "
               f"({summary['order']['quarter_wave_nm']:.0f} nm)")
    centre = img.get("face_centre")
    if frac["evaluated_at"] == "edge column":
        out.append("F is evaluated at the edge column, not at the face centre where the central length is "
                   "defined (the block-platen wedge makes the two differ): the outline of the face is incomplete; "
                   "give the face dimensions (--face-mm) or the centre (--face-centre)")
    else:
        out.append("F at the face centre extends the side lines beyond the measured bands; u(F) includes how the "
                   "choice of bands moves it there, but not the scatter of that extension")
    if centre:
        if centre["source"] == "outline and dimensions":
            width, length = centre["face_mm"]
            origin = inputs["face_mm"][1]
            out.append(f"the face centre rests on the declared face dimensions {width:g} x {length:g} mm ({origin}); "
                       "with dimensions not measured on the block it is only as good as they are")
            if inputs["u_face_mm"][1] == "assumed exact":
                out.append("the face dimensions have no uncertainty (u_cara_mm in the sidecar or --u-face-mm): "
                           "u(F) leaves it out")
        elif centre["source"] == "manual column":
            out.append("--face-centre-x moves F to that column at the evaluation row, not to the row of the face "
                       "centre; give --face-centre Y X or the face dimensions")
        out += [f"face centre: {note}" for note in centre["notes"]]
        if not np.isfinite(frac["u_centre_position"]):
            out.append("the position of the face centre has no uncertainty (give --u-face-centre-px for a manual "
                       "centre): u(F) leaves it out")
        measured_aspect, expected_aspect = centre["aspect"]
        if np.isfinite(measured_aspect) and abs(measured_aspect / expected_aspect - 1) > ASPECT_WARN:
            out.append(f"the outline gives a length-to-width ratio of {measured_aspect:.3f} against "
                       f"{expected_aspect:.3f} from the declared dimensions: a feature or a dimension is wrong")
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
             "humidity_pct": "%", "block_temperature_c": "C", "cte_per_k": "1/K", "platen_side": "", "order_sign": "",
             "face_mm": "mm", "u_face_mm": "mm"}
    for name, item in s["inputs"].items():
        value = item["value"]
        shown = " x ".join(f"{v:g}" for v in value) if isinstance(value, tuple) else _fmt(value, "g")
        out.append(f"  {name:<22}{shown:>12} {units[name]:<4} [{item['origin']}]")
    air = s["air"]
    out.append(f"\nAir            n - 1 = {air['n_minus_1']:.4e}   lambda_air = {air['wavelength_air_nm']:.5f} nm")
    img = s["image"]
    out.append("\nImage")
    out.append(f"  source        {img['source']} ({img['size_px'][0]} x {img['size_px'][1]} px)")
    geo = img.get("geometry", {})
    out.append(f"  detection     {geo.get('method', '-')}; platen on the {geo.get('platen_side', '-')}"
               + ("; image mirrored" if geo.get("mirrored") else ""))
    if "angle_deg" in geo:
        found = (f"phase step in {100 * geo['support']:.0f} % of the row bands" if np.isfinite(geo["support"])
                 else "no phase-step line")
        out.append(f"  edge line     angle {geo['angle_deg']:+.2f} deg (rotation applied {geo['rotation_deg']:+.2f}); "
                   f"{found}; dark line depth {geo['dark_line_depth']:.2f}; fringes at {geo['fringe_angle_deg']:+.1f} deg")
        out.append(f"  widths        platen {geo['platen_width_px']:.0f} px, face {geo['face_width_px']:.0f} px; "
                   f"face with fringes in rows {geo['face_rows'][0]}-{geo['face_rows'][1]}")
    out.append(f"  edge          x = {img['edge_x_px']:.2f} px ({img['edge_origin']}); evaluation row {img['eval_row_px']}")
    centre = img.get("face_centre")
    if centre:
        scale = (f"; scale {centre['scale_px_per_mm']:.2f} px/mm ({centre['scale_source']})"
                 if np.isfinite(centre["scale_px_per_mm"]) else "")
        out.append(f"  face centre   y = {centre['y']:.1f}, x = {centre['x']:.1f} px (u {_fmt(centre['u_y'], '.1f')}, "
                   f"{_fmt(centre['u_x'], '.1f')} px) from the {centre['source']}{scale}")
        if centre["features"]:
            found = ", ".join(f"{name} {value:.1f} ({origin})" for name, (value, origin) in centre["features"].items())
            out.append(f"  outline       {found}")
    out.append(f"  fringes       period {img['period_px']:.2f} px; {img['fringes_in_fit_window']:.1f} fringes "
               f"in the fit window {img['fit_rows']}")
    out.append(f"  visibility    platen {img['visibility_platen']:.2f}, block {img['visibility_block']:.2f}; "
               f"saturated {100 * img['saturated_fraction']:.2f} %; median band R^2 {_fmt(img['median_band_r2_sine'], '.3f')}")
    wedge = (f"  wedge         dg_x = {_fmt(img['wedge_x_fringes_per_px'], '.2e')}, "
             f"dg_y = {_fmt(img['wedge_y_fringes_per_px'], '.2e')} fringe/px")
    if img["wedge_x_urad"] is not None:
        wedge += f"  ({_fmt(img['wedge_x_urad'], '.1f')}, {_fmt(img['wedge_y_urad'], '.1f')} urad)"
    out.append(wedge)
    if img.get("fringe_motion_between_frames"):
        out.append(f"  frames        fringe motion between frames {img['fringe_motion_between_frames']:.3f} fringe")
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
    variants = ", ".join(f"{name} {f % 1.0:.4f}" for name, f in frac["band_variants"].items())
    out.append(f"  band choice   gap-span px: {variants}; u_m(F) = {frac['u_band_choice']:.4f} "
               f"({frac['u_band_choice'] * s['air']['half_wavelength_nm']:.1f} nm)")
    if centre:
        out.append(f"  centre        u_c(F) = {_fmt(frac['u_centre_position'])} "
                   f"({_fmt(frac['u_centre_position'] * s['air']['half_wavelength_nm'], '.1f')} nm) from its position")
    out.append(f"  F used = {frac['fraction_used']:.4f} at the {frac['evaluated_at']}, u(F) = {_fmt(frac['u_fraction'])}")
    order = s["order"]
    out.append(f"\nInteger order  kappa = {order['integer_order']}, nominal fraction F' = {order['nominal_fraction']:.4f}, "
               f"margin to lambda/4 = {order['margin_to_quarter_wave_nm']:.1f} nm")
    if "mefm_length_dev_nm" in order:
        lo, hi = order["mefm_acceptance_interval_nm"]
        out.append(f"  MEFM          d = {order['mefm_length_dev_nm']:+.1f} nm (prior interval {lo:+.1f} to {hi:+.1f} nm)")
    par = s["parity"]
    out.append(f"\nParity (order_sign {par['order_sign']:+d}, platen on the {par['platen_side']}): {par['verdict']}")
    out.append(f"  declared      F = {par['fraction']:.4f}   d = {par['length_dev_nm']:+.1f} nm")
    out.append(f"  flipped       F = {par['flipped_fraction']:.4f}   d = {par['flipped_length_dev_nm']:+.1f} nm   "
               f"(shift {par['shift_nm']:+.1f} nm; the combs are {par['separation_nm']:.1f} nm apart at least)")
    if par["prior_u_nm"] is not None:
        out.append(f"  prior         d = {par['prior_dev_nm']:+.1f} nm, u = {par['prior_u_nm']:.1f} nm (U / {PRIOR_K:g}), "
                   f"u optical = {par['u_optical_nm']:.1f} nm;  z = {par['z']:+.2f} declared, {par['flipped_z']:+.2f} "
                   f"flipped (log10 likelihood ratio {par['log10_likelihood_ratio']:+.1f})")
    if "truth" in s:
        t = s["truth"]
        out.append(f"\nTruth          F = {t['fraction']:.4f}, L = {t['length_20c_nm'] / 1e6:.7f} mm;   "
                   f"error F {t['fraction_error']:+.4f}, L {t['length_error_nm']:+.2f} nm")
        if t.get("edge_x_px") is not None:
            out.append(f"               edge from the mask {t['edge_x_px']:.2f} px (found {img['edge_x_px']:.2f} px)")
        if t.get("centre_error_yx_px") is not None:
            dy, dx = t["centre_error_yx_px"]
            out.append(f"               face centre {t['centre_yx_px'][0]:.1f}, {t['centre_yx_px'][1]:.1f} px; "
                       f"error {dy:+.2f}, {dx:+.2f} px")
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
    if data.get("face_rows"):
        for y_face in data["face_rows"]:
            ax_frame.axhline(y_face, color=BLOCK_COLOUR, ls=":", lw=1.0)
        ax_frame.plot([], [], color=BLOCK_COLOUR, ls=":", lw=1.0, label="face rows with fringes")
    ax_frame.axvline(x_e, color=EDGE_COLOUR, lw=1.2, label=f"edge x = {x_e:.1f}")
    zx = (max(0, int(x_e) - 270), min(width, int(x_e) + 270))
    zy = (max(0, y_e - 300), min(height, y_e + 300))
    ax_frame.add_patch(Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], fill=False, ec="w", lw=1.2, ls="--"))
    geo = data.get("geometry", {})
    note = []
    if geo.get("rotation_deg"):
        note.append(f"rotated {geo['rotation_deg']:+.2f} deg")
    if geo.get("mirrored"):
        note.append("mirrored")
    ax_frame.set_title("(a) normalised frame" + (f" ({', '.join(note)})" if note else "") + "\nwhite box: zoom in (b)")
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
    centre = data.get("face_centre")
    if centre is not None:
        ax_zoom.axvline(centre["x"], color="w", lw=1.0, ls="-.", label=f"face centre x = {centre['x']:.1f}")
        ax_frame.plot(centre["x"], centre["y"], "*", ms=12, mfc=BLOCK_COLOUR, mec="w",
                      label=f"face centre ({centre['source']})")
        ax_frame.legend(fontsize=7, loc="lower left", framealpha=0.85)
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
    ap.add_argument("--platen-side", choices=("left", "right"),
                    help="side of the block edge where the platen is (sidecar: lado_platina; default left)")
    ap.add_argument("--method", choices=METHODS, default="fourier", help="primary fraction reader")
    ap.add_argument("--eval-row", type=int, help="evaluation row (bench default 1050; simulated: centre)")
    ap.add_argument("--edge-x", type=float, help="edge column, px (default: located on the image)")
    ap.add_argument("--edge-search", type=int, nargs=2, metavar=("X0", "X1"), help="columns searched for the edge")
    ap.add_argument("--face-mm", type=float, nargs=2, metavar=("WIDTH", "LENGTH"),
                    help="measuring face, mm: width across the edge and length along it; with part of the outline "
                         "it places the face centre (sidecar: cara_mm)")
    ap.add_argument("--u-face-mm", type=float, nargs=2, metavar=("U_WIDTH", "U_LENGTH"),
                    help="standard uncertainties of the face dimensions, mm (sidecar: u_cara_mm; default 0)")
    ap.add_argument("--face-centre", type=float, nargs=2, metavar=("Y", "X"),
                    help="face centre given by hand, input image px (overrides the detection)")
    ap.add_argument("--u-face-centre-px", type=float, help="standard uncertainty of --face-centre on each axis, px")
    ap.add_argument("--face-centre-x", type=float,
                    help="column of the face centre, px: F extended to it at the evaluation row (prefer --face-centre)")
    ap.add_argument("--pixel-mm", type=float, help="pixel size on the object, mm (wedge in microradians)")
    ap.add_argument("--prior-dev-nm", type=float, help="prior deviation from nominal for MEFM, nm")
    ap.add_argument("--prior-U-nm", type=float,
                    help="expanded uncertainty of that prior (k = 2), nm; with --prior-dev-nm it also tests the parity")
    ap.add_argument("--u-other-nm", type=float,
                    help="standard uncertainty of d from all but F (air, lambda_0, temperature), nm, for the parity test")
    ap.add_argument("--bit-depth", type=int, default=12, help="sensor bit depth, for saturation (default 12)")
    ap.add_argument("--json", metavar="PATH", help="write the summary to PATH")
    ap.add_argument("--plot", metavar="PATH", help="save a figure of the areas used (frame, edge zoom, band phases)")
    sim = ap.add_argument_group("simulated scene (when --series is not given)")
    sim.add_argument("--true-dev-nm", type=float, default=51.0, help="true deviation from nominal, nm")
    sim.add_argument("--scale", type=float, default=0.5, help="frame scale of the 2448 x 2048 sensor")
    sim.add_argument("--frames", type=int, default=5, help="frames with independent noise")
    sim.add_argument("--fringes-y", type=float, default=30.0, help="reference tilt, fringes along y")
    sim.add_argument("--block-rotation-deg", type=float, default=0.0, help="in-plane rotation of the block, deg")
    sim.add_argument("--block-centre-mm", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"),
                     help="offset of the block from the frame centre, mm")
    sim.add_argument("--block-size-mm", type=float, nargs=2, default=(9.0, 35.0), metavar=("WIDTH", "LENGTH"),
                     help="measuring face of the simulated block, mm (default 9 x 35: its ends are out of the field)")
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
