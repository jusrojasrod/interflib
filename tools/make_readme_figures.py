"""Regenerate the figures of the README and of the documentation in ``docs/images/``.

Every figure comes from the layered simulator or from published reference values, so it
can be rebuilt anywhere and shows a known truth. Run from the repository root:
``python tools/make_readme_figures.py``.
"""
import importlib.util
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from interflib.dimensional.coincidence import CoincidenceEFSolver  # noqa: E402
from interflib.dimensional.mefm import MonochromaticEFMSolver  # noqa: E402
from interflib.dimensional.parity import check_fraction_parity  # noqa: E402
from interflib.dimensional.refraction import EdlenRefractiveIndex  # noqa: E402
from interflib.fringe.diagnostics import field_overview  # noqa: E402
from interflib.fringe.edge_comparison import (column_phases, detect_edge_column, edge_fraction,  # noqa: E402
                                              side_band_centres, side_fit)
from interflib.fringe.fourier import _bands, demodulate_2d, estimate_carrier, unwrap_phase_map  # noqa: E402
from interflib.fringe.phase_shifting import edge_fraction_from_phase, hariharan_five_step  # noqa: E402
from interflib.simulation import PhaseModel, SceneGeometry, SensorModel, synthesize_interferogram  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
PLATEN, BLOCK, EDGE = "#1f5482", "#c2141b", "#e0a100"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 9, "axes.titlesize": 9,
                     "savefig.bbox": "tight"})

# The README scene: 2 mm + 51 nm block at 543.5 nm, CS505MU noise, half-resolution frame
GEOMETRY = SceneGeometry.cs505mu(scale=0.5, fov_mm=25.0)
MODEL = PhaseModel(wavelength_nm=543.5, central_length_nm=2_000_051.0, fringes_y=30.0, fringes_x=1.0)
SENSOR = SensorModel.cs505mu()


def scene(**phase):
    return synthesize_interferogram(GEOMETRY, replace(MODEL, **phase), sensor=SENSOR, seed=0)


def fig_interferogram(img, edge_x, rows, gap=30, span=180):
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.imshow(img, cmap="gray")
    ax.axvline(edge_x, color=EDGE, lw=1.2, ls="--", label="edge (platen | block)")
    for x0, colour, name in ((edge_x - span, PLATEN, "platen bands"), (edge_x + gap, BLOCK, "block bands")):
        ax.add_patch(plt.Rectangle((x0, rows[0]), span - gap, rows[1] - rows[0], fill=False, ec=colour, lw=1.5,
                                   label=name))
    ax.axhline((rows[0] + rows[1]) / 2, color="w", lw=0.6, ls=":")
    ax.set_title("Synthetic CS505MU frame: 9 mm gauge block wrung to a platen, ~30 fringes")
    ax.set_xlabel("column x (px)")
    ax.set_ylabel("row y (px)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    fig.savefig(OUT / "interferogram.png")
    plt.close(fig)


def fig_edge_comparison(img, edge_x, rows, eval_row, period, gap=30, span=180):
    xs_p, xs_b = side_band_centres(edge_x, gap, span, 5)
    platen = column_phases(img, xs_p, rows, eval_row, period)
    block = column_phases(img, xs_b, rows, eval_row, period)
    cp = side_fit(platen, (edge_x - span, edge_x - gap))
    cb = side_fit(block, (edge_x + gap, edge_x + span))
    ref = np.polyval(cp, edge_x)
    shift = np.floor((np.polyval(cb, edge_x) - ref) / (2 * np.pi)) * 2 * np.pi   # put the block 0..1 fringe above
    f = lambda v: (v - ref) / (2 * np.pi)  # noqa: E731
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(platen.x, f(platen.phase_unwrapped), "s", ms=3, color=PLATEN, label="platen bands")
    ax.plot(block.x, f(block.phase_unwrapped - shift), "o", ms=3, color=BLOCK, label="block bands")
    xx = np.array([edge_x - span, edge_x])
    ax.plot(xx, f(np.polyval(cp, xx)), "-", color=PLATEN, lw=1)
    xx = np.array([edge_x, edge_x + span])
    ax.plot(xx, f(np.polyval(cb, xx) - shift), "-", color=BLOCK, lw=1)
    top = f(np.polyval(cb, edge_x) - shift)
    ax.axvline(edge_x, color=EDGE, lw=1, ls="--")
    ax.annotate("", xy=(edge_x + 4, top), xytext=(edge_x + 4, 0), arrowprops=dict(arrowstyle="<->", lw=1.2))
    ax.text(edge_x + 10, top / 2, f"F = {top:.4f}", va="center")
    ax.set_xlabel("column x (px)")
    ax.set_ylabel("phase at the evaluation row (fringes)")
    ax.set_title("Edge comparison: each side extrapolated to the edge column, where common terms cancel")
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(OUT / "edge_comparison.png")
    plt.close(fig)


def fig_fourier(img, edge_x, height):
    xs = np.arange(int(edge_x) - 180, int(edge_x) - 29, 5)
    profiles = _bands(img, xs, (100, height - 100), 2)
    n = profiles.shape[0]
    centred = (profiles - profiles.mean(axis=0)) * np.hanning(n)[:, None]
    power = (np.abs(np.fft.rfft(centred, n=4 * n, axis=0)) ** 2).mean(axis=1)
    freqs = np.fft.rfftfreq(4 * n)
    carrier = estimate_carrier(profiles)
    curved = scene(block_flatness_pv_nm=600.0).image.astype(float)
    x0 = int(edge_x) + 20
    pmap = demodulate_2d(curved, (100, height - 100), (x0, x0 + 400))
    relief = unwrap_phase_map(pmap).object_phase / (2 * np.pi)
    yy, xx = np.indices(relief.shape)
    plane = np.linalg.lstsq(np.column_stack([np.ones(relief.size), xx.ravel(), yy.ravel()]), relief.ravel(),
                            rcond=None)[0]
    relief = relief - (plane[0] + plane[1] * xx + plane[2] * yy)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    ax.semilogy(freqs, power / power.max(), color="k", lw=0.8)
    ax.axvspan(0.5 * carrier, 1.5 * carrier, color=EDGE, alpha=0.25, label="band-pass (bandwidth 0.5)")
    ax.axvline(carrier, color=EDGE, lw=1)
    ax.set_xlim(0, 4 * carrier)
    ax.set_ylim(1e-6, 2)
    ax.set_xlabel("spatial frequency along y (fringes/px)")
    ax.set_ylabel("power (normalised)")
    ax.set_title("(a) column spectrum: carrier lobe and filter")
    ax.legend(fontsize=7)
    h = axes[1].imshow(np.angle(np.exp(1j * pmap.object_phase)), cmap="twilight", aspect="auto")
    axes[1].set_title("(b) carrier-free phase, wrapped (rad)")
    fig.colorbar(h, ax=axes[1], fraction=0.046, pad=0.04)
    h = axes[2].imshow(relief, cmap="viridis", aspect="auto")
    axes[2].set_title("(c) unwrapped relief of a curved face (fringes)")
    fig.colorbar(h, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes[1:]:
        ax.set_xlabel("column in region (px)")
    axes[1].set_ylabel("row in region (px)")
    fig.tight_layout()
    fig.savefig(OUT / "fourier.png")
    plt.close(fig)


def fig_phase_stepping(edge_x, rows, eval_row):
    frames = [synthesize_interferogram(GEOMETRY, replace(MODEL, phase_shift_rad=k * np.pi / 2), sensor=SENSOR,
                                       seed=k).image.astype(float) for k in range(5)]
    psi = hariharan_five_step(frames)
    res = edge_fraction_from_phase(psi.phase, edge_x, rows, eval_row, modulation=psi.modulation)
    crop = (slice(380, 640), slice(int(edge_x) - 160, int(edge_x) + 160))
    fig = plt.figure(figsize=(10, 4.6))
    grid = fig.add_gridspec(2, 5, height_ratios=[1, 1.6], hspace=0.35)
    for k in range(5):
        ax = fig.add_subplot(grid[0, k])
        ax.imshow(frames[k][crop], cmap="gray")
        ax.set_title(f"frame {k}: δ = {k}π/2" if k else "frame 0: δ = 0")
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(grid[1, 1:4])
    h = ax.imshow(psi.phase[crop], cmap="twilight", aspect="auto",
                  extent=(crop[1].start, crop[1].stop, crop[0].stop, crop[0].start))
    ax.axvline(edge_x, color=EDGE, lw=1, ls="--")
    ax.set_xlabel("column x (px)")
    ax.set_ylabel("row y (px)")
    ax.set_title(f"wrapped phase φ (rad); the step at the edge is 2πF, F = {res.fraction:.4f}")
    fig.colorbar(h, ax=ax, fraction=0.03, pad=0.02)
    fig.savefig(OUT / "phase_stepping.png")
    plt.close(fig)


def fig_exact_fractions():
    wavelengths = [667.8184, 587.5649, 501.5702, 447.1501]
    sol = CoincidenceEFSolver(wavelengths).solve(1_400_000.0, 1_000.0, [0.28, 0.85, 0.90, 0.23])
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    colours = ["#c2141b", "#e0a100", "#2a9d3c", "#1f5482"]
    for i, cands in enumerate(sol.candidate_lengths_per_colour):
        ax.plot((np.asarray(cands) - 1_400_000.0) / 1000, np.full(len(cands), i), "|", ms=16, mew=2, color=colours[i])
    ax.axvline((sol.length - 1_400_000.0) / 1000, color="k", lw=1, ls="--",
               label=f"coincidence: L = {sol.length / 1000:.4f} µm")
    ax.set_yticks(range(4), [f"λ = {lam:.1f} nm" for lam in wavelengths])
    ax.set_ylim(-0.7, 3.7)
    ax.set_xlabel("L − L' (µm), L' = 1400 µm")
    ax.set_title("Exact fractions: each wavelength allows a comb of lengths; only one coincides")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.savefig(OUT / "exact_fractions.png")
    plt.close(fig)


def fig_simulation_layers(sim):
    crop = (slice(470, 530), slice(560, 640))                     # a small patch, to see the sensor noise
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.2))
    panels = [(sim.opd_nm, "viridis", "(a) layer 1: optical path difference (nm)"),
              (sim.phase, "twilight", "(b) layer 1: wrapped phase (rad)"),
              (sim.irradiance, "gray", "(c) layer 2: irradiance on the sensor"),
              (sim.image[crop].astype(float), "gray", "(d) layer 3: 12-bit counts, 60 × 80 px patch")]
    for ax, (data, cmap, title) in zip(axes, panels, strict=True):
        h = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(h, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(OUT / "simulation_layers.png")
    plt.close(fig)


def fig_sign_convention(img, edge_x, rows, eval_row, period):
    reversed_tilt = scene(fringes_y=-30.0).image.astype(float)
    right = edge_fraction(img, edge_x, rows, eval_row, period)
    wrong = edge_fraction(reversed_tilt, edge_x, rows, eval_row, period)
    fixed = edge_fraction(reversed_tilt, edge_x, rows, eval_row, period, order_sign=-1)
    crop = (slice(eval_row - 60, eval_row + 60), slice(int(edge_x) - 90, int(edge_x) + 90))
    extent = (crop[1].start, crop[1].stop, crop[0].stop, crop[0].start)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, frame, title in (
            (axes[0], img, f"order increasing down the image\norder_sign = +1 reads F = {right.fraction:.4f}"),
            (axes[1], reversed_tilt, f"reference tilt reversed: order increasing up\n"
                                     f"+1 reads {wrong.fraction:.4f} = 1 − F;  −1 reads {fixed.fraction:.4f}")):
        ax.imshow(frame[crop], cmap="gray", extent=extent)
        ax.axvline(edge_x, color=EDGE, lw=1.2, ls="--")
        ax.axhline(eval_row, color="w", lw=0.6, ls=":")
        ax.set_title(title)
        ax.set_xlabel("column x (px)")
    axes[0].set_ylabel("row y (px)")
    axes[0].text(edge_x - 85, eval_row - 50, "platen", color="w", fontsize=8)
    axes[0].text(edge_x + 10, eval_row - 50, "block", color="w", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "sign_convention.png")
    plt.close(fig)


def fig_air_index(wavelength_vac=543.5163):
    pressures = np.linspace(60_000.0, 110_000.0, 51)
    humidities = np.linspace(0.0, 80.0, 41)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for t, colour in ((15.0, PLATEN), (20.0, "k"), (25.0, BLOCK)):
        n1 = [EdlenRefractiveIndex(t, p, relative_humidity=50.0).index_refraction(wavelength_vac) - 1 for p in pressures]
        axes[0].plot(pressures / 1000, np.array(n1) * 1e6, color=colour, label=f"{t:.0f} °C")
    n1 = [EdlenRefractiveIndex(20.0, 75_300.0, relative_humidity=h).index_refraction(wavelength_vac) - 1
          for h in humidities]
    axes[1].plot(humidities, np.array(n1) * 1e6, color="k")
    example = (EdlenRefractiveIndex(20.0, 75_300.0, relative_humidity=50.0).index_refraction(wavelength_vac) - 1) * 1e6
    axes[0].plot(75.3, example, "o", color=EDGE, ms=6, zorder=3, label="example of the guide")
    axes[1].plot(50.0, example, "o", color=EDGE, ms=6, zorder=3)
    axes[0].set_xlabel("air pressure (kPa)")
    axes[0].set_title("(a) against pressure, 50 % relative humidity")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("relative humidity (%)")
    axes[1].set_title("(b) against humidity, 20 °C and 75.3 kPa")
    for ax in axes:
        ax.set_ylabel("(n − 1) × 10⁶")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Refractive index of air at λ_vac = {wavelength_vac} nm (modified Edlén, Birch & Downs)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "air_index.png")
    plt.close(fig)


def fig_mefm(wavelength=543.5, fraction=0.8934, nominal=2e6, prior=(55.0, 30.0)):
    half = wavelength / 2
    block = MonochromaticEFMSolver(wavelength=wavelength).solve(
        nominal_length=nominal, prior_length_dev=prior[0], prior_U=prior[1], fringe_fraction=fraction)
    kappa0 = int(np.floor(nominal / half))
    candidates = np.array([half * (k + fraction) - nominal for k in range(kappa0 - 4, kappa0 + 5)])
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.axvspan(-half / 2, half / 2, ymin=0.05, ymax=0.45, color=PLATEN, alpha=0.2,
               label="nominal prior: |d| < λ/4 (GaugeBlockLength)")
    ax.axvspan(prior[0] - prior[1], prior[0] + prior[1], ymin=0.55, ymax=0.95, color=EDGE, alpha=0.35,
               label=f"mechanical prior: δ ± U = {prior[0]:.0f} ± {prior[1]:.0f} nm (MEFM)")
    for d in candidates:
        ax.axvline(d, color="0.4", lw=1)
    ax.axvline(block.length_dev, color=BLOCK, lw=2.2, label=f"accepted: d = {block.length_dev:.1f} nm")
    ax.set_xlim(-650, 650)
    ax.set_yticks([])
    ax.set_xlabel("deviation from nominal d = L − L' (nm)")
    ax.set_title(f"One wavelength: candidates spaced λ/2 = {half:.2f} nm apart (F = {fraction}); "
                 "the prior keeps one")
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.savefig(OUT / "mefm.png")
    plt.close(fig)


def fig_parity(half=271.7, d=51.3, fraction=0.1416, prior=(51.0, 10.0), u_length=10.0):
    ff = np.linspace(0, 1, 401)
    check = check_fraction_parity(half, d, fraction, prior_length_dev=prior[0], prior_u=prior[1], u_length=u_length)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    ax.plot(ff, half * np.abs(2 * ff - np.round(2 * ff)), color="k")
    ax.plot(fraction, abs(check.flipped_length_dev - check.length_dev), "o", color=EDGE, ms=6,
            label=f"2 mm block, F = {fraction}: {abs(check.flipped_length_dev - check.length_dev):.1f} nm")
    ax.set_xlabel("fringe fraction F of the declared reading")
    ax.set_ylabel("least shift of d with 1 − F (nm)")
    ax.set_title("(a) what a wrong parity costs: (λ/2)·|2F − round(2F)|")
    ax.set_ylim(0, 1.3 * half / 2)
    ax.legend(fontsize=8, loc="upper center")
    ax.grid(alpha=0.3)
    ax = axes[1]
    u = np.hypot(prior[1], u_length)
    xx = np.linspace(-80, 110, 400)
    ax.fill_between(xx, np.exp(-0.5 * ((xx - prior[0]) / u) ** 2), color=EDGE, alpha=0.3,
                    label=f"prior {prior[0]:.0f} nm, combined u = {u:.1f} nm")
    ax.errorbar([check.length_dev], [0.5], xerr=[u_length], fmt="o", color=BLOCK, capsize=4,
                label=f"declared reading: d = {check.length_dev:+.1f} nm, z = {check.z:+.2f}")
    ax.errorbar([check.flipped_length_dev], [0.3], xerr=[u_length], fmt="s", color=PLATEN, capsize=4,
                label=f"other parity: d = {check.flipped_length_dev:+.1f} nm, z = {check.flipped_z:+.2f}")
    ax.set_yticks([])
    ax.set_xlabel("deviation from nominal d (nm)")
    ax.set_title(f"(b) test against a prior: verdict '{check.verdict}'")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "parity.png")
    plt.close(fig)


def fig_field_overview(img, edge_x, height, width):
    rows, cols = (100, height - 100), (0, width)
    pmap = field_overview(img, rows, cols, carrier_cols=(40, int(edge_x) - 40))
    extent = (cols[0], cols[1], rows[1], rows[0])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    h = axes[0].imshow(pmap.object_phase / (2 * np.pi), cmap="twilight", extent=extent, aspect="auto")
    axes[0].set_title("(a) carrier-free phase (fringes): the block stands out by F")
    fig.colorbar(h, ax=axes[0], fraction=0.046, pad=0.03)
    h = axes[1].imshow(pmap.modulation, cmap="magma", extent=extent, aspect="auto")
    axes[1].set_title("(b) fringe modulation: the dip marks the block edge")
    fig.colorbar(h, ax=axes[1], fraction=0.046, pad=0.03)
    for ax in axes:
        ax.axvline(edge_x, color=EDGE, lw=0.8, ls="--")
        ax.set_xlabel("column x (px)")
    axes[0].set_ylabel("row y (px)")
    fig.tight_layout()
    fig.savefig(OUT / "field_overview.png")
    plt.close(fig)


def fig_run_calibration():
    tool = Path(__file__).resolve().parent / "run_calibration.py"
    spec = importlib.util.spec_from_file_location("run_calibration", tool)
    run_calibration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_calibration)
    artifacts = {}
    summary = run_calibration.run(run_calibration.parse_args(["--block-size-mm", "9", "14", "--frames", "2"]),
                                  artifacts)
    run_calibration.plot_regions(OUT / "run_calibration_areas.png", summary, artifacts)


def fig_pipeline(img, edge_x, rows, eval_row, period, gap=30, span=180, wavelength=543.5):
    res = edge_fraction(img, edge_x, rows, eval_row, period)
    block = MonochromaticEFMSolver(wavelength=wavelength).solve(
        nominal_length=2e6, prior_length_dev=55.0, prior_U=30.0, fringe_fraction=res.fraction)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4), gridspec_kw={"width_ratios": [1, 1.2, 1.2], "wspace": 0.5})
    ax = axes[0]
    crop = (slice(eval_row - 120, eval_row + 120), slice(int(edge_x) - 200, int(edge_x) + 200))
    ax.imshow(img[crop], cmap="gray", extent=(crop[1].start, crop[1].stop, crop[0].stop, crop[0].start),
              aspect="auto")
    ax.axvline(edge_x, color=EDGE, lw=1.2, ls="--")
    for x0, colour in ((edge_x - span, PLATEN), (edge_x + gap, BLOCK)):
        ax.add_patch(plt.Rectangle((x0, rows[0]), span - gap, rows[1] - rows[0], fill=False, ec=colour, lw=1.5))
    ax.set_title("1. interferogram: bands on each side")
    ax.set_xticks([])
    ax.set_yticks([])
    ax = axes[1]
    for side, colour, marker in ((res.platen, PLATEN, "s"), (res.block, BLOCK, "o")):
        phase = side.phase_unwrapped / (2 * np.pi)
        ax.plot(side.x, phase - phase.mean() + (0 if side is res.platen else res.fraction), marker, ms=3,
                color=colour)
    ax.axvline(edge_x, color=EDGE, lw=1, ls="--")
    ax.set_title(f"2. phases at the edge: F = {res.fraction:.4f}")
    ax.set_xlabel("column x (px)")
    ax.set_ylabel("phase (fringes)")
    ax = axes[2]
    half = wavelength / 2
    kappa0 = int(np.floor(2e6 / half))
    for k in range(kappa0 - 3, kappa0 + 4):
        ax.axvline(half * (k + res.fraction) - 2e6, color="0.5", lw=1)
    ax.axvspan(25, 85, color=EDGE, alpha=0.35)
    ax.axvline(block.length_dev, color=BLOCK, lw=2.2)
    ax.set_xlim(-500, 500)
    ax.set_yticks([])
    ax.set_xlabel("L − L' (nm)")
    ax.set_title(f"3. integer order and length: L − L' = {block.length_dev:.1f} nm")
    for left, right in ((axes[0], axes[1]), (axes[1], axes[2])):   # an arrow in each gap, above the axis labels
        x = (left.get_position().x1 + right.get_position().x0) / 2
        fig.text(x, 0.82, "→", fontsize=24, ha="center", va="center")
    fig.savefig(OUT / "pipeline.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sim = scene()
    img = sim.image.astype(float)
    height, width = img.shape
    edge_x = detect_edge_column(img, search=(0, width // 2)) - 0.5
    eval_row = height // 2
    rows = (eval_row - 75, eval_row + 75)
    fig_interferogram(img, edge_x, rows)
    fig_edge_comparison(img, edge_x, rows, eval_row, height / 30)
    fig_fourier(img, edge_x, height)
    fig_phase_stepping(edge_x, rows, eval_row)
    fig_exact_fractions()
    fig_simulation_layers(sim)
    fig_sign_convention(img, edge_x, rows, eval_row, height / 30)
    fig_air_index()
    fig_mefm()
    fig_parity()
    fig_field_overview(img, edge_x, height, width)
    fig_run_calibration()
    fig_pipeline(img, edge_x, rows, eval_row, height / 30)
    for p in sorted(OUT.glob("*.png")):
        print(p.name, p.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
