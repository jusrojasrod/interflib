"""Regenerate the figures of the README in ``docs/images/``.

Every figure comes from the layered simulator, so it can be rebuilt anywhere and shows
a known truth. Run from the repository root: ``python tools/make_readme_figures.py``.
"""
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from interflib.dimensional.coincidence import CoincidenceEFSolver  # noqa: E402
from interflib.fringe.edge_comparison import column_phases, detect_edge_column, side_band_centres, side_fit  # noqa: E402
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
    for p in sorted(OUT.glob("*.png")):
        print(p.name, p.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
