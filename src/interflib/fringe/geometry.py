"""Locate the gauge block in an interferogram and normalise the geometry for the edge readers.

The edge readers (:mod:`interflib.fringe.edge_comparison`, :mod:`interflib.fringe.fourier`,
:mod:`interflib.fringe.phase_shifting`) compare column bands on either side of a vertical
edge with the platen on the left. :func:`locate_block` finds the block edge without
assuming where it is or how it is oriented, and returns the image rotated and mirrored so
that those assumptions hold.

It works on the whole-field Fourier demodulation (Takeda, Ina & Kobayashi, *J. Opt. Soc.
Am.* 72, 156-160, 1982):

1. the **fringe zone** is where the recovered modulation reaches a fraction of its 95th
   percentile (the median of a frame with a dark surround is too low a reference);
2. the **carrier-free phase** of block and platen differs by ``2 pi F`` plus the wedge, so
   the magnitude of its local wrapped difference draws the block outline in any
   orientation (it vanishes only for ``F`` near 0);
3. the block edge is a line of phase steps **inside** the zone, with fringes on both sides,
   found collinear in at least six bands of 40 rows; the borders of the zone are excluded;
4. which side of that line is the platen cannot be told from the phase (it would read
   ``1 - F``), so it is **declared** (``platen_side``) and only checked against the
   expected face width when the pixel scale is known.

The edge found is refined to the centre of the dark line that the chamfer draws when the
line is present within a few pixels. With few fringes in the field the demodulation, and
therefore this function, fails (``ValueError``).
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import affine_transform, binary_erosion, uniform_filter, uniform_filter1d
from scipy.signal import find_peaks

from .diagnostics import field_overview
from .edge_comparison import edge_line_centre
from .fourier import estimate_carrier_2d, modulation_mask

ZONE_FRACTION = 0.3        # fringe zone: modulation above this fraction of its 95th percentile
BORDER_MARGIN = 40         # px: a phase-step peak closer than this to the zone border is the border
SLAB_ROWS = 40             # height of each band of rows searched for the edge
MIN_SLABS = 6              # an edge line must be found, collinear, in at least this many bands (~240 rows)
STRONG_FRACTION = 0.7      # only lines with this fraction of the best support compete for the edge
INLIER_PX = 3.0            # px: distance of a band peak to its line to count as support
MAX_ANGLE_DEG = 30.0       # steepest edge line considered
REFINE_PX = 4.0            # px: the dark chamfer line refines the edge when it lies this close
DARK_LINE_DEPTH = 0.1      # and is this much darker than the columns 10-20 px on either side
                           # (0.47-0.52 on the 2026-10 bench frames, 0.004 where there is no line)


@dataclass
class BlockGeometry:
    """Block edge and image normalised so that the edge is vertical with the platen on the left.

    Attributes
    ----------
    image : numpy.ndarray
        The normalised image (rotated by ``rotation_deg`` and mirrored if ``mirrored``).
    edge_x : float
        Edge column in the normalised image, px (boundary between platen and block columns).
    edge_x_phase : float
        The same edge from the phase step alone, before the dark-line refinement, px.
    refined_by_dark_line : bool
        Whether the dark chamfer line was found close enough to refine the edge.
    angle_deg : float
        Angle of the edge from the vertical in the input image, degrees (as found).
    rotation_deg : float
        Rotation applied to make the edge vertical: ``angle_deg``, or 0 below
        ``min_angle_deg``.
    mirrored : bool
        Whether the image was mirrored (platen declared on the right).
    support : float
        Fraction of the bands of rows in which the edge line was found.
    platen_width_px, face_width_px : float
        Width of the fringe zone left of the edge, and of the face right of it (up to the
        next strong edge line or the end of the zone), at the reference row, px.
    face_rows : tuple of int
        Rows, around the reference row, over which the block face extends along the edge.
    zone : numpy.ndarray
        Boolean fringe zone in the normalised image.
    carrier : tuple of float
        ``(g_x, g_y)`` of the input image, fringes per px.
    centre : tuple of float
        ``(y, x)`` centre of the rotation, input pixels.
    """

    image: np.ndarray
    edge_x: float
    edge_x_phase: float
    refined_by_dark_line: bool
    angle_deg: float
    rotation_deg: float
    mirrored: bool
    support: float
    platen_width_px: float
    face_width_px: float
    face_rows: tuple
    zone: np.ndarray
    carrier: tuple
    centre: tuple

    def to_normalised(self, y, x):
        """Map an input pixel ``(y, x)`` to the normalised image."""
        return _forward(y, x, np.radians(self.rotation_deg), self.centre, self.image.shape[1], self.mirrored)

    def normalise(self, img):
        """Apply the same rotation and mirroring to another frame of the series."""
        return _transform(np.asarray(img, dtype=float), np.radians(self.rotation_deg), self.centre, self.mirrored)


def fringe_zone(img, fraction=ZONE_FRACTION):
    """Whole-field demodulation and fringe zone of an image.

    The carrier is estimated on the columns of high intensity contrast, over the middle half
    of the rows, so that a dark surround does not hide it.

    Returns
    -------
    tuple
        ``(phase_map, zone)``: the :class:`interflib.fringe.fourier.PhaseMap` of the whole
        field and the boolean fringe zone.
    """
    img = np.asarray(img, dtype=float)
    height, width = img.shape
    rows = slice(height // 4, 3 * height // 4)
    contrast = uniform_filter1d(img[rows].std(axis=0), 9)
    cols = np.flatnonzero(contrast > 0.4 * np.percentile(contrast, 95))
    carrier = estimate_carrier_2d(img[rows, cols.min():cols.max() + 1])
    phase_map = field_overview(img, (0, height), (0, width), carrier=carrier)
    b = phase_map.modulation
    zone = modulation_mask(b, fraction, reference=np.nanpercentile(b, 95))
    return phase_map, zone


def phase_step_map(phase_map, smooth=5):
    """Magnitude of the local wrapped difference of the carrier-free phase, radians.

    Central differences of unit phasors along both axes, combined in quadrature and
    smoothed over ``smooth`` px; it peaks along lines where the phase jumps (block outline).
    """
    z = np.exp(1j * phase_map.object_phase)
    step = np.zeros(z.shape)
    gx = np.abs(np.angle(z[1:-1, 2:] * np.conj(z[1:-1, :-2])))
    gy = np.abs(np.angle(z[2:, 1:-1] * np.conj(z[:-2, 1:-1])))
    step[1:-1, 1:-1] = np.hypot(gx, gy)
    return uniform_filter(step, smooth)


def _interior_peaks(step, zone, y0, y1):
    # Phase-step peaks of one band of rows that lie inside the zone, away from its borders.
    inner = binary_erosion(zone[y0:y1], iterations=5)
    if inner.sum() == 0:
        return []
    counts = inner.sum(axis=0)
    profile = np.where(inner, step[y0:y1], 0.0).sum(axis=0) / np.maximum(counts, 1)
    covered = np.flatnonzero(counts > 0)
    peaks, props = find_peaks(profile, prominence=0.3 * profile.max() if profile.max() > 0 else np.inf)
    lo, hi = covered.min() + BORDER_MARGIN, covered.max() - BORDER_MARGIN
    keep = [(p, prom) for p, prom in zip(peaks, props["prominences"], strict=True) if lo <= p <= hi]
    out = []
    for p, prom in keep:
        delta = 0.0
        if 0 < p < len(profile) - 1:
            left, centre, right = profile[p - 1:p + 2]
            curvature = left - 2 * centre + right
            delta = 0.5 * (left - right) / curvature if curvature < 0 else 0.0
        out.append((p + delta, prom))
    return out


def _edge_lines(points, n_slabs, rng):
    # Lines x = a + b y supported by the band peaks (a small RANSAC over pairs of points).
    points = np.asarray(points, dtype=float)
    lines = []
    max_slope = np.tan(np.radians(MAX_ANGLE_DEG))
    while len(points) >= 2:
        best = None
        for _ in range(300):
            i, j = rng.choice(len(points), 2, replace=False)
            (yi, xi), (yj, xj) = points[i, :2], points[j, :2]
            if yi == yj:
                continue
            b = (xj - xi) / (yj - yi)
            if abs(b) > max_slope:
                continue
            a = xi - b * yi
            inliers = np.abs(points[:, 1] - (a + b * points[:, 0])) <= INLIER_PX
            if best is None or inliers.sum() > best.sum():
                best = inliers
        if best is None or len(np.unique(points[best, 0])) < MIN_SLABS:
            break
        b, a = np.polyfit(points[best, 0], points[best, 1], 1)
        lines.append((a, b, len(np.unique(points[best, 0])) / n_slabs))
        points = points[~best]
    return lines


def _detect(img, platen_side, row_span):
    phase_map, zone = fringe_zone(img)
    step = phase_step_map(phase_map)
    height = img.shape[0]
    rows_with_fringes = np.flatnonzero(zone.any(axis=1))
    y_lo, y_hi = rows_with_fringes.min(), rows_with_fringes.max()
    if row_span is not None:
        y_lo, y_hi = max(y_lo, row_span[0]), min(y_hi, row_span[1])
    starts = np.arange(y_lo, max(y_lo + 1, y_hi - SLAB_ROWS), SLAB_ROWS)
    points = [((y0 + min(y0 + SLAB_ROWS, height)) / 2, x, prom)
              for y0 in starts for x, prom in _interior_peaks(step, zone, y0, min(y0 + SLAB_ROWS, height))]
    lines = _edge_lines(points, len(starts), np.random.default_rng(0)) if points else []
    if not lines:
        raise ValueError("no edge line of phase steps found inside the fringe zone")
    y_ref = (y_lo + y_hi) / 2
    # Weak lines (scratches, stray fringes) do not compete. Of the strong ones (a block with
    # platen around it has two), the one nearest the declared platen side is the edge with
    # the platen on that side and the block on the other.
    best = max(line[2] for line in lines)
    lines = sorted((line for line in lines if line[2] >= STRONG_FRACTION * best),
                   key=lambda line: line[0] + line[1] * y_ref)
    a, b, support = lines[0] if platen_side == "left" else lines[-1]
    return phase_map, zone, step, a, b, support, y_ref, lines


def _forward(y, x, theta, centre, width, mirrored):
    # Input pixel -> normalised pixel: o = R^T (i - c) + c, then the optional mirror.
    cy, cx = centre
    dy, dx = y - cy, x - cx
    yo = np.cos(theta) * dy + np.sin(theta) * dx + cy
    xo = -np.sin(theta) * dy + np.cos(theta) * dx + cx
    if mirrored:
        xo = width - 1 - xo
    return yo, xo


def _transform(img, theta, centre, mirrored):
    # Normalised image: output pixel o samples the input at i = R (o - c) + c.
    if theta != 0.0:
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        c = np.asarray(centre, dtype=float)
        img = affine_transform(img, rot, offset=c - rot @ c, order=3, mode="nearest")
    return img[:, ::-1] if mirrored else img


def locate_block(img, platen_side="left", row_span=None, eval_row=None, min_angle_deg=1.0):
    """Find the block edge and return the geometry normalised for the edge readers.

    Parameters
    ----------
    img : numpy.ndarray
        Interferogram, rows x columns (a single frame or the mean of a burst).
    platen_side : {"left", "right"}
        Side of the edge on which the platen lies in the input image. It is declared, not
        detected: swapping the sides reads ``1 - F``.
    row_span : tuple of int, optional
        Rows searched for the edge; all rows with fringes if omitted.
    eval_row : int, optional
        Reference row for the widths and the face extent; the middle of the searched rows
        if omitted.
    min_angle_deg : float
        Edges closer to vertical than this are not rotated, to avoid the interpolation: at
        1 degree the edge moves 2.6 px over a 150-row fit window, well inside the 30 px gap
        of the readers. ``angle_deg`` reports the angle found either way.

    Returns
    -------
    BlockGeometry

    Raises
    ------
    ValueError
        If the field cannot be demodulated (too few fringes) or no edge line is found.
    """
    if platen_side not in ("left", "right"):
        raise ValueError("platen_side must be 'left' or 'right'")
    img = np.asarray(img, dtype=float)
    height, width = img.shape
    centre = ((height - 1) / 2, (width - 1) / 2)
    phase_map, zone, step, a, b, support, y_ref, lines = _detect(img, platen_side, row_span)
    carrier = tuple(float(g) for g in phase_map.carrier)
    angle = float(np.arctan(b))
    theta = angle if abs(np.degrees(angle)) >= min_angle_deg else 0.0
    mirrored = platen_side == "right"
    norm = _transform(img, theta, centre, mirrored)
    if theta != 0.0 or mirrored:
        span = None if row_span is None else row_span
        _, zone, step, a, b, support, y_ref, lines = _detect(norm, "left", span)
    y_e = int(round(y_ref if eval_row is None else eval_row))
    edge_phase = float(a + b * y_e)

    # Refine with the dark chamfer line when it is present close to the phase-step edge.
    rows = (max(0, y_e - 200), min(height, y_e + 200))
    try:
        dark = edge_line_centre(norm, int(round(edge_phase)), rows=rows, half=8)
        refined = abs(dark - edge_phase) <= REFINE_PX and _dark_line_depth(norm, dark, rows) >= DARK_LINE_DEPTH
    except (ValueError, IndexError):
        dark, refined = edge_phase, False
    edge_x = float(dark) if refined else edge_phase

    # Widths of the zone on each side, and the extent of the face along the edge.
    zone_row = np.flatnonzero(zone[y_e])
    platen_width = edge_x - zone_row.min() if zone_row.size else 0.0
    # The face ends at the next strong edge line on the block side (the far edge of the block,
    # when platen surrounds it) or at the end of the fringe zone, whichever comes first.
    face_end = zone_row.max() if zone_row.size else edge_x
    for la, lb, _ in lines:
        x_line = la + lb * y_e
        if x_line > edge_x + 2 * BORDER_MARGIN:
            face_end = min(face_end, x_line)
    face_width = face_end - edge_x
    period = 1.0 / max(abs(carrier[1]), 1e-6)
    face_rows = _face_rows(step, zone, edge_x, y_e, period)
    return BlockGeometry(image=norm, edge_x=edge_x, edge_x_phase=edge_phase, refined_by_dark_line=bool(refined),
                         angle_deg=float(np.degrees(angle)), rotation_deg=float(np.degrees(theta)),
                         mirrored=mirrored, support=float(support),
                         platen_width_px=float(platen_width), face_width_px=float(face_width),
                         face_rows=face_rows, zone=zone, carrier=carrier,
                         centre=centre)


def _dark_line_depth(img, x, rows):
    # Relative depth of the column-mean minimum at x against the columns 10-20 px on either side.
    means = np.convolve(img[rows[0]:rows[1]].mean(axis=0), np.ones(3) / 3, "same")
    c = int(round(x))
    if c < 20 or c + 20 > len(means):
        return 0.0
    ring = np.r_[means[c - 20:c - 10], means[c + 10:c + 20]].mean()
    return float(1.0 - means[c - 2:c + 3].min() / ring) if ring > 0 else 0.0


def _face_rows(step, zone, edge_x, y_e, period, inset=(60, 160)):
    # Rows around y_e over which a strip of the face (60-160 px right of the edge) stays in
    # the fringe zone. A horizontal line of phase steps across the strip is the top or bottom
    # end of the block; the face stops one fringe period short of it, where the
    # demodulation is no longer disturbed by the step.
    x0, x1 = int(edge_x + inset[0]), int(min(step.shape[1], edge_x + inset[1]))
    inside = zone[:, x0:x1].mean(axis=1) > 0.9
    rows = np.flatnonzero(inside)
    if rows.size == 0:
        return int(y_e), int(y_e) + 1
    top, bottom = y_e, y_e
    while top > 0 and inside[top - 1]:
        top -= 1
    while bottom < len(inside) - 1 and inside[bottom + 1]:
        bottom += 1
    profile = uniform_filter1d(step[:, x0:x1].mean(axis=1), 9)
    ref = profile[top:bottom + 1]
    spread = np.median(np.abs(ref - np.median(ref)))
    peaks, _ = find_peaks(np.where(inside, profile, 0.0),
                          prominence=max(0.3 * ref.max(), 6 * spread))
    guard = int(np.ceil(period))
    above = peaks[(peaks < y_e) & (peaks >= top)]
    below = peaks[(peaks > y_e) & (peaks <= bottom)]
    if above.size:
        top = int(above.max()) + guard
    if below.size:
        bottom = int(below.min()) - guard
    return int(top), int(bottom + 1)
