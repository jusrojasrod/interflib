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

A line of phase steps is not always the edge: with wide fringes the demodulation draws stray
lines with as much support as the edge, and the phase-step line moves off the edge in
position and angle. The **dark line** that the chamfer draws along the edge confirms it:

5. of the strong phase-step lines, only those along which the dark line runs compete, and
   the edge is moved onto the dark line (its angle too, when the dark line contradicts it);
6. when none has it, the edge is the dark line nearest the platen side inside the zone,
   searched in the column means (a near-vertical edge);
7. when there is no dark line either, the edge rests on the phase step alone and
   ``edge_source`` says so: it may be a stray line.

With few fringes in the field the demodulation, and therefore this function, fails
(``ValueError``).
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import affine_transform, binary_erosion, map_coordinates, uniform_filter, uniform_filter1d
from scipy.signal import argrelmin, find_peaks

from .diagnostics import field_overview
from .fourier import estimate_carrier_2d, modulation_mask
from .models import parabolic_vertex

ZONE_FRACTION = 0.3        # fringe zone: modulation above this fraction of its 95th percentile
BORDER_MARGIN = 40         # px: a phase-step peak closer than this to the zone border is the border
SLAB_ROWS = 40             # height of each band of rows searched for the edge
MIN_SLABS = 6              # an edge line must be found, collinear, in at least this many bands (~240 rows)
STRONG_FRACTION = 0.7      # only lines with this fraction of the best support compete for the edge
INLIER_PX = 3.0            # px: distance of a band peak to its line to count as support
MAX_ANGLE_DEG = 30.0       # steepest edge line considered
REFINE_PX = 4.0            # px: the dark chamfer line confirms and refines the edge when it lies
REFINE_PERIODS = 0.15      # within this distance of the phase-step line, or this fraction of the
                           # fringe period if larger (the phase-step line moves off the edge as the
                           # fringes widen: 7 px at a 133 px period on the 2026-10 bench frames)
DARK_LINE_DEPTH = 0.25     # and is this much darker than the profile 10-20 px on its darker side
                           # (edges 0.41-0.53 on the 2026-10 bench frames; beam borders 0.11-0.16;
                           # below 0.06 along stray phase-step lines)
DARK_SLAB_ROWS = 100       # bands of rows in which the dark line is followed to check the edge angle,
DARK_FOLLOW_ROWS = 400     # over this many rows on either side of the reference row


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
        The same edge from the phase step alone, before the dark-line refinement, px
        (``nan`` when the edge is the dark line alone).
    refined_by_dark_line : bool
        Whether the edge lies on the dark chamfer line (``edge_source`` other than
        ``"phase step"``).
    edge_source : str
        ``"phase step and dark line"`` (a phase-step line confirmed by the dark line),
        ``"dark line"`` (no phase-step line had it; the dark line nearest the platen side)
        or ``"phase step"`` (no dark line: the edge is unconfirmed and may be a stray line).
    dark_line_depth : float
        Relative depth of the dark line at the edge (``DARK_LINE_DEPTH`` or more confirms it).
    angle_deg : float
        Angle of the edge from the vertical in the input image, degrees (as found).
    rotation_deg : float
        Rotation applied to make the edge vertical: ``angle_deg``, or 0 below
        ``min_angle_deg``.
    mirrored : bool
        Whether the image was mirrored (platen declared on the right).
    support : float
        Fraction of the bands of rows in which the phase-step edge line was found (``nan``
        when the edge is the dark line alone).
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
    edge_source: str
    dark_line_depth: float
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
    # Weak lines (scratches, stray fringes) do not compete.
    best = max(line[2] for line in lines)
    lines = sorted((line for line in lines if line[2] >= STRONG_FRACTION * best),
                   key=lambda line: line[0] + line[1] * y_ref)
    return phase_map, zone, step, y_ref, lines


def _choose_edge(img, zone, lines, platen_side, y_c, period):
    # Of the strong lines (a block with platen around it has two), those along which the
    # dark chamfer line runs are the block outline; a line without it is a stray phase step
    # (on the 2026-10 bench frames with 5 to 20 fringes such lines lay up to 150 px from the
    # edge, with as much support as the edge). When none has it, the dark lines inside the
    # zone stand in for them. Of the candidates, the one nearest the declared platen side is
    # the edge with the platen on that side and the block on the other.
    # Returns (phase line or None, edge line (a, b), dark-line depth, edge source).
    half = int(min(DARK_FOLLOW_ROWS, max(200, 3 * period)))     # three fringes at least
    rows = (max(0, int(y_c) - half), min(img.shape[0], int(y_c) + half))
    reach = max(REFINE_PX, REFINE_PERIODS * period)
    dark = [_dark_line_across(img, a, b, rows, reach) for a, b, _ in lines]
    confirmed = [i for i, (_, depth) in enumerate(dark) if depth >= DARK_LINE_DEPTH]
    if confirmed:
        i = confirmed[0] if platen_side == "left" else confirmed[-1]
        a, b, _ = lines[i]
        source = "phase step and dark line"
    else:
        columns = _dark_columns(img, zone, rows)
        if not columns:
            i = 0 if platen_side == "left" else len(lines) - 1
            a, b, _ = lines[i]
            return lines[i], (float(a), float(b)), dark[i][1], "phase step"
        i, a, b, reach = None, float(columns[0] if platen_side == "left" else columns[-1]), 0.0, REFINE_PX
        source = "dark line"
    b_edge = _dark_line_slope(img, a, b, y_c, reach)
    a = a + (b - b_edge) * y_c                             # same point at y_c, new slope
    offset, depth = _dark_line_across(img, a, b_edge, rows, reach)
    return (None if i is None else lines[i]), (float(a + offset), float(b_edge)), depth, source


def _dark_columns(img, zone, rows):
    # Columns of the dark lines inside the fringe zone (away from its borders) over ``rows``:
    # local minima of the column means with the depth of an edge (a near-vertical edge).
    covered = np.flatnonzero(zone[rows[0]:rows[1]].mean(axis=0) > 0.5)    # the zone has holes with wide fringes
    if covered.size == 0:
        return []
    lo, hi = covered.min() + BORDER_MARGIN, covered.max() - BORDER_MARGIN
    profile = np.convolve(img[rows[0]:rows[1]].mean(axis=0), np.ones(3) / 3, "same")
    return [int(c) for c in argrelmin(profile, order=3)[0]
            if lo <= c <= hi and _depth(profile, c) >= DARK_LINE_DEPTH]


def _depth(profile, c):
    # Relative depth of the minimum at c against the profile 10-20 px on its darker side, so
    # that an intensity step (one bright side) is not taken for a line.
    if c < 22 or c + 22 > len(profile):
        return 0.0
    ring = min(profile[c - 20:c - 10].mean(), profile[c + 11:c + 21].mean())
    return float(1.0 - profile[c - 2:c + 3].min() / ring) if ring > 0 else 0.0


def _dark_line_slope(img, a, b, y_c, reach):
    # Slope of the edge: that of the phase-step line x = a + b y unless the dark-line centres
    # of bands of DARK_SLAB_ROWS rows, over DARK_FOLLOW_ROWS rows on either side of y_c,
    # contradict it by more than three standard errors. With wide fringes the phase-step
    # line is off in angle (-1.9 deg where the dark line runs at +0.5 deg, on a 2026-10 bench
    # frame at a 133 px period); with dense ones it is the steadier estimate, since the fringes pull the band
    # centres of the dark line some 3 px about.
    ys, xs = [], []
    y_start = max(0, int(y_c) - DARK_FOLLOW_ROWS)
    y_stop = min(img.shape[0], int(y_c) + DARK_FOLLOW_ROWS)
    for y0 in range(y_start, y_stop - DARK_SLAB_ROWS + 1, DARK_SLAB_ROWS):
        offset, depth = _dark_line_across(img, a, b, (y0, y0 + DARK_SLAB_ROWS), reach)
        if depth >= DARK_LINE_DEPTH:
            y = y0 + (DARK_SLAB_ROWS - 1) / 2
            ys.append(y)
            xs.append(a + b * y + offset)
    if len(ys) < 4:
        return b
    ys, xs = np.asarray(ys), np.asarray(xs)
    slope, intercept = np.polyfit(ys, xs, 1)
    residual = xs - (intercept + slope * ys)
    se = np.sqrt(residual @ residual / (len(ys) - 2) / np.sum((ys - ys.mean()) ** 2))
    return float(slope) if abs(slope - b) > 3 * se else b


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
    phase_map, zone, step, y_ref, lines = _detect(img, platen_side, row_span)
    carrier = tuple(float(g) for g in phase_map.carrier)
    period = 1.0 / max(np.hypot(*carrier), 1e-6)
    y_c = y_ref if eval_row is None else eval_row
    choice = _choose_edge(img, zone, lines, platen_side, y_c, period)
    angle = float(np.arctan(choice[1][1]))
    theta = angle if abs(np.degrees(angle)) >= min_angle_deg else 0.0
    mirrored = platen_side == "right"
    norm = _transform(img, theta, centre, mirrored)
    if theta != 0.0 or mirrored:
        _, zone, step, y_ref, lines = _detect(norm, "left", row_span)
        y_c = y_ref if eval_row is None else eval_row
        choice = _choose_edge(norm, zone, lines, "left", y_c, period)
    phase_line, (a, b), depth, source = choice
    y_e = int(round(y_c))
    edge_x = float(a + b * y_e)
    if phase_line is None:
        edge_phase, support = float("nan"), float("nan")
    else:
        edge_phase, support = float(phase_line[0] + phase_line[1] * y_e), phase_line[2]

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
    face_rows = _face_rows(step, zone, edge_x, y_e, 1.0 / max(abs(carrier[1]), 1e-6))
    return BlockGeometry(image=norm, edge_x=edge_x, edge_x_phase=edge_phase, refined_by_dark_line=source != "phase step",
                         edge_source=source, dark_line_depth=float(depth),
                         angle_deg=float(np.degrees(angle)), rotation_deg=float(np.degrees(theta)),
                         mirrored=mirrored, support=float(support),
                         platen_width_px=float(platen_width), face_width_px=float(face_width),
                         face_rows=face_rows, zone=zone, carrier=carrier,
                         centre=centre)


def _dark_line_across(img, a, b, rows, reach):
    # Darkest point of the mean profile across the line x = a + b y (sampled along the line
    # over ``rows``) within ``reach`` px of it: (offset from the line, px; relative depth).
    half = int(np.ceil(reach)) + 22
    y = np.arange(*rows, dtype=float)
    k = np.arange(-half, half + 1, dtype=float)
    x = a + b * y[:, None] + k[None, :]
    if x.min() < 0 or x.max() > img.shape[1] - 1:
        return 0.0, 0.0
    samples = map_coordinates(img, [np.broadcast_to(y[:, None], x.shape), x], order=1)
    profile = np.convolve(samples.mean(axis=0), np.ones(3) / 3, "same")
    inner = np.flatnonzero(np.abs(k) <= reach)
    c = int(inner[np.argmin(profile[inner])])
    if profile[c] > min(profile[c - 1], profile[c + 1]):
        return float(k[c]), 0.0                 # the profile still falls at the end of the reach
    return float(k[c] + parabolic_vertex(*profile[c - 1:c + 2])), _depth(profile, c)


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
