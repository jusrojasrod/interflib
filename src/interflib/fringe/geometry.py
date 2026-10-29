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

The central length of a gauge block is defined at the centre of its free measuring face (ISO
3650:1998, clause 3). :func:`locate_block` also looks for the rest of the face outline: the
**far edge** (the lateral edge opposite the platen, with the same order of preference as the
edge: phase-step line confirmed by the dark line, dark line, phase-step line; else the end of
the lit field, which may be the border of the beam) and the **ends** of the face (horizontal
lines of phase steps across the face that the platen strip does not share at the same row,
with fringes on both sides). :func:`locate_face_centre` then places the centre:

- from the whole outline (both lateral edges and both ends);
- from part of it and the face dimensions in mm (each axis with both features uses their
  midpoint; with one, the feature plus half the dimension times the scale);
- from coordinates given by hand, as the last resort.
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
FACE_INSET = 60            # px: strips on the face and the platen start this far from the edges
END_PROMINENCE = 0.5       # rad: a block end is a peak of the face-strip phase-step profile at least
                           # this prominent divided by the fringe period in px (the peak falls as the
                           # period grows: 0.17-0.38 at the ends of the 2026-10 bench frames, periods
                           # 37-47 px; 0.06 to 0.01 on simulated ends, F = 0.25, periods 35 to 128 px)
END_PLATEN_RATIO = 0.5     # and the platen strip, at the same rows, steps less than this fraction of
                           # it (0.03-0.18 at the bench ends, up to 0.38 on simulated ones with wide
                           # fringes; 0.69 and more at the beam borders, which cross the platen too)
END_MODULATION = 0.015     # and the face has fringes one period above and below it: modulation of at
                           # least this fraction of its 95th percentile (0.025 and more at the ends;
                           # 0.007 at the beam borders in the dark)
FIELD_BORDER_LEVEL = 0.5   # the lit field ends where the face profile falls below this fraction of
FIELD_BORDER_RUN = 15      # its level near the edge, for at least this many px (a dark line is narrower)
FEATURE_U_PX = 2.0         # px: standard uncertainty of each outline feature (working assumption,
                           # not yet validated on real blocks; simulated edges within 0.3 px)
FAR_EDGE_SCALE_TOL = 0.05  # a far edge without the dark line is used with the dimensions only if the face width
                           # it gives is within this fraction of the declared pixel size (2026-10 bench: 17.7-18.2
                           # um/px with confirmed edges; 19.9-30 um/px where the beam clips the block)
END_U_PERIODS = 0.05       # and of each end, at least this fraction of the fringe period (the
                           # demodulation blurs the ends along the fringes: simulated centre rows off
                           # by 1.4 px at a 35 px period and by up to 3.6 px at 86-128 px)


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
    eval_row : float
        Reference row at which ``edge_x`` and ``far_edge_x`` are given, px.
    edge_slope : float
        ``dx/dy`` of the edge in the normalised image (not zero when the edge was left
        unrotated, below ``min_angle_deg``).
    far_edge_x : float
        Far lateral edge of the face (opposite the platen) at ``eval_row``, px; ``nan`` if
        not found.
    far_edge_source : str
        ``"phase step and dark line"``, ``"dark line"`` or ``"phase step"`` (as for the
        edge), ``"field border"`` (the end of the lit field: it may be the border of the
        beam, not of the block) or ``"none"``.
    face_ends : tuple of float
        ``(top, bottom)`` rows of the ends of the face, px; ``nan`` for an end not found.
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
    eval_row: float
    edge_slope: float
    far_edge_x: float
    far_edge_source: str
    face_ends: tuple

    def to_normalised(self, y, x):
        """Map an input pixel ``(y, x)`` to the normalised image."""
        return _forward(y, x, np.radians(self.rotation_deg), self.centre, self.image.shape[1], self.mirrored)

    def edge_x_at(self, y):
        """Edge column at row ``y`` of the normalised image, px."""
        return self.edge_x + self.edge_slope * (y - self.eval_row)

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
        phase_map, zone, step, y_ref, lines = _detect(norm, "left", row_span)
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
    period_y = 1.0 / max(abs(carrier[1]), 1e-6)
    face_rows = _face_rows(step, zone, edge_x, y_e, period_y)

    # The rest of the outline: far edge and ends of the face.
    half = int(min(DARK_FOLLOW_ROWS, max(200, 3 * period)))
    rows = (max(0, y_e - half), min(height, y_e + half))
    far_x, far_source = _far_edge(norm, lines, a, b, y_e, rows, period)
    x1 = far_x - FACE_INSET if np.isfinite(far_x) else edge_x + face_width - 20
    face_ends = _face_ends(phase_map, step, edge_x, x1, y_e, period_y)
    return BlockGeometry(image=norm, edge_x=edge_x, edge_x_phase=edge_phase, refined_by_dark_line=source != "phase step",
                         edge_source=source, dark_line_depth=float(depth),
                         angle_deg=float(np.degrees(angle)), rotation_deg=float(np.degrees(theta)),
                         mirrored=mirrored, support=float(support),
                         platen_width_px=float(platen_width), face_width_px=float(face_width),
                         face_rows=face_rows, zone=zone, carrier=carrier,
                         centre=centre, eval_row=float(y_e), edge_slope=float(b),
                         far_edge_x=float(far_x), far_edge_source=far_source, face_ends=face_ends)


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


def _profile_along(img, a, b, rows, k0):
    # Mean of the image along lines parallel to the edge x = a + b y, over ``rows``, at the
    # offsets k >= k0 px to its right that stay inside the image: (k, profile).
    y = np.arange(*rows, dtype=float)
    k1 = int(np.floor(img.shape[1] - 1 - (a + b * y).max()))
    if k1 <= k0 + 2:
        return np.array([]), np.array([])
    k = np.arange(k0, k1 + 1, dtype=float)
    x = a + b * y[:, None] + k[None, :]
    samples = map_coordinates(img, [np.broadcast_to(y[:, None], x.shape), x], order=1)
    return k, np.convolve(samples.mean(axis=0), np.ones(3) / 3, "same")


def _field_border(k, profile):
    # Offset at which the lit field ends: the profile falls below FIELD_BORDER_LEVEL of its
    # level near the edge and stays there for FIELD_BORDER_RUN px (nan if it never does).
    if profile.size < 100 + FIELD_BORDER_RUN:
        return float("nan")
    threshold = FIELD_BORDER_LEVEL * np.median(profile[:100])
    low = (profile < threshold).astype(int)
    starts = np.flatnonzero(np.convolve(low, np.ones(FIELD_BORDER_RUN, dtype=int), "valid") == FIELD_BORDER_RUN)
    if starts.size == 0:
        return float("nan")
    i = int(starts[0])
    if i == 0:
        return float(k[0])
    return float(k[i - 1] + (profile[i - 1] - threshold) / (profile[i - 1] - profile[i]))


def _far_edge(img, lines, a, b, y_e, rows, period):
    # Far lateral edge of the face at y_e, with the order of preference of the edge itself:
    # a phase-step line with the dark chamfer line, the dark line alone, a phase-step line;
    # else the end of the lit field. Only features inside the lit field compete.
    # Returns (x, source), or (nan, "none").
    edge_x = a + b * y_e
    k, profile = _profile_along(img, a, b, rows, 2 * BORDER_MARGIN)
    if profile.size == 0:
        return float("nan"), "none"
    border = _field_border(k, profile)
    limit = border if np.isfinite(border) else np.inf
    reach = max(REFINE_PX, REFINE_PERIODS * period)
    beyond = sorted(((la, lb) for la, lb, _ in lines if k[0] < la + lb * y_e - edge_x < limit),
                    key=lambda line: line[0] + line[1] * y_e)
    for la, lb in beyond:
        offset, depth = _dark_line_across(img, la, lb, rows, reach)
        if depth >= DARK_LINE_DEPTH:
            return float(la + lb * y_e + offset), "phase step and dark line"
    for c in argrelmin(profile, order=3)[0]:
        if k[c] < limit - 10 and _depth(profile, c) >= DARK_LINE_DEPTH:
            return float(edge_x + k[c] + parabolic_vertex(*profile[c - 1:c + 2])), "dark line"
    if beyond:
        la, lb = beyond[0]
        return float(la + lb * y_e), "phase step"
    if np.isfinite(border):
        return float(edge_x + border), "field border"
    return float("nan"), "none"


def _face_ends(phase_map, step, edge_x, x1, y_e, period):
    # Rows (top, bottom) of the ends of the face: peaks of the phase-step profile of a strip
    # of the face, from FACE_INSET px right of the edge to x1, that the platen strip (100 px
    # wide, FACE_INSET px left of the edge) does not share within one period (the border of
    # the beam crosses both), with fringes one period above and below. nan when not found.
    nan = float("nan")
    x0, x1 = int(round(edge_x + FACE_INSET)), int(round(min(x1, step.shape[1])))
    if x1 - x0 < 20:
        return nan, nan
    face = uniform_filter1d(step[:, x0:x1].mean(axis=1), 9)
    p1 = int(round(edge_x - FACE_INSET))
    p0 = max(0, p1 - 100)
    platen = uniform_filter1d(step[:, p0:p1].mean(axis=1), 9) if p1 - p0 >= 20 else None
    modulation = phase_map.modulation
    fringes = uniform_filter1d(np.nanmean(modulation[:, x0:x1], axis=1), 9)
    floor = END_MODULATION * np.nanpercentile(modulation, 95)
    guard = int(np.ceil(period))
    peaks, _ = find_peaks(face, prominence=END_PROMINENCE / period)
    ends = []
    for p in peaks:
        lo, hi = max(0, p - guard), min(len(face) - 1, p + guard)
        if platen is not None and platen[lo:hi + 1].max() > END_PLATEN_RATIO * face[p]:
            continue
        if min(fringes[lo], fringes[hi]) < floor:
            continue
        ends.append(p + (parabolic_vertex(*face[p - 1:p + 2]) if 0 < p < len(face) - 1 else 0.0))
    ends = np.asarray(ends, dtype=float)
    above, below = ends[ends < y_e], ends[ends > y_e]
    return (float(above.max()) if above.size else nan, float(below.min()) if below.size else nan)


@dataclass
class FaceCentre:
    """Centre of the measuring face in the normalised image of a :class:`BlockGeometry`.

    Attributes
    ----------
    y, x : float
        Row and column of the centre, px.
    u_y, u_x : float
        Standard uncertainties of ``y`` and ``x``, px (``nan`` for a manual centre given
        without one).
    source : str
        ``"outline"`` (both lateral edges and both ends detected), ``"outline and
        dimensions"`` (part of the outline and the face dimensions) or ``"manual"``.
    scale_px_per_mm : float
        Image scale on the object, px/mm (``nan`` when not used).
    scale_source : str
        ``"face width"``, ``"face length"``, ``"pixel size"`` or ``"none"``.
    features : dict
        The outline features found, ``name: (px, origin)``.
    aspect : tuple of float
        ``(measured, expected)`` length-to-width ratio of the face when the whole outline
        and the dimensions are known (a check of the features), else ``(nan, nan)``.
    notes : tuple of str
        Caveats: a far edge that is only the end of the lit field, a scale taken from the
        declared pixel size.
    """

    y: float
    x: float
    u_y: float
    u_x: float
    source: str
    scale_px_per_mm: float
    scale_source: str
    features: dict
    aspect: tuple
    notes: tuple


def locate_face_centre(geo, face_mm=None, u_face_mm=(0.0, 0.0), pixel_mm=None, manual=None, u_manual_px=None,
                       u_feature_px=FEATURE_U_PX):
    """Place the centre of the measuring face, where the central length is defined.

    The modes, in order: the whole outline found by :func:`locate_block`; part of it with
    the face dimensions; coordinates given by hand. ``manual`` overrides the other two.

    Parameters
    ----------
    geo : BlockGeometry
        Result of :func:`locate_block`.
    face_mm : tuple of float, optional
        ``(width, length)`` of the measuring face, mm: across the edge and along it.
    u_face_mm : tuple of float
        Standard uncertainties of ``face_mm``, mm.
    pixel_mm : float, optional
        Size of one pixel on the object, mm. Used for the scale only when the outline does
        not give it (no pair of opposite features), and with the dimensions to check a far
        edge without the dark line: the end of the lit field is used only if it agrees within
        ``FAR_EDGE_SCALE_TOL``; an unconfirmed phase-step line is dropped if it disagrees.
    manual : tuple of float, optional
        ``(y, x)`` of the centre in the **input** image, px.
    u_manual_px : float, optional
        Standard uncertainty of ``manual`` along each axis, px.
    u_feature_px : float
        Standard uncertainty of the position of each outline feature, px; for the ends, at
        least ``END_U_PERIODS`` fringe periods.

    Returns
    -------
    FaceCentre

    Raises
    ------
    ValueError
        If the outline is incomplete and no dimensions are given, or if not even the
        dimensions place the centre (no end of the face visible, no scale).
    """
    nan = float("nan")
    if manual is not None:
        y, x = (float(v) for v in geo.to_normalised(*manual))
        u = nan if u_manual_px is None else float(u_manual_px)
        return FaceCentre(y=y, x=x, u_y=u, u_x=u, source="manual", scale_px_per_mm=nan, scale_source="none",
                          features={}, aspect=(nan, nan), notes=())

    x_e, x_f = geo.edge_x, geo.far_edge_x
    y_t, y_b = geo.face_ends
    u_f = float(u_feature_px)
    u_end = max(u_f, END_U_PERIODS / max(np.hypot(*geo.carrier), 1e-6))
    has_far = bool(np.isfinite(x_f))
    far_confirmed = has_far and geo.far_edge_source != "field border"
    has_top, has_bottom = bool(np.isfinite(y_t)), bool(np.isfinite(y_b))
    features = {"edge": (x_e, geo.edge_source)}
    if has_far:
        features["far edge"] = (x_f, geo.far_edge_source)
    if has_top:
        features["top end"] = (y_t, "phase step")
    if has_bottom:
        features["bottom end"] = (y_b, "phase step")
    notes = []
    # A far edge without the dark line may lie off the block: the end of the lit field is the
    # border of the beam when the beam clips the block, and a phase-step line may be stray. With
    # the dimensions, the width it gives must agree with the declared pixel size; the end of the
    # lit field is used only then.
    if has_far and geo.far_edge_source in ("field border", "phase step") and face_mm is not None:
        what = "the end of the lit field" if geo.far_edge_source == "field border" else "an unconfirmed phase-step line"
        if pixel_mm:
            mismatch = (x_f - x_e) / float(face_mm[0]) * float(pixel_mm) - 1.0
            agrees = abs(mismatch) <= FAR_EDGE_SCALE_TOL
            notes.append(f"the far edge is {what}; the face width it gives is {100 * mismatch:+.1f} % off the "
                         f"declared pixel size: {'used' if agrees else 'not used'}")
        else:
            agrees = geo.far_edge_source == "phase step"
            if not agrees:
                notes.append(f"the far edge is {what}, which may be the border of the beam, and no pixel size "
                             "confirms it: not used")
        if not agrees:
            has_far = far_confirmed = False

    if face_mm is None:
        if not (far_confirmed and has_top and has_bottom):
            missing = [name for name, ok in (("far edge", far_confirmed), ("top end", has_top),
                                             ("bottom end", has_bottom)) if not ok]
            raise ValueError(f"the outline of the face is incomplete (no {', '.join(missing)}): "
                             "give the face dimensions or the centre")
        y, x = (y_t + y_b) / 2, (x_e + x_f) / 2
        u_y, u_x = u_end / np.sqrt(2), u_f / np.sqrt(2)
        scale, scale_source, aspect = nan, "none", (nan, nan)
    else:
        width, length = (float(v) for v in face_mm)
        u_width, u_length = (float(v) for v in u_face_mm)
        if has_far:
            span, dim, u_dim, u_span, scale_source = x_f - x_e, width, u_width, np.sqrt(2) * u_f, "face width"
        elif has_top and has_bottom:
            span, dim, u_dim, u_span, scale_source = y_b - y_t, length, u_length, np.sqrt(2) * u_end, "face length"
        else:
            span, scale_source = None, "pixel size"
        if span is not None:
            scale = span / dim
            u_scale = float(np.hypot(u_span / dim, span * u_dim / dim ** 2))
        elif pixel_mm:
            scale, u_scale = 1.0 / float(pixel_mm), 0.0
            notes.append("the scale is the declared pixel size, taken without uncertainty")
        else:
            raise ValueError("the outline gives no scale (no pair of opposite features): give pixel_mm")
        if has_far:
            x, u_x = (x_e + x_f) / 2, u_f / np.sqrt(2)
        else:
            x = x_e + scale * width / 2
            u_x = float(np.sqrt(u_f ** 2 + (width / 2 * u_scale) ** 2 + (scale / 2 * u_width) ** 2))
        if has_top and has_bottom:
            y, u_y = (y_t + y_b) / 2, u_end / np.sqrt(2)
        elif has_top or has_bottom:
            y = y_t + scale * length / 2 if has_top else y_b - scale * length / 2
            u_y = float(np.sqrt(u_end ** 2 + (length / 2 * u_scale) ** 2 + (scale / 2 * u_length) ** 2))
        else:
            raise ValueError("no end of the face is visible: the dimensions cannot place the centre along the edge; "
                             "give the centre")
        aspect = ((y_b - y_t) / (x_f - x_e), length / width) if has_far and has_top and has_bottom else (nan, nan)
    source = "outline" if far_confirmed and has_top and has_bottom else "outline and dimensions"
    x = x + geo.edge_slope * (y - geo.eval_row)             # the lateral edges at the centre row
    return FaceCentre(y=float(y), x=float(x), u_y=float(u_y), u_x=float(u_x), source=source,
                      scale_px_per_mm=float(scale), scale_source=scale_source, features=features,
                      aspect=(float(aspect[0]), float(aspect[1])), notes=tuple(notes))
