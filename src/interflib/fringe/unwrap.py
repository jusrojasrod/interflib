"""Two-dimensional phase unwrapping for demodulated interferogram phase maps.

Phase-stepping and profile fitting recover the phase *wrapped* into ``(-pi, pi]``. Turning
that back into a continuous surface (topography, wavefront figure, plane fitting) means
adding the missing integer multiples of ``2 pi`` -- unwrapping. The problem is only hard
where the wrapped gradient is unreliable: noise, undersampling or invalid pixels create
*residues*, points around which the unwrapped phase is path-dependent, so a naive scan
smears a local error across the whole field.

Three back-ends are provided, from cheapest to most robust:

* :func:`unwrap_itoh` -- Itoh's row/column integration. Exact on a clean, fully valid,
  well-sampled field; the baseline. It ignores residues, so noise or holes corrupt it.
* :func:`unwrap_quality_guided` -- flood fill that visits pixels in order of decreasing
  quality (the phase-stepping ``modulation`` map is the natural quality metric) and honours
  a validity ``mask``. Steering integration through the reliable pixels first keeps a bad
  region from poisoning the good ones, and the mask lets the block and the platen be
  unwrapped independently (their shared edge is a real height step, not a wrapping jump).
* :func:`unwrap_goldstein` -- Goldstein, Zebker & Werner's branch-cut method. It locates
  residues, joins balancing pairs (and the border) with branch cuts, and unwraps by a flood
  fill that may not cross a cut, so the path-dependence is confined to the cuts instead of
  leaking outward.

The framing follows fringe-analysis practice: path integration, quality guidance and the
handling of 2-D inconsistencies (Servin, Quiroga & Padilla, *Fringe Pattern Analysis for
Optical Metrology*, 2014, ch. 6; Ghiglia & Pritt, *Two-Dimensional Phase Unwrapping*, 1998).

Unwrapped phase is returned as a float array; pixels outside ``mask`` (and any the algorithm
could not reach) are ``nan``. No I/O side effects.
"""
import itertools
from collections import deque
from heapq import heappop, heappush

import numpy as np

TWO_PI = 2.0 * np.pi


def wrap_phase(phase) -> np.ndarray:
    """Wrap phase into ``(-pi, pi]``."""
    return np.mod(np.asarray(phase, dtype=float) + np.pi, TWO_PI) - np.pi


def _full_mask(shape, mask):
    if mask is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} does not match phase shape {shape}.")
    return mask


def find_residues(phase, mask=None) -> np.ndarray:
    """Residue charge of every 2x2 loop, as an ``(H-1, W-1)`` integer array.

    Each entry is the sum of the four wrapped gradients around the loop whose top-left
    corner is that pixel, divided by ``2 pi`` and rounded: ``0`` where the field is locally
    consistent, ``+-1`` at a residue. A clean, well-sampled field has none. With a ``mask``,
    loops touching an invalid pixel are reported as ``0``.
    """
    phase = np.asarray(phase, dtype=float)
    top = wrap_phase(phase[:-1, 1:] - phase[:-1, :-1])
    right = wrap_phase(phase[1:, 1:] - phase[:-1, 1:])
    bottom = wrap_phase(phase[1:, :-1] - phase[1:, 1:])
    left = wrap_phase(phase[:-1, :-1] - phase[1:, :-1])
    residues = np.round((top + right + bottom + left) / TWO_PI).astype(int)

    if mask is not None:
        mask = _full_mask(phase.shape, mask)
        loop_valid = (mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, 1:] & mask[1:, :-1])
        residues[~loop_valid] = 0
    return residues


def unwrap_itoh(phase) -> np.ndarray:
    """Itoh row/column unwrap: integrate along rows, then down columns.

    Exact on a clean, fully valid, well-sampled field; the baseline. It follows a fixed
    scan and so cannot avoid residues -- use :func:`unwrap_quality_guided` or
    :func:`unwrap_goldstein` for noisy or masked data.
    """
    phase = np.asarray(phase, dtype=float)
    return np.unwrap(np.unwrap(phase, axis=1), axis=0)


_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _flood_fill_unwrap(phase, allowed, quality=None) -> np.ndarray:
    """Path-following unwrap by flood fill over ``allowed`` pixels.

    Pixels are added in order of decreasing ``quality`` (uniform if ``None``), each unwrapped
    against the already-unwrapped neighbour it was reached from. Disconnected regions of
    ``allowed`` are each seeded independently, so every region is defined up to its own
    piston.
    """
    phase = np.asarray(phase, dtype=float)
    height, width = phase.shape
    if quality is None:
        quality = np.zeros((height, width))
    else:
        quality = np.asarray(quality, dtype=float)

    unwrapped = np.full((height, width), np.nan)
    visited = np.zeros((height, width), dtype=bool)
    counter = itertools.count()
    remaining = set(map(tuple, np.argwhere(allowed)))

    def push_neighbours(heap, i, j):
        for di, dj in _NEIGHBOURS:
            ni, nj = i + di, j + dj
            if 0 <= ni < height and 0 <= nj < width and allowed[ni, nj] \
                    and not visited[ni, nj]:
                heappush(heap, (-quality[ni, nj], next(counter), ni, nj, i, j))

    while remaining:
        si, sj = max(remaining, key=lambda p: quality[p])
        unwrapped[si, sj] = phase[si, sj]
        visited[si, sj] = True
        remaining.discard((si, sj))
        heap = []
        push_neighbours(heap, si, sj)
        while heap:
            _, _, i, j, ri, rj = heappop(heap)
            if visited[i, j]:
                continue
            unwrapped[i, j] = unwrapped[ri, rj] + wrap_phase(phase[i, j] - phase[ri, rj])
            visited[i, j] = True
            remaining.discard((i, j))
            push_neighbours(heap, i, j)
    return unwrapped


def unwrap_quality_guided(phase, quality=None, mask=None) -> np.ndarray:
    """Quality-guided flood-fill unwrap.

    Parameters
    ----------
    phase : array_like
        Wrapped phase map, radians.
    quality : array_like, optional
        Per-pixel reliability; higher is unwrapped first. The phase-stepping ``modulation``
        map is the natural choice. Uniform if omitted (arbitrary order, still exact on clean
        data).
    mask : array_like of bool, optional
        Pixels to unwrap. Disconnected regions are unwrapped independently, so passing the
        block or the platen footprint keeps their real height step out of the integration.

    Returns
    -------
    numpy.ndarray
        Unwrapped phase; pixels outside ``mask`` are ``nan``.
    """
    phase = np.asarray(phase, dtype=float)
    allowed = _full_mask(phase.shape, mask)
    return _flood_fill_unwrap(phase, allowed, quality)


# --------------------------------------------------------------------------------------
# Goldstein branch-cut method
# --------------------------------------------------------------------------------------
def _draw_line(cuts, r0, c0, r1, c1):
    """Mark a Bresenham line between two nodes as branch-cut pixels (endpoints included)."""
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        cuts[r, c] = True
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc


def _cut_to_border(cuts, r, c, height, width):
    """Connect a residue to the nearest image border with a straight cut."""
    distances = {"top": r, "bottom": height - 1 - r, "left": c, "right": width - 1 - c}
    side = min(distances, key=distances.get)
    if side == "top":
        _draw_line(cuts, r, c, 0, c)
    elif side == "bottom":
        _draw_line(cuts, r, c, height - 1, c)
    elif side == "left":
        _draw_line(cuts, r, c, r, 0)
    else:
        _draw_line(cuts, r, c, r, width - 1)


def _place_branch_cuts(residues, height, width) -> np.ndarray:
    """Goldstein box-growing: join residues into charge-balanced groups with cuts.

    Residues are addressed at the pixel of their loop's top-left corner. Each unbalanced
    residue grows a search box; residues found inside are joined by a cut and their charge
    added, and a box that reaches the border ties the group to it (the border is a charge
    sink). The group closes once its net charge is zero.
    """
    cuts = np.zeros((height, width), dtype=bool)
    charged = [(int(i), int(j), int(residues[i, j]))
               for i, j in np.argwhere(residues != 0)]
    charge = {(i, j): q for i, j, q in charged}
    balanced = set()

    for start_i, start_j, start_q in charged:
        if (start_i, start_j) in balanced:
            continue
        group = [(start_i, start_j)]
        group_charge = start_q
        box = 1
        while group_charge != 0:
            progressed = False
            for ci, cj in list(group):
                if ci - box < 0 or ci + box > height - 1 \
                        or cj - box < 0 or cj + box > width - 1:
                    _cut_to_border(cuts, ci, cj, height, width)
                    group_charge = 0
                    progressed = True
                    break
                for si, sj, sq in charged:
                    if (si, sj) in group or (si, sj) in balanced:
                        continue
                    if abs(si - ci) <= box and abs(sj - cj) <= box:
                        _draw_line(cuts, ci, cj, si, sj)
                        group.append((si, sj))
                        group_charge += sq
                        progressed = True
                        break
                if progressed:
                    break
            if not progressed:
                box += 1
                if box > height + width:            # safety: tie the group to the border
                    ci, cj = group[0]
                    _cut_to_border(cuts, ci, cj, height, width)
                    group_charge = 0
        balanced.update(group)
    return cuts


def _fill_barrier_pixels(phase, unwrapped, valid):
    """Unwrap valid pixels left on a branch cut from any already-unwrapped neighbour."""
    height, width = phase.shape
    pending = deque(map(tuple, np.argwhere(valid & np.isnan(unwrapped))))
    stalled = 0
    while pending and stalled < len(pending):
        i, j = pending.popleft()
        for di, dj in _NEIGHBOURS:
            ni, nj = i + di, j + dj
            if 0 <= ni < height and 0 <= nj < width and not np.isnan(unwrapped[ni, nj]):
                unwrapped[i, j] = unwrapped[ni, nj] + wrap_phase(phase[i, j] - phase[ni, nj])
                break
        if np.isnan(unwrapped[i, j]):
            pending.append((i, j))
            stalled += 1
        else:
            stalled = 0
    return unwrapped


def unwrap_goldstein(phase, mask=None, return_cuts=False):
    """Goldstein branch-cut unwrap.

    Locates residues, balances them with branch cuts (:func:`_place_branch_cuts`), and
    unwraps by a flood fill that treats cuts (and invalid pixels) as barriers, confining any
    path-dependence to the cuts (Goldstein, Zebker & Werner, *Radio Sci.* 23, 713, 1988).

    Parameters
    ----------
    phase : array_like
        Wrapped phase map, radians.
    mask : array_like of bool, optional
        Valid pixels; invalid pixels act as barriers.
    return_cuts : bool, optional
        If true, also return the boolean branch-cut map.

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, numpy.ndarray)
        Unwrapped phase (``nan`` outside ``mask``), and the branch-cut map if requested.
    """
    phase = np.asarray(phase, dtype=float)
    height, width = phase.shape
    valid = _full_mask(phase.shape, mask)

    residues = find_residues(phase, mask)
    cuts = _place_branch_cuts(residues, height, width)

    allowed = valid & ~cuts
    unwrapped = _flood_fill_unwrap(phase, allowed, quality=None)
    unwrapped = _fill_barrier_pixels(phase, unwrapped, valid)

    if return_cuts:
        return unwrapped, cuts
    return unwrapped
