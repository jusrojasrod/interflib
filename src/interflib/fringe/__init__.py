"""L2 - fringe pattern and phase analysis.

Scope: phase-shifting algorithms, phase unwrapping, Zernike polynomial fitting and
surface topography.

Present (ported from the Excess_fractions prototype):

- ``models``      interference model and curve-fit quality metrics
- ``profile_fit`` 1D fringe-profile extraction, sinusoidal fit, fringe fraction

Pending: phase-shifting and Fourier back-ends for the same fringe-to-phase stage; the
unwrapping algorithm is a design decision to be recorded as an ADR before it is written.
"""
