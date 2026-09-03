"""L2 - fringe pattern and phase analysis.

Scope: phase-shifting algorithms, phase unwrapping, Zernike polynomial fitting and
surface topography.

Present:

- ``models``         interference model and curve-fit quality metrics
- ``profile_fit``    1D fringe-profile extraction, sinusoidal fit, fringe fraction
                     (single-shot, ported from the Excess_fractions prototype)
- ``phase_shifting`` phase stepping (PSI): wrapped-phase recovery from an N-frame stack
                     (general least-squares plus 3/4-step and Schwider-Hariharan
                     estimators), and the block-vs-platen fringe fraction from a phase map
- ``unwrap``         2-D phase unwrapping: Itoh baseline, quality-guided flood fill and
                     the Goldstein branch-cut method (residue detection + branch cuts)

Pending: the Fourier (single-shot spatial-carrier) back-end for the same fringe-to-phase
stage.
"""
