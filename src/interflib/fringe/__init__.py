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
- ``edge_comparison`` single-shot fringe fraction with block and platen phases compared
                     at the edge column (column-resolved phase extrapolated to the edge),
                     which cancels any phase term common to both surfaces
- ``wedge``          block-platen wedge along x and y from the same comparison, in
                     fringes/px and, given the pixel scale, in microradians
- ``fourier``        single-shot Fourier (spatial-carrier) demodulation along each column
                     and the edge-column fringe fraction from the demodulated phases
"""
