"""L2 - fringe pattern and phase analysis.

Scope: from a fringe image (or a phase-stepped stack) to the block-platen fringe fraction,
the phase map and the block-platen wedge.

Present:

- ``models``         interference model, sine-fit seed and branch, fit quality, the
                     ``order_sign`` convention of the fraction
- ``profile_fit``    1D fringe-profile extraction, sinusoidal fit, fringe fraction
                     between two strips (single shot)
- ``phase_shifting`` phase stepping (PSI): wrapped-phase recovery from an N-frame stack
                     (general least-squares plus 3/4-step and Schwider-Hariharan
                     estimators), and the block-vs-platen fringe fraction from a phase map,
                     compared at the edge column or between strips
- ``unwrap``         2-D phase unwrapping: Itoh baseline, quality-guided flood fill and
                     the Goldstein branch-cut method (residue detection + branch cuts)
- ``edge_comparison`` single-shot fringe fraction with block and platen phases compared
                     at the edge column (column-resolved phase extrapolated to the edge),
                     which cancels any phase term common to both surfaces
- ``wedge``          block-platen wedge along x and y from the same comparison, in
                     fringes/px and, given the pixel scale, in microradians
- ``fourier``        single-shot Fourier (spatial-carrier) demodulation, along each column
                     or in two dimensions over a region on one surface (synchronous
                     detection), and the edge-column fringe fraction from either
- ``diagnostics``    inspection tools on top of ``fourier``: whole-field overview maps, the
                     block edge from the modulation dip, and the sensitivity of the
                     fraction to the filter width
"""
