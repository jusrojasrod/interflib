"""interflib - analysis library for the INM interferometry project.

Submodules mirror the project's work lines:

- ``io``            file readers and ``.meta.yaml`` sidecar validation (data contract)
- ``preprocessing`` interferogram ROI, background flattening, denoising pipeline
- ``fringe``        L2: fringe -> phase/fraction (sinusoidal fit, edge comparison, Fourier,
                    phase stepping, unwrapping, diagnostics)
- ``dimensional``   L1: refractive index (Edlen/Birch-Downs), exact fractions (multi-wavelength,
                    coincidence, single wavelength, MEFM)
- ``simulation``    layered synthetic interferograms with ground truth
- ``vibration``     L3: heterodyne detection, fringe counting, spectral analysis
- ``uncertainty``   L4: GUM propagation, Monte Carlo, budget generation

The shared figure style is *not* part of the library: it lives outside the package
because report figures are not fringe analysis.

Each subpackage states its implementation status in its docstring; ``io.meta``,
``preprocessing``, ``vibration`` and ``uncertainty`` are still placeholders.
"""

__version__ = "0.1.0"
