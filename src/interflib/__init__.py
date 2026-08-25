"""interflib - analysis library for the INM interferometry project.

Submodules mirror the project's work lines:

- ``io``            file readers and ``.meta.yaml`` sidecar validation (data contract)
- ``preprocessing`` interferogram ROI, background flattening, denoising pipeline
- ``fringe``        L2: fringe -> phase/fraction (sinusoidal fit, phase-shifting, Fourier)
- ``dimensional``   L1: refractive index (Edlen/Birch-Downs), excess fraction, MEFM
- ``vibration``     L3: heterodyne detection, fringe counting, spectral analysis
- ``uncertainty``   L4: GUM propagation, Monte Carlo, budget generation

The shared figure style is *not* part of the library: it lives outside the package
because report figures are not fringe analysis.

Being ported from the ``Exccess_fractions`` prototype one method at a time; each module
carries its own implementation status in its docstring.
"""

__version__ = "0.1.0"
