"""L1 - dimensional laser interferometry.

Wiki: wiki/lineas/L1-dimensional.md

Scope: fringe order determination, method of exact fractions, and the refractive
index of air (Edlen / Ciddor formulations) needed to convert optical path
difference into length traceable to the metre.

Modules (ported from the Excess_fractions prototype; internal refinement pending per
wiki/decisiones/ADR-004-arquitectura-interflib.md):

- ``refraction``       refractive index of air (modified Edlen, Birch & Downs)
- ``excess_fraction``  single-wavelength excess fraction (GaugeBlockLength); the air
                       index is delegated to ``refraction`` via ``from_environment``
- ``mefm``             monochromatic excess fraction solver with mechanical priors
"""
