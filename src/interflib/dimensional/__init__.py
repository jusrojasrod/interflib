"""L1 - dimensional laser interferometry.

Scope: fringe order determination, method of exact fractions, and the refractive
index of air (Edlen / Ciddor formulations) needed to convert optical path
difference into length traceable to the metre.

Modules:

- ``refraction``       refractive index of air (modified Edlen, Birch & Downs)
- ``multiwavelength``  classical multi-wavelength method of exact fractions
                       (MultiwavelengthEFSolver); resolves the integer order from two or
                       more wavelengths, the standard Michelson-Benoit / Decker approach
- ``coincidence``      the same method by brute-force comb coincidence (CoincidenceEFSolver,
                       Doiron & Beers / Tilford); symmetric and ambiguity-explicit, an
                       independent cross-check of ``multiwavelength``
- ``excess_fraction``  single-wavelength excess fraction (GaugeBlockLength); the air
                       index is delegated to ``refraction`` via ``from_environment``
- ``mefm``             monochromatic excess fraction solver: the single-wavelength
                       adaptation that resolves the order from a prior calibration bound
- ``parity``           the reading F against 1 - F (order direction and platen side
                       declared wrong): what a wrong parity gives, and its test against a prior
"""
