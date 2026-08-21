"""Synthesis of interferograms for testing the analysis pipelines.

Wiki: wiki/metodos/simulacion-interferograma.md

Scope: physically layered generation of Twyman-Green gauge-block interferograms
(optical phase -> image formation -> sensor), used as ground truth for the L1 excess-
fraction and L2 fringe-analysis pipelines.

The single-frame primitive :func:`synthesize_interferogram` carries the hooks
(``phase_shift_rad``, per-wavelength phase model) needed to later assemble phase-shifted
stacks and multi-wavelength sets without changing its physics.
"""
from interflib.simulation.interferogram import (
    Illumination,
    PhaseModel,
    SceneGeometry,
    SensorModel,
    SyntheticInterferogram,
    fringe_fraction_of_length,
    synthesize_interferogram,
)

__all__ = [
    "SceneGeometry",
    "PhaseModel",
    "Illumination",
    "SensorModel",
    "SyntheticInterferogram",
    "synthesize_interferogram",
    "fringe_fraction_of_length",
]
