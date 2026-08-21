"""Interferogram preprocessing pipeline.

Wiki: wiki/index.md

Region-of-interest definition, background/illumination flattening, noise filtering
and a composable pipeline that prepares raw interferograms for fringe analysis. This
stage is shared by every fringe-to-phase method (sinusoidal fit, phase-shifting,
Fourier), which is why it is its own submodule rather than living inside ``fringe``.

Not implemented yet. Ported and refactored from the ``Exccess_fractions`` prototype in
a later integration step (see wiki/decisiones/ADR-004-arquitectura-interflib.md).
"""
