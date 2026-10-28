"""Parity of the fringe-fraction reading: ``F`` against ``1 - F``.

A single interferogram cannot tell which way the interference order runs, nor, from the
phase alone, which side of the edge is the platen. Each wrong declaration reverses the sign
of the block-platen phase difference, so the reading becomes ``1 - F``; two wrong
declarations cancel. Only their product, the *parity* of the reading, matters, and only a
value of the length known beforehand can test it.

The length is ``L = (kappa + F) lambda / 2`` (T. Doiron, J. Beers, "The Gage Block
Handbook", NIST Monograph 180, 1995, sec. 6.5), so each reading gives a comb of candidate
deviations ``lambda / 2`` apart, and the comb of ``1 - F`` is the comb of ``F`` shifted by
``(1 - 2 F) lambda / 2``. The order is resolved by taking, in each comb, the candidate
nearest the prior, as in the monochromatic method (J. S. Rojas-Rodriguez, J. L. Galvis
Arroyave and H. Vinck-Posada, "Absolute Gauge Block Calibration: A Single-Wavelength Excess
Fraction Method Assisted by High-Precision Mechanical Pre-measurement", preprint, INM
(2026), sec. III.C). The two candidates are at least ``(lambda / 2) |2 F - round(2 F)|``
apart: a wrong parity moves the result by at least that much, and a prior can only tell
the parities apart when its uncertainty is well below that separation.

The thermal reduction of the half-wave (parts in 10^6 of ``lambda / 2``) is neglected in the
shift between the combs.
"""
from dataclasses import dataclass

import numpy as np

from ._fractions import circular_distance

COVERAGE = 3.0   # |z| up to which a candidate is consistent with the prior


@dataclass
class ParityCheck:
    """Length from the declared reading and from the flipped one, and their test against a prior.

    Attributes
    ----------
    fraction, flipped_fraction : float
        Declared reading ``F`` and the reading of the other parity, ``1 - F``.
    length_dev, flipped_length_dev : float
        Candidate deviation nearest the prior in each comb, nm.
    shift : float
        ``flipped_length_dev - length_dev``: what a wrong parity would do to the result, nm.
    separation : float
        Smallest distance between the two combs, ``(lambda / 2) |2 F - round(2 F)|``, nm.
    z, flipped_z : float
        Gap of each candidate to the prior in standard uncertainties; ``nan`` without one.
    log10_likelihood_ratio : float
        Base-10 logarithm of the Gaussian likelihood of the declared parity over the flipped
        one (positive favours the declared); ``nan`` without a prior.
    verdict : str
        ``"unverified"`` (no prior uncertainty), ``"confirmed"`` (only the declared candidate
        is within ``coverage``), ``"contradicted"`` (only the flipped one is), ``"undecided"``
        (both are) or ``"inconsistent"`` (neither is).
    """

    fraction: float
    flipped_fraction: float
    length_dev: float
    flipped_length_dev: float
    shift: float
    separation: float
    z: float
    flipped_z: float
    log10_likelihood_ratio: float
    verdict: str


def _nearest(candidate, spacing, target):
    # Member of the comb candidate + m spacing closest to target.
    return candidate + spacing * np.round((target - candidate) / spacing)


def check_fraction_parity(half_wavelength, length_dev, fringe_fraction, prior_length_dev=0.0, prior_u=None,
                          u_length=0.0, coverage=COVERAGE) -> ParityCheck:
    """Compare the declared reading of the fraction with the flipped one, against a prior.

    Parameters
    ----------
    half_wavelength : float
        ``lambda / 2`` in the medium of the measurement, nm.
    length_dev : float
        A deviation from nominal obtained with the declared reading ``fringe_fraction``, at
        any integer order, nm.
    fringe_fraction : float
        Declared reading ``F``, in ``[0, 1)``.
    prior_length_dev : float
        Deviation known beforehand (calibration certificate of a check standard, mechanical
        pre-measurement), nm. The default 0 is the nominal length.
    prior_u : float, optional
        Standard uncertainty of that prior, nm. Without it the parity is not tested and the
        result only says what a wrong parity would give.
    u_length : float
        Standard uncertainty of the optical deviation itself, nm (it adds in quadrature to
        ``prior_u`` in the gaps).
    coverage : float
        Largest ``|z|`` for a candidate to be consistent with the prior.

    Returns
    -------
    ParityCheck
    """
    h = float(half_wavelength)
    fraction = float(fringe_fraction) % 1.0
    flipped = (1.0 - fraction) % 1.0
    declared = float(_nearest(length_dev, h, prior_length_dev))
    other = float(_nearest(length_dev + (1.0 - 2.0 * fraction) * h, h, prior_length_dev))
    separation = float(h * circular_distance(2.0 * fraction, 0.0))
    z = z_flip = log_ratio = float("nan")
    verdict = "unverified"
    if prior_u is not None:
        sigma = float(np.hypot(prior_u, u_length))
        if sigma <= 0:
            raise ValueError("the prior and the optical deviation need a positive combined uncertainty")
        z, z_flip = (declared - prior_length_dev) / sigma, (other - prior_length_dev) / sigma
        log_ratio = float((z_flip ** 2 - z ** 2) / 2.0 / np.log(10.0))
        ok, ok_flip = abs(z) <= coverage, abs(z_flip) <= coverage
        verdict = {(True, False): "confirmed", (False, True): "contradicted",
                   (True, True): "undecided", (False, False): "inconsistent"}[(ok, ok_flip)]
    return ParityCheck(fraction=fraction, flipped_fraction=flipped, length_dev=declared,
                       flipped_length_dev=other, shift=other - declared, separation=separation,
                       z=float(z), flipped_z=float(z_flip), log10_likelihood_ratio=log_ratio,
                       verdict=verdict)
