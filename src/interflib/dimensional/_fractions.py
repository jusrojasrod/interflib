"""Fringe-fraction arithmetic shared by the exact-fractions solvers."""
import numpy as np


def fractional_part(x):
    """Positive fractional part of ``x`` (matches Liu's INT() = floor for x > 0)."""
    return x - np.floor(x)


def circular_distance(a, b):
    """Shortest distance between two fringe fractions on the unit circle, in [0, 0.5].

    Fringe fractions live on a circle: 0.99 and 0.01 are 0.02 apart, not 0.98. This
    wraps the difference into [-0.5, 0.5) and returns its magnitude.
    """
    return np.abs(fractional_part(a - b + 0.5) - 0.5)
