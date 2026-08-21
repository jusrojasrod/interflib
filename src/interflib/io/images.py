"""Interferogram image loading.

Wiki: data/CONTRATO.md

Reads a grayscale interferogram from disk in a Unicode-path-safe way (via
``cv2.imdecode`` over ``numpy.fromfile``, which tolerates non-ASCII paths that
``cv2.imread`` mishandles on Windows).
"""
from pathlib import Path

import cv2
import numpy as np


def load_image(path) -> np.ndarray:
    """Load a grayscale interferogram from ``path``.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the image file.

    Returns
    -------
    numpy.ndarray
        The image as a 2-D ``uint8`` grayscale array.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    buffer = np.fromfile(path, np.uint8)
    img = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"File found but could not be decoded as an image: {path}")
    return img
