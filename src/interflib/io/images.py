"""Interferogram image loading.

Reads a grayscale interferogram from disk in a Unicode-path-safe way (via
``cv2.imdecode`` over ``numpy.fromfile``, which tolerates non-ASCII paths that
``cv2.imread`` mishandles on Windows).

The image is returned at its native bit depth: an 8-bit file yields a ``uint8``
array and a 16-bit file (e.g. a 12-bit CMOS frame stored as 16-bit TIFF) yields a
``uint16`` array. Preserving the depth matters for phase/fraction estimation, which
would otherwise be limited by 8-bit quantisation.
"""
from pathlib import Path

import cv2
import numpy as np


def load_image(path) -> np.ndarray:
    """Load a grayscale interferogram from ``path`` at its native bit depth.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the image file.

    Returns
    -------
    numpy.ndarray
        The image as a 2-D grayscale array. The dtype matches the file's bit
        depth (``uint8`` for an 8-bit file, ``uint16`` for a 16-bit file); a
        multi-channel file is collapsed to grayscale without changing the depth.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    buffer = np.fromfile(path, np.uint8)
    # IMREAD_UNCHANGED keeps the native bit depth. IMREAD_GRAYSCALE would force
    # 8-bit and discard the 12 bits of a CMOS frame stored as a 16-bit TIFF.
    img = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"File found but could not be decoded as an image: {path}")
    if img.ndim == 3:
        # Collapse a multi-channel image to grayscale, preserving the bit depth.
        channels = img.shape[2]
        if channels == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = img[..., 0]
    return img
