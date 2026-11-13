# select_roi.py

A legacy helper from the first prototype. It opens one of the simulated interferograms in
`simulations/interferograms/` with OpenCV, scaled to at most 1000 px wide, lets you draw a
rectangle on it with the mouse (confirm with Enter), and prints the rectangle in pixels of
the full-size image, ready to paste as `img[y1:y2, x1:x2]`. The file name is fixed in the
script and it takes no arguments; its messages are in Spanish.

```bash
python tools/select_roi.py
```

It needs a desktop session, because it opens an OpenCV window. The region of real frames
is now found automatically (`interflib.fringe.geometry.locate_block`, used by
[`run_calibration.py`](run-calibration.md)), so this script is not part of the measurement
chain.
