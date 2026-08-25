"""Physical constants used across the project.

Rules
-----
No magic numbers anywhere else in the package. Every constant defined here carries
its source as a comment citing the actual reference (paper, standard or datasheet).

A constant without a citation is a defect. If a value is provisional, mark it
``[provisional]`` in the comment.
"""

# Standard reference temperature for dimensional metrology; gauge-block lengths are
# defined and reported at this temperature.  ISO 3650 (via ISO 1)
REFERENCE_TEMPERATURE_C = 20.0  # degrees Celsius
