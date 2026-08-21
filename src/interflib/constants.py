"""Physical constants used across the project.

Wiki: wiki/nomenclatura.md

Rules
-----
No magic numbers anywhere else in the package. Every constant defined here carries
its source as a comment using the wiki citation format ``[source-slug section]``.

A constant without a citation is a defect. If a value is provisional, mark it
``[sin fuente - hipotesis de trabajo]`` and open a page under ``wiki/preguntas/``.
"""

# Standard reference temperature for dimensional metrology; gauge-block lengths are
# defined and reported at this temperature.  [iso-3650] (via ISO 1)
REFERENCE_TEMPERATURE_C = 20.0  # degrees Celsius
