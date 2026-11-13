"""Physical constants used across the project.

Rules
-----
No magic numbers anywhere else in the package. Every constant defined here carries
its source as a comment citing the actual reference (paper, standard or datasheet).

A constant without a citation is a defect. If a value is provisional, mark it
``[provisional]`` in the comment.

References
----------
[NIST] J. A. Stone and J. H. Zimmerman, "Index of Refraction of Air", NIST
Engineering Metrology Toolbox, https://emtoolbox.nist.gov/Wavelength/Documentation.asp
(Appendix A).

[BD] K. P. Birch and M. J. Downs, "An updated Edlen equation for the refractive index
of air", Metrologia 30, 155-162 (1993); correction, Metrologia 31, 315-316 (1994).
"""

# Standard reference temperature for dimensional metrology; gauge-block lengths are
# defined and reported at this temperature.  ISO 3650 (via ISO 1)
REFERENCE_TEMPERATURE_C = 20.0  # degrees Celsius

# Offset between the Celsius and Kelvin scales.  [NIST] eq. (A2), (A10)
CELSIUS_TO_KELVIN = 273.15  # K

# --------------------------------------------------------------------------------------
# Modified Edlen equation for the refractive index of air  [BD]; as summarised in
# [NIST] Appendix A-IV, eq. (A42)-(A49), with sigma = 1/lambda^2 in um^-2.
# --------------------------------------------------------------------------------------
# Dispersion of standard air: (n - 1)_s * 1e8 = A + B / (130 - sigma) + C / (38.9 - sigma)
EDLEN_A = 8342.54
EDLEN_B = 2406147.0
EDLEN_C = 15998.0
EDLEN_DISPERSION_POLE_1 = 130.0   # um^-2
EDLEN_DISPERSION_POLE_2 = 38.9    # um^-2
# Density factor: (n - 1)_tp = p (n - 1)_s X / D,  X = (1 + 1e-8 (E - F t) p) / (1 + G t)
EDLEN_D = 96095.43                # Pa
EDLEN_E = 0.601
EDLEN_F = 0.00972                 # 1/degC
EDLEN_G = 0.003661                # 1/degC
# Water vapour: n = n_tp - 1e-10 (T_ref / T) (H1 - H2 sigma) p_v.  The factor T_ref / T,
# with T_ref = 292.75 K, is the NIST modification of the Birch & Downs vapour term
# (fitted at 19.6 C) that extends it to other temperatures.  [NIST] eq. (A49)
EDLEN_VAPOUR_H1 = 3.7345
EDLEN_VAPOUR_H2 = 0.0401          # um^2
EDLEN_VAPOUR_T_REF = 292.75       # K

# --------------------------------------------------------------------------------------
# Saturation vapour pressure (IAPWS)  [NIST] Appendix A-I
# --------------------------------------------------------------------------------------
# Over water, eq. (A1)-(A7), coefficients K1..K10 of Table A1 (from P. H. Huang, 1998)
IAPWS_WATER_K = (
    1.16705214528e3, -7.24213167032e5, -1.70738469401e1, 1.20208247025e4,
    -3.23255503223e6, 1.49151086135e1, -4.82326573616e3, 4.05113405421e5,
    -2.38555575678e-1, 6.50175348448e2,
)
# Over ice, eq. (A8)-(A13)
IAPWS_ICE_A1 = -13.928169
IAPWS_ICE_A2 = 34.7078238
WATER_TRIPLE_POINT_K = 273.16          # K, eq. (A11)
WATER_TRIPLE_POINT_PRESSURE_PA = 611.657  # Pa, eq. (A13)

# --------------------------------------------------------------------------------------
# Input limits of the refractive-index calculation  [NIST] section IV
# --------------------------------------------------------------------------------------
AIR_TEMPERATURE_RANGE_C = (-40.0, 100.0)        # accepted range (Ciddor's validity)
AIR_PRESSURE_RANGE_PA = (10e3, 140e3)           # accepted range
AIR_PRESSURE_REALISTIC_PA = (60e3, 120e3)       # warning outside this range
AIR_HUMIDITY_CONDENSATION_PCT = 85.0            # warning above: risk of condensation
VACUUM_WAVELENGTH_RANGE_NM = (300.0, 1700.0)    # accepted range (Ciddor's validity)
EDLEN_WAVELENGTH_RANGE_NM = (350.0, 650.0)      # range claimed by Birch & Downs
