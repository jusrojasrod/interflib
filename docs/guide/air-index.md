# Refractive index of air

The fraction is read at the wavelength in air, $\lambda = \lambda_\text{vac}/n$.
`EdlenRefractiveIndex` computes the index $n$ with the modified Edlén equation of Birch &
Downs (1993, 1994), in the form and with the input limits of the NIST Engineering
Metrology Toolbox (Stone & Zimmerman). It reproduces that document's reference table to
within 4·10⁻¹⁰.

```python
from interflib.dimensional.refraction import EdlenRefractiveIndex

air = EdlenRefractiveIndex(temperature=20.0, pressure=75_300.0, relative_humidity=50.0)
lambda_air = air.wavelength_in_air(543.5163)     # vacuum wavelength in, nm
print(f"{air.index_refraction(543.5163) - 1:.4e} {lambda_air:.4f}")   # -> 2.0260e-04 543.4062
```

```{figure} ../images/air_index.png
:alt: Refractive index of air against pressure and humidity

`n − 1` at 543.5163 nm. Pressure dominates: (a) it is nearly proportional to it, and a few kelvin move it less. (b) Humidity lowers it slightly. The dot is the example above.
```

The inputs are the air temperature in °C, the pressure in Pa and one humidity state: the
relative humidity in %, the dew point or the frost point. An input outside the range the calculation accepts raises `EdlenInputBoundsError`
instead of being extrapolated. A pressure outside 60–120 kPa or a humidity above 85 % only
warns, and so does a wavelength outside the 350–650 nm range of Birch & Downs.

The solvers of [Exact fractions](exact-fractions.md) take the wavelength in air. The
single-wavelength solver can also take the vacuum wavelength and the conditions, and
compute the index itself (`GaugeBlockLength.from_environment`). The package keeps one
implementation of the air index.
