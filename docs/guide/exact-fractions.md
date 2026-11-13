# Exact fractions

An image shows only the fraction $F$ of the interference order $2L/\lambda = \kappa + F$.
The integer order $\kappa$ has to come from elsewhere: from other wavelengths, from a
mechanical pre-measurement or from the nominal length. `interflib.dimensional` has four
solvers for it, all with lengths and wavelengths in nanometres and the wavelength in air
(see [Refractive index of air](air-index.md)).

| Solver | Prior | Use |
|---|---|---|
| `MultiwavelengthEFSolver` | nominal length and tolerance | the standard method with two or more wavelengths |
| `CoincidenceEFSolver` | nominal length and tolerance | the same, by comb coincidence; an independent cross-check |
| `MonochromaticEFMSolver` (MEFM) | a mechanical deviation and its uncertainty | one wavelength |
| `GaugeBlockLength` | the nominal length | one wavelength, block within `λ/4` of nominal |

## Several wavelengths

Each wavelength allows a comb of lengths spaced `λ/2` apart; only one length is
consistent with all of them. `MultiwavelengthEFSolver` parametrises the length by its
correction to the nominal (Liu et al. 2022), anchors the sub-fringe position on the first
wavelength and picks the integer orders that minimise the residual. It also reports the
`margin` to the runner-up, which warns of an ambiguous choice. `CoincidenceEFSolver`
scans the combs directly (Doiron & Beers 1995, §6.8) and serves as an independent
cross-check. The example is the four-wavelength measurement of a 1400 µm block published
by Liu et al. (2022).

```python
from interflib.dimensional.coincidence import CoincidenceEFSolver
from interflib.dimensional.multiwavelength import MultiwavelengthEFSolver

wavelengths = [667.8184, 587.5649, 501.5702, 447.1501]   # in air, nm; the first is the anchor
fractions = [0.28, 0.85, 0.90, 0.23]

multi = MultiwavelengthEFSolver(wavelengths).solve(
    nominal_length=1_400_000.0, nominal_tolerance=1_000.0, fringe_fractions=fractions)
print(f"{multi.length / 1000:.6f} {multi.integer_orders}")   # -> 1399.840861 [4192, 4764, 5581, 6261]

comb = CoincidenceEFSolver(wavelengths).solve(1_400_000.0, 1_000.0, fractions)
print(f"{comb.length / 1000:.4f} {comb.is_unique}")          # -> 1399.8481 True
```

The two solvers agree on the integer orders. Their lengths differ by 7 nm because each
places the length within the fringe differently:

- `CoincidenceEFSolver` averages the nearest comb line of every wavelength, weighted by
  `(2/λ)²`.
- `MultiwavelengthEFSolver` takes the position within the fringe from the first wavelength.

The spread reflects how consistent the published fractions are, which are given to two
decimals only.

```{image} ../images/exact_fractions.png
:alt: Combs of candidate lengths of four wavelengths and their coincidence
```

## One wavelength and a mechanical prior (MEFM)

`MonochromaticEFMSolver` picks the only candidate that falls within `δ ± U` of the
mechanically measured deviation (Rojas-Rodríguez, Galvis Arroyave & Vinck-Posada 2026).
For the candidate to be unique, `U` must stay below `λ/4`. A certificate gives the
deviation at 20 °C while the fraction is measured at the block temperature: with `cte` and
`temp_c` the solver expands the nominal to that temperature and reduces each candidate to
20 °C before testing it. Without them it is only right at 20 °C; a 100 mm steel block moves
115 nm per 0.1 K. The fraction below is the one read on the [edge comparison](edge-comparison.md)
page from the simulated 2 mm block, 51 nm long.

```python
from interflib.dimensional.mefm import MonochromaticEFMSolver

fraction = 0.8934                                  # read at 543.5 nm on the simulated block

mefm = MonochromaticEFMSolver(wavelength=543.5)
block = mefm.solve(nominal_length=2e6,             # L', nm
                   prior_length_dev=55.0,          # mechanical deviation from nominal, nm
                   prior_U=30.0,                   # its expanded uncertainty, nm
                   fringe_fraction=fraction,
                   cte=11.5e-6, temp_c=20.0)       # block expansion, 1/K, and temperature, C
# deviation from nominal accepted by the optics, nm
print(f"{block.length_dev:.1f}")                    # -> 51.0
```

```{figure} ../images/mefm.png
:alt: Candidate lengths of one wavelength and the two priors

With one wavelength the fraction allows a comb of deviations spaced `λ/2` apart (grey lines). The nominal prior of `GaugeBlockLength` keeps the candidate within `λ/4` of the nominal (blue); the mechanical prior of MEFM keeps the one within `δ ± U` (yellow). Both keep `d = 51.0 nm` here; the mechanical prior also works when the block is further than `λ/4` from its nominal.
```

## One wavelength, nominal length as prior

`GaugeBlockLength` is the classical single-wavelength computation (Doiron & Beers 1995,
§6.5). It works when the block is within `λ/4` of its nominal length, and it reduces the
result to 20 °C. `GaugeBlockLength.from_environment` takes the vacuum wavelength and the
air conditions, and computes the air index itself.

```python
from interflib.dimensional.excess_fraction import GaugeBlockLength

classic = GaugeBlockLength(wavelength=543.5, nominal_length=2e6, cte=11.5e-6, temp_c=20.0,
                           fringe_fraction=0.8934).solve()
# deviation from nominal, nm
print(f"{classic.length_dev:.1f}")                  # -> 51.0
```

With one wavelength the result also depends on the parity of the reading: see
[Parity of the reading](parity.md).
