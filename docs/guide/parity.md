# Parity of the reading

A wrong `order_sign` and a platen declared on the wrong side each turn the reading into
`1 − F`, and two wrong declarations cancel. Neither shows in the image (see
[Sign of the fraction](sign-convention.md)).

`check_fraction_parity` gives the deviation that the other parity would give, which is what
a wrong declaration would cost. It lies at least $(\lambda/2)\,|2F - \operatorname{round}(2F)|$
away: up to `λ/4` at `F = 0.25`, and nothing at `F = 0` or `0.5`. With a prior and its
standard uncertainty, such as the certificate of a check standard or a pre-measurement, it
also tests both readings. The verdict is `confirmed`, `contradicted`, `undecided` (both
fit) or `inconsistent` (neither fits); without a prior it is `unverified`. The values below
are the certified 2 mm block of the 2026-10 campaign, read at the face centre.

```python
from interflib.dimensional.parity import check_fraction_parity

check = check_fraction_parity(271.7, 51.3, 0.1416)    # lambda/2, d and F of the declared reading
print(f"{check.flipped_length_dev:+.1f} {check.verdict}")    # -> -25.6 unverified
check = check_fraction_parity(271.7, 51.3, 0.1416, prior_length_dev=51.0, prior_u=10.0, u_length=10.0)
print(f"{check.z:+.2f} {check.flipped_z:+.2f} {check.verdict}")   # -> +0.02 -5.42 confirmed
```

[`run_calibration.py`](../tools/run-calibration.md) always reports the length the other
parity would give, and tests both against the prior when one is given. The
[parity notebook](../notebooks/_generated/20261027-paridad-de-la-lectura.ipynb) shows the
check on simulated scenes and on the campaign series.
