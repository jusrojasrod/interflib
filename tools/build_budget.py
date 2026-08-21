#!/usr/bin/env python3
"""Generate an uncertainty budget table from the wiki.

Wiki: wiki/lineas/L4-incertidumbre.md

The uncertainty budget is a *database query*, not a document. Each contribution
lives as one page under ``wiki/incertidumbre/`` with machine-readable frontmatter
(see ``wiki/plantillas/incertidumbre.md``). This script reads those pages and emits
the budget. A budget written by hand in a deliverable is technical debt.

Functional specification
------------------------
1. Take ``--modelo <id>`` and read ``wiki/incertidumbre/modelos/<id>.md``.
2. For each id listed in the model's ``contribuciones:``, read the corresponding
   page and parse its frontmatter.
3. Validate: required fields present, units coherent, contributions declared in the
   model with no page of their own (and pages claiming a model that does not list
   them).
4. Compute the standard uncertainty by applying ``divisor`` according to
   ``distribucion``, apply ``coeficiente_sensibilidad``, and combine in quadrature
   taking ``correlaciones`` into account.
5. Emit a Markdown table: quantity, symbol, estimate, u(x_i), distribution, divisor,
   c_i, contribution u_i*c_i, percentage of the total, state, source.
6. Visibly flag every row whose page is in ``hipotesis`` or ``en-validacion``.
7. Write to ``entregables/<destino>/presupuesto-<id>.md`` with a header stating the
   generation date and a "generated file - do not edit by hand" warning.
8. State in that header that the budget is FIRST ORDER. For the L1 model the GUM
   itself warns that second-order terms are not negligible and raise the combined
   uncertainty by about 6 % [norma-jcgm-100-2008 H.1.7], so a first-order budget is
   a known underestimate and must not be presented as complete.
   See wiki/preguntas/terminos-segundo-orden-modelo-l1.md.

Status
------
Skeleton. The first measurement model now exists
(wiki/incertidumbre/modelos/modelo-bloque-patron-comparacion.md) with its six
contribution pages, so the frontmatter shape is real and the divisor table below is
sourced. What is still missing is the data: every contribution has ``valor: null``
pending its own certificate and characterisation, so there is nothing to compute.
The parser is written once the first values land.

Two shapes the parser must handle and that a naive implementation would get wrong:
a contribution whose ``distribucion`` is ``compuesta`` has no single divisor, and a
contribution whose ``coeficiente_sensibilidad`` evaluates to zero still belongs in
the table (it may carry a second-order term).

Usage
-----
    python code/tools/build_budget.py --modelo modelo-bloque-patron
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

WIKI = Path(__file__).resolve().parents[2] / "wiki"
DELIVERABLES = Path(__file__).resolve().parents[2] / "entregables"

# Divisor applied to convert a half-width `a` (or an expanded uncertainty) into a
# standard uncertainty, per distribution.
# Wiki: wiki/conceptos/evaluacion-tipo-a-y-tipo-b.md
#
# Sources, per entry:
#   rectangular  u = a/sqrt(3)   [norma-jcgm-100-2008 4.3.7, eq. 7]
#   triangular   u = a/sqrt(6)   [norma-jcgm-100-2008 4.3.9]
#   arcsine (U)  u = a/sqrt(2)   [norma-jcgm-100-2008 H.1.3.4] -- example only, not a clause
DIVISORS: dict[str, float] = {
    "rectangular": math.sqrt(3.0),
    "triangular": math.sqrt(6.0),
    "arcoseno": math.sqrt(2.0),
}

# Distributions whose divisor is not a constant and must be read from the page.
#   normal        u = U/k, or U_p/t_p(nu)  [norma-jcgm-100-2008 4.3.3, 4.3.4]
#   trapezoidal   u = a*sqrt((1+beta**2)/6), depends on the plateau ratio beta
#                                            [norma-jcgm-100-2008 4.3.9]
#   compuesta     several subcomponents with different distributions, combined in
#                 quadrature on the page itself; there is no single divisor
#                 [norma-jcgm-100-2008 H.1.3.4]
VARIABLE_DIVISOR = frozenset({"normal", "trapezoidal", "compuesta"})

# Asymmetric bounds: u = (b_plus + b_minus)/sqrt(12) [norma-jcgm-100-2008 4.3.8]
ASYMMETRIC_DIVISOR = math.sqrt(12.0)


def load_model(model_id: str) -> dict:
    """Read and parse ``wiki/incertidumbre/modelos/<model_id>.md``."""
    raise NotImplementedError("Pending: no measurement model exists yet.")


def load_contributions(ids: list[str]) -> list[dict]:
    """Read and parse the frontmatter of each contribution page."""
    raise NotImplementedError("Pending: no contribution page exists yet.")


def validate(model: dict, contributions: list[dict]) -> list[str]:
    """Return the list of contract violations found. Empty means valid."""
    raise NotImplementedError


def combine(model: dict, contributions: list[dict]) -> dict:
    """Combine standard uncertainties in quadrature, honouring correlations."""
    raise NotImplementedError


def render(model: dict, result: dict) -> str:
    """Render the budget as Markdown, flagging non-established rows."""
    raise NotImplementedError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modelo", required=True, help="measurement model id")
    parser.add_argument("--destino", default="informes", help="subfolder of entregables/")
    parser.add_argument("--dry-run", action="store_true", help="print instead of writing")
    args = parser.parse_args()

    model = load_model(args.modelo)
    contributions = load_contributions(model["contribuciones"])
    problems = validate(model, contributions)
    if problems:
        for p in problems:
            print(f"ERROR: {p}")
        return 1

    output = render(model, combine(model, contributions))
    if args.dry_run:
        print(output)
        return 0

    target = DELIVERABLES / args.destino / f"presupuesto-{args.modelo}.md"
    target.write_text(output, encoding="utf-8")
    print(f"written: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
