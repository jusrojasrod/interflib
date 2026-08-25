"""First-order propagation of uncertainty following the GUM.

Combines standard uncertainties of the input quantities through the sensitivity
coefficients of the measurement model, accounting for declared correlations, and
computes the combined standard uncertainty, the effective degrees of freedom and
the expanded uncertainty for a chosen coverage factor.

Not implemented yet. JCGM 100 (the GUM) has not been fully worked through, so the exact
clauses this must comply with are not settled yet. Implementing before that would produce
untraceable code.
"""
