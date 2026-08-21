"""First-order propagation of uncertainty following the GUM.

Wiki: wiki/lineas/L4-incertidumbre.md

Combines standard uncertainties of the input quantities through the sensitivity
coefficients of the measurement model, accounting for declared correlations, and
computes the combined standard uncertainty, the effective degrees of freedom and
the expanded uncertainty for a chosen coverage factor.

Not implemented yet. JCGM 100 has not been ingested, so the exact clauses this must
comply with are not yet recorded in the wiki. Implementing before that would produce
untraceable code.
"""
