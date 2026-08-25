"""Instrument file readers and data-contract validation.

Responsibilities
----------------
- Read raw instrument formats into tidy structures.
- Validate that every file in ``data/raw/`` has a well-formed ``.meta.yaml``
  sidecar before any processing happens.

A file without a valid sidecar is not processed.

Modules: ``images`` (interferogram loading, Unicode-path-safe) and ``meta``
(``.meta.yaml`` sidecar validation, a stub until the first instrument file arrives).
"""
