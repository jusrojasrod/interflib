"""Validation of the ``.meta.yaml`` sidecar required by the data contract.

Every file in ``data/raw/`` must be named ``AAAAMMDD_LN_instrument_serie-NNN.ext``
and be accompanied by a sidecar of the same stem with the ``.meta.yaml`` extension,
carrying at minimum: ``fecha``, ``linea``, ``operador``, ``instrumento``,
``mensurando``, ``condiciones``, ``patron`` and ``metodo``.

Not implemented yet. No measurement data exists; the validator is written when the
first file arrives, so that it is checked against a real instrument output rather
than against an assumed one.
"""
