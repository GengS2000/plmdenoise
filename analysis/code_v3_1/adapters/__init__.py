"""Thin adapters around published longitudinal microbiome denoisers.

Every module here obeys `external_method_contract`: it runs the authors'
software unchanged (optionally through a recorded, additive patch that only
exports an already-computed quantity) and returns an `ExternalResult` whose
`comp_mean` is a full relative composition on the v3.1 grid.

An adapter module defines:

    NAME                 method name used in file names and tables
    SOURCE_REPOSITORY    author repository the fit comes from
    DEFAULTS             dict of settings; the store refuses mixed settings
    fit(truth, task, settings, context) -> ExternalResult

`context` carries the working directory, the seed and the allocated CPUs.
"""
from __future__ import annotations

import importlib

# method name -> module under this package
REGISTRY = {
    "tempted_gemelli": "tempted_gemelli",
    "microtensor": "microtensor",
    "embed": "embed",
    "parafac2": "parafac2",
    "mallard_pilot": "mallard",
}


def load(method):
    if method not in REGISTRY:
        raise KeyError(f"no adapter registered for {method!r}; have {sorted(REGISTRY)}")
    return importlib.import_module(f"{__name__}.{REGISTRY[method]}")


def defaults(method):
    """Settings dict without importing the method's heavy dependencies."""
    return dict(load(method).DEFAULTS)
