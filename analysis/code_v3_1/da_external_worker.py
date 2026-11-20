"""Run one existing denoiser on saved DA observations, in its original environment."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys


def main():
    snapshot = os.environ.get("DA_FROZEN_SOURCE")
    if snapshot:
        sys.path.insert(0, snapshot)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        from da_io import clean
        settings = dict(value=1)
        settings.update(dict(value=2))
        assert clean(settings) == {"value": 2}
        print("External worker compatibility passed:", sys.version, sys.executable)
        return
    if args.request is None:
        parser.error("--request is required unless --check-only is used")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    import numpy as np
    from adapters import load
    from external_method_contract import counts_fingerprint, validate_comp_mean, source_commit
    from da_io import atomic_json, atomic_npz, export_matrix

    method = request["method"]
    module = load(method)
    settings = dict(module.DEFAULTS)
    settings.update(request["settings"])
    with np.load(request["observed"], allow_pickle=False) as archive:
        # No DA labels or simulated latent abundance enter external adapters.
        observed = {key: archive[key] for key in ("Y", "mask", "depth", "times")}
    before = counts_fingerprint(observed)
    work = Path(request["workdir"])
    work.mkdir(parents=True, exist_ok=True)
    roots = {"gemelli": "gemelli", "microtensor": "microTensor",
             "embed": "EMBED", "parafac2": "parafac2_microbiome"}
    commits = {key: source_commit(Path(request["external_root"]) / value) for key, value in roots.items()}
    context = dict(seed=request["seed"], workdir=work, alloc_cpus=request["threads"], commits=commits)
    result = module.fit(observed, {}, settings, context)
    if counts_fingerprint(observed) != before:
        raise ValueError(f"{method} modified its observed input")
    validate_comp_mean(result.comp_mean, observed, method)
    export_matrix(work / "composition.csv", result.comp_mean)
    arrays = dict(result.arrays)
    arrays["composition_mean"] = result.comp_mean
    atomic_npz(work / "reconstruction.npz", arrays)
    packages = sorted({(dist.metadata.get("Name", "unknown"), dist.version)
                       for dist in importlib.metadata.distributions()})
    atomic_json(work / "diagnostics.json", dict(metadata=result.meta, warnings=result.warnings,
        settings=settings, observation_fingerprint=before, source_commits=commits,
        python=sys.version, executable=sys.executable, platform=platform.platform(), packages=packages))


if __name__ == "__main__":
    main()
