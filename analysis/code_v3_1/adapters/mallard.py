"""MALLARD pilot (Silverman et al.) -- the authors' Stan model through Rscript.

This is deliberately a *pilot*, not a full-benchmark method.  MALLARD's state
space is the ILR of the whole composition: the observation covariance V is a
full (J-1) x (J-1) matrix with an LKJ prior, so the parameter count grows
quadratically in the number of taxa and a 500-taxon dataset is out of reach by
construction, not by tuning.  The pilot therefore runs on small datasets and
reports its own diagnostics; `mallard_runner.R` compiles the authors'
`toblerone.stan` unchanged.

Estimand.  `comp_mean` is the posterior mean of the *biological* state theta
mapped through the ILR inverse -- averaged on the composition scale, one draw
at a time.  The observation state eta, which still contains the technical noise
V, is not used.

Pre-registered success thresholds (a run that misses them fails; it is never
reported as a successful fit): no divergent transitions, R-hat <= 1.01 on theta
and the covariances, minimum bulk ESS >= 100, a complete composition for every
observed visit, and completion inside the job's wall-clock budget.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np

from external_method_contract import (ContractError, ExternalResult, MethodUnsupported,
                                      Timer, build_meta, f32)
from adapters import tabular
from adapters.microtensor import require_common_grid

NAME = "mallard_pilot"
SOURCE_REPOSITORY = "https://github.com/LAD-LAB/MALLARD-Paper-Code"
RUNNER = Path(__file__).with_name("mallard_runner.R")

DEFAULTS = dict(
    chains=4,
    iter=2000,
    warmup=1000,
    adapt_delta=0.9,
    max_treedepth=12,
    cores=1,
    stan_dir="",            # MALLARD-Paper-Code/mdlm_toblerone; required
    rscript="Rscript",
    timeout=86400,
    max_taxa=40,            # refuse sizes the model cannot carry (see docstring)
    max_divergences=0,
    max_rhat=1.01,
    min_ess=100.0,
)


class DiagnosticFailure(RuntimeError):
    """Sampling finished but missed a pre-registered threshold."""

    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


def fit(truth, task, settings, context):
    stan_dir = Path(str(settings["stan_dir"] or ""))
    if not (stan_dir / "toblerone.stan").is_file():
        raise FileNotFoundError(
            f"MALLARD Stan model not found: {stan_dir}/toblerone.stan "
            "(set settings stan_dir to MALLARD-Paper-Code/mdlm_toblerone)")
    if not RUNNER.is_file():
        raise FileNotFoundError(f"missing R runner {RUNNER}")
    I, T, J = truth["Y"].shape
    if J > int(settings["max_taxa"]):
        raise MethodUnsupported(
            f"MALLARD's observation covariance is a full {J - 1}x{J - 1} LKJ-prior matrix; "
            f"this dataset has {J} taxa, above the pilot limit of {settings['max_taxa']}. "
            "This is a structural limit of the published model, not a tuning choice.")
    require_common_grid(truth, 1e-6)
    observed = np.asarray(truth["mask"]).all(axis=2)
    seed = int(context["seed"])
    workdir = Path(context["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    counts = workdir / "counts.tsv.gz"
    output = workdir / "comp.tsv.gz"
    diagnostics = workdir / "diagnostics.json"
    tabular.write_counts(counts, truth, observed)
    command = [str(settings["rscript"]), str(RUNNER), "--input", str(counts),
               "--output", str(output), "--diagnostics", str(diagnostics),
               "--stan-dir", str(stan_dir.resolve()),
               "--chains", str(int(settings["chains"])), "--iter", str(int(settings["iter"])),
               "--warmup", str(int(settings["warmup"])),
               "--adapt-delta", repr(float(settings["adapt_delta"])),
               "--max-treedepth", str(int(settings["max_treedepth"])),
               "--cores", str(int(settings["cores"])), "--seed", str(seed)]
    with Timer() as timer:
        completed = subprocess.run(command, cwd=str(workdir), capture_output=True, encoding="utf-8", errors="replace",
                                   timeout=float(settings["timeout"]))
    (workdir / "stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
    (workdir / "stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"MALLARD Rscript exited {completed.returncode}; "
                           f"see {workdir / 'stderr.txt'}\n{(completed.stderr or '')[-2000:]}")
    info = json.loads(diagnostics.read_text(encoding="utf-8"))
    values = tabular.read_values(output, truth["Y"].shape)
    comp = np.full(values.shape, np.nan)
    block = values[observed]
    if not np.isfinite(block).all() or (block < 0).any():
        raise ContractError(f"{NAME}: posterior mean composition is not finite and non-negative")
    comp[observed] = block / block.sum(axis=-1, keepdims=True)

    thresholds = dict(max_divergences=int(settings["max_divergences"]),
                      max_rhat=float(settings["max_rhat"]), min_ess=float(settings["min_ess"]))
    failed = []
    if info.get("divergences", 0) > thresholds["max_divergences"]:
        failed.append(f"{info['divergences']} divergent transitions "
                      f"(threshold {thresholds['max_divergences']})")
    if not (info.get("rhat_max") is not None and info["rhat_max"] <= thresholds["max_rhat"]):
        failed.append(f"max R-hat {info.get('rhat_max')} > {thresholds['max_rhat']}")
    if not (info.get("ess_bulk_min") is not None and info["ess_bulk_min"] >= thresholds["min_ess"]):
        failed.append(f"min ESS {info.get('ess_bulk_min')} < {thresholds['min_ess']}")
    details = dict(diagnostics=info, thresholds=thresholds, command=" ".join(command),
                   wall_seconds=timer.wall, workdir=str(workdir))
    if failed:
        raise DiagnosticFailure(
            "MALLARD pilot missed its pre-registered thresholds: " + "; ".join(failed), details)

    meta = build_meta(
        NAME, method_version=f"MALLARD toblerone.stan via rstan {info['rstan_version']}",
        source_repository=SOURCE_REPOSITORY,
        source_commit_id=context.get("commits", {}).get("mallard", "see ENV_INFO.txt"),
        settings=dict(settings), rank=None, seed=seed, wall_seconds=timer.wall,
        alloc_cpus=context.get("alloc_cpus"), cpu_hours_used=info.get("cpu_seconds", 0) / 3600.0,
        transform="ILR of the composition (orthonormal basis); multinomial likelihood on counts",
        inverse_transform="ILR inverse of each theta draw, averaged on the composition scale",
        status="sampled, thresholds met", input_shape=truth["Y"].shape, output_shape=comp.shape,
        rank_selection=dict(rule="not a low-rank method; state dimension is J-1 by construction"),
        command=" ".join(command), stan_diagnostics=info, thresholds=thresholds,
        estimand_note="posterior mean of the biological state theta, not the observation state eta")
    return ExternalResult(comp_mean=f32(comp), meta=meta)
