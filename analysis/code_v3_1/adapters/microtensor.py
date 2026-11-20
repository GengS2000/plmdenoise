"""microTensor (Ma & Li) through a non-interactive Rscript subprocess.

Estimand.  microTensor decomposes the multinomial *linear predictor* of a
feature x subject x time count tensor.  The relative composition is the
authors' own `phat`: a stable softmax of `get_Yhat()` over the feature mode,
copied from `R/helpers_evaluation.R`.  The adapter never touches the fit; it
only exports that quantity (see `microtensor_runner.R`).

Applicability domain.  The authors' time mode is a *shared discrete grid*: the
third tensor index is one time point for every subject.  Reading "visit k" as a
common time is only correct when the design really is a common grid, so a
dataset whose visit times differ across subjects is refused here rather than
quietly relabelled.  Missing visits stay NA across every feature, exactly as in
the authors' own simulation script (`rmd/1_Simulation_TrueMod.Rmd`).
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from r_process import run_r

import numpy as np

from external_method_contract import (ContractError, ExternalResult, MethodUnsupported,
                                      Timer, build_meta, f32, select_rank, truncate_rank_grid)
from adapters import tabular

NAME = "microtensor"
SOURCE_REPOSITORY = "https://github.com/syma-research/microTensor"
RUNNER = Path(__file__).with_name("microtensor_runner.R")

# Defaults follow the authors' own simulation study (rmd/1_Simulation_TrueMod.Rmd).
DEFAULTS = dict(
    rank=0,                # 0 = choose by the shared masked-sample CV
    rank_grid=[2, 3, 5],
    weighted=True,
    ortho_m=False,
    nn_t=False,
    init="ctf",
    maxit=5000,
    abs_tol=1e-4,
    rel_tol=1e-4,
    L_init=3,
    gamma=2,
    rscript="Rscript",
    timeout=7200,
    common_grid_tolerance=1e-6,
)


def require_common_grid(truth, tolerance):
    times = np.asarray(truth["times"], dtype=float)
    spread = np.nanmax(np.abs(times - times[0]), axis=0)
    if np.nanmax(spread) > tolerance * max(1.0, float(np.nanmax(np.abs(times)))):
        raise MethodUnsupported(
            "microTensor's third tensor mode is one shared time point per index; this dataset "
            f"has subject-specific visit times (max spread {float(np.nanmax(spread)):.4g}). "
            "Calling visit k a common time would change the estimand.")
    return times[0].tolist()


def _run(truth, keep, rank, settings, workdir, seed, tag):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    counts = workdir / f"counts_{tag}.tsv.gz"
    output = workdir / f"comp_{tag}.tsv.gz"
    diagnostics = workdir / f"diagnostics_{tag}.json"
    tabular.write_counts(counts, truth, keep)
    launcher = ([str(settings["r_binary"]), "--vanilla", "--slave", "-f", str(RUNNER), "--args"]
                if settings.get("r_binary") else [str(settings["rscript"]), str(RUNNER)])
    command = launcher + [
               "--input", str(counts), "--output", str(output),
               "--diagnostics", str(diagnostics), "--rank", str(int(rank)),
               "--weighted", str(bool(settings["weighted"])).upper(),
               "--ortho-m", str(bool(settings["ortho_m"])).upper(),
               "--nn-t", str(bool(settings["nn_t"])).upper(),
               "--init", str(settings["init"]), "--maxit", str(int(settings["maxit"])),
               "--abs-tol", repr(float(settings["abs_tol"])),
               "--rel-tol", repr(float(settings["rel_tol"])),
               "--L-init", repr(float(settings["L_init"])),
               "--gamma", repr(float(settings["gamma"])), "--seed", str(int(seed))]
    completed = run_r(command, cwd=str(workdir), capture_output=True, encoding="utf-8", errors="replace",
                               timeout=float(settings["timeout"]))
    (workdir / f"stdout_{tag}.txt").write_text(completed.stdout or "", encoding="utf-8")
    (workdir / f"stderr_{tag}.txt").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"microTensor Rscript exited {completed.returncode}; "
                           f"see {workdir / f'stderr_{tag}.txt'}\n{(completed.stderr or '')[-2000:]}")
    values = tabular.read_values(output, truth["Y"].shape)
    info = json.loads(diagnostics.read_text(encoding="utf-8"))
    return values, info, " ".join(command)


def _composition(values, observed):
    comp = np.full(values.shape, np.nan)
    comp[observed] = values[observed]
    total = comp[observed].sum(axis=-1, keepdims=True)
    if np.any(total <= 0) or not np.isfinite(comp[observed]).all():
        raise ContractError(f"{NAME}: phat is not a positive composition at some observed visit")
    comp[observed] = comp[observed] / total
    return comp


def fit(truth, task, settings, context):
    if not RUNNER.is_file():
        raise FileNotFoundError(f"missing R runner {RUNNER}")
    grid_times = require_common_grid(truth, float(settings["common_grid_tolerance"]))
    observed = np.asarray(truth["mask"]).all(axis=2)
    seed = int(context["seed"])
    workdir = Path(context["workdir"])
    grid = truncate_rank_grid(truth, settings.get("rank_grid") or DEFAULTS["rank_grid"])

    def fit_at_rank(rank, held):
        values, _, _ = _run(truth, observed & ~held, rank, settings, workdir, seed, f"cv_r{rank}")
        return _composition(values, observed)

    with Timer() as timer:
        if int(settings.get("rank", 0)):
            rank = int(settings["rank"])
            selection = dict(rank=rank, rule="fixed by settings", scores={}, failures={})
        else:
            selection = select_rank(fit_at_rank, truth, grid, seed)
            rank = selection["rank"]
        values, info, command = _run(truth, observed, rank, settings, workdir, seed, "final")
    comp = _composition(values, observed)
    meta = build_meta(
        NAME, method_version=f"microTensor {info['package_version']} (R {info['r_version']})",
        source_repository=SOURCE_REPOSITORY,
        source_commit_id=context.get("commits", {}).get("microtensor", "see ENV_INFO.txt"),
        settings=dict(settings, rank_grid=grid), rank=int(rank), seed=seed,
        wall_seconds=timer.wall, alloc_cpus=context.get("alloc_cpus"),
        cpu_hours_used=info.get("cpu_seconds", 0.0) / 3600.0,
        transform="multinomial linear predictor (authors' Yhat), no transform of the counts",
        inverse_transform="stable softmax over the feature mode (authors' helpers_evaluation.R phat)",
        status="converged" if info.get("converged") else "reached maxit",
        input_shape=truth["Y"].shape, output_shape=comp.shape,
        rank_selection=selection, command=command, r_diagnostics=info,
        time_representation="shared discrete visit grid", grid_times=grid_times,
        warnings=[] if info.get("converged") else
                 [f"components not converged: {info.get('convergence')}"])
    return ExternalResult(comp_mean=f32(comp), meta=meta,
                          warnings=list(meta["warnings"]))
