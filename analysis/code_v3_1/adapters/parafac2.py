"""PARAFAC2 as applied to microbiome data by Serdos et al. (parafac2_microbiome).

Everything numerical comes from the authors' repository and from MatCouply: the
CLR with their 0.5 pseudo-count (`aux_funcs.clr`), their global Frobenius
scaling (`aux_funcs.norm_tensor`), their constraint/penalty set and AO-ADMM
call (`aux_funcs.fit_PARAFAC2_`, which fixes n_iter_max, tolerances and the
feasibility scale), and their run-selection rules from `collect_results.py`
(drop unfeasible, non-converged and degenerate runs, keep the lowest
reconstruction error).  This adapter supplies the tensor, the initialisations
and the inverse transform.

Reconstruction order, as required by the contract:

    model reconstruction (MatCouply cmf_to_matrices)
      -> undo the global Frobenius normalisation
      -> reconstructed CLR
      -> stable softmax
      -> comp_mean

Missing visits.  The authors' own branch switch is kept: a tensor containing
NaN is fitted with their EM option and their FARMM graph-Laplacian time
smoother (lambda = 1e-3, second-difference matrix), a complete tensor without
it.  Nothing is zero-filled.

Rank.  PARAFAC2's time factors are subject-specific matrices, so holding out a
sample can only be done through the authors' EM branch, which also turns on
their graph-Laplacian penalty -- the validation fit would not be the estimator
that is finally refit.  Rather than fake a comparable cross-validation, this
method uses a pre-specified rank (3, the middle of the shared grid) and can
report the whole grid as a sensitivity analysis.  This is stated in
EXTERNAL_BENCHMARKS.md and in every run's metadata.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib
import os
from pathlib import Path
import pickle
import sys

import numpy as np

from external_method_contract import (ContractError, ExternalResult, MethodUnsupported,
                                      Timer, build_meta, f32, stable_softmax,
                                      truncate_rank_grid)

NAME = "parafac2"
SOURCE_REPOSITORY = "https://github.com/blzserdos/parafac2_microbiome"

DEFAULTS = dict(
    rank=3,                 # pre-specified; see the module docstring
    rank_grid=[2, 3, 5],    # used only when sensitivity=True
    sensitivity=False,
    n_init=5,               # the paper uses 60; a recorded budget reduction
    source_dir="",          # checkout of parafac2_microbiome; required
    graph_laplacian_lambda=1e-3,   # the authors' FARMM setting
    max_iterations=5000,    # the value hard-coded in the authors' fit function
)


def _author_module(settings):
    source = Path(str(settings["source_dir"] or ""))
    functions = source / "functions"
    if not (functions / "aux_funcs.py").is_file():
        raise FileNotFoundError(
            f"parafac2_microbiome not found: {functions}/aux_funcs.py "
            "(set settings source_dir to the repository checkout)")
    if str(functions) not in sys.path:
        sys.path.insert(0, str(functions))
    return importlib.import_module("aux_funcs"), source


def _graph_laplacian(n_time, scale):
    """The authors' FARMM time smoother: second-difference matrix times lambda."""
    import tensorly as tl
    M = 2 * np.eye(n_time) - np.eye(n_time, k=1) - np.eye(n_time, k=-1)
    M[0, 0] = 1
    M[-1, -1] = 1
    return scale * tl.tensor(M)


def _prepare(truth, keep, aux, settings):
    """Counts -> authors' CLR -> authors' Frobenius scaling; returns the scale."""
    Y = np.asarray(truth["Y"], dtype=float).copy()      # aux.clr edits in place
    Y[~np.asarray(keep, bool)] = np.nan
    clr = aux.clr(Y)
    mask = np.isfinite(clr)
    norm = float(np.sqrt(np.nansum(clr ** 2)))
    scaled = aux.norm_tensor(clr.copy())
    finite = np.isfinite(scaled) & (np.abs(clr) > 1e-12)
    if finite.any():
        implied = float(np.median(clr[finite] / scaled[finite]))
        if not np.isclose(implied, norm, rtol=1e-6):
            raise ContractError(f"{NAME}: recovered Frobenius scale {implied:.6g} does not match "
                                f"the recomputed {norm:.6g}")
    return scaled, mask, norm


def _fit_rank(aux, source, tensor, mask, rank, settings, workdir, seed):
    """Run the authors' fit function for n_init random starts and pick their way."""
    from copy import deepcopy
    tag = f"R{int(rank)}"
    base = Path(workdir) / "analysis_results" / "models" / "bench" / "parafac2" / tag
    (base / "inits").mkdir(parents=True, exist_ok=True)
    (Path(workdir) / "analysis_results" / "models" / "logs").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    states = [int(v) for v in rng.integers(0, 1000000, size=int(settings["n_init"]))]
    has_missing = bool(np.isnan(tensor).any())
    penalty = (_graph_laplacian(tensor.shape[1], float(settings["graph_laplacian_lambda"]))
               if has_missing else None)
    here = Path.cwd()
    log = Path(workdir) / f"parafac2_fit_{tag}.log"
    try:
        os.chdir(workdir)                 # the authors' function writes relative paths
        # fit_PARAFAC2_ hard-codes verbose=100; keep its trace but out of the
        # driver's own stdout, where the Slurm log expects task progress only.
        with log.open("a", encoding="utf-8") as handle, redirect_stdout(handle):
            for state in states:
                aux.fit_PARAFAC2_(dict(dataset_name="bench", method="parafac2", rank=tag,
                                       tensor=deepcopy(tensor), init_no=state,
                                       mask=deepcopy(mask), l2_reg=1e-3, gl_penalty=penalty))
    finally:
        os.chdir(here)
    runs = []
    for path in sorted((base / "inits").glob("init*.pkl")):
        with path.open("rb") as handle:
            frame = pickle.load(handle)
        for _, row in frame.iterrows():
            runs.append(dict(row))
    return runs, states, has_missing


def _select_run(aux, runs, settings):
    """collect_results.py: drop broken, unfeasible, non-converged and degenerate
    runs, then keep the lowest final reconstruction error."""
    kept, rejected = [], []
    for run in runs:
        if run.get("exit") != "OK" or run.get("factors") is None:
            rejected.append(dict(init_no=run.get("init_no"), reason=str(run.get("exit"))))
            continue
        C, B, A = run["factors"]
        if np.isnan(C).any() or np.isnan(np.vstack(B)).any() or not C.any(axis=0).all():
            rejected.append(dict(init_no=run["init_no"], reason="NaN or all-zero factor column"))
            continue
        factors = aux.scale_factors((C, B, A), "parafac2")
        iterations = len(run["rec_errors"])
        gap = max(run["feasibility_gaps"][-1]) if run["feasibility_gaps"] else float("inf")
        if gap > 1e-6:
            rejected.append(dict(init_no=run["init_no"], reason=f"feasibility gap {gap:.3g}"))
            continue
        if iterations > int(settings["max_iterations"]):
            rejected.append(dict(init_no=run["init_no"], reason=f"{iterations} iterations"))
            continue
        if aux.check_degenerate(factors, "parafac2"):
            rejected.append(dict(init_no=run["init_no"], reason="degenerate"))
            continue
        kept.append(dict(run, factors=factors, iterations=iterations,
                         final_rec_error=float(run["rec_errors"][-1]), feasibility_gap=float(gap)))
    if not kept:
        raise MethodUnsupported(
            "every PARAFAC2 initialisation was rejected by the authors' own model-selection "
            f"rules ({rejected[:6]}); no model is reported rather than an unfeasible one")
    best = min(kept, key=lambda run: run["final_rec_error"])
    return best, kept, rejected


def _reconstruct(best, norm, shape):
    from matcouply.coupled_matrices import cmf_to_matrices
    C, B, A = best["factors"]
    matrices = cmf_to_matrices((None, (np.asarray(C), [np.asarray(b) for b in B], np.asarray(A))))
    clr = np.stack([np.asarray(m, dtype=float) for m in matrices], axis=0) * norm
    if clr.shape != shape:
        raise ContractError(f"{NAME}: reconstruction shape {clr.shape}, expected {shape}")
    return clr


def fit(truth, task, settings, context):
    aux, source = _author_module(settings)
    import matcouply
    observed = np.asarray(truth["mask"]).all(axis=2)
    seed = int(context["seed"])
    workdir = Path(context["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    grid = truncate_rank_grid(truth, settings.get("rank_grid") or DEFAULTS["rank_grid"])
    rank = int(settings.get("rank") or 3)
    if rank not in grid:
        raise MethodUnsupported(f"pre-specified rank {rank} does not fit this tensor (grid {grid})")

    tensor, mask, norm = _prepare(truth, observed, aux, settings)
    sensitivity = {}
    with Timer() as timer:
        runs, states, has_missing = _fit_rank(aux, source, tensor, mask, rank, settings,
                                              workdir, seed)
        best, kept, rejected = _select_run(aux, runs, settings)
        if settings.get("sensitivity"):
            for other in grid:
                if other == rank:
                    continue
                try:
                    other_runs, _, _ = _fit_rank(aux, source, tensor, mask, other, settings,
                                                 workdir, seed)
                    other_best, _, _ = _select_run(aux, other_runs, settings)
                    sensitivity[str(other)] = dict(final_rec_error=other_best["final_rec_error"],
                                                   iterations=other_best["iterations"])
                except Exception as error:
                    sensitivity[str(other)] = dict(error=f"{type(error).__name__}: {error}")
    clr = _reconstruct(best, norm, truth["Y"].shape)
    comp = np.full(clr.shape, np.nan)
    comp[observed] = stable_softmax(clr[observed], axis=-1)
    meta = build_meta(
        NAME, method_version=f"parafac2_microbiome + matcouply {matcouply.__version__}",
        source_repository=SOURCE_REPOSITORY,
        source_commit_id=context.get("commits", {}).get("parafac2", "see ENV_INFO.txt"),
        settings=dict(settings, rank_grid=grid), rank=rank, seed=seed,
        wall_seconds=timer.wall, alloc_cpus=context.get("alloc_cpus"), cpu_hours_used=None,
        transform="authors' clr(counts + 0.5) then global Frobenius normalisation",
        inverse_transform="cmf_to_matrices, multiply by the saved Frobenius norm, stable softmax",
        status="selected by the authors' collect_results rules",
        input_shape=truth["Y"].shape, output_shape=comp.shape,
        rank_selection=dict(rank=rank, rule="pre-specified (no comparable entry-wise CV; see module docstring)",
                            grid=grid, sensitivity=sensitivity),
        frobenius_norm=norm, em_branch=has_missing,
        graph_laplacian=bool(has_missing),
        initialisations=dict(requested=int(settings["n_init"]), random_states=states,
                             kept=len(kept), rejected=rejected,
                             selected_init=int(best["init_no"])),
        final_rec_error=best["final_rec_error"], iterations=int(best["iterations"]),
        feasibility_gap=best["feasibility_gap"],
        model_fit_percent=float(aux.get_model_fit(best["factors"], tensor, mask)))
    return ExternalResult(comp_mean=f32(comp), meta=meta,
                          arrays=dict(parafac2_clr_reconstruction=f32(clr)))
