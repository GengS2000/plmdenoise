"""TEMPTED through the Python Gemelli implementation (biocore/gemelli).

Estimand.  TEMPTED decomposes the *rclr-transformed* tensor into subject,
feature and smooth temporal loadings.  With gemelli's own default
`pseudo_count = 1` the rclr of `counts + 1` has no masked entries, so the
transform is an ordinary centred log-ratio and the low-rank reconstruction is a
full CLR tensor: every taxon of every observed visit has a finite value and the
inverse transform is an unambiguous stable softmax.  This is checked at runtime
(`transform_is_full_clr`); a configuration that leaves masked rclr entries is
refused rather than softmaxed on a guess.

Pipeline (all of it the authors' code):

    counts (+1) --matrix_rclr--> CLR --svd_centralize--> centred tensor
        --tempted_helper--> sum_s lambda_s a_i b (x) phi(t)
        --+ per-subject SVD centre--> CLR estimate --stable softmax--> comp_mean

The only added code is `gemelli_tempted_patch`, which exports the rank-one
reconstruction the author loop already computes; see that module for the patch
text and the guard on the pinned source.

`gemelli.tempted.tempted()` is not called directly because it wraps the fit in
`ctf_table_processing`, whose feature/sample filters would shrink the taxon set;
with gemelli's default thresholds (all zero) that step is a no-op, so this
adapter starts from `build_sparse`, which is exactly where `tempted()` arrives.
"""
from __future__ import annotations

import numpy as np

from external_method_contract import (ContractError, ExternalResult, MethodUnsupported,
                                      Timer, build_meta, check_taxa_preserved, f32,
                                      select_rank, stable_softmax, truncate_rank_grid)

NAME = "tempted_gemelli"
SOURCE_REPOSITORY = "https://github.com/biocore/gemelli"

DEFAULTS = dict(
    rank=0,                     # 0 = choose by the shared masked-sample CV
    rank_grid=[2, 3, 5],
    smooth=1e-6,                # gemelli DEFAULT_TEMPTED_SMTH
    resolution=101,             # gemelli DEFAULT_TEMPTED_RES
    maxiter=20,                 # gemelli DEFAULT_TEMPTED_MAXITER
    epsilon=1e-4,               # gemelli DEFAULT_TEMPTED_EP
    svd_centralized=True,       # gemelli DEFAULT_TEMPTED_SVDC
    n_components_centralize=1,  # gemelli DEFAULT_TEMPTED_SVDCN
    pseudo_count=1,             # gemelli DEFAULT_TEMPTED_PC -> rclr(counts+1) == clr
)


def taxon_names(J):
    return [f"taxon_{j:05d}" for j in range(J)]


def subject_names(I):
    return [f"subject_{i:05d}" for i in range(I)]


def _build_tables(truth, keep, settings):
    """The authors' preprocessing (`build_sparse`) on the kept samples."""
    import pandas as pd
    from gemelli.preprocessing import build_sparse

    I, T, J = truth["Y"].shape
    features, subjects = taxon_names(J), subject_names(I)
    columns, records = [], []
    for i, t in zip(*np.nonzero(keep)):
        columns.append(np.asarray(truth["Y"][i, t], dtype=float))
        records.append((f"s{i:05d}_v{t:03d}", subjects[i], float(truth["times"][i, t]), i, t))
    if not records:
        raise MethodUnsupported("no observed samples to fit")
    sample_ids = [r[0] for r in records]
    table = pd.DataFrame(np.asarray(columns).T, index=features, columns=sample_ids)
    metadata = pd.DataFrame(dict(host_subject_id=[r[1] for r in records],
                                 time_point=[r[2] for r in records]), index=sample_ids)
    counts = metadata["host_subject_id"].value_counts()
    if counts.min() <= 1:
        raise MethodUnsupported("TEMPTED needs at least two visits per subject after masking")
    tensor = build_sparse().construct(
        table, metadata, "host_subject_id", "time_point",
        pseudo_count=settings["pseudo_count"],
        replicate_handling="error",   # the simulator has one sample per (subject, visit)
        svd_centralized=bool(settings["svd_centralized"]),
        n_components_centralize=int(settings["n_components_centralize"]))
    check_taxa_preserved(NAME, J, len(tensor.feature_order))
    if sorted(tensor.feature_order) != sorted(features):
        raise ContractError(f"{NAME}: gemelli changed the feature set")
    position = {name: j for j, name in enumerate(features)}
    feature_rows = np.array([position[name] for name in tensor.feature_order])
    cell = {r[0]: (r[3], r[4]) for r in records}
    return tensor, feature_rows, cell


def _grid_index(times, low, high, resolution):
    """The author mapping from a time to a state-loading grid column."""
    if high <= low:
        raise MethodUnsupported("all samples share one time point")
    scaled = (resolution - 1) * (np.asarray(times, dtype=float) - low) / (high - low)
    return np.clip(scaled.astype(int), 0, resolution - 1)


def _fit_once(truth, keep, rank, settings, seed, patched):
    """One TEMPTED fit; returns (I, T, J) CLR reconstruction on the whole grid."""
    I, T, J = truth["Y"].shape
    # Both svd_centralize and the per-component initialisation call scipy's
    # svds, which draws its ARPACK start vector from the legacy global RNG.
    # Seed before the first of them so a fit does not depend on how many fits
    # ran before it in the same process (rank selection runs several).
    np.random.seed(int(seed) % (2 ** 32 - 1))
    try:
        tensor, feature_rows, cell = _build_tables(truth, keep, settings)
    except (np.linalg.LinAlgError, SystemError) as error:
        # The pinned gemelli environments ship NumPy old enough that a
        # non-converged svds inside svd_centralize surfaces as SystemError
        # wrapping LinAlgError instead of LinAlgError itself.
        raise MethodUnsupported(
            f"TEMPTED's svd_centralize did not converge ({I} subjects x {T} visits x {J} taxa, "
            f"{int(np.asarray(keep).sum())} samples fitted): {error}. The authors' "
            "implementation does not guard against this tensor.") from error
    tables = (tensor.individual_id_tables_centralized if settings["svd_centralized"]
              else tensor.individual_id_tables)
    if not np.isfinite(np.concatenate([m.values.ravel() for m in tables.values()])).all():
        raise MethodUnsupported(
            "the rclr transform left masked (non-finite) entries: with pseudo_count=0 the "
            "reconstruction is not a full CLR tensor and no softmax is defined for the "
            "missing taxa. Use the gemelli default pseudo_count=1.")
    # Per-subject SVD centre that must be added back to return to the CLR scale.
    centre = {}
    for key, table in tensor.individual_id_tables.items():
        shift = table.values - tables[key].values
        if not np.allclose(shift, shift[:, [0]], atol=1e-8):
            raise ContractError(f"{NAME}: svd_centralize shift is not constant in time")
        centre[key] = shift[:, 0]
    inputs = {key: table.values.copy() for key, table in tables.items()}
    try:
        result = patched(tables, tensor.individual_id_state_orders, tensor.feature_order,
                         n_components=int(rank), smooth=float(settings["smooth"]),
                         resolution=int(settings["resolution"]),
                         maxiter=int(settings["maxiter"]), epsilon=float(settings["epsilon"]))
    except (np.linalg.LinAlgError, SystemError) as error:
        raise MethodUnsupported(
            f"TEMPTED did not converge at rank {rank} on this tensor "
            f"({I} subjects x {T} visits x {J} taxa, {int(np.asarray(keep).sum())} samples "
            f"fitted): {error}. The rank-one updates produced a degenerate loading, which "
            "the authors' implementation does not guard against.") from error
    (_individual, _feature, state_loadings, _time_return, lambda_coeff, rsquared,
     reconstruction, reconstruction_grid, residual, ti, order) = result
    # The exported reconstruction must be exactly what the author loop removed
    # from its working tables: input - residual. If this ever fails, the patch
    # is no longer tracking the fit and no estimate is reported.
    for key, start in inputs.items():
        if not np.isfinite(reconstruction[key]).all():
            raise MethodUnsupported(
                f"TEMPTED produced a non-finite reconstruction at rank {rank} for {key}; "
                "the rank-one updates degenerated (a loading normalised by zero).")
        if not np.allclose(start - residual[key], reconstruction[key], atol=1e-8):
            raise ContractError(f"{NAME}: exported reconstruction differs from the fit's "
                                f"own residual update for {key}")
    resolution = int(settings["resolution"])
    every_time = np.concatenate([np.asarray(v, dtype=float)
                                 for v in tensor.individual_id_state_orders.values()])
    low, high = float(every_time.min()), float(every_time.max())
    clr = np.full((I, T, J), np.nan)
    for position, key in enumerate(order):
        grid = np.asarray(reconstruction_grid[key], dtype=float)
        fitted = np.asarray(reconstruction[key], dtype=float)
        index = np.asarray(ti[position], dtype=int)
        if not np.allclose(grid[:, index], fitted, atol=1e-8):
            raise ContractError(f"{NAME}: grid and sample reconstructions disagree")
        i = int(key.split("_")[1])
        full = grid + centre[key][:, None]                       # CLR scale
        columns = _grid_index(truth["times"][i], low, high, resolution)
        clr[i, :, feature_rows] = full[:, columns]
    # Visits the simulation never observed stay NaN: the temporal loading
    # function could be evaluated there, but nothing in the benchmark scores it.
    clr[~np.asarray(truth["mask"]).all(axis=2)] = np.nan
    return clr, dict(lambda_coeff=np.asarray(lambda_coeff, dtype=float).tolist(),
                     r_squared=np.asarray(rsquared, dtype=float).tolist(),
                     state_loadings=np.asarray(state_loadings, dtype=float),
                     svd_centre=np.asarray([centre[k] for k in order], dtype=float),
                     subject_order=[int(k.split("_")[1]) for k in order])


def _composition(clr, observed):
    comp = np.full(clr.shape, np.nan)
    comp[observed] = stable_softmax(clr[observed], axis=-1)
    return comp


def fit(truth, task, settings, context):
    import gemelli
    from adapters.gemelli_tempted_patch import build as build_patch

    patched, patch_info = build_patch()
    observed = np.asarray(truth["mask"]).all(axis=2)
    seed = int(context["seed"])
    grid = truncate_rank_grid(truth, settings.get("rank_grid") or DEFAULTS["rank_grid"])

    def fit_at_rank(rank, held):
        clr, _ = _fit_once(truth, observed & ~held, rank, settings, seed, patched)
        return _composition(clr, observed)

    with Timer() as timer:
        if int(settings.get("rank", 0)):
            rank = int(settings["rank"])
            selection = dict(rank=rank, rule="fixed by settings", scores={}, failures={})
        else:
            selection = select_rank(fit_at_rank, truth, grid, seed)
            rank = selection["rank"]
        clr, extra = _fit_once(truth, observed, rank, settings, seed, patched)
    comp = _composition(clr, observed)
    meta = build_meta(
        NAME, method_version=f"gemelli {gemelli.__version__}",
        source_repository=SOURCE_REPOSITORY,
        source_commit_id=context.get("commits", {}).get("gemelli", "see ENV_INFO.txt"),
        settings=dict(settings, rank_grid=grid), rank=int(rank), seed=seed,
        wall_seconds=timer.wall, alloc_cpus=context.get("alloc_cpus"),
        cpu_hours_used=timer.cpu / 3600.0,
        transform="rclr(counts + pseudo_count) == clr, then svd_centralize (rank 1)",
        inverse_transform="reconstruction + per-subject svd centre, then stable softmax",
        status="converged" if np.isfinite(clr[observed]).all() else "non-finite reconstruction",
        input_shape=truth["Y"].shape, output_shape=comp.shape,
        rank_selection=selection, lambda_coeff=extra["lambda_coeff"],
        r_squared=extra["r_squared"], gemelli_patch=patch_info,
        transform_is_full_clr=True)
    return ExternalResult(comp_mean=f32(comp), meta=meta,
                          arrays=dict(tempted_state_loadings=f32(extra["state_loadings"]),
                                      tempted_svd_centre=f32(extra["svd_centre"]),
                                      tempted_clr_reconstruction=f32(clr)))
