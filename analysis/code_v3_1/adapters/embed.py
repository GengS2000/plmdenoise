"""EMBED / Thermodynamic Manifold Inference (Shahin, Ji & Dixit) via Octave or MATLAB.

The authors' implementation is MATLAB (`EMBED/Scripts/source_codes/TMI.m` and
`TMI_masked_samples.m`); the repository's `python_directory` holds data and
results only, no Python model code.  This adapter therefore drives the authors'
`.m` files from a non-interactive engine rather than re-writing the model in
Python: `embed_runner.m` builds their input, calls `TMI_masked_samples`
unchanged, and exports the composition through `embed_q.m`, a verbatim copy of
the nested `q()` that MATLAB does not let a caller reach.

Estimand.  TMI fits q = exp(-theta z) normalised per (subject, time), i.e. the
model's own relative composition.  `comp_mean` is that q with the authors' 1/S
factor undone; no separate inverse transform is invented.

Applicability domain.  The authors' data matrix is (subjects x taxa) rows by
*shared* time columns, so EMBED needs a common visit grid; datasets with
subject-specific times are refused.  Missing and held-out samples go through the
authors' own `TMI_masked_samples` mask, so nothing is interpolated.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from external_method_contract import (ContractError, ExternalResult, MethodUnsupported,
                                      Timer, build_meta, f32, select_rank, truncate_rank_grid)
from adapters import tabular
from adapters.microtensor import require_common_grid

NAME = "embed"
SOURCE_REPOSITORY = "https://github.com/mayar-shahin/EMBED"
RUNNER = Path(__file__).with_name("embed_runner.m")

# Learning rates, step budget and stopping rule from the authors' own analysis
# scripts (Fig3a.m: 0.06 / 1.5e5 / 1e-6; Fig4a.m: 0.02 / 2.5e6 / 1e-4); two
# restarts keeping the lowest KL is the procedure their README prescribes.
DEFAULTS = dict(
    rank=0,                 # 0 = choose by the shared masked-sample CV
    rank_grid=[2, 3, 5],
    learn_z=0.02,
    learn_theta=0.02,
    max_steps=250000,
    min_fitting=0.0,
    min_gradients=1e-4,
    restarts=2,
    accept_unconverged=False,
    engine="octave",        # octave | matlab
    octave="octave",
    matlab="matlab",
    source_dir="",          # EMBED/Scripts/source_codes; required
    timeout=14400,
)


def _engine_command(settings, workdir, args_file):
    engine = str(settings["engine"]).lower()
    if engine == "octave":
        binary = shutil.which(str(settings["octave"])) or shutil.which("octave-cli")
        if not binary:
            raise MethodUnsupported(
                "no Octave on PATH; EMBED's author code is MATLAB. Install octave in the "
                "`embed` environment (slurm/setup_external_envs_v2.sh) or set engine=matlab.")
        return [binary, "--no-gui", "--quiet", str(RUNNER)]
    if engine == "matlab":
        binary = shutil.which(str(settings["matlab"]))
        if not binary:
            raise MethodUnsupported("engine=matlab but no matlab on PATH")
        return [binary, "-batch", f"run('{RUNNER.as_posix()}')"]
    raise ValueError(f"unknown EMBED engine {settings['engine']!r}")


def _run(truth, keep, rank, settings, workdir, seed, tag):
    source_dir = Path(str(settings["source_dir"] or ""))
    if not (source_dir / "TMI_masked_samples.m").is_file():
        raise FileNotFoundError(
            f"EMBED source not found: {source_dir}/TMI_masked_samples.m "
            "(set settings source_dir to EMBED/Scripts/source_codes)")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    counts = workdir / f"counts_{tag}.csv"
    output = workdir / f"comp_{tag}.csv"
    diagnostics = workdir / f"diagnostics_{tag}.json"
    tabular.write_counts_csv(counts, truth, keep)
    values = [str(counts), str(output), str(diagnostics), str(source_dir.resolve()),
              str(int(rank)), repr(float(settings["learn_z"])), repr(float(settings["learn_theta"])),
              str(int(settings["max_steps"])), repr(float(settings["min_fitting"])),
              repr(float(settings["min_gradients"])), str(int(settings["restarts"])), str(int(seed))]
    args_file = workdir / f"args_{tag}.txt"
    args_file.write_text("\n".join(values) + "\n", encoding="utf-8", newline="\n")
    command = _engine_command(settings, workdir, args_file) + values
    environment = dict(os.environ, EMBED_ARGS_FILE=str(args_file))
    completed = subprocess.run(command, cwd=str(workdir), capture_output=True, encoding="utf-8", errors="replace",
                               timeout=float(settings["timeout"]), env=environment)
    (workdir / f"stdout_{tag}.txt").write_text(completed.stdout or "", encoding="utf-8")
    (workdir / f"stderr_{tag}.txt").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode or not output.is_file():
        raise RuntimeError(f"EMBED engine exited {completed.returncode}; "
                           f"see {workdir / f'stderr_{tag}.txt'}\n{(completed.stderr or '')[-2000:]}")
    values_out = tabular.read_values_csv(output, truth["Y"].shape)
    info = json.loads(diagnostics.read_text(encoding="utf-8"))
    return values_out, info, " ".join(command)


def _composition(values, observed):
    comp = np.full(values.shape, np.nan)
    block = values[observed]
    if not np.isfinite(block).all() or (block < 0).any():
        raise ContractError(f"{NAME}: q(z, theta) is not a finite non-negative composition")
    total = block.sum(axis=-1, keepdims=True)
    if np.any(total <= 0):
        raise ContractError(f"{NAME}: q(z, theta) has zero mass at some observed visit")
    comp[observed] = block / total
    return comp


def fit(truth, task, settings, context):
    grid_times = require_common_grid(truth, 1e-6)
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
    if not (info.get("kl_masked_best") is not None and np.isfinite(info["kl_masked_best"])):
        raise ContractError(f"{NAME}: the engine returned a non-finite KL divergence")
    # TMI stops on its relative-gradient criterion; a run that simply used every
    # step is recorded as unconverged, never silently accepted as a success.
    converged = bool(info.get("converged"))
    warnings = [] if converged else [
        f"TMI reached max_steps={settings['max_steps']} with relative theta gradient "
        f"{info.get('relative_gradient_theta')} above min_gradients={settings['min_gradients']}"]
    if not converged and not settings.get("accept_unconverged"):
        raise ContractError(
            f"{NAME}: " + warnings[0] + ". Raise max_steps, or set accept_unconverged=true "
            "to record the last iterate as a (clearly flagged) result.")
    meta = build_meta(
        NAME, method_version=f"EMBED TMI via {info.get('engine', 'unknown engine')}",
        source_repository=SOURCE_REPOSITORY,
        source_commit_id=context.get("commits", {}).get("embed", "see ENV_INFO.txt"),
        settings=dict(settings, rank_grid=grid), rank=int(rank), seed=seed,
        wall_seconds=timer.wall, alloc_cpus=context.get("alloc_cpus"),
        cpu_hours_used=None,
        transform="per-(subject, visit) closure of the counts (authors' normalize(...,'norm',1))",
        inverse_transform="q(z, theta) = exp(-theta z) renormalised per sample; no extra transform",
        status="converged" if converged else "max_steps reached", input_shape=truth["Y"].shape, output_shape=comp.shape,
        rank_selection=selection, command=command, engine_diagnostics=info,
        time_representation="shared discrete visit grid", grid_times=grid_times,
        warnings=warnings)
    return ExternalResult(comp_mean=f32(comp), meta=meta, warnings=warnings)
