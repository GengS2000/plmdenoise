"""Frozen DA plans, server-only generation/fitting/testing, and failure-aware summaries."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback

from r_process import run_r

ROOT = Path(__file__).resolve().parent
DEFAULT_R_BIN = "/public/software/VersionHub/R/4.2.2/R"
DEFAULT_MAASLIN2_R_BIN = "/public/software/VersionHub/R/4.4.2/R"
SOURCE_FILES = (
    "compare_da_denoising.py", "compare_methods.py", "external_benchmarks.py", "evaluate.py",
    "slurm/compare_da_denoising.sbatch",
    "tests/test_da_comparison.py",
    "run_da.py", "check_da_server.py", "r_process.py", "da_design.py", "simulate_microbiome.py", "event_aware_denoise.py",
    "adapters/da_methods.R", "adapters/da_dependencies.R", "adapters/check_microtensor.R", "DA_SERVER_GUIDE.md",
    "slurm/submit_da.sh", "slurm/resume_da_existing_envs.sh", "slurm/da_array.sbatch", "slurm/da_merge.sbatch",
    "tests/test_da_design.py", "tests/check_da_adapters.R",
    "adapters/da_warning_policy.R", "da_effect_diagnostics.py", "da_local_pilot.py", "DA_ORACLE_GUIDE.md",
    "slurm/submit_da_oracle.sh", "tests/test_da_oracle.py", "da_mechanism_check.py",
    "da_external_worker.py", "da_io.py", "da_external_settings.json", "da_method_eligibility.csv",
    "external_method_contract.py", "adapters/__init__.py", "adapters/tabular.py",
    "adapters/tempted_gemelli.py", "adapters/gemelli_tempted_patch.py",
    "adapters/microtensor.py", "adapters/microtensor_runner.R",
    "adapters/embed.py", "adapters/embed_runner.m", "adapters/embed_q.m",
    "adapters/octave_compat/nansum.m", "adapters/octave_compat/normalize.m",
    "adapters/parafac2.py",
)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True).encode()).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def prepare(args):
    from da_design import effect_cells, effect_grid, simulation_config, METHODS, INPUTS
    from event_aware_denoise import DenoiseConfig
    seeds = args.seeds if args.seeds is not None else list(range(5100, 5115))
    if len(set(seeds)) != len(seeds) or not seeds or min(seeds) < 0 or max(seeds) > 2146783647:
        raise ValueError("Seeds must be unique integers in 0..2146783647 (R seed range)")
    if not 0 < args.alpha < 1 or not math.isfinite(args.log_pseudocount) or args.log_pseudocount <= 0:
        raise ValueError("Invalid alpha or log pseudocount")
    tasks = []
    strengths = getattr(args, "effect_strengths", None)
    if strengths is not None and args.levels != [0, 1]:
        raise ValueError("Use either explicit effect strengths or legacy levels")
    cells = effect_grid(strengths) if strengths is not None else effect_cells(args.levels)
    for seed in seeds:
        for cell in cells:
            tasks.append(dict(task_id=f"seed_{seed}_{cell['series']}_s{cell['s']}", seed=seed, **cell))
    fit = asdict(DenoiseConfig(adaptive_budget=True, joint_rank=4,
                              hierarchical_observation=True, flexible_event=True,
                              map_steps=600, map_starts=2, vi_steps=2000,
                              posterior_samples=400, early_stopping=True, verbose=False))
    # Same fit RNG seed across effect levels, as for paired simple32 comparisons.
    fit.pop("seed")
    external = json.loads((ROOT / "da_external_settings.json").read_text(encoding="utf-8"))
    if args.external_config:
        supplied = json.loads(args.external_config.read_text(encoding="utf-8"))
        if set(supplied) != set(external):
            raise ValueError("External config must retain exactly the four eligible methods")
        for name, entry in supplied.items():
            # Execution paths/environment may change; scientific settings stay fixed.
            if set(entry) - {"environment", "python", "settings", "r_library"}:
                raise ValueError(f"Unsupported configuration fields for {name}")
            scientific = external[name]["settings"]
            supplied_settings = entry.get("settings", {})
            execution_keys = {"source_dir", "engine", "octave", "matlab", "timeout"}
            if set(supplied_settings) - set(scientific) - execution_keys:
                raise ValueError(f"Unknown external settings for {name}")
            if any(k in scientific and v != scientific[k] for k, v in supplied_settings.items()):
                raise ValueError("Use the successful eight-scenario scientific settings unchanged")
            external[name] = external[name] | entry
            external[name]["settings"] = scientific | supplied_settings
    external_root = args.external_root.resolve()
    external["embed"]["settings"].setdefault("source_dir", str(external_root / "EMBED/Scripts/source_codes"))
    external["parafac2"]["settings"].setdefault("source_dir", str(external_root / "parafac2_microbiome"))
    external["microtensor"].setdefault("r_library", str(external_root / "R_library"))
    methods = list(getattr(args, "methods", None) or METHODS)
    inputs = list(INPUTS) + (["oracle_clean"] if getattr(args, "include_oracle", False) else [])
    if getattr(args, "abundance_inputs", False):
        if "oracle_clean" not in inputs:
            raise ValueError("--abundance-inputs requires --include-oracle")
        inputs += ["m4_flex_abundance", "oracle_abundance"]
    if len(set(methods)) != len(methods):
        raise ValueError("Duplicate DA methods")
    schema = ("da_v4_abundance" if "oracle_abundance" in inputs else
              "da_v3_oracle" if "oracle_clean" in inputs else "da_v2_multi")
    plan = dict(schema=schema,
                external=external, external_root=str(external_root), tasks=tasks, simulation=asdict(simulation_config(0)),
                fit=fit, denoise_group=args.denoise_group, methods=methods, inputs=inputs,
                alpha=args.alpha, log_pseudocount=args.log_pseudocount,
                r_seed_offset=700000, fixed_effects=["X", "visit", "event_post"],
                target="gamma != 0; injected log-abundance coefficient, not marginal relative DA",
                source_hashes={name: file_hash(ROOT / name) for name in SOURCE_FILES})
    if "oracle_clean" in inputs:
        plan["oracle_note"] = "Diagnostic only: saved clean composition with realised structural zeros; never supplied to denoisers"
        plan["paired_reference"] = "naive; additionally compare each reconstruction with oracle_clean in oracle_paired_summary.csv"
        plan["effect_diagnostics"] = True
    if "oracle_abundance" in inputs:
        plan["abundance_note"] = ("m4_flex_abundance = exp(posterior mean Z) closed over taxa, from the same blind "
                                  "m4_flex fit (absence channel dropped); oracle_abundance = exp(Z_true) closed, "
                                  "no structural zeros. gamma is injected in Z, so these are the abundance-channel "
                                  "pair; m4_flex/oracle_clean remain the composition-channel pair.")
    plan["plan_id"] = digest(plan)
    out = args.out.resolve()
    if (out / "plan.json").exists():
        saved = load_plan(out, require_snapshot=False)
        if saved != clean(plan):
            raise ValueError("Existing immutable plan differs; choose a new OUT")
        print(f"Existing verified plan: {len(tasks)} datasets")
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError("New OUT must be empty; do not mix experiments")
    out.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        dest = out / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    atomic_json(out / "plan.json", plan)
    print(f"Prepared {len(tasks)} datasets; source frozen. No data generated or jobs submitted.")


def load_plan(out, require_snapshot=True):
    out = Path(out).resolve()
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    body = {k: v for k, v in plan.items() if k != "plan_id"}
    if digest(body) != plan["plan_id"]:
        raise ValueError("Plan checksum mismatch")
    for name, checksum in plan["source_hashes"].items():
        if file_hash(out / "source" / name) != checksum:
            raise ValueError(f"Frozen source changed: {name}; create a new experiment")
    if require_snapshot and ROOT != out / "source":
        raise ValueError("Use OUT/source/run_da.py for all operations after prepare")
    return plan


def artifact_done(marker, plan_id, retry_partial=False):
    if not marker.exists():
        return False
    try:
        info = json.loads(marker.read_text(encoding="utf-8"))
        allowed = {"complete"} if retry_partial else {"complete", "partial"}
        return (info["plan_id"] == plan_id and info["status"] in allowed and
                bool(info.get("artifacts")) and
                all((marker.parent / name).is_file() and file_hash(marker.parent / name) == checksum
                    for name, checksum in (info["artifacts"] | info.get("dependencies", {})).items()))
    except (ValueError, KeyError, OSError):
        return False


def finish(marker, plan, status, files, dependencies=(), **extra):
    atomic_json(marker, dict(plan_id=plan["plan_id"], status=status,
                            artifacts={str(Path(f).relative_to(marker.parent)): file_hash(f) for f in files},
                            dependencies={os.path.relpath(f, marker.parent): file_hash(f) for f in dependencies},
                            **extra))


def atomic_npz(path, arrays):
    import numpy as np
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temp.replace(path)


def generate(directory, plan, task):
    import numpy as np
    from simulate_microbiome import SimulationConfig
    from da_design import simulate_da, export_observed, export_matrix, abundance_composition
    marker = directory / "generation.json"
    if artifact_done(marker, plan["plan_id"]):
        return
    truth = simulate_da(SimulationConfig(**(plan["simulation"] | dict(seed=task["seed"]))), task["strength"])
    atomic_npz(directory / "truth.npz", truth)
    atomic_npz(directory / "observed.npz", {key: truth[key] for key in
               ("Y", "times", "depth", "batch", "event_times", "group", "mask")})
    export_observed(directory, truth)
    names = ["truth.npz", "observed.npz", "counts.csv", "naive.csv", "metadata.csv", "truth.csv"]
    if "oracle_clean" in plan["inputs"]:
        export_matrix(directory / "oracle_clean.csv", truth["clean_composition"])
        names.append("oracle_clean.csv")
    if "oracle_abundance" in plan["inputs"]:
        export_matrix(directory / "oracle_abundance.csv", abundance_composition(truth["Z_true"]))
        names.append("oracle_abundance.csv")
    finish(marker, plan, "complete", [directory / name for name in names],
           task=task, runtime=dict(python=sys.version, numpy=np.__version__))


def denoise(directory, plan, task, threads):
    import numpy as np
    import torch
    from event_aware_denoise import DenoiseConfig, EventAwareDenoiser
    from da_design import export_matrix, abundance_composition
    marker = directory / "denoise_m4_flex.json"
    if artifact_done(marker, plan["plan_id"]):
        return
    torch.set_num_threads(threads)
    with np.load(directory / "observed.npz", allow_pickle=False) as saved:
        # Explicit observation-only input: never pass gamma, S, Z or true coefficients.
        observed = {key: saved[key] for key in ("Y", "times", "depth", "batch", "event_times", "group", "mask")}
    group = observed["group"] if plan["denoise_group"] == "aware" else None
    started = time.monotonic()
    estimator = EventAwareDenoiser(DenoiseConfig(**(plan["fit"] | dict(seed=task["seed"]))))
    estimator.fit(observed["Y"], observed["times"], observed["depth"], observed["batch"],
                  observed["event_times"], group, observed["mask"])
    result = estimator.posterior()
    composition = result["composition_mean"]
    if not np.isfinite(composition).all() or (composition < 0).any() or not np.allclose(composition.sum(-1), 1.):
        raise ValueError("Invalid posterior composition")
    export_matrix(directory / "m4_flex.csv", composition)
    artifacts = [directory / "m4_flex.csv", directory / "posterior_da.npz"]
    if "m4_flex_abundance" in plan["inputs"]:
        # Same blind fit; only the absence channel (1 - rho) is dropped.
        export_matrix(directory / "m4_flex_abundance.csv", abundance_composition(result["Z_mean"]))
        artifacts.append(directory / "m4_flex_abundance.csv")
    atomic_npz(directory / "posterior_da.npz", {key: result[key] for key in
               ("composition_mean", "X_mean", "Z_mean", "absence_probability_mean", "metadata_json")})
    finish(marker, plan, "complete", artifacts,
           dependencies=[directory / "observed.npz"],
           seconds=time.monotonic() - started, diagnostics=estimator.metadata_,
           converged=bool(estimator.metadata_.get("map_stable") and estimator.metadata_.get("vi_stable")),
           runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                        platform=platform.platform(), threads=threads), denoise_group=plan["denoise_group"])


def denoise_external(directory, plan, task, source, args):
    marker = directory / f"denoise_{source}.json"
    if artifact_done(marker, plan["plan_id"]):
        return
    config = plan["external"][source]
    attempt = directory / "denoisers" / source / ("attempt_" + str(time.time_ns()))
    attempt.mkdir(parents=True)
    settings = dict(config["settings"])
    if source == "microtensor":
        # Explicit site R for this method, independent of DA/Rscript overrides.
        settings["r_binary"] = os.environ.get("MICROTENSOR_R_BIN", DEFAULT_R_BIN)
    request = dict(method=source, settings=settings, observed=str(directory / "observed.npz"),
                   workdir=str(attempt), seed=task["seed"], threads=args.threads,
                   external_root=plan["external_root"])
    atomic_json(attempt / "request.json", request)
    command = ([config["python"]] if config.get("python") else
               [args.conda, "run", "--no-capture-output", "-n", config["environment"], "python"])
    command += [str(ROOT / "da_external_worker.py"), "--request", str(attempt / "request.json")]
    env = os.environ.copy()
    if source == "microtensor" and config.get("r_library"):
        env["R_LIBS_USER"] = config["r_library"]
    started = time.monotonic()
    log_path = attempt / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:])
        raise RuntimeError(f"{source} exited {completed.returncode}; log: {log_path}\n{tail}")
    diagnostics = json.loads((attempt / "diagnostics.json").read_text(encoding="utf-8"))
    dest = directory / f"{source}.csv"
    shutil.copyfile(attempt / "composition.csv", dest.with_suffix(".csv.tmp"))
    dest.with_suffix(".csv.tmp").replace(dest)
    status = diagnostics["metadata"].get("status")
    converged = status in {"converged", "selected by the authors' collect_results rules"}
    finish(marker, plan, "complete", [dest, attempt / "composition.csv", attempt / "reconstruction.npz",
           attempt / "diagnostics.json", attempt / "request.json", attempt / "run.log"],
           dependencies=[directory / "observed.npz"], converged=converged,
           seconds=time.monotonic() - started, attempt=str(attempt.relative_to(directory)),
           diagnostics=diagnostics)


def r_command(args, script, *arguments, method=None):
    """Use the server R executable directly; retain an explicit Rscript override."""
    if method == "maaslin2":
        return [args.maaslin2_r_bin, "--vanilla", "--slave", "-f", str(script), "--args", *arguments]
    if args.rscript:
        return [args.rscript, "--vanilla", str(script), *arguments]
    return [args.r_bin, "--vanilla", "--slave", "-f", str(script), "--args", *arguments]


def r_environment(method):
    env = os.environ.copy()
    # DA R processes may run side by side within one task; keep each single-threaded.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = "1"
    if method == "maaslin2":
        # Do not carry the R 4.2 user/site libraries into R 4.4.
        for key in ("R_LIBS", "R_LIBS_USER", "R_LIBS_SITE"):
            env.pop(key, None)
        if env.get("MAASLIN2_R_LIBS_USER"):
            env["R_LIBS_USER"] = env["MAASLIN2_R_LIBS_USER"]
    return env


def check_dependencies(args, methods=None):
    from da_design import METHODS
    for method in (methods or METHODS):
        command = r_command(args, ROOT / "adapters/da_dependencies.R", "--check", method, method=method)
        run_r(command, env=r_environment(method), check=True)


def test_method(directory, plan, task, source, method, args):
    from da_design import score_results, write_csv
    output = directory / "da" / source / method
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "status.json"
    if artifact_done(marker, plan["plan_id"], args.retry_partial):
        return
    # Unique attempt directory prevents stale package outputs being reused after failure.
    attempt = output / ("attempt_" + str(time.time_ns()))
    attempt.mkdir()
    command = r_command(args, ROOT / "adapters/da_methods.R", "--method", method,
               "--input", str(directory / f"{source}.csv"),
               "--metadata", str(directory / "metadata.csv"), "--out", str(attempt),
               "--pseudocount", str(plan["log_pseudocount"]),
               "--seed", str(task["seed"] + plan["r_seed_offset"]), method=method)
    started = time.monotonic()
    with (attempt / "run.log").open("w", encoding="utf-8") as log:
        run_r(command, env=r_environment(method), stdout=log, stderr=subprocess.STDOUT, check=True)
    result_file = attempt / "results.csv"
    metrics, rows = score_results(directory / "truth.csv", result_file, plan["alpha"])
    write_csv(output / "taxa.csv", list(rows[0]), rows)
    status = "complete" if metrics["n_failed"] == 0 else "partial" if metrics["n_tested"] else "failed"
    native_files = [p for p in attempt.rglob("*") if p.is_file()]
    finish(marker, plan, status, [output / "taxa.csv", *native_files],
           dependencies=[directory / f"{source}.csv", directory / "metadata.csv", directory / "truth.csv"],
           task=task, input=source, method=method, metrics=metrics,
           seconds=time.monotonic() - started, attempt=attempt.name)
    return status


def check_external_imports(plan, args):
    probes = {
        "tempted_gemelli": "import numpy, scipy, pandas, gemelli; from adapters.gemelli_tempted_patch import build; build()",
        "microtensor": "import numpy, scipy",
        "embed": "import numpy, scipy",
        "parafac2": "import numpy, scipy, tensorly, matcouply",
    }
    for method, config in plan["external"].items():
        command = ([config["python"]] if config.get("python") else
                   [args.conda, "run", "--no-capture-output", "-n", config["environment"], "python"])
        print(f"Checking external Python environment: {method}", flush=True)
        subprocess.run(command + ["-c", probes[method]], cwd=str(ROOT), check=True)


def work_units(plan, unit):
    """dataset: one unit per dataset; input: one unit per dataset x input (DA only)."""
    if unit == "dataset":
        return [(task, None) for task in plan["tasks"]]
    if unit == "input":
        return [(task, source) for task in plan["tasks"] for source in plan["inputs"]]
    raise ValueError(f"Unknown work unit: {unit}")


def run_methods(directory, plan, task, source, args):
    """All DA methods for one input; up to --threads single-threaded R processes at once."""
    from concurrent.futures import ThreadPoolExecutor

    def one(method):
        try:
            return test_method(directory, plan, task, source, method, args) == "failed"
        except Exception:
            atomic_json(directory / "da" / source / method / "status.json",
                        dict(status="failed", plan_id=plan["plan_id"], task=task,
                             input=source, method=method, error=traceback.format_exc()))
            traceback.print_exc()
            return True

    workers = max(1, min(args.threads, len(plan["methods"])))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(one, plan["methods"]))


def run(args):
    from da_design import fit_source
    plan = load_plan(args.out)
    unit = getattr(args, "unit", "dataset")
    if args.shards < 1 or not 0 <= args.shard < args.shards or args.threads < 1:
        raise ValueError("Invalid shard/thread arguments")
    if unit == "input" and args.phase != "da":
        raise ValueError("--unit input only splits the DA phase")
    # data = generate + all denoiser fits + effect diagnostics (no DA).
    generating = args.phase in ("all", "generate", "data")
    fitting = args.phase in ("all", "fit", "data")
    testing = args.phase in ("all", "da")
    if testing:
        check_dependencies(args, plan["methods"])
    if fitting:
        check_external_imports(plan, args)
    failures = 0
    for index, (task, only_source) in enumerate(work_units(plan, unit)):
        if index % args.shards != args.shard:
            continue
        directory = args.out / "datasets" / task["task_id"]
        directory.mkdir(parents=True, exist_ok=True)
        # Per-input locks let the inputs of one dataset be tested concurrently.
        lock = directory / ".running" if only_source is None else directory / "da" / only_source / ".running"
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            lock.mkdir()
        except FileExistsError:
            raise RuntimeError(f"Task locked: {lock}; check running jobs before removing a stale lock")
        atomic_json(lock / "owner.json", dict(pid=os.getpid(), host=platform.node(),
                    job=os.environ.get("SLURM_JOB_ID"), phase=args.phase, unit=unit))
        try:
            print(f"{task['task_id']} input={only_source or 'all'} phase={args.phase}", flush=True)
            if generating:
                generate(directory, plan, task)
            if not artifact_done(directory / "generation.json", plan["plan_id"]):
                raise RuntimeError("Generation incomplete; run the data phase first")
            if fitting:
                for source in plan["inputs"]:
                    if fit_source(source) != source:
                        continue  # generated, or derived from another fit
                    try:
                        if source == "m4_flex":
                            denoise(directory, plan, task, args.threads)
                        else:
                            denoise_external(directory, plan, task, source, args)
                    except Exception:
                        failures += 1
                        atomic_json(directory / f"denoise_{source}.json", dict(status="failed", plan_id=plan["plan_id"],
                                    error=traceback.format_exc()))
                        traceback.print_exc()
            if testing:
                for source in ([only_source] if only_source else plan["inputs"]):
                    fit = fit_source(source)
                    if fit and not artifact_done(directory / f"denoise_{fit}.json", plan["plan_id"]):
                        print(f"Skipping {source}: {fit} fit not complete", flush=True)
                        continue
                    failures += run_methods(directory, plan, task, source, args)
            # Diagnostics need matrices only, so they run once per dataset, never per input unit.
            if plan.get("effect_diagnostics") and only_source is None and (fitting or testing):
                from da_effect_diagnostics import diagnose_dataset
                diagnose_dataset(directory, plan)
        except Exception:
            failures += 1
            error_file = directory / ("task_error.json" if only_source is None else f"da/{only_source}/task_error.json")
            atomic_json(error_file, dict(plan_id=plan["plan_id"], error=traceback.format_exc()))
            traceback.print_exc()
        finally:
            (lock / "owner.json").unlink()
            lock.rmdir()
    if failures:
        raise SystemExit(f"{failures} failed stages; inspect logs/status and resume")


def summarise(args):
    from da_design import write_csv, GENERATED_INPUTS, fit_source
    plan = load_plan(args.out)
    rows, missing = [], []
    diagnostics, input_index = [], []
    for task in plan["tasks"]:
        directory = args.out / "datasets" / task["task_id"]
        generation_ok = artifact_done(directory / "generation.json", plan["plan_id"])
        for source in plan["inputs"]:
            fit_marker = directory / f"denoise_{fit_source(source) or source}.json"
            fit_ok = generation_ok if source in GENERATED_INPUTS else artifact_done(fit_marker, plan["plan_id"])
            fit = json.loads(fit_marker.read_text(encoding="utf-8")) if fit_ok and source not in GENERATED_INPUTS else {}
            diagnostics.append(task | dict(input=source, generation_complete=generation_ok,
                                          fit_complete=fit_ok, fit_converged=fit.get("converged")))
            matrix = directory / f"{source}.csv"
            input_index.append(task | dict(input=source, valid=generation_ok and fit_ok,
                counts=str((directory / "counts.csv").relative_to(args.out)),
                observed=str((directory / "observed.npz").relative_to(args.out)),
                composition=str(matrix.relative_to(args.out)),
                sha256=file_hash(matrix) if matrix.is_file() else None,
                da_directory=str((directory / "da" / source).relative_to(args.out))))
            for method in plan["methods"]:
                marker = directory / "da" / source / method / "status.json"
                try:
                    info = json.loads(marker.read_text(encoding="utf-8"))
                    if info["plan_id"] != plan["plan_id"]:
                        raise ValueError("Wrong plan")
                except (OSError, ValueError, KeyError):
                    info = dict(status="missing")
                valid = generation_ok and (source in GENERATED_INPUTS or fit_ok) and artifact_done(marker, plan["plan_id"])
                row = task | dict(input=source, method=method, status=info["status"] if valid else "failed_or_missing",
                                  fit_converged=fit.get("converged") if source not in GENERATED_INPUTS else None)
                if valid:
                    row.update(info["metrics"])
                else:
                    missing.append(dict(task_id=task["task_id"], input=source, method=method,
                                        reported_status=info["status"]))
                rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(args.out / "da_metrics.csv", fields, rows)
    write_csv(args.out / "fit_status.csv", list(diagnostics[0]), diagnostics)
    write_csv(args.out / "input_index.csv", list(input_index[0]), input_index)
    summaries = []
    for series, s, source, method in sorted({(r["series"], r["s"], r["input"], r["method"]) for r in rows}):
        subset = [r for r in rows if (r["series"], r["s"], r["input"], r["method"]) == (series, s, source, method)]
        item = dict(series=series, s=s, input=source, method=method, n_expected=len(subset),
                    n_complete=sum(r["status"] == "complete" for r in subset),
                    n_partial=sum(r["status"] == "partial" for r in subset),
                    n_missing_or_failed=sum(r["status"] == "failed_or_missing" for r in subset),
                    n_fit_converged=sum(r["fit_converged"] is True for r in subset))
        # Partial fits remain visible separately; primary estimates require all taxa returned.
        for scope, eligible in (("complete", [r for r in subset if r["status"] == "complete"]),
                                ("complete_converged", [r for r in subset if r["status"] == "complete"
                                                        and (source in GENERATED_INPUTS or r["fit_converged"] is True)]),
                                ("including_partial", [r for r in subset if "fdp" in r])):
            for metric in ("fdp", "power", "precision", "f1", "fpr", "any_false_discovery", "failure_fraction", "boundary_fraction"):
                vals = [r[metric] for r in eligible if r.get(metric) is not None]
                name = "fdr" if metric == "fdp" else metric
                item[f"{scope}_{name}_n"] = len(vals)
                item[f"{scope}_{name}_mean"] = statistics.mean(vals) if vals else None
                item[f"{scope}_{name}_mcse"] = statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else None
        summaries.append(item)
    write_csv(args.out / "da_summary.csv", list(summaries[0]), summaries)
    paired = []
    lookup = {(r["task_id"], r["input"], r["method"]): r for r in rows}
    for task in plan["tasks"]:
        for method in plan["methods"]:
            baseline = lookup[(task["task_id"], "naive", method)]
            common = all(lookup[(task["task_id"], src, method)]["status"] == "complete"
                         for src in plan["inputs"])
            for source in plan["inputs"]:
                if source == "naive":
                    continue
                den = lookup[(task["task_id"], source, method)]
                paired_ok = baseline["status"] == den["status"] == "complete"
                item = task | dict(input=source, baseline="naive", method=method,
                    pair_complete=paired_ok, all_inputs_complete=common, fit_converged=den["fit_converged"])
                for metric in ("fdp", "power", "f1", "fpr"):
                    a, b = baseline.get(metric), den.get(metric)
                    item[metric + "_delta"] = b - a if paired_ok and a is not None and b is not None else None
                paired.append(item)
    write_csv(args.out / "paired_deltas.csv", list(paired[0]), paired)
    paired_summary = []
    for series, s, source, method in sorted({(r["series"], r["s"], r["input"], r["method"]) for r in paired}):
        expected = [r for r in paired if (r["series"], r["s"], r["input"], r["method"]) == (series, s, source, method)]
        for scope in ("pairwise", "all_inputs_common"):
            subset = [r for r in expected if r["pair_complete"] and
                      (scope == "pairwise" or r["all_inputs_complete"])]
            item = dict(series=series, s=s, input=source, baseline="naive", method=method, scope=scope,
                        n_expected=len(expected), n_complete_pairs=len(subset),
                        n_fit_converged=sum(r["fit_converged"] is True for r in subset))
            for metric in ("fdp_delta", "power_delta", "f1_delta", "fpr_delta"):
                vals = [r[metric] for r in subset if r[metric] is not None]
                item[metric + "_n"] = len(vals)
                item[metric + "_mean"] = statistics.mean(vals) if vals else None
                item[metric + "_mcse"] = statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else None
            paired_summary.append(item)
    write_csv(args.out / "paired_summary.csv", list(paired_summary[0]), paired_summary)
    atomic_json(args.out / "status.json", dict(plan_id=plan["plan_id"], n_datasets=len(plan["tasks"]),
                expected_method_inputs=len(rows), complete=sum(r["status"] == "complete" for r in rows),
                partial=sum(r["status"] == "partial" for r in rows), missing_or_failed=missing,
                # Derived inputs share their producing fit; count each fit once.
                fits_expected=len(plan["tasks"]) * sum(fit_source(src) == src for src in plan["inputs"]),
                fits_complete=sum(r["fit_complete"] for r in diagnostics if fit_source(r["input"]) == r["input"]),
                fits_converged=sum(r["fit_converged"] is True for r in diagnostics
                                   if fit_source(r["input"]) == r["input"]),
                note="FDR = mean replicate FDP (no calls => FDP=0). Failed runs are not zeros. Null shared once."))
    print(f"Summarised {len(rows)} expected method/input results; {len(missing)} missing/failed")
    if plan.get("effect_diagnostics"):
        from da_effect_diagnostics import summarise_diagnostics
        summarise_diagnostics(args.out, plan, rows)


def archive_results(args):
    """Downloadable DA results include all taxa/attempts; matrices remain separately retained."""
    import tarfile
    load_plan(args.out)
    destination = args.out / ("da_full_data.tar.gz" if args.full_data else "da_results.tar.gz")
    temporary = destination.with_suffix(".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        for path in sorted(args.out.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(args.out)
            if path == temporary or path.name.endswith((".tar.gz", ".tmp")):
                continue
            parts = relative.parts
            include = args.full_data or parts[0] != "datasets" or "da" in parts or "diagnostics" in parts or path.suffix == ".json" or path.name in {"truth.csv", "metadata.csv"}
            if include:
                archive.add(path, arcname=str(relative), recursive=False)
    temporary.replace(destination)
    print(f"Archived {destination}; source datasets were retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "count", "run", "summarise", "archive", "check"):
        sub = commands.add_parser(name)
        sub.add_argument("--out", type=Path, required=True)
        if name in ("run", "count"):
            sub.add_argument("--unit", choices=["dataset", "input"], default="dataset")
        if name == "archive":
            sub.add_argument("--full-data", action="store_true")
        if name == "prepare":
            sub.add_argument("--external-root", type=Path, default=Path.cwd() / "external")
            sub.add_argument("--external-config", type=Path, help="JSON execution overrides for four external methods")
            sub.add_argument("--seeds", type=int, nargs="+")
            sub.add_argument("--levels", type=int, nargs="+", default=[0, 1])
            sub.add_argument("--effect-strengths", type=float, nargs="+",
                             help="Explicit natural-log effects including 0; overrides the legacy strong/weak grid")
            sub.add_argument("--include-oracle", action="store_true")
            sub.add_argument("--abundance-inputs", action="store_true",
                             help="Also test m4_flex_abundance and oracle_abundance (requires --include-oracle)")
            sub.add_argument("--methods", nargs="+", choices=["linda", "maaslin2", "glmmpql", "zigmm"])
            sub.add_argument("--alpha", type=float, default=.05)
            sub.add_argument("--log-pseudocount", type=float, default=1e-6)
            sub.add_argument("--denoise-group", choices=["blind", "aware"], default="blind")
        if name == "run":
            sub.add_argument("--phase", choices=["all", "generate", "fit", "data", "da"], default="all")
            sub.add_argument("--shard", type=int, default=0)
            sub.add_argument("--shards", type=int, default=1)
            sub.add_argument("--threads", type=int, default=1)
            sub.add_argument("--conda", default=os.environ.get("CONDA_EXE", "conda"))
        if name in ("run", "check"):
            sub.add_argument("--maaslin2-r-bin", default=os.environ.get("MAASLIN2_R_BIN", DEFAULT_MAASLIN2_R_BIN))
            r_launcher = sub.add_mutually_exclusive_group()
            r_launcher.add_argument("--r-bin", default=DEFAULT_R_BIN)
            r_launcher.add_argument("--rscript", help="Optional explicit Rscript executable instead of R")
            sub.add_argument("--retry-partial", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "count":
        print(len(work_units(load_plan(args.out), args.unit)))
    elif args.command == "run":
        run(args)
    elif args.command == "check":
        plan = load_plan(args.out)
        check_dependencies(args, plan["methods"])
    elif args.command == "archive":
        archive_results(args)
    else:
        summarise(args)


if __name__ == "__main__":
    main()
