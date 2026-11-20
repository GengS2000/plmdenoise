"""Predeclared v3 development screen; not a scientific pass certificate."""
import numpy as np

MARGINS = {"count_log_rmse": .02, "comp_bray_curtis": .02, "event_plus_rmse": .05}
ADJACENT = {"m1_budget": "v2", "m2_joint": "m1_budget", "m3_pool": "m2_joint",
            "m4_flex": "m3_pool", "m5_guard": "m4_flex"}

def acceptance(rows):
    """Paired relative changes, averaged over fit repeats before data seeds.

    1.96*SE is a descriptive normal approximation, NOT multiplicity-adjusted
    inference. Small samples and unstable fits remain inconclusive.
    """
    lookup = {(r["scenario"],r["condition"],r["seed"],r["fit_repeat"],r["method"]):r for r in rows}
    output = []
    for method in sorted({r["method"] for r in rows} - {"v2"}):
        for reference in dict.fromkeys(["v2", ADJACENT.get(method, "v2")]):
            for scenario, condition in sorted({(r["scenario"],r["condition"]) for r in rows if r["method"] == method}):
                group = [r for r in rows if (r["method"],r["scenario"],r["condition"]) == (method,scenario,condition)]
                for metric, margin in MARGINS.items():
                    by_seed = {}; stable = True
                    for row in group:
                        base = lookup.get((scenario,condition,row["seed"],row["fit_repeat"],reference))
                        if base is None: continue
                        a,b = row.get(metric),base.get(metric)
                        if a is None or b is None or not np.isfinite([a,b]).all(): continue
                        by_seed.setdefault(row["seed"],[]).append((a-b)/max(abs(b),1e-8))
                        stable &= bool(row.get("converged")) and bool(base.get("converged"))
                    values = np.array([np.mean(v) for v in by_seed.values()])
                    if not values.size: continue
                    mean = float(values.mean())
                    se = float(values.std(ddof=1)/np.sqrt(values.size)) if values.size > 1 else None
                    upper = mean + 1.96*se if se is not None else None
                    verdict = "inconclusive"
                    if values.size >= 10 and stable and upper is not None:
                        verdict = "within_margin_screen" if upper <= margin else "not_within_margin"
                    output.append(dict(method=method,reference=reference,scenario=scenario,condition=condition,
                                       metric=metric,n_seeds=int(values.size),relative_delta=mean,se=se,
                                       upper_approx95=upper,margin=margin,stable=stable,verdict=verdict))
    return dict(rows=output, note="Exploratory paired screen. No automatic model promotion. Also inspect power, FDR, coverage, width and testable fraction.")
