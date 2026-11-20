"""Reproducible subject x visit x taxon count simulations. See FORMULAS.md."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

import numpy as np
from scipy.special import expit


SCENARIOS = {
    1: "baseline_nb", 2: "group_time", 3: "low_rank_and_specific",
    4: "local_events", 5: "long_events", 6: "two_channel_zinb",
    7: "null_events", 8: "compositional_misspecification",
}


@dataclass
class SimulationConfig:
    scenario: int = 6
    n_subjects: int = 40
    n_visits: int = 16
    n_taxa: int = 80
    n_factors: int = 3
    n_batches: int = 3
    n_events: int = 1
    horizon: float = 100.0
    seed: int = 2026
    mechanism: str | None = None
    median_depth: float = 12000.0
    depth_log_sd: float = 0.5
    batch_sd: float = 0.35
    nb_size: float = 12.0
    dm_concentration: float = 250.0
    effect_scale: float = 0.8
    response_fraction: float = 0.3
    exposure_fraction: float = 0.85
    irregular: bool = False
    missing_rate: float = 0.0
    dropout_rate: float = 0.0
    batch_dispersion: bool = False
    batch_drift: float = 0.0
    outlier_rate: float = 0.0
    persistent_absence: bool = False
    confounding: str = "balanced"
    basis_match: str = "matched"
    trend_mode: str = "smooth"
    residual_multiplier: float = 1.0
    heterogeneity: str = "legacy"  # legacy | gain | shared
    hetero_sd: float = 0.2  # log-gain sd for gain/shared; Gaussian sd for legacy
    subject_delay_sd: float = 0.0  # fraction of horizon
    event_time_error: float = 0.0  # observed event-time error, fraction of horizon
    random_lengths: bool = False


def _gp(rng, times, n_processes, length, smooth=False):
    distance = np.abs(times[:, None] - times[None, :])
    kernel = np.exp(-0.5 * (distance / length) ** 2) if smooth else np.exp(-distance / length)
    return np.linalg.cholesky(kernel + np.eye(len(times)) * 1e-7) @ rng.normal(size=(len(times), n_processes))


def _event_curve(s, kind, decay=.08):
    """s in fractions of the study horizon; every kernel is zero pre-event."""
    x = np.maximum(s, 0.0)
    on = s >= 0
    if kind == 0:
        return on * np.exp(-x / decay)
    if kind == 1:
        return ((s >= 0) & (s <= 0.16)).astype(float)
    if kind == 2:
        return on.astype(float)
    if kind == 3:
        return x
    if kind == 4:
        return on * (0.35 + 0.65 * np.exp(-x / decay))
    if kind == 5:
        return (s >= 0.08) * np.exp(-np.maximum(s - 0.08, 0) / decay)
    if kind == 6:  # gamma-shaped rise and recovery, unit peak
        z = x / decay
        return on * z * np.exp(1 - z)
    if kind == 7:  # overshoot and rebound
        return on * (np.exp(-x / decay) - .65 * (x / decay) * np.exp(1 - x / decay))
    if kind == 8:  # damped oscillation
        return on * np.exp(-x / (2 * decay)) * np.sin(2 * np.pi * x / (3 * decay))
    if kind == 9:  # narrow pulse with delayed peak
        return on * np.exp(-.5 * ((x - decay) / (decay / 3)) ** 2)
    raise ValueError("Unknown event shape")


def simulate(config: SimulationConfig) -> dict[str, np.ndarray]:
    """Return numeric arrays plus scalar JSON metadata; NPZ needs no pickle.

    Y=-1 means unobserved. `mask` is authoritative; no missing count is zero-filled.
    Event times are NaN for unexposed subjects. All biological components are saved.
    """
    c = config
    if c.scenario not in SCENARIOS:
        raise ValueError("scenario must be 1..8")
    if min(c.n_subjects, c.n_visits, c.n_taxa) < 2 or c.n_factors < 1 or c.n_events < 1 or c.n_batches < 1:
        raise ValueError("Need >=2 subjects/visits/taxa and positive factors/events/batches")
    for name in ["response_fraction", "exposure_fraction", "missing_rate", "dropout_rate", "outlier_rate"]:
        if not 0 <= getattr(c, name) <= 1:
            raise ValueError(f"{name} must be in [0,1]")
    if min(c.horizon, c.median_depth, c.nb_size, c.dm_concentration) <= 0:
        raise ValueError("horizon, depth, size, concentration must be positive")
    if min(c.depth_log_sd, c.batch_sd, c.effect_scale, c.batch_drift) < 0:
        raise ValueError("noise/effect scales must be nonnegative")
    if c.confounding not in {"balanced", "partial", "complete"}:
        raise ValueError("Invalid confounding")
    if c.basis_match not in {"matched", "mismatched"} or c.trend_mode not in {"smooth", "local"}:
        raise ValueError("Invalid basis/trend mode")
    if c.heterogeneity not in {"legacy", "gain", "shared"}:
        raise ValueError("Invalid heterogeneity mode")
    if min(c.residual_multiplier, c.hetero_sd, c.subject_delay_sd, c.event_time_error) < 0:
        raise ValueError("New simulation scales must be nonnegative")
    mechanism = c.mechanism or ("zinb" if c.scenario in (6, 7) else "dm" if c.scenario == 8 else "nb")
    if mechanism not in {"nb", "zinb", "multinomial", "dm"}:
        raise ValueError("Unknown count mechanism")
    rng = np.random.default_rng(c.seed)
    # Extra axes use a separate stream; default inputs preserve the original draws.
    extra_rng = np.random.default_rng(np.random.SeedSequence([c.seed, 271828]))
    I, T, J, K, E = c.n_subjects, c.n_visits, c.n_taxa, c.n_factors, c.n_events
    x = np.broadcast_to(np.linspace(0, 1, T), (I, T)).copy()
    if c.irregular:
        x[:, 1:-1] += rng.uniform(-0.4, 0.4, (I, T - 2)) / (T - 1)
    times = x * c.horizon
    group = np.arange(I) % 2
    rng.shuffle(group)
    tau = rng.uniform(0.25, 0.65, (I, E))
    tau.sort(axis=1)
    tau[rng.random((I, E)) > c.exposure_fraction] = np.nan
    tau_eval = np.where(np.isfinite(tau), tau, np.inf)
    batch = rng.integers(c.n_batches, size=(I, T))
    if c.confounding != "balanced":
        if c.n_batches < 2:
            raise ValueError("Confounding experiments need >=2 batches")
        target = (x >= tau_eval[:, :1]).astype(int)
        choose = rng.random((I, T)) < (0.8 if c.confounding == "partial" else 1.0)
        batch[choose] = target[choose]

    alpha = rng.normal(-np.log(J) - 0.5, 1.0, J)
    u = rng.normal(0, 0.3, (I, J))
    kind = rng.integers(0, 6, J)
    G = np.zeros((I, T, J))
    templates = [np.zeros_like(x), x, -x, expit(12 * (x - .5)),
                 np.exp(-((x - .55) / .18) ** 2), np.sin(2 * np.pi * x)]
    for j in range(J):
        g = templates[kind[j]]
        G[:, :, j] = rng.uniform(.25, .65) * (g - g.mean())
    if c.trend_mode == "local":
        for j in range(J):
            centre = extra_rng.uniform(.25, .75)
            g = ((x >= centre).astype(float) if j % 3 == 0 else
                 np.exp(-.5 * ((x - centre) / .05) ** 2) if j % 3 == 1 else
                 np.maximum(x - centre, 0) * 3)
            G[:, :, j] += extra_rng.uniform(.3, .7) * (g - g.mean())
    delta = np.zeros(J)
    D = np.zeros_like(G)
    group_enabled = c.scenario >= 2
    if group_enabled:
        affected = rng.random(J) < .25
        delta[affected] = c.effect_scale * rng.choice([-1, 1], affected.sum())
        dynamic = rng.random(J) < .25
        D[:, :, dynamic] = (x - .5)[:, :, None] * rng.choice([-1, 1], dynamic.sum()) * c.effect_scale
    group_component = group[:, None, None] * (delta[None, None, :] + D)

    F = np.zeros((I, T, K))
    loading = np.zeros((J, K))
    R = np.zeros_like(G)
    factor_length = float(extra_rng.lognormal(np.log(.22), .4)) if c.random_lengths else .22
    residual_length = float(extra_rng.lognormal(np.log(.10), .4)) if c.random_lengths else .10
    if c.scenario >= 3:
        for i in range(I):
            F[i] = _gp(rng, x[i], K, factor_length)
        loading = rng.normal(0, .08, (J, K))
        loading[np.arange(J), np.arange(J) % K] += rng.choice([-1, 1], J) * .3
        specific = rng.random(J) < .25
        for i in range(I):
            R[i][:, specific] = .18 * c.residual_multiplier * _gp(rng, x[i], int(specific.sum()), residual_length, smooth=True)
    shared = np.einsum("itk,jk->itj", F, loading)

    active_events = c.scenario in (4, 5, 6, 8)
    zero_layer = mechanism == "zinb" or (mechanism in {"dm", "multinomial"} and c.scenario >= 6)
    channels = np.zeros((J, E), dtype=int)  # 0 null, 1 abundance, 2 absence, 3 both
    shapes = np.zeros((J, E), dtype=int)
    amp = np.zeros((J, E))
    amp0 = np.zeros((J, E))
    individual_amp = np.zeros((I, J, E))
    event_population = np.zeros((I, T, J))
    event_plus = np.zeros_like(G)
    event_zero = np.zeros_like(G)
    decays = np.full((J, E), .08)
    if c.basis_match == "mismatched":
        decays = extra_rng.lognormal(np.log(.08), .4, (J, E))
    subject_log_gain = extra_rng.normal(size=(I, 1))
    delays = np.abs(extra_rng.normal(0, c.subject_delay_sd, (I, E)))
    for e in range(E):
        if active_events:
            responding = rng.random(J) < c.response_fraction
            channels[responding, e] = rng.integers(1, 4, responding.sum()) if zero_layer else 1
        choices = [0, 1, 5] if c.scenario == 4 else [2, 3, 4] if c.scenario == 5 else list(range(6))
        shapes[:, e] = rng.choice(choices, J)
        if c.basis_match == "mismatched":
            shapes[:, e] = extra_rng.integers(6, 10, J)
        amp[:, e] = c.effect_scale * rng.choice([-1, 1], J) * np.isin(channels[:, e], [1, 3])
        amp0[:, e] = 1.2 * c.effect_scale * rng.choice([-1, 1], J) * np.isin(channels[:, e], [2, 3])
        hetero = rng.normal(1, c.hetero_sd, (I, J)) * (rng.random((I, J)) < .85) / .85
        if c.heterogeneity != "legacy":
            z = extra_rng.normal(size=(I, J))
            if c.heterogeneity == "shared":
                z = (subject_log_gain + z) / np.sqrt(2)
            hetero = np.exp(c.hetero_sd * z - .5 * c.hetero_sd ** 2)
        individual_amp[:, :, e] = amp[:, e][None, :] * hetero * (1 + .3 * group[:, None])
        for j in range(J):
            curve = _event_curve(x - tau_eval[:, e, None], shapes[j, e], decays[j, e])
            event_population[:, :, j] += curve * amp[j, e] * (1 + .3 * group[:, None])
            individual_curve = _event_curve(x - tau_eval[:, e, None] - delays[:, e, None], shapes[j, e], decays[j, e])
            event_plus[:, :, j] += individual_curve * individual_amp[:, j, e, None]
            event_zero[:, :, j] += curve * amp0[j, e]
    Z = alpha + u[:, None, :] + G + group_component + shared + R + event_plus
    absent_intercept = rng.normal(-1.8, .7, J)
    absent_individual = rng.normal(0, .3, (I, J))
    absent_age = rng.normal(0, .25, J)
    absent_group = rng.normal(0, .2, J) if group_enabled else np.zeros(J)
    eta = expit(absent_intercept + absent_individual[:, None, :] +
                (x - .5)[:, :, None] * absent_age + group[:, None, None] * absent_group + event_zero)
    if not zero_layer:
        eta[:] = 0
    S = rng.random((I, T, J)) < eta
    if c.persistent_absence and zero_layer:
        for t in range(1, T):
            persistence = np.exp(-(x[:, t] - x[:, t - 1]) / .15)[:, None]
            S[:, t] = rng.random((I, J)) < (persistence * S[:, t - 1] + (1 - persistence) * eta[:, t])
    if mechanism in {"dm", "multinomial"}:
        for i, t in np.argwhere(S.all(axis=2)):
            S[i, t, np.argmax(Z[i, t])] = False
    X = (~S) * np.exp(Z)
    clean_sum = X.sum(axis=2, keepdims=True)
    clean_composition = np.divide(X, clean_sum, out=np.zeros_like(X), where=clean_sum > 0)
    depth = np.maximum(1, np.rint(rng.lognormal(np.log(c.median_depth), c.depth_log_sd, (I, T)))).astype(np.int64)
    kappa = rng.normal(0, c.batch_sd, (c.n_batches, J))
    kappa[0] = 0.0
    batch_effect = kappa[batch]
    order = rng.permutation(I * T).reshape(I, T) / max(I * T - 1, 1)
    drift = c.batch_drift * (order - .5)[:, :, None] * rng.normal(size=J)
    outlier_sample = rng.random((I, T)) < c.outlier_rate
    outlier = outlier_sample[:, :, None] * rng.normal(0, 1.5, (I, T, J))
    technical = batch_effect + drift + outlier
    r = c.nb_size * np.exp(rng.normal(0, .3, J))
    r_batch = np.broadcast_to(r, (c.n_batches, J)).copy()
    if c.batch_dispersion:
        r_batch *= np.exp(rng.normal(0, .5, (c.n_batches, 1)))
    mu_present = depth[:, :, None] * np.exp(Z + technical)
    observed_prob = np.zeros_like(X)
    if mechanism in {"nb", "zinb"}:
        r_it = r_batch[batch]
        Y = rng.negative_binomial(r_it, r_it / (r_it + mu_present))
        Y[S] = 0
    else:
        # Normalise in log space, and restrict Dirichlet draws to present taxa.
        logw = np.where(S, -np.inf, Z + technical)
        weight = np.exp(logw - np.max(logw, axis=2, keepdims=True))
        observed_prob = weight / weight.sum(axis=2, keepdims=True)
        Y = np.zeros((I, T, J), dtype=np.int64)
        for i in range(I):
            for t in range(T):
                present = ~S[i, t]
                prob = observed_prob[i, t, present]
                if mechanism == "dm":
                    prob = rng.dirichlet(c.dm_concentration * prob)
                Y[i, t, present] = rng.multinomial(int(depth[i, t]), prob)
    dropout_probability = c.dropout_rate * expit((np.log(10) - np.log1p(mu_present)) + batch[:, :, None] * .3) * 2
    dropout = rng.random((I, T, J)) < np.clip(dropout_probability, 0, 1)
    Y[dropout] = 0
    visit_mask = rng.random((I, T)) >= c.missing_rate
    # Every individual has at least two observed visits, for a usable dataset.
    visit_mask[:, [0, -1]] = True
    mask = np.broadcast_to(visit_mask[:, :, None], Y.shape).copy()
    Y[~mask] = -1
    metadata = asdict(c) | {"mechanism": mechanism, "scenario_name": SCENARIOS[c.scenario],
                           "target": "clean_composition" if mechanism in {"dm", "multinomial"} else "X_true",
                           "schema_version": 2, "factor_length_true": factor_length,
                           "residual_length_true": residual_length}
    observed_tau = tau.copy()
    if c.event_time_error:
        observed_tau += extra_rng.normal(0, c.event_time_error, tau.shape)
    # Store dense event truths independently of the inference dictionary. The
    # evaluator interpolates these, including for future non-dictionary shapes.
    truth_grid = np.linspace(-.3, 1.5, 3601)
    truth_curves = np.array([[_event_curve(truth_grid, int(shapes[j, e]), decays[j, e])
                             for e in range(E)] for j in range(J)])
    return dict(Y=Y, mask=mask, times=times, group=group, batch=batch, depth=depth,
                event_times=observed_tau * c.horizon, event_times_true=tau * c.horizon,
                event_subject_delay=delays * c.horizon, event_decay_true=decays,
                event_truth_grid=truth_grid * c.horizon,
                H_plus_true=truth_curves * amp[..., None], H_zero_true=truth_curves * amp0[..., None],
                Z_true=Z, X_true=X, clean_composition=clean_composition,
                S_true=S, eta_true=eta, alpha=alpha, individual=u, trend=G,
                group_effect=group_component, group_delta=delta, group_trend=D,
                factors=F, loadings=loading, shared=shared, specific=R,
                event_plus=event_plus, event_zero=event_zero, event_population=event_population,
                event_channels=channels, event_shapes=shapes, event_amplitude=amp,
                event_amplitude_zero=amp0, event_individual_amplitude=individual_amp,
                batch_coefficients=kappa, batch_effect=batch_effect, technical_effect=technical,
                nb_size=r, nb_size_batch=r_batch, mu_present=mu_present,
                observed_probability=observed_prob, technical_dropout=dropout,
                outlier_sample=outlier_sample, processing_order=order,
                metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)))


ALIASES = {"n_subjects": "--subjects", "n_visits": "--visits", "n_taxa": "--taxa"}
CHOICES = {"mechanism": ["nb", "zinb", "multinomial", "dm"],
           "confounding": ["balanced", "partial", "complete"],
           "basis_match": ["matched", "mismatched"], "trend_mode": ["smooth", "local"],
           "heterogeneity": ["legacy", "gain", "shared"]}


def add_config_flags(parser):
    """Expose every SimulationConfig field, typed by its default, so no knob is
    reachable only from Python. Booleans become store_true switches."""
    for field in fields(SimulationConfig):
        if field.name == "scenario":
            continue
        names = ["--" + field.name.replace("_", "-")] + ([ALIASES[field.name]] if field.name in ALIASES else [])
        if field.name in CHOICES:
            parser.add_argument(*names, choices=CHOICES[field.name], default=field.default)
        elif isinstance(field.default, bool):
            parser.add_argument(*names, action="store_true")
        else:
            parser.add_argument(*names, type=type(field.default), default=field.default)
    return parser


def config_from_namespace(args, **override):
    return SimulationConfig(**{f.name: getattr(args, f.name) for f in fields(SimulationConfig)
                               if f.name != "scenario"} | override)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=int, choices=SCENARIOS, default=6)
    parser.add_argument("--all", action="store_true", help="Write all eight scenarios")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    args = add_config_flags(parser).parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS if args.all else [args.scenario]:
        c = config_from_namespace(args, scenario=scenario)
        path = args.out / f"scenario_{scenario}_{SCENARIOS[scenario]}.npz"
        data = simulate(c)
        np.savez_compressed(path, **data)
        print(f"{path}: shape={data['Y'].shape}, observed={data['mask'].mean():.3f}")


if __name__ == "__main__":
    main()
