"""Event-aware longitudinal NB/ZINB denoising (PyTorch, CPU or CUDA).

Concrete implementation of the supplied method draft: marginalised zero state,
multi-start MAP, structured Gaussian VI, and joint posterior zero responsibility.
See README.md for the public interface and output interpretation.
This module is standalone: it does not import the simulation code.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import warnings

import numpy as np
from scipy.interpolate import BSpline
import torch
from torch import nn
from torch.nn import functional as fn


EVENT_NAMES = ("decay", "pulse", "step", "ramp", "delayed_decay")


@dataclass
class DenoiseConfig:
    # v3 extensions are opt-in; all-off retains the v2 stochastic path.
    adaptive_budget: bool = False
    budget_multiplier: int = 3
    joint_rank: int = 0
    joint_scale: float = .10
    vi_max_recoveries: int = 5
    vi_recovery_factor: float = .20
    hierarchical_observation: bool = False
    flexible_event: bool = False
    flex_scale: float = .25
    observation: str = "count"  # count (NB/ZINB) or dm (composition only)
    support_guard: bool = False
    support_radius: float = .10
    min_pre_visits: int = 2
    batch_scale: float = .5
    n_factors: int = 3
    spline_df: int = 5
    zero_inflated: bool = True
    event_basis: tuple[str, ...] = EVENT_NAMES
    event_decay: float = 0.08  # fractions of training time range
    event_window: float = 0.16
    event_delay: float = 0.08
    factor_length: float = 0.22
    residual_length: float = 0.10
    residual: str = "basis"  # none | basis | full; full reproduces the original GP
    residual_df: int = 3
    residual_scale: float = 0.05
    individual_event: str = "gain"  # none | gain | basis
    gain_scale: float = 0.3
    individual_covariance: str = "diagonal"  # full remains available for ablations
    event_prior: str = "regularized"  # legacy | regularized (continuous, not PIP)
    event_global_scale: float = 0.05
    zero_global_scale: float = 0.03
    event_local_log_sd: float = 1.5
    event_slab_scale: float = 1.0
    early_stopping: bool = True
    check_every: int = 50
    patience: int = 3
    min_map_steps: int = 150
    min_vi_steps: int = 300
    convergence_tol: float = 1e-4  # objective improvement per observed cell
    validation_samples: int = 8
    event_support_fraction: float = 0.5
    map_steps: int = 600
    map_starts: int = 2
    map_lr: float = 0.025
    vi_steps: int = 2000
    vi_lr: float = 0.01
    vi_samples: int = 2
    posterior_samples: int = 200
    interval: float = 0.95
    effect_threshold: float = 0.2
    seed: int = 2026
    device: str = "cpu"
    verbose: bool = True


def m4_flex_config(**overrides) -> DenoiseConfig:
    """Return the M4_flex preset from code_v3_1, with optional overrides."""
    options = dict(adaptive_budget=True, joint_rank=4,
                   hierarchical_observation=True, flexible_event=True)
    return DenoiseConfig(**(options | overrides))


def legacy_config(**overrides):
    """Explicit original model; also used to load pre-v2 checkpoints."""
    values = dict(residual="full", residual_scale=.12, individual_event="basis",
                  individual_covariance="full", event_prior="legacy", early_stopping=False,
                  map_steps=300, vi_steps=500)
    return DenoiseConfig(**(values | overrides))


def event_basis(relative_time, names=EVENT_NAMES, decay=.08, window=.16, delay=.08):
    """Dimensionless relative time; NaN event times produce all-zero bases."""
    s = np.asarray(relative_time, dtype=float)
    x = np.maximum(np.nan_to_num(s, nan=-1e10), 0)
    on = np.isfinite(s) & (s >= 0)
    basis = {
        "decay": on * np.exp(-x / decay),
        "pulse": (np.isfinite(s) & (s >= 0) & (s <= window)).astype(float),
        "step": on.astype(float), "ramp": on * x,
        "delayed_decay": (np.isfinite(s) & (s >= delay)) * np.exp(-np.maximum(x - delay, 0) / decay),
    }
    for index, center in enumerate((.04, .12, .28, .50)):
        # Compact triangular correction bases, zero before the event.
        basis[f"flex{index}"] = on * np.maximum(1 - np.abs(x - center) / (.06, .10, .18, .25)[index], 0)
    if not names or len(set(names)) != len(names) or any(n not in basis for n in names):
        raise ValueError(f"event_basis must be a nonempty unique subset of {EVENT_NAMES}")
    return np.stack([basis[n] for n in names], axis=-1)


def nb_log_prob(y, log_mu, size):
    """Stable NB2 log probability: Var(Y)=mu+mu^2/size."""
    log_r = torch.log(size)
    log_sum = torch.logaddexp(log_r, log_mu)
    return (torch.lgamma(y + size) - torch.lgamma(size) - torch.lgamma(y + 1)
            + size * (log_r - log_sum) + y * (log_mu - log_sum))


def zinb_log_prob(y, log_mu, size, eta_logit):
    log_nb = nb_log_prob(y, log_mu, size)
    present = fn.logsigmoid(-eta_logit) + log_nb
    return torch.where(y == 0, torch.logaddexp(fn.logsigmoid(eta_logit), present), present)


def zero_responsibility(y, log_mu, size, eta_logit):
    log_nb0 = nb_log_prob(torch.zeros_like(y), log_mu, size)
    # posterior odds of absence = prior odds / P_NB(Y=0)
    return torch.where(y == 0, torch.sigmoid(eta_logit - log_nb0), torch.zeros_like(y))


def dm_log_prob(y, logits, concentration):
    """Joint Dirichlet-multinomial probability conditional on visit total."""
    alpha = torch.softmax(logits, -1).clamp_min(1e-12) * concentration
    total = y.sum(-1)
    mass = alpha.sum(-1)
    return (torch.lgamma(total + 1) - torch.lgamma(y + 1).sum(-1)
            + torch.lgamma(mass) - torch.lgamma(mass + total)
            + (torch.lgamma(alpha + y) - torch.lgamma(alpha)).sum(-1))


class _GaussianBlock(nn.Module):
    """Normal q with full covariance in the last axis, or independent scalars.

    Variables have standard normal priors. Non-centred hyperparameter transforms
    occur in the model; no estimated normalising constants are dropped.
    """
    def __init__(self, shape, full=False, initial=None, device="cpu"):
        super().__init__()
        self.full = full
        dtype = torch.float64
        self.loc = nn.Parameter(torch.zeros(shape, dtype=dtype, device=device))
        if initial is not None:
            with torch.no_grad():
                self.loc.copy_(torch.as_tensor(initial, dtype=dtype, device=device))
        covariance_shape = (*shape, shape[-1]) if full else shape
        self.raw = nn.Parameter(torch.zeros(covariance_shape, dtype=dtype, device=device))
        with torch.no_grad():
            if full:
                self.raw.diagonal(dim1=-2, dim2=-1).fill_(math.log(.04))
            else:
                self.raw.fill_(math.log(.04))

    def scale(self):
        if self.full:
            return torch.tril(self.raw, -1) + torch.diag_embed(self.raw.diagonal(dim1=-2, dim2=-1).exp())
        return self.raw.exp()

    def draw(self):
        eps = torch.randn_like(self.loc)
        scale = self.scale()
        return self.loc + ((scale @ eps.unsqueeze(-1)).squeeze(-1) if self.full else scale * eps)

    def kl(self):
        scale = self.scale()
        logdet = self.raw.diagonal(dim1=-2, dim2=-1).sum() if self.full else self.raw.sum()
        return .5 * (self.loc.square().sum() + scale.square().sum() - self.loc.numel() - 2 * logdet)


class _Model(nn.Module):
    def __init__(self, arrays, config, seed):
        super().__init__()
        self.c = config
        torch.manual_seed(seed)
        self.device = config.device
        dtype = torch.float64
        for key, value in arrays.items():
            if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
                tensor = torch.as_tensor(value, dtype=torch.bool if value.dtype.kind == "b" else dtype, device=self.device)
                self.register_buffer(key, tensor)
        I, T, J = arrays["y"].shape
        self.I, self.T, self.J = I, T, J
        K, P = config.n_factors, arrays["spline"].shape[-1]
        E, Q = arrays["q_event"].shape[-2:]
        B = arrays["batch_design"].shape[-1]
        self.blocks = nn.ModuleDict()

        def block(name, shape, full=False, initial=None):
            self.blocks[name] = _GaussianBlock(shape, full, initial, self.device)

        rates = arrays["y"] / arrays["depth"][:, :, None]
        avg = np.sum(rates * arrays["mask"], axis=(0, 1)) / np.maximum(arrays["mask"].sum(axis=(0, 1)), 1)
        block("alpha", (J,), initial=(np.log(np.maximum(avg, 1e-7)) + np.log(J)) / 2)
        block("u", (I, J)); block("scale_u", (J,))
        block("G", (J, P), True)
        block("group", (J,)); block("group_time", (J, P), True)
        block("loading", (J, K), initial=np.random.default_rng(seed).normal(0, .2, (J, K)))
        block("F", (I, K, T), True, np.random.default_rng(seed + 1).normal(0, .1, (I, K, T)))
        individual_full = config.individual_covariance == "full"
        if config.residual != "none":
            block("R", (I, J, T if config.residual == "full" else arrays["residual_basis"].shape[-1]), individual_full)
            block("scale_R", (J,))
        block("theta", (J, E, Q), True); block("scale_theta", (J,))
        block("theta_group", (J, E, Q), True)
        if config.individual_event != "none":
            block("d", (I, J, E, Q) if config.individual_event == "basis" else (I, J, E),
                  individual_full if config.individual_event == "basis" else False)
            block("scale_d", (J,))
        if config.event_prior == "regularized":
            block("global_theta", (1,)); block("local_theta", (J, E))
        block("kappa", (B, J)); block("log_r", (J,))
        if config.zero_inflated:
            block("c", (J,)); block("v", (I, J)); block("scale_v", (J,))
            block("A", (J, P), True); block("zero_group", (J,))
            block("theta0", (J, E, Q), True); block("scale_theta0", (J,))
            if config.event_prior == "regularized":
                block("global_theta0", (1,)); block("local_theta0", (J, E))

        # Legacy scales are not part of the regularized prior; remove unused blocks.
        if config.event_prior == "regularized":
            del self.blocks["scale_theta"]
            if config.zero_inflated:
                del self.blocks["scale_theta0"]
        if config.hierarchical_observation:
            block("r_global", (1,)); block("r_scale", (1,))
            if config.zero_inflated:
                block("c_global", (1,)); block("c_scale", (1,))
        if config.observation == "dm":
            block("dm_concentration", (1,))
        self.joint = nn.ParameterDict()
        if config.joint_rank:
            for name in ("theta", "theta_group", "theta0", "G", "group_time", "kappa", "R", "F"):
                if name in self.blocks:
                    self.joint[name] = nn.Parameter(torch.randn(
                        (*self.blocks[name].loc.shape, config.joint_rank), dtype=dtype, device=self.device) * .01)

    def joint_factors(self):
        """Bound shared factors so one Adam proposal cannot explode q samples."""
        return {name: self.c.joint_scale * torch.tanh(raw) for name, raw in self.joint.items()}

    def event_scale(self, w, zero=False):
        suffix = "theta0" if zero else "theta"
        if self.c.event_prior == "legacy":
            return torch.exp(math.log(.35 if zero else .25) + .7 * w["scale_" + suffix])[:, None, None]
        # Proper lognormal global/local priors, with a finite upper slab scale.
        # Every underlying w has an N(0,1) prior; the Gaussian KL remains exact.
        base = self.c.zero_global_scale if zero else self.c.event_global_scale
        log_scale = math.log(base) + .8 * w["global_" + suffix] + self.c.event_local_log_sd * w["local_" + suffix]
        log_slab = math.log(self.c.event_slab_scale)
        scale = torch.exp(log_scale - .5 * fn.softplus(2 * (log_scale - log_slab)))
        return scale[..., None]

    def values(self, sample):
        values = {name: b.draw() if sample else b.loc for name, b in self.blocks.items()}
        if sample and self.c.joint_rank:
            shared = torch.randn(self.c.joint_rank, dtype=torch.float64, device=self.device)
            for name, factor in self.joint_factors().items():
                values[name] = values[name] + factor @ shared
        return values

    def kl(self):
        value = sum(b.kl() for b in self.blocks.values())
        if self.c.joint_rank:
            gram = torch.eye(self.c.joint_rank, dtype=torch.float64, device=self.device)
            trace = torch.zeros((), dtype=torch.float64, device=self.device)
            for name, factor in self.joint_factors().items():
                block = self.blocks[name]
                whitened = (torch.linalg.solve_triangular(block.scale(), factor, upper=False)
                            if block.full else factor / block.scale()[..., None])
                flat = whitened.reshape(-1, self.c.joint_rank)
                gram = gram + flat.T @ flat
                trace = trace + factor.square().sum()
            value = value + .5 * (trace - torch.linalg.slogdet(gram)[1])
        return value

    def decode(self, w):
        # Lognormal shrinkage scales with proper N(0,1) hyperpriors.
        su = torch.exp(math.log(.3) + .5 * w["scale_u"])
        st = self.event_scale(w)
        alpha = -math.log(self.J) + 2 * w["alpha"]
        individual = w["u"] * su
        trend = torch.einsum("itp,jp->itj", self.spline, .7 * w["G"])
        group = self.group[:, None, None] * (.7 * w["group"] + torch.einsum("itp,jp->itj", self.spline, .5 * w["group_time"]))
        factors = torch.einsum("itv,ikv->itk", self.factor_chol, w["F"])
        loading = .4 * w["loading"]  # factor prior marginal variance fixed to one
        shared = torch.einsum("itk,jk->itj", factors, loading)
        specific = torch.zeros_like(shared)
        if self.c.residual != "none":
            sr = torch.exp(math.log(self.c.residual_scale) + .7 * w["scale_R"])
            if self.c.residual == "full":
                # Original hyperprior width, for a clean legacy comparison.
                sr = torch.exp(math.log(self.c.residual_scale) + .5 * w["scale_R"])
            basis = self.residual_chol if self.c.residual == "full" else self.residual_basis
            specific = torch.einsum("itv,ijv->itj", basis, w["R"]) * sr
        theta = w["theta"] * st
        theta_group = w["theta_group"] * st * .5
        flex_weight = torch.tensor([self.c.flex_scale if name.startswith("flex") else 1.
                                    for name in self.c.event_basis], dtype=torch.float64, device=self.device)
        theta = theta * flex_weight
        theta_group = theta_group * flex_weight
        population_coeff = theta[None] + self.group[:, None, None, None] * theta_group[None]
        d = torch.zeros_like(population_coeff)
        if self.c.individual_event == "basis":
            sd = torch.exp(math.log(.10) + .5 * w["scale_d"])
            d = w["d"] * sd[None, :, None, None]
            if self.c.event_prior == "regularized":
                # Prevent the individual channel from bypassing event shrinkage.
                d = d * st[None] / self.c.event_slab_scale
        elif self.c.individual_event == "gain":
            sd = torch.exp(math.log(self.c.gain_scale) + .5 * w["scale_d"])
            gain = w["d"] * sd[None, :, None]
            d = gain[..., None] * population_coeff
        event_population = torch.einsum("iteq,ijeq->itj", self.q_event, population_coeff)
        event = event_population + torch.einsum("iteq,ijeq->itj", self.q_event, d)
        Z = alpha + individual[:, None, :] + trend + group + shared + specific + event
        batch = torch.einsum("itb,bj->itj", self.batch_design, self.c.batch_scale * w["kappa"])
        log_mu = self.depth.log()[:, :, None] + Z + batch
        size = torch.exp(math.log(10) + w["log_r"])
        if self.c.hierarchical_observation:
            size = torch.exp(math.log(10) + .7 * w["r_global"] +
                             .5 * torch.exp(.3 * w["r_scale"]) * w["log_r"])
        eta_logit = torch.full_like(Z, -torch.inf)
        theta0 = torch.zeros_like(theta)
        if self.c.zero_inflated:
            sv = torch.exp(math.log(.3) + .5 * w["scale_v"])
            theta0 = w["theta0"] * self.event_scale(w, zero=True) * flex_weight
            eta_logit = (-2 + 1.5 * w["c"] + (w["v"] * sv)[:, None, :]
                         + torch.einsum("itp,jp->itj", self.spline, .5 * w["A"])
                         + self.group[:, None, None] * .5 * w["zero_group"]
                         + torch.einsum("iteq,jeq->itj", self.q_event, theta0))
            if self.c.hierarchical_observation:
                eta_logit = eta_logit - 1.5 * w["c"] + w["c_global"] + torch.exp(.3 * w["c_scale"]) * w["c"]
        if self.c.observation == "dm":
            size = torch.exp(math.log(250) + w["dm_concentration"])
        return dict(Z=Z, log_mu=log_mu, size=size, eta_logit=eta_logit, theta=theta,
                    theta_group=theta_group, theta0=theta0, d=d, trend=trend,
                    group_effect=group, shared=shared, specific=specific, event_plus=event,
                    event_population=event_population,
                    event_zero=torch.einsum("iteq,jeq->itj", self.q_event, theta0),
                    batch_effect=batch, factors=factors, loadings=loading)

    def log_likelihood(self, decoded):
        if self.c.observation == "dm":
            return dm_log_prob(self.y, decoded["Z"] + decoded["batch_effect"], decoded["size"])[self.mask.all(-1)].sum()
        if self.c.zero_inflated:
            ll = zinb_log_prob(self.y, decoded["log_mu"], decoded["size"], decoded["eta_logit"])
        else:
            ll = nb_log_prob(self.y, decoded["log_mu"], decoded["size"])
        return ll[self.mask].sum()

    def map_objective(self):
        w = self.values(False)
        return -self.log_likelihood(self.decode(w)) + .5 * sum(v.square().sum() for v in w.values())

    def negative_elbo(self):
        ll = sum(self.log_likelihood(self.decode(self.values(True))) for _ in range(self.c.vi_samples)) / self.c.vi_samples
        return -ll + self.kl()


def _prepare(Y, times, depth, batch, event_times, group, mask, config):
    y = np.asarray(Y, dtype=float)
    if y.ndim != 3:
        raise ValueError("Y must be subject x visit x taxon")
    I, T, J = y.shape
    if min(I, T, J) < 1 or T < 2:
        raise ValueError("Need at least one subject/taxon and two visit slots")
    mask = (np.isfinite(y) & (y >= 0)) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != y.shape or not mask.any():
        raise ValueError("mask must match Y and include observations")
    if np.any(~np.isfinite(y[mask])) or np.any(y[mask] < 0) or np.any(y[mask] != np.floor(y[mask])):
        raise ValueError("Observed counts must be finite nonnegative integers")
    if np.any(mask.sum(axis=(0, 1)) == 0):
        raise ValueError("Each taxon needs an observed entry")
    times, depth = np.asarray(times, float), np.asarray(depth, float)
    batch = np.asarray(batch)
    if times.shape != (I, T) or depth.shape != (I, T) or batch.shape != (I, T):
        raise ValueError("times, depth, batch must have shape (subjects, visits)")
    if not np.all(np.isfinite(times)) or not np.all(np.diff(times, axis=1) > 0):
        raise ValueError("All visit times must be finite and strictly increasing per subject")
    if not np.all(np.isfinite(depth) & (depth > 0)):
        raise ValueError("Depth/exposure must be finite and positive, including missing visit slots")
    if batch.dtype.kind not in "iu" or np.any(batch < 0):
        raise ValueError("Batch labels must be nonnegative integers")
    group = np.zeros(I) if group is None else np.asarray(group, float)
    if group.shape != (I,) or not np.all(np.isin(group, [0, 1])):
        raise ValueError("group must be a binary vector of length subjects")
    event_times = np.full((I, 1), np.nan) if event_times is None else np.asarray(event_times, float)
    if event_times.ndim == 1:
        event_times = event_times[:, None]
    if event_times.ndim != 2 or event_times.shape[0] != I or event_times.shape[1] < 1 or np.any(np.isinf(event_times)):
        raise ValueError("event_times must be (subjects, event types); use NaN for unexposed")
    span = float(times.max() - times.min())
    x = (times - times.min()) / span
    event_x = (event_times - times.min()) / span
    q = event_basis(x[:, :, None] - event_x[:, None, :], config.event_basis,
                    config.event_decay, config.event_window, config.event_delay)
    n = config.spline_df
    knots = np.r_[np.zeros(4), np.linspace(0, 1, n - 2)[1:-1], np.ones(4)]
    design = BSpline(knots, np.eye(n), 3)(x)
    observed_visit = mask.any(axis=2)
    if config.observation == "dm" and np.any(mask.any(-1) != mask.all(-1)):
        raise ValueError("DM requires complete taxon vectors or fully missing visits")
    # Drop one basis column after centring to remove the constant direction.
    center = design[observed_visit].mean(axis=0)
    spline = (design - center)[..., :-1]
    levels = np.unique(batch[observed_visit])
    if np.any(~np.isin(batch, levels)):
        raise ValueError("A batch has no observed visits; cannot estimate its effect")
    batch_design = (batch[:, :, None] == levels[1:]).astype(float)
    distance = np.abs(x[:, :, None] - x[:, None, :])
    eye = np.eye(T)[None]
    factor_chol = np.linalg.cholesky(np.exp(-distance / config.factor_length) + 1e-6 * eye)
    residual_chol = (np.linalg.cholesky(np.exp(-.5 * (distance / config.residual_length) ** 2) + 1e-6 * eye)
                     if config.residual == "full" else np.zeros((I, T, 0)))
    # Coarse cubic basis, centred within subject to separate it from u_ij.
    nr = config.residual_df + 1
    rk = np.r_[np.zeros(4), np.linspace(0, 1, nr - 2)[1:-1], np.ones(4)]
    rb = BSpline(rk, np.eye(nr), 3)(x)[..., :-1]
    rb -= (rb * observed_visit[..., None]).sum(1, keepdims=True) / np.maximum(observed_visit.sum(1)[:, None, None], 1)
    rms = np.sqrt(np.mean(np.sum(rb[observed_visit] ** 2, axis=-1)))
    rb /= max(rms, 1e-12)  # unit average marginal variance under iid coefficients
    diagnostic = []
    relative = x[:, :, None] - event_x[:, None, :]
    bounds, pre_counts, correlations = [], [], []
    for i in range(I):
        bounds.append([]); pre_counts.append([])
        for e in range(event_x.shape[1]):
            s = relative[i, observed_visit[i], e]
            finite = s[np.isfinite(s)]
            bounds[i].append([float(finite.min()), float(finite.max())] if finite.size else [None, None])
            pre_counts[i].append(int((finite < 0).sum()) if finite.size else None)
        a = q[i, observed_visit[i]].reshape(int(observed_visit[i].sum()), -1)
        btrend = spline[i, observed_visit[i]]
        def orthogonal_columns(matrix):
            matrix = matrix - matrix.mean(0)
            u, s, _ = np.linalg.svd(matrix, full_matrices=False)
            return u[:, s > max(float(s.max()) if s.size else 0, 1) * 1e-9]
        ua, ub = orthogonal_columns(a), orthogonal_columns(btrend)
        correlations.append(float(np.linalg.svd(ua.T @ ub, compute_uv=False).max())
                            if ua.shape[1] and ub.shape[1] else None)
    valid_pre = [v for row in pre_counts for v in row if v is not None]
    valid_corr = [v for v in correlations if v is not None]
    if valid_pre and min(valid_pre) < 3:
        diagnostic.append("Some exposed subjects have fewer than three pre-event visits.")
    if valid_corr and max(valid_corr) > .95:
        diagnostic.append("High event/trend basis overlap; separated event curves may be prior-sensitive.")
    bio = np.concatenate([np.ones((I, T, 1)), np.broadcast_to(group[:, None, None], (I, T, 1)),
                          spline, spline * group[:, None, None], q.reshape(I, T, -1),
                          q.reshape(I, T, -1) * group[:, None, None]], axis=-1)[observed_visit]
    b = batch_design[observed_visit]
    if b.shape[1]:
        residual = b - bio @ np.linalg.lstsq(bio, b, rcond=None)[0]
        if np.linalg.matrix_rank(residual, tol=1e-7) < b.shape[1]:
            diagnostic.append("Batch effects are confounded with biological design; separation is prior-dependent.")
        elif np.linalg.norm(residual) / max(np.linalg.norm(b), 1e-12) < .15:
            diagnostic.append("Strong batch/biological confounding: interpret separated components cautiously.")
    if np.linalg.matrix_rank(bio) < bio.shape[1]:
        diagnostic.append("Some biological design columns are collinear or unobserved; event coefficients may be prior-driven.")
    if config.n_factors > min(J, I * T):
        raise ValueError("n_factors exceeds available dimensions")
    for message in diagnostic:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return dict(y=np.where(mask, y, 0), mask=mask, times=times, depth=depth, group=group,
                spline=spline, batch_design=batch_design, q_event=q,
                factor_chol=factor_chol, residual_chol=residual_chol, residual_basis=rb), dict(
                    time_origin=float(times.min()), time_span=span, batch_levels=levels.tolist(),
                    diagnostics=diagnostic, spline_knots=knots.tolist(), spline_center=center.tolist(),
                    event_relative_bounds=bounds, event_pre_visits=pre_counts,
                    event_relative_visits=[[relative[i, observed_visit[i], e].tolist()
                                            for e in range(event_x.shape[1])] for i in range(I)],
                    batch_biological_confounded=any("separation is prior-dependent" in s for s in diagnostic),
                    event_trend_canonical_correlation=correlations)


class EventAwareDenoiser:
    """Fit on one longitudinal tensor. No simulated truth is consumed by fit()."""
    def __init__(self, config: DenoiseConfig | None = None):
        self.config = m4_flex_config() if config is None else config
        c = self.config
        if c.flexible_event and not any(n.startswith("flex") for n in c.event_basis):
            c.event_basis = tuple(c.event_basis) + tuple(f"flex{k}" for k in range(4))
        if c.joint_rank < 0 or c.joint_rank > 8 or c.budget_multiplier < 1 or c.flex_scale <= 0 or c.joint_scale <= 0:
            raise ValueError("Invalid v3 rank, budget multiplier or flex scale")
        if c.vi_max_recoveries < 0 or not 0 < c.vi_recovery_factor < 1:
            raise ValueError("Invalid VI recovery configuration")
        if c.support_radius <= 0 or c.min_pre_visits < 0 or c.batch_scale <= 0:
            raise ValueError("Invalid support or batch prior configuration")
        if c.observation not in {"count", "dm"} or (c.observation == "dm" and c.zero_inflated):
            raise ValueError("DM must explicitly disable zero_inflated")
        if c.n_factors < 1 or c.spline_df < 4 or c.map_steps < 1 or c.map_starts < 1:
            raise ValueError("Invalid factor/spline/MAP configuration")
        if min(c.vi_steps, c.vi_samples, c.posterior_samples) < 1:
            raise ValueError("VI steps, MC samples and posterior samples must be positive")
        if min(c.map_lr, c.vi_lr, c.factor_length, c.residual_length, c.event_decay, c.event_window) <= 0:
            raise ValueError("Learning rates and length scales must be positive")
        if not 0 < c.interval < 1 or c.effect_threshold < 0 or c.event_delay < 0:
            raise ValueError("Invalid interval/threshold/delay")
        if c.residual not in {"none", "basis", "full"} or c.individual_event not in {"none", "gain", "basis"}:
            raise ValueError("Invalid residual or individual_event mode")
        if c.individual_covariance not in {"diagonal", "full"} or c.event_prior not in {"legacy", "regularized"}:
            raise ValueError("Invalid covariance or event prior")
        if c.residual_df < 3 or min(c.residual_scale, c.gain_scale, c.event_global_scale,
                                   c.zero_global_scale, c.event_local_log_sd, c.event_slab_scale) <= 0:
            raise ValueError("Residual df must be >=3 and shrinkage scales positive")
        if min(c.check_every, c.patience, c.validation_samples, c.min_map_steps, c.min_vi_steps) < 1:
            raise ValueError("Stopping budgets must be positive")
        if c.convergence_tol <= 0 or not 0 < c.event_support_fraction <= 1:
            raise ValueError("Invalid stopping tolerance/support fraction")

    def fit(self, Y, times, depth, batch, event_times=None, group=None, mask=None):
        c = self.config
        arrays, metadata = _prepare(Y, times, depth, batch, event_times, group, mask, c)
        normalizer = int(arrays["mask"].sum())
        histories, best_state, best_value = [], None, float("inf")
        for start in range(c.map_starts):
            model = _Model(arrays, c, c.seed + start)
            params = [b.loc for b in model.blocks.values()]
            optimizer = torch.optim.Adam(params, lr=c.map_lr)
            value = float(model.map_objective().detach())
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite initial MAP objective; inspect counts/depth")
            history = [value]
            rejected = 0
            stable_checks = 0
            map_stable = False
            # Adam proposes a direction, then an objective-based backtracking step
            # enforces non-increase. Fall back to clipped negative gradient.
            for step in range(c.map_steps * (c.budget_multiplier if c.adaptive_budget else 1)):
                optimizer.zero_grad()
                loss = model.map_objective() / normalizer
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 20)
                old = [p.detach().clone() for p in params]
                grads = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params]
                optimizer.step()
                proposed = [p.detach().clone() for p in params]
                accepted = False
                with torch.no_grad():
                    for fallback in (False, True):
                        for backtrack in range(10):
                            rate = .5 ** backtrack
                            for p, previous, proposal, grad in zip(params, old, proposed, grads):
                                direction = -c.map_lr * grad if fallback else proposal - previous
                                p.copy_(previous + rate * direction)
                            candidate = float(model.map_objective())
                            if np.isfinite(candidate) and candidate <= value:
                                value, accepted = candidate, True
                                break
                        if accepted:
                            break
                    if not accepted:
                        for p, previous in zip(params, old):
                            p.copy_(previous)
                        rejected += 1
                history.append(value)
                if (step + 1) % c.check_every == 0 and step + 1 >= c.min_map_steps:
                    improvement = (history[-c.check_every - 1] - history[-1]) / normalizer
                    stable_checks = stable_checks + 1 if improvement < c.convergence_tol else 0
                    map_stable = stable_checks >= c.patience
                    if (c.early_stopping or c.adaptive_budget) and map_stable:
                        break
                if c.verbose and (step == 0 or (step + 1) % 100 == 0):
                    print(f"MAP start {start+1}/{c.map_starts}, step {step+1}: objective={value:.3f}", flush=True)
            histories.append({"objective": history, "rejected_steps": rejected, "stable": map_stable,
                              "steps": len(history) - 1})
            if value < best_value:
                best_value = value
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        torch.manual_seed(c.seed + 10000)
        optimizer = torch.optim.Adam(model.parameters(), lr=c.vi_lr)
        vi_history = []
        validation_history, vi_best_state = [], None
        vi_best_step = 0
        vi_best = float("inf")
        stable_checks, vi_stable = 0, False
        vi_cap = c.vi_steps * (c.budget_multiplier if c.adaptive_budget else 1)
        vi_recoveries = []
        safe_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        current_lr = c.vi_lr
        for step in range(vi_cap):
            if c.adaptive_budget and step and step % c.vi_steps == 0:
                current_lr *= .5
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
            optimizer.zero_grad()
            loss = model.negative_elbo() / normalizer
            if not torch.isfinite(loss):
                if len(vi_recoveries) >= c.vi_max_recoveries:
                    raise FloatingPointError("Nonfinite ELBO after bounded rollback retries")
                model.load_state_dict(safe_state)
                current_lr *= c.vi_recovery_factor
                optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
                stable_checks = 0
                vi_recoveries.append({"attempt_step": step + 1, "reason": "nonfinite_elbo", "new_lr": current_lr})
                continue
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
            if not torch.isfinite(gradient_norm):
                if len(vi_recoveries) >= c.vi_max_recoveries:
                    raise FloatingPointError("Nonfinite VI gradient after bounded rollback retries")
                model.load_state_dict(safe_state)
                current_lr *= c.vi_recovery_factor
                optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
                stable_checks = 0
                vi_recoveries.append({"attempt_step": step + 1, "reason": "nonfinite_gradient", "new_lr": current_lr})
                continue
            optimizer.step()
            vi_history.append(float(loss.detach()) * normalizer)
            if (step + 1) % c.check_every == 0 or step + 1 == vi_cap:
                # Common random numbers: validation does not disturb training RNG.
                devices = [torch.device(c.device).index or 0] if str(c.device).startswith("cuda") else []
                with torch.random.fork_rng(devices=devices), torch.no_grad():
                    torch.manual_seed(c.seed + 30000)
                    ll = sum(model.log_likelihood(model.decode(model.values(True))) for _ in range(c.validation_samples)) / c.validation_samples
                    check = float((-ll + model.kl()) / normalizer)
                if not np.isfinite(check):
                    if len(vi_recoveries) >= c.vi_max_recoveries:
                        raise FloatingPointError("Nonfinite validation ELBO after bounded rollback retries")
                    model.load_state_dict(safe_state)
                    current_lr *= c.vi_recovery_factor
                    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
                    stable_checks = 0
                    vi_recoveries.append({"attempt_step": step + 1, "reason": "nonfinite_validation", "new_lr": current_lr})
                    continue
                if validation_history and step + 1 >= c.min_vi_steps:
                    delta = validation_history[-1]["negative_elbo_per_cell"] - check
                    stable_checks = stable_checks + 1 if abs(delta) < c.convergence_tol else 0
                validation_history.append({"step": step + 1, "negative_elbo_per_cell": check})
                safe_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if check < vi_best:
                    vi_best = check
                    vi_best_step = step + 1
                    vi_best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                vi_stable = stable_checks >= c.patience
                if (c.early_stopping or c.adaptive_budget) and vi_stable:
                    break
            if c.verbose and (step == 0 or (step + 1) % 100 == 0):
                print(f"VI step {step+1}: negative MC ELBO={vi_history[-1]:.3f}", flush=True)
        if c.early_stopping and vi_best_state is not None:
            model.load_state_dict(vi_best_state)
        self.model_ = model
        budget = {name: {"means": b.loc.numel(), "covariance_storage": b.raw.numel(),
                         "covariance_effective": int(b.loc.numel() * (b.loc.shape[-1] + 1) / 2) if b.full else b.raw.numel()}
                  for name, b in model.blocks.items()}
        best_start = int(np.argmin([h["objective"][-1] for h in histories]))
        self.metadata_ = metadata | {"config": asdict(c), "map_best_objective": best_value,
                                    "map_histories": histories, "vi_negative_elbo": vi_history,
                                    "schema_version": 2, "parameter_budget": budget,
                                    "n_observed": normalizer, "vi_validation": validation_history,
                                    "map_stable": histories[best_start]["stable"], "vi_stable": vi_stable,
                                    "vi_recoveries": vi_recoveries,
                                    "vi_final_lr": current_lr,
                                    "vi_steps_run": len(vi_history), "vi_best_validation": vi_best,
                                    "vi_selected_step": vi_best_step if c.early_stopping else len(vi_history),
                                    "interval_warning": "Variational intervals are approximate and require simulation calibration.",
                                    "algorithm": "noncentred_MAP_then_block_Gaussian_VI",
                                    "converged": bool(histories[best_start]["stable"] and vi_stable)}
        self.metadata_["joint_covariance_parameters"] = sum(v.numel() for v in model.joint.values())
        self.metadata_["map_budget_exhausted"] = not histories[best_start]["stable"]
        self.metadata_["vi_budget_exhausted"] = not vi_stable
        self.metadata_["estimand"] = "relative_composition_only" if c.observation == "dm" else "depth_corrected_signal"
        # 'converged' is an objective-stability heuristic, not posterior calibration.
        return self

    @torch.no_grad()
    def posterior(self, event_grid=None):
        """Posterior means, pointwise intervals and event probability summaries.

        X interval integrates the discrete S state (drawn conditionally), whereas
        X_mean averages Rao-Blackwell conditional expectations for lower MC error.
        Missing entries use P(S=1|continuous parameters)=eta, not zero responsibility.
        """
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit before posterior")
        c, m = self.config, self.model_
        torch.manual_seed(c.seed + 20000)
        span = self.metadata_["time_span"]
        grid = np.linspace(-.2 * span, .7 * span, 91) if event_grid is None else np.asarray(event_grid, float)
        if grid.ndim != 1 or not np.isfinite(grid).all():
            raise ValueError("event_grid must be a finite vector in original time units")
        q = torch.as_tensor(event_basis(grid / span, c.event_basis, c.event_decay, c.event_window, c.event_delay),
                            dtype=torch.float64, device=c.device)
        names = ["X", "X_latent", "composition", "Z", "absence_probability",
                 "H_plus", "H_plus_group1", "H_zero", "theta_plus", "theta_zero",
                 "trend", "group_effect", "shared", "specific", "event_plus", "event_zero", "batch_effect", "event_population"]
        interval_names = {"X_latent", "Z", "H_plus", "H_plus_group1", "H_zero", "theta_plus", "theta_zero"}
        # Store only draws needed for quantiles; stream component means.
        draws, means = {}, {}
        for draw_index in range(c.posterior_samples):
            d = m.decode(m.values(True))
            if not torch.isfinite(d["Z"]).all() or not torch.isfinite(d["log_mu"]).all():
                raise FloatingPointError("Nonfinite posterior draw; fit is numerically unstable")
            rho = zero_responsibility(m.y, d["log_mu"], d["size"], d["eta_logit"]) if c.zero_inflated else torch.zeros_like(m.y)
            rho = torch.where(m.mask, rho, torch.sigmoid(d["eta_logit"]))
            positive = torch.softmax(d["Z"], -1) if c.observation == "dm" else d["Z"].exp()
            X = (1 - rho) * positive
            X_latent = (torch.rand_like(rho) >= rho) * positive
            comp = X / X.sum(dim=2, keepdim=True).clamp_min(torch.finfo(X.dtype).tiny)
            decoded = dict(X=X, X_latent=X_latent, composition=comp, Z=d["Z"], absence_probability=rho,
                           H_plus=torch.einsum("sq,jeq->jes", q, d["theta"]),
                           H_plus_group1=torch.einsum("sq,jeq->jes", q, d["theta"] + d["theta_group"]),
                           H_zero=torch.einsum("sq,jeq->jes", q, d["theta0"]),
                           theta_plus=d["theta"], theta_zero=d["theta0"])
            for name in names:
                tensor = decoded[name] if name in decoded else d[name]
                array = tensor.cpu().numpy()
                if not np.isfinite(array).all():
                    raise FloatingPointError(f"Nonfinite posterior output: {name}")
                if draw_index == 0:
                    means[name] = np.zeros_like(array)
                    if name in interval_names:
                        draws[name] = np.empty((c.posterior_samples, *array.shape), dtype=np.float32)
                means[name] += array / c.posterior_samples
                if name in interval_names:
                    draws[name][draw_index] = array
        tail = (1 - c.interval) / 2
        result = dict(event_grid=grid, observed_mask=m.mask.cpu().numpy())
        for name, array in means.items():
            if name == "X_latent":
                continue
            result[name + "_mean"] = array
            if name == "X" or name in interval_names:
                interval_array = draws["X_latent"] if name == "X" else draws[name]
                result[name + "_lower"], result[name + "_upper"] = np.quantile(interval_array, [tail, 1-tail], axis=0)
        # composition_mean is a plug-in normalisation of RB estimates per draw,
        # not the exact posterior expectation of a discrete-state composition.
        bounds = self.metadata_.get("event_relative_bounds")
        support = np.ones((m.q_event.shape[-2], grid.size), dtype=bool)
        if bounds is not None:
            bounds = np.asarray(bounds, float)
            exposed = np.isfinite(bounds[..., 0])
            inside = ((grid[None, None] / span >= bounds[..., 0, None]) &
                      (grid[None, None] / span <= bounds[..., 1, None]))
            support = inside.sum(0) / np.maximum(exposed.sum(0)[:, None], 1) >= c.event_support_fraction
        support &= grid[None] >= 0
        # Keep the original support for unguarded recovery scoring in runner.
        result["event_support_v2"] = support.copy()
        if c.support_guard:
            local = np.zeros_like(support, dtype=float)
            eligible_count = np.zeros(support.shape[0])
            for subject in self.metadata_["event_relative_visits"]:
                for e, visits in enumerate(subject):
                    visits = np.asarray(visits, float)
                    visits = visits[np.isfinite(visits)]
                    if not visits.size:
                        continue
                    eligible_count[e] += 1
                    if (visits < 0).sum() >= c.min_pre_visits:
                        post = visits[visits >= 0]
                        if post.size:
                            local[e] += (np.abs(post[:, None] - grid[None] / span) <= c.support_radius).any(0)
            support &= local / np.maximum(eligible_count[:, None], 1) >= c.event_support_fraction
            if self.metadata_["batch_biological_confounded"]:
                support[:] = False
        result["event_support"] = support
        result["event_testable"] = support.any(-1)
        for name in ("H_plus", "H_plus_group1", "H_zero"):
            result[name + "_p_positive"] = (draws[name] > c.effect_threshold).mean(axis=0)
            result[name + "_p_negative"] = (draws[name] < -c.effect_threshold).mean(axis=0)
            result[name + "_p_any"] = (np.abs(draws[name]).max(axis=-1) > c.effect_threshold).mean(axis=0)
            result[name + "_p_any_supported"] = ((np.abs(draws[name]) > c.effect_threshold) & support[None, None]).any(-1).mean(0)
        null_plus = np.abs(draws["H_plus"]).max(axis=-1) <= c.effect_threshold
        null_zero = np.abs(draws["H_zero"]).max(axis=-1) <= c.effect_threshold
        result["event_p_joint_null_group0"] = (null_plus & null_zero).mean(axis=0)
        result["event_p_joint_null_group1"] = ((np.abs(draws["H_plus_group1"]).max(axis=-1) <= c.effect_threshold) & null_zero).mean(axis=0)
        null_zero_supported = ~((np.abs(draws["H_zero"]) > c.effect_threshold) & support[None, None]).any(-1)
        for group_id, name in ((0, "H_plus"), (1, "H_plus_group1")):
            null = ~((np.abs(draws[name]) > c.effect_threshold) & support[None, None]).any(-1)
            result[f"event_p_joint_null_group{group_id}_supported"] = (null & null_zero_supported).mean(0)
        result["metadata_json"] = np.array(json.dumps(self.metadata_, ensure_ascii=False))
        return result

    def save_checkpoint(self, path):
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit before saving")
        torch.save({"state_dict": self.model_.state_dict(), "metadata": self.metadata_}, path)

    @classmethod
    def load_checkpoint(cls, path, device="cpu"):
        """Reload this implementation's checkpoint without rerunning fitting."""
        saved = torch.load(path, map_location=device, weights_only=True)
        stored_config = saved["metadata"]["config"] | {"device": device}
        config = DenoiseConfig(**stored_config) if saved["metadata"].get("schema_version", 1) >= 2 else legacy_config(**stored_config)
        estimator = cls(config)
        buffer_names = ("y", "mask", "times", "depth", "group", "spline", "batch_design",
                        "q_event", "factor_chol", "residual_chol")
        arrays = {key: saved["state_dict"][key].cpu().numpy() for key in buffer_names}
        if "residual_basis" in saved["state_dict"]:
            arrays["residual_basis"] = saved["state_dict"]["residual_basis"].cpu().numpy()
        estimator.model_ = _Model(arrays, config, config.seed)
        estimator.model_.load_state_dict(saved["state_dict"])
        estimator.metadata_ = saved["metadata"]
        estimator.metadata_["config"]["device"] = device
        return estimator


def fit_npz(path, config=None):
    """Read only observed inputs from an NPZ (no access to truth fields)."""
    with np.load(path, allow_pickle=False) as data:
        return EventAwareDenoiser(config).fit(data["Y"], data["times"], data["depth"], data["batch"],
                                            data["event_times"] if "event_times" in data else None,
                                            data["group"] if "group" in data else None,
                                            data["mask"] if "mask" in data else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--nb", action="store_true", help="Disable the biological-absence layer")
    parser.add_argument("--factors", type=int, default=3)
    parser.add_argument("--map-steps", type=int)
    parser.add_argument("--map-starts", type=int, default=2)
    parser.add_argument("--vi-steps", type=int)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--legacy", action="store_true", help="Original model and fixed budgets")
    parser.add_argument("--residual", choices=["none", "basis", "full"])
    parser.add_argument("--individual-event", choices=["none", "gain", "basis"])
    parser.add_argument("--event-prior", choices=["legacy", "regularized"])
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    factory = legacy_config if args.legacy else m4_flex_config
    overrides = {key: getattr(args, key) for key in ("residual", "individual_event", "event_prior") if getattr(args, key) is not None}
    overrides.update({key: getattr(args, key) for key in ("map_steps", "vi_steps") if getattr(args, key) is not None})
    c = factory(n_factors=args.factors, zero_inflated=not args.nb,
                      map_starts=args.map_starts, posterior_samples=args.draws,
                      seed=args.seed, device=args.device, **overrides)
    model = fit_npz(args.input, c)
    result = model.posterior()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / (args.input.stem + "_denoised.npz")
    np.savez_compressed(path, **result)
    model.save_checkpoint(args.out / (args.input.stem + "_fit.pt"))
    (args.out / (args.input.stem + "_diagnostics.json")).write_text(json.dumps(model.metadata_, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
