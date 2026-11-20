"""Check actual curve-scale priors (no simulated truth or fitted data used).

Reports practical-effect probabilities, NOT inclusion probabilities. This checks
the prior implication of the sup-norm ROPE on a fixed post-event interval.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import warnings

import numpy as np
import torch

from event_aware_denoise import DenoiseConfig, _Model, _prepare, event_basis, legacy_config


def inspect(config, draws=2000, taxa=80, events=1):
    I, T = 2, 6
    times = np.broadcast_to(np.linspace(0, 1, T), (I, T)).copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arrays, _ = _prepare(np.ones((I, T, taxa)), times, np.full((I, T), 1000.),
                             np.zeros((I, T), dtype=int), np.full((I, events), .3),
                             np.array([0, 1]), None, config)
    model = _Model(arrays, config, 2026)
    q = torch.as_tensor(event_basis(np.linspace(0, .7, 141), config.event_basis,
                        config.event_decay, config.event_window, config.event_delay), dtype=torch.float64)
    torch.manual_seed(4567)
    peaks = {"plus": [], "zero": []}
    with torch.no_grad():
        for _ in range(draws):
            w = {name: torch.randn_like(block.loc) for name, block in model.blocks.items() if "theta" in name}
            for channel, key in (("plus", "theta"), ("zero", "theta0")):
                theta = w[key] * model.event_scale(w, zero=channel == "zero")
                curve = torch.einsum("sq,jeq->jes", q, theta)
                peaks[channel].append(curve.abs().amax(-1).numpy())
    result = dict(config=asdict(config), draws=draws, taxa=taxa, events=events,
                  grid_interval=[0, .7], threshold=config.effect_threshold)
    for name, arrays in peaks.items():
        peak = np.stack(arrays)
        result[name] = dict(prior_probability_any_effect=float((peak > config.effect_threshold).mean()),
                            median_peak=float(np.median(peak)), peak_q95=float(np.quantile(peak, .95)))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.draws < 1:
        p.error("draws must be positive")
    torch.set_num_threads(1)
    output = {name: inspect(c, args.draws) for name, c in (("legacy", legacy_config()), ("regularized", DenoiseConfig()))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for name, row in output.items():
        print(name, {c: row[c] for c in ("plus", "zero")})


if __name__ == "__main__":
    main()
