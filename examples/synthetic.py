"""Run a small CPU fit using simulated counts."""

import numpy as np
import torch

from plmdenoise import EventAwareDenoiser, m4_flex_config

torch.set_num_threads(1)
rng = np.random.default_rng(12)
times = np.tile(np.arange(6, dtype=float), (4, 1))
depth = np.full(times.shape, 2000.0)
batch = np.zeros(times.shape, dtype=int)
events = np.array([2.5, 2.5, np.nan, np.nan])
rates = np.broadcast_to([12., 20., 30.], (*times.shape, 3)).copy()
rates[:2, 3:, 0] *= 2
counts = rng.poisson(rates)
# Small budgets are for demonstrating the interface.
config = m4_flex_config(map_steps=5, vi_steps=8, map_starts=1,
                        posterior_samples=8, early_stopping=False)
fit = EventAwareDenoiser(config).fit(counts, times, depth, batch, event_times=events)
print("Recovered composition shape:", fit.posterior()["composition_mean"].shape)
