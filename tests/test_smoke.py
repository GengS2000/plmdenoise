import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from plmdenoise import EventAwareDenoiser, m4_flex_config


class InstalledPackageTest(unittest.TestCase):
    def test_default_preset_and_checkpoint(self):
        torch.set_num_threads(1)
        default = EventAwareDenoiser().config
        self.assertTrue(default.adaptive_budget)
        self.assertTrue(default.hierarchical_observation)
        self.assertEqual(default.joint_rank, 4)
        self.assertEqual(sum(name.startswith("flex") for name in default.event_basis), 4)

        times = np.tile(np.arange(6, dtype=float), (4, 1))
        depth = np.full(times.shape, 2000.0)
        batch = np.zeros(times.shape, dtype=int)
        events = np.array([2.3, 2.6, 3.1, np.nan])
        counts = np.random.default_rng(12).poisson(20, (*times.shape, 3))
        config = m4_flex_config(map_steps=5, vi_steps=8, map_starts=1,
                                posterior_samples=8, early_stopping=False)
        model = EventAwareDenoiser(config).fit(counts, times, depth, batch, events)
        expected = model.posterior()["composition_mean"]
        self.assertEqual(expected.shape, counts.shape)
        np.testing.assert_allclose(expected.sum(axis=-1), 1., atol=1e-6)
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "fit.pt"
            model.save_checkpoint(path)
            actual = EventAwareDenoiser.load_checkpoint(path).posterior()["composition_mean"]
        np.testing.assert_array_equal(actual, expected)
