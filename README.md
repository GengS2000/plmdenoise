# PLMdenoise

Probabilistic Longitudinal Microbiome Denoising with flexible post-event responses.
The default estimator implements the `m4_flex` preset from `code_v3_1`.
It fits longitudinal count data with a zero-inflated negative binomial observation
model, smooth background trajectories and an event dictionary. It uses MAP
initialization followed by structured variational inference.

Analysis scripts for the manuscript are available in the [analysis directory](analysis/README.md).
## Install

Python 3.10 or later. From the package directory:

```sh
python -m pip install .
python examples/synthetic.py
```

Once the project is published on PyPI, install with `python -m pip install plmdenoise`.
For CUDA, install the appropriate PyTorch distribution for your system before installing PLMdenoise.

## Use

```python
from plmdenoise import EventAwareDenoiser, m4_flex_config

config = m4_flex_config(seed=2026, device="cpu")
fit = EventAwareDenoiser(config).fit(Y, times, depth, batch, event_times=event_times)
posterior = fit.posterior()
fit.save_checkpoint("fit.pt")
restored = EventAwareDenoiser.load_checkpoint("fit.pt")
```

`Y` contains nonnegative integer counts in subject × visit × taxon order.
`times`, `depth` and `batch` have shape subject × visit. Times must be finite
and strictly increasing within each subject, exposure must be positive, and
batch labels are nonnegative integers. These requirements also apply to
masked visit slots. Use zeros for batch labels when there is one batch.
Optional `mask` matches `Y`, with true indicating an observed count.
Without a mask, finite nonnegative entries are treated as observed. A
measured zero is an observed count. Each taxon requires an observation.

Optional `event_times` uses the same time units as `times`. Pass one
value per subject, or one column per event type, with NaN for unexposed
subjects. Optional `group` contains one binary label per subject.

`composition_mean` averages the within-draw normalized denoised signal
after accounting for structural absence. `Z_mean` is the mean latent log
abundance before structural absence. For the abundance channel used in
downstream analyses, apply softmax to `Z_mean` across taxa.
`absence_probability_mean` reports the inferred probability of absence.
Event curves describe associations on the model scale. Interpret them in
light of visit support, confounding and model assumptions.

The CLI reads the same array names from an NPZ file and writes a posterior
NPZ, a checkpoint and JSON diagnostics:

```sh
plmdenoise input.npz --out results
```

## Release

Build a wheel and source distribution with `python -m build`.
After creating the GitHub repository, register a
[PyPI pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
with project name `plmdenoise`, the exact GitHub owner and repository,
workflow `publish.yml` and environment `pypi`. Publishing a GitHub
Release tagged `v0.1.0` triggers the workflow for this version.
Confirm the PyPI name, author attribution and license before public release.
