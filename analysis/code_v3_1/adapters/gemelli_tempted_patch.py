"""Export the low-rank reconstruction that Gemelli's TEMPTED already builds.

`gemelli.tempted.tempted_helper` fits TEMPTED one rank-one component at a time
and, after each component, subtracts that component's reconstruction from the
working tables:

    tables_update[id] -= lambda_coeff[[s]] * (a_i * b (x) phi[ti_i])

The public function returns ordinations, loadings and eigenvalues only, so the
reconstructed tensor -- which is what a denoising benchmark needs -- is thrown
away.  Worse, the returned loadings cannot be multiplied back together: the
function flips the sign of `state_loadings` at the end without compensating it
elsewhere, so the product of the *returned* factors is the reconstruction up to
an unrecoverable per-component sign.

Rather than reimplement the optimisation, this module takes the authors' own
source for `tempted_helper`, applies additive edits and compiles the result
against the authors' module globals.  The algorithm, the iteration order, the
convergence rule and every returned value are untouched; the patch only
accumulates the term that is already computed on the line above it, and returns
it.

One edit is a guard rather than an export.  Each component starts the subject
loading at `sign(sum(b_hat)) / sqrt(n)`, where `b_hat` is the leading left
singular vector.  On CLR input every column sums to zero, so `b_hat` is
orthogonal to the ones vector and `sum(b_hat)` is round-off (a few multiples of
2.2e-16 on simple32).  Its sign is then arbitrary, and often exactly 0.  A zero
sign makes `a_hat` all zero, `phi_hat` all zero, the normalisation 0/0 and the
fit NaN; `lstsq` then raises "SVD did not converge in Linear Least Squares".
The guard maps sign 0 to +1, the start the authors' R package always uses
(`+1/sqrt(n)`).  Whenever the author code would have produced a finite fit, its
behaviour is unchanged.

The exact patch, the gemelli version and a hash of the original source are
recorded in the run metadata; if a different gemelli version changes that
source, `build()` fails loudly instead of silently patching something else.
"""
from __future__ import annotations

import hashlib
import inspect
import textwrap

# (anchor, replacement) applied once each, in order, to the author source.
PATCH = (
    (
        "    # calculate rank-1 component sequentially.\n"
        "    X = []\n",
        "    # calculate rank-1 component sequentially.\n"
        "    reconstruction = {k: np.zeros(v.shape, dtype=float)\n"
        "                      for k, v in tables_update.items()}\n"
        "    reconstruction_grid = {k: np.zeros((n_features, resolution), dtype=float)\n"
        "                           for k in tables_update}\n"
        "    X = []\n",
    ),
    (
        "        consistent_sign = np.sign(np.sum(b_hat))\n",
        "        consistent_sign = np.sign(np.sum(b_hat))\n"
        "        if consistent_sign == 0:\n"
        "            consistent_sign = 1.0\n",
    ),
    (
        "            tables_update[individual_id] -= (lambda_coeff[[s]] * scale_tmp)\n",
        "            tables_update[individual_id] -= (lambda_coeff[[s]] * scale_tmp)\n"
        "            reconstruction[individual_id] += (lambda_coeff[[s]] * scale_tmp)\n"
        "            grid_tmp = ft_tmp_.values.dot(state_loadings.iloc[:, [s]].values.T)\n"
        "            grid_tmp = individual_loadings.iloc[i, [s]].values * grid_tmp\n"
        "            reconstruction_grid[individual_id] += (lambda_coeff[[s]] * grid_tmp)\n",
    ),
    (
        "    return (individual_loadings, feature_loadings,\n"
        "            state_loadings, time_return,\n"
        "            lambda_coeff, rsquared)\n",
        "    return (individual_loadings, feature_loadings,\n"
        "            state_loadings, time_return,\n"
        "            lambda_coeff, rsquared,\n"
        "            reconstruction, reconstruction_grid,\n"
        "            {k: v.values.copy() for k, v in tables_update.items()},\n"
        "            [t.copy() for t in ti], list(tables_update.keys()))\n",
    ),
)

PATCHED_NAME = "tempted_helper_with_reconstruction"


def patch_text():
    """Human-readable record of the edit, stored in every run's metadata."""
    return "\n".join(f"@@ replace @@\n{old}@@ with @@\n{new}" for old, new in PATCH)


def patch_id():
    return hashlib.sha256(patch_text().encode()).hexdigest()[:16]


def build():
    """Compile the patched helper against gemelli's own module globals.

    Returns (function, provenance dict).  Raises RuntimeError if the authors'
    source no longer contains an anchor, i.e. the pinned version changed.
    """
    import gemelli
    from gemelli import tempted as author_module

    source = textwrap.dedent(inspect.getsource(author_module.tempted_helper))
    original_sha = hashlib.sha256(source.encode()).hexdigest()
    patched = source
    for old, new in PATCH:
        if patched.count(old) != 1:
            raise RuntimeError(
                f"gemelli {gemelli.__version__}: tempted_helper source does not contain "
                f"exactly one copy of the patch anchor {old.strip()[:60]!r}. "
                "Re-pin the gemelli version or update adapters/gemelli_tempted_patch.py.")
        patched = patched.replace(old, new, 1)
    patched = patched.replace("def tempted_helper(", f"def {PATCHED_NAME}(", 1)
    namespace = dict(vars(author_module))
    exec(compile(patched, "<gemelli tempted_helper (benchmark patch)>", "exec"), namespace)
    provenance = dict(gemelli_version=gemelli.__version__,
                      original_source_sha256=original_sha,
                      patched_source_sha256=hashlib.sha256(patched.encode()).hexdigest(),
                      patch_id=patch_id(), patch=patch_text(),
                      patched_function=PATCHED_NAME,
                      note="additive export of the rank-one reconstruction the author "
                           "loop already subtracts, plus a sign(0) -> +1 guard on the "
                           "subject-loading start (zero sign otherwise gives 0/0 in "
                           "phi_hat on CLR input); no other optimisation code changed")
    return namespace[PATCHED_NAME], provenance
