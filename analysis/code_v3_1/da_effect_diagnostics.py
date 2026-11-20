"""Post-fit oracle comparisons; truth is never passed to any denoiser.

The response projections below are descriptive OLS coefficients, not DA tests.
Their residual decompositions are not mixed-model variance estimates.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from da_design import GENERATED_INPUTS, identifiers, write_csv


def read_table(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_matrix(path, shape):
    samples, taxa = identifiers(shape)
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader) != ['sample_id'] + taxa:
            raise ValueError(f"Taxon alignment mismatch: {path}")
        rows = list(reader)
    if [r[0] for r in rows] != samples:
        raise ValueError(f"Sample alignment mismatch: {path}")
    values = np.array([r[1:] for r in rows], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(1), 1, atol=1e-7):
        raise ValueError(f"Invalid composition: {path}")
    return values


def projection(values, design, subjects, visits):
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    residual = (values - design @ coefficients).reshape(subjects, visits, -1)
    between = residual.mean(1)
    within = residual - between[:, None, :]
    a, b = within[:, :-1].reshape(-1, values.shape[1]), within[:, 1:].reshape(-1, values.shape[1])
    a, b = a-a.mean(0), b-b.mean(0)
    denom = np.sqrt((a*a).sum(0)*(b*b).sum(0))
    lag = np.divide((a*b).sum(0), denom, out=np.full_like(denom, np.nan), where=denom>0)
    return dict(beta=coefficients[1], residual_sd=residual.reshape(-1, values.shape[1]).std(0, ddof=1),
                between_subject_sd=between.std(0, ddof=1),
                within_subject_sd=within.reshape(-1, values.shape[1]).std(0, ddof=1), lag1=lag)


REFERENCES = ('oracle_clean', 'oracle_abundance')
SCALES = ('log2_pseudocount', 'asin_sqrt', 'clr_halfmin')


def response(values, scale, pseudocount):
    """The response scales actually used downstream (clr_halfmin mirrors LinDA's
    proportion path: zero -> half the taxon's smallest positive value, log2, CLR)."""
    if scale == 'log2_pseudocount':
        return np.log2(values + pseudocount)
    if scale == 'asin_sqrt':
        return np.arcsin(np.sqrt(values))
    filled = values.copy()
    for j in range(filled.shape[1]):
        column = filled[:, j]
        positive = column[column > 0]
        column[column == 0] = .5 * positive.min() if positive.size else np.finfo(float).tiny
    logged = np.log2(filled)
    return logged - logged.mean(1, keepdims=True)


def closure_shift(z_true, gamma, group):
    """Group contrast of uninjected taxa on the zero-free log2 composition: what
    closure alone does to taxa whose injection label is false."""
    from da_design import abundance_composition
    logged = np.log2(abundance_composition(z_true))
    contrast = logged[group == 1].mean((0, 1)) - logged[group == 0].mean((0, 1))
    return float(np.median(contrast[gamma == 0])) if (gamma == 0).any() else None


def diagnose_dataset(directory, plan):
    from run_da import artifact_done, finish
    from da_design import fit_source
    marker = directory / 'effect_diagnostics.json'
    if artifact_done(marker, plan['plan_id']):
        return
    if not artifact_done(directory / 'generation.json', plan['plan_id']):
        raise ValueError('Cannot diagnose unverified truth')
    with np.load(directory / 'truth.npz', allow_pickle=False) as truth:
        shape = truth['Y'].shape
        gamma = truth['da_gamma'].copy()
        group = truth['group'].copy()
        shift = closure_shift(truth['Z_true'], gamma, group)
    samples, taxa = identifiers(shape)
    meta = read_table(directory / 'metadata.csv')
    if [r['sample_id'] for r in meta] != samples:
        raise ValueError('Metadata order mismatch')
    visit = np.array([int(r['visit']) for r in meta])
    design = np.column_stack([np.ones(len(meta)), [int(r['X']) for r in meta],
                             *[(visit == v).astype(float) for v in sorted(set(visit))[1:]],
                             [float(r['event_post']) for r in meta]])
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError('Rank-deficient diagnostic design')
    matrices = {}
    for source in plan['inputs']:
        fit = fit_source(source)
        valid = fit is None or artifact_done(directory / f'denoise_{fit}.json', plan['plan_id'])
        if valid and (directory / f'{source}.csv').exists():
            matrices[source] = read_matrix(directory / f'{source}.csv', shape)
    references = [r for r in REFERENCES if r in matrices]
    if not references:
        return
    rows, summaries = [], []
    for scale in SCALES:
        projections = {source: projection(response(value, scale, plan['log_pseudocount']), design, shape[0], shape[1])
                       for source, value in matrices.items()}
        for source, current in projections.items():
            for j, taxon in enumerate(taxa):
                row = dict(input=source, scale=scale, taxon=taxon, gamma=float(gamma[j]),
                           **{key: float(val[j]) for key, val in current.items()})
                row.update({f'{reference}_beta': float(projections[reference]['beta'][j]) for reference in references})
                rows.append(row)
            for reference in references:
                oracle = projections[reference]
                for label, mask in (('all', np.ones(len(taxa), bool)), ('injected', gamma != 0),
                                    ('uninjected', gamma == 0)):
                    if not mask.any():
                        continue
                    a, b, g = oracle['beta'][mask], current['beta'][mask], gamma[mask]
                    record = dict(input=source, reference=reference, scale=scale, subset=label,
                        n_taxa=int(mask.sum()), closure_shift_uninjected_log2=shift,
                        beta_rmse_to_oracle=float(np.sqrt(np.mean((a-b)**2))),
                        beta_slope_to_oracle=float(np.dot(a,b)/np.dot(a,a)) if np.dot(a,a)>0 else None,
                        beta_corr_to_oracle=float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None,
                        # Compression against the injected truth itself (ideal 1/ln2 on log2 scales).
                        beta_slope_to_gamma=float(np.polyfit(g, b, 1)[0]) if g.std() > 0 else None,
                        oracle_beta_slope_to_gamma=float(np.polyfit(g, a, 1)[0]) if g.std() > 0 else None,
                        median_beta=float(np.median(b)))
                    for key in ('residual_sd', 'between_subject_sd', 'within_subject_sd', 'lag1'):
                        record['median_'+key] = float(np.nanmedian(current[key][mask]))
                        record['median_oracle_'+key] = float(np.nanmedian(oracle[key][mask]))
                    summaries.append(record)
    folder = directory / 'diagnostics'
    write_csv(folder / 'effect_projection_taxa.csv', list(rows[0]), rows)
    write_csv(folder / 'effect_projection_summary.csv', list(summaries[0]), summaries)
    # Only mark final once all inputs are available; a resume can add missing methods.
    if set(matrices) == set(plan['inputs']):
        finish(marker, plan, 'complete', [folder / 'effect_projection_taxa.csv', folder / 'effect_projection_summary.csv'],
               dependencies=[directory / f'{source}.csv' for source in matrices] +
                            [directory / 'truth.npz', directory / 'metadata.csv'],
               note='Descriptive response-scale OLS projection, not mixed-model inference')


def bootstrap_summary(values, rng):
    values = np.array(values, dtype=float)
    n = len(values)
    if n == 0:
        return dict(n=0, mean=None, mcse=None, ci_lo=None, ci_hi=None)
    if n == 1:
        return dict(n=1, mean=float(values[0]), mcse=None, ci_lo=None, ci_hi=None)
    means = values[rng.integers(n, size=(2000, n))].mean(1)
    return dict(n=n, mean=float(values.mean()), mcse=float(values.std(ddof=1)/np.sqrt(n)),
                ci_lo=float(np.quantile(means,.025)), ci_hi=float(np.quantile(means,.975)))


def summarise_diagnostics(out, plan, da_rows):
    projection_rows = []
    for task in plan['tasks']:
        path = out / 'datasets' / task['task_id'] / 'diagnostics/effect_projection_summary.csv'
        if path.exists():
            projection_rows.extend(dict(row, dataset_id=task['task_id'], seed=task['seed'],
                                        series=task['series'], strength=task['strength']) for row in read_table(path))
    if projection_rows:
        write_csv(out / 'effect_projection_summary.csv', list(projection_rows[0]), projection_rows)
    lookup = {(r['task_id'],r['input'],r['method']):r for r in da_rows}
    references = [r for r in REFERENCES if r in plan['inputs']]
    fitted = []
    for task, method, reference in ((t, m, r) for t in plan['tasks'] for m in plan['methods'] for r in references):
        oracle_row = lookup[(task['task_id'],reference,method)]
        if oracle_row['status'] not in ('complete','partial'):
            continue
        folder = out / 'datasets' / task['task_id'] / 'da'
        oracle_taxa = {r['taxon']:r for r in read_table(folder / reference / method / 'taxa.csv')}
        for source in plan['inputs']:
            if lookup[(task['task_id'],source,method)]['status'] not in ('complete','partial'):
                continue
            combined = []
            for row in read_table(folder / source / method / 'taxa.csv'):
                other = oracle_taxa.get(row['taxon'])
                if other is None or row['fit_status'] not in ('ok','warning') or other['fit_status'] not in ('ok','warning'):
                    continue
                try:
                    values = [float(row[k]) for k in ('estimate','std_error','gamma')]
                    values += [float(other[k]) for k in ('estimate','std_error')]
                except (ValueError, TypeError):
                    continue
                if np.isfinite(values).all():
                    combined.append(values)
            if not combined:
                continue
            values = np.array(combined)
            for subset, mask in (('all',np.ones(len(values),bool)), ('injected',values[:,2]!=0)):
                if not mask.any():
                    continue
                beta,se,_,ref,ref_se = values[mask].T
                fitted.append(dict(dataset_id=task['task_id'], seed=task['seed'], series=task['series'],
                    strength=task['strength'], input=source, reference=reference, method=method,
                    subset=subset, n_common=int(mask.sum()),
                    beta_rmse_to_oracle=float(np.sqrt(np.mean((beta-ref)**2))),
                    beta_slope_to_oracle=float(np.dot(beta,ref)/np.dot(ref,ref)) if np.dot(ref,ref)>0 else None,
                    median_se=float(np.median(se)), median_oracle_se=float(np.median(ref_se)),
                    median_se_ratio_to_oracle=float(np.median(se[ref_se>0]/ref_se[ref_se>0])) if (ref_se>0).any() else None))
    if fitted:
        write_csv(out / 'fitted_effect_comparison.csv', list(fitted[0]), fitted)
    paired = []
    for task in plan['tasks']:
        for method in plan['methods']:
            for reference in ('naive', *references):
                baseline = lookup[(task['task_id'],reference,method)]
                for source in plan['inputs']:
                    if source == reference:
                        continue
                    candidate = lookup[(task['task_id'],source,method)]
                    valid = baseline['status'] in ('complete','partial') and candidate['status'] in ('complete','partial')
                    item = dict(dataset_id=task['task_id'], seed=task['seed'], series=task['series'],
                        strength=task['strength'], method=method, reference=reference, input=source,
                        paired_usable=valid, input_status=candidate['status'], reference_status=baseline['status'])
                    for metric in ('fdp','power','f1','failure_fraction'):
                        a,b = baseline.get(metric),candidate.get(metric)
                        item[metric+'_delta'] = b-a if valid and a is not None and b is not None else None
                    paired.append(item)
    if paired:
        write_csv(out / 'oracle_paired_deltas.csv', list(paired[0]), paired)
    summary = []
    rng = np.random.default_rng(20261005)
    keys = ('series','strength','method','reference','input')
    for group in sorted({tuple(row[k] for k in keys) for row in paired}):
        selected = [r for r in paired if tuple(r[k] for k in keys)==group]
        for metric in ('fdp','power','f1','failure_fraction'):
            values = [r[metric+'_delta'] for r in selected if r[metric+'_delta'] is not None]
            summary.append(dict(zip(keys,group), metric=metric, expected=len(selected),
                                scope='paired_including_partial', **bootstrap_summary(values,rng)))
    if summary:
        write_csv(out / 'oracle_paired_summary.csv', list(summary[0]), summary)
    from run_da import atomic_json
    atomic_json(out / 'oracle_analysis_notes.json', dict(
        estimand='injected absolute log-abundance coefficient gamma != 0',
        warning='FDP against injection labels is not necessarily model-null FDR for transformed relative abundances',
        oracle='oracle_clean: realised clean composition with structural zeros; oracle_abundance: exp(Z_true) closed, '
               'no zeros. Diagnostic references, not guaranteed inference upper bounds',
        matched_pairs='m4_flex <-> oracle_clean (composition channel); m4_flex_abundance <-> oracle_abundance (abundance channel)',
        closure='closure_shift_uninjected_log2: true zero-free log2 group contrast of uninjected taxa; nonzero values '
                'mean injection-label FDP is not model-null FDR for log-proportion responses',
        effect_projection='Descriptive adjusted OLS contrast; not an additional DA test',
        uncertainty='2000 seed-level paired bootstrap resamples, separately within each effect; MCSE also reported',
        selection='All predeclared effects and methods reported, regardless of direction of benefit'))
