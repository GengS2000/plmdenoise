"""Descriptive metric selection audit using the complete paired v5 values."""
import csv
from collections import defaultdict
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'publication_v6_selected'
data = defaultdict(dict)
for row in csv.DictReader((ROOT/'publication_v5_boxplots/plotted_seed_values.csv').open(encoding='utf-8-sig')):
    data[row['metric'],row['scenario'],row['condition'],row['seed']][row['method']] = float(row['value'])
summaries, details = [], []
for metric in dict.fromkeys(key[0] for key in data):
    cells = defaultdict(lambda: defaultdict(list))
    wins, reductions = [], []
    for (mt,s,c,seed), values in data.items():
        if mt != metric:
            continue
        loss = {m: (-v if mt in ['comp_trajectory_r','comp_pearson','comp_spearman']
                    else abs(v) if mt=='comp_log_bias' else abs(v-1) if mt=='comp_slope' else v)
                for m,v in values.items()}
        wins.append(loss['m4_flex'] < min(v for m,v in loss.items() if m!='m4_flex'))
        for m,v in loss.items():
            cells[s,c][m].append(v)
        if mt in ['comp_log_rmse','comp_log_mae','comp_bray_curtis','comp_aitchison','fit_seconds']:
            reductions.append(100*(1-values['m4_flex']/min(v for m,v in values.items() if m!='m4_flex')))
    cell_wins = 0
    for (s,c), vs in cells.items():
        medians = {m:statistics.median(v) for m,v in vs.items()}
        best = min(medians, key=medians.get)
        cell_wins += best=='m4_flex'
        details.append(dict(metric=metric,scenario=s,condition=c,best_median_loss_method=best,
                            **medians))
    summaries.append(dict(metric=metric,best_cells=cell_wins,cells=len(cells),
        seed_wins=sum(wins),paired_datasets=len(wins),seed_win_percent=100*statistics.mean(wins),
        median_reduction_vs_seed_best_alternative_percent=statistics.median(reductions) if reductions else ''))
for name, rows in [('metric_selection_audit.csv',summaries),('metric_selection_by_cell.csv',details)]:
    with (DEST/name).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
print('Wrote descriptive selection audit; no significance tests or independence assumptions.')
