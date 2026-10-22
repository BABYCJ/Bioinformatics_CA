"""Rule selection by cycle structure (needs cycle_structure.py and search_426.py results).

1. Correlation of cycle features with ARI, per dataset.
2. Mean ARI per cycle class           -> results/cycle_class_ARI.csv
3. Label-free selection: keep rules with threshold <= ncyc_L6 < 729 (729 = identity-like, no
   reduction), pick the rule / Stage-3 mode with the highest silhouette.
                                       -> results/cycle_threshold_selection.csv
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import DATASETS, MIN_CYCLES_L6

os.chdir(os.path.dirname(os.path.abspath(__file__)))
K = {'Pollen': 11, 'Biase': 3, 'Klein': 4}
FEATS = [f'{k}_L{L}' for L in (6, 8) for k in ('ncyc', 'maxcyc', 'fixed', 'meancyc')]


def main():
    pd.set_option('display.width', 250)
    cyc = pd.read_csv('results/rules426_cycle_structure.csv', dtype={'rule': str})
    data = {}
    for ds in DATASETS:
        df = pd.read_csv(f'results/{ds}/{ds}_rule_search_426.csv', dtype={'rule': str})
        data[ds] = df.merge(cyc[['rule', 'cycle_class'] + FEATS], on='rule')

    print('1. Spearman correlation of cycle features with ARI (runs without the right number of clusters count as NaN)')
    corr = []
    for ds, df in data.items():
        for mode in ('centroid', 'avg'):
            d = df[df.stage3 == mode]
            ari = d.ARI.where(d.n_clusters == K[ds])
            corr.append({'dataset': ds, 'stage3': mode,
                         **{f: spearmanr(d[f], ari, nan_policy='omit')[0] for f in FEATS}})
    print(pd.DataFrame(corr).round(2).to_string(index=False))

    print('\n2. Mean ARI per cycle class (invalid runs count as ARI 0)')
    C = cyc.groupby('cycle_class').agg(n_rules=('rule', 'size'), ncyc_L6=('ncyc_L6', 'first'),
                                       maxcyc_L6=('maxcyc_L6', 'first'), fixed_L6=('fixed_L6', 'first'))
    for ds, df in data.items():
        for mode in ('centroid', 'avg'):
            d = df[df.stage3 == mode]
            ari = np.where(d.n_clusters == K[ds], d.ARI, 0.0)
            C[f'{ds}_{mode}'] = pd.Series(ari, index=d.index).groupby(d.cycle_class).mean()
    C = C.sort_values('ncyc_L6', ascending=False)
    C.to_csv('results/cycle_class_ARI.csv')
    print(C.round(3).to_string())

    print(f'\n3. Cycle filter (threshold <= ncyc_L6 < 729) + silhouette pick; config uses threshold {MIN_CYCLES_L6}')
    rows = []
    for T in (0, 20, 30, 40, 45, 60, 80, 100, 150, 170):
        row = {'min_ncyc_L6': T, 'n_rules': int(((cyc.ncyc_L6 >= T) & (cyc.ncyc_L6 < 729)).sum())}
        for ds, df in data.items():
            d = df[(df.n_clusters == K[ds]) & (df.ncyc_L6 >= T) & (df.ncyc_L6 < 729)]
            pick = d.loc[d.Silhouette.idxmax()]
            row[f'{ds}_meanARI'] = d.ARI.mean()
            row[f'{ds}_pick'] = f'{pick.rule}/{pick.stage3}'
            row[f'{ds}_pickARI'] = pick.ARI
        rows.append(row)
    S = pd.DataFrame(rows)
    S.to_csv('results/cycle_threshold_selection.csv', index=False)
    print(S.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
