"""Run the selected 426-family rules on each dataset and save clusters, metrics and plots.

usage:  python run_best.py                 # all datasets, both rule choices
        python run_best.py Pollen Klein    # some datasets
"""
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from config import DATASETS, RULES_FILE, BEST_ARI, CYCLE_SILHOUETTE
from rca import Dataset, load_rules

os.chdir(os.path.dirname(os.path.abspath(__file__)))
CHOICES = {'best_ARI': BEST_ARI, 'cycle_silhouette': CYCLE_SILHOUETTE}


def main(names):
    rules, tags = load_rules(RULES_FILE)
    summary = []
    for name in names:
        ds = Dataset(DATASETS[name])
        out = f'results/{name}'
        os.makedirs(out, exist_ok=True)
        for choice, table in CHOICES.items():
            rule, mode = table[name]
            t0 = time.time()
            pred = ds.run_rule(rules[rule], mode)
            res = ds.evaluate(pred)
            print(f"\n[{name}] {choice}: rule {rule} ({rules[rule]}, {tags[rule]}), "
                  f"Stage 3 = {mode}  [{time.time() - t0:.1f}s]")
            print(pd.Series(res).round(4).to_string())
            print(pd.crosstab(pd.Series(ds.true_labels, name='true cell type'),
                              pd.Series(pred, name='predicted cluster')))
            pd.DataFrame({'cell': ds.cells, 'true_label': ds.true_labels, 'predicted_cluster': pred}) \
              .to_csv(f'{out}/{name}_clusters_{choice}.csv', index=False)

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            ax[0].scatter(ds.P[:, 0], ds.P[:, 1], c=pd.factorize(ds.true_labels)[0], s=12, cmap='tab20')
            ax[0].set_title(f"{name}: true cell types")
            ax[1].scatter(ds.P[:, 0], ds.P[:, 1], c=pred, s=12, cmap='tab20')
            ax[1].set_title(f"RCA rule {rule} ({mode}), ARI = {res['ARI']:.3f}")
            for a in ax:
                a.set_xlabel('PC1'); a.set_ylabel('PC2')
            plt.tight_layout(); plt.savefig(f'{out}/{name}_clusters_{choice}.png', dpi=150); plt.close(fig)
            summary.append({'dataset': name, 'choice': choice, 'rule': rule, 'rule_str': rules[rule],
                            'stage3': mode, **res})
    df = pd.DataFrame(summary)
    print('\n' + df.drop(columns='rule_str').round(4).to_string(index=False))
    if set(names) == set(DATASETS):
        df.to_csv('results/summary.csv', index=False)
        print('Saved results/summary.csv')


if __name__ == '__main__':
    main(sys.argv[1:] or list(DATASETS))
