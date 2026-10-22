"""Run all 426 rules (both Stage-3 modes) on each dataset -> results/<dataset>/<dataset>_rule_search_426.csv

usage:  python search_426.py                # all datasets (Klein takes ~2-3 minutes)
        python search_426.py Pollen Biase
"""
import os
import sys
import time

import numpy as np
import pandas as pd

from config import DATASETS, RULES_FILE
from rca import Dataset, load_rules

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def search(name, rules, tags):
    ds = Dataset(DATASETS[name])
    rows, cache = [], {}
    t0 = time.time()
    for k, r in enumerate(rules, 1):
        for mode in ('centroid', 'avg'):
            try:
                lab = ds.run_rule(rules[r], mode)
            except RuntimeError as e:                    # e.g. the identity-like rules never reduce
                rows.append({'rule': r, 'rule_str': rules[r], 'tags': tags[r], 'stage3': mode, 'error': str(e)})
                continue
            sizes = np.bincount(lab)
            sizes = sizes[sizes > 0]
            key = lab.tobytes()
            if key not in cache:
                cache[key] = ds.evaluate(lab) if len(sizes) > 1 else {}
            rows.append({'rule': r, 'rule_str': rules[r], 'tags': tags[r], 'stage3': mode,
                         'n_clusters': len(sizes), 'min_size': int(sizes.min()), **cache[key]})
        if k % 100 == 0:
            print(f'  {k}/{len(rules)} rules, {time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(f'results/{name}', exist_ok=True)
    df.to_csv(f'results/{name}/{name}_rule_search_426.csv', index=False)

    ok = df[df.n_clusters == ds.num_clusters]
    cols = ['rule', 'rule_str', 'stage3', 'ARI', 'NMI', 'ACC', 'F1', 'Silhouette']
    print(f'[{name}] {len(ok)} runs with {ds.num_clusters} clusters. Top 5 by ARI:')
    print(ok.sort_values(['ARI', 'NMI'], ascending=False)[cols].head(5).round(4).to_string(index=False))


if __name__ == '__main__':
    pd.set_option('display.width', 200)
    rules, tags = load_rules(RULES_FILE)
    for name in sys.argv[1:] or list(DATASETS):
        search(name, rules, tags)
