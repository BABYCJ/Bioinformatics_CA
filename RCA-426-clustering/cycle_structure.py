"""Cycle structure of every 426-family rule on configuration lengths 3..8 (null boundaries).
-> results/rules426_cycle_structure.csv

For each length L: number of cycles (ncyc), longest cycle (maxcyc), fixed points (fixed),
mean cycle length (meancyc) and the cycle type (length^count ...).
"""
import collections
import os

import numpy as np
import pandas as pd

from config import RULES_FILE
from rca import load_rules, rule_table, next_state_map

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def cycle_lengths(nxt):
    seen = np.zeros(len(nxt), bool)
    out = []
    for s in range(len(nxt)):
        if seen[s]:
            continue
        k, u = 0, s
        while not seen[u]:
            seen[u] = True
            u = nxt[u]
            k += 1
        out.append(k)
    return out


def main():
    rules, tags = load_rules(RULES_FILE)
    rows = []
    for rid, rs in rules.items():
        t = rule_table(rs)
        row = {'rule': rid, 'rule_str': rs, 'tags': tags[rid]}
        for L in range(3, 9):
            nxt = next_state_map(t, L)
            assert len(set(nxt)) == 3 ** L, (rid, L, 'not a permutation')
            cl = cycle_lengths(nxt)
            c = collections.Counter(cl)
            row[f'ncyc_L{L}'] = len(cl)
            row[f'maxcyc_L{L}'] = max(cl)
            row[f'fixed_L{L}'] = c.get(1, 0)
            row[f'meancyc_L{L}'] = 3 ** L / len(cl)
            row[f'type_L{L}'] = ' '.join(f'{k}^{v}' for k, v in sorted(c.items()))
        rows.append(row)
    df = pd.DataFrame(rows)
    df['signature'] = df[[f'type_L{L}' for L in range(3, 9)]].agg(' | '.join, axis=1)
    df['cycle_class'] = df['signature'].factorize()[0]
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/rules426_cycle_structure.csv', index=False)
    print(len(df), 'rules |', df['cycle_class'].nunique(), 'distinct cycle structures (L=3..8)')
    print('Saved results/rules426_cycle_structure.csv')


if __name__ == '__main__':
    main()
