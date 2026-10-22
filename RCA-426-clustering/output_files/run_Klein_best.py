"""Klein: RCA clustering with the best rule of the 426-rule reversible family (standalone).

Best rule : 7348218507513  (222000111000222111222111000), Stage 3 = avg (original HF_average)
Expected  : ARI 0.7463 | NMI 0.6905 | ACC 0.8892 | F1 0.8768

usage:  python run_Klein_best.py
output: results/Klein/Klein_best_rule_clusters.csv and .png
"""
import bisect
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, f1_score,
                             silhouette_score, davies_bouldin_score, calinski_harabasz_score)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- settings
NAME          = 'Klein'
DATA_FILE     = 'data/Klein/klein_full_dataset.csv.gz'   # genes x cells, raw UMI counts
LABEL_FILE    = 'data/Klein/Klein_cell_label.csv'
GENES_IN_ROWS = True
NORMALISATION = 'counts'
DTYPE         = np.float32
MIN_CELLS     = 10
EPS           = 0.0

RULE_ID  = '7348218507513'
RULE_STR = '222000111000222111222111000'

N_HVG, N_PCS, DIGITS = 2000, 6, 2
n, min_split_size, max_split_size, MAX_PASSES = 6, 3, 8, 100


# ---------------------------------------------------------------- cellular automaton
def rule_table(rule_str):
    return np.array([int(rule_str[26 - v]) for v in range(27)], dtype=np.int64)


def next_state_map(table, L):
    p = 3 ** np.arange(L - 1, -1, -1)
    d = (np.arange(3 ** L)[:, None] // p) % 3
    padded = np.pad(d, ((0, 0), (1, 1)))
    idx = 9 * padded[:, :-2] + 3 * padded[:, 1:-1] + padded[:, 2:]
    return (table[idx] @ p).tolist()


class CA:
    def __init__(self, rule_str):
        self.table = rule_table(rule_str)
        self.maps = {}

    def nxt(self, L):
        if L not in self.maps:
            self.maps[L] = next_state_map(self.table, L)
        return self.maps[L]


def to_values(D):
    return D @ (3 ** np.arange(D.shape[1] - 1, -1, -1))


def to_digits(values, width):
    return (values[:, None] // 3 ** np.arange(width - 1, -1, -1)) % 3


def apply_rule(states, L, ca):
    nxt = ca.nxt(L)
    present = set(states.tolist())
    visited, cycles = set(), []
    for v in sorted(present):
        if v in visited:
            continue
        cyc = [v]
        visited.add(v)
        u, steps = nxt[v], 0
        while u != v and steps < 3 ** L:
            if u in present and u not in visited:
                cyc.append(u)
                visited.add(u)
            u = nxt[u]
            steps += 1
        cycles.append(cyc)
    return cycles


def code_width(s):
    w = 1
    while 3 ** w < s:
        w += 1
    return w


# ---------------------------------------------------------------- data + encoding
data = pd.read_csv(DATA_FILE, index_col=0)
if GENES_IN_ROWS:
    data = data.T
labels_df = pd.read_csv(LABEL_FILE)
assert len(labels_df) == len(data), "label file and data must have the same number of cells"
cells = data.index
true_labels = labels_df.iloc[:, 1].astype(str).to_numpy()
X = data.to_numpy().astype(np.float32).astype(DTYPE)
del data
N = len(X)
num_clusters = len(np.unique(true_labels))
print(f"[{NAME}] cells x genes: {X.shape} | {num_clusters} cell types")

if NORMALISATION == 'counts':
    norm = np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4)
elif NORMALISATION == 'log1p':
    norm = np.log1p(X)
else:
    norm = X
expressed = (X > 0).sum(axis=0) >= MIN_CELLS
dispersion = np.where(expressed, norm.var(axis=0) / (norm.mean(axis=0) + 1e-9), -np.inf)
hvg = np.argsort(dispersion)[::-1][:min(N_HVG, int(expressed.sum()))]
Z = norm[:, hvg]
Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-9)
P = PCA(n_components=min(30, N - 1, Z.shape[1]), random_state=0).fit_transform(Z)
EVAL = P[:, :10]

LEVELS = 3 ** DIGITS
F = P[:, :N_PCS]
levels = np.empty(F.shape, dtype=np.int64)
for j in range(N_PCS):
    lo, hi = np.percentile(F[:, j], [1, 99])
    levels[:, j] = np.clip(np.floor((F[:, j] - lo) / (hi - lo + EPS) * LEVELS), 0, LEVELS - 1)
pw = 3 ** np.arange(DIGITS - 1, -1, -1)
digits = ((levels[:, :, None] // pw) % 3).reshape(N, -1)
padded_length = ((digits.shape[1] + n - 1) // n) * n
enc_digits = np.hstack([np.zeros((N, padded_length - digits.shape[1]), dtype=np.int64), digits])


# ---------------------------------------------------------------- Stages 1-3
def reduce_with_rule1(ca1):
    D = enc_digits
    for _ in range(MAX_PASSES + 1):
        blocks = to_values(D.reshape(-1, n)).reshape(N, -1)
        pieces = []
        for i in range(blocks.shape[1]):
            uniq, inv = np.unique(blocks[:, i], return_inverse=True)
            cycles = sorted(apply_rule(uniq, n, ca1), key=lambda c: (sum(c) / len(c), c))
            code_of = np.empty(len(uniq), dtype=np.int64)
            for l, cyc in enumerate(cycles):
                code_of[np.searchsorted(uniq, cyc)] = l
            pieces.append(to_digits(code_of[inv], code_width(len(cycles))))
        M = np.hstack(pieces)
        if M.shape[1] < min_split_size:
            M = np.hstack([M[:, :n], np.zeros((N, n - min(n, M.shape[1])), dtype=np.int64)])
        if M.shape[1] <= max_split_size:
            return M
        target = ((M.shape[1] + n - 1) // n) * n
        D = np.hstack([np.zeros((N, target - M.shape[1]), dtype=np.int64), M])
    raise RuntimeError(f"no reduction after {MAX_PASSES} passes")


def _key(x):
    return (-len(x), x)


def stage3_avg(cycles, values):
    """Original HF_average Stage 3: dissolve the smallest cycle into the cycle whose
    (rounded) average value changes least."""
    cyc = sorted((list(c) for c in cycles), key=_key)
    while len(cyc) > num_clusters:
        last = cyc.pop()
        S = np.array([sum(c) for c in cyc], dtype=np.float64)
        Ln = np.array([len(c) for c in cyc], dtype=np.float64)
        touched = set()
        for e in last:
            diff = np.abs(np.rint((S + e) / (Ln + 1)) - np.rint(S / Ln))
            cand = np.flatnonzero(diff == diff.min())
            idx = int(cand[np.argmin(S[cand])])
            cyc[idx].append(e); S[idx] += e; Ln[idx] += 1; touched.add(idx)
        moved = [cyc[i] for i in sorted(touched)]
        for i in sorted(touched, reverse=True):
            del cyc[i]
        for c in moved:
            bisect.insort(cyc, c, key=_key)
    where = {}
    for i, c in enumerate(cyc):
        for e in c:
            where.setdefault(e, i)
    return np.array([where.get(v, -1) for v in values.tolist()])


ca = CA(RULE_STR)
merged = reduce_with_rule1(ca)
values = to_values(merged)
pred = stage3_avg(apply_rule(np.unique(values), merged.shape[1], ca), values)


# ---------------------------------------------------------------- evaluation + output
t_ids, p_ids = np.unique(true_labels), np.unique(pred)
cont = np.array([[np.sum((pred == p) & (true_labels == t)) for t in t_ids] for p in p_ids])
r, c = linear_sum_assignment(-cont)
mapping = {p_ids[i]: t_ids[j] for i, j in zip(r, c)}
mapped = np.array([mapping.get(p, '?') for p in pred])
res = {'ARI': adjusted_rand_score(true_labels, pred),
       'NMI': normalized_mutual_info_score(true_labels, pred),
       'ACC': np.mean(mapped == true_labels),
       'F1': f1_score(true_labels, mapped, average='macro'),
       'Silhouette': silhouette_score(EVAL, pred),
       'Davies-Bouldin': davies_bouldin_score(EVAL, pred),
       'Calinski-Harabasz': calinski_harabasz_score(EVAL, pred)}

print(f"\n[{NAME}] rule {RULE_ID} ({RULE_STR}), Stage 3 = avg")
print(pd.Series(res).round(4).to_string())
print(pd.crosstab(pd.Series(true_labels, name='true cell type'), pd.Series(pred, name='predicted cluster')))

out = f'results/{NAME}'
os.makedirs(out, exist_ok=True)
pd.DataFrame({'cell': cells, 'true_label': true_labels, 'predicted_cluster': pred}) \
  .to_csv(f'{out}/{NAME}_best_rule_clusters.csv', index=False)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(P[:, 0], P[:, 1], c=pd.factorize(true_labels)[0], s=12, cmap='tab20')
ax[0].set_title(f"{NAME}: true cell types")
ax[1].scatter(P[:, 0], P[:, 1], c=pred, s=12, cmap='tab20')
ax[1].set_title(f"RCA rule {RULE_ID} (avg), ARI = {res['ARI']:.3f}")
for a in ax:
    a.set_xlabel('PC1'); a.set_ylabel('PC2')
plt.tight_layout(); plt.savefig(f'{out}/{NAME}_best_rule_clusters.png', dpi=150); plt.close(fig)
print(f"Saved {out}/{NAME}_best_rule_clusters.csv and .png")
