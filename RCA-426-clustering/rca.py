"""RCA clustering: 3-state reversible cellular automata (radius 1) applied to scRNA-seq data.

Pipeline (HF_average Stage 1 -> Stage 2 -> Stage 3, improved input encoding):
  1. Normalise -> 2,000 highly variable genes -> PCA
  2. Top 6 principal components, each quantised to 9 levels = 2 ternary digits (0/1/2)
  3. Stage 1/2: 6-digit blocks are replaced by the index of their CA cycle until <= 8 digits remain
  4. Stage 2 rule-2 cycles, then Stage 3 merges groups down to the number of cell types
"""
import bisect

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, f1_score,
                             silhouette_score, davies_bouldin_score, calinski_harabasz_score)

N_HVG          = 2000    # highly variable genes for PCA
N_PCS          = 6       # principal components encoded
DIGITS         = 2       # ternary digits per component (3**2 = 9 levels)
n              = 6       # CA block length
min_split_size = 3
max_split_size = 8
MAX_PASSES     = 100


# ---------------------------------------------------------------- rules

def load_rules(path):
    """Rules file: <rule id> <TAB> <27-digit rule string> [<TAB> tags ...]."""
    rules, tags = {}, {}
    with open(path, encoding='utf-8-sig') as f:          # file starts with a BOM
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 2 and len(parts[1]) == 27:
                rules[parts[0]] = parts[1]
                tags[parts[0]] = ' '.join(parts[2:])
    return rules, tags


def rule_table(rule_str):
    """27-digit rule string (most significant digit first, Wolfram order)
    -> lookup table indexed by 9*left + 3*centre + right."""
    return np.array([int(rule_str[26 - v]) for v in range(27)], dtype=np.int64)


def next_state_map(table, L):
    """next[v] = CA successor (null boundaries) of every length-L configuration v (base-3 number)."""
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


# ---------------------------------------------------------------- helpers

def to_values(D):
    return D @ (3 ** np.arange(D.shape[1] - 1, -1, -1))


def to_digits(values, width):
    return (values[:, None] // 3 ** np.arange(width - 1, -1, -1)) % 3


def apply_rule(states, L, ca):
    """Distinct data states grouped by CA cycle (as in HF_average apply_rule)."""
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


def _key(x):
    return (-len(x), x)


# ---------------------------------------------------------------- dataset

class Dataset:
    """Loads one dataset, preprocesses it and holds the ternary encoding."""

    def __init__(self, cfg, verbose=True):
        self.cfg = cfg
        data = pd.read_csv(cfg['data_file'], index_col=0)
        if cfg['genes_in_rows']:                       # genes x cells -> cells x genes
            data = data.T
        labels_df = pd.read_csv(cfg['label_file'])
        assert len(labels_df) == len(data), "label file and data must have the same number of cells"
        self.cells = data.index
        self.true_labels = labels_df.iloc[:, 1].astype(str).to_numpy()   # matched by row order

        X = data.to_numpy().astype(np.float32).astype(cfg['dtype'])
        del data
        self.N = N = len(X)
        self.num_clusters = len(np.unique(self.true_labels))
        if verbose:
            print(f"[{cfg['name']}] cells x genes: {X.shape} | {self.num_clusters} cell types:",
                  pd.Series(self.true_labels).value_counts().to_dict())

        # normalise -> highly variable genes -> PCA
        if cfg['normalisation'] == 'counts':           # raw counts: library-size normalisation + log
            norm = np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4)
        elif cfg['normalisation'] == 'log1p':          # FPKM / TPM
            norm = np.log1p(X)
        else:                                          # already log-transformed
            norm = X
        expressed = (X > 0).sum(axis=0) >= cfg['min_cells']
        dispersion = np.where(expressed, norm.var(axis=0) / (norm.mean(axis=0) + 1e-9), -np.inf)
        hvg = np.argsort(dispersion)[::-1][:min(N_HVG, int(expressed.sum()))]
        Z = norm[:, hvg]
        Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-9)
        self.P = PCA(n_components=min(30, N - 1, Z.shape[1]), random_state=0).fit_transform(Z)
        self.EVAL = self.P[:, :10]                     # space for silhouette / DB / CH

        # ternary (0/1/2) encoding
        LEVELS = 3 ** DIGITS
        F = self.P[:, :N_PCS]
        levels = np.empty(F.shape, dtype=np.int64)
        for j in range(N_PCS):
            lo, hi = np.percentile(F[:, j], [1, 99])
            levels[:, j] = np.clip(np.floor((F[:, j] - lo) / (hi - lo + cfg['eps']) * LEVELS), 0, LEVELS - 1)
        pw = 3 ** np.arange(DIGITS - 1, -1, -1)
        digits = ((levels[:, :, None] // pw) % 3).reshape(N, -1)          # PC1 first
        padded_length = ((digits.shape[1] + n - 1) // n) * n
        self.levels = levels
        self.enc_digits = np.hstack([np.zeros((N, padded_length - digits.shape[1]), dtype=np.int64), digits])

    # ------------------------------------------------ Stages 1-3

    def reduce_with_rule1(self, ca1):
        """Stage 1 + Stage 2: blocks -> cycle codes (cycles sorted by average) until <= max_split_size digits."""
        N, D = self.N, self.enc_digits
        for depth in range(MAX_PASSES + 1):
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

    def stage3_avg(self, cycles, values):
        """Original HF_average Stage 3: dissolve the smallest cycle into the cycle whose
        (rounded) average value changes least."""
        cyc = sorted((list(c) for c in cycles), key=_key)
        while len(cyc) > self.num_clusters:
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

    def stage3_centroid(self, cycles, values):
        """Improved Stage 3: merge the smallest CA group into the group with the nearest
        centroid of encoded levels (size-weighted) until num_clusters groups remain."""
        cyc_of = {v: i for i, c in enumerate(cycles) for v in c}
        groups = np.array([cyc_of[v] for v in values.tolist()])
        G = len(cycles)
        size = np.bincount(groups, minlength=G).astype(np.float64)
        cen = np.zeros((G, self.levels.shape[1]))
        np.add.at(cen, groups, self.levels)
        cen /= size[:, None]
        parent = np.arange(G)
        alive = np.ones(G, dtype=bool)
        for _ in range(G - self.num_clusters):
            cand = np.flatnonzero(alive)
            g = cand[np.argmin(size[cand])]
            alive[g] = False
            others = np.flatnonzero(alive)
            t = others[np.argmin(((cen[others] - cen[g]) ** 2).sum(axis=1))]
            cen[t] = (cen[t] * size[t] + cen[g] * size[g]) / (size[t] + size[g])
            size[t] += size[g]
            parent[g] = t
        root = np.arange(G)
        for g in range(G):
            r = g
            while parent[r] != r:
                r = parent[r]
            root[g] = r
        return np.searchsorted(np.flatnonzero(alive), root[groups])

    def run_rule(self, rule_str, mode):
        """Full pipeline for one rule (used as both Rule 1 and Rule 2). mode: 'centroid' or 'avg'."""
        ca = CA(rule_str)
        merged = self.reduce_with_rule1(ca)
        values = to_values(merged)
        cycles = apply_rule(np.unique(values), merged.shape[1], ca)
        if mode == 'centroid':
            return self.stage3_centroid(cycles, values)
        return self.stage3_avg(cycles, values)

    # ------------------------------------------------ evaluation

    def evaluate(self, pred):
        true = self.true_labels
        t_ids, p_ids = np.unique(true), np.unique(pred)
        cont = np.array([[np.sum((pred == p) & (true == t)) for t in t_ids] for p in p_ids])
        r, c = linear_sum_assignment(-cont)                     # best cluster -> cell type matching
        mapping = {p_ids[i]: t_ids[j] for i, j in zip(r, c)}
        mapped = np.array([mapping.get(p, '?') for p in pred])
        return {'ARI': adjusted_rand_score(true, pred),
                'NMI': normalized_mutual_info_score(true, pred),
                'ACC': np.mean(mapped == true),
                'F1': f1_score(true, mapped, average='macro'),
                'Silhouette': silhouette_score(self.EVAL, pred),
                'Davies-Bouldin': davies_bouldin_score(self.EVAL, pred),
                'Calinski-Harabasz': calinski_harabasz_score(self.EVAL, pred)}
