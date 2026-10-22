# RCA clustering with the 426-rule reversible permutation family

Clustering of single-cell RNA-seq data with 3-state, radius-1 **reversible cellular automata (RCA)**.
All rules come from a family of 426 reversible rules (`rules/reversible_426_permutation_family.txt`).
This repository runs every rule on three datasets, relates clustering quality to each rule's
**cycle structure**, and selects rules for clustering.

## Results

| Dataset | Cells | Types | Selection | Rule | Stage 3 | ARI | NMI | ACC | F1 |
|---|---|---|---|---|---|---|---|---|---|
| Pollen | 249 | 11 | best ARI | `7207549538319` | centroid | 0.9482 | 0.9525 | 0.9518 | 0.9200 |
| Pollen | | | cycle filter + silhouette | `1838403620885` | centroid | 0.9419 | 0.9499 | 0.9357 | 0.8920 |
| Biase | 49 | 3 | best ARI | `277192716489` | centroid | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Biase | | | cycle filter + silhouette | `146064945221` | centroid | 0.8048 | 0.7975 | 0.9388 | 0.9396 |
| Klein | 2717 | 4 | best ARI | `7348218507513` | avg | 0.7463 | 0.6905 | 0.8892 | 0.8768 |
| Klein | | | cycle filter + silhouette | `7348218507513` | avg | 0.7463 | 0.6905 | 0.8892 | 0.8768 |

- **Best ARI**: the rule with the highest ARI among the 426, chosen using the true labels (upper bound).
- **Cycle filter + silhouette**: label-free. Keep rules with 45 ≤ number of cycles on length-6
  configurations < 729, then take the rule and Stage-3 mode with the highest silhouette.

The pipeline is deterministic, so re-running gives exactly these numbers (full table: `results/summary.csv`).

## Cycle structure and clustering quality

Each rule acts as a permutation of the 3^L configurations of length L (null boundaries).
The 426 rules fall into **58 distinct cycle structures** (`results/rules426_cycle_structure.csv`).

- The number of cycles on length 6 (the CA block length), `ncyc_L6`, is the cycle feature that best
  predicts ARI. Spearman correlation with ARI (Stage 3 = centroid): Pollen 0.89, Biase 0.60, Klein 0.69.
- Rules with few, long cycles (fewer than ~20 cycles on length 6) give ARI close to 0 on all datasets.
- The identity-like class (729 fixed points) never reduces the encoding and fails on Klein, so it is excluded.
- A strict filter (`ncyc_L6` ≥ 170, 73 rules) raises the mean ARI (centroid) from 0.63 to 0.91 on Pollen,
  0.43 to 0.62 on Biase and 0.22 to 0.42 on Klein, but it removes the best rules for Biase (89 cycles)
  and Klein (48 cycles). Thresholds from 0 to 45 all give the silhouette picks above
  (`results/cycle_threshold_selection.csv`).

Note: the threshold of 45 was set after looking at these three datasets.

## Pipeline

1. Normalise → 2,000 highly variable genes → PCA
2. Top 6 principal components, each quantised to 9 levels = 2 ternary digits (0/1/2)
3. Stage 1/2: 6-digit blocks are replaced by the index of their CA cycle until ≤ 8 digits remain
4. Stage 2 rule-2 cycles (Rule 2 = Rule 1), then Stage 3 merges groups down to the number of cell types,
   either by nearest centroid (`centroid`) or by the original HF_average rule (`avg`)

## How to run

Requires Python 3.10+.

```bash
pip install -r requirements.txt

python run_best.py           # selected rules on all datasets (~1 min)
python cycle_structure.py    # cycle structure of the 426 rules (~10 s)
python search_426.py         # all 426 rules x 2 Stage-3 modes on all datasets (~3 min, mostly Klein)
python select_by_cycles.py   # cycle structure vs ARI, threshold table (needs the two above)
```

`run_best.py` and `search_426.py` also accept dataset names, e.g. `python run_best.py Pollen`.

## Repository layout

```
rca.py              pipeline: preprocessing, ternary encoding, CA, Stages 1-3, evaluation
config.py           dataset settings and the selected rules
run_best.py         run the selected rules -> results/<dataset>/
search_426.py       run all 426 rules -> results/<dataset>/<dataset>_rule_search_426.csv
cycle_structure.py  cycle structure of every rule -> results/rules426_cycle_structure.csv
select_by_cycles.py correlation, per-class ARI, threshold selection -> results/
rules/              reversible_426_permutation_family.txt (rule id, 27-digit rule, tags)
data/               the three datasets
results/            pre-computed outputs of all scripts
```

## Data

| Dataset | Description | Source | File in `data/` |
|---|---|---|---|
| Pollen | human cell lines/tissues, already log-transformed | SIMLR repository (Test_3_Pollen) | `Pollen/Pollen_data.csv.gz` (cells × genes, gzip) |
| Biase | mouse zygote, 2-cell and 4-cell embryo blastomeres, FPKM | GEO GSE57249 | `Biase/Biase_data.csv` (cells × genes) |
| Klein | mouse embryonic stem cells, day 0/2/4/7, inDrop UMI counts | GEO GSE65525 | `Klein/klein_full_dataset.csv.gz` (genes × cells, gzip) |

Labels are matched to the data by row order. The large data files are gzip-compressed to stay
under GitHub's file-size limits; pandas reads them directly.
