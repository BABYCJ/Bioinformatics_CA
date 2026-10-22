"""Dataset settings and the selected rules (all from rules/reversible_426_permutation_family.txt)."""
import numpy as np

RULES_FILE = 'rules/reversible_426_permutation_family.txt'

DATASETS = {
    'Pollen': dict(
        name='Pollen',
        data_file='data/Pollen/Pollen_data.csv.gz',       # cells x genes (log-transformed), SIMLR Test_3_Pollen
        label_file='data/Pollen/Pollen_labels.csv',
        genes_in_rows=False, normalisation='none', dtype=np.float64, min_cells=3, eps=1e-12),
    'Biase': dict(
        name='Biase',
        data_file='data/Biase/Biase_data.csv',            # cells x genes (FPKM), GEO GSE57249
        label_file='data/Biase/Biase_labels.csv',
        genes_in_rows=False, normalisation='log1p', dtype=np.float64, min_cells=3, eps=1e-12),
    'Klein': dict(
        name='Klein',
        data_file='data/Klein/klein_full_dataset.csv.gz',  # genes x cells (raw UMI counts), GEO GSE65525
        label_file='data/Klein/Klein_cell_label.csv',
        genes_in_rows=True, normalisation='counts', dtype=np.float32, min_cells=10, eps=0.0),
}

# Best ARI among the 426 rules (chosen with the true labels).
BEST_ARI = {
    'Pollen': ('7207549538319', 'centroid'),
    'Biase':  ('277192716489',  'centroid'),
    'Klein':  ('7348218507513', 'avg'),
}

# Label-free choice: keep rules with MIN_CYCLES_L6 <= #cycles on length-6 configurations < 729,
# then take the rule / Stage-3 mode with the highest silhouette (see select_by_cycles.py).
MIN_CYCLES_L6 = 45
CYCLE_SILHOUETTE = {
    'Pollen': ('1838403620885', 'centroid'),
    'Biase':  ('146064945221',  'centroid'),
    'Klein':  ('7348218507513', 'avg'),
}
