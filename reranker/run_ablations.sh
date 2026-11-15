#!/bin/bash
# Design-ablation grid for Table "tab:designablation".  Every run uses the same
# leave-one-target-out protocol and the same feature table; one design choice
# changes per run.  CPU only, ~80 s each.
set -e
VP="C:/Users/samuel/AppData/Local/Temp/claude/e60efc1e4d6f85de/c3813cdd-517a-4498-bfb6-a20ce9f97e58/scratchpad/venv/Scripts/python.exe"
F=features_folds.csv
mkdir -p ablations

# Feature count: 5 and 21 AF3 features, on the SAME table as the 37-feature
# reference, so the three rows of that block differ only in feature count.
"$VP" train_reranker.py --features $F --feature-set af3core   --out ablations/ab_feat5.csv
"$VP" train_reranker.py --features $F --feature-set af3       --out ablations/ab_feat21.csv

# Capacity, at the reference feature set (37).
"$VP" train_reranker.py --features $F --feature-set af3+input --hidden        --out ablations/ab_linear.csv
"$VP" train_reranker.py --features $F --feature-set af3+input --hidden 16     --out ablations/ab_h16.csv
"$VP" train_reranker.py --features $F --feature-set af3+input --hidden 128 64 --out ablations/ab_wide.csv

# Loss and normalisation.
"$VP" train_reranker.py --features $F --feature-set af3+input --loss bce            --out ablations/ab_bce.csv
"$VP" train_reranker.py --features $F --feature-set af3+input --standardise pooled  --out ablations/ab_pooled.csv
echo "ALL ABLATION RUNS COMPLETE"
