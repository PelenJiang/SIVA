# !/bin/bash
CUDA_VISIBLE_DEVICES=0 python ./run_SIVA.py \
--input-rna "./Data/Processed_MISAR_Data/E15_5-S1/E15_5-S1_mRNA_w_histo_labels_pp.h5ad" \
--input-atac "./Processed_MISAR_Data/E15_5-S1/E15_5-S1_ATAC_w_histo_labels_pp.h5ad" \
--input-anchor "./Data/Processed_MISAR_Data/seurat_results/E15_S1_seurat_anchors_matrix.csv" \
--train-dir "./Results/"  \
--GP_dim 4 \
--Normal_dim 16 \
--inducing_point_steps 14 \
--lam-mmd 5 \
--lam-mag 1 \
--random-seed 3 \
-p