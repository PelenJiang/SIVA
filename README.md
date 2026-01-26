# SIVA (Spatially-Informed Variational Autoencoders and Anchor Guidance)
Code for: "SIVA: Diagonal Integration of Spatial Multi-Omics Data via Spatially-Informed Variational Autoencoders and Anchor Guidance"

## Installation

```
conda create -n siva python=3.9
pip install -r requirements.txt
```
for the installation of scib (used in evaluation code), there is an alternative option:
```
pip install scib --no-cache-dir --no-binary :all:
```
## Data:
Link of demo data (processed MISAR-seq E15-S1 sample) used for evaluation:
[Link](https://pan.baidu.com/s/1F3chjYEvHBM3G_MIxQzAFg?pwd=6vc8)  
The data from different omics groups are stored using .h5ad file. The anchor matrix generated from Seurat is stored using .csv file.  
**.obs['histo_labels']** is the manual annotation label.  
**.obs['domain']** is the omics layer label.  
**.obsm['spatial']** saves the spatial position of each spot.  


## Usage:
Run SIVA Training script
```
bash ./run_SIVA.sh
```

## Parameters:
Illustration of parameters in SIVA Training script.  
**--input-rna**：Path to input RNA dataset (.h5ad)  
**--input-atac**：Path to input ATAC dataset (.h5ad)  
**--input-anchor**：Path to input anchor matrix (.csv)  
**--train-dir**：Training directory used for storing the results.  
**--GP_dim**：Dimension of the latent Gaussian process embedding  
**--Normal_dim**：Dimension of the latent standard Gaussian embeddin  
**--inducing_point_steps**：The number of inducing point steps used to generate inducing points.  
**--lam-mmd**：MMD loss weight  
**--lam-mag**: MAG loss weight  
**--random-seed**: The number of random seeds used for repeated experiments.  
**-p, --paired**: Whether the data is paired. When used, calculates FOSCTTM metric for evaluation.  