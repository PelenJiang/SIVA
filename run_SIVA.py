import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import sys
import time
import pathlib
import argparse
import anndata as ad
import scanpy as sc
import numpy as np
import torch
import pandas as pd
import functools
from model import SIVATrainer, SIVAModel
from data import configure_dataset
from sklearn.preprocessing import MinMaxScaler
import scib
import yaml
import metrics as mc
import matplotlib.pyplot as plt
from matplotlib import rcParams


def parse_args() -> argparse.Namespace:
    r"""
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', dest='result_dir', type=str, default="MyResults/",
        help="Path to directory where training logs and checkpoints are stored")
    parser.add_argument('--input-rna', dest='input_rna', type=pathlib.Path, required=True,
                        help="Path to input RNA dataset (.h5ad)")
    parser.add_argument('--input-atac', dest='input_atac', type=pathlib.Path, required=True, 
                        help="Path to input ATAC dataset (.h5ad)")
    parser.add_argument('--input-anchor', dest='input_anchor', type=str, required=True)
    parser.add_argument('--dropoutE', default=0, type=float,
                        help='dropout probability for encoder')
    parser.add_argument('--dropoutD', default=0, type=float,
                        help='dropout probability for decoder')
    parser.add_argument('--rna_encoder_layers', nargs="+", default=[128, 64], type=int)
    parser.add_argument('--rna_decoder_layers', nargs="+", default=[64, 128], type=int)
    parser.add_argument('--atac_encoder_layers', nargs="+", default=[1024, 128], type=int)
    parser.add_argument('--GP_dim', default=4, type=int,help='dimension of the latent Gaussian process embedding')
    parser.add_argument('--Normal_dim', default=16, type=int,help='dimension of the latent standard Gaussian embedding')
    parser.add_argument('--atac_decoder_layers', nargs="+", default=[128, 1024], type=int)
    parser.add_argument('--dynamicVAE', action='store_false', default=True, 
                        help='whether to use dynamicVAE to tune the value of beta, if setting to false, then beta is fixed to initial value')
    parser.add_argument('--init_beta', default=10, type=float, help='initial coefficient of the KL loss')
    parser.add_argument('--min_beta', default=4, type=float, help='minimal coefficient of the KL loss')
    parser.add_argument('--max_beta', default=25, type=float, help='maximal coefficient of the KL loss')
    parser.add_argument('--KL_loss', default=0.025, type=float, help='desired KL_divergence value')
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--fix_inducing_points', default=True, type=bool)
    parser.add_argument('--grid_inducing_points', default=True, type=bool, 
                        help='whether to generate grid inducing points or use k-means centroids on locations as inducing points')
    parser.add_argument('--inducing_point_steps', default=14, type=int)
    parser.add_argument('--inducing_point_nums', default=None, type=int)
    parser.add_argument('--fixed_gp_params', default=False, type=bool)
    parser.add_argument('--loc_range', default=20., type=float)
    parser.add_argument('--kernel_scale', default=20., type=float)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-epochs', dest='max_epochs', type=int, default=2000, help="Maximum iteration")
    parser.add_argument('--lam-mag', dest='lam_mag', type=float, default=0.05,help="MAG loss weight")
    parser.add_argument('--lam-mmd', dest="lam_mmd", type=float, default=0.05,help="MMD loss weight")
    parser.add_argument('--lam-gaualign', dest="lam_gaualign", type=float, default=1.0, help="gaussian MMD alignment weight")
    parser.add_argument('--lam-data', dest="lam_data", type=float, default=1.0,help="Modality weight")
    parser.add_argument('--lam-kl', dest="lam_kl", type=float, default=1.0,help="KL weight")
    parser.add_argument('--lr', dest="lr", type=float, default=1e-3,help="Learning rate")
    parser.add_argument('--patience', dest="patience", type=int, default=50, help="Patience for earlystopping")
    parser.add_argument('--data-batch-size', dest="data_batch_size", type=int, default=512,
                        help="Number of cells in each data minibatch")
    parser.add_argument( '-s', '--random-seed', dest="random_seed", type=int, default=0, help="Random seed")
    parser.add_argument('-p', "--paired", dest="paired", default=False, action="store_true",
                        help="Whether the latent embeddings are paired")
    parser.add_argument('--train-dir', dest='train_dir', type=str)
    return parser.parse_args()


def cal_metrics(rna, atac, result_dir):
    combined = ad.concat([rna, atac])
    rcParams["figure.figsize"] = (4, 4)
    sc.pp.neighbors(combined, use_rep="X_emb", metric="cosine")
    sc.tl.umap(combined)
    sc.pl.umap(combined, color=["histo_labels","domain"], wspace=0.65, show=False)
    plt.savefig(result_dir + '/cluster_plot.png')

    datasets = [rna, atac]
    cell_types = [dataset.obs['histo_labels'].to_numpy() for dataset in datasets]
    domains = [dataset.obs["domain"].to_numpy() for dataset in datasets]
    latents = [dataset.obsm["X_emb"] for dataset in datasets]

    combined_cell_type = np.concatenate(cell_types)
    combined_domain = np.concatenate(domains)
    combined_latent = np.concatenate(latents)
    metrics_results = {
        "MAP":
            mc.mean_average_precision(combined_latent, combined_cell_type),
        "ASW_class":
            mc.avg_silhouette_width(combined_latent, combined_cell_type),
        "GC":
            mc.graph_connectivity(combined_latent, combined_cell_type),
        "SAS":
            mc.seurat_alignment_score(combined_latent, combined_domain, random_state=0),
        "ASW_domain":
            mc.avg_silhouette_width_batch(combined_latent, combined_domain, combined_cell_type),
    }
    if args.paired:
        metrics_results["FOSCTTM"] = np.concatenate(
            mc.foscttm(*latents)
        ).mean().item()
    else:
        metrics_results["FOSCTTM"] = None
    adata_integrated = sc.concat([rna, atac])
    res_max, nmi_max, nmi_all = scib.clustering.cluster_optimal_resolution(
        adata_integrated,
        label_key="histo_labels",
        cluster_key="cluster",
        use_rep="X_emb",
        force=True,
        verbose=True,
        return_all=True,
    )
    adata_integrated.obs['domain'] = adata_integrated.obs['domain'].astype('category')
    adata_integrated.obs['histo_labels'] = adata_integrated.obs['histo_labels'].astype('category')
    clisi = scib.me.clisi_graph(adata_integrated, label_key="histo_labels", type_="embed", use_rep="X_emb")
    ari = scib.me.ari(adata_integrated, cluster_key="cluster", label_key="histo_labels")
    ilisi = scib.me.ilisi_graph(adata_integrated, batch_key="domain", type_="embed", use_rep="X_emb")
    kbet = scib.me.kBET(adata_integrated, batch_key="domain", label_key="histo_labels", type_="embed", embed="X_emb")
    metrics_results.update({
        'cLISI':  float(clisi),
        'ARI':  float(ari),
        'NMI':  float(nmi_max),
        'iLISI':  float(ilisi),
        'kBET':  float(kbet)
    })
    biology_conservation =  (metrics_results['MAP'] + metrics_results['ASW_class'] + metrics_results['ARI']+ metrics_results['NMI']+ metrics_results['cLISI'])/5.0
    omics_mixing =  (metrics_results['SAS'] + metrics_results['ASW_domain'] + metrics_results['GC']+ metrics_results['iLISI']+ metrics_results['kBET'])/5.0

    metrics_computed = { "mean_biology_score": biology_conservation,
                        "mean_omics_score": omics_mixing
    }
    metrics_results.update(metrics_computed)
    print("Final metrics:", metrics_results)
    metrics_output_file = pathlib.Path(result_dir + '/metrics_results.yaml')
    print("[3/3] Saving results...")
    with metrics_output_file.open("w") as f:
        yaml.dump( metrics_results, f)


def main(args):
    start_time = time.time()

    rna = ad.read_h5ad(args.input_rna)
    atac = ad.read_h5ad(args.input_atac)

    n_counts =  rna.layers['counts'].toarray().sum(axis=1)
    rna.obs['size_factors'] = n_counts / np.median(n_counts)

    atac.layers['counts'] = atac.X.copy()
    atac.X[atac.X>0] = 1

    rna.obsm['raw_spatial'] = rna.obsm['spatial']
    atac.obsm['raw_spatial'] = atac.obsm['spatial']

    rna_scaler = MinMaxScaler()
    rna_loc = rna_scaler.fit_transform(rna.obsm['spatial']) * args.loc_range
    rna.obsm['spatial'] = rna_loc

    atac_scaler = MinMaxScaler()
    atac_loc = atac_scaler.fit_transform(atac.obsm['spatial']) * args.loc_range 
    atac.obsm['spatial'] = atac_loc

    anchors_matrix = pd.read_csv(args.input_anchor, index_col=0)
    anchors = anchors_matrix.rename(columns={'cell1': 'rna_idx','cell2': 'atac_idx'})
    anchors = anchors[anchors['score']>=0.5]
    rna_config = configure_dataset(
        rna,  use_highly_variable=True, use_size_factors="size_factors",
        use_layer="counts", use_spatial_pos="spatial"
    )
    atac_config = configure_dataset(
        atac,  use_highly_variable=True,use_spatial_pos="spatial"
    )
    data_configs = {"rna":rna_config,"atac":atac_config}
    eps = 1e-5
    initial_inducing_points = np.mgrid[0:(1+eps):(1./args.inducing_point_steps), 0:(1+eps):(1./args.inducing_point_steps)].reshape(2, -1).T * args.loc_range

    mymodel = SIVAModel(data_configs =data_configs, rna_input_dim = len(rna_config["features"]), atac_input_dim =len(atac_config["features"]),GP_dim=args.GP_dim, Normal_dim=args.Normal_dim,rna_encoder_layers=args.rna_encoder_layers, rna_decoder_layers=args.rna_decoder_layers, atac_encoder_layers=args.atac_encoder_layers, atac_decoder_layers=args.atac_decoder_layers, encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD, fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points, fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, rna_N_train=rna_config["total_cells"], atac_N_train=atac_config["total_cells"], dtype=torch.float64, device=args.device)

    trainer = SIVATrainer(net=mymodel, max_epochs=args.max_epochs, random_seed =args.random_seed, dynamicVAE = args.dynamicVAE,
                             lam_data = args.lam_data,lam_kl = args.lam_kl, lam_mag=args.lam_mag,init_beta = args.init_beta,
                             lam_mmd=args.lam_mmd,lam_gaualign=args.lam_gaualign,lr = args.lr,patience=args.patience)


    trainer.fit(adatas = {"rna": rna, "atac": atac}, anchor_matrix = anchors, 
                data_batch_size = args.data_batch_size,directory= args.result_dir + '/finetune')
    rna.obsm["X_emb"] = mymodel.encode_data("rna", rna)
    atac.obsm["X_emb"] = mymodel.encode_data("atac", atac)

    rna.write(args.result_dir+"/rna_SIVA.h5ad")
    atac.write(args.result_dir+"/atac_SIVA.h5ad")
    print("Evaluating results metrics...")

    cal_metrics(rna, atac, args.result_dir)

    print("Saving results...")
    pd.DataFrame(rna.obsm["X_emb"], index=rna.obs_names).to_csv(args.result_dir+"/rna_latent.csv", header=False)
    pd.DataFrame(atac.obsm["X_emb"], index=atac.obs_names).to_csv(args.result_dir+"/atac_latent.csv", header=False)
    mymodel.save(args.result_dir+"/final_model.dill")

    elapsed_time = time.time() - start_time
    run_info_path = pathlib.Path(args.result_dir + '/run_info.yaml')
    with run_info_path.open("w") as f:
        yaml.dump({
            "cmd": " ".join(sys.argv),
            "args": vars(args),
            "training_time": elapsed_time,
            "n_cells": atac.shape[0] + rna.shape[0]
        }, f)

if __name__ == "__main__":
    args = parse_args()
    for seed in range(args.random_seed):
        args.random_seed = seed
        args.result_dir = args.train_dir +"seed_"+ str(seed)
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir, exist_ok=True)
        main(args)