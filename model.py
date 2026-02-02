r"""
Graph-linked unified embedding (GLUE) for single-cell multi-omics
data integration
"""

import time
import os
import pathlib
import dill
from typing import List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.distributions as D
import torch.nn.functional as F
from anndata import AnnData
from data import SimpleBalancedLoader, AnchorPairedDataset,MixedDataset, SingleModalDataset,SingleModal_collate_fn
from torch.utils.data import DataLoader
from collections import deque

from utils import *


class SIVAModel(nn.Module):
    def __init__(self, data_configs,rna_input_dim,atac_input_dim, GP_dim, Normal_dim, rna_encoder_layers, rna_decoder_layers, atac_encoder_layers, atac_decoder_layers,encoder_dropout, decoder_dropout, fixed_inducing_points, initial_inducing_points, fixed_gp_params, kernel_scale, rna_N_train,atac_N_train, dtype, device):
        super(SIVAModel, self).__init__()
        torch.set_default_dtype(dtype)
        self.keys = ['rna','atac']
        self.modalities = data_configs
        self.rna_input_dim = rna_input_dim

        self.dtype = dtype
        self.GP_dim = GP_dim            # dimension of latent Gaussian process embedding
        self.Normal_dim = Normal_dim    # dimension of latent standard Gaussian embedding

        self.device = device
        self.rna_encoder = DenseEncoder(input_dim=rna_input_dim, hidden_dims=rna_encoder_layers, output_dim=GP_dim+Normal_dim, activation="elu", dropout=encoder_dropout)
        self.rna_decoder = buildNetwork([GP_dim+Normal_dim]+rna_decoder_layers, activation="elu", dropout=decoder_dropout)

        self.rna_dec_mean = nn.Sequential(nn.Linear(rna_decoder_layers[-1], rna_input_dim), MeanAct())
        self.rna_dec_disp = nn.Parameter(torch.randn(self.rna_input_dim), requires_grad=True)       # trainable dispersion parameter for NB loss
        if GP_dim>0:
            self.rna_svgp = ZSVGP(fixed_inducing_points=fixed_inducing_points, initial_inducing_points=initial_inducing_points,
                    fixed_gp_params=fixed_gp_params, kernel_scale=kernel_scale, N_train=rna_N_train,GP_dim=GP_dim, dtype=dtype, device=device)
            self.atac_svgp = ZSVGP(fixed_inducing_points=fixed_inducing_points, initial_inducing_points=initial_inducing_points,
                    fixed_gp_params=fixed_gp_params, kernel_scale=kernel_scale, N_train=atac_N_train,GP_dim=GP_dim, dtype=dtype, device=device)
        else:
            self.rna_svgp = None
            self.atac_svgp = None

        self.atac_input_dim = atac_input_dim
        self.atac_encoder = DenseEncoder(input_dim=atac_input_dim, hidden_dims=atac_encoder_layers, output_dim=GP_dim+Normal_dim, activation="elu", dropout=encoder_dropout)
        self.atac_decoder = buildNetwork([GP_dim+Normal_dim]+ atac_decoder_layers, activation="elu", dropout=decoder_dropout)

        self.atac_dec_mean = nn.Sequential(nn.Linear(atac_decoder_layers[-1], atac_input_dim,), nn.Sigmoid())

        self.l_encoder = buildNetwork([atac_input_dim,]+atac_encoder_layers, activation="elu", dropout=encoder_dropout)
        self.l_encoder.append(nn.Linear(atac_encoder_layers[-1], 1))

        self.peak_bias = nn.Parameter(torch.zeros(atac_input_dim), requires_grad=True)

        self.BCE_loss = nn.BCELoss(reduction="none").to(self.device)
        self.NB_loss = NBLoss().to(self.device)
        self.to(device)
    
    
    def save(self, fname: os.PathLike) -> None:
        fname = pathlib.Path(fname)
        with fname.open("wb") as f:
            dill.dump(self, f, protocol=4, byref=False, recurse=True)

    @torch.no_grad()
    def encode_data(
            self, key: str, adata: AnnData, batch_size: int = 128, 
            n_sample: Optional[int] = None
    ) -> np.ndarray:

        self.eval()
        if key =='rna':
            encoder = self.rna_encoder
            svgp  = self.rna_svgp
        if key =='atac':
            encoder = self.atac_encoder
            svgp  = self.atac_svgp
        data = SingleModalDataset(
            key,adata, self.modalities[key]
        )
        data_loader = DataLoader(
            data, batch_size=batch_size, shuffle=False, drop_last=False,
            collate_fn=SingleModal_collate_fn
        )
        result = []
        for batch_data in data_loader:
            x = batch_data['counts'].to(self.device)
            xpos = batch_data['xpos'].to(self.device)
            mu, var = encoder(x)
            if svgp:
                result.append(svgp.forward_eval(mu, var,xpos))
            else:
                result.append(mu)

        return torch.cat(result).cpu().numpy()

class SIVATrainer:

    def __init__(
            self, net: SIVAModel,
            max_epochs = 1000,
            random_seed: int = 0, 
            lam_data: float = 1.0,
            lam_kl: float = 1.0,
            lam_gp_kl: float = 1.0,
            KL_LOSS: float = 0.025,
            dynamicVAE: bool = True, 
            init_beta: float = 10, 
            min_beta: float = 5, 
            max_beta: float = 25,
            lam_mag: float = 1.0,
            lam_mmd: float = 5.0,
            lam_gaualign: float = 1.0,
            modality_weight: Optional[Mapping[str, float]] = None,
            optim: str = "AdamW", 
            lr: float = 1e-3, 
            patience: int = 30,
            **kwargs
    ) -> None:  
        self.net = net
        self.max_epochs = max_epochs
        self.random_seed = random_seed
        self.edge_seed  = random_seed
        self.lam_data = lam_data
        self.lam_kl = lam_kl
        self.lam_gp_kl = lam_gp_kl
        
        self.PID = {k: PIDControl(Kp=0.01, Ki=-0.005, init_beta=init_beta, min_beta=min_beta, max_beta=max_beta)
                     for k in self.net.keys}
        self.KL_LOSS = KL_LOSS          # expected KL loss value
        self.dynamicVAE = dynamicVAE
        self.beta = {k:init_beta for k in self.net.keys}          # beta controls the weight of reconstruction loss

        self.lam_mag = lam_mag
        self.lam_mmd = lam_mmd
        self.lam_gaualign = lam_gaualign
        if modality_weight is None:
            self.modality_weight = {k: 1.0 for k in self.net.keys}
        self.lr = lr
        self.patience = patience

        self.vae_optim = getattr(torch.optim, optim)(
                self.net.parameters(), lr=self.lr, **kwargs
            )

        self.earlystop = EarlyStopping(patience=self.patience)
        print("SIVATrainer initialization finished!")


    def compute_losses(
            self, batch_data
    ) -> Mapping[str, torch.Tensor]:
        net = self.net

        for modality_key in net.keys:
            for datatype_key in batch_data[modality_key].keys():
                batch_data[modality_key][datatype_key] = batch_data[modality_key][datatype_key].to(net.device, non_blocking=True)
        batch_data['is_anchor'] = batch_data['is_anchor'].to(net.device)
        batch_size = batch_data['rna']['counts'].shape[0]
        rna_mu, rna_var = net.rna_encoder(batch_data['rna']['counts'])
        atac_mu, atac_var = net.atac_encoder(batch_data['atac']['counts'])
        if net.GP_dim >0:
            rna_ugp, rna_gp_kl, rna_gaussian_kl  = net.rna_svgp(rna_mu, rna_var,batch_data['rna']['xpos'])
            atac_ugp, atac_gp_kl, atac_gaussian_kl  = net.atac_svgp(atac_mu, atac_var,batch_data['atac']['xpos'])
            mmd_loss_gp = imq_kernel(rna_ugp.mean[:net.GP_dim], atac_ugp.mean[:net.GP_dim], h_dim=net.GP_dim) 
            mmd_loss_gaussian = imq_kernel(rna_ugp.mean[net.GP_dim:], atac_ugp.mean[net.GP_dim:], h_dim=net.Normal_dim)
            mmd_loss = mmd_loss_gp + self.lam_gaualign * mmd_loss_gaussian
        else:
            mmd_loss = imq_kernel(rna_mu, atac_mu, h_dim=rna_mu.shape[1]) 

        knn_loss = 0.0
        anchor_batch_size = batch_data['is_anchor'].sum().item()
        knn_loss +=  F.mse_loss(rna_mu[batch_data['is_anchor']], atac_mu[batch_data['is_anchor']], reduction='none').sum() / anchor_batch_size

        # rsample()
        if net.GP_dim >0: 
            rna_z = rna_ugp.rsample()
            atac_z = atac_ugp.rsample()
        else:
            rna_dist = D.Normal(rna_mu, torch.sqrt(rna_var))
            atac_dist = D.Normal(atac_mu, torch.sqrt(atac_var))
            rna_z = rna_dist.rsample()
            atac_z = atac_dist.rsample()

        # decode data and reconstruction loss
        rna_hidden = net.rna_decoder(rna_z)
        rna_mean = net.rna_dec_mean(rna_hidden)
        rna_disp = (torch.exp(torch.clamp(net.rna_dec_disp, -15., 15.))).unsqueeze(0)

        rna_recon_loss = net.NB_loss(x=batch_data['rna']['xraw'], mean=rna_mean , disp=rna_disp, scale_factor=batch_data['rna']['xsf'])/batch_size
        
        atac_hidden = net.atac_decoder(atac_z)
        atac_mean = net.atac_dec_mean(atac_hidden)

        l_samples = net.l_encoder(batch_data['atac']['xraw'])
        atac_recon_loss = net.BCE_loss(torch.sigmoid(l_samples)[:None] * atac_mean * torch.sigmoid(net.peak_bias).unsqueeze(0), batch_data['atac']['xraw']).sum()/batch_size

        if net.GP_dim >0: 
            rna_kl = (rna_gp_kl + rna_gaussian_kl)/batch_size
            atac_kl = (atac_gp_kl + atac_gaussian_kl)/batch_size
        else:
            prior_dist = D.Normal(torch.zeros_like(rna_mu), torch.ones_like(rna_mu))

            rna_kl = D.kl_divergence(rna_dist, prior_dist).sum()/batch_size
            atac_kl = D.kl_divergence(atac_dist, prior_dist).sum()/batch_size

        rna_elbo = rna_recon_loss + self.beta['rna'] * rna_kl 
        atac_elbo = atac_recon_loss + self.beta['atac'] * atac_kl

        vae_loss = rna_elbo + atac_elbo 
        gen_loss = vae_loss + self.lam_mmd * mmd_loss  + self.lam_mag * knn_loss

        losses = {
            "mmd_loss": mmd_loss, "knn_loss": knn_loss, "vae_loss": vae_loss, "gen_loss": gen_loss
            
        }

        if net.GP_dim >0:
            losses.update({ 
                            "rna_gp_kl": rna_gp_kl/batch_size,
                            "rna_gaussian_kl": rna_gaussian_kl/batch_size,
                            "rna_kl": rna_kl,
                            "rna_recon": rna_recon_loss,
                            "rna_elbo": rna_elbo,
                            "atac_gp_kl": atac_gp_kl/batch_size,
                            "atac_gaussian_kl": atac_gaussian_kl/batch_size,
                            "atac_kl": atac_kl,
                            "atac_recon": atac_recon_loss,
                            "atac_elbo": atac_elbo
                        })
        else:
            losses.update({ 
                            "rna_kl": rna_kl,
                            "rna_recon": rna_recon_loss,
                            "rna_elbo": rna_elbo,
                            "atac_kl": atac_kl,
                            "atac_recon": atac_recon_loss,
                            "atac_elbo": atac_elbo
                        })
        return losses,batch_size

    def train_step(
            self, batch_data
    ) -> Mapping[str, torch.Tensor]:
        self.net.train()
        losses,batch_size = self.compute_losses(batch_data)
        self.net.zero_grad(set_to_none=True)
        losses["gen_loss"].backward()
        self.vae_optim.step()

        return losses,batch_size

    def train(self, adatas, train_loader,  max_epochs,directory):

        earlystop_model_dir = directory / 'Eearly_Stop_Models'
        self.earlystop.model_file = directory / 'Earlystop_model.pt'

        if not os.path.exists(earlystop_model_dir):
            os.makedirs(earlystop_model_dir, exist_ok=True)
        queue ={'rna':deque(), 'atac':deque()} 
        for epoch in range(max_epochs):
            epoch_start_time = time.time()
            train_losses = {}
            batch_train_count = 0
            KL_val = {}
            avg_KL= {}
            for idx, train_batch in enumerate(train_loader):
                batch_train_losses,train_batch_size = self.train_step(train_batch)
                train_losses = {
                    key: train_losses.get(key, 0) + value
                    for key, value in batch_train_losses.items()
                }
                batch_train_count += 1
                if self.dynamicVAE:
                    for k in self.net.keys:
                        KL_val[k] = train_losses[f"{k}_kl"].item()
                        queue[k].append(KL_val[k])
                        avg_KL[k] = np.mean(queue[k])
                        self.beta[k], _ = self.PID[k].pid(self.KL_LOSS*(self.net.GP_dim +self.net.Normal_dim), avg_KL[k])
                        if len(queue[k]) >= 10:
                            queue[k].popleft()
                
            train_metrics = {
                    key: float(f"{(value/ batch_train_count):.6f}")
                    for key, value in train_losses.items()
                }
            if epoch+1 > 300:
                self.earlystop(train_metrics["vae_loss"], self.net)
                if self.earlystop.early_stop:
                        print("EarlyStopping: run {} epochs".format(epoch+1))
                        self.log_file.write('EarlyStopping: run {} epochs'.format(epoch+1) + '\n')
                        break

            epoch_end_time = time.time()
            elapsed_time = epoch_end_time - epoch_start_time
            print("[Epoch: {}] train={}, current beta={}, {:.2f}s elapsed".format(epoch+1, train_metrics, self.beta, elapsed_time))

            self.log_file.write("[Epoch: {}] train={}, current beta={}, {:.2f}s elapsed".format(epoch+1, train_metrics, self.beta, elapsed_time) + '\n')
        
        torch.save(self.net.state_dict(), directory / f"model_final_epoch.pt")
        

    def fit(
            self, adatas: Mapping[str, AnnData], anchor_matrix: Mapping[str, torch.Tensor],
            data_batch_size: int = 128,
            directory: Optional[os.PathLike] = None
    ) -> None:

        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.log_file = open(directory / 'log_file.txt',mode='a+')

        anchor_dataset = AnchorPairedDataset(adatas, self.net.modalities, anchor_matrix)
        mix_datasat = MixedDataset(adatas, self.net.modalities)

        dataloader = SimpleBalancedLoader(
            anchor_dataset, mix_datasat, 
            batch_size=data_batch_size, paired_ratio=0.3  # 30%来自anchor对
        )
        print("Begin SIVA Training! ")
        try:
            self.train(adatas,dataloader,self.max_epochs,directory)
        finally:
            self.log_file.close()
    


def load_model(fname: os.PathLike) -> SIVAModel:

    fname =pathlib.Path(fname)
    with fname.open("rb") as f:
        model = dill.load(f)
    # model.device = device  # pylint: disable=no-member
    return model
