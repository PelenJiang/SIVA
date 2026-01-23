import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import anndata as ad
from collections import defaultdict
import random
from typing import Any, List, Mapping, Optional, Tuple
import pandas as pd
import scipy.sparse
from anndata import AnnData
from itertools import cycle

def configure_dataset(
        adata: AnnData,
        use_highly_variable: bool = True,
        use_spatial_pos: Optional[str] = None,
        use_size_factors:Optional[str] = None,
        use_layer: Optional[str] = None,
        use_rep: Optional[str] = None,
        use_batch: Optional[str] = None,
        use_obs_names: bool = False
) -> None:
    r"""
    Configure dataset for model training

    Parameters
    ----------
    adata
        Dataset to be configured
    use_highly_variable
        Whether to use highly variable features
    use_layer
        Data layer to use (key in ``adata.layers``)
    use_rep
        Data representation to use as the first encoder transformation
        (key in ``adata.obsm``)
    use_batch
        Data batch to use (key in ``adata.obs``)
    use_cell_type
        Data cell type to use (key in ``adata.obs``)
    use_dsc_weight
        Discriminator sample weight to use (key in ``adata.obs``)
    use_obs_names
        Whether to use ``obs_names`` to mark paired cells across
        different datasets

    Note
    -----
    The ``use_rep`` option applies to encoder inputs, but not the decoders,
    which are always fitted on data in the original space.
    """
    data_config = {}
    data_config["total_cells"] = adata.shape[0]
    if use_highly_variable:
        if "highly_variable" not in adata.var:
            raise ValueError("Please mark highly variable features first!")
        data_config["use_highly_variable"] = True
        data_config["features"] = adata.var.query("highly_variable").index.to_numpy().tolist()
    else:
        data_config["use_highly_variable"] = False
        data_config["features"] = adata.var_names.to_numpy().tolist()

    if use_spatial_pos:
        if use_spatial_pos not in adata.obsm:
            raise ValueError("Invalid `use_spatial_pos`!")

        data_config["use_spatial_pos"] = use_spatial_pos
    else:
        data_config["use_spatial_pos"] = None

    if use_size_factors:
        if use_size_factors not in adata.obs:
            raise ValueError("Invalid `use_size_factors`!")

        data_config["use_size_factors"] = use_size_factors
    else:
        data_config["use_size_factors"] = None  
    if use_layer:
        if use_layer not in adata.layers:
            raise ValueError("Invalid `use_layer`!")
        data_config["use_layer"] = use_layer
    else:
        data_config["use_layer"] = None
    if use_rep:
        if use_rep not in adata.obsm:
            raise ValueError("Invalid `use_rep`!")
        data_config["use_rep"] = use_rep
        data_config["rep_dim"] = adata.obsm[use_rep].shape[1]
    else:
        data_config["use_rep"] = None
        data_config["rep_dim"] = None
    if use_batch:
        if use_batch not in adata.obs:
            raise ValueError("Invalid `use_batch`!")
        data_config["use_batch"] = use_batch
        data_config["batches"] = pd.Index(
            adata.obs[use_batch]
        ).dropna().drop_duplicates().sort_values().to_numpy()  # AnnData does not support saving pd.Index in uns
    else:
        data_config["use_batch"] = None
        data_config["batches"] = None
    data_config["use_obs_names"] = use_obs_names
    
    return data_config
    # adata.uns[config.ANNDATA_KEY] = data_config


class SingleModalDataset(Dataset):
    def __init__(self, modal_key,  adata, data_config, default_dtype = 'float64'):
        self.modal_key = modal_key
        self.default_dtype = default_dtype
        self.adata = adata
        self.atac = adata
        
        # extract data
        self.counts = self._extract_x(self.adata,data_config)
        self.xraw = self._extract_xraw(self.adata,data_config)
        self.xpos = self._extract_xpos(self.adata,data_config)

        if self.modal_key == 'rna':
            self.xsf = self._extract_xsf(self.adata,data_config)
        
        print(f"SingleModalDataset: {len(self.adata)} cells!")
    
    def _extract_x(self, adata, data_config):
        # print(self.default_dtype)
        default_dtype = self.default_dtype
        features = data_config["features"]
        if not np.array_equal(adata.var_names, features):
            adata = adata[:, features]  # This will load all data to memory if backed
        x = adata.X
        if x.dtype.type is not default_dtype:
            x = x.astype(default_dtype)
        if scipy.sparse.issparse(x):
            x = x.toarray()
        return np.asarray(x)
    
    def _extract_xidx(self, adata, data_config):
        use_self_index = data_config["use_self_index"]
        if use_self_index:
            if use_self_index not in adata.obsm:
                raise ValueError(
                    f"Configured data representation '{use_self_index}' "
                    f"cannot be found in input data!"
                )
            xidx = np.asarray(adata.obsm[use_self_index],dtype=int)
            return xidx
        return np.zeros(adata.shape[0], dtype=int)

    def _extract_xraw(self, adata, data_config):
        default_dtype = self.default_dtype
        features = data_config["features"]
        use_layer = data_config["use_layer"]
        if not np.array_equal(adata.var_names, features):
            adata = adata[:, features]  # This will load all data to memory if backed
        if use_layer:
            if use_layer not in adata.layers:
                raise ValueError(
                    f"Configured data layer '{use_layer}' "
                    f"cannot be found in input data!"
                )
            xraw = adata.layers[use_layer]
        else:
            xraw = adata.X
        if xraw.dtype.type is not default_dtype:
            xraw = xraw.astype(default_dtype)
        if scipy.sparse.issparse(xraw):
            xraw = xraw.toarray()
        return np.asarray(xraw)

    def _extract_xpos(self, adata, data_config):
        default_dtype = self.default_dtype
        use_spatial_pos = data_config["use_spatial_pos"]
        if use_spatial_pos:

            if use_spatial_pos not in adata.obsm:
                raise ValueError(
                    f"Configured data representation '{use_spatial_pos}' "
                    f"cannot be found in input data!"
                )
            xpos = np.asarray(adata.obsm[use_spatial_pos]).astype(default_dtype)
            # print("xpos的shape是",xpos.shape)
            return xpos
        return np.empty((adata.shape[0], 0), dtype=default_dtype)

    def _extract_xsf(self, adata, data_config):
        default_dtype = self.default_dtype
        use_size_factors = data_config["use_size_factors"]
        if use_size_factors:
            if use_size_factors not in adata.obs:
                raise ValueError(
                    f"Configured data batch '{use_size_factors}' "
                    f"cannot be found in input data!"
                )
            return adata.obs[use_size_factors].to_numpy().astype(default_dtype)
        return np.ones(adata.shape[0], dtype=int)

    def __len__(self):
        return len(self.adata)
    
    def __getitem__(self, idx):
        """获取单个细胞数据（随机配对）"""
        data_idx = idx % len(self.counts)
        
        data = {
            'counts': self.counts[data_idx],
            'xraw': self.xraw[data_idx],
            'xpos': self.xpos[data_idx],
        }

        if self.modal_key == 'rna':
            data['xsf'] =  self.xsf[data_idx]

        return data

def SingleModal_collate_fn(batch):
    """
    batch: List, each iter is a dictionary from dataset.__getitem__
    """
    collated = {}
    # 遍历每个字段（counts, xraw, xpos, xsf）
    for key in batch[0].keys():
        # 收集该字段的所有样本
        samples = [d[key] for d in batch]
        collated[key] = torch.stack([torch.tensor(s) for s in samples])

    return collated



class AnchorPairedDataset(Dataset):
    
    def __init__(self, adatas, data_configs, anchors, default_dtype = 'float64', use_pca_lsi=True):
        self.default_dtype = default_dtype
        self.rna_adata = adatas['rna']
        self.atac_adata = adatas['atac']
        self.use_pca_lsi = use_pca_lsi
        self.anchors = self._process_anchors(anchors)

        self.rna_counts = self._extract_x(self.rna_adata,data_configs['rna'])
        self.atac_counts = self._extract_x(self.atac_adata,data_configs['atac'])
        self.rna_xraw = self._extract_xraw(self.rna_adata,data_configs['rna'])
        self.atac_xraw = self._extract_xraw(self.atac_adata,data_configs['atac'])
        self.rna_xpos = self._extract_xpos(self.rna_adata,data_configs['rna'])
        self.atac_xpos = self._extract_xpos(self.atac_adata,data_configs['atac'])

        self.rna_xsf = self._extract_xsf(self.rna_adata,data_configs['rna'])

        self.paired_indices = self._build_paired_indices()
        
        print(f"AnchorPairedDataset: Construct {len(self.paired_indices)} anchors")
    
    def _extract_x(self, adata, data_config):
        # print(self.default_dtype)
        default_dtype = self.default_dtype
        features = data_config["features"]
        if not np.array_equal(adata.var_names, features):
            adata = adata[:, features]  # This will load all data to memory if backed
        x = adata.X
        if x.dtype.type is not default_dtype:
            x = x.astype(default_dtype)
        if scipy.sparse.issparse(x):
            x = x.toarray()
        return np.asarray(x)

    def _extract_xraw(self, adata, data_config):
        default_dtype = self.default_dtype
        features = data_config["features"]
        use_layer = data_config["use_layer"]
        if not np.array_equal(adata.var_names, features):
            adata = adata[:, features]  # This will load all data to memory if backed
        if use_layer:
            if use_layer not in adata.layers:
                raise ValueError(
                    f"Configured data layer '{use_layer}' "
                    f"cannot be found in input data!"
                )
            xraw = adata.layers[use_layer]
        else:
            xraw = adata.X
        if xraw.dtype.type is not default_dtype:
            xraw = xraw.astype(default_dtype)
        if scipy.sparse.issparse(xraw):
            xraw = xraw.toarray()
        return np.asarray(xraw)

    def _extract_xpos(self, adata, data_config):
        default_dtype = self.default_dtype
        use_spatial_pos = data_config["use_spatial_pos"]
        if use_spatial_pos:

            if use_spatial_pos not in adata.obsm:
                raise ValueError(
                    f"Configured data representation '{use_spatial_pos}' "
                    f"cannot be found in input data!"
                )
            xpos = np.asarray(adata.obsm[use_spatial_pos]).astype(default_dtype)
            # print("xpos的shape是",xpos.shape)
            return xpos
        return np.empty((adata.shape[0], 0), dtype=default_dtype)

    def _extract_xsf(self, adata, data_config):
        default_dtype = self.default_dtype
        use_size_factors = data_config["use_size_factors"]
        if use_size_factors:
            if use_size_factors not in adata.obs:
                raise ValueError(
                    f"Configured data batch '{use_size_factors}' "
                    f"cannot be found in input data!"
                )
            return adata.obs[use_size_factors].to_numpy().astype(default_dtype)
        return np.ones(adata.shape[0], dtype=int)

    def _process_anchors(self, anchors):
        """Process Seurat anchor indices (Convert 1 to 0)"""
        processed = []
        for index, anchor in anchors.iterrows():
            rna_idx = anchor['rna_idx'] - 1 if anchor['rna_idx'] > 0 else 0
            atac_idx = anchor['atac_idx'] - 1 if anchor['atac_idx'] > 0 else 0
            processed.append({
                    'rna_idx': int(rna_idx),
                    'atac_idx': int(atac_idx),
                    'score': anchor.get('score', 1.0)
                })
        return processed
    
    def _build_paired_indices(self):
        paired_indices = []
        for anchor in self.anchors:
            paired_indices.append((anchor['rna_idx'], anchor['atac_idx'], anchor['score']))
        return paired_indices
    
    def __len__(self):
        return len(self.paired_indices)
    
    def __getitem__(self, idx):
        """获取配对的RNA和ATAC数据"""
        rna_idx, atac_idx, score = self.paired_indices[idx]
        # RNA data
        rna_data = {
            'counts': self.rna_counts[rna_idx],
            'xraw': self.rna_xraw[rna_idx],
            'xpos': self.rna_xpos[rna_idx],
            'xsf': self.rna_xsf[rna_idx]
        }
        
        # ATAC data
        atac_data = {
            'counts': self.atac_counts[atac_idx],
            'xraw': self.atac_xraw[atac_idx],
            'xpos': self.atac_xpos[atac_idx],
            # 'idx': atac_idx,
        }
        
        return {
            'rna': rna_data,
            'atac': atac_data,
            'is_anchor_pair': True
        }


class MixedDataset(Dataset):
    
    def __init__(self,  adatas, data_configs,default_dtype = 'float64', use_pca_lsi=True):
        self.default_dtype = default_dtype
        self.rna_adata = adatas['rna']
        self.atac_adata = adatas['atac']

        self.rna_counts = self._extract_x(self.rna_adata,data_configs['rna'])
        self.atac_counts = self._extract_x(self.atac_adata,data_configs['atac'])

        self.rna_xraw = self._extract_xraw(self.rna_adata,data_configs['rna'])
        self.atac_xraw = self._extract_xraw(self.atac_adata,data_configs['atac'])

        self.rna_xpos = self._extract_xpos(self.rna_adata,data_configs['rna'])
        self.atac_xpos = self._extract_xpos(self.atac_adata,data_configs['atac'])

        self.rna_xsf = self._extract_xsf(self.rna_adata,data_configs['rna'])

        self.all_rna_indices = list(range(len(self.rna_adata)))
        self.all_atac_indices = list(range(len(self.atac_adata)))
        
        print(f"MixedDataset: {len(self.rna_adata)} RNA cells, {len(self.atac_adata)} ATAC cells")
    
    def _extract_x(self, adata, data_config):
        default_dtype = self.default_dtype
        features = data_config["features"]
        if not np.array_equal(adata.var_names, features):
            adata = adata[:, features]  
        x = adata.X
        if x.dtype.type is not default_dtype:
            x = x.astype(default_dtype)
        if scipy.sparse.issparse(x):
            x = x.toarray()
        return np.asarray(x)
    
    def _extract_xidx(self, adata, data_config):
        use_self_index = data_config["use_self_index"]
        if use_self_index:
            if use_self_index not in adata.obsm:
                raise ValueError(
                    f"Configured data representation '{use_self_index}' "
                    f"cannot be found in input data!"
                )
            xidx = np.asarray(adata.obsm[use_self_index],dtype=int)
            return xidx
        return np.zeros(adata.shape[0], dtype=int)

    def _extract_xraw(self, adata, data_config):
        # print(self.default_dtype)
        default_dtype = self.default_dtype
        features = data_config["features"]
        use_layer = data_config["use_layer"]
        if not np.array_equal(adata.var_names, features):
            adata = adata[:, features]  # This will load all data to memory if backed
        if use_layer:
            if use_layer not in adata.layers:
                raise ValueError(
                    f"Configured data layer '{use_layer}' "
                    f"cannot be found in input data!"
                )
            xraw = adata.layers[use_layer]
        else:
            xraw = adata.X
        if xraw.dtype.type is not default_dtype:
            xraw = xraw.astype(default_dtype)
        if scipy.sparse.issparse(xraw):
            xraw = xraw.toarray()
        return np.asarray(xraw)

    def _extract_xpos(self, adata, data_config):
        default_dtype = self.default_dtype
        use_spatial_pos = data_config["use_spatial_pos"]
        if use_spatial_pos:

            if use_spatial_pos not in adata.obsm:
                raise ValueError(
                    f"Configured data representation '{use_spatial_pos}' "
                    f"cannot be found in input data!"
                )
            xpos = np.asarray(adata.obsm[use_spatial_pos]).astype(default_dtype)
            # print("xpos的shape是",xpos.shape)
            return xpos
        return np.empty((adata.shape[0], 0), dtype=default_dtype)

    def _extract_xsf(self, adata, data_config):
        default_dtype = self.default_dtype
        use_size_factors = data_config["use_size_factors"]
        if use_size_factors:
            if use_size_factors not in adata.obs:
                raise ValueError(
                    f"Configured data batch '{use_size_factors}' "
                    f"cannot be found in input data!"
                )
            return adata.obs[use_size_factors].to_numpy().astype(default_dtype)
        return np.ones(adata.shape[0], dtype=int)



    def __len__(self):
        return max(len(self.rna_adata), len(self.atac_adata))
    
    def __getitem__(self, idx):
        rna_idx = idx % len(self.rna_counts)
        atac_idx = idx % len(self.atac_counts)

        rna_data = {
            'counts': self.rna_counts[rna_idx],
            'xraw': self.rna_xraw[rna_idx],
            'xpos': self.rna_xpos[rna_idx],
            'xsf': self.rna_xsf[rna_idx]
        }

        atac_data = {
            'counts': self.atac_counts[atac_idx],
            'xraw': self.atac_xraw[atac_idx],
            'xpos': self.atac_xpos[atac_idx],
        }
        
        return {
            'rna': rna_data,
            'atac': atac_data,
            'is_anchor_pair': False
        }


class SimpleBalancedLoader:
    """Balanced DataLoader"""
    
    def __init__(self, paired_dataset, mixed_dataset, batch_size=64, paired_ratio=0.3, shuffle=True):
        self.batch_size = batch_size
        self.paired_ratio = paired_ratio
        
        self.paired_batch_size = max(1, int(batch_size * paired_ratio))
        self.mixed_batch_size = batch_size - self.paired_batch_size
        
        self.paired_loader = DataLoader(
            paired_dataset, 
            batch_size=self.paired_batch_size, num_workers=8,
            shuffle=shuffle,pin_memory=True, persistent_workers=True
        )
        self.mixed_loader = DataLoader(
            mixed_dataset, 
            batch_size=self.mixed_batch_size, num_workers=8,
            shuffle=shuffle,pin_memory=True, persistent_workers=True
        )

        self.epoch_length = max(len(self.paired_loader), len(self.mixed_loader))
        self.paired_iter = None
        self.mixed_iter = None
    
    def __iter__(self):
        paired_iter = cycle(iter(self.paired_loader))
        mixed_iter = cycle(iter(self.mixed_loader))

        for _ in range(self.epoch_length):
            try:
                # Get paired batch
                paired_batch = next(paired_iter)
                # Get mixed batch
                mixed_batch = next(mixed_iter)            
                # Merge batch
                merged_batch = self._merge_batches(paired_batch, mixed_batch)
                yield merged_batch
                
            except StopIteration:
                break

    
    def _merge_batches(self, paired_batch, mixed_batch):
        rna_data = {}
        atac_data = {}
        for key in paired_batch['rna'].keys():
            rna_data[key] = torch.cat([paired_batch['rna'][key], mixed_batch['rna'][key]])
        for key in paired_batch['atac'].keys():
            atac_data[key] = torch.cat([paired_batch['atac'][key], mixed_batch['atac'][key]])
        merged = {
            'rna': rna_data,
            'atac': atac_data,
            'is_anchor': torch.cat([paired_batch['is_anchor_pair'], mixed_batch['is_anchor_pair']])
        }
        return merged
    
    def __len__(self):
        return self.epoch_length
