library(Signac)
library(Seurat)
library(GenomeInfoDb)
library(GenomicRanges)
library(future)
library(reticulate)
library(anndata)
library(ggplot2)
library(Matrix)
library(EnsDb.Mmusculus.v79)
set.seed(1234)
library(rtracklayer)

rna <- read_h5ad('E15_5-S1_mRNA.h5ad')
atac <- read_h5ad('E15_5-S1_ATAC.h5ad')

rna_counts <- t(rna$X)  
rownames(rna_counts) <- rna$var_names  
colnames(rna_counts) <- rna$obs_names  
rna_obj <- CreateSeuratObject(counts = rna_counts, assay = "RNA")

fragpath <- 'E15_5-S1_filtered_fragments.tsv.bgz'
annotation <- GetGRangesFromEnsDb(ensdb = EnsDb.Mmusculus.v79)
seqlevelsStyle(annotation) <- "UCSC"
genome(annotation) <- "mm10"

atac_counts <- t(atac$X)  
rownames(atac_counts) <- atac$var_names  
colnames(atac_counts) <- atac$obs_names  
ATAC_assay <- CreateChromatinAssay(
  counts = atac_counts,
  sep = c("-", "-"),
  genome = "mm10",
  fragments = fragpath,
)
atac_obj <- CreateSeuratObject(
  counts = ATAC_assay,
  assay = 'ATAC',
  annotation = annotation
)

# Perform standard analysis of each modality independently 
rna_obj <- NormalizeData(rna_obj)
rna_obj <- FindVariableFeatures(rna_obj)
rna_obj <- ScaleData(rna_obj)
rna_obj <- RunPCA(rna_obj)

atac_obj <- RunTFIDF(atac_obj)
atac_obj <- FindTopFeatures(atac_obj, min.cutoff = "q0")
atac_obj <- RunSVD(atac_obj)

Annotation(atac_obj) <- annotation
gene.activities <- GeneActivity(atac_obj)

# add gene activities as a new assay
atac_obj[["ACTIVITY"]] <- CreateAssayObject(counts = gene.activities)

# normalize gene activities
DefaultAssay(atac_obj) <- "ACTIVITY"
atac_obj <- NormalizeData(atac_obj)
atac_obj <- ScaleData(atac_obj, features = rownames(atac_obj))

# Identify anchors
transfer.anchors <- FindTransferAnchors(reference = rna_obj, query = atac_obj, 
    reference.assay = "RNA", query.assay = "ACTIVITY", reduction = "cca")

anchors_matrix <- transfer.anchors@anchors

write.csv(anchors_matrix, "seurat_anchors_matrix.csv")
