## GCImpute: A Genomic-Cytology Collaborative Imputation Network for Single-cell RNA Sequencing Data
This repository contains the Python implementation for GCImpute. 
<!--
Further details about GCImpute can be found in our paper:
> GCImpute: A Genomic-Cytology Collaborative Imputation Network for Single-cell RNA Sequencing Data
-->

## Requirements

=================
* torch==2.1.0
* python==3.8
* scanpy

## Overview
<!--
![framework](framework.jpg)
-->

## Tutorial

=================
### Preprocess

Raw single-cell count matrices in formats such as CSV, h5ad, 10x mtx, or TXT were imported into AnnData objects. We verified that the inputs contained unnormalized counts, removed empty droplets and duplicated labels, and applied optional filters to exclude lowly expressed genes and low-quality cells. Preprocessing was performed using Scanpy, optionally including library-size normalization, log(1+count) transformation, and other standard steps. After preprocessing, we obtained a cell-by-gene expression matrix with cells in rows and genes in columns, which was used as the direct input to the model.

See the `sc_load_process()` function for more preprocess details.

### Hyperparameter

**1. Input/format parameters**

* `--sc_data`: Path to the input data file (CSV, h5ad, 10x mtx, or TXT).
* `--csv_to_load`, `--h5ad_to_load`, `--x10_to_load`, `--txt_to_load`: Flags specifying the file format.
* `-t/--transpose`: Transpose the input matrix if needed (default: False).

**2. Normalization and preprocessing options** (all disabled by default; users must activate explicitly)

* `--logtrans_input`: Apply log(1+count) transformation.
* `--normalize_amount`: Apply library-size normalization.
* `--normalize_expression_profile`: Apply scaling/standardization.

**3. Hardware and output settings**

* `--gpu`: Specify the GPU ID (default: 0).
* `--output_dir`: Directory where outputs are stored (default: `outputs/`). 

**4. Model training parameters**

* `--graph_AE_epoch`: Total training epochs (default: 500).
* `--graph_AE_patience`: Early stopping patience (default: 10 epochs).
* `--graph_AE_learning_rate`: Learning rate (default: 1e-3).
* `--graph_AE_factor`: Learning rate scaling factor when validation does not improve (default: 0.1).
* `--graph_AE_cell_batch_size`: Batch size for cells (default: 1024).
* `--graph_AE_gene_batch_size`: Batch size for genes (default: 1024).
* `--graph_AE_use_GAT`: If enabled, Graph Attention Networks are used for gene layers; otherwise Graph Convolutional Networks (default: False).

**5. Example usage**
For example, to run GCImpute with a CSV input file on GPU 1:

```bash
python main.py --sc_data="./data/X.csv" --csv_to_load -t --gpu=1
```
See the `parse_args()` function for more parameter details.

### outputs

Output folder contains the main output file, along with additional training logs.

- `imputed.csv` is the main output of the method which represents the imputed scRNA-seq data. It is formatted as a `cell x gene` matrix. 


