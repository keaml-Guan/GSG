# GSG: A generative self-supervised graph learning framework for spatial transcriptomics
![GitHub Repo stars](https://img.shields.io/github/stars/keaml-Guan/GSG?style=social) ![GitHub forks](https://img.shields.io/github/forks/keaml-Guan/GSG?style=social) ![GitHub watchers](https://img.shields.io/github/watchers/keaml-Guan/GSG?style=social)
#
![](https://github.com/keaml-Guan/GSG/tree/main/figurees/GSG.jpg)
<br>
## Overview

GSG takes ST data as input and outputs,its spatially coherent graph representation learning. The ST data contain two components: gene expression and spatial location information. The gene expression of a spot is initially reduced by principal component analysis (PCA) or  initial extraction of highly variable genes (HVGs) as the initial spot features, which are used as nodes in the graph. Then, an adjacency of the graph is constructed based on the location  information. Specifically, the location information is used to calculate the relative distance matrix among the spots. According to the biological assumption that cells influence their neighbours according to the diffusion principle, we choose a certain distance threshold to generate a 0-1 adjacency matrix for the graph. To calculate the representation learning of the graph, we introduced a self-supervised masked graph autoencoder. GSG selects a random number of nodes and masks their initial node features using a mask token [MASK]. Then, a GNN encoder is used to obtain the corrupted graph embedding. The selected nodes are remasked with another token (DMASK) in the extracted embedding and passed through a decoder composed of GNNs to reproduce the initial features. The decoder output is used to reconstruct the node feature of the masked node, using the scaled cosine error as the loss function. By using MASK, GSG enhances the utilization of features of neighbouring nodes for better representation of ST data. With generative self-supervised graph learning, GSG learns to encode a spot/cell node embedding that contains gene expression information, which is further used to visualize the data with a UMAP plot and for other downstream analyses34.

## Requirements
You'll need to install the following packages in order to run the codes.
* python==3.7.12
* torch==1.8.0
* cudnn==8.4
* numpy==1.21.6
* scanpy==1.8.2
* anndata==0.8.0
* dgl==0.9.0
* pandas==1.2.4
* scipy==1.7.3
* scikit-learn==1.0.1 
* tqdm==4.64.1
* matplotlib==3.5.3
* tensorboardX==2.5.1

## Installation

### From source
Start by grabbing this source codes:
```sh
git clone https://github.com/keaml-Guan/GSG.git
cd GSG
```
### Use python virutal environment with conda
```sh
conda creat -n gsg python=3.7
conda activate gsg
pip install -r requirements.txt
```
## Quick Start
```sh
python GSG_cluster.py --device 0 --cluster_label layer_guess_reordered_short
```


<!--
## Citation
-->
