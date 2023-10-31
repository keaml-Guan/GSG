# Generative Self-Supervised Graphs Enhance Spatial Transcriptomics Integration, Imputation and Domain Identification
![GitHub Repo stars](https://img.shields.io/github/stars/keaml-Guan/GSG) &nbsp;&nbsp; ![GitHub watchers](https://img.shields.io/github/watchers/keaml-Guan/GSG) &nbsp;&nbsp; ![GitHub](https://img.shields.io/github/license/keaml-Guan/GSG)
#
![](https://github.com/keaml-Guan/GSG/blob/main/figures/GSG_overview.jpg)
<br>
## Overview

Recent advances in spatial transcriptomics (ST) have opened new avenues for preserving spatial information while measuring gene expression. Yet, the challenge of seamlessly integrating this data into accurate and transferable representation remains. Here, we introduce a generative self-supervised graph (GSG) learning framework to achieve an effective joint embedding of location and gene expression within ST data. Our approach surpasses existing methods in identifying spatial domains within the human dorsolateral prefrontal cortex. Moreover, it can offer reliable analyses across various techniques, including Stereo-seq, Slide-seq, and seqFISH, irrespective of spatial resolution. Furthermore, GSG addresses dropout defects, enhancing gene expression by smoothing spatial patterns, extracting critical features, reducing batch effects, and enabling the integration of disparate datasets. Additionally, we performed spatial transcriptomic analysis on fetal human hearts, and applied GSG to extract biological insights. These experiments highlight GSG's accuracy in identifying spatial domains, uncovering specific APCDD1 expression in fetal endocardium, and implicating its role in congenital heart disease. Our results showcase GSG's superiority and underscore its valuable contributions to advancing spatial-omics analysis.


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

See our model document details from [Docs](https://keaml-guan.github.io/GSG/).

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
Before using, you need to unzip the data:
```sh
cd ./data/10X
cat 151673.zip* > 151673.zip
unzip -d ./ 151673.zip
cd ../..
```
And then, you can start using code following:
    
```sh
python GSG_cluster.py --device 0 --cluster_label layer_guess_reordered_short --feature_dim_method "PCA"
# feature_dim_method default is "PCA", and another is "HVG"
```


<!--
## Citation
-->

![](https://github.com/keaml-Guan/GSG/blob/main/figures/Result.jpg)
