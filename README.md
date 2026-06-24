# A masked generative graph representation learning framework empowering precise spatial domain identification
![GitHub Repo stars](https://img.shields.io/github/stars/keaml-Guan/GSG) &nbsp;&nbsp; ![GitHub watchers](https://img.shields.io/github/watchers/keaml-Guan/GSG) &nbsp;&nbsp; ![GitHub](https://img.shields.io/github/license/keaml-Guan/GSG)
#
![](https://github.com/keaml-Guan/GSG/blob/main/figures/Fig1_11_reduce.jpg)
<br>
## ✨ Overview

Recent advances in spatial transcriptomics (ST) have opened new avenues for preserving spatial information while measuring gene expression. Yet, the challenge of seamlessly integrating this data into accurate and transferable representation remains. Here, we introduce a generative self-supervised graph (GSG) learning framework to achieve an effective joint embedding of location and gene expression within ST data. Our approach surpasses existing methods in identifying spatial domains within the human dorsolateral prefrontal cortex. Moreover, it can offer reliable analyses across various techniques, including Stereo-seq, Slide-seq, and seqFISH, irrespective of spatial resolution. Furthermore, GSG addresses dropout defects, enhancing gene expression by smoothing spatial patterns, extracting critical features, reducing batch effects, and enabling the integration of disparate datasets. Additionally, we performed spatial transcriptomic analysis on fetal human hearts, and applied GSG to extract biological insights. These experiments highlight GSG's accuracy in identifying spatial domains, uncovering specific APCDD1 expression in fetal endocardium, and implicating its role in congenital heart disease. Our results showcase GSG's superiority and underscore its valuable contributions to advancing spatial-omics analysis.


## 🛠️ Installation

> [!NOTE]
> **!!! The recommended operating system is Ubuntu 18.04 LTS.** Some packages may not download correctly on Windows.
### Use python virutal environment with conda
```sh
conda create -n gsg python=3.7 -y
conda activate gsg
# Need install cudnn based on your CUDA version.Refer to https://developer.nvidia.com/cudnn-archive
# conda install cudnn[=version]
```
### Install GSG
```sh
pip install GSG
```


## 🚀 Quick Start
See our model document details from [Docs](https://keaml-guan.github.io/GSG/).

We provide the jupyter notebook for reproducing the quantitative and visualization results of the paper in [/docs/tutorials/](https://github.com/keaml-Guan/GSG/tree/main/docs/tutorials/).
 
The core workflow of GSG can be summarized in three main steps. You can get started with the following code:
    
```sh
adata = GSG.pp.read_10X_Visium_with_label(args.folder_name + args.sample_name)     # Read in data
adata, graph = GSG.pp.Graph_10X(adata, args)                                       # preprocess
adata, model = GSG.train.GSG_train(adata, graph, args)                             # graph representation learning
```

<!-- ## Issues on experiment
We found that SpaceFlow has different versions on GitHub and PyPi. The version installed in the recommended way is backward. In addition, the new version on GitHub has corrections to the old version, while the code on PyPi has fatal problems, which leads to serious problems in spatial domain identification. -->

## 📚 Citation
Wang C, Zhang T, Sun H, et al. A masked generative graph representation learning framework empowering precise spatial domain identification[J]. *Bioinformatics*, 2026, 42(6). [https://doi.org/10.1093/bioinformatics/btag333.](https://doi.org/10.1093/bioinformatics/btag333)

## 📩 Contact
If you have any questions, feel free to contact [chuyao25@mails.jlu.edu.cn](mailto:chuyao25@mails.jlu.edu.cn).
