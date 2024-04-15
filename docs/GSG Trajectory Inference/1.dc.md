---
sort: 1
---

# Result output

The anndata object is the same as the data preprocessing and training model in the previous domain identification task. We use the anndata object after kmeans clustering for trajectory inference.

## GSG.GSG_plot_trajectory
This function is used to read anndata objects for paga trace inference analysis and draw a trace inference graph to save it in a specified file.


### Paramaters
```
adata: anndata
       Spatial transcriptome data.

args: argparse
      The arguments of constructing graph. 
      
result_file: str
      Path to restore the pdf file.
```

#### Examples

```
GSG.GSG_plot_trajectory(adata, args, result_file, filename = "GSG_Cluster_Embedding_trajectory.pdf")
```

<img src="../pics/paga.jpg">


