--
sort: 1
---
{% include list.liquid %}
# Graph construction
## GSG.read_10X_Visium_with_label

This function reads the files in the specified folder and returns anndata object for subsequent operations.

## GSG.Graph_10X

Graph_10X is a function to construct the neighbor structure of the graph using spatial information.
The input parameters of this function are the anndata object and the args object, the anndata object is the 10X spatial transcriptome data, and the args object holds various training parameters. The anndata object holds the gene expression information of the spatial transcriptome data as well as the spatial location of each captured point, and by using this method we use the spatial information of the points to construct the neighbor-joining map used in the subsequent model training and save the constructed graph in the dgl object. In addition, this function performs preprocessing operations on the spatial transcriptome data, including removing genes with low expression, normalizing the gene expression, and downscaling the data, and saves the gene expression after preprocessing in the dgl object as well.The return values of the function are the preprocessed anndata object and the dgl object.


### Parameters
```
adata: anndata
       10X spatial transcriptome data.

args: argparse
      The arguments of constructing graph. 
```
### Returns
```
adata: anndata
       10X spatial transcriptome data.

graph: dgl object
       The neighbor structure of the graph with spatial information.
```

### Examples
```
adata, graph = GSG.Graph_10X(adata, args)
```
