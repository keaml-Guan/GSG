import os
import warnings
import itertools
warnings.filterwarnings("ignore")

import dgl
import torch
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from scipy.spatial.distance import pdist, squareform

from . import utils


def read_10X_Visium(path,
                    genome=None,
                    count_file='filtered_feature_bc_matrix.h5', 
                    library_id=None, 
                    load_images=True, 
                    quality='hires',
                    image_path = None):
    adata = sc.read_visium(path, 
                        genome=genome,
                        count_file=count_file,
                        library_id=library_id,
                        load_images=load_images,)
    adata.var_names_make_unique()
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
    if quality == "fulres":
        image_coor = adata.obsm["spatial"]
        img = plt.imread(image_path, 0)
        adata.uns["spatial"][library_id]["images"]["fulres"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + quality + "_scalef"]
        image_coor = adata.obsm["spatial"] * scale
    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = quality
    return adata

def read_10X_Visium_with_label(path, 
                    genome=None,
                    count_file='filtered_feature_bc_matrix.h5', 
                    library_id=None, 
                    load_images=True, 
                    quality='hires',
                    image_path = None):
    adata = sc.read_visium(path, 
                        genome=genome,
                        count_file=count_file,
                        library_id=library_id,
                        load_images=load_images,)
    adata.var_names_make_unique()
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
    if quality == "fulres":
        image_coor = adata.obsm["spatial"]
        img = plt.imread(image_path, 0)
        adata.uns["spatial"][library_id]["images"]["fulres"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + quality + "_scalef"]
        image_coor = adata.obsm["spatial"] * scale
    if(os.path.exists(path + "/metadata.tsv")):
        adata.obs = pd.read_table(path + "/metadata.tsv",sep="\t",index_col=0)
    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = quality
    return adata


def read_stereo_seq(counts_data_path, position_path):
    counts_file = os.path.join(counts_data_path)
    coor_file = os.path.join(position_path)
    coor_df = pd.read_csv(coor_file, sep='\t')
    counts = pd.read_csv(counts_file, sep='\t', index_col=0)
    counts.columns = ['Spot_' + str(x) for x in counts.columns]
    coor_df.index = coor_df['label'].map(lambda x: 'Spot_' + str(x))
    adata = sc.AnnData(counts.T)
    adata.obs = coor_df
    adata.var_names_make_unique()
    coor_df = coor_df.loc[adata.obs_names, ['y', 'x']]
    adata.obsm["spatial"] = coor_df.to_numpy()
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata.obs['imagecol'] = coor_df.iloc[:, 1]
    adata.obs['imagerow'] = coor_df.iloc[:, 0]
    return adata

def read_slide_seq(path,
                      library_id=None,
                      scale=None,
                      quality="hires",
                      spot_diameter_fullres=50,
                      background_color="white",):
    count = pd.read_csv(os.path.join(path, "count_matrix.count"))
    meta = pd.read_csv(os.path.join(path, "spatial.idx"))
    adata = AnnData(count.iloc[:, 1:].set_index("gene").T)
    adata.var["ENSEMBL"] = count["ENSEMBL"].values
    adata.obs["index"] = meta["index"].values
    if scale == None:
        max_coor = np.max(meta[["x", "y"]].values)
        scale = 2000 / max_coor
    adata.obs["imagecol"] = meta["x"].values * scale
    adata.obs["imagerow"] = meta["y"].values * scale
    # Create image
    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)
    if background_color == "black":
        image = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)
    if library_id is None:
        library_id = "Slide-seq"
    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"][
        "tissue_" + quality + "_scalef"] = scale
    adata.uns["spatial"][library_id]["scalefactors"][
        "spot_diameter_fullres"
    ] = spot_diameter_fullres
    adata.obsm["spatial"] = meta[["x", "y"]].values
    return adata


def Graph_10X(adata, args):
    cell_loc = adata.obs[["imagerow", "imagecol"]].values
    if args.graph == 'radius':
        distance_np = pdist(cell_loc, metric = "euclidean")
        distance_np_X = squareform(distance_np)
        threshold = args.threshold_radius
        num_big = np.where((0< distance_np_X)&(distance_np_X < threshold))[0].shape[0]
        adj_matrix = np.zeros(distance_np_X.shape)
        non_zero_point = np.where((0< distance_np_X)&(distance_np_X<threshold))
        for i in tqdm(range(num_big)):
            x = non_zero_point[0][i]
            y = non_zero_point[1][i]
            adj_matrix[x][y] = 1 
        adj_matrix = adj_matrix + np.eye(distance_np_X.shape[0])
        adj_matrix  = np.float32(adj_matrix)
        adj_matrix_crs = sparse.csr_matrix(adj_matrix)
    elif args.graph == 'knn':
        tree = BallTree(cell_loc)
        distances, tail_list = tree.query(cell_loc, k=args.num_neighbors)
        head_list = []
        head_list = [head_list + [i] * len(tail_list[i]) for i in range(len(tail_list))]
        head_list = list(itertools.chain.from_iterable(head_list))
        tail_list = list(itertools.chain.from_iterable(tail_list))
        distances = np.ones_like(head_list)
        adj_matrix_crs = sparse.coo_matrix((distances, (head_list, tail_list)), shape=(cell_loc.shape[0], cell_loc.shape[0])).tocsr()
    graph = dgl.from_scipy(adj_matrix_crs, eweight_name='w')

    adata.var_names=[i.upper() for i in list(adata.var_names)]
    adata.var["genename"] = adata.var.index.astype("str")
    adata.var_names_make_unique()
    if(args.feature_dim_method == "PCA"):
        sc.pp.filter_genes(adata, min_cells=5)
        adata_X = sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
        adata_X = sc.pp.scale(adata_X)
        adata_X = sc.pp.pca(adata_X, n_comps=args.num_features)
    else:
        sc.pp.filter_genes(adata, min_cells=5)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=args.num_features)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata_Vars =  adata[:, adata.var['highly_variable']]
        adata_X = adata_Vars.X.todense()
    graph.ndata["feat"] = torch.tensor(adata_X.copy())
    return adata,graph

