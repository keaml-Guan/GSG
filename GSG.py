from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, 
                             silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score,fowlkes_mallows_score)
from tqdm import tqdm
from models import build_model
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import dgl
import torch
import os
import warnings
import matplotlib.pyplot as plt
from tools.utils import (
    create_optimizer,
    set_random_seed,
    pretrain,
    KMeans_use,
    drawPicture,
    mkdir,
)
warnings.filterwarnings("ignore")
from scipy.spatial.distance import pdist, squareform
from scipy import sparse
import argparse

def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=0.2,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=0.1,
                        help="attention dropout")
    parser.add_argument("--weight_decay", type=float, default=2e-4,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--max_epoch_f", type=int, default=300)
    parser.add_argument("--lr_f", type=float, default=0.01, help="learning rate for evaluation")
    parser.add_argument("--weight_decay_f", type=float, default=1e-4, help="weight decay for evaluation")
    parser.add_argument("--linear_prob", action="store_true", default=True)
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--scheduler", action="store_true", default=True)
    parser.add_argument("--concat_hidden", action="store_true", default=False)
    # for graph classification
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--deg4feat", action="store_true", default=False, help="use node degree as input feature")
    parser.add_argument("--batch_size", type=int, default=32)
    # adjustable parameters
    parser.add_argument("--mask_rate", type=float, default=0.8)
    parser.add_argument("--encoder", type=str, default="gin")
    parser.add_argument("--decoder", type=str, default="gin")
    parser.add_argument("--num_hidden", type=int, default=128,
                        help="number of hidden units")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="number of hidden layers")
    parser.add_argument("--activation", type=str, default="elu")
    parser.add_argument("--max_epoch", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001 ,
                        help="learning rate")
    parser.add_argument("--alpha_l", type=float, default=4, help="`pow`inddex for `sce` loss")
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--replace_rate", type=float, default=0.05)
    parser.add_argument("--norm", type=str, default="batchnorm")
    # GSG parameter
    parser.add_argument("--feature_dim_method", type=str, default="PCA")
    parser.add_argument("--num_features", type=int, default=600)
    parser.add_argument("--threshold_radius", type=int, default=25)
    parser.add_argument("--folder_name", type=str, default="data/10X/")
    parser.add_argument("--sample_name", type=str, default="151673")
    parser.add_argument("--cluster_label", type=str, default= "")
    parser.add_argument("--num_classes", type=int, default=7,help = "The number of clusters")
    # read parameters
    args = parser.parse_args(args=[])
    return args

def GSG_Spatial_Pic(adata,args, result_file_path):
    if(args.cluster_label != ""):
        adata.obs[args.cluster_label] = adata.obs[args.cluster_label].astype('category').cat.add_categories(['None'])
        adata.obs[args.cluster_label] = adata.obs[args.cluster_label].fillna("None")
        drawPicture(adata.obs,col_name ="imagecol",row_name = "imagerow",colorattribute=args.cluster_label,save_file = result_file_path + "/Ground_true.pdf",is_show=False,is_save= True)
        drawPicture(adata.obs,col_name ="imagecol",row_name = "imagerow",colorattribute="GSG_Kmeans_cluster_str",save_file = result_file_path + "/GSG_Cluster_Spatial.pdf",is_show=False,is_save= True)
        k_means_score_ari = adjusted_rand_score(adata.obs["GSG_Kmeans_cluster_str"].values, adata.obs[args.cluster_label].values)
        k_means_score_silhouette = silhouette_score(adata.obsm["GSG_embedding"], adata.obs["GSG_Kmeans_cluster_str"].values, metric="sqeuclidean")
        k_means_score_nmi = normalized_mutual_info_score(adata.obs["GSG_Kmeans_cluster_str"].values, adata.obs[args.cluster_label].values)
        k_means_score_fmi = fowlkes_mallows_score(adata.obs["GSG_Kmeans_cluster_str"].values, adata.obs[args.cluster_label].values)
        k_means_score_DB = davies_bouldin_score(adata.obsm["GSG_embedding"], adata.obs["GSG_Kmeans_cluster_str"].values)
        k_means_score_chs = calinski_harabasz_score(adata.obsm["GSG_embedding"], adata.obs["GSG_Kmeans_cluster_str"].values)
        print("k_means_score_ari:" + str(k_means_score_ari))
        print("k_means_score_silhouette:" + str(k_means_score_silhouette))
        print("k_means_score_nmi:" + str(k_means_score_nmi))
        print("k_means_score_fmi:" + str(k_means_score_fmi))
        print("k_means_score_DB:" + str(k_means_score_DB))
        print("k_means_score_chs:" + str(k_means_score_chs))
    else:
        drawPicture(adata.obs,col_name ="imagecol",row_name = "imagerow",colorattribute="GSG_Kmeans_cluster_str",save_file = result_file_path + "/GSG_Cluster_Spatial.pdf",is_show=False,is_save= True)
        k_means_score_silhouette = silhouette_score(adata.obsm["GSG_embedding"], adata.obs["GSG_Kmeans_cluster_str"].values, metric="sqeuclidean")
        k_means_score_DB = davies_bouldin_score(adata.obsm["GSG_embedding"], adata.obs["GSG_Kmeans_cluster_str"].values)
        k_means_score_chs = calinski_harabasz_score(adata.obsm["GSG_embedding"], adata.obs["GSG_Kmeans_cluster_str"].values)
        print("k_means_score_silhouette:" + str(k_means_score_silhouette))
        print("k_means_score_DB:" + str(k_means_score_DB))
        print("k_means_score_chs:" + str(k_means_score_chs))
    return adata

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

def Graph_Construction(adata,args):
    cell_loc = adata.obs[["imagerow","imagecol"]].values
    distance_np = pdist(cell_loc, metric = "euclidean")
    distance_np_X =squareform(distance_np)
    threshold = args.threshold_radius
    num_big = np.where((0< distance_np_X)&(distance_np_X < threshold))[0].shape[0]
    adj_matrix = np.zeros(distance_np_X.shape)
    non_zero_point = np.where((0< distance_np_X)&(distance_np_X<threshold))
    for i in range(num_big):
        x = non_zero_point[0][i]
        y = non_zero_point[1][i]
        adj_matrix[x][y] = 1 
    adj_matrix = adj_matrix + np.eye(distance_np_X.shape[0])
    adj_matrix  = np.float32(adj_matrix)
    adj_matrix_crs = sparse.csr_matrix(adj_matrix)
    graph = dgl.from_scipy(adj_matrix_crs,eweight_name='w')
    adata.var_names=[i.upper() for i in list(adata.var_names)]
    adata.var["genename"]=adata.var.index.astype("str")
    adata.var_names_make_unique
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

def Graph_get(adata,threshold = 30,feature_dim_method = "PCA",num_features = 600,row_name = "imagerow",col_name = "imagecol"):
    cell_loc = adata.obs[[row_name,col_name]].values
    distance_np = pdist(cell_loc, metric = "euclidean")
    distance_np_X =squareform(distance_np)
    num_big = np.where((0< distance_np_X)&(distance_np_X < threshold))[0].shape[0]
    adj_matrix = np.zeros(distance_np_X.shape)
    non_zero_point = np.where((0 < distance_np_X) & (distance_np_X < threshold))
    adj_matrix = np.zeros(distance_np_X.shape)
    non_zero_point = np.where((0< distance_np_X)&(distance_np_X<threshold))
    for i in range(num_big):
        x = non_zero_point[0][i]
        y = non_zero_point[1][i]
        adj_matrix[x][y] = 1 
    adj_matrix = adj_matrix + np.eye(distance_np_X.shape[0])
    adj_matrix  = np.float32(adj_matrix)
    adj_matrix_crs = sparse.csr_matrix(adj_matrix)
    graph = dgl.from_scipy(adj_matrix_crs,eweight_name='w')
    if(feature_dim_method == "PCA"):
        sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
        sc.pp.scale(adata)
        sc.pp.pca(adata, n_comps=num_features)
        adata_X = adata.obsm["X_pca"]
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=num_features)
        adata_Vars =  adata[:, adata.var['highly_variable']]
        adata_X = adata_Vars.X.todense()
    graph.ndata["feat"] = torch.tensor(adata_X.copy())
    print(graph)
    return graph

def GSG_train(adata,graph,args):
    device = args.device if args.device >= 0 else "cpu"
    max_epoch = args.max_epoch
    optim_type = args.optimizer 
    lr = args.lr
    weight_decay = args.weight_decay
    load_model = args.load_model
    args.num_features = args.num_features
    seed = 0
    set_random_seed(seed)
    logger = None
    model = build_model(args)
    model.to(device)
    optimizer = create_optimizer(optim_type, model, lr, weight_decay)
    scheduler = None
    x = graph.ndata["feat"]
    if not load_model:
        model = pretrain(model, graph, x, optimizer, max_epoch, device, scheduler, logger)
    model.train(False)
    x = graph.ndata["feat"]
    embedding = model.embed(graph.to(device), x.to(device))
    adata.obsm["GSG_embedding"] = embedding.cpu().detach().numpy()
    if args.feature_dim_method == "HVG":
        latten_embedding = model.encoder_to_decoder(embedding)
        imputation_embedding =  model.decoder(graph.to(device),latten_embedding)
        adata.obsm["GSG_imputation_embedding"] = imputation_embedding.cpu().detach().numpy()
    return adata,model

def Data_read_and_comb(args,min_cells = 5):
    folder_name = args.folder_name
    before_sample_name = args.before_sample_name
    after_sample_name = args.after_sample_name
    before_folder_name = folder_name + before_sample_name
    adata_before = read_10X_Visium(before_folder_name)
    adata_before.var_names=[i.upper() for i in list(adata_before.var_names)]
    adata_before.var["genename"]=adata_before.var.index.astype("str")
    adata_before.var_names_make_unique()
    sc.pp.filter_genes(adata_before, min_cells)
    after_folder_name = folder_name + after_sample_name
    adata_after = read_10X_Visium(after_folder_name)
    adata_after.var_names=[i.upper() for i in list(adata_after.var_names)]
    adata_after.var["genename"]=adata_after.var.index.astype("str")
    adata_after.var_names_make_unique()
    sc.pp.filter_genes(adata_after, min_cells)
    adata_before.obs_names_make_unique()
    adata_after.obs_names_make_unique()
    adata_before.var_names_make_unique()
    adata_after.var_names_make_unique()
    adata_before.obs["sample_number"] = before_sample_name
    adata_after.obs["sample_number"] = after_sample_name
    merge_gene_list = list(set(adata_before.var.index).intersection(set(adata_after.var.index)))
    adata_before.var["comb_gene"] = True
    for i in adata_before.var.index:
        if i not in merge_gene_list:
            adata_before.var.loc[i,"comb_gene"] = False
    adata_after.var["comb_gene"] = True
    for i in adata_after.var.index:
        if i not in merge_gene_list:
            adata_after.var.loc[i,"comb_gene"] = False
    adata_before.obs.index=['sub_before-'+x for x in adata_before.obs.index]
    adata_before.obs.index
    adata_after.obs.index=['sub_after-'+x for x in adata_after.obs.index]
    adata_after.obs.index
    adata_subset_before = adata_before[:,adata_before.var['comb_gene']]
    adata_subset_after = adata_after[:,adata_after.var['comb_gene']]
    adatas=[adata_subset_before,adata_subset_after]
    adatas = ad.concat(adatas,merge= "same")
    return adatas

def Graph_Get_And_Data_Charge(adatas,args):
    feature_dim_method = args.feature_dim_method
    num_features = args.num_features
    before_sample_name = args.before_sample_name
    after_sample_name = args.after_sample_name
    adata_before = adatas[adatas.obs.sample_number == before_sample_name,]
    adata_after = adatas[adatas.obs.sample_number ==  after_sample_name,]
    graph_before = Graph_get(adata=adata_before,feature_dim_method = feature_dim_method , num_features= num_features, threshold =30)
    num_features = graph_before.ndata["feat"].shape[1]
    graph_after = Graph_get(adata=adata_after,feature_dim_method = feature_dim_method , num_features= num_features, threshold = 30)
    num_features = graph_after.ndata["feat"].shape[1]
    sc.pp.neighbors(adatas, n_neighbors=10, n_pcs=100,use_rep='X')
    sc.pp.normalize_total(adatas, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    sc.pp.scale(adatas)
    sc.pp.pca(adatas, n_comps=num_features)
    two_graph = [graph_before,graph_after]
    two_adata = [adata_before,adata_after]
    return two_graph,two_adata

def Origin_PCA_show(adatas,result_file,args):
    before_sample_name = args.before_sample_name
    after_sample_name = args.after_sample_name
    sc.pp.neighbors(adatas, n_neighbors=10, n_pcs=100,use_rep='X')
    fig, ax = plt.subplots(2,1,figsize=(10, 20))
    sc.tl.umap(adatas)
    sc.pl.umap(adatas, color="sample_number",title = "comb_umap_Origin",show = False ,ax=ax[0])
    sc.pp.neighbors(adatas, n_neighbors=10, n_pcs=600)
    sc.tl.umap(adatas)
    sc.pl.umap(adatas, color="sample_number",title = "comb_umap_PCA",show = False ,ax=ax[1])
    fig.savefig(result_file + "/Origin_and_PCA_umap.pdf",bbox_inches = 'tight')

def GSG_batch_effect_train(args,two_graph):
    device = args.device if args.device >= 0 else "cpu"
    max_epoch = args.max_epoch
    optim_type = args.optimizer 
    lr = args.lr
    weight_decay = args.weight_decay
    seed = 0
    set_random_seed(seed)
    model = build_model(args)
    graph_before = two_graph[0]
    graph_after = two_graph[1]
    model.to(device)
    x_after = graph_after.ndata["feat"]
    x_before = graph_before.ndata["feat"]
    graph_before = graph_before.to(device)
    x_before = x_before.to(device)
    graph_after = graph_after.to(device)
    x_after = x_after.to(device)
    optimizer = create_optimizer(optim_type, model, lr, weight_decay)
    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()
        loss, loss_dict = model(graph_before, x_before)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.train()
        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
    return model

def GSG_Embedding_Get(model,two_graph,two_adata,args):
    device = args.device if args.device >= 0 else "cpu"
    graph_before = two_graph[0]
    graph_after = two_graph[1]
    adata_before = two_adata[0]
    adata_after = two_adata[1]
    x = graph_after.ndata["feat"]
    embedding = model.embed(graph_after.to(device), x.to(device))
    adata_after.obsm["GSG_embedding"] = embedding.cpu().detach().numpy()
    if args.feature_dim_method == "HVG":
        latten_embedding = model.encoder_to_decoder(embedding)
        imputation_embedding =  model.decoder(graph_after.to(device),latten_embedding)
        adata_after.obsm["GSG_imputation_embedding"] = imputation_embedding.cpu().detach().numpy()
    x = graph_before.ndata["feat"]
    embedding = model.embed(graph_before.to(device), x.to(device))
    adata_before.obsm["GSG_embedding"] = embedding.cpu().detach().numpy()
    if args.feature_dim_method == "HVG":
        latten_embedding = model.encoder_to_decoder(embedding)
        imputation_embedding =  model.decoder(graph_before.to(device),latten_embedding)
        adata_before.obsm["GSG_imputation_embedding"] = imputation_embedding.cpu().detach().numpy()

def GSG_Cluster_Comb(two_adata,args,result_file):
    color_list = {
        "0" : "#E41A1C",
        "1": "#377EB8",
        "2":"#4DAF4A",
        "3":"#984EA3",
        "4":"#FF7F00",
        "5":"#FFFF33",
        "6":"#A65628",
        "7" :"#F781BF",
        "8":"#999999",
        "9":"#E41A80",
        "10":"#377F1C",
        "11":"#4DAFAE",
        "12":"#984F07",
        "13":"#FF7F64",
        "14":"#FFFF97",
        "15":"#A6568C", 
        "16":"#F78223", 
        "17":"#9999FD", 
        "18" :"#E41AE4", 
        "19" : "#377F80", 
        "20":"#4DB012",
        "21":"#984F6B",
        "22":"#FF7FC8",  
        "23":"#A656F0", 
        "24":"#F78287", 
        "25":"#999A61", 
        "26":"#E41B48", 
        "27":"#377FE4",
        "28":"#4DB076", 
        "29":"#984FCF", 
        "30":"#FF802C",
    }
    flatui = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999", "#E41A80",
    "#377F1C", "#4DAFAE", "#984F07", "#FF7F64", "#FFFF97", "#A6568C", "#F78223", "#9999FD", "#E41AE4", "#377F80", "#4DB012",
    "#984F6B", "#FF7FC8",  "#A656F0", "#F78287", "#999A61", "#E41B48", "#377FE4", "#4DB076", "#984FCF", "#FF802C"]
    import seaborn as sns
    adata_before = two_adata[0]
    adata_after = two_adata[1]
    adatas_GSG =[adata_before,adata_after]
    adatas_GSG = ad.concat(adatas_GSG,merge = "same")
    K_means_value = KMeans_use(adatas_GSG.obsm['GSG_embedding'],args.num_classes)
    adatas_GSG.obs["Comb_K_means_Number"] = K_means_value
    sc.pp.neighbors(adatas_GSG, n_neighbors=10,use_rep='GSG_embedding')
    sc.tl.umap(adatas_GSG)
    fig_new = sc.pl.umap(adatas_GSG, color="sample_number",title = "GSG_Comb_UMAP",return_fig = True)
    fig_new.savefig(result_file + "/GSG_Comb_UMAP.pdf",bbox_inches = 'tight')
    adatas_GSG.obs["Comb_K_means_Number_str"] = adatas_GSG.obs["Comb_K_means_Number"].astype("str")
    fig_new = sc.pl.umap(adatas_GSG, color="Comb_K_means_Number_str",title = "GSG_Kmeans_Comb_Cluster_UMAP",return_fig = True,palette=sns.color_palette(flatui))
    fig_new.savefig(result_file + "/GSG_Kmeans_Comb_Cluster_UMAP.pdf",bbox_inches = 'tight')
    adata_before = adatas_GSG[adatas_GSG.obs.sample_number == args.before_sample_name,]
    adata_after = adatas_GSG[adatas_GSG.obs.sample_number == args.after_sample_name,]
    adata_before.obs["cluster_color"] = ""
    adata_after.obs["cluster_color"] = ""
    for i in adata_before.obs.index:
        adata_before.obs.loc[i,"cluster_color"] = color_list[adata_before.obs.loc[i,"Comb_K_means_Number_str"]]
    for i in adata_after.obs.index:
        adata_after.obs.loc[i,"cluster_color"] = color_list[adata_after.obs.loc[i,"Comb_K_means_Number_str"]]
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    col_name ="imagecol"
    row_name = "imagerow"
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Scatter(x=adata_before.obs[col_name], y=adata_before.obs[row_name],
                        mode='markers',
                        name='markers',
                        marker=dict(size=14,color = adata_before.obs["cluster_color"]),
                    ),row=1, col=1)
    fig.add_trace(go.Scatter(x=adata_after.obs[col_name], y= adata_after.obs[row_name],
                        mode='markers',
                        name='markers',
                        marker=dict(size=14,color = adata_after.obs["cluster_color"]),
                    ),row=1, col=2)
    fig.update_layout(height=1000, width=2000, title_text="batch_show")
    fig.show()
    fig.write_image(result_file + "/" + args.before_sample_name + "_" +args.after_sample_name+ "_spatial_comb_kmeans.pdf")
    return adatas_GSG

def GSG_Kmean_Separate_cluster(adatas_GSG,args):
    adata_before = adatas_GSG[adatas_GSG.obs.sample_number == args.before_sample_name,]
    adata_after = adatas_GSG[adatas_GSG.obs.sample_number == args.after_sample_name,]
    k_means_value = KMeans_use(adata_after.obsm['GSG_embedding'],args.num_classes)
    adata_after.obs["GSG_Kmeans_Cluster_Separate_number"] = k_means_value
    k_means_value = KMeans_use(adata_before.obsm['GSG_embedding'],args.num_classes)
    adata_before.obs["GSG_Kmeans_Cluster_Separate_number"] = k_means_value
    adatas_GSG =[adata_before,adata_after]
    adatas_GSG = ad.concat(adatas_GSG,merge = "same")
    return adatas_GSG

def GSG_KMean_cluster_Separate_Result_Show(adatas_GSG,args,result_file):
    flatui = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999", "#E41A80",
    "#377F1C", "#4DAFAE", "#984F07", "#FF7F64", "#FFFF97", "#A6568C", "#F78223", "#9999FD", "#E41AE4", "#377F80", "#4DB012",
    "#984F6B", "#FF7FC8",  "#A656F0", "#F78287", "#999A61", "#E41B48", "#377FE4", "#4DB076", "#984FCF", "#FF802C"]
    sc.pp.neighbors(adatas_GSG, n_neighbors=10,use_rep='GSG_embedding')
    sc.tl.umap(adatas_GSG)
    fig_new = sc.pl.umap(adatas_GSG, color="sample_number",title = "GSG_Comb_UMAP",return_fig = True)
    fig_new.savefig(result_file + "/GSG_UMAP.pdf",bbox_inches = 'tight')
    adatas_GSG.obs["GSG_Kmeans_Cluster_Separate_number_str"] = adatas_GSG.obs["GSG_Kmeans_Cluster_Separate_number"].astype("str")
    adata_before = adatas_GSG[adatas_GSG.obs.sample_number == args.before_sample_name,]
    adata_after = adatas_GSG[adatas_GSG.obs.sample_number == args.after_sample_name,]
    GSG_adata_before_cluster_file = result_file + "/" + args.before_sample_name + "_Separate_Cluster.pdf"
    drawPicture(adata_before.obs,col_name ="imagecol",row_name = "imagerow",colorattribute="GSG_Kmeans_Cluster_Separate_number_str",save_file = GSG_adata_before_cluster_file,is_show=False,is_save= True)
    GSG_adata_after_cluster_file = result_file + "/" + args.after_sample_name + "_Separate_Cluster.pdf"
    drawPicture(adata_after.obs,col_name ="imagecol",row_name = "imagerow",colorattribute="GSG_Kmeans_Cluster_Separate_number_str",save_file = GSG_adata_after_cluster_file,is_show=False,is_save= True)
    return adatas_GSG

def GSG_plot_batch_effect_imputation(adata,args,result_file):
    diff_gene_floder = result_file + "/differential_genes/"
    mkdir(diff_gene_floder)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.pca(adata, n_comps=600)
    adata.obs["Comb_k_means_number_str"] = adata.obs["Comb_K_means_Number_str"].astype("category")
    adata.var_names = adata.var.index
    sc.tl.rank_genes_groups(adata, "Comb_k_means_number_str" , use_raw=False,n_genes = 300,method="wilcoxon")
    sc.tl.filter_rank_genes_groups(adata)
    celltype_list = adata.uns["rank_genes_groups_filtered"]["names"].dtype.names
    for celltype in celltype_list:
        gene_list = []
        for gene in adata.uns["rank_genes_groups_filtered"]["names"][celltype]:
            if(type(gene) == str):
                gene_list.append(gene)
            if(len(gene_list) >= 3):
                break
        if(len(gene_list) != 0 ):
            sc.pl.violin(adata, gene_list, groupby='Comb_k_means_number_str',show=False)
            plt.savefig(diff_gene_floder + celltype+ "_gene_violin.pdf")
    adata.var['features'] = adata.var.index
    result_csv = pd.DataFrame()
    gene_num = 0
    celltype_list = adata.uns["rank_genes_groups_filtered"]["names"].dtype.names
    for celltype in celltype_list:
        gene_list = adata.uns["rank_genes_groups_filtered"]["names"][celltype]
        for gene in gene_list:
            if(type(gene) == str):
                result_csv.loc[gene_num,"gene_num"] = gene
                result_csv.loc[gene_num,"gene"] = adata.var.loc[gene,"features"]
                result_csv.loc[gene_num,"celltype"] = celltype
                result_csv.loc[gene_num,"type"] = "after"
                gene_num = gene_num + 1
                print(gene)
                print(celltype)
    result_csv.to_csv(diff_gene_floder + "all_differential_genes_result.csv")
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    col_name ="imagecol"
    row_name = "imagerow"    
    adata_before = adata[adata.obs.sample_number == args.before_sample_name,]
    before_cell_csv = adata_before.obs[[row_name,col_name]]
    before_celltype_list = adata_before.uns["rank_genes_groups"]["names"].dtype.names
    sc.pp.normalize_total(adata_before, target_sum=1e4)
    sc.pp.log1p(adata_before)
    sc.pp.highly_variable_genes(adata_before, flavor='seurat_v3', n_top_genes=args.num_features)
    before_adata_Vars =  adata_before[:, adata_before.var['highly_variable']]
    for celltype in before_celltype_list:
        gene_list = adata_before.uns["rank_genes_groups"]["names"][celltype][:5]
        for gene in gene_list:
            if(gene in before_adata_Vars.var.index):
                print(gene)
                print(celltype)
                celltype = celltype.replace("/","_")
                before_gene_csv = pd.DataFrame(index =adata_before.obs.index,columns= adata_before.var.index,data =adata_before.X.todense())
                after_gene_csv = pd.DataFrame(index =before_adata_Vars.obs.index,columns= before_adata_Vars.var.index,data =adata_before.obsm["GSG_imputation_embedding"] )
                before_gene_csv_merge = pd.merge(before_gene_csv,before_cell_csv,left_index=True,right_index=True)
                after_gene_csv_merge = pd.merge(after_gene_csv,before_cell_csv,left_index=True,right_index=True)
                width = 1000
                height = 2000
                marker_size = 10
                fig = make_subplots(rows=1, cols=2)
                fig.add_trace(go.Scatter(x=before_gene_csv_merge[col_name], y=before_gene_csv_merge[row_name],
                                    mode='markers',
                                    name='markers',
                                    marker=dict(size=10,color = before_gene_csv_merge[gene],colorscale = px.colors.sequential.Viridis),
                                ),row=1, col=1)
                fig.add_trace(go.Scatter(x=after_gene_csv_merge[col_name], y=after_gene_csv_merge[row_name],
                                    mode='markers',
                                    name='markers',
                                    marker=dict(size=10,color = after_gene_csv_merge[gene],colorscale = px.colors.sequential.Viridis),
                                ),row=1, col=2)
                fig.update_layout(height=1000, width=2000, title_text="before and after " + gene + " "+ celltype + " gene express ")
                fig.write_image(diff_gene_floder + "first_sample_before_and_after_" + gene + "_"+ celltype + "_gene express.pdf")
    adata_after = adata[adata.obs.sample_number == args.after_sample_name,]
    after_cell_csv = adata_after.obs[[row_name,col_name]]
    after_celltype_list = adata_after.uns["rank_genes_groups"]["names"].dtype.names
    sc.pp.normalize_total(adata_after, target_sum=1e4)
    sc.pp.log1p(adata_after)
    sc.pp.highly_variable_genes(adata_after, flavor='seurat_v3', n_top_genes=args.num_features)
    after_adata_Vars =  adata_after[:, adata_after.var['highly_variable']]
    for celltype in after_celltype_list:
        gene_list = adata_after.uns["rank_genes_groups"]["names"][celltype][:5]
        for gene in gene_list:
            if(gene in after_adata_Vars.var.index):
                print(gene)
                print(celltype)
                celltype = celltype.replace("/","_")
                before_gene_csv = pd.DataFrame(index =adata_after.obs.index,columns= adata_after.var.index,data =adata_after.X.todense())
                after_gene_csv = pd.DataFrame(index =after_adata_Vars.obs.index,columns= after_adata_Vars.var.index,data =adata_after.obsm["GSG_imputation_embedding"] )
                before_gene_csv_merge = pd.merge(before_gene_csv,after_cell_csv,left_index=True,right_index=True)
                after_gene_csv_merge = pd.merge(after_gene_csv,after_cell_csv,left_index=True,right_index=True)
                width = 1000
                height = 2000
                marker_size = 10
                fig = make_subplots(rows=1, cols=2)
                fig.add_trace(go.Scatter(x=before_gene_csv_merge[col_name], y=before_gene_csv_merge[row_name],
                                    mode='markers',
                                    name='markers',
                                    marker=dict(size=10,color = before_gene_csv_merge[gene],colorscale = px.colors.sequential.Viridis),
                                ),row=1, col=1)
                fig.add_trace(go.Scatter(x=after_gene_csv_merge[col_name], y=after_gene_csv_merge[row_name],
                                    mode='markers',
                                    name='markers',
                                    marker=dict(size=10,color = after_gene_csv_merge[gene],colorscale = px.colors.sequential.Viridis),
                                ),row=1, col=2)
                fig.update_layout(height=1000, width=2000, title_text="before and after " + gene + " "+ celltype + " gene express ")
                fig.write_image(diff_gene_floder + "second_sample_before_and_after_" + gene + "_"+ celltype + "_gene express.pdf")
    del adata.uns['rank_genes_groups_filtered']

def GSG_plot_imputation_gene(adata,args,result_file):
    diff_gene_floder = result_file + "/" + args.sample_name + "_differential_genes/"
    mkdir(diff_gene_floder)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.pca(adata, n_comps=600)
    adata.obs["GSG_kmeans_cluster_str"] =  adata.obs["GSG_Kmeans_cluster_str"].astype("category")
    adata.var_names = adata.var.index
    sc.tl.rank_genes_groups(adata, "GSG_kmeans_cluster_str" , use_raw=False,n_genes = 300,method="wilcoxon")
    sc.tl.filter_rank_genes_groups(adata)
    celltype_list = adata.uns["rank_genes_groups_filtered"]["names"].dtype.names
    for celltype in celltype_list:
        gene_list = []
        for gene in adata.uns["rank_genes_groups_filtered"]["names"][celltype]:
            if(type(gene) == str):
                gene_list.append(gene)
            if(len(gene_list) >= 3):
                break
        if(len(gene_list) != 0 ):
            sc.pl.violin(adata, gene_list, groupby='GSG_kmeans_cluster_str',show=False)
            plt.savefig(diff_gene_floder + celltype+ "_gene_violin.pdf")
    adata.var['features'] = adata.var.index
    result_csv = pd.DataFrame()
    gene_num = 0
    celltype_list = adata.uns["rank_genes_groups_filtered"]["names"].dtype.names
    for celltype in celltype_list:
        gene_list = adata.uns["rank_genes_groups_filtered"]["names"][celltype]
        for gene in gene_list:
            if(type(gene) == str):
                result_csv.loc[gene_num,"gene_num"] = gene
                result_csv.loc[gene_num,"gene"] = adata.var.loc[gene,"features"]
                result_csv.loc[gene_num,"celltype"] = celltype
                result_csv.loc[gene_num,"type"] = "after"
                gene_num = gene_num + 1
    result_csv.to_csv(diff_gene_floder + "all_differential_genes_result.csv")
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    col_name ="imagecol"
    row_name = "imagerow"
    cell_csv = adata.obs[[row_name,col_name]]
    celltype_list = adata.uns["rank_genes_groups"]["names"].dtype.names
    adata_Vars =  adata[:, adata.var['highly_variable']]
    for celltype in celltype_list:
        gene_list = adata.uns["rank_genes_groups"]["names"][celltype][:5]
        for gene in gene_list:
            if(gene in adata_Vars.var.index):
                print(gene)
                print(celltype)
                celltype = celltype.replace("/","_")
                before_gene_csv = pd.DataFrame(index =adata.obs.index,columns= adata.var.index,data =adata.X.todense())
                after_gene_csv = pd.DataFrame(index =adata_Vars.obs.index,columns= adata_Vars.var.index,data =adata.obsm["GSG_imputation_embedding"] )
                before_gene_csv_merge = pd.merge(before_gene_csv,cell_csv,left_index=True,right_index=True)
                after_gene_csv_merge = pd.merge(after_gene_csv,cell_csv,left_index=True,right_index=True)
                width = 1000
                height = 2000
                marker_size = 10
                fig = make_subplots(rows=1, cols=2)
                fig.add_trace(go.Scatter(x=before_gene_csv_merge[col_name], y=before_gene_csv_merge[row_name],
                                    mode='markers',
                                    name='markers',
                                    marker=dict(size=10,color = before_gene_csv_merge[gene],colorscale = px.colors.sequential.Viridis),
                                ),row=1, col=1)
                fig.add_trace(go.Scatter(x=after_gene_csv_merge[col_name], y=after_gene_csv_merge[row_name],
                                    mode='markers',
                                    name='markers',
                                    marker=dict(size=10,color = after_gene_csv_merge[gene],colorscale = px.colors.sequential.Viridis),
                                ),row=1, col=2)
                fig.update_layout(height=1000, width=2000, title_text="before and after " + gene + " "+ celltype + " gene express ")
                fig.write_image(diff_gene_floder + "before_and_after_" + gene + "_"+ celltype + "_gene express.pdf")
    del adata.uns['rank_genes_groups_filtered']

def GSG_plot_trajectory(adata, args, result_file, filename = "GSG_Cluster_Embedding_trajectory.pdf"):
    if(args.cluster_label != ""):
        used_adata = adata[adata.obs[args.cluster_label]!= "None",]
        sc.tl.paga(used_adata, groups=args.cluster_label)
    else:
        used_adata = adata
        sc.tl.paga(used_adata, groups="GSG_Kmeans_cluster_str")
    sc.pl.paga_compare(used_adata, legend_fontsize=15, node_size_scale= 4, frameon=False, size=50,max_edge_width = 10,right_margin=0.2,
                        legend_fontoutline=5,title="PAGA_SHOW", show=False)
    plt.savefig( result_file + "/" + filename,bbox_inches='tight',dpi =1000)


# def save_file(adata,result_file):
#     adata_X = adata.X
