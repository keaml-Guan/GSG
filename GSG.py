from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, 
                             silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score,fowlkes_mallows_score)
from graphmae.models import build_model
from sklearn.cluster import KMeans
#import calculate_adj
import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse
import dgl
import torch
import os
import warnings
import matplotlib.pyplot as plt
from graphmae.utils import ( 
    create_optimizer,
    set_random_seed,
    pretrain,    
)
warnings.filterwarnings("ignore")
from scipy.spatial.distance import pdist, squareform
from scipy import sparse

def KMeans_use(embedding,cluster_number):
    kmeans = KMeans(n_clusters=cluster_number,
                init="k-means++",
                random_state=0)
    pred = kmeans.fit_predict(embedding)
    return pred

def drawPicture(dataframe,col_name, row_name,colorattribute,save_file,celltype_colors =  ("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999", "#E41A80",
    "#377F1C", "#4DAFAE", "#984F07", "#FF7F64", "#FFFF97", "#A6568C", "#F78223", "#9999FD", "#E41AE4", "#377F80", "#4DB012",
    "#984F6B", "#FF7FC8",  "#A656F0", "#F78287", "#999A61", "#E41B48", "#377FE4", "#4DB076", "#984FCF", "#FF802C",
    "#00005F", "#A65754", "#F782EB", "#999AC5", "#E41BAC", "#378048", "#4DB0DA", "#985033", "#FF8090", "#0000C3", "#A657B8",
    "#F7834F", "#999B29", "#E41C10", "#3780AC", "#4DB13E", "#985097", "#FF80F4", "#000127", "#A6581C", "#F783B3", "#999B8D",
    "#E41C74", "#378110", "#4DB1A2", "#9850FB", "#FF8158", "#00018B", "#A65880", "#F78417", "#999BF1", "#E41CD8", "#378174",
    "#4DB206", "#98515F", "#FF81BC", "#0001EF", "#A658E4", "#F7847B", "#999C55", "#E41D3C", "#3781D8", "#4DB26A", "#9851C3",
    "#FF8220", "#000253", "#A65948", "#F784DF", "#999CB9", "#E41DA0", "#37823C", "#4DB2CE", "#985227", "#FF8284", "#0002B7",
    "#A659AC", "#F78543", "#999D1D", "#E41E04", "#3782A0", "#4DB332", "#98528B", "#FF82E8", "#00031B", "#A65A10", "#F785A7",
    "#999D81", "#E41E68", "#378304", "#4DB396", "#9852EF", "#FF834C", "#00037F", "#A65A74", "#F7860B", "#999DE5", "#E41ECC",
    "#378368", "#4DB3FA", "#985353", "#FF83B0", "#0003E3", "#A65AD8", "#F7866F", "#999E49", "#E41F30", "#3783CC", "#4DB45E",
    "#9853B7", "#FF8414", "#000447", "#A65B3C", "#F786D3", "#999EAD", "#E41F94", "#378430", "#4DB4C2", "#98541B", "#FF8478",
    "#0004AB", "#A65BA0", "#F78737", "#999F11", "#E41FF8", "#378494", "#4DB526", "#98547F", "#FF84DC", "#00050F", "#A65C04",
    "#F7879B", "#999F75"
    ),width = 1000,height = 1000,marker_size = 10,is_show = True,is_save = False,save_type = "pdf"):
    import plotly.express as px
    length_col = max(dataframe[col_name]) - min(dataframe[col_name])
    length_row = max(dataframe[row_name]) - min(dataframe[row_name])
    max_length = max(length_col,length_row) + 2
    fig = px.scatter(dataframe, x = col_name, y= row_name,color = colorattribute,color_discrete_sequence=celltype_colors)
    fig.update_traces(marker_size=marker_size)
    fig.update_layout(
        xaxis = dict(
            tickmode = 'linear',
            tick0 = min(dataframe[row_name]),   # 起始点
            dtick = max_length  # 间距
        ),
        yaxis = dict(
            tickmode = 'linear',
            tick0 = min(dataframe[col_name]),   # 起始点
            dtick = max_length  # 间距
        ),
    )
    fig.update_layout(
        autosize=False,
        width=width,
        height = height)
    if(is_show):
        fig.show()
    if(is_save):
        if(save_type == "pdf"):
            fig.write_image(save_file)
        if(save_type == "html"):
            fig.write_html(save_file)

def GSG_Spatial_Pic(adata,args,result_file):
    if(args.cluster_label != ""):
        adata.obs[args.cluster_label] = adata.obs[args.cluster_label].cat.add_categories(['None'])
        adata.obs[args.cluster_label] = adata.obs[args.cluster_label].fillna("None")
        drawPicture(adata.obs,col_name ="imagecol",row_name = "imagerow",colorattribute=args.cluster_label,save_file = result_file + "/Ground_true.pdf",is_show=False,is_save= True)
        drawPicture(adata.obs,col_name ="imagecol",row_name = "imagerow",colorattribute="GSG_Kmeans_cluster_str",save_file = result_file + "/GSG_Cluster_Spatial.pdf",is_show=False,is_save= True)
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
        drawPicture(adata.obs,col_name ="imagecol",row_name = "imagerow",colorattribute="GSG_Kmeans_cluster_str",save_file = result_file + "/GSG_Cluster_Spatial.pdf",is_show=False,is_save= True)
        k_means_score_silhouette = silhouette_score(adata.obsm["GSG_embedding"], adata.obs["GSG_Kmeans_cluster_str"].values, metric="sqeuclidean")
        k_means_score_DB = davies_bouldin_score(adata.obsm["GSG_embedding"], adata.obs["GSG_Kmeans_cluster_str"].values)
        k_means_score_chs = calinski_harabasz_score(adata.obsm["GSG_embedding"], adata.obs["GSG_Kmeans_cluster_str"].values)
        print("k_means_score_silhouette:" + str(k_means_score_silhouette))
        print("k_means_score_DB:" + str(k_means_score_DB))
        print("k_means_score_chs:" + str(k_means_score_chs))
    return adata

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   
		os.makedirs(path)            
		print("---  new folder...  ---")
		print("---  OK  ---")
	else:
		print("---  There is this folder!  ---")

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

def Graph_10X(adata,args):
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
    # train model
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
        gene_list = adata.uns["rank_genes_groups"]["names"][celltype].head(5)
        for gene in gene_list:
            if(gene in adata_Vars.var.index):
                print(gene)
                print(celltype)
                celltype = celltype.replace("/","_")
                before_gene_csv = pd.DataFrame(index =adata.obs.index,columns= adata.var.index,data =adata.X.todense())
                after_gene_csv = pd.DataFrame(index =adata_Vars.obs.index,columns= adata_Vars.var.index,data =adata.obsm["GSG_imputation_embedding"] )
                before_gene_csv_merge = pd.merge(before_gene_csv,cell_csv,left_index=True,right_index=True)
                after_gene_csv_merge = pd.merge(after_gene_csv,cell_csv,left_index=True,right_index=True)
                width = 2000
                height = 1000
                marker_size = 10
                fig = make_subplots(rows=1, cols=2)
                fig.add_trace(go.Scatter(x=before_gene_csv_merge[col_name], y=before_gene_csv_merge[row_name],
                                    mode='markers',
                                    name='markers',
                                    marker=dict(size=marker_size,color = before_gene_csv_merge[gene],colorscale = px.colors.sequential.Viridis),
                                ),row=1, col=1)
                fig.add_trace(go.Scatter(x=after_gene_csv_merge[col_name], y=after_gene_csv_merge[row_name],
                                    mode='markers',
                                    name='markers',
                                    marker=dict(size=marker_size,color = after_gene_csv_merge[gene],colorscale = px.colors.sequential.Viridis),
                                ),row=1, col=2)
                fig.update_layout(height=height, width=width, title_text="before and after " + gene + " "+ celltype + " gene express ")
                fig.write_image(diff_gene_floder + "before_and_after_" + gene + "_"+ celltype + "_gene express.pdf")
