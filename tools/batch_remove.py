from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score)
import logging
from tqdm import tqdm
from utils import (
    create_optimizer,
    set_random_seed,
    get_current_lr,
)
from graphmae.models import build_model
from sklearn.cluster import KMeans
import scanpy as sc
from operator import index
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np
from scipy import sparse
import dgl
import torch
import warnings
import argparse

warnings.filterwarnings("ignore")
class myDict(object):
    def __init__(self,mydict):
        self.mydict = mydict
        self.length = []
        self.keys = []
        for key,values in self.mydict.items():
            self.keys.append(key)
            self.length.append(len(values))
        self.nums = [1] * len(self.length)
        for i in range(len(self.length)):
            for j in range(i,len(self.length)):
                self.nums[i] *= self.length[j]
        self.para_dis = []
        print(self.length)
        print(self.nums)
                
    def getindex(self,index):
        result = []
        value = index
        for i in range(len(self.nums) - 1):
            result.append(value // self.nums[i+1])
            value = value - result[i] * self.nums[i+1]
        result.append(value) 
        result_dict = dict()
        for index,value in enumerate(result):
            result_dict[self.keys[index]] = self.mydict.get(self.keys[index])[value]
        return result_dict
    
    #para_dis = []
    def myiter(self):
        #para_dis = []
        for i in range(0,self.nums[0]):
            self.para_dis.append(self.getindex(i))
        return self.para_dis
def kMeans_use(embedding,cluster_number):
    kmeans = KMeans(n_clusters=cluster_number,
                init="k-means++",
                random_state=0)
    pred = kmeans.fit_predict(embedding)
    return pred
def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()

        loss, loss_dict = model(graph, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)
    return model
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
    ),width = 1000,height = 1000,marker_size = 14,is_show = True,is_save = False,save_type = "pdf"):

    import plotly.express as px
    size = len(set(dataframe[colorattribute]))
    #dataframe.sort_values(by = colorattribute,inplace=True, ascending=True)
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
    #     xaxis_range = [min(it_mapping_csv.row)-10,min(it_mapping_csv.row) + max_length ],
    #     yaxis_range =[min(it_mapping_csv.col)-10,min(it_mapping_csv.col) + max_length]
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
def mclust_R(adata, num_cluster,obs_name = "mclust" ,modelNames='EEE', used_obsm='scGraphMAE',random_seed = 2022):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster,modelNames)
    #print(mclust_res)
    mclust_res = np.array(res[-2])
    #print(mclust_res)
    adata.obs[obs_name] = mclust_res
    adata.obs[obs_name] = adata.obs[obs_name].astype('int')
    adata.obs[obs_name] = adata.obs[obs_name].astype('category')
    return adata
def Graph_get(adata,threshold = 30,feature_dim_method = "PCA",num_features = 600,row_name = "imagerow",col_name = "imagecol"):
    cell_loc = adata.obsm["spatial"]
    distance_np = pdist(cell_loc, metric = "euclidean")
    distance_np_X =squareform(distance_np)
    #distance_loc_csv = pd.DataFrame(index=adata.obs.index, columns=adata.obs.index,data = distance_np_X)
    num_big = np.where((0< distance_np_X)&(distance_np_X < threshold))[0].shape[0]
    adj_matrix = np.zeros(distance_np_X.shape)
    non_zero_point = np.where((0 < distance_np_X) & (distance_np_X < threshold))
    for i in range(num_big):
        x = non_zero_point[0][i]
        y = non_zero_point[1][i]
        adj_matrix[x][y] = 1 
    adj_matrix = adj_matrix + np.eye(distance_np_X.shape[0])
    adj_matrix  = np.float32(adj_matrix)
    adj_matrix_crs = sparse.csr_matrix(adj_matrix)
    graph = dgl.from_scipy(adj_matrix_crs,eweight_name='w')
    if(feature_dim_method == "PCA"):
        adata_X = adata.obsm["X_pca"]
    else:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=num_features)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata_Vars =  adata[:, adata.var['highly_variable']]
        adata_X = adata_Vars.X.todense()
    graph.ndata["feat"] = torch.tensor(adata_X.copy())
    print(graph)
    return graph
def train_model(adata,graph,args,device = 0 ,seed = 0):
    max_epoch = args.max_epoch
    #训练模型
    #提取到的embedding保存在adata里的obsm["scGraphMAE"]、
    lr = args.lr
    weight_decay = args.weight_decay
    optim_type = args.optimizer 
    set_random_seed(seed)
    model = build_model(args)
    model.to(device)
    optimizer = create_optimizer(optim_type, model, lr, weight_decay)
    x = graph.ndata["feat"]
    scheduler  = None
    logger = None
    if not load_model:
        model = pretrain(model, graph, x, optimizer, max_epoch, device, scheduler)
    embedding = model.embed(graph.to(device), x.to(device))
    adata.obsm["scGraphMAE"] = embedding.cpu().detach().numpy()
    return adata
def cluster_comp(adata,num_classes, used_obsm='scGraphMAE',K_means_value = "K_means_value",m_clust_EEE = "mclust_EEE",m_clust_VII = "mclust_VII",m_clust_EII = "mclust_EII"):
    k_means_value = kMeans_use(adata.obsm[used_obsm],num_classes)
    adata.obs[K_means_value] = k_means_value
    #k_means_score = adjusted_rand_score(adata.obs.lay_num.values, k_means_value )
    try:
        adata = mclust_R(adata,used_obsm=used_obsm,obs_name = m_clust_EEE,num_cluster=num_classes,modelNames = "EEE")
    except:
        adata.obs[m_clust_EEE] = 0
    try:
        adata = mclust_R(adata,used_obsm=used_obsm,obs_name = m_clust_VII,num_cluster=num_classes,modelNames = "VII")
    except:
        adata.obs[m_clust_VII] = 0
    try:
        adata = mclust_R(adata,used_obsm=used_obsm,obs_name = m_clust_EII,num_cluster=num_classes,modelNames = "EII")
    except:
        adata.obs[m_clust_EII] = 0
    return adata
def evaluate_ari_comp(adata,true_columns = "lay_num",K_means_value = "K_means_value",m_clust_EEE = "mclust_EEE",m_clust_VII = "mclust_VII",m_clust_EII = "mclust_EII"):
    if(K_means_value != None):
        k_means_score = adjusted_rand_score(adata.obs[true_columns], adata.obs[K_means_value] )
        adata.uns[K_means_value] = k_means_score
    if(m_clust_EEE != None):
        m_clust_EEE_score = adjusted_rand_score(adata.obs[true_columns], adata.obs[m_clust_EEE] )
        adata.uns[m_clust_EEE] = m_clust_EEE_score
    if(m_clust_VII != None):
        m_clust_VII_score = adjusted_rand_score(adata.obs[true_columns], adata.obs[m_clust_VII] )
        adata.uns[m_clust_VII] = m_clust_VII_score
    if(m_clust_EII != None):
        m_clust_EII_score = adjusted_rand_score(adata.obs[true_columns], adata.obs[m_clust_EII] )
        adata.uns[m_clust_EII] = m_clust_EII_score
    return adata
parser = argparse.ArgumentParser(description="GAT")
parser.add_argument("--seeds", type=int, nargs="+", default=[0])
parser.add_argument("--dataset", type=str, default="cora")
parser.add_argument("--device", type=int, default=-1)
parser.add_argument("--max_epoch", type=int, default=200,
                    help="number of training epochs")
parser.add_argument("--warmup_steps", type=int, default=-1)

parser.add_argument("--num_heads", type=int, default=4,
                    help="number of hidden attention heads")
parser.add_argument("--num_out_heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--num_layers", type=int, default=2,
                    help="number of hidden layers")
parser.add_argument("--num_hidden", type=int, default=256,
                    help="number of hidden units")
parser.add_argument("--residual", action="store_true", default=False,
                    help="use residual connection")
parser.add_argument("--in_drop", type=float, default=.2,
                    help="input feature dropout")
parser.add_argument("--attn_drop", type=float, default=.1,
                    help="attention dropout")
parser.add_argument("--norm", type=str, default=None)
parser.add_argument("--lr", type=float, default=0.005,
                    help="learning rate")
parser.add_argument("--weight_decay", type=float, default=5e-4,
                    help="weight decay")
parser.add_argument("--negative_slope", type=float, default=0.2,
                    help="the negative slope of leaky relu for GAT")
parser.add_argument("--activation", type=str, default="prelu")
parser.add_argument("--mask_rate", type=float, default=0.5)
parser.add_argument("--drop_edge_rate", type=float, default=0.0)
parser.add_argument("--replace_rate", type=float, default=0.0)

parser.add_argument("--encoder", type=str, default="gat")
parser.add_argument("--decoder", type=str, default="gat")
parser.add_argument("--loss_fn", type=str, default="byol")
parser.add_argument("--alpha_l", type=float, default=2, help="`pow`inddex for `sce` loss")
parser.add_argument("--optimizer", type=str, default="adam")

parser.add_argument("--max_epoch_f", type=int, default=30)
parser.add_argument("--lr_f", type=float, default=0.001, help="learning rate for evaluation")
parser.add_argument("--weight_decay_f", type=float, default=0.0, help="weight decay for evaluation")
parser.add_argument("--linear_prob", action="store_true", default=False)

parser.add_argument("--load_model", action="store_true")
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--use_cfg", action="store_true")
parser.add_argument("--logging", action="store_true")
parser.add_argument("--scheduler", action="store_true", default=False)
parser.add_argument("--concat_hidden", action="store_true", default=False)

# for graph classification
parser.add_argument("--pooling", type=str, default="mean")
parser.add_argument("--deg4feat", action="store_true", default=False, help="use node degree as input feature")
parser.add_argument("--batch_size", type=int, default=32)
sample_names = ["151507","151508","151509","151510","151669","151670","151671","151672","151673","151674","151675","151676"]
#sample_names = ["151507"]
scGraph_fold = "/home/sunhang/Embedding/baseline/10X_batch/"
result_csv = pd.DataFrame()