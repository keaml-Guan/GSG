import os
import argparse
import random
import yaml
import logging
from functools import partial
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import dgl
import scanpy as sc
import pandas as pd
import torch
import torch.nn as nn
from torch import optim as optim
from scipy.spatial.distance import pdist, squareform
from scipy import sparse
from tensorboardX import SummaryWriter



logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path)
		print("---  new folder...  ---")
		print("---  OK  ---")
	else:
		print("---  There is this folder!  ---")

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
    size = len(set(dataframe[colorattribute]))
    length_col = max(dataframe[col_name]) - min(dataframe[col_name])
    length_row = max(dataframe[row_name]) - min(dataframe[row_name])
    max_length = max(length_col,length_row) + 2
    fig = px.scatter(dataframe, x = col_name, y= row_name,color = colorattribute,color_discrete_sequence=celltype_colors)
    fig.update_traces(marker_size=marker_size)
    fig.update_layout(
        xaxis = dict(
            tickmode = 'linear',
            tick0 = min(dataframe[row_name]),
            dtick = max_length
        ),
        yaxis = dict(
            tickmode = 'linear',
            tick0 = min(dataframe[col_name]),
            dtick = max_length
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

def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--dataset", type=str, default="Stere-seq")
    parser.add_argument("--data_path", type=str, default="./data/151673.h5ad") # Modified by zpwu
    parser.add_argument("--image_row_name", type=str, default="imagerow",
                        help="row of spatial position") # Modified by zpwu
    parser.add_argument("--image_col_name", type=str, default="imagecol",
                        help="column of spatial position") # Modified by zpwu
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--max_epoch", type=int, default=500,
                        help="number of training epochs") # Modified by zpwu
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--threshold_num", type=int, default=120) # Modified by zpwu
    parser.add_argument("--feature_dim", type=str, default="HVG") # Modified by zpwu
    parser.add_argument("--feature_dim_num", type=int, default=3000) # Modified by zpwu
    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="number of hidden layers") # Modified by zpwu
    parser.add_argument("--num_hidden", type=int, default=128,
                        help="number of hidden units")
    parser.add_argument("--num_classes", type=int, default=8,
                        help="number of domains") # Modified by zpwu
    parser.add_argument("--groud_truth_column_name", type=str, default="",
                        help="column name of groud truth") # Modified by zpwu
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.2,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.1,
                        help="attention dropout")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate") # Modified by zpwu
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--activation", type=str, default="elu") # Modified by zpwu
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
    args = parser.parse_args()
    return args


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return None


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


# -------------------
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
def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args

# ------ logging ------

class TBLogger(object):
    def __init__(self, log_path="./logging_data", name="run"):
        super(TBLogger, self).__init__()

        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        self.last_step = 0
        self.log_path = log_path
        raw_name = os.path.join(log_path, name)
        name = raw_name
        for i in range(1000):
            name = raw_name + str(f"_{i}")
            if not os.path.exists(name):
                break
        self.writer = SummaryWriter(logdir=name)

    def note(self, metrics, step=None):
        if step is None:
            step = self.last_step
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        self.last_step = step

    def finish(self):
        self.writer.close()


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError
        
    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor
        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)
        sub = tensor - mean * self.mean_scale
        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias
