from xml.sax.handler import feature_validation
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, 
                             silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)
from sklearn.preprocessing import normalize
import logging
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from .utils import (
    build_args,
    create_optimizer,
    mask_edge,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
    KMeans_use,
    mkdir,
    drawPicture
)
import os
import matplotlib.pyplot as plt
from scanpy import read_10x_h5
from graphmae.models import build_model
from sklearn.cluster import KMeans
#import calculate_adj
import anndata as ad
import scanpy as sc
from operator import index
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np
from scipy import sparse
import dgl
import torch
import warnings
import sys
from .parameters_dict import parametersDict
warnings.filterwarnings("ignore")

def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
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
    # return best_model
    return model

def test(args, graph, spatial_fold, adata):
    result_pd = pd.read_csv(spatial_fold + "result.csv",index_col=0)
    if args.groud_truth_column_name != "" and args.groud_truth_column_name in adata.obs.columns:
        parameters = {
                "dataset_name":(args.dataset,),
                "graph_devise":("CCST",),
                "threshold_num":(args.threshold_num,),
                "feature_dim":(args.feature_dim,),
                "feature_dim_num" : (args.feature_dim_num,),
                "mask_rate" : (float(result_pd[result_pd['ARI'] == result_pd['ARI'].max()].loc[:,'mask_rate']),),
                "code_networt_and_norm" : (["gin","gin","batchnorm"],),
                "num_hidden" : (args.num_hidden,),
                "num_layers" : (args.num_layers,),
                "activation" : (args.activation,),
                "max_epoch" : (args.max_epoch,),
                "lr" : (args.lr,),
                "alpha_l" : (float(result_pd[result_pd['ARI'] == result_pd['ARI'].max()].loc[:,'alpha_l']),),
                "loss":("sce",),
                "replace_rate":(0.05,)
        }
    else:
        parameters = {
                "dataset_name":(args.dataset,),
                "graph_devise":("CCST",),
                "threshold_num":(args.threshold_num,),
                "feature_dim":(args.feature_dim,),
                "feature_dim_num" : (args.feature_dim_num,),
                "mask_rate" : (float(result_pd[result_pd['silhouette_score'] == result_pd['silhouette_score'].min()].loc[:,'mask_rate']),),
                "code_networt_and_norm" : (["gin","gin","batchnorm"],),
                "num_hidden" : (args.num_hidden,),
                "num_layers" : (args.num_layers,),
                "activation" : (args.activation,),
                "max_epoch" : (args.max_epoch,),
                "lr" : (args.lr,),
                "alpha_l" : (float(result_pd[result_pd['silhouette_score'] == result_pd['silhouette_score'].min()].loc[:,'alpha_l']),),
                "loss":("sce",),
                "replace_rate":(0.05,)
        }
    parameters_dict = parametersDict(parameters)
    choose_parameter  = parameters_dict.myiter()


    # args.lr = choose_parameter[0]["lr"]
    args.lr_f = 0.01
    # args.num_hidden = choose_parameter[0]["num_hidden"]
    args.num_heads = 4
    args.weight_decay = 2e-4
    args.weight_decay_f= 1e-4
    # args.max_epoch= choose_parameter[0]["max_epoch"]
    args.max_epoch_f= 300
    args.mask_rate= choose_parameter[0]["mask_rate"]
    # args.num_layers= choose_parameter[0]["num_layers"]
    args.encoder= choose_parameter[0]["code_networt_and_norm"][0]
    args.decoder= choose_parameter[0]["code_networt_and_norm"][1]
    args.norm = choose_parameter[0]["code_networt_and_norm"][2]
    # args.activation= choose_parameter[0]["activation"]
    args.in_drop= 0.2
    args.attn_drop= 0.1
    args.linear_prob= True
    args.loss_fn= choose_parameter[0]["loss"]
    args.drop_edge_rate=0.0
    args.optimizer= "adam"
    args.replace_rate= 0.05 
    args.alpha_l = choose_parameter[0]["alpha_l"]
    args.scheduler= True
    # args.dataset = choose_parameter[0]["dataset_name"]
    activation = choose_parameter[0]["activation"]
    

    #默认参数传递
    device = args.device if args.device >= 0 else "cpu"
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_classes = args.num_classes
    optim_type = args.optimizer 

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model 
    row_name = args.image_row_name
    col_name = args.image_col_name

    seed = 0
    set_random_seed(seed)
    logger = None
    model = build_model(args)
    # device = 0
    model.to(device)
    optimizer = create_optimizer(optim_type, model, lr, weight_decay)
    
    scheduler = None
    x = graph.ndata["feat"]
    if not load_model:
        model = pretrain(model, graph, x, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
        #model = model.cpu()
    if load_model:
        logging.info("Loading Model ... ")
        model.load_state_dict(torch.load("saved_model.pt"))
    if save_model:
        logging.info("Saveing Model ...")
        torch.save(model.state_dict(), "saved_model.pt")

    model.train(False)
    x = graph.ndata["feat"]
    embedding = model.embed(graph.to(device), x.to(device))
    adata.obsm["scGraph_embedding"]  = embedding.cpu().detach().numpy()
    middle_embedding = model.encoder_to_decoder(embedding)
    result_embedding =  model.decoder(graph.to(device),middle_embedding)
    adata.obsm["scGraphMAE_gene"] = result_embedding.cpu().detach().numpy()
    new_pred = KMeans_use(embedding.cpu().detach().numpy(),num_classes)
    adata.obs["pre"] = new_pred
    
    # score = adjusted_rand_score(adata.obs.layer_num.values, new_pred)
    kmeans = KMeans(n_clusters=num_classes, init="k-means++", random_state=0)
    embedding = embedding.cpu().detach().numpy()
    kmeans_pred = kmeans.fit_predict(embedding)
    adata.obs["cluster_num"] = kmeans_pred
    adata.obs["cluster_str"] = adata.obs["cluster_num"].apply(str)
    feature_matrix = middle_embedding.cpu().detach().numpy()
    imputation_matrix = result_embedding.cpu().detach().numpy()
    if args.feature_dim == "PCA":
        imputation_df = pd.DataFrame(imputation_matrix,index=adata.obs.index)        
    else:
        adata_Vars =  adata[:, adata.var['highly_variable']]
        imputation_df = pd.DataFrame(imputation_matrix,index=adata_Vars.obs.index,columns=adata_Vars.var.index)
    feature_df = pd.DataFrame(feature_matrix,index=adata.obs.index)
    feature_df.to_csv(spatial_fold + "feature_matrix.csv")
    imputation_df.to_csv(spatial_fold + "imputation_matrix.csv")
    adata.obs['cluster_str'].to_csv(spatial_fold +  "cluster_SEA.csv")
    drawPicture(adata.obs,col_name =col_name,row_name = row_name, save_file = spatial_fold + "/SEA_clusters.pdf",colorattribute = "cluster_str",marker_size = 11,is_show=False,is_save= True,)
