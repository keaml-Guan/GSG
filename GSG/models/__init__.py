from .edcoder import PreModel

#####################################################################################################################################
#   Adapted from:                                                                                                                   # 
#   @inproceedings{hou2022graphmae,                                                                                                 #
#    title={GraphMAE: Self-Supervised Masked Graph Autoencoders},                                                                   #
#    author={Hou, Zhenyu and Liu, Xiao and Cen, Yukuo and Dong, Yuxiao and Yang, Hongxia and Wang, Chunjie and Tang, Jie},          #
#    booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},                              #
#    pages={594--604},                                                                                                              #
#    year={2022}                                                                                                                    #
#   }                                                                                                                               #
#####################################################################################################################################

def build_model(args):
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    in_drop = args.in_drop
    norm = args.norm
    encoder_type = "gin"
    if args.imputation:
        decoder_type = "mlp"
    else:
        decoder_type = 'gin'
    mask_rate = args.mask_rate
    replace_rate = args.replace_rate


    activation = args.activation
    alpha_l = args.alpha_l
    num_features = args.num_features


    model = PreModel(
        in_dim=num_features,
        num_hidden=num_hidden,
        num_layers=num_layers,
        activation=activation,
        feat_drop=in_drop,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        mask_rate=mask_rate,
        norm=norm,
        replace_rate=replace_rate,
        alpha_l=alpha_l,
    )
    return model
