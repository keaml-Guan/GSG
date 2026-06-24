from . import models
from . import utils

def GSG_train(adata, graph, args):
    device = args.device if args.device >= 0 else "cpu"
    utils.set_random_seed(args.seeds)
    model = models.build_model(args)
    model.to(device)
    optimizer = models.utils.create_optimizer(args.optimizer, model, args.lr, args.weight_decay)
    x = graph.ndata["feat"]
    if not args.load_model:
        model = models.utils.pretrain(model, graph, x, optimizer, args.max_epoch, device)
    model.train(False)
    x = graph.ndata["feat"]
    embedding = model.embed(graph.to(device), x.to(device))
    adata.obsm["GSG_embedding"] = embedding.cpu().detach().numpy()
    if args.imputation:
        latten_embedding = model.encoder_to_decoder(embedding)
        imputation_embedding =  model.decoder(graph.to(device),latten_embedding)
        adata.obsm["GSG_imputation"] = imputation_embedding.cpu().detach().numpy()
    return adata, model