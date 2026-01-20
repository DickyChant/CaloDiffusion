import click
from calodiffusion.utils import utils
from calodiffusion.train.train_diffusion import TrainDiffusion
from calodiffusion.train.train_layer_model import TrainLayerModel
from calodiffusion.train.train_meanflow import TrainMeanFlow, TrainMeanFlowDDP

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


@click.group()
@click.option(
    "-d", "--data-folder", default="../data/", help="Folder containing data and MC files"
)

@click.option(
    "-c",
    "--config",
    default="configs/test.json",
    help="Config file with training parameters",
)
@click.option(
    "--checkpoint",
    "checkpoint_folder",
    default="../models",
    help="Folder with checkpoints",
)
@click.option(
    "-n", "--nevts", type=int, default=-1, help="Number of events to load"
)
@click.option(
    "--frac",
    type=float,
    default=0.85,
    help="Fraction of total events used for training",
)
@click.option(
    "--load",
    is_flag=True,
    default=False,
    help="Load pretrained weights to continue the training",
)
@click.option("--seed", type=int, default=1234, help="Pytorch seed")
@click.option('--reclean/--no-reclean', default=False, help='Reclean data')
@click.option(
    "--reset_training", is_flag=True, default=False, help="Retrain"
)
@click.option("--hgcal/--no-hgcal", default=None, is_flag=True, help="Use HGCal settings (overwrites config)")
@click.option("--model-loc", default=None, help="Specify existing model to load")
@click.option("--gmm-prior", default=None, help="Path to GMM checkpoint (.pt file) for sampling prior on-the-fly (for MeanFlow with GMM)")
@click.pass_context
def train(ctx, config, data_folder, checkpoint_folder, nevts, frac, load, seed, reclean, reset_training, model_loc, hgcal, gmm_prior): 
    ctx.ensure_object(dotdict)

    ctx.obj.config = utils.LoadJson(config)

    ctx.obj.data_folder = data_folder  
    ctx.obj.checkpoint_folder = checkpoint_folder
    ctx.obj.nevts = nevts
    ctx.obj.frac = frac
    ctx.obj.load = load
    ctx.obj.seed = seed
    ctx.obj.reclean = reclean
    ctx.obj.reset_training = reset_training
    ctx.obj.hgcal = hgcal
    ctx.obj.model_loc = model_loc
    ctx.obj.gmm_prior = gmm_prior

    if hgcal is not None: 
        ctx.obj.config['HGCAL'] = hgcal
        ctx.obj.hgcal = hgcal
    else: 
        ctx.obj.hgcal = ctx.obj.config.get("HGCAL", False)


@train.command()
@click.pass_context
def diffusion(ctx): 
    ctx.obj.model = "diffusion"
    TrainDiffusion(ctx.obj, ctx.obj.config).train()

@train.command()
@click.option("--layer-model-loc", default=None, help="Specify existing layer model to load")
@click.pass_context
def layer(ctx, layer_model_loc):
    ctx.obj.model = "layer"
    if (layer_model_loc is not None) and ctx.obj.load: 
        ctx.obj.config['layer_model'] = layer_model_loc 

    #self.layer_steps = self.config.get("LAYER_STEPS")
    #sampler_algo = self.config.get("LAYER_SAMPLER", "DDim")

    TrainLayerModel(ctx.obj, ctx.obj.config).train()


@train.command()
@click.pass_context
def meanflow(ctx):
    """Train MeanFlow diffusion model (with optional GMM prior)."""
    ctx.obj.model = "meanflow"
    TrainMeanFlow(ctx.obj, ctx.obj.config).train()


@train.command("meanflow-ddp")
@click.pass_context
def meanflow_ddp(ctx):
    """
    Train MeanFlow diffusion model with multi-GPU support using DDP.
    
    This command enables efficient multi-GPU training on a single node or 
    across multiple nodes using PyTorch's DistributedDataParallel (DDP).
    
    Usage:
    
    \b
    # Single node, multi-GPU (4 GPUs):
    torchrun --nproc_per_node=4 -m calodiffusion.training \\
        -c config.json -d /path/to/data meanflow-ddp
    
    \b
    # Multi-node training (2 nodes, 4 GPUs each):
    # On node 0:
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \\
        --master_addr=<master_ip> --master_port=29500 \\
        -m calodiffusion.training -c config.json meanflow-ddp
    
    \b
    # On node 1:
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \\
        --master_addr=<master_ip> --master_port=29500 \\
        -m calodiffusion.training -c config.json meanflow-ddp
    
    The batch size per GPU can be configured using BATCH_MEANFLOW in the config.
    The effective total batch size will be BATCH_MEANFLOW * num_gpus.
    """
    ctx.obj.model = "meanflow-ddp"
    TrainMeanFlowDDP(ctx.obj, ctx.obj.config).train()


if __name__ == "__main__": 
    train()
