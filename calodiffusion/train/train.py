from abc import ABC, abstractmethod
import json
import os

import torch

from calodiffusion.utils import utils
from calodiffusion.utils import distributed as dist_utils

tqdm = utils.import_tqdm()

class Train(ABC): 
    def __init__(self, flags, config, load_data:bool=True, save_model: bool = True) -> None:
        # Initialize distributed training if requested
        self.use_ddp = self._should_use_ddp(flags)
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        
        if self.use_ddp:
            self._setup_distributed(flags)
        
        # Set device based on distributed setup
        if self.use_ddp:
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.local_rank)
        else:
            self.device = utils.get_device()
        
        self.save_model = save_model

        if load_data: 
            self.loader_train, self.loader_val, self.sampler_train, self.sampler_val = utils.load_data(
                flags, config, distributed=self.use_ddp
            )
        else:
            # Initialize samplers as None if not loading data
            self.sampler_train = None
            self.sampler_val = None
        
        self.config = config
        self.flags = flags
        self.batch_size = self.config.get("BATCH", 256)
        if self.save_model: 
            self.checkpoint_folder = f"{flags.checkpoint_folder.strip('/')}/{config['CHECKPOINT_NAME']}_{flags.model}/"
            if not os.path.exists(self.checkpoint_folder):
                os.makedirs(self.checkpoint_folder)

        self.checkpoint_folder = os.path.join(flags.checkpoint_folder, f"{config['CHECKPOINT_NAME']}_{self.__class__.__name__.removeprefix('Train')}")
        
        if hasattr(flags, "sample_algo"): 
            if flags.sample_algo is not None: 
                self.config['SAMPLER'] == flags.sample_algo

        if hasattr(flags, "model_loc"): 
            if flags.model_loc is not None: 
                self.checkpoint_folder = os.path.dirname(flags.model_loc)

        if not os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)

        with open(os.path.join(self.checkpoint_folder, "config.json"), "w") as config_file:
            json.dump(flags.config, config_file) 
    
    def _should_use_ddp(self, flags):
        """Determine if DDP should be enabled based on flags and environment."""
        # Check SLURM environment first
        slurm_info = dist_utils.get_distributed_info_from_slurm()
        if slurm_info is not None and slurm_info['world_size'] > 1:
            return True
        
        # Explicit DDP flag
        if hasattr(flags, 'enable_ddp') and flags.enable_ddp:
            return True
        
        # Multi-GPU or multi-node configuration
        if hasattr(flags, 'n_nodes') and hasattr(flags, 'gpus_per_node'):
            if flags.n_nodes > 1 or flags.gpus_per_node > 1:
                return True
        
        return False
    
    def _setup_distributed(self, flags):
        """Initialize distributed training."""
        # First check if running in SLURM environment
        slurm_info = dist_utils.get_distributed_info_from_slurm()
        
        if slurm_info is not None:
            # Use SLURM environment variables
            self.rank = slurm_info['global_rank']
            self.world_size = slurm_info['world_size']
            self.local_rank = slurm_info['local_rank']
            master_addr = slurm_info['master_addr']
            master_port = slurm_info['master_port']
            backend = getattr(flags, 'backend', 'nccl')
        else:
            # Use command-line arguments or defaults
            n_nodes = getattr(flags, 'n_nodes', 1)
            gpus_per_node = getattr(flags, 'gpus_per_node', 1)
            
            # Auto-detect GPUs if not specified explicitly (and > 1)
            if gpus_per_node == 1 and torch.cuda.is_available():
                available_gpus = torch.cuda.device_count()
                if available_gpus > 1 and not hasattr(flags, 'gpus_per_node'):
                    gpus_per_node = available_gpus
            
            self.world_size = n_nodes * gpus_per_node
            
            # Get rank from environment (set by torchrun or similar)
            self.rank = int(os.environ.get('RANK', 0))
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            master_addr = getattr(flags, 'master_addr', 'localhost')
            master_port = getattr(flags, 'master_port', '29500')
            backend = getattr(flags, 'backend', 'nccl')
        
        # Initialize the process group
        dist_utils.setup_distributed(
            backend=backend,
            master_addr=master_addr,
            master_port=master_port,
            rank=self.rank,
            world_size=self.world_size
        ) 

    @abstractmethod
    def init_model(self): 
        raise NotImplementedError

    @abstractmethod
    def training_loop(
        self, 
        optimizer, 
        scheduler, 
        early_stopper, 
        start_epoch, 
        num_epochs, 
        training_losses, 
        val_losses): 

        raise NotImplementedError
    
    def pickup_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        early_stopper,
        n_epochs,
        restart_training,
    ):
        
        if not hasattr(self.flags, "model_loc") or (self.flags.model_loc is None)  :
            checkpoint_path = os.path.join(self.checkpoint_folder, "checkpoint.pth")

            if os.path.exists(checkpoint_path):
                print("Loading training checkpoint from %s" % checkpoint_path, flush=True)
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            else:
                raise ValueError("No checkpoint at %s" % checkpoint_path)
        else: 
            checkpoint = torch.load(self.flags.model_loc, map_location=self.device, weights_only=False)

        if "model_state_dict" in checkpoint.keys():
            state_dict = checkpoint["model_state_dict"]
            
            # Handle loading checkpoints with or without DDP module prefix
            # Check if the checkpoint has 'module.' prefix
            checkpoint_has_module = any(k.startswith('module.') for k in state_dict.keys())
            # Check if current model has 'module.' (is wrapped in DDP)
            model_has_module = hasattr(model, 'module')
            
            # If there's a mismatch, adjust the state dict
            if checkpoint_has_module and not model_has_module:
                # Remove 'module.' prefix from checkpoint
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            elif not checkpoint_has_module and model_has_module:
                # Add 'module.' prefix to checkpoint
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict)
        elif len(checkpoint.keys()) > 1:
            model.load_state_dict(checkpoint)

        if "optimizer_state_dict" in checkpoint.keys() and not restart_training:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint.keys() and not restart_training:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "early_stop_dict" in checkpoint.keys() and not restart_training:
            early_stopper.__dict__ = checkpoint["early_stop_dict"]

        training_losses = {}
        val_losses = {}
        start_epoch = 0

        if "train_loss_hist" in checkpoint.keys() and not restart_training:
            training_losses = checkpoint["train_loss_hist"]
            val_losses = checkpoint["val_loss_hist"]
            start_epoch = checkpoint["epoch"] + 1

        return model, optimizer, scheduler, start_epoch, training_losses, val_losses

    def save(
        self,
        model_state,
        epoch,
        name,
        training_losses,
        validation_losses,
        optimizer,
        scheduler,
        early_stopper,
    ):
        # Only save from the main process (rank 0)
        if not dist_utils.is_main_process(self.rank):
            return
        
        if self.save_model: 
            final_path = os.path.join(self.checkpoint_folder, f"{name}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss_hist": training_losses,
                    "val_loss_hist": validation_losses,
                    "early_stop_dict": early_stopper.__dict__,
                },
                final_path,
            )

        with open(self.checkpoint_folder + f"/{name}_training_losses.txt", "w") as tfileout:
            tfileout.write("\n".join(str(loss) for loss in training_losses.values()) + "\n")
        with open(self.checkpoint_folder + f"/{name}_validation_losses.txt", "w") as vfileout:
            vfileout.write("\n".join(str(vl) for vl in validation_losses.values()) + "\n")


    def train(self): 
        if not hasattr(self, "model"): 
            self.init_model()

        num_epochs = self.config.get("MAXEPOCH", 30)
        early_stopper = utils.EarlyStopper(
            patience=self.config["EARLYSTOP"], mode="val_loss", min_delta=1e-5
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config["LR"]))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, factor=0.1, patience=15
        )

        start_epoch = 0
        if self.flags.load:
            model, optimizer, scheduler, start_epoch, training_losses, val_losses = (
                self.pickup_checkpoint(
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    early_stopper=early_stopper,
                    n_epochs=num_epochs,
                    restart_training=self.flags.reset_training,
                )
            )
        else: 
            training_losses = dict()
            val_losses = dict()

        model, epoch, training_losses, val_losses, optimizer, scheduler, early_stopper = self.training_loop(
            optimizer, 
            scheduler, 
            early_stopper, 
            start_epoch, 
            num_epochs, 
            training_losses, 
            val_losses
        )
        # Also save at the end of training
        # Get the underlying model state dict (unwrap DDP if needed)
        if self.use_ddp:
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        
        self.save(
            model_state,
            epoch=epoch,
            name="final",
            training_losses=training_losses,
            validation_losses=val_losses,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopper=early_stopper,
        )
        
        # Cleanup distributed training
        if self.use_ddp:
            dist_utils.cleanup_distributed()
