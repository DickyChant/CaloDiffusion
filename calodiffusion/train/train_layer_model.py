from calodiffusion.models.layerdiffusion import LayerDiffusion
from calodiffusion.train.train_diffusion import TrainDiffusion
import torch

class TrainLayerModel(TrainDiffusion):
    def __init__(self, flags, config, load_data = True, inference=False):
        super().__init__(flags, config, load_data)
        self.init_model()
        if inference: 
            # Unwrap DDP if needed to access model methods
            model_ref = self.model.module if self.use_ddp else self.model
            model_ref.set_layer_state(False)
        else: 
            # Unwrap DDP if needed to access model methods
            model_ref = self.model.module if self.use_ddp else self.model
            model_ref.set_layer_state(True)

    def init_model(self):
        self.config['checkpoint'] = self.checkpoint_folder
        self.model = LayerDiffusion(
            self.config, n_steps = self.config["NSTEPS"], loss_type = self.config['LOSS_TYPE']
        )
        self.model = self.model.to(self.device)
        
        # Wrap with DDP if enabled
        if self.use_ddp:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )