import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.nn.utils import clip_grad_norm_

from calodiffusion.utils import utils
from calodiffusion.train.train import Train
from calodiffusion.models.calodiffusion import CaloDiffusion


class TrainMeanFlow(Train):
    """Training class for MeanFlow diffusion model using standard CaloDiffusion framework."""
    
    def __init__(self, flags, config, load_data=True, save_model=True):
        # Check if GMM prior is provided
        self.use_gmm = hasattr(flags, 'gmm_prior') and flags.gmm_prior is not None
        if self.use_gmm:
            print(f"Using GMM prior file: {flags.gmm_prior}", flush=True)
        
        super().__init__(flags, config, load_data=load_data, save_model=save_model)
        
        # Load GMM prior if provided
        self.gmm_prior_data = None
        if self.use_gmm and load_data:
            self._load_gmm_prior(flags.gmm_prior)
    
    def _load_gmm_prior(self, prior_path):
        """Load GMM prior data from H5 file."""
        import h5py as h5
        import numpy as np
        
        if not os.path.exists(prior_path):
            raise FileNotFoundError(f"GMM prior file not found: {prior_path}")
        
        print(f"Loading GMM prior from {prior_path}", flush=True)
        with h5.File(prior_path, "r") as f:
            prior = f["showers"][:].astype(np.float32)
        
        # Reshape to match data shape
        shape = self.config.get("SHAPE_PAD")
        if shape is None:
            shape = self.config.get("SHAPE_FINAL")
        
        if prior.ndim == 2 and prior.shape[1] == np.prod(shape[1:]):
            prior = prior.reshape(-1, *shape[1:])
        elif prior.ndim == len(shape) and prior.shape[1:] == tuple(shape[1:]):
            pass  # Already correct shape
        else:
            print(f"[WARNING] Prior shape {prior.shape} doesn't match expected {shape}, attempting reshape...")
            try:
                prior = prior.reshape(-1, *shape[1:])
            except:
                raise ValueError(f"Cannot reshape prior from {prior.shape} to {shape}")
        
        # Match number of events with training data
        # We'll handle this in training_loop by indexing
        self.gmm_prior_data = torch.from_numpy(prior).cpu()
        print(f"Loaded GMM prior with shape: {self.gmm_prior_data.shape}", flush=True)
    
    def init_model(self):
        """Initialize the CaloDiffusion model for MeanFlow training."""
        shape = self.config.get("SHAPE_PAD")
        if shape is None:
            shape = self.config.get("SHAPE_FINAL")
        
        # Remove batch dimension for model init
        model_shape = shape[1:] if len(shape) > 1 else shape
        
        self.model = CaloDiffusion(
            self.config,
            n_steps=self.config.get("NSTEPS", 400),
            loss_type=self.config.get("LOSS_TYPE", "l2")
        ).to(device=self.device)
    
    def training_loop(
        self, optimizer, scheduler, early_stopper, start_epoch, num_epochs, training_losses, val_losses
    ):
        """Training loop for MeanFlow model."""
        tqdm = utils.import_tqdm()
        cold_diffu = self.config.get("COLD_DIFFU", False)
        cold_noise_scale = self.config.get("COLD_NOISE", 1.0)
        energy_loss_scale = self.config.get("ENERGY_LOSS_SCALE", 0.0)
        weight_pidm = self.config.get("WEIGHT_PIDM", 0.03)
        MAX_GRAD_NORM = 1.0
        
        # Fixed noise levels for validation loss stability
        if self.loader_val is not None:
            val_rnd = torch.randn(
                (len(self.loader_val), self.batch_size), device=self.device
            )
        
        min_validation_loss = 99999.0
        
        for epoch in range(start_epoch, num_epochs):
            print("Beginning epoch %i" % epoch, flush=True)
            train_loss = 0
            
            self.model.train()
            for i, (E, layers, data) in tqdm(
                enumerate(self.loader_train, 0), unit="batch", total=len(self.loader_train)
            ):
                self.model.zero_grad()
                optimizer.zero_grad()
                
                data = data.to(device=self.device)
                E = E.to(device=self.device)
                layers = layers.to(device=self.device)
                
                noise = torch.randn_like(data)
                
                if cold_diffu:
                    noise = self.model.gen_cold_image(E, cold_noise_scale, noise)
                
                # Load GMM prior for this batch if using GMM
                prior = None
                if self.use_gmm and self.gmm_prior_data is not None:
                    batch_size = data.shape[0]
                    # Use modulo to cycle through prior data if needed
                    start_idx = (i * self.batch_size) % len(self.gmm_prior_data)
                    end_idx = min(start_idx + batch_size, len(self.gmm_prior_data))
                    prior_batch = self.gmm_prior_data[start_idx:end_idx]
                    
                    # Pad if needed
                    if len(prior_batch) < batch_size:
                        pad_size = batch_size - len(prior_batch)
                        prior_batch = torch.cat([
                            prior_batch,
                            self.gmm_prior_data[:pad_size]
                        ], dim=0)
                    
                    prior = prior_batch.to(device=self.device)
                
                # Compute loss
                if self.use_gmm and prior is not None:
                    loss_vec, loss_ref = self.model.compute_loss_meanflow_gmm(
                        data, E, gmm_prior=prior, noise=noise, 
                        energy_loss_scale=energy_loss_scale, layers=layers
                    )
                else:
                    loss_vec, loss_ref = self.model.compute_loss_meanflow(
                        data, E, noise=noise, 
                        energy_loss_scale=energy_loss_scale, layers=layers
                    )
                
                batch_loss = loss_vec.mean()
                batch_loss_ref = loss_ref.mean()
                
                batch_loss.backward()
                
                # Gradient clipping
                grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=MAX_GRAD_NORM)
                
                optimizer.step()
                train_loss += batch_loss.item()
                
                del data, E, layers, noise, batch_loss, batch_loss_ref
                if prior is not None:
                    del prior
            
            train_loss = train_loss / len(self.loader_train)
            training_losses[epoch] = train_loss
            print("loss: " + str(train_loss))
            
            val_loss = 0
            self.model.eval()
            if self.loader_val is not None:
                for i, (vE, vlayers, vdata) in tqdm(
                    enumerate(self.loader_val, 0), unit="batch", total=len(self.loader_val)
                ):
                    vdata = vdata.to(device=self.device)
                    vE = vE.to(device=self.device)
                    vlayers = vlayers.to(device=self.device)
                    
                    noise = torch.randn_like(vdata)
                    
                    if cold_diffu:
                        noise = self.model.gen_cold_image(vE, cold_noise_scale, noise)
                    
                    # Load GMM prior for validation batch if using GMM
                    vprior = None
                    if self.use_gmm and self.gmm_prior_data is not None:
                        batch_size = vdata.shape[0]
                        # Use different indexing for validation
                        val_start_idx = (i * self.batch_size + len(self.loader_train) * self.batch_size) % len(self.gmm_prior_data)
                        val_end_idx = min(val_start_idx + batch_size, len(self.gmm_prior_data))
                        prior_batch = self.gmm_prior_data[val_start_idx:val_end_idx]
                        
                        if len(prior_batch) < batch_size:
                            pad_size = batch_size - len(prior_batch)
                            prior_batch = torch.cat([
                                prior_batch,
                                self.gmm_prior_data[:pad_size]
                            ], dim=0)
                        
                        vprior = prior_batch.to(device=self.device)
                    
                    # Compute validation loss
                    if self.use_gmm and vprior is not None:
                        loss_vec, loss_ref = self.model.compute_loss_meanflow_gmm(
                            vdata, vE, gmm_prior=vprior, noise=noise,
                            energy_loss_scale=energy_loss_scale, layers=vlayers
                        )
                    else:
                        loss_vec, loss_ref = self.model.compute_loss_meanflow(
                            vdata, vE, noise=noise,
                            energy_loss_scale=energy_loss_scale, layers=vlayers
                        )
                    
                    batch_loss = loss_vec.mean()
                    val_loss += batch_loss.item()
                    
                    del vdata, vE, vlayers, noise, batch_loss
                    if vprior is not None:
                        del vprior
                
                val_loss = val_loss / len(self.loader_val)
                val_losses[epoch] = val_loss
                print("val_loss: " + str(val_loss), flush=True)
            
            scheduler.step(torch.tensor([train_loss]))
            
            if val_loss < min_validation_loss:
                if self.save_model:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.checkpoint_folder, "best_val.pth")
                    )
                min_validation_loss = val_loss
            
            if early_stopper.early_stop(val_loss):
                print("Early stopping!")
                break
            
            # Save checkpoint
            self.model.eval()
            print("SAVING")
            self.save(
                self.model.state_dict(),
                epoch=epoch,
                name="checkpoint",
                training_losses=training_losses,
                validation_losses=val_losses,
                optimizer=optimizer,
                scheduler=scheduler,
                early_stopper=early_stopper,
            )
        
        return self.model, epoch, training_losses, val_losses, optimizer, scheduler, early_stopper

