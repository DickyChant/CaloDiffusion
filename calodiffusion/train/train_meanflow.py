import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler

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


class TrainMeanFlowMultiGPU(TrainMeanFlow):
    """
    Training class for MeanFlow with multi-GPU support using manual gradient synchronization.
    
    This class extends TrainMeanFlow to enable multi-GPU training on a single node.
    Unlike DDP, it does NOT wrap the model with DistributedDataParallel, which is 
    important because MeanFlow uses JVP (Jacobian-vector product) that has 
    compatibility issues with DDP's gradient hooks.
    
    Instead, this class:
    1. Uses DistributedSampler to shard data across GPUs
    2. Each GPU computes gradients independently (JVP works normally)
    3. Manually synchronizes gradients using all-reduce after backward()
    4. Only main process (rank 0) saves checkpoints
    
    Usage:
        # Launch with torchrun for multi-GPU training:
        torchrun --nproc_per_node=4 -m calodiffusion.training meanflow-ddp ...
        
        # Or use the legacy torch.distributed.launch:
        python -m torch.distributed.launch --nproc_per_node=4 -m calodiffusion.training meanflow-ddp ...
    """
    
    def __init__(self, flags, config, load_data=True, save_model=True):
        # Initialize distributed environment before everything else
        self.rank, self.world_size, self.device = utils.setup_ddp()
        self.is_main = utils.is_main_process(self.rank)
        
        # Only main process should print
        if not self.is_main:
            import sys
            # Keep stderr for errors but suppress stdout
            sys.stdout = utils.NullWriter()
        
        # Set save_model to True only for main process
        save_model = save_model and self.is_main
        
        # Check if GMM prior is provided
        self.use_gmm = hasattr(flags, 'gmm_prior') and flags.gmm_prior is not None
        if self.use_gmm and self.is_main:
            print(f"Using GMM prior file: {flags.gmm_prior}", flush=True)
        
        # Initialize base Train class (skip TrainMeanFlow's __init__ and call Train's directly)
        # We do this to avoid double data loading
        Train.__init__(self, flags, config, load_data=False, save_model=save_model)
        
        # Load data with distributed sampler support
        if load_data:
            self._load_data_distributed(flags, config)
        
        # Load GMM prior if provided
        self.gmm_prior_data = None
        if self.use_gmm and load_data:
            self._load_gmm_prior(flags.gmm_prior)
        
        # Synchronize all processes after initialization
        utils.barrier()
        
        if self.is_main:
            print(f"[Multi-GPU] Initialized with {self.world_size} GPUs (no DDP wrapper for JVP compatibility)", flush=True)
    
    def _load_data_distributed(self, flags, config):
        """Load data with DistributedSampler for multi-GPU training."""
        import torch.utils.data as torchdata
        
        # First, load data using standard loader (only on main process if needed for caching)
        if self.is_main:
            print("[Multi-GPU] Loading and preparing data...", flush=True)
        
        # Load the data normally first
        loader_train, loader_val = utils.load_data(flags, config)
        
        # Synchronize to ensure data files are created
        utils.barrier()
        
        # Now create distributed samplers and loaders
        train_dataset = loader_train.dataset
        val_dataset = loader_val.dataset if loader_val is not None else None
        
        # Get batch size (use smaller batch for MeanFlow due to jvp memory usage)
        default_batch = min(32, config.get("BATCH", 256) // 4)
        batch_size = config.get("BATCH_MEANFLOW", default_batch)
        self.batch_size = batch_size
        
        if self.is_main:
            print(f"[Multi-GPU] Using batch size {batch_size} per GPU (total effective: {batch_size * self.world_size})", flush=True)
        
        # Create distributed samplers
        self.train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True
        )
        
        self.val_sampler = None
        if val_dataset is not None:
            self.val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False
            )
        
        # Create data loaders with distributed samplers
        # Note: num_workers=0 to avoid issues with multiprocessing and NCCL
        self.loader_train = torchdata.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=self.train_sampler,
            pin_memory=True,
            num_workers=0,
            drop_last=True
        )
        
        self.loader_val = None
        if val_dataset is not None:
            self.loader_val = torchdata.DataLoader(
                val_dataset,
                batch_size=batch_size,
                sampler=self.val_sampler,
                pin_memory=True,
                num_workers=0,
                drop_last=False
            )
    
    def init_model(self):
        """Initialize the CaloDiffusion model WITHOUT DDP wrapper (for JVP compatibility)."""
        shape = self.config.get("SHAPE_PAD")
        if shape is None:
            shape = self.config.get("SHAPE_FINAL")
        
        # Create model on the correct device for this process
        self.model = CaloDiffusion(
            self.config,
            n_steps=self.config.get("NSTEPS", 400),
            loss_type=self.config.get("LOSS_TYPE", "l2")
        ).to(device=self.device)
        
        # Important: Do NOT wrap with DDP - JVP doesn't work with DDP hooks
        # Instead, we manually synchronize gradients after backward()
        if self.is_main:
            print(f"[Multi-GPU] Model initialized on device {self.device} (no DDP wrapper)", flush=True)
    
    def _sync_gradients(self):
        """Manually synchronize gradients across all processes using all-reduce."""
        if self.world_size <= 1:
            return
        
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size
    
    def _sync_model_params(self):
        """Broadcast model parameters from rank 0 to all other processes."""
        if self.world_size <= 1:
            return
        
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)
    
    def training_loop(
        self, optimizer, scheduler, early_stopper, start_epoch, num_epochs, training_losses, val_losses
    ):
        """Training loop for MeanFlow model with multi-GPU support (manual gradient sync)."""
        tqdm_func = utils.import_tqdm()
        cold_diffu = self.config.get("COLD_DIFFU", False)
        cold_noise_scale = self.config.get("COLD_NOISE", 1.0)
        energy_loss_scale = self.config.get("ENERGY_LOSS_SCALE", 0.0)
        weight_pidm = self.config.get("WEIGHT_PIDM", 0.03)
        MAX_GRAD_NORM = 1.0
        
        min_validation_loss = 99999.0
        
        # Ensure all processes start with the same model weights
        self._sync_model_params()
        
        for epoch in range(start_epoch, num_epochs):
            # Set epoch for distributed sampler (ensures different shuffle each epoch)
            self.train_sampler.set_epoch(epoch)
            if self.val_sampler is not None:
                self.val_sampler.set_epoch(epoch)
            
            if self.is_main:
                print(f"Beginning epoch {epoch}", flush=True)
            
            train_loss = torch.tensor(0.0, device=self.device)
            num_batches = torch.tensor(0, device=self.device)
            
            self.model.train()
            
            # Only show progress bar on main process
            loader_iter = tqdm_func(
                enumerate(self.loader_train, 0), 
                unit="batch", 
                total=len(self.loader_train)
            ) if self.is_main else enumerate(self.loader_train, 0)
            
            for i, (E, layers, data) in loader_iter:
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
                    # Account for rank in indexing to get different prior samples per GPU
                    global_batch_idx = (epoch * len(self.loader_train) + i) * self.world_size + self.rank
                    start_idx = (global_batch_idx * batch_size) % len(self.gmm_prior_data)
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
                
                # Compute loss - JVP works here because model is not wrapped with DDP
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
                
                # Backward pass - computes gradients locally on this GPU
                batch_loss.backward()
                
                # Manually synchronize gradients across all GPUs
                self._sync_gradients()
                
                # Gradient clipping
                grad_norm = clip_grad_norm_(self.model.parameters(), max_norm=MAX_GRAD_NORM)
                
                optimizer.step()
                train_loss += batch_loss.detach()
                num_batches += 1
                
                # Clean up GPU memory (JVP is memory-intensive)
                del data, E, layers, noise, batch_loss, loss_vec, loss_ref
                if prior is not None:
                    del prior
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Reduce training loss across all processes for logging
            train_loss = utils.reduce_tensor(train_loss, self.world_size)
            num_batches_total = utils.reduce_tensor(num_batches.float(), self.world_size)
            train_loss = (train_loss / num_batches_total).item()
            
            training_losses[epoch] = train_loss
            if self.is_main:
                print(f"loss: {train_loss}", flush=True)
            
            # Validation
            val_loss = torch.tensor(0.0, device=self.device)
            val_batches = torch.tensor(0, device=self.device)
            
            self.model.eval()
            if self.loader_val is not None:
                with torch.no_grad():
                    val_iter = tqdm_func(
                        enumerate(self.loader_val, 0),
                        unit="batch",
                        total=len(self.loader_val)
                    ) if self.is_main else enumerate(self.loader_val, 0)
                    
                    for i, (vE, vlayers, vdata) in val_iter:
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
                            global_val_batch_idx = (epoch * len(self.loader_val) + i) * self.world_size + self.rank
                            val_start_idx = (global_val_batch_idx * batch_size + len(self.loader_train) * self.batch_size) % len(self.gmm_prior_data)
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
                        val_loss += batch_loss.detach()
                        val_batches += 1
                        
                        del vdata, vE, vlayers, noise, batch_loss
                        if vprior is not None:
                            del vprior
                
                # Reduce validation loss across all processes
                val_loss = utils.reduce_tensor(val_loss, self.world_size)
                val_batches_total = utils.reduce_tensor(val_batches.float(), self.world_size)
                val_loss = (val_loss / val_batches_total).item()
                
                val_losses[epoch] = val_loss
                if self.is_main:
                    print(f"val_loss: {val_loss}", flush=True)
            
            scheduler.step(torch.tensor([train_loss]))
            
            # Only main process saves checkpoints
            if self.is_main:
                if val_loss < min_validation_loss:
                    if self.save_model:
                        torch.save(
                            self.model.state_dict(),
                            os.path.join(self.checkpoint_folder, "best_val.pth")
                        )
                    min_validation_loss = val_loss
                
                if early_stopper.early_stop(val_loss):
                    print("Early stopping!", flush=True)
                    # Broadcast early stop signal to all processes
                    early_stop_signal = torch.tensor([1], device=self.device)
                else:
                    early_stop_signal = torch.tensor([0], device=self.device)
                
                # Save checkpoint
                self.model.eval()
                print("SAVING", flush=True)
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
            else:
                early_stop_signal = torch.tensor([0], device=self.device)
            
            # Broadcast early stop signal
            if dist.is_initialized():
                dist.broadcast(early_stop_signal, src=0)
            
            if early_stop_signal.item() == 1:
                break
            
            # Synchronize all processes at end of epoch
            utils.barrier()
        
        return self.model, epoch, training_losses, val_losses, optimizer, scheduler, early_stopper
    
    def train(self):
        """Override train to add cleanup."""
        try:
            super().train()
        finally:
            # Clean up distributed environment
            utils.cleanup_ddp()


# Alias for backward compatibility
TrainMeanFlowDDP = TrainMeanFlowMultiGPU
