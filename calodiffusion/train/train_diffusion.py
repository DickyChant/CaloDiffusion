import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from calodiffusion.utils import utils
from calodiffusion.utils import distributed as dist_utils
from calodiffusion.train.train import Train
from calodiffusion.models.calodiffusion import CaloDiffusion


class TrainDiffusion(Train): 
    def __init__(self, flags, config, load_data=True, save_model:bool=True) -> None:
        super().__init__(flags, config, load_data=load_data, save_model=save_model)

    def init_model(self):
        self.model = CaloDiffusion(
            self.config, n_steps=self.config["NSTEPS"], loss_type=self.config['LOSS_TYPE']
        )
        self.model = self.model.to(self.device)
        
        # Wrap with DDP if enabled
        if self.use_ddp:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )
    
    def training_loop(self, optimizer, scheduler, early_stopper, start_epoch, num_epochs, training_losses, val_losses):
    
        tqdm = utils.import_tqdm()
        cold_diffu = self.config.get("COLD_DIFFU", False)
        cold_noise_scale = self.config.get("COLD_NOISE", 1.0)

        #fixed noise levels for  the validation loss for stability
        if(self.loader_val is not None):
            val_rnd = torch.randn( (len(self.loader_val)+1,self.batch_size,), device=self.device)
            print(val_rnd.shape)


        # training loop
        min_validation_loss = 99999.0
        epoch = start_epoch
        for epoch in range(start_epoch, num_epochs):
            # Set epoch for distributed sampler (ensures different shuffling per epoch)
            if self.use_ddp and self.sampler_train is not None:
                self.sampler_train.set_epoch(epoch)
            
            if dist_utils.is_main_process(self.rank):
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

                # Access nsteps from the model (unwrap DDP if needed)
                model_nsteps = self.model.module.nsteps if self.use_ddp else self.model.nsteps
                t = torch.randint(0, model_nsteps, (data.size()[0],), device=self.device).long()
                noise = torch.randn_like(data)

                if cold_diffu:  # cold diffusion interpolates from avg showers instead of pure noise
                    # Access gen_cold_image from the model (unwrap DDP if needed)
                    model_ref = self.model.module if self.use_ddp else self.model
                    noise = model_ref.gen_cold_image(E, cold_noise_scale, noise)

                # Access compute_loss from the model (unwrap DDP if needed)
                model_ref = self.model.module if self.use_ddp else self.model
                batch_loss = model_ref.compute_loss(
                    data=data, energy=E, noise=noise, layers=layers, time=t
                )
                batch_loss.backward()

                optimizer.step()
                train_loss += batch_loss.item()

                del data, E, layers, noise, batch_loss

            train_loss = train_loss / len(self.loader_train)
            
            # Reduce loss across all processes if using DDP
            if self.use_ddp:
                train_loss_tensor = torch.tensor(train_loss, device=self.device)
                train_loss_tensor = dist_utils.reduce_tensor(train_loss_tensor, self.world_size)
                train_loss = train_loss_tensor.item()
            
            training_losses[epoch] = train_loss

            if dist_utils.is_main_process(self.rank):
                print("loss: " + str(train_loss))

            val_loss = 0
            self.model.eval()
            if(self.loader_val is not None):
                # Set epoch for validation sampler too
                if self.use_ddp and self.sampler_val is not None:
                    self.sampler_val.set_epoch(epoch)
                
                for i, (vE, vlayers, vdata) in tqdm(
                    enumerate(self.loader_val, 0), unit="batch", total=len(self.loader_val)
                ):
                    #dumb fix
                    if(i >= val_rnd.shape[0]): break

                    vdata = vdata.to(device=self.device)
                    vE = vE.to(device=self.device)
                    vlayers = vlayers.to(device=self.device)


                    noise = torch.randn_like(vdata)

                    #use fixed time steps for stable val loss
                    rnd_normal = val_rnd[i].to(device=self.device)

                    #make sure shape of last batch handled properly
                    if(vE.shape[0] != self.batch_size):
                        rnd_normal = rnd_normal[:vE.shape[0]]

                    if cold_diffu:
                        model_ref = self.model.module if self.use_ddp else self.model
                        noise = model_ref.gen_cold_image(vE, cold_noise_scale, noise)

                    model_ref = self.model.module if self.use_ddp else self.model
                    batch_loss = model_ref.compute_loss(
                        vdata, vE, noise=noise, layers=vlayers, rnd_normal=rnd_normal,
                    )

                    val_loss += batch_loss.item()
                    del vdata, vE, vlayers, noise, batch_loss

                val_loss = val_loss / len(self.loader_val)
                
                # Reduce validation loss across all processes if using DDP
                if self.use_ddp:
                    val_loss_tensor = torch.tensor(val_loss, device=self.device)
                    val_loss_tensor = dist_utils.reduce_tensor(val_loss_tensor, self.world_size)
                    val_loss = val_loss_tensor.item()
                
                val_losses[epoch] = val_loss
                if dist_utils.is_main_process(self.rank):
                    print("val_loss: " + str(val_loss), flush=True)

            scheduler.step(torch.tensor([train_loss]))

            if val_loss < min_validation_loss:
                # Synchronize before saving
                if self.use_ddp:
                    torch.distributed.barrier()
                
                if self.save_model and dist_utils.is_main_process(self.rank):
                    # Get the underlying model state (unwrap DDP if needed)
                    model_state = self.model.module.state_dict() if self.use_ddp else self.model.state_dict()
                    torch.save(
                        model_state, os.path.join(self.checkpoint_folder, "best_val.pth")
                    )
                min_validation_loss = val_loss

            if early_stopper.early_stop(val_loss):
                if dist_utils.is_main_process(self.rank):
                    print("Early stopping!")
                break

            # save the model for each checkpoint
            # Synchronize before saving
            if self.use_ddp:
                torch.distributed.barrier()
            
            self.model.eval()
            if dist_utils.is_main_process(self.rank):
                print("SAVING")
            
            # Get the underlying model state (unwrap DDP if needed)
            model_state = self.model.module.state_dict() if self.use_ddp else self.model.state_dict()
            self.save(
                model_state,
                epoch=epoch,
                name="checkpoint",
                training_losses=training_losses,
                validation_losses=val_losses,
                optimizer=optimizer,
                scheduler=scheduler,
                early_stopper=early_stopper,
            )
            
        return self.model, epoch, training_losses, val_losses, optimizer, scheduler, early_stopper
