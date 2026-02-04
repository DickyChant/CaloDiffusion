import copy
import os
from typing import Union
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import NamedTuple
from scipy.ndimage import zoom
from torch.func import jvp
from einops import rearrange
from calodiffusion.models.diffusion import Diffusion
from calodiffusion.models.models import ResNet, CondUnet, PureDiT, MeanFlowDiT, MeanFlowDiT_v1, cosine_beta_schedule, extract
from calodiffusion.models import models as models_module
from calodiffusion.utils import sampling
from calodiffusion.utils import utils
from calodiffusion.utils.utils import ReverseNorm
import calodiffusion.utils.HGCal_utils as hgcal_utils

# Alias for CaloDiffu class compatibility
models = models_module

class CaloDiffusion(Diffusion): 
    def __init__(self, config: Union[str, dict], n_steps: int = 400, loss_type: str = 'l2'):
        super().__init__(config, n_steps, loss_type)
        self.pre_embed = "pre-embed" in self.config['SHOWER_EMBED']
        self.hgcal = self.config.get("HGCAL", False)

        self.fully_connected = "FCN" in self.config.get("SHOWER_EMBED", "")
        self.time_embed = self.config.get("TIME_EMBED", "sin")
        self.dataset_num = self.config.get("DATASET_NUM", 2)

        self.R_image, self.Z_image = utils.create_R_Z_image(
            self.device, dataset_num = self.dataset_num, scaled=True, shape=self.config["SHAPE_FINAL"][1:]
        )
        self.phi_image = utils.create_phi_image(self.device, shape=self.config["SHAPE_FINAL"][1:])
        self.training_objective = self.config.get("TRAINING_OBJ", "noise_pred")
        # Whether to append per-layer energy conditioning to the conditional vector.
        # Note: This is independent from whether SHOWERMAP includes "layer" (which controls
        # preprocessing and optional ReverseNorm layer renormalization).
        self.layer_cond = "layer" in self.config.get("SHOWERMAP", "")
        if self.config.get("DISABLE_LAYER_COND", False):
            self.layer_cond = False

        self.model = self.init_model()
        self.NN_embed = self.init_embedding_model()
        self.do_embed = self.NN_embed is not None and (not self.pre_embed)
        
        # Track if using MeanFlowDiT architecture (requires r parameter in forward)
        self.is_meanflow_dit = self.shower_embed in ("MFAttn", "MFAttn_v1")


    def load_state_dict(self, state_dict, strict = True):
        base_model_name = list(state_dict.keys())[10].split('.')[0]
        if base_model_name!="model": 
            state_dict = {
                key.removeprefix(f"{base_model_name}."): value for key, value in state_dict.items() if key.split('.')[0] == base_model_name
            }
        return super().load_state_dict(state_dict, strict)
    
    def init_model(self):
        """Initialize the network model based on SHOWER_EMBED configuration."""
        self.shower_embed = self.config.get("SHOWER_EMBED", "")
        self.fully_connected = "FCN" in self.shower_embed

        if self.fully_connected: 
            model = ResNet(
                cond_emb_dim=self.config["COND_SIZE_UNET"],
                dim_in=self.config["SHAPE_ORIG"][1],
                num_layers=self.config["NUM_LAYERS_LINEAR"],
                hidden_dim=512,
            ).to(device=self.device)

        else: 
            in_channels = 1
            if self.config.get("R_Z_INPUT", False):
                in_channels = 3

            if self.config.get("PHI_INPUT", False):
                in_channels += 1
            
            # Compute cond_size - use LEGACY_COND_SIZE for backward compatibility with old checkpoints
            if self.config.get("LEGACY_COND_SIZE", False):
                # Old checkpoints used only energy conditioning; disable layer conditioning
                cond_size = 1
                self.layer_cond = False
            elif self.config.get("DISABLE_LAYER_COND", False):
                # Keep SHOWERMAP="layer-..." for preprocessing/ReverseNorm, but do not condition the
                # network on layerE. Use only the generator-level conditioning (energy [+ angles]).
                cond_size = 1
                if self.hgcal:
                    cond_size += 2
                self.layer_cond = False
            else:
                cond_size = 2 + self.config["SHAPE_FINAL"][2] if "layer" in self.config.get("SHOWERMAP", "") else 1
                # extra conditioning info for hgcal
                if(self.hgcal): cond_size +=2
            calo_summary_shape = [1, in_channels] + list(copy.copy(self.config["SHAPE_FINAL"][1:]))

            # Select model architecture based on SHOWER_EMBED
            if "PuAttn" in self.shower_embed:
                # PureDiT backbone (used for DiT-style MeanFlow)
                patch_shape = self.config.get("SHAPE_PAD", self.config["SHAPE_FINAL"])[2:]
                model = PureDiT(
                    hidden_dim=self.config["COND_SIZE_UNET"],
                    in_dim=in_channels,
                    depth=self.config.get("NUM_LAYERS", 4),
                    num_heads=self.config.get("NUM_HEADS", 8),
                    patch_shape=patch_shape,
                    mlp_ratio=self.config.get("MLP_RATIO", 4.0),
                ).to(device=self.device)
                print(f"[CaloDiffusion] Initialized PureDiT with patch_shape={patch_shape}")

            elif "MFAttn" in self.shower_embed and "v1" not in self.shower_embed:
                # MeanFlowDiT architecture for MeanFlow with attention
                patch_shape = self.config.get("SHAPE_PAD", self.config["SHAPE_FINAL"])[2:]
                model = MeanFlowDiT(
                    hidden_dim=self.config["COND_SIZE_UNET"],
                    in_dim=in_channels,
                    depth=self.config.get("NUM_LAYERS", 4),
                    num_heads=self.config.get("NUM_HEADS", 8),
                    patch_shape=patch_shape,
                    mlp_ratio=self.config.get("MLP_RATIO", 4.0),
                    time_embed=(self.config.get("TIME_EMBED", "sin") == "sin"),
                    cond_embed=(self.config.get("COND_EMBED", "sin") == "sin"),
                ).to(device=self.device)
                print(f"[CaloDiffusion] Initialized MeanFlowDiT with patch_shape={patch_shape}")
                
            elif "MFAttn_v1" in self.shower_embed:
                # MeanFlowDiT_v1 architecture (alternate r embedding)
                patch_shape = self.config.get("SHAPE_PAD", self.config["SHAPE_FINAL"])[2:]
                model = MeanFlowDiT_v1(
                    hidden_dim=self.config["COND_SIZE_UNET"],
                    in_dim=in_channels,
                    depth=self.config.get("NUM_LAYERS", 4),
                    num_heads=self.config.get("NUM_HEADS", 8),
                    patch_shape=patch_shape,
                    mlp_ratio=self.config.get("MLP_RATIO", 4.0),
                    time_embed=(self.config.get("TIME_EMBED", "sin") == "sin"),
                    cond_embed=(self.config.get("COND_EMBED", "sin") == "sin"),
                ).to(device=self.device)
                print(f"[CaloDiffusion] Initialized MeanFlowDiT_v1 with patch_shape={patch_shape}")
                
            else:
                # Default: CondUnet architecture
                model = CondUnet(
                    cond_dim=self.config["COND_SIZE_UNET"],
                    out_dim=1,
                    channels=in_channels,
                    layer_sizes=self.config["LAYER_SIZE_UNET"],
                    block_attn=self.config.get("BLOCK_ATTN", False),
                    mid_attn= self.config.get("MID_ATTN", False),
                    cylindrical=self.config.get("CYLINDRICAL", False),
                    compress_Z=self.config.get("COMPRESS_Z", False),
                    resnet_block_groups=self.config.get("BLOCK_GROUPS", 8), 
                    data_shape=calo_summary_shape,
                    cond_embed=(self.config.get("COND_EMBED", "sin") == "sin"),
                    cond_size=cond_size,
                    time_embed=(self.config.get("TIME_EMBED", "sin") == "sin"),
                ).to(device=self.device)

        return model.to(self.device)

    def noise_generation(self, shape):
        return super().noise_generation(shape)

    def forward(self, x, E, time, layers=None, controls=None, r=None):
        """Forward pass through the model.
        
        Args:
            x: Input tensor
            E: Energy conditioning
            time: Time embedding
            layers: Optional layer conditioning
            controls: Optional control parameters
            r: Optional r parameter for MeanFlowDiT (flow progress)
        """
        if (self.do_embed):
            x = self.NN_embed.enc(x.to(torch.float32)).to(x.device)
        if (self.layer_cond) and (layers is not None):
            E = torch.cat([E, layers], dim=1)

        # Legacy checkpoints expect energy-only conditioning
        if self.config.get("LEGACY_COND_SIZE", False) and E is not None:
            if E.ndim == 2 and E.shape[1] > 1:
                E = E[:, :1]

        # PureDiT expects scalar conditioning; squeeze to (B,)
        if "PuAttn" in self.shower_embed and E is not None:
            if E.ndim > 1:
                E = E[:, 0]
        rz_phi = self.add_RZPhi(x).float()
        
        # MeanFlowDiT requires r parameter
        if self.is_meanflow_dit:
            # For MFDiT: model(data, cond=E, time=time, r=r)
            # If r is None, default to 0 (endpoint of flow)
            if r is None:
                r = torch.zeros_like(time)
            out = self.model(rz_phi, cond=E.float(), time=time.float(), r=r.float())
        else:
            # PureDiT/CondUnet may not accept controls; call safely
            import inspect
            sig = inspect.signature(self.model.forward)
            if "controls" in sig.parameters:
                out = self.model(rz_phi, cond=E.float(), time=time.float(), controls=controls)
            else:
                out = self.model(rz_phi, cond=E.float(), time=time.float())

        if (self.do_embed):
            out = self.NN_embed.dec(out).to(x.device)

        return out
    
    def init_embedding_model(self):
        dataset_num = self.config.get("DATASET_NUM", 2)
        shower_embed = self.config.get("SHOWER_EMBED", "")
        
        NN_embed = None
        if ("NN" in shower_embed and not self.hgcal):
            if dataset_num == 1:
                bins = utils.XMLHandler("photon", self.config["BIN_FILE"])
            else:
                bins = utils.XMLHandler("pion", self.config["BIN_FILE"])

            NN_embed = utils.NNConverter(bins=bins).to(device=self.device)

        elif(self.hgcal and not self.pre_embed):
            trainable = self.config.get('TRAINABLE_EMBED', False)
            NN_embed = hgcal_utils.HGCalConverter(bins = self.config['SHAPE_FINAL'], geom_file = self.config['BIN_FILE'], device = self.device, trainable = trainable).to(device = self.device)
            if not trainable: 
                NN_embed.init(norm = self.pre_embed, dataset_num = dataset_num)

        return NN_embed

    def add_RZPhi(self, x):

        if len(x.shape) < 3:
            return x
        cats = [x]
        const_shape = (x.shape[0], *((1,) * (len(x.shape) - 1)))
        target_shape = x.shape[1:]  # (C, L, H, W)

        if not self.fully_connected and self.config.get("R_Z_INPUT", False): 
            # Rebuild R/Z images if cached shapes don't match current input
            if tuple(self.R_image.shape) != tuple(target_shape):
                self.R_image, self.Z_image = utils.create_R_Z_image(
                    self.device,
                    dataset_num=self.dataset_num,
                    scaled=True,
                    shape=target_shape,
                )
            batch_R_image = self.R_image.repeat(const_shape).to(device=self.device)
            batch_Z_image = self.Z_image.repeat(const_shape).to(device=self.device)

            cats += [batch_R_image, batch_Z_image]

        if not self.fully_connected and self.config.get("PHI_INPUT", False):
            if tuple(self.phi_image.shape) != tuple(target_shape):
                self.phi_image = utils.create_phi_image(self.device, shape=target_shape)
            batch_phi_image = self.phi_image.repeat(const_shape).to(device=self.device)

            cats += [batch_phi_image]

        if len(cats) > 1:
            return torch.cat(cats, axis=1)
        else:
            return x

    def do_time_embed(
        self,
        sigma=None,
    ):
        embed: dict[str, callable] = {
            "sigma": lambda sigma: sigma / (1 + sigma**2).sqrt(), 
            "log": lambda sigma:  0.5 * torch.log(sigma)
        }
        return embed[self.time_embed](sigma)
    
    def denoise(self, x, E=None, sigma=None, layers=None, controls=None, r=None):
        """Denoise input x at noise level sigma.
        
        Args:
            x: Noisy input tensor
            E: Energy conditioning
            sigma: Noise level (time parameter for MeanFlow)
            layers: Optional layer conditioning
            controls: Optional control parameters
            r: Optional r parameter for MeanFlowDiT (flow progress)
        """
        t_emb = self.do_time_embed(sigma=sigma.reshape(-1)).to(float)
        loss_function_name = type(self.loss_function).__name__
        
        # For MeanFlowDiT, we need to pass r parameter
        # If r is not provided for MFDiT, derive it from sigma
        if self.is_meanflow_dit and r is None:
            # For MeanFlow ODE, r can be derived from or equal to time/sigma
            r = sigma.reshape(-1)

        scales = self.loss_function.get_scaling(sigma)
        # Ensure scaling tensors broadcast over spatial dims (sigma is (B,) for MeanFlow)
        if x.ndim > 1 and scales["c_in"].ndim == 1:
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            scales = {
                "c_in": scales["c_in"].reshape(shape),
                "c_skip": scales["c_skip"].reshape(shape),
                "c_out": scales["c_out"].reshape(shape),
            }
        pred = self.forward(x * scales['c_in'], E, t_emb, layers=layers, r=r)

        if('noise_pred' in loss_function_name):
            return (x - sigma * pred)

        elif('mean_pred' in loss_function_name):
            return pred
        elif ('hybrid' or 'minsnr') in loss_function_name:
            return (scales['c_skip'] * x + scales['c_out'] * pred)
        else:
            raise ValueError("??? Training obj %s" % loss_function_name)


    def __call__(self, x, **kwargs):
        return self.denoise(x, **kwargs)


# Helper functions and classes from CaloDiffu.py

def stopgrad(x):
    return x.detach()

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def _layer_sums(x, layer_dim=2):
    """
    x: (B, C, L, H, W) or (B, L, H, W)
    returns: (B, L)
    """
    if x.ndim == 5:   # (B, C, L, H, W)
        return x.sum(dim=(1, 3, 4)).squeeze(1)  # sum over C,H,W -> (B,L)
    elif x.ndim == 4: # (B, L, H, W)
        return x.sum(dim=(2, 3))                # sum over H,W   -> (B,L)
    else:
        raise ValueError(f"Unexpected shape for layer_sums: {x.shape}")

def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    if error.ndim == 4:
        # old: (B, C, H, W)
        delta_sq = torch.mean(error**2, dim=(1,2,3))
    elif error.ndim == 5:
        delta_sq = torch.mean(error**2, dim=(1, 2, 3, 4))
    elif error.ndim == 2:
        # new: (B, D)
        delta_sq = torch.mean(error**2, dim=1)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()   

def reverse_logit(x, alpha = 1e-6):
    exp = np.exp(x)    
    o = exp/(1+exp)
    o = (o-alpha)/(1 - 2*alpha)
    return o

def compute_diffusion(t_cur):
    return 2 * t_cur

def expand_t_like_x(t, x_cur):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x_cur.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

# helper for time tensor with same broadcast shape as training
def make_time_const_shape(bsz, like_tensor):
    const_shape = (bsz,) + (1,) * (like_tensor.ndim - 1)
    return const_shape

def get_score_from_velocity(vt, xt, t, path_type="linear"):
    """Wrapper function: transfrom velocity prediction model to score
    Args:
        velocity: [batch_dim, ...] shaped tensor; velocity model output
        x: [batch_dim, ...] shaped tensor; x_t data point
        t: [batch_dim,] time tensor
    """
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
        sigma_t, d_sigma_t = t, torch.ones_like(xt, device=xt.device)
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError

    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var

    return score

def _to_scalar_map_3d(x_np):
    """
    Convert a numpy tensor to a 3D scalar map (D,H,W).
    - If 3D: (D,H,W) -> as-is
    - If 4D: (C,D,H,W) -> L2-norm over channel -> (D,H,W)
    - If anything else: try last-dim L2 and then reshape to 3D if possible
    """
    if x_np.ndim == 3:
        return x_np
    if x_np.ndim == 4:  # (C,D,H,W)
        return np.linalg.norm(x_np, axis=0)
    # Fallback: treat last axis as "channel/features"
    if x_np.ndim >= 2:
        x_np = np.linalg.norm(x_np, axis=-1)
    # If still not 3D, try to flatten to match (D,H,W) of target later
    return x_np

def _per_layer_fraction_diff(A, B, eps=1e-12):
    """
    Given two 3D maps A,B with shape (D,H,W), compute per-layer fraction maps:
      frac_A[z] = A[z]/A[z].sum(), frac_B[z] = B[z]/B[z].sum()
    Return a list of 2D arrays: diff[z] = frac_A[z] - frac_B[z]
    Negative values are clipped to zero BEFORE fraction to avoid sign issues.
    """
    A = np.clip(A, 0, None)
    B = np.clip(B, 0, None)
    assert A.shape == B.shape and A.ndim == 3, "A and B must both be (D,H,W)"

    D = A.shape[0]
    diffs = []
    for z in range(D):
        Az = A[z]
        Bz = B[z]
        Az_sum = Az.sum() + eps
        Bz_sum = Bz.sum() + eps
        fracA = Az / Az_sum
        fracB = Bz / Bz_sum
        diffs.append(fracA - fracB)
    return diffs


class CaloDiffu(nn.Module):
    """Diffusion based generative model"""
    
    def _parallel_jvp(self, fn, primals, tangents, energy, layers=None):
        """
        Manual multi-GPU jvp: split batch across GPUs to reduce memory usage.
        jvp doesn't work with DataParallel, but we can manually split the batch.
        See: https://github.com/pytorch/pytorch/issues/102197
        
        Args:
            fn: Function that takes (z, cur_r, cur_t, energy_chunk, layers_chunk)
            primals: Tuple of (z_t, r, t)
            tangents: Tuple of (v_t, zeros_like(r), ones_like(t))
            energy: Energy tensor for conditioning
            layers: Optional layers tensor
        """
        z_t, r, t = primals
        v_t, _, _ = tangents
        batch_size = z_t.shape[0]
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        original_device = z_t.device
        
        # Only split if explicitly enabled (multi-GPU jvp can cause device mismatches)
        enable_multi_gpu_jvp = os.environ.get("MF_JVP_MULTI_GPU", "0") == "1"
        if enable_multi_gpu_jvp and num_gpus > 1 and batch_size > 4:
            # Split batch across GPUs
            chunk_size = max(batch_size // num_gpus, 1)  # At least 1 per GPU
            
            u_chunks = []
            dudt_chunks = []
            
            for i in range(num_gpus):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, batch_size)
                if start_idx >= batch_size:
                    break
                    
                device_i = torch.device(f"cuda:{i}")
                
                # Move chunk to GPU i
                z_t_chunk = z_t[start_idx:end_idx].to(device_i)
                r_chunk = r[start_idx:end_idx].to(device_i)
                t_chunk = t[start_idx:end_idx].to(device_i)
                v_t_chunk = v_t[start_idx:end_idx].to(device_i)
                energy_chunk = energy[start_idx:end_idx].to(device_i)
                layers_chunk = layers[start_idx:end_idx].to(device_i) if layers is not None else None
                
                # Temporarily move model to this GPU for jvp
                original_model_device = next(self.parameters()).device
                self.to(device_i)
                
                primals_chunk = (z_t_chunk, r_chunk, t_chunk)
                tangents_chunk = (v_t_chunk, torch.zeros_like(r_chunk), torch.ones_like(t_chunk))
                
                def fn_chunk(z, cur_r, cur_t):
                    return fn(z, cur_r, cur_t, energy_chunk, layers_chunk)
                
                u_chunk, dudt_chunk = jvp(fn_chunk, primals_chunk, tangents_chunk)
                
                # Move results back to original device
                u_chunks.append(u_chunk.to(original_device))
                dudt_chunks.append(dudt_chunk.to(original_device))
                
                # Move model back
                self.to(original_model_device)
            
            # Concatenate results
            u = torch.cat(u_chunks, dim=0)
            dudt = torch.cat(dudt_chunks, dim=0)
            return u, dudt
        else:
            # Single GPU or small batch - use original approach
            # For single GPU, fn should work with full energy/layers
            def fn_single(z, cur_r, cur_t):
                return fn(z, cur_r, cur_t, energy, layers)
            return jvp(fn_single, primals, tangents)
    def __init__(self, data_shape, config=None, R_Z_inputs = False, training_obj = 'noise_pred', nsteps = 400,
                    cold_diffu = False, E_bins = None, avg_showers = None, std_showers = None, NN_embed = None):
        super(CaloDiffu, self).__init__()
        self._data_shape = data_shape
        self.nvoxels = np.prod(self._data_shape)
        self.config = config
        self._num_embed = self.config['EMBED']
        self.num_heads=1
        self.nsteps = nsteps
        self.cold_diffu = cold_diffu
        self.E_bins = E_bins
        self.avg_showers = avg_showers
        self.std_showers = std_showers
        self.training_obj = training_obj
        self.shower_embed = self.config.get('SHOWER_EMBED', '')
        self.fully_connected = ('FCN' in self.shower_embed)
        self.puredit = ('PuAttn' in self.shower_embed)
        self.meanflow = (
            ('MFAttn' in self.shower_embed) or
            ('MFAttn_v1' in self.shower_embed) or
            ('MFUnet' in self.shower_embed) or
            ('MFUnet_v2' in self.shower_embed) or
            ('MFDiC' in self.shower_embed) or
            ('MF' in self.shower_embed)  # Allow any MF prefix for MeanFlow training
        )
        self.NN_embed = NN_embed
        self.restart_info = self.config.get('restart_info', '')
        self.total_generated_samples = 0  # To keep track of total generated samples
        self.total_sampling_time = 0.0    # To keep track of total sampling time
        


        

        supported = ['noise_pred', 'mean_pred', 'hybrid']
        is_obj = [s in self.training_obj for s in supported]
        if(not any(is_obj)):
            print("Training objective %s not supported!" % self.training_obj)
            exit(1)


        if config is None:
            raise ValueError("Config file not given")
        
        self.verbose = 1

        
        if(torch.cuda.is_available()): 
            device = torch.device('cuda')
        else: 
            device = torch.device('cpu')
        self.device = device

        #Minimum and maximum maximum variance of noise
        self.beta_start = 0.0001
        self.beta_end = config.get("BETA_MAX", 0.02)

        #linear schedule
        schedd = config.get("NOISE_SCHED", "linear")
        self.discrete_time = True

        
        if("linear" in schedd): self.betas = torch.linspace(self.beta_start, self.beta_end, self.nsteps)
        elif("cosine" in schedd): 
            self.betas = models.cosine_beta_schedule(self.nsteps)
        elif("log" in schedd):
            self.discrete_time = False
            self.P_mean = -1.2
            self.P_std = 1.2
            self.sigma_data = 0.5
            self.scales = 1
        else:
            print("Invalid NOISE_SCHEDD param %s" % schedd)
            exit(1)

        if(self.discrete_time):
            #precompute useful quantities for training
            self.alphas = 1. - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, axis = 0)

            #shift all elements over by inserting unit value in first place
            alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

            self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

            self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.time_embed = config.get("TIME_EMBED", 'sin')
        self.E_embed = config.get("COND_EMBED", 'sin')
        cond_dim = config['COND_SIZE_UNET']
        layer_sizes = config['LAYER_SIZE_UNET']
        block_attn = config.get("BLOCK_ATTN", False)
        mid_attn = config.get("MID_ATTN", False)
        compress_Z = config.get("COMPRESS_Z", False)
        
        if('layer' in config.get('SHOWERMAP', '')): 
            self.layer_cond = True
            #gen energy + total deposited energy + layer energy fractions
            cond_size = 2 + config['SHAPE_PAD'][2]
        else: 
            self.layer_cond = False
            cond_size = 1


        if(self.fully_connected):
            #fully connected network architecture
            self.model = models.FCN(cond_dim = cond_dim, dim_in = config['SHAPE_ORIG'][1], num_layers = config['NUM_LAYERS_LINEAR'],
                    cond_embed = (self.E_embed == 'sin'), time_embed = (self.time_embed == 'sin') )

            self.R_Z_inputs = False

            summary_shape = [[1,config['SHAPE_ORIG'][1]], [1], [1]]

        RZ_shape = config['SHAPE_PAD'][1:]
        self.R_Z_inputs = config.get('R_Z_INPUT', False)
        self.phi_inputs = config.get('PHI_INPUT', False)

        # Set input channels based on options
        in_channels = 1
        if self.R_Z_inputs:
            in_channels = 3
        if self.phi_inputs:
            in_channels += 1

        # Create RZ and phi images
        self.R_image, self.Z_image = utils.create_R_Z_image(self.device, dataset_num=config.get('DATASET_NUM', 2), scaled=True, shape=RZ_shape)
        self.phi_image = utils.create_phi_image(self.device, shape=RZ_shape)

        # Define shape configurations
        calo_summary_shape = [1] + [in_channels] + list(RZ_shape)
        summary_shape = [calo_summary_shape, [1], [1]]

        # Select model based on conditions
        if self.puredit:
            self.model = models.PureDiT(
                hidden_dim=cond_dim, 
                in_dim=in_channels, 
                depth=config['NUM_LAYERS'], 
                num_heads=config['NUM_HEADS'],
                patch_shape=config['SHAPE_PAD'][2:],
                mlp_ratio=config['MLP_RATIO'], 
            )
        elif self.meanflow:
            if self.shower_embed == 'MFAttn':
                    self.model = MeanFlowDiT(
                        hidden_dim=cond_dim, 
                        in_dim=in_channels, 
                        depth=config['NUM_LAYERS'], 
                        num_heads=config['NUM_HEADS'],
                        patch_shape=config['SHAPE_PAD'][2:],
                        mlp_ratio=config['MLP_RATIO'], 
                    )
            if self.shower_embed == 'MFAttn_v1':
                self.model = MeanFlowDiT_v1(
                    hidden_dim=cond_dim, 
                    in_dim=in_channels, 
                    depth=config['NUM_LAYERS'], 
                    num_heads=config['NUM_HEADS'],
                    patch_shape=config['SHAPE_PAD'][2:],
                    mlp_ratio=config['MLP_RATIO'], 
                )
                print(self.model)
            if self.shower_embed == 'MFUnet':
                self.model = MeanFlowCondUnet(
                    cond_dim=cond_dim, 
                    out_dim=1, 
                    channels=in_channels, 
                    layer_sizes=layer_sizes, 
                    block_attn=block_attn, 
                    mid_attn=mid_attn,
                    cylindrical=config.get('CYLINDRICAL', False), 
                    compress_Z=compress_Z, 
                    data_shape=calo_summary_shape,
                    cond_embed = (self.E_embed == 'sin'), #cond_size = cond_size,
                    time_embed=(self.time_embed == 'sin')
                )
            if self.shower_embed == 'MFUnet_v2':
                self.model = MeanFlowCondUnet_v2(
                    cond_dim=cond_dim, 
                    out_dim=1, 
                    channels=in_channels, 
                    layer_sizes=layer_sizes, 
                    block_attn=block_attn, 
                    mid_attn=mid_attn,
                    cylindrical=config.get('CYLINDRICAL', False), 
                    compress_Z=compress_Z, 
                    data_shape=calo_summary_shape,
                    cond_embed = (self.E_embed == 'sin'), #cond_size = cond_size,
                    time_embed=(self.time_embed == 'sin')
                )
            if self.shower_embed == 'MFDiC':
                self.model = MeanFlowCondDiC(
                    cond_dim=cond_dim, 
                    out_dim=1, 
                    channels=in_channels, 
                    layer_sizes=layer_sizes, 
                    block_attn=block_attn, 
                    mid_attn=mid_attn,
                    cylindrical=config.get('CYLINDRICAL', False), 
                    compress_Z=compress_Z, 
                    data_shape=calo_summary_shape,
                    cond_embed = (self.E_embed == 'sin'), #cond_size = cond_size,
                    time_embed=(self.time_embed == 'sin')
                )
                print(self.model)
            # If MeanFlow is enabled but no specific backbone matched, use regular CondUnet
            # This allows MeanFlow training with CondUnet backbone (e.g., SHOWER_EMBED="MF-condunet")
            if not hasattr(self, 'model') or self.model is None:
                self.model = models.CondUnet(
                    cond_dim=cond_dim, 
                    out_dim=1, 
                    channels=in_channels, 
                    layer_sizes=layer_sizes, 
                    block_attn=block_attn, 
                    mid_attn=mid_attn,
                    cylindrical=config.get('CYLINDRICAL', False), 
                    compress_Z=compress_Z, 
                    data_shape=calo_summary_shape,
                    cond_embed = (self.E_embed == 'sin'), #cond_size = cond_size,
                    time_embed=(self.time_embed == 'sin')
                )
        else:
            self.model = models.CondUnet(
                cond_dim=cond_dim, 
                out_dim=1, 
                channels=in_channels, 
                layer_sizes=layer_sizes, 
                block_attn=block_attn, 
                mid_attn=mid_attn,
                cylindrical=config.get('CYLINDRICAL', False), 
                compress_Z=compress_Z, 
                data_shape=calo_summary_shape,
                cond_embed = (self.E_embed == 'sin'), #cond_size = cond_size,
                time_embed=(self.time_embed == 'sin')
            )

        #print("\n\n Model: \n")
        #summary(self.model, summary_shape)

    #wrapper for backwards compatability
    def load_state_dict(self, d):
        if('noise_predictor' in list(d.keys())[0]):
            d_new = dict()
            for key in d.keys():
                key_new = key.replace('noise_predictor', 'model')
                d_new[key_new] = d[key]
        else: d_new = d

        return super().load_state_dict(d_new)

    def add_RZPhi(self, x):
        cats = [x]
        if(self.R_Z_inputs):

            batch_R_image = self.R_image.repeat([x.shape[0], 1,1,1,1]).to(device=x.device)
            batch_Z_image = self.Z_image.repeat([x.shape[0], 1,1,1,1]).to(device=x.device)

            cats+= [batch_R_image, batch_Z_image]
        if(self.phi_inputs):
            batch_phi_image = self.phi_image.repeat([x.shape[0], 1,1,1,1]).to(device=x.device)

            cats += [batch_phi_image]

        if(len(cats) > 2):
            cats[0] = cats[0].reshape(-1, *cats[1].shape[1:])
            return torch.cat(cats, axis = 1)
        else: 
            return x
            
    
    def lookup_avg_std_shower(self, inputEs):
        idxs = torch.bucketize(inputEs, self.E_bins)  - 1 #NP indexes bins starting at 1 
        return self.avg_showers[idxs], self.std_showers[idxs]

    
    def noise_image(self, data = None, t = None, noise = None):

        if(noise is None): noise = torch.randn_like(data)

        if(t[0] <=0): return data

        if(self.discrete_time):
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, data.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, data.shape)
            out = sqrt_alphas_cumprod_t * data + sqrt_one_minus_alphas_cumprod_t * noise
            return out
        else:
            print("non discrete time not supported")
            exit(1)
            
            
            
    def set_sampling_steps(self, nsteps):
        self.nsteps = nsteps
        #precompute useful quantities for sampling
        self.betas = models.cosine_beta_schedule(self.nsteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis = 0)

        #shift all elements over by inserting unit value in first place
        self.alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


    def compute_loss(self, data, energy, noise = None, t = None, layers = None, loss_type = "l2", rnd_normal = None, energy_loss_scale = 1e-2):
        if noise is None:
            noise = torch.randn_like(data)

        if(self.discrete_time): 
            if(t is None): t = torch.randint(0, self.nsteps, (data.size()[0],), device=data.device).long()
            x_noisy = self.noise_image(data, t, noise=noise)
            sigma = None
            sigma2 = extract(self.sqrt_one_minus_alphas_cumprod, t, data.shape)**2
        else:
            if(rnd_normal is None): rnd_normal = torch.randn((data.size()[0],), device=data.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            x_noisy = data + torch.reshape(sigma, (data.shape[0], 1,1,1,1)) * noise
            sigma2 = sigma**2



        t_emb = self.do_time_embed(t, self.time_embed, sigma)


        pred = self.pred(x_noisy, energy, t_emb)

        weight = 1.
        x0_pred = None
        if('hybrid' in self.training_obj ):

            c_skip = torch.reshape(1. / (sigma2 + 1.), (data.shape[0], 1,1,1,1))
            c_out = torch.reshape(1./ (1. + 1./sigma2).sqrt(), (data.shape[0], 1,1,1,1))
            weight = torch.reshape(1. + (1./ sigma2), (data.shape[0], 1,1,1,1))

            #target = (data - c_skip * x_noisy)/c_out


            x0_pred = pred = c_skip * x_noisy + c_out * pred
            target = data

        elif('noise_pred' in self.training_obj):
            target = noise
            weight = 1.
            if('energy' in self.training_obj): 
                sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, data.shape)
                sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, data.shape)
                x0_pred = (x_noisy - sqrt_one_minus_alphas_cumprod_t * pred)/sqrt_alphas_cumprod_t
        elif('mean_pred' in self.training_obj):
            target = data
            weight = 1./ sigma2
            x0_pred = pred


        if loss_type == 'l1':
            loss = torch.nn.functional.l1_loss(target, pred)
        elif loss_type == 'l2':
            if('weight' in self.training_obj):
                loss = (weight * ((pred - data) ** 2)).sum() / (torch.mean(weight) * self.nvoxels)
            else:
                loss = torch.nn.functional.mse_loss(target, pred)
        elif loss_type == "huber":
            loss =torch.nn.functional.smooth_l1_loss(target, pred)
        else:
            raise NotImplementedError()

        if('energy' in self.training_obj):
            #sum total energy
            dims = [i for i in range(1,len(data.shape))]
            tot_energy_pred = torch.sum(x0_pred, dim = dims)
            tot_energy_data = torch.sum(data, dim = dims)
            loss_en = energy_loss_scale * torch.nn.functional.mse_loss(tot_energy_data, tot_energy_pred) / self.nvoxels
            loss += loss_en

        return loss
    
    
    
    #utils function for min snr weighting
    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def weighting_soft_min_snr(self, sigma, k = 2):
        return (sigma * self.sigma_data) ** 2 / (sigma ** 2 + self.sigma_data ** k) ** 2

    #loss function for min snr weighting adopted from the above styles    
    def compute_loss_karras(self, data, energy, noise = None, t = None, layers = None, rnd_normal = None, energy_loss_scale = 1e-2, scales=1):
        self.scales = scales
        const_shape = (data.shape[0], *((1,) * (len(data.shape) - 1)))
        if noise is None:
            noise = torch.randn_like(data)
        if(rnd_normal is None): rnd_normal = torch.randn((data.size()[0],), device=data.device)
        sigma0 = (rnd_normal * self.P_std + self.P_mean).exp()
        sigma = torch.reshape((rnd_normal * self.P_std + self.P_mean).exp(),const_shape)
        
        
        
        x_noisy = data + torch.reshape(sigma, const_shape) * noise
        sigma2 = sigma**2
        

        c_skip, c_out, c_in = self.get_scalings(sigma)
        c_weight = self.weighting_soft_min_snr(sigma0)

        x_pred = self.pred(x_noisy*c_in, energy, sigma0, layers = layers)
        target = (data - c_skip * x_noisy) / c_out

        
        if('hybrid_weight_karras' in self.training_obj ) and (self.scales == 1):
            loss = (((x_pred - target) ** 2).flatten(1).mean(1) * c_weight).sum()
        elif ('hybrid_weight_karras_scaled' in self.training_obj ) and (self.scales != 1):
            sq_error = dct(model_output - target) ** 2
            f_weight = freq_weight_nd(sq_error.shape[2:], self.scales, dtype=sq_error.dtype, device=sq_error.device)
            loss = ((sq_error * f_weight).flatten(1).mean(1) * c_weight).sum()
                                 
            
            
        return loss
    
    #loss function for min snr weighting adopted from the above styles    
    def compute_loss_dino(self, data, energy, zs=None, noise = None, t = None, layers = None, rnd_normal = None, energy_loss_scale = 1e-2, scales=1):
        self.scales = scales
        const_shape = (data.shape[0], *((1,) * (len(data.shape) - 1)))
        if noise is None:
            noise = torch.randn_like(data)
        if(rnd_normal is None): rnd_normal = torch.randn((data.size()[0],), device=data.device)
        sigma0 = (rnd_normal * self.P_std + self.P_mean).exp()
        sigma = torch.reshape((rnd_normal * self.P_std + self.P_mean).exp(),const_shape)
        
        
        
        x_noisy = data + torch.reshape(sigma, const_shape) * noise
        sigma2 = sigma**2
        

        c_skip, c_out, c_in = self.get_scalings(sigma)
        c_weight = self.weighting_soft_min_snr(sigma0)

        x_pred, zs_tilde = self.pred(x_noisy*c_in, energy, sigma0, layers = layers, return_patch=False, encoder_patch=True)
        
        #print(zs_tilde.shape)
        #print(zs.shape)
        
        target = (data - c_skip * x_noisy) / c_out
        
        if('hybrid_weight_karras' in self.training_obj ) and (self.scales == 1):
            diffusion_loss = (((x_pred - target) ** 2).flatten(1).mean(1) * c_weight).sum()
        elif ('hybrid_weight_karras_scaled' in self.training_obj ) and (self.scales != 1):
            sq_error = dct(model_output - target) ** 2
            f_weight = freq_weight_nd(sq_error.shape[2:], self.scales, dtype=sq_error.dtype, device=sq_error.device)
            diffusion_loss = ((sq_error * f_weight).flatten(1).mean(1) * c_weight).sum()
        proj_loss = 0.                         

        B, N, D = zs.shape

        # zs: (B, N, 32)

        # normalize per token
        #zs       = torch.nn.functional.normalize(zs,       dim=-1)
        #zs_tilde = torch.nn.functional.normalize(zs_tilde, dim=-1)

        # negative cosine per token, then mean over tokens and batch
        #proj_loss = (-(zs * zs_tilde).sum(dim=-1)).mean()
        
        
        proj_loss = torch.tensor(0., device=zs.device)
        zs = [zs]
        bsz = zs[0].shape[0]  # = B
        #print(zs[0].shape)

        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):      
            for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)): 
                z_tilde_j = F.normalize(z_tilde_j, dim=-1)  # (D,)
                z_j       = F.normalize(z_j,       dim=-1)  # (D,)

                # negative cosine for this token, reduce to scalar via your mean_flat
                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))

        # average over all (B * N) terms
        proj_loss /= (len(zs) * bsz)
     
            
        return diffusion_loss, proj_loss
    
    
    def interpolant(self, t, path_type='linear'):
        if path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
    def compute_loss_sit(self, data, energy, noise = None, layers = None, weighting = 'uniform', energy_loss_scale = 1e-2, scales=1):
        self.scales = scales
        device = data.device
        dtype  = data.dtype

        if noise is None:
            noise = torch.randn_like(data)

        # sample t
        const_shape = (data.shape[0], *((1,) * (len(data.shape) - 1)))
        if weighting == "uniform":
            time_input = torch.rand(const_shape, device=device, dtype=dtype)
        elif weighting == "lognormal":
            # log-normal over sigma (EDM-style); clamp to avoid tails exploding
            rnd_normal = torch.randn(const_shape, device=device, dtype=dtype)
            sigma = rnd_normal.exp().clamp(1e-3, 1e3)
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
        elif self.path_type == "cosine":
                time_input = (2.0 / torch.pi) * torch.atan(sigma)
        else:
            raise ValueError(f"Unknown weighting: {weighting}")

        # interpolant
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)

        # model IO
        model_input = alpha_t * data + sigma_t * noise
        model_output, zs_tilde = self.pred(
            model_input, energy, time_input, layers=layers,
            return_patch=False, encoder_patch=True
        )

        # target (SiT/DDPM-style velocity/score target)
        model_target = d_alpha_t * data + d_sigma_t * noise
        diffusion_loss = mean_flat((model_output - model_target) ** 2).mean()

        # Optional PIDM regularizer (consistent with meanflow PIDM path)
        loss_ref = torch.tensor(0.0, device=device)
        loss_pidm_vec = None
        weight_pidm = float(self.config.get("WEIGHT_PIDM", 0.0) or 0.0)
        if weight_pidm > 0.0:
            pidm_steps = int(self.config.get("PIDM_SAMPLE_STEPS", 10))
            pidm_algo = self.config.get("PIDM_SAMPLE_ALGO", "sit")
            with torch.no_grad():
                # Use SiT sampler to generate samples for PIDM
                sample_data, _, _ = self.sit_sampler(
                    noise, energy, layers=layers, num_steps=pidm_steps, sample_algo=pidm_algo
                )

            dataset_config = self.config
            is_hgcal = dataset_config.get("HGCAL", False) or dataset_config.get("DATASET_NUM", 0) == 2

            # Convert to numpy for ReverseNorm
            real_std_np = data.detach().cpu().numpy()
            gen_std_np = sample_data.detach().cpu().numpy()
            E_std_np = energy.detach().cpu().numpy()
            if E_std_np.ndim == 1:
                E_std_np = E_std_np.reshape(-1, 1)

            shower_map = dataset_config.get("SHOWERMAP", "")
            layerE_np = None
            if layers is not None:
                layerE_np = layers.detach().cpu().numpy()
                if layerE_np.ndim == 1:
                    layerE_np = layerE_np.reshape(-1, 1)
            elif "layer" in shower_map:
                batch_size = E_std_np.shape[0]
                num_layers = 47
                layerE_np = np.zeros((batch_size, 1 + num_layers))
                layerE_np[:, 0] = E_std_np[:, 0]
                layerE_np[:, 1:] = 0.0

            shape_key = dataset_config.get("SHAPE_ORIG") or dataset_config.get("SHAPE") or dataset_config.get("SHAPE_PAD")
            emin_val = dataset_config["EMIN"]
            emax_val = dataset_config["EMAX"]
            if isinstance(emin_val, (list, tuple, np.ndarray)):
                emin_val = float(emin_val[0]) if len(emin_val) > 0 else float(emin_val)
            else:
                emin_val = float(emin_val)
            if isinstance(emax_val, (list, tuple, np.ndarray)):
                emax_val = float(emax_val[0]) if len(emax_val) > 0 else float(emax_val)
            else:
                emax_val = float(emax_val)

            real_phys, _ = ReverseNorm(
                real_std_np,
                E_std_np,
                hgcal=is_hgcal,
                layerE=layerE_np,
                shape=shape_key,
                logE=dataset_config["logE"],
                max_deposit=dataset_config["MAXDEP"],
                emax=emax_val,
                emin=emin_val,
                showerMap=dataset_config["SHOWERMAP"],
                dataset_num=dataset_config["DATASET_NUM"],
                orig_shape=False,
                ecut=dataset_config["ECUT"],
            )

            gen_phys, _ = ReverseNorm(
                gen_std_np,
                E_std_np,
                hgcal=is_hgcal,
                layerE=layerE_np,
                shape=shape_key,
                logE=dataset_config["logE"],
                max_deposit=dataset_config["MAXDEP"],
                emax=emax_val,
                emin=emin_val,
                showerMap=dataset_config["SHOWERMAP"],
                dataset_num=dataset_config["DATASET_NUM"],
                orig_shape=False,
                ecut=dataset_config["ECUT"],
            )

            real_phys_tensor = torch.from_numpy(real_phys).to(device=device, dtype=data.dtype)
            gen_phys_tensor = torch.from_numpy(gen_phys).to(device=device, dtype=data.dtype)
            if real_phys_tensor.ndim == 4:
                real_phys_tensor = real_phys_tensor.unsqueeze(1)
            elif real_phys_tensor.ndim == 2:
                real_phys_tensor = real_phys_tensor.reshape(data.shape)
            if gen_phys_tensor.ndim == 4:
                gen_phys_tensor = gen_phys_tensor.unsqueeze(1)
            elif gen_phys_tensor.ndim == 2:
                gen_phys_tensor = gen_phys_tensor.reshape(data.shape)

            S_true = _layer_sums(real_phys_tensor, layer_dim=2)
            S_sample = _layer_sums(gen_phys_tensor, layer_dim=2)
            raw_pidm = ((S_sample - S_true.detach()) ** 2).mean(dim=1)
            loss_pidm_vec = raw_pidm

        return diffusion_loss, loss_ref, loss_pidm_vec
        
    def compute_loss_dino_sit(self, data, energy, zs=None, noise = None, t = None, layers = None, weighting = 'uniform', energy_loss_scale = 1e-2, scales=1):
        self.scales = scales
        device = data.device
        dtype  = data.dtype

        if noise is None:
            noise = torch.randn_like(data)

        # sample t
        const_shape = (data.shape[0], *((1,) * (len(data.shape) - 1)))
        if weighting == "uniform":
            time_input = torch.rand(const_shape, device=device, dtype=dtype)
        elif weighting == "lognormal":
            # log-normal over sigma (EDM-style); clamp to avoid tails exploding
            rnd_normal = torch.randn(const_shape, device=device, dtype=dtype)
            sigma = rnd_normal.exp().clamp(1e-3, 1e3)
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
        elif self.path_type == "cosine":
                time_input = (2.0 / torch.pi) * torch.atan(sigma)
        else:
            raise ValueError(f"Unknown weighting: {weighting}")

        # interpolant
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)

        # model IO
        model_input = alpha_t * data + sigma_t * noise
        model_output, zs_tilde = self.pred(
            model_input, energy, time_input, layers=layers,
            return_patch=False, encoder_patch=True
        )

        # target (SiT/DDPM-style velocity/score target)
        model_target = d_alpha_t * data + d_sigma_t * noise
        diffusion_loss = mean_flat((model_output - model_target) ** 2).mean()
        
        proj_loss = 0.                         

        B, N, D = zs.shape

        
        
        proj_loss = torch.tensor(0., device=zs.device)
        zs = [zs]
        bsz = zs[0].shape[0]  # = B
        #print(zs[0].shape)

        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):      
            for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)): 
                z_tilde_j = F.normalize(z_tilde_j, dim=-1)  # (D,)
                z_j       = F.normalize(z_j,       dim=-1)  # (D,)

                # negative cosine for this token, reduce to scalar via your mean_flat
                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))

        # average over all (B * N) terms
        proj_loss /= (len(zs) * bsz)
     
            
        return diffusion_loss, proj_loss
     
    def eval_patch_karras(self, data, energy, noise = None, t = None, layers = None, rnd_normal = None, energy_loss_scale = 1e-2, scales=1):
        self.scales = scales
        const_shape = (data.shape[0], *((1,) * (len(data.shape) - 1)))
        if noise is None:
            noise = torch.randn_like(data)
        if(rnd_normal is None): rnd_normal = torch.randn((data.size()[0],), device=data.device)
        sigma0 = (rnd_normal * self.P_std + self.P_mean).exp()
        sigma = torch.reshape((rnd_normal * self.P_std + self.P_mean).exp(),const_shape)
        
        
        
        x_noisy = data + torch.reshape(sigma, const_shape) * noise
        sigma2 = sigma**2
        

        c_skip, c_out, c_in = self.get_scalings(sigma)
        c_weight = self.weighting_soft_min_snr(sigma0)

        patch_pred = self.pred(x_noisy*c_in, energy, sigma0, layers = layers, return_patch=True)
        target = (data - c_skip * x_noisy) / c_out
        
        # ---- 1) Build patch_map (10,10,10) from patch_pred ----
        pp = patch_pred.detach().float().cpu().numpy()   # (16, 1000, 64)
        pp0 = pp[0]                                      # (1000, 64)
        patch_norm = np.linalg.norm(pp0, axis=-1)        # (1000,)
        patch_map = patch_norm.reshape(10, 10, 10)       # (10,10,10)

        # ---- 2) Prepare target & E_ori (zoom to 10x10x10) ----
        # target: pick sample 0, channel 0 (assumed), then zoom
        target_np = target[0, 0].detach().float().cpu().numpy()   # (Dz, Dy, Dx)
        E_ori_np  = data[0, 0].detach().float().cpu().numpy()     # (Dz, Dy, Dx)

        Dz_t, Dy_t, Dx_t = target_np.shape
        Dz_e, Dy_e, Dx_e = E_ori_np.shape

        # target -> (10,10,10)
        tz = 10.0 / Dz_t
        ty = 10.0 / Dy_t
        tx = 10.0 / Dx_t
        target_map = zoom(target_np, (tz, ty, tx), order=1)

        # E_ori -> (10,10,10)
        ez = 10.0 / Dz_e
        ey = 10.0 / Dy_e
        ex = 10.0 / Dx_e
        E_up = zoom(E_ori_np, (ez, ey, ex), order=1)

        print("Shapes -> patch_map:", patch_map.shape, " target_map:", target_map.shape, " E_up:", E_up.shape)

        # ---- 3) Fraction helpers ----
        def per_layer_fraction_maps(A, eps=1e-12):
            """Return list of per-Z fractional maps for 3D A (10,10,10). Negative values clipped to 0."""
            A = np.clip(A, 0, None)
            D = A.shape[0]
            out = []
            for z in range(D):
                plane = A[z]
                s = plane.sum() + eps
                out.append(plane / s)
            return out

        def print_fraction_diff(name, A, B):
            """Print per-Z ( frac(A) - frac(B) )."""
            fA = per_layer_fraction_maps(A)
            fB = per_layer_fraction_maps(B)
            print(f"\n--- Fraction difference per Z ({name}) ---")
            for z in range(len(fA)):
                diff = fA[z] - fB[z]
                print(f"z={z}:")
                print(np.round(diff, 4))

        # ---- 4) Print the requested fraction differences ----
        print_fraction_diff("patch - target", patch_map, target_map)
        print_fraction_diff("patch - E_ori_up", patch_map, E_up)
                                 
            
            
        return loss
    
    
    
    
    # fix: r should be always not larger than t
    def sample_t_r(self, batch_size, device, flow_ratio=None):
        if flow_ratio is None:
            flow_ratio = self.config.get("FLOW_RATIO", 0.75)
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

        # Assign t = max, r = min, for each pair
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r
    
        
        
    def sample_time_steps(self, batch_size, device, time_sampler='logit_normal', flow_ratio=None):
        """Sample time steps (r, t) according to the configured sampler"""
        # Step1: Sample two time points
        if time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
        elif time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, 2, device=device)
            normal_samples = normal_samples * 1 - 0.4
            time_samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")
        
        # Step2: Ensure t > r by sorting
        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0], sorted_samples[:, 1]
        
        # Step3: Control the proportion of r=t samples
        if flow_ratio is None:
            flow_ratio = self.config.get("FLOW_RATIO", 0.75)
        fraction_equal = flow_ratio  # e.g., 0.75 means 75% of samples have r=t
        # Create a mask for samples where r should equal t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal
        # Apply the mask: where equal_mask is True, set r=t (replace)
        r = torch.where(equal_mask, t, r)
        
        return r, t 

    def compute_loss_meanflow(self, data, energy, noise = None, t = None, layers = None, weighting = 'uniform', energy_loss_scale = 1e-2, scales=1, time_sampler="uniform", time_mu=-0.4,time_sigma=1.0, adaptive=False):
        self.scales = scales
        device = data.device
        dtype  = data.dtype
        
        time_dist=['lognorm', -0.4, 1.0]
        self.time_dist = time_dist

        if noise is None:
            noise = torch.randn_like(data)

        # sample t
        batch_size = data.shape[0]
        const_shape = (data.shape[0], *((1,) * (len(data.shape) - 1)))

            
        # Sample time steps
        #r, t = self.sample_time_steps(batch_size, device, time_sampler)
        flow_ratio = self.config.get("FLOW_RATIO", 0.75)
        r, t = self.sample_t_r(batch_size, device, flow_ratio)
        
        t_ = torch.reshape(t, const_shape).detach().clone()
        r_ = torch.reshape(r, const_shape).detach().clone()

        # interpolant
        #alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t.view(const_shape))

        # model IO
        #z_t = alpha_t * data + sigma_t * noise
        #v_t = d_alpha_t * data + d_sigma_t * noise
        
        z_t = (1 - t_) * data + t_ * noise
        v_t = noise - data
        
        time_diff = (t - r).view(const_shape)
                
        u_target = torch.zeros_like(v_t)
        
        energy = energy.to(dtype=dtype)
        
        
        u = self.pred_meanflow(z_t, energy, t,r_emb = r, layers=layers)
        
        primals = (z_t, r, t)
        tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))
        
        # Use manual multi-GPU jvp to reduce memory usage
        # jvp doesn't work with DataParallel, but we can manually split the batch
        def fn_current(z, cur_r, cur_t, energy_chunk, layers_chunk):
            return self.pred_meanflow(z, energy_chunk, cur_t, r_emb = cur_r, layers=layers_chunk)
        
        u, dudt = self._parallel_jvp(fn_current, primals, tangents, energy, layers)
        
        
        u_target = v_t - time_diff * dudt

        
        error = u - stopgrad(u_target)
        loss_mid = adaptive_l2_loss(error)
        # loss = F.mse_loss(u, stopgrad(u_tgt))

        loss_mean_ref = (stopgrad(error) ** 2).mean()

        
        if adaptive:
            weights = 1.0 / (loss_mid.detach() + 1e-3).pow(1)
            loss = weights * loss_mid          
        else:
            loss = loss_mid
        
        # Add energy conservation physics constraint if energy_loss_scale > 0
        if energy_loss_scale > 0:
            # Compute predicted x0 from the flow prediction
            # For MeanFlow, we can approximate x0 from the current prediction
            # Using the flow: x0 ≈ z_t - t * u (approximate, may need adjustment based on your flow definition)
            x0_pred = z_t - t_.view(-1, *((1,) * (len(data.shape) - 1))) * u
            
            # Sum total energy over spatial dimensions
            dims = [i for i in range(1, len(data.shape))]
            tot_energy_pred = torch.sum(x0_pred, dim=dims)
            tot_energy_data = torch.sum(data, dim=dims)
            loss_en = energy_loss_scale * torch.nn.functional.mse_loss(tot_energy_data, tot_energy_pred) / self.nvoxels
            loss = loss + loss_en
        #loss_mean_ref = torch.mean((error**2))
     
            
        return loss, loss_mean_ref
    
    
    def compute_loss_meanflow_pidm(self, data, energy, noise = None, t = None, layers = None, weighting = 'uniform', energy_loss_scale = 1e-2, scales=1, time_sampler="uniform", time_mu=-0.4,time_sigma=1.0, adaptive=False):
        self.scales = scales
        device = data.device
        dtype  = data.dtype
        #print(f'data dtype {dtype}')
        
        time_dist=['lognorm', -0.4, 1.0]
        self.time_dist = time_dist

        if noise is None:
            noise = torch.randn_like(data)

        # sample t
        batch_size = data.shape[0]
        const_shape = (data.shape[0], *((1,) * (len(data.shape) - 1)))


        flow_ratio = self.config.get("FLOW_RATIO", 0.75)
        r, t = self.sample_t_r(batch_size, device, flow_ratio)
        
        t_ = torch.reshape(t, const_shape).detach().clone()
        r_ = torch.reshape(r, const_shape).detach().clone()

        
        z_t = (1 - t_) * data + t_ * noise
        v_t = noise - data
        
        time_diff = (t - r).view(const_shape)
                
        u_target = torch.zeros_like(v_t)
        
        energy = energy.to(device, dtype=dtype)
        
        u = self.pred_meanflow(z_t, energy, t,r_emb = r, layers=layers)
        
        primals = (z_t, r, t)
        tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))
        
        # Use manual multi-GPU jvp to reduce memory usage
        # jvp doesn't work with DataParallel, but we can manually split the batch
        def fn_current(z, cur_r, cur_t, energy_chunk, layers_chunk):
            return self.pred_meanflow(z, energy_chunk, cur_t, r_emb = cur_r, layers=layers_chunk)
        
        u, dudt = self._parallel_jvp(fn_current, primals, tangents, energy, layers)
        
        
        u_target = v_t - time_diff * dudt

        
        error = u - stopgrad(u_target)
        loss_mid = adaptive_l2_loss(error)

        loss_mean_ref = (stopgrad(error) ** 2).mean()
        
        
        with torch.no_grad():  # treat as regularizer; no backprop through sampler
            sample_data, _, _ = self.meanflow_sampler(
                #data, energy, layers=layers, sample_algo='meanflow', num_steps=10
                noise, energy, layers=layers, sample_algo='meanflow', num_steps=10
            )
            # Optionally clone if ReverseNorm does in-place ops
            real_std   = data
            gen_std    = sample_data.to(device=device)
            E_std      = energy.to(device=device)
            
            dataset_config = self.config

            # ---- reverse normalisation back to physical space ----
            # Check if HGCal (dataset_num=2 typically indicates HGCal)
            is_hgcal = dataset_config.get('HGCAL', False) or dataset_config.get('DATASET_NUM', 0) == 2
            # Convert tensors to numpy for ReverseNorm
            real_std_np = real_std.detach().cpu().numpy() if torch.is_tensor(real_std) else real_std
            gen_std_np = gen_std.detach().cpu().numpy() if torch.is_tensor(gen_std) else gen_std
            E_std_np = E_std.detach().cpu().numpy() if torch.is_tensor(E_std) else E_std
            
            # Ensure E_std_np is 2D: (batch_size, num_features)
            # ReverseNormHGCal expects e to be 2D for indexing e[:, 0]
            if E_std_np.ndim == 1:
                E_std_np = E_std_np.reshape(-1, 1)
            
            # Handle layerE: ReverseNormHGCal requires it if "layer" is in showerMap
            shower_map = dataset_config.get('SHOWERMAP', '')
            layerE_np = None
            if layers is not None:
                layerE_np = layers.detach().cpu().numpy() if torch.is_tensor(layers) else layers
                if layerE_np.ndim == 1:
                    layerE_np = layerE_np.reshape(-1, 1)
            elif "layer" in shower_map:
                # If showerMap contains "layer" but layers is None, create dummy layerE
                # ReverseNormHGCal expects layerE with shape (batch_size, 1+num_layers)
                # First column is totalE (normalized), rest are layer energies (normalized/logit)
                batch_size = E_std_np.shape[0]
                num_layers = 47  # HGCal has 47 layers
                # Create dummy layerE: (totalE, layer1, layer2, ...)
                # Use E_std_np for totalE (already normalized), zeros for layers (will normalize to uniform)
                layerE_np = np.zeros((batch_size, 1 + num_layers))
                layerE_np[:, 0] = E_std_np[:, 0]  # totalE from E_std (already normalized)
                # Layer energies: zeros will normalize to uniform distribution after reverse_logit
                # This is a reasonable default when actual layer energies are not available
                layerE_np[:, 1:] = 0.0
            
            # Use SHAPE_ORIG if available, otherwise fall back to SHAPE or SHAPE_PAD
            shape_key = dataset_config.get('SHAPE_ORIG') or dataset_config.get('SHAPE') or dataset_config.get('SHAPE_PAD')
            
            # Ensure emin/emax are scalars, not arrays
            # ReverseNormHGCal expects scalars, but config might have arrays (e.g., [50, 1.99, 1.57])
            emin_val = dataset_config['EMIN']
            emax_val = dataset_config['EMAX']
            if isinstance(emin_val, (list, tuple, np.ndarray)):
                emin_val = float(emin_val[0]) if len(emin_val) > 0 else float(emin_val)
            else:
                emin_val = float(emin_val)
            if isinstance(emax_val, (list, tuple, np.ndarray)):
                emax_val = float(emax_val[0]) if len(emax_val) > 0 else float(emax_val)
            else:
                emax_val = float(emax_val)
            
            real_phys, E_phys = ReverseNorm(
                real_std_np,
                E_std_np,
                hgcal      = is_hgcal,
                layerE     = layerE_np,
                shape      = shape_key,
                logE       = dataset_config['logE'],
                max_deposit= dataset_config['MAXDEP'],
                emax       = emax_val,
                emin       = emin_val,
                showerMap  = dataset_config['SHOWERMAP'],
                dataset_num= dataset_config['DATASET_NUM'],
                orig_shape = False,
                ecut       = dataset_config['ECUT'],
            )

            gen_phys, _ = ReverseNorm(
                gen_std_np,
                E_std_np,                       # same incident energies for this batch
                hgcal      = is_hgcal,
                layerE     = layerE_np,
                shape      = shape_key,
                logE       = dataset_config['logE'],
                max_deposit= dataset_config['MAXDEP'],
                emax       = emax_val,
                emin       = emin_val,
                showerMap  = dataset_config['SHOWERMAP'],
                dataset_num= dataset_config['DATASET_NUM'],
                orig_shape = False,
                ecut       = dataset_config['ECUT'],
            )
        # Convert numpy arrays to PyTorch tensors and reshape to match data shape
        # real_phys and gen_phys are numpy arrays from ReverseNorm
        # data.shape is (B, 1, L, H, W) = (B, 1, 47, 12, 21)
        real_phys_tensor = torch.from_numpy(real_phys).to(device=device, dtype=data.dtype)
        gen_phys_tensor = torch.from_numpy(gen_phys).to(device=device, dtype=data.dtype)
        
        # Reshape to match data shape: (B, 1, L, H, W)
        # real_phys/gen_phys might be (B, L, H, W) or (B, L*H*W) depending on ReverseNorm output
        if real_phys_tensor.ndim == 4:  # (B, L, H, W)
            real_phys_tensor = real_phys_tensor.unsqueeze(1)  # -> (B, 1, L, H, W)
        elif real_phys_tensor.ndim == 2:  # (B, L*H*W) - flattened
            real_phys_tensor = real_phys_tensor.reshape(data.shape)  # -> (B, 1, L, H, W)
        
        if gen_phys_tensor.ndim == 4:  # (B, L, H, W)
            gen_phys_tensor = gen_phys_tensor.unsqueeze(1)  # -> (B, 1, L, H, W)
        elif gen_phys_tensor.ndim == 2:  # (B, L*H*W) - flattened
            gen_phys_tensor = gen_phys_tensor.reshape(data.shape)  # -> (B, 1, L, H, W)
        
        S_true   = _layer_sums(real_phys_tensor, layer_dim=2)      # target from data
        S_sample = _layer_sums(gen_phys_tensor, layer_dim=2)
        
        #w = (S_true / (S_true.sum(dim=1, keepdim=True) + 1e-8)).detach()
        w = 1
        
        N_cells = sample_data.shape[-1] * sample_data.shape[-2] 

        # MSE (weighted) across layers, then mean over batch
        raw_pidm = ((S_sample - S_true.detach())**2 * w).mean(dim=1)  # (B,)
        
        loss_pidm_vec = raw_pidm


        return loss_mid, loss_mean_ref, loss_pidm_vec
    
    
    def compute_loss_meanflow_gmm(self, data, energy, gmm_prior, noise = None, t = None, layers = None, weighting = 'uniform', energy_loss_scale = 1e-2, scales=1, time_sampler="uniform", time_mu=-0.4,time_sigma=1.0, adaptive=False):
        self.scales = scales
        device = data.device
        dtype  = data.dtype
        
        time_dist=['lognorm', -0.4, 1.0]
        self.time_dist = time_dist

        #if noise is None:
        #    noise = torch.randn_like(data)

        # sample t
        batch_size = data.shape[0]
        const_shape = (data.shape[0], *((1,) * (len(data.shape) - 1)))

        flow_ratio = self.config.get("FLOW_RATIO", 0.75)
        r, t = self.sample_t_r(batch_size, device, flow_ratio)
        
        t_ = torch.reshape(t, const_shape).detach().clone()
        r_ = torch.reshape(r, const_shape).detach().clone()

        
        #z_t = (1 - t_) * data + t_ * noise
        #v_t = noise - data
        
        #print(data.shape)
        #print(gmm_prior.shape)
        
        gmm_prior = gmm_prior.view(data.shape)
        
        z_t = (1 - t_) * data + t_ * gmm_prior
        v_t = gmm_prior - data
        #print(z_t.shape)
        #print(v_t.shape)
        
        time_diff = (t - r).view(const_shape)
                
        u_target = torch.zeros_like(v_t)
        
        energy = energy.to(device, dtype=dtype)
        
        
        u = self.pred_meanflow(z_t, energy, t,r_emb = r, layers=layers)
        
        primals = (z_t, r, t)
        tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))
        
        # Use manual multi-GPU jvp to reduce memory usage
        def fn_current(z, cur_r, cur_t, energy_chunk=None, layers_chunk=None):
            e = energy_chunk if energy_chunk is not None else energy
            l = layers_chunk if layers_chunk is not None else layers
            return self.pred_meanflow(z, e, cur_t, r_emb = cur_r, layers=l)
        
        u, dudt = self._parallel_jvp(fn_current, primals, tangents, energy, layers)
        
        
        u_target = v_t - time_diff * dudt

        error = u - stopgrad(u_target)
        loss_mid = adaptive_l2_loss(error)
        # loss = F.mse_loss(u, stopgrad(u_tgt))

        loss_mean_ref = (stopgrad(error) ** 2).mean()

        loss = loss_mid

     
            
        return loss, loss_mean_ref
    
    
    
    
    def compute_loss_meanflow_gmm_pidm(self, data, energy, gmm_prior, noise = None, t = None, layers = None, weighting = 'uniform', energy_loss_scale = 1e-2, scales=1, time_sampler="uniform", time_mu=-0.4,time_sigma=1.0, adaptive=False):
        self.scales = scales
        device = data.device
        dtype  = data.dtype
        
        time_dist=['lognorm', -0.4, 1.0]
        self.time_dist = time_dist

        #if noise is None:
        #    noise = torch.randn_like(data)

        # sample t
        batch_size = data.shape[0]
        const_shape = (data.shape[0], *((1,) * (len(data.shape) - 1)))

        flow_ratio = self.config.get("FLOW_RATIO", 0.75)
        r, t = self.sample_t_r(batch_size, device, flow_ratio)
        
        t_ = torch.reshape(t, const_shape).detach().clone()
        r_ = torch.reshape(r, const_shape).detach().clone()

        
        gmm_prior = gmm_prior.view(data.shape)
        
        z_t = (1 - t_) * data + t_ * gmm_prior
        v_t = gmm_prior - data
 
        time_diff = (t - r).view(const_shape)
                
        u_target = torch.zeros_like(v_t)
        
        energy = energy.to(device, dtype=dtype)
        
        
        u = self.pred_meanflow(z_t, energy, t,r_emb = r, layers=layers)
        
        primals = (z_t, r, t)
        tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))
        
        # Use manual multi-GPU jvp to reduce memory usage
        def fn_current(z, cur_r, cur_t, energy_chunk=None, layers_chunk=None):
            e = energy_chunk if energy_chunk is not None else energy
            l = layers_chunk if layers_chunk is not None else layers
            return self.pred_meanflow(z, e, cur_t, r_emb = cur_r, layers=l)
        
        u, dudt = self._parallel_jvp(fn_current, primals, tangents, energy, layers)
        
        
        u_target = v_t - time_diff * dudt

        error = u - stopgrad(u_target)
        loss_mid = adaptive_l2_loss(error)
        # loss = F.mse_loss(u, stopgrad(u_tgt))

        loss_mean_ref = (stopgrad(error) ** 2).mean()

        loss = loss_mid
        
        with torch.no_grad():  # treat as regularizer; no backprop through sampler
            sample_data, _, _ = self.meanflow_gmm_sampler(
                #data, energy, layers=layers, sample_algo='meanflow', num_steps=10
                gmm_prior, energy, layers=layers, sample_algo='meanflow_gmm', num_steps=10
            )
        
            # Optionally clone if ReverseNorm does in-place ops
            real_std   = data
            gen_std    = sample_data.to(device=device)
            E_std      = energy.to(device=device)
            
            dataset_config = self.config

            # ---- reverse normalisation back to physical space ----
            # Check if HGCal (dataset_num=2 typically indicates HGCal)
            is_hgcal = dataset_config.get('HGCAL', False) or dataset_config.get('DATASET_NUM', 0) == 2
            # Convert tensors to numpy for ReverseNorm
            real_std_np = real_std.detach().cpu().numpy() if torch.is_tensor(real_std) else real_std
            gen_std_np = gen_std.detach().cpu().numpy() if torch.is_tensor(gen_std) else gen_std
            E_std_np = E_std.detach().cpu().numpy() if torch.is_tensor(E_std) else E_std
            
            # Ensure E_std_np is 2D: (batch_size, num_features)
            # ReverseNormHGCal expects e to be 2D for indexing e[:, 0]
            if E_std_np.ndim == 1:
                E_std_np = E_std_np.reshape(-1, 1)
            
            # Handle layerE: ReverseNormHGCal requires it if "layer" is in showerMap
            shower_map = dataset_config.get('SHOWERMAP', '')
            layerE_np = None
            if layers is not None:
                layerE_np = layers.detach().cpu().numpy() if torch.is_tensor(layers) else layers
                if layerE_np.ndim == 1:
                    layerE_np = layerE_np.reshape(-1, 1)
            elif "layer" in shower_map:
                # If showerMap contains "layer" but layers is None, create dummy layerE
                # ReverseNormHGCal expects layerE with shape (batch_size, 1+num_layers)
                # First column is totalE (normalized), rest are layer energies (normalized/logit)
                batch_size = E_std_np.shape[0]
                num_layers = 47  # HGCal has 47 layers
                # Create dummy layerE: (totalE, layer1, layer2, ...)
                # Use E_std_np for totalE (already normalized), zeros for layers (will normalize to uniform)
                layerE_np = np.zeros((batch_size, 1 + num_layers))
                layerE_np[:, 0] = E_std_np[:, 0]  # totalE from E_std (already normalized)
                # Layer energies: zeros will normalize to uniform distribution after reverse_logit
                # This is a reasonable default when actual layer energies are not available
                layerE_np[:, 1:] = 0.0
            
            # Use SHAPE_ORIG if available, otherwise fall back to SHAPE or SHAPE_PAD
            shape_key = dataset_config.get('SHAPE_ORIG') or dataset_config.get('SHAPE') or dataset_config.get('SHAPE_PAD')
            
            # Ensure emin/emax are scalars, not arrays
            # ReverseNormHGCal expects scalars, but config might have arrays (e.g., [50, 1.99, 1.57])
            emin_val = dataset_config['EMIN']
            emax_val = dataset_config['EMAX']
            if isinstance(emin_val, (list, tuple, np.ndarray)):
                emin_val = float(emin_val[0]) if len(emin_val) > 0 else float(emin_val)
            else:
                emin_val = float(emin_val)
            if isinstance(emax_val, (list, tuple, np.ndarray)):
                emax_val = float(emax_val[0]) if len(emax_val) > 0 else float(emax_val)
            else:
                emax_val = float(emax_val)
            
            real_phys, E_phys = ReverseNorm(
                real_std_np,
                E_std_np,
                hgcal      = is_hgcal,
                layerE     = layerE_np,
                shape      = shape_key,
                logE       = dataset_config['logE'],
                max_deposit= dataset_config['MAXDEP'],
                emax       = emax_val,
                emin       = emin_val,
                showerMap  = dataset_config['SHOWERMAP'],
                dataset_num= dataset_config['DATASET_NUM'],
                orig_shape = False,
                ecut       = dataset_config['ECUT'],
            )

            gen_phys, _ = ReverseNorm(
                gen_std_np,
                E_std_np,                       # same incident energies for this batch
                hgcal      = is_hgcal,
                layerE     = layerE_np,
                shape      = shape_key,
                logE       = dataset_config['logE'],
                max_deposit= dataset_config['MAXDEP'],
                emax       = emax_val,
                emin       = emin_val,
                showerMap  = dataset_config['SHOWERMAP'],
                dataset_num= dataset_config['DATASET_NUM'],
                orig_shape = False,
                ecut       = dataset_config['ECUT'],
            )
        # Convert numpy arrays to PyTorch tensors and reshape to match data shape
        # real_phys and gen_phys are numpy arrays from ReverseNorm
        # data.shape is (B, 1, L, H, W) = (B, 1, 47, 12, 21)
        real_phys_tensor = torch.from_numpy(real_phys).to(device=device, dtype=data.dtype)
        gen_phys_tensor = torch.from_numpy(gen_phys).to(device=device, dtype=data.dtype)
        
        # Reshape to match data shape: (B, 1, L, H, W)
        # real_phys/gen_phys might be (B, L, H, W) or (B, L*H*W) depending on ReverseNorm output
        if real_phys_tensor.ndim == 4:  # (B, L, H, W)
            real_phys_tensor = real_phys_tensor.unsqueeze(1)  # -> (B, 1, L, H, W)
        elif real_phys_tensor.ndim == 2:  # (B, L*H*W) - flattened
            real_phys_tensor = real_phys_tensor.reshape(data.shape)  # -> (B, 1, L, H, W)
        
        if gen_phys_tensor.ndim == 4:  # (B, L, H, W)
            gen_phys_tensor = gen_phys_tensor.unsqueeze(1)  # -> (B, 1, L, H, W)
        elif gen_phys_tensor.ndim == 2:  # (B, L*H*W) - flattened
            gen_phys_tensor = gen_phys_tensor.reshape(data.shape)  # -> (B, 1, L, H, W)
        
        S_true   = _layer_sums(real_phys_tensor, layer_dim=2)      # target from data
        S_sample = _layer_sums(gen_phys_tensor, layer_dim=2)
        
        #w = (S_true / (S_true.sum(dim=1, keepdim=True) + 1e-8)).detach()
        w = 1
        
        N_cells = sample_data.shape[-1] * sample_data.shape[-2] 

        # MSE (weighted) across layers, then mean over batch
        raw_pidm = ((S_sample - S_true.detach())**2 * w).mean(dim=1)  # (B,)
        
        loss_pidm_vec = raw_pidm

            
        return loss, loss_mean_ref, loss_pidm_vec 
        




    def do_time_embed(self, t = None, embed_type = "identity",  sigma = None,):
        if(self.discrete_time):
            if(sigma is None): sigma = self.sqrt_one_minus_alphas_cumprod.to(t.device)[t]

            if(embed_type == "identity" or embed_type == 'sin'):
                return t
            if(embed_type == "scaled"):
                return t/self.nsteps
            if(embed_type == "sigma"):
                my_sigma = sigma / (1 + sigma**2).sqrt()
                return my_sigma
            if(embed_type == "log"):
                #return 0.5 * torch.log(sigma).to(t.device)
                return 0.5 * torch.log(sigma)
        else:
            if(embed_type == "log"):
                #return 0.5 * torch.log(sigma).to(t.device)
                return 0.5 * torch.log(sigma)
            else:
                return sigma
            
            
    #util function for LMS, order default = 4         
            
    def linear_multistep_coeff(self, order, t, i, j):
        if order - 1 > i:
            raise ValueError(f'Order {order} too high for step {i}')
        def fn(tau):
            prod = 1.
            for k in range(order):
                if j == k:
                    continue
                prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
            return prod
        return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]
    
    ##########
    # work in progress EDM_G++ Sampler, need a pretrained EDM model and extra training on discriminator
    # code from https://github.com/alsdudrla10/DG
    def compute_tau(self, std_wve_t):
        self.beta_0 = 0.1
        self.beta_1 = 20.
        tau = -self.beta_0 + torch.sqrt(self.beta_0 ** 2 + 2. * (self.beta_1 - self.beta_0) * torch.log(1. + std_wve_t ** 2))
        tau /= self.beta_1 - self.beta_0
        return tau

    def marginal_prob(self, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def transform_unnormalized_wve_to_normalized_vp(self, t, std_out=False):
        tau = self.compute_tau(t)
        mean_vp_tau, std_vp_tau = self.marginal_prob(tau)
        if std_out:
            return mean_vp_tau, std_vp_tau, tau
        return mean_vp_tau, tau

    def compute_t_cos_from_t_lin(self, t_lin):
        
        self.s = 0.008
        self.f_0 = np.cos(self.s / (1. + self.s) * np.pi / 2.) ** 2

        sqrt_alpha_t_bar = torch.exp(-0.25 * t_lin ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t_lin * self.beta_0)
        time = torch.arccos(np.sqrt(self.f_0) * sqrt_alpha_t_bar)
        t_cos = self.T * ((1. + self.s) * 2. / np.pi * time - self.s)
        return t_cos
    
    
    def get_grad_log_ratio(discriminator, unnormalized_input, std_wve_t, resolution, time_min, time_max, E, log=False):
        mean_vp_tau, tau = self.transform_unnormalized_wve_to_normalized_vp(std_wve_t) ## VP pretrained classifier
        if tau.min() > time_max or tau.min() < time_min or discriminator == None:
            if log:
                return torch.zeros_like(unnormalized_input), 10000000. * torch.ones(unnormalized_input.shape[0], device=unnormalized_input.device)
            return torch.zeros_like(unnormalized_input)
        else:
            x = mean_vp_tau[:,None,None,None,None] * unnormalized_input
        with torch.enable_grad():
            x_ = x.float().clone().detach().requires_grad_()
            if resolution == 64: 
                tau = vpsde.compute_t_cos_from_t_lin(tau)
            tau = torch.ones(input.shape[0], device=tau.device) * tau
            log_ratio = get_log_ratio(discriminator, x_, tau, E)
            discriminator_guidance_score = torch.autograd.grad(outputs=log_ratio.sum(), inputs=x_, retain_graph=False)[0]
            discriminator_guidance_score *= - ((std_wve_t[:,None,None,None,None] ** 2) * mean_vp_tau[:,None,None,None,None])
        if log:
            return discriminator_guidance_score, log_ratio
        return discriminator_guidance_score
    
    
    def get_log_ratio(discriminator, x, time, E):
        if discriminator == None:
            return torch.zeros(x.shape[0], device=x.device)
        else:
            logits = discriminator(x, timesteps=time, condition=E)
            prediction = torch.clip(logits, 1e-5, 1. - 1e-5)
            log_ratio = torch.log(prediction / (1. - prediction))
            return log_ratio
        
    ##########
            
    def get_steps(self, x, num_step, min_t, max_t, rho):

        step_indices = torch.arange(num_step, dtype=torch.float32, device=x.device)
        t_steps = (max_t ** (1 / rho) + step_indices / (num_step - 1) * (min_t ** (1 / rho) - max_t ** (1 / rho))) ** rho
        return t_steps
    
    
    @torch.no_grad()
    def estimate_gamma(self, x0, energy, layers=None, n_batches=1):
        """
        x0: a small batch of REAL training-space samples (after the same normalization + geom conversion).
        energy: matching conditioning for x0.
        """
        num = 0.0
        den = 0.0
        for _ in range(n_batches):
            # t ~ U[0,1] with broadcast shape like training
            const_shape = (x0.shape[0],) + (1,) * (x0.ndim - 1)
            t = torch.rand(const_shape, device=x0.device, dtype=x0.dtype)

            noise = torch.randn_like(x0)

            # same interpolant as training
            alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t)

            # build path sample and target velocity (same as your loss)
            x_t    = alpha_t * x0 + sigma_t * noise
            v_tgt  = d_alpha_t * x0 + d_sigma_t * noise

            # model prediction (same call signature as in training)
            v_pred, _ = self.pred(
                x_t, energy, t, layers=layers,
                return_patch=False, encoder_patch=True
            )

            num += (v_pred * v_tgt).sum().item()
            den += (v_pred * v_pred).sum().item()

        den = max(den, 1e-12)
        return num / den
    
    
    def meanflow_sampler(
        self, x, E, layers = None, sample_algo = 'meanflow', randn_like=torch.randn_like,
        num_steps=1):
        
        
        xs = []
        x0s = None
        batch_size = x.shape[0]
        

        #x_next = x.to(torch.float32)
        device = x.device
        _dtype = x.dtype  # model runs at original dtype
        
        E = E.to(dtype=_dtype)

        if num_steps == 1:
            r = 0.
            t = 1.
            
            t_in = torch.full(
                    make_time_const_shape(x.shape[0], x),
                    fill_value=float(t),
                    device=device, dtype=_dtype
                )
            
            r_in = torch.full(
                    make_time_const_shape(x.shape[0], x),
                    fill_value=float(r),
                    device=device, dtype=_dtype
                )
            
            u = self.pred_meanflow(
                    x.to(dtype=_dtype), E, t_in.to(dtype=_dtype), r_emb=r_in, layers=layers
                ).to(torch.float64)
            
            x_next = x - u
            
        else:
            x_next = x
        
            time_steps = torch.linspace(1, 0, num_steps + 1, device=device)
            
            for i in range(num_steps):
                t_cur = time_steps[i]
                t_next = time_steps[i + 1]

                #t = torch.full((batch_size,), t_cur, device=device)
                #r = torch.full((batch_size,), t_next, device=device)
                
                
                t_in = torch.full(
                        make_time_const_shape(x.shape[0], x),
                        fill_value=float(t_cur.item()),
                        device=device, dtype=_dtype
                    )

                r_in = torch.full(
                        make_time_const_shape(x.shape[0], x),
                        fill_value=float(t_next.item()),
                        device=device, dtype=_dtype
                    )

                u = self.pred_meanflow(
                        x_next.to(dtype=_dtype), E, t_in.to(dtype=_dtype), r_emb=r_in, layers=layers
                    ).to(torch.float64)
                
                
                x_next = x_next - (t_cur - t_next) * u
                


        
        return x_next, xs, x0s
    
    
    def meanflow_gmm_sampler(
        self, x, E, layers = None, sample_algo = 'meanflow_gmm', randn_like=torch.randn_like,
        num_steps=1):
        
        
        xs = []
        x0s = None
        batch_size = x.shape[0]
        

        #x_next = x.to(torch.float32)
        device = x.device
        _dtype = x.dtype  # model runs at original dtype
        
        E = E.to(dtype=_dtype)

        if num_steps == 1:
            r = 0.
            t = 1.
            
            t_in = torch.full(
                    make_time_const_shape(x.shape[0], x),
                    fill_value=float(t),
                    device=device, dtype=_dtype
                )
            
            r_in = torch.full(
                    make_time_const_shape(x.shape[0], x),
                    fill_value=float(r),
                    device=device, dtype=_dtype
                )
            
            u = self.pred_meanflow(
                    x.to(dtype=_dtype), E, t_in.to(dtype=_dtype), r_emb=r_in, layers=layers
                ).to(torch.float64)
            
            x_next = x - u
            
        else:
            x_next = x
        
            time_steps = torch.linspace(1, 0, num_steps + 1, device=device)
            
            for i in range(num_steps):
                t_cur = time_steps[i]
                t_next = time_steps[i + 1]

                #t = torch.full((batch_size,), t_cur, device=device)
                #r = torch.full((batch_size,), t_next, device=device)
                
                
                t_in = torch.full(
                        make_time_const_shape(x.shape[0], x),
                        fill_value=float(t_cur.item()),
                        device=device, dtype=_dtype
                    )

                r_in = torch.full(
                        make_time_const_shape(x.shape[0], x),
                        fill_value=float(t_next.item()),
                        device=device, dtype=_dtype
                    )

                u = self.pred_meanflow(
                        x_next.to(dtype=_dtype), E, t_in.to(dtype=_dtype), r_emb=r_in, layers=layers
                    ).to(torch.float64)
                
                
                x_next = x_next - (t_cur - t_next) * u
                


        
        return x_next, xs, x0s
    
    def sit_sampler(
        self, x, E, layers = None, sample_algo = 'sit', randn_like=torch.randn_like,
        num_steps=100, sigma_min=0.04, sigma_max=1., gamma=1., path_type='linear',heun=False):
        
        
        xs = []
        x0s = None
        gen_size = x.shape[0]
        
        _dtype = x.dtype
        if sample_algo == 'sit_euler':
            # ensure float64 integration like your code
            #t1 = torch.ones((x.shape[0],)+(1,)*(x.ndim-1), device=x.device, dtype=torch.float64)
            #_, sigma1, _, _ = self.interpolant(t1)
            #x_next = (torch.randn_like(x) * sigma1.to(x.dtype)).to(torch.float64)
            x_next = x.to(torch.float32)
            device = x_next.device
            _dtype = x.dtype  # model runs at original dtype

            # time grid (descending); match training time domain [0,1]
            t_steps = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64, device=device)

            

            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                x_cur = x_next

                t_in = torch.full(
                    make_time_const_shape(x_cur.shape[0], x_cur),
                    fill_value=float(t_cur.item()),
                    device=device, dtype=torch.float32
                )

                v_cur = self.pred(
                    x_cur.to(dtype=_dtype), E, t_in.to(dtype=_dtype), layers=layers
                ).to(torch.float32)

                # --- NEW: apply gamma if available ---
                if gamma is not None:
                    v_cur = float(gamma) * v_cur
                # ------------------------------------

                dt = (t_next - t_cur)
                x_euler = x_cur + dt * v_cur

                if heun and (i < num_steps - 1):
                    t_in_next = torch.full(
                        make_time_const_shape(x_cur.shape[0], x_cur),
                        fill_value=float(t_next.item()),
                        device=device, dtype=torch.float32
                    )
                    v_next = self.pred(
                        x_euler.to(dtype=_dtype), E, t_in_next.to(dtype=_dtype), layers=layers
                    ).to(torch.float32)

                    # --- NEW: apply gamma to corrector too ---
                    if gamma is not None:
                        v_next = float(gamma) * v_next
                    # ----------------------------------------

                    x_next = x_cur + dt * 0.5 * (v_cur + v_next)
            else:
                    x_next = x_euler

                
                #print(f"[t={float(t_cur):.3f}] mean={x_cur.mean().item():.6g} std={x_cur.std().item():.6g} |v|={v_cur.std().item():.6g}")
                
        else:
    
            t_steps = torch.linspace(1., 0.04, num_steps, dtype=torch.float64)
            t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
            x_next = x.to(torch.float64)
            device = x_next.device

            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
                dt = t_next - t_cur
                x_cur = x_next

                model_input = x_cur
                time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
                diffusion = compute_diffusion(t_cur)            
                eps_i = torch.randn_like(x_cur).to(device)
                deps = eps_i * torch.sqrt(torch.abs(dt))


                v_cur = self.pred(model_input.to(dtype=_dtype), E, time_input.to(dtype=_dtype), layers = layers).to(torch.float64)

                s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
                d_cur = v_cur - 0.5 * diffusion * s_cur

                x_next =  x_cur + d_cur * dt + torch.sqrt(diffusion) * deps

            t_cur, t_next = t_steps[-2], t_steps[-1]
            dt = t_next - t_cur
            x_cur = x_next

            model_input = x_cur
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur

            # compute drift
            v_cur = self.pred(model_input.to(dtype=_dtype), E, time_input.to(dtype=_dtype), layers = layers).to(torch.float64)
            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
            diffusion = compute_diffusion(t_cur)
            d_cur = v_cur - 0.5 * diffusion * s_cur

            x_next = x_cur + dt * d_cur
        
        return x_next, xs, x0s
        

    def edm_sampler(
        self, x, E, layers = None, sample_algo = 'euler', randn_like=torch.randn_like,
        num_steps=18, sigma_min=0.002, sigma_max=1, rho=1,order=4,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1.,eta=1.,beta_d=19.9, beta_min=0.1, eps_s=1e-3,
        restart_info='{"0": [3, 1, 19.35, 40.79], "1": [4, 1, 1.09, 1.92], "2": [4, 4, 0.59, 1.09], "3": [4, 1, 0.30, 0.59], "4": [4, 4, 0.06, 0.30]}', restart_gamma=0.05, boosting=0, time_min=0.01, time_max=1.0, dg_weight_1st_order=2., dg_weight_2nd_order=0.
    ):

        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()
        old_denoised = None
        h_last = None
        
        # Adjust noise levels based on what's supported by the network.
        #sigma_min = max(sigma_min, net.sigma_min)
        #sigma_max = min(sigma_max, net.sigma_max)
        xs = []
        x0s = None

        gen_size = x.shape[0]
        #print(f'x shape {x.shape}')

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=x.device)
        
        
        #default edm karras
        t_steps = self.get_steps(num_step = num_steps, x=x, min_t=sigma_min, max_t=sigma_max, rho=rho)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                    sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        
        
        
        #dpm-solver lu 
        #rho=1
        #lambda_min=np.log(sigma_min)
        #lambda_max=np.log(sigma_max)
        #t_steps = self.get_steps(num_step = num_steps, x=x, min_t=lambda_min, max_t=lambda_max, rho=rho)
        #sigmas = (lambda_max ** (1 / rho) + step_indices / (num_steps - 1) * (
        #            lambda_min ** (1 / rho) - lambda_max ** (1 / rho))) ** rho
        #t_steps = torch.exp(sigmas)
        
        #vp
        #t1 = torch.linspace(1, eps_s, num_steps, device=x.device)
        #t_steps = torch.sqrt(torch.exp(beta_d * t1 ** 2 / 2 + beta_min * t1) - 1)
        
        
        

        
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

        total_step = len(t_steps)

        #print('initial timestep is {}'.format(t_steps[0]))
        x_next = x.to(torch.float32) * t_steps[0]

        # Main sampling loop.
        

        
        # {[num_steps, number of restart iteration (K), t_min, t_max], ... }
        #some option
        #multi level for imagenet {"0": [3, 1, 19.35, 40.79], "1": [4, 1, 1.09, 1.92], "2": [4, 4, 0.59, 1.09], "3": [4, 1, 0.30, 0.59], "4": [4, 4, 0.06, 0.30]}
        #single level for cifar-10 {"0": [3, 2, 0.14, 0.30]}
        import json
        #print(restart_info)
        restart_list = json.loads(restart_info) if restart_info != '' else {}
        # cast t_min to the index of nearest value in t_steps
        restart_list = {int(torch.argmin(abs(t_steps - v[2]), dim=0)): v for k, v in restart_list.items()}

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N_main -1
            x_cur = x_next
            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
            # Euler step.


            t_hat_full = torch.full((gen_size,), t_hat, device=x.device)


            denoised = self.denoise(x_hat, E, t_hat_full, layers = layers).to(torch.float32)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply edm 2nd order correction.
            if (sample_algo == 'edm')  and (i < num_steps - 1):
                t_next_full = torch.full((gen_size,), t_next, device=x.device)
                denoised = self.denoise(x_next, E, t_next_full, layers = layers).to(torch.float32)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                #print(x_next)


            # custom dpm2/edm 2nd order correction.    
            if (sample_algo == 'dpm2')  and (i < num_steps - 1):
                t_mid = t_hat.log().lerp(t_next.log(), 0.5).exp()
                dt_1 = t_mid - t_hat
                dt_2 = t_next - t_hat
                x_2 = x_hat + d_cur * dt_1
                t_mid_full = torch.full((gen_size,), t_mid, device=x.device)
                denoised_2 = self.denoise(x_2, E, t_mid_full, layers = layers).to(torch.float32)
                d_2 = (x_2 - denoised_2) / t_mid
                x_next = x_hat + d_2*dt_2


            if (sample_algo == 'restart'):

                # ================= restart ================== #
                if i + 1 in restart_list.keys():
                    restart_idx = i + 1

                    for restart_iter in range(restart_list[restart_idx][1]):

                        new_t_steps = self.get_steps(min_t=t_steps[restart_idx], max_t=restart_list[restart_idx][3], num_step=restart_list[restart_idx][0], rho=rho, x=x)
                        #print(f"restart at {restart_idx} with {new_t_steps}")
                        new_total_step = len(new_t_steps)

                        x_next = x_next + randn_like(x_next) * (new_t_steps[0] ** 2 - new_t_steps[-1] ** 2).sqrt() * S_noise


                        for j, (t_cur, t_next) in enumerate(zip(new_t_steps[:-1], new_t_steps[1:])):  # 0, ..., N_restart -1

                            x_cur = x_next
                            gamma = restart_gamma if S_min <= t_cur <= S_max else 0
                            t_hat = torch.as_tensor(t_cur + gamma * t_cur)

                            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)


                            t_hat_full = torch.full((gen_size,), t_hat, device=x.device)


                            denoised = self.denoise(x_hat, E, t_hat_full, layers = layers).to(torch.float32)
                            d_cur = (x_hat - denoised) / (t_hat)
                            x_next = x_hat + (t_next - t_hat) * d_cur

                            # Apply 2nd order correction.
                            if (sample_algo == 'restart') and (j < new_total_step - 2 or new_t_steps[-1] != 0):
                                t_next_full = torch.full((gen_size,), t_next, device=x.device)
                                denoised = self.denoise(x_next, E, t_next_full, layers = layers).to(torch.float32)
                                d_prime = (x_next - denoised) / t_next
                                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

                                xs.append(x_next)
                                    

        return x_next, xs, x0s
    
    def pred_meanflow(self, x, E, t_emb, r_emb=None, layers=None,):
        import inspect
        import torch.nn as nn

        # Ensure E has the right shape: (batch_size, 1) or (batch_size, cond_size)
        if E.ndim == 1:
            E = E.reshape(-1, 1)
        
        # layer cond
        if self.layer_cond and layers is not None:
            E = torch.cat([E, layers], dim=1)

        # Unwrap model if it's wrapped in DataParallel/DistributedDataParallel
        # jvp doesn't work with parallel wrappers (see https://github.com/pytorch/pytorch/issues/102197)
        model_to_use = self.model
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model_to_use = self.model.module

        # Check if model's forward method accepts 'r' parameter
        forward_sig = inspect.signature(model_to_use.forward)
        accepts_r = 'r' in forward_sig.parameters

        # PureDiT does not accept r; use same conditioning style as pred()
        if self.puredit:
            out = model_to_use(
                self.add_RZPhi(x),
                time=t_emb.reshape(-1,),
                cond=E.reshape(-1,),
            )
            return out

        # CondUnet / MF variants
        # Pass E with shape (batch_size, cond_size), not flattened
        if accepts_r and r_emb is not None:
            out = model_to_use(
                self.add_RZPhi(x),
                time=t_emb.reshape(-1,),
                cond=E, r=r_emb.reshape(-1,)
            )
        else:
            # Standard CondUnet doesn't accept 'r', so don't pass it
            out = model_to_use(
                self.add_RZPhi(x),
                time=t_emb.reshape(-1,),
                cond=E
            )

        return out
    def pred(self, x, E, t_emb, r_emb=None, layers=None,
             return_patch=False, encoder_patch=False):
        # optional extra encoder
        if self.NN_embed is not None:
            x = self.NN_embed.enc(x).to(x.device)

        # layer cond
        if self.layer_cond and layers is not None:
            E = torch.cat([E.reshape(-1, 1), layers], dim=1)

        # ---- CASE 1: user wants patch(es) ----
        if return_patch:
            if encoder_patch:
                # -> model returns (final_patch, second_layer_patch)
                out_patch, layer2_patch = self.model(
                    self.add_RZPhi(x),
                    time=t_emb.reshape(-1,),
                    cond=E.reshape(-1,),
                    return_patch=True,
                    encoder_patch=True,
                )
                return out_patch, layer2_patch
        else:
            # -> normal prediction (not patch)
            if encoder_patch:
                out, layer2_patch = self.model(
                    self.add_RZPhi(x),
                    time=t_emb.reshape(-1,),
                    cond=E.reshape(-1,),
                    encoder_patch=True,
                )
                return out, layer2_patch
            out = self.model(
                self.add_RZPhi(x),
                time=t_emb.reshape(-1,),
                cond=E.reshape(-1,),
            )
            return out

        # ---- CASE 2: normal diffusion prediction ----
        if self.lowhigh:
            # your lowhigh path already expects 2 returns from the model
            out, layer_out = self.model(
                self.add_RZPhi(x),
                time=t_emb.reshape(-1,),
                cond=E.reshape(-1, 1),
            )
        if self.meanflow:
            # your lowhigh path already expects 2 returns from the model
            out = self.model(
                self.add_RZPhi(x),
                time=t_emb.reshape(-1,),
                cond=E.reshape(-1,), r=r_emb.reshape(-1,)
            )
        else:
            # ADD: if encoder_patch=True, model will now return (preds, layer2_patch)
            if encoder_patch:
                out, layer2_patch = self.model(
                    self.add_RZPhi(x),
                    time=t_emb.reshape(-1,),
                    cond=E.reshape(-1,),
                    encoder_patch=True,
                )
            else:
                out = self.model(
                    self.add_RZPhi(x),
                    time=t_emb.reshape(-1,),
                    cond=E.reshape(-1,),
                )

        # decode if you have NN_embed
        if self.NN_embed is not None:
            # only for af4
            out = out.view(*out.shape[:-2], 1, 280)
            out = self.NN_embed.dec(out).to(x.device)

        # return according to branch
        if self.lowhigh:
            return out, layer_out
        else:
            if encoder_patch:
                return out, layer2_patch
            else:
                return out


    def denoise(self, x, E,  sigma, layers = None):
        
        t_emb = self.do_time_embed(embed_type = self.time_embed, sigma = sigma.reshape(-1))
        sigma = sigma.reshape(-1, *(1,)*(len(x.shape)-1))
        c_in = 1 / (sigma**2 + 1).sqrt()


        if('noise_pred' in self.training_obj):
            pred = self.pred(x * c_in, E, t_emb)
            return (x - sigma * pred)
        if('mean_pred' in self.training_obj):
            pred = self.pred(x, E, t_emb)
            return pred
        elif(self.training_obj == 'hybrid_weight'):
            pred = self.pred(x, E, t_emb)
            sigma2 = (t_emb**2).reshape(-1, *(1,)*(len(x.shape)-1))
            c_skip = 1. / (sigma2 + 1.)
            c_out = torch.sqrt(sigma2) / (sigma2 + 1.).sqrt()
            return c_skip * x + c_out * pred
        
        elif(self.training_obj == 'hybrid_weight_karras'):
            c_skip, c_out, c_in = self.get_scalings(t_emb.reshape(-1, *(1,)*(len(x.shape)-1)))
            pred = self.pred(x*c_in, E, t_emb, layers = layers)

            sigma2 = (t_emb**2).reshape(-1, *(1,)*(len(x.shape)-1))
            return c_skip * x + c_out * pred

        
    @torch.no_grad()    
    def dpm2_sampler(self, x_start, E, num_steps = 400, sample_algo = 'dpm', debug = False, use_corrector = False, x_t=None, variants = 'vc'):
        #dpm family of samplers
        
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()
        old_denoised = None
        h_last = None

        old_nsteps = self.nsteps
        if(self.nsteps != num_steps):
            self.set_sampling_steps(num_steps)

        gen_size = E.shape[0]
        device = x_start.device 
        

        xs = []
        x0s = []

        time_steps = list(range(0, num_steps))
        time_steps.reverse()
        
        
        
        #sigma_min = 0.00045

        #sigma_max = 23
        
        sigma_min = 0.00013514
        sigma_max = 71.69926
        rho = 7
        step_indices = torch.arange(num_steps, dtype=torch.float32)
        sigmas = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                   sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        
        #sigmas = torch.linspace(sigma_max**(1. / rho), sigma_min**(1. / rho), num_steps + 1).pow(rho)
        #sigmas = torch.Tensor(, device=step_indices.device)
        
        
        #optimizer = StepOptim(NoiseScheduleVP(), sigmas)
        #sigmas, _ = optimizer.get_ts_lambdas(num_steps, sigma_min, 'edm')
        #sigmas = sigmas.to(device).to(torch.float32)
        
        #print(sigmas)
        
        #eps_s = 1e-3
        #beta_d=19.9
        #beta_min=0.002
        #t1 = torch.linspace(1, eps_s, num_steps, device=device)
        #sigmas = torch.sqrt(torch.exp(beta_d * t1 ** 2 / 2 + beta_min * t1) - 1)
        
        #sigmas = -0.25 * t1 ** 2 * (beta_d - beta_min) - 0.5 * t1 * beta_min
        
        #t1 = torch.linspace(1, eps_s, num_steps+1, device=device)
        #sigmas = t1
        

        x = x_start * sigmas[0]
        print(sigmas[0])
        
        s_in = x.new_ones([x.shape[0]])
        
        if('adapt' in sample_algo):
            x = sampling.sample_dpm_adaptive(self, x, sigma_min, sigma_max, extra_args={'E':E})
        elif('++' in sample_algo and 'sde' in sample_algo):
            x = sampling.sample_dpmpp_2m_sde(self, x, sigmas, extra_args={'E':E})
        elif('++' in sample_algo and '2m' in sample_algo):
            for i in range(len(sigmas) - 1):
                denoised = self.denoise(x, E=E, sigma=sigmas[i] * s_in)
               
                t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
                h = t_next - t
                if old_denoised is None or sigmas[i + 1] == 0:
                    x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
                else:
                    h_last = t - t_fn(sigmas[i - 1])
                    r = h_last / h
                    denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
                    x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
                old_denoised = denoised
                h_last = h
        elif ('++' in sample_algo and 'fast' in sample_algo):
            x = sampling.sample_dpm_fast(self, x, sigma_min, sigma_max, num_steps, extra_args={'E':E})
            

        return x, None,None


    @torch.no_grad()
    def p_sample(self, x, E, t, cold_noise_scale = 0., noise = None, sample_algo = 'ddpm', debug = False):
        #reverse the diffusion process (one step)

        if(noise is None): 
            noise = torch.randn(x.shape, device = x.device)
            if(self.cold_diffu): #cold diffusion interpolates from avg showers instead of pure noise
                noise = self.gen_cold_image(E, cold_noise_scale, noise)

        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        posterior_variance_t = extract(self.posterior_variance, t, x.shape)

        t_emb = self.do_time_embed(t, self.time_embed)


        pred = self.pred(x, E, t_emb)
        if('noise_pred' in self.training_obj):
            noise_pred = pred
            x0_pred = None
        elif('mean_pred' in self.training_obj):
            x0_pred = pred
            noise_pred = (x - sqrt_alphas_cumprod_t * x0_pred)/sqrt_one_minus_alphas_cumprod_t
        elif('hybrid' in self.training_obj):

            sigma2 = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)**2
            c_skip = 1. / (sigma2 + 1.)
            c_out = torch.sqrt(sigma2) / (sigma2 + 1.).sqrt()

            x0_pred = c_skip * x + c_out * pred
            noise_pred = (x - sqrt_alphas_cumprod_t * x0_pred)/sqrt_one_minus_alphas_cumprod_t

        


        if(sample_algo == 'ddpm'):
            # Sampling algo from https://arxiv.org/abs/2006.11239
            # Use results from our model (noise predictor) to predict the mean of posterior distribution of prev step
            post_mean = sqrt_recip_alphas_t * ( x - betas_t * noise_pred  / sqrt_one_minus_alphas_cumprod_t)
            out = post_mean + torch.sqrt(posterior_variance_t) * noise 
            if t[0] == 0: out = post_mean
        else:
            print("Algo %s not supported!" % sample_algo)
            exit(1)



        if(debug): 
            if(x0_pred is None):
                x0_pred = (x - sqrt_one_minus_alphas_cumprod_t * noise_pred)/sqrt_alphas_cumprod_t
            return out, x0_pred
        return out

    def gen_cold_image(self, E, cold_noise_scale, noise = None):

        avg_shower, std_shower = self.lookup_avg_std_shower(E)

        if(noise is None):
            noise = torch.randn_like(avg_shower, dtype = torch.float32)

        cold_scales = cold_noise_scale

        return torch.add(avg_shower, cold_scales * (noise * std_shower))




    @torch.no_grad()
    def Sample(self, E, d_batch=None, prior=None, layers = None, num_steps = 200, cold_noise_scale = 0., sample_algo = 'ddpm', debug = False, sample_offset = 0, sample_step = 1):
        """Generate samples from diffusion model.
        
        Args:
        E: Energies
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
        
        Returns: 
        Samples.
        """

        print("SAMPLE ALGO : %s" % sample_algo)

        # Full sample (all steps)
        device = next(self.parameters()).device
        



        gen_size = E.shape[0]
        self.total_generated_samples += gen_size  # Update the total generated samples
        # start from pure noise (for each example in the batch)
        gen_shape = list(copy.copy(self._data_shape))
        gen_shape.insert(0,gen_size)

        #start from pure noise
        if(sample_algo != 'meanflow_gmm'):
            x_start = torch.randn(gen_shape, device=device)
        else:
            x_start = prior.view(gen_shape)

        avg_shower = std_shower = None
        if(self.cold_diffu): #cold diffu starts using avg images
            x_start = self.gen_cold_image(E, cold_noise_scale)


        start = time.time()
        
        
        if(sample_algo == 'euler' or sample_algo == 'lms' or sample_algo == 'edm' or sample_algo == 'restart' or sample_algo == 'dpm2' or sample_algo == 'dpmpp_2m_karras' or sample_algo == 'dpmpp_2m_sde'):
            S_churn = 30  ##30, 0 
            S_min = 0.01
            S_max = 40  ##40, 1
            S_noise = 1.003
            sigma_min = 0.002
            sigma_max = 80
            eta = 1.0
            rho = 7.0
            restart_gamma = 0.05

            x, xs, x0s = self.edm_sampler(x_start,E, layers = layers, num_steps = num_steps, sample_algo = sample_algo, sigma_min = sigma_min, sigma_max = sigma_max, rho=rho,
                    S_churn = S_churn, S_min = S_min, S_max = S_max, S_noise = S_noise, eta=eta, restart_info=self.restart_info,restart_gamma=restart_gamma)
        elif(sample_algo == 'sit' or sample_algo == 'sit_euler'):
            sigma_min = 0.04
            sigma_max = 1.
            #gamma = self.estimate_gamma(d_batch, E, layers=layers)
            #print(gamma)
            x, xs, x0s = self.sit_sampler(x_start,E, layers = layers, num_steps = num_steps, sample_algo = sample_algo, sigma_min = sigma_min, sigma_max = sigma_max)
            
        elif(sample_algo == 'meanflow'):
            x, xs, x0s = self.meanflow_sampler(x_start,E, layers = layers, num_steps = num_steps, sample_algo = sample_algo)
        elif(sample_algo == 'meanflow_gmm'):
            x, xs, x0s = self.meanflow_gmm_sampler(x_start,E, layers = layers, num_steps = num_steps, sample_algo = sample_algo)
        elif('dpm+' in sample_algo):
            x, xs, x0s = self.dpm2_sampler(x_start, E, num_steps = num_steps, sample_algo = sample_algo, debug = debug)

        else:


            x = x_start
            fixed_noise = None
            if('fixed' in sample_algo): 
                print("Fixing noise to constant for sampling!")
                fixed_noise = x_start
            xs = []
            x0s = []
            self.prev_noise = x_start

            time_steps = list(range(0, num_steps - sample_offset, sample_step))
            time_steps.reverse()

            for time_step in time_steps:      
                times = torch.full((gen_size,), time_step, device=device, dtype=torch.long)
                out = self.p_sample(x, E, times, noise = fixed_noise, cold_noise_scale = cold_noise_scale, sample_algo = sample_algo, debug = debug)
                if(debug): 
                    x, x0_pred = out
                    xs.append(x.detach().cpu().numpy())
                    x0s.append(x0_pred.detach().cpu().numpy())
            else: x = out

        end = time.time()
        sampling_time = end - start
        self.total_sampling_time += sampling_time  # Update total sampling time
        #print("Time for sampling {} events is {} seconds".format(gen_size,end - start), flush=True)
        print("Total generated samples so far: {}".format(self.total_generated_samples), flush=True)
        print("Total time spent sampling so far: {:.2f} seconds".format(self.total_sampling_time), flush=True)

        if(debug):
            return x.detach().cpu().numpy(), xs, x0s
        else:   
            return x.detach().cpu().numpy()

    
        
