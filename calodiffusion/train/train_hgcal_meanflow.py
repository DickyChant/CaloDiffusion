import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import h5py as h5
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from calodiffusion.utils import utils
from calodiffusion.utils import HGCal_utils
from calodiffusion.models.calodiffusion import CaloDiffu


if __name__ == '__main__':
    print("TRAIN DIFFU")

    if(torch.cuda.is_available()): device = torch.device('cuda')
    else: device = torch.device('cpu')
        
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', default='../data', help='Folder containing data and MC files')
    parser.add_argument('--model', default='Diffu', help='Diffusion model to train')
    parser.add_argument('-c', '--config', default='configs/test.json', help='Config file with training parameters')
    parser.add_argument('--files', nargs='+', default=None, help='Override FILES list from config (space-separated filenames, relative to data_folder)')
    parser.add_argument('--nevts', type=int,default=-1, help='Number of events to load')
    parser.add_argument('--frac', type=float,default=0.85, help='Fraction of total events used for training')
    parser.add_argument('--load', action='store_true', default=False,help='Load pretrained weights to continue the training')
    parser.add_argument('--seed', type=int, default=1234,help='Pytorch seed')
    parser.add_argument('--reset_training', action='store_true', default=False,help='Retrain')
    parser.add_argument('--weight_pidm', type=float,default=0.03, help='weight for pidm')
    parser.add_argument('--gmm_prior', type=str, default=None, 
                       help='Path to GMM prior H5 file. If provided, enables GMM training mode.')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint folder path. If not provided, uses ../models/{CHECKPOINT_NAME}_{model}/')
    flags = parser.parse_args()

    # Determine if GMM mode is enabled
    use_gmm = flags.gmm_prior is not None
    if use_gmm:
        print(f"Using GMM checkpoint for on-the-fly sampling: {flags.gmm_prior}", flush=True)
        print(f"[INFO] GMM will sample prior during training (not loading pre-generated H5)", flush=True)

    dataset_config = utils.LoadJson(flags.config)

    # Override FILES from config if --files is provided
    if flags.files is not None:
        original_files = dataset_config.get('FILES', [])
        dataset_config['FILES'] = flags.files
        dataset_config['EVAL'] = flags.files  # Also override EVAL to match
        print("[CONFIG OVERRIDE] FILES list changed from config:", flush=True)
        print(f"  Original: {original_files}", flush=True)
        print(f"  New: {flags.files}", flush=True)
        print(f"  Total files: {len(flags.files)}", flush=True)

    print("TRAINING OPTIONS")
    print(dataset_config, flush = True)

    torch.manual_seed(flags.seed)

    cold_diffu = dataset_config.get('COLD_DIFFU', False)
    cold_noise_scale = dataset_config.get('COLD_NOISE', 1.0)

    nholdout  = dataset_config.get('HOLDOUT', 0)

    # Use batch size from config (can be overridden with BATCH_MEANFLOW)
    # MeanFlow with jvp is very memory-intensive - use smaller batch size
    # DataParallel doesn't help with jvp since it runs on single GPU
    default_batch_meanflow = min(32, dataset_config.get('BATCH', 256) // 4)  # Much smaller default
    batch_size = dataset_config.get('BATCH_MEANFLOW', default_batch_meanflow)
    print(f"Using batch size {batch_size} for MeanFlow training (jvp is memory-intensive)", flush=True)
    if batch_size > 64:
        print(f"[WARNING] Batch size {batch_size} may be too large for jvp. Consider reducing BATCH_MEANFLOW in config.", flush=True)
    
    num_epochs = dataset_config['MAXEPOCH']
    early_stop = dataset_config['EARLYSTOP']
    training_obj = dataset_config.get('TRAINING_OBJ', 'noise_pred')
    loss_type = dataset_config.get("LOSS_TYPE", "l2")
    dataset_num = dataset_config.get('DATASET_NUM', 5)
    shower_embed = dataset_config.get('SHOWER_EMBED', '')
    orig_shape = ('orig' in shower_embed)
    energy_loss_scale = dataset_config.get('ENERGY_LOSS_SCALE', 0.0)
    weight_pidm = dataset_config.get('WEIGHT_PIDM', flags.weight_pidm)
    
    # Check if pre-embedding is needed
    pre_embed = ('pre-embed' in shower_embed) or ('NN' in shower_embed)
    geom_file = dataset_config.get('BIN_FILE', '')
    shower_scale = dataset_config.get('SHOWERSCALE', 200.0)
    max_cells = dataset_config.get('MAX_CELLS', None)
    
    # Initialize HGCalConverter if pre-embedding is needed
    NN_embed = None
    if pre_embed:
        trainable = dataset_config.get('TRAINABLE_EMBED', False)
        NN_embed = HGCal_utils.HGCalConverter(
            bins=dataset_config['SHAPE_FINAL'],
            geom_file=geom_file,
            trainable=trainable,
            device=device,
        ).to(device=device)
        NN_embed.init(norm=True, dataset_num=dataset_num)
        print(f"Initialized HGCalConverter for pre-embedding: {geom_file}", flush=True)
    
    # Will initialize GMM model after detecting data format
    gmm_model = None
    gmm_mean = None
    gmm_std = None
    detected_voxel_shape = None  # Will be set after first data load

    data = []
    energies = []
    prior = [] if use_gmm else None
    
    # Determine expected raw spatial size for GMM prior conversion
    # IMPORTANT: Use MAX_CELLS from config, not the actual data file size,
    # because the embedding layer was initialized with MAX_CELLS
    expected_raw_spatial_size = None
    if use_gmm:
        # Use MAX_CELLS from config (this is what the embedding expects)
        expected_raw_spatial_size = dataset_config.get('MAX_CELLS', None)
        if expected_raw_spatial_size is None:
            # Fall back to SHAPE_ORIG if MAX_CELLS not set
            expected_raw_spatial_size = dataset_config.get('SHAPE_ORIG', [None, None, None])[2]
        
        if expected_raw_spatial_size is None:
            # Last resort: peek at data file
            first_data_file = os.path.join(flags.data_folder, dataset_config['FILES'][0])
            if os.path.exists(first_data_file):
                with h5.File(first_data_file, "r") as h5f_data:
                    raw_data_sample = h5f_data["showers"][:1].astype(np.float32)
                    if raw_data_sample.ndim >= 3:
                        expected_raw_spatial_size = raw_data_sample.shape[2]
                        print(f"[WARNING] MAX_CELLS not in config, detected from data file: {expected_raw_spatial_size}", flush=True)
        
        print(f"[INFO] Using MAX_CELLS from config for embedding: {expected_raw_spatial_size}", flush=True)

    for i, dataset in enumerate(dataset_config['FILES']):
        # Load data
        data_, gen_info_, layers_ = utils.DataLoader(
                os.path.join(flags.data_folder,dataset),
                hgcal=True,  # Use HGCal loader
                shape=dataset_config['SHAPE_PAD'],
                emax = dataset_config['EMAX'],emin = dataset_config['EMIN'],
                nevts = flags.nevts,
                max_deposit=dataset_config['MAXDEP'], #noise can generate more deposited energy than generated
                logE=dataset_config['logE'],
                showerMap = dataset_config['SHOWERMAP'],
                nholdout = nholdout if (i == len(dataset_config['FILES']) -1 ) else 0,
                dataset_num  = dataset_num,
                orig_shape = orig_shape,
            embed=pre_embed,
            NN_embed=NN_embed,
            config=dataset_config,
            binning_file=geom_file,
            shower_scale=shower_scale,
            max_cells=max_cells,
        )
        # Extract energy from gen_info (first column)
        e_ = gen_info_[:, 0] if gen_info_.ndim > 1 else gen_info_
        
        # Sample from GMM prior if needed (instead of loading pre-generated H5)
        if use_gmm:
            # Detect voxel shape from first data batch
            if detected_voxel_shape is None:
                # Peek at raw data to detect format
                first_data_file = os.path.join(flags.data_folder, dataset_config['FILES'][0])
                with h5.File(first_data_file, "r") as h5f_data:
                    raw_data_sample = h5f_data["showers"][:1].astype(np.float32)
                    if raw_data_sample.ndim == 3:
                        detected_voxel_shape = (raw_data_sample.shape[1], raw_data_sample.shape[2])  # (layers, spatial_size)
                    elif raw_data_sample.ndim == 4:
                        detected_voxel_shape = (raw_data_sample.shape[1], raw_data_sample.shape[2], raw_data_sample.shape[3])  # (layers, H, W)
                    else:
                        raise ValueError(f"Unexpected raw data shape: {raw_data_sample.shape}")
                print(f"[INFO] Detected voxel shape: {detected_voxel_shape}", flush=True)
            
            # Initialize GMM model once
            if gmm_model is None:
                from calodiffusion.train.gmm_hgcal import OriginalMDNConditionalGMM
                from calodiffusion.train.gmm_hgcal import invert_to_physical_logit
                
                gmm_ckpt_path = flags.gmm_prior  # Checkpoint path
                if not os.path.exists(gmm_ckpt_path):
                    raise FileNotFoundError(f"GMM checkpoint not found: {gmm_ckpt_path}")
                
                # Check if file is actually a checkpoint (should be .pt file)
                if not gmm_ckpt_path.endswith('.pt'):
                    raise ValueError(
                        f"GMM checkpoint path should be a .pt file, got: {gmm_ckpt_path}\n"
                        f"Note: You should pass the GMM checkpoint path (e.g., gmm_prior_checkpoint.pt), "
                        f"not the H5 prior file. The code will sample from the checkpoint on-the-fly."
                    )
                
                print(f"[INFO] Loading GMM checkpoint from: {gmm_ckpt_path}", flush=True)
                try:
                    # Load checkpoint (weights_only=False because checkpoint contains model state, mean, std, etc.)
                    ckpt = torch.load(gmm_ckpt_path, map_location=device, weights_only=False)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load GMM checkpoint from {gmm_ckpt_path}.\n"
                        f"Error: {e}\n"
                        f"The file may be corrupted or not a valid PyTorch checkpoint.\n"
                        f"Please verify the checkpoint file or retrain the GMM model."
                    ) from e
                
                # Verify checkpoint has required keys
                required_keys = ['model_state', 'cond_dim', 'data_dim', 'K', 'hidden', 'mean', 'std']
                missing_keys = [k for k in required_keys if k not in ckpt]
                if missing_keys:
                    raise ValueError(
                        f"GMM checkpoint is missing required keys: {missing_keys}\n"
                        f"Found keys: {list(ckpt.keys())}\n"
                        f"The checkpoint may be from an older version. Please retrain the GMM model."
                    )
                
                # Get data_dim from checkpoint (should match what GMM was trained on)
                data_dim = ckpt.get('data_dim', None)
                if data_dim is None:
                    # Calculate from detected shape as fallback
                    if len(detected_voxel_shape) == 2:
                        data_dim = detected_voxel_shape[0] * detected_voxel_shape[1]
                    elif len(detected_voxel_shape) == 3:
                        data_dim = np.prod(detected_voxel_shape)
                    else:
                        raise ValueError(f"Cannot determine data_dim")
                
                gmm_model = OriginalMDNConditionalGMM(
                    cond_dim=ckpt['cond_dim'],
                    data_dim=data_dim,
                    K=ckpt['K'],
                    hidden=ckpt['hidden']
                ).to(device)
                gmm_model.load_state_dict(ckpt["model_state"])
                gmm_model.eval()
                gmm_mean = ckpt['mean']
                gmm_std = ckpt['std']
                print(f"[INFO] Loaded GMM model from checkpoint: {gmm_ckpt_path}", flush=True)
                print(f"[INFO] GMM: cond_dim={ckpt['cond_dim']}, data_dim={data_dim}, K={ckpt['K']}, hidden={ckpt['hidden']}", flush=True)
            
            # Sample from GMM using the same energies as data
            n_events = data_.shape[0]
            e_prior = e_[:n_events] if len(e_) >= n_events else e_
            
            # Convert energies to log-space for GMM conditioning (same as GMM training)
            E_min, E_max = 1.0, 1000.0  # Same as in gmm_hgcal.py
            e_prior_GeV = e_prior / 1000.0  # Convert to GeV if needed
            cE = (np.log10(np.clip(e_prior_GeV, E_min, E_max)) - np.log10(E_min)) / (np.log10(E_max) - np.log10(E_min))
            cE = np.clip(cE, 0.0, 1.0).astype(np.float32)
            cE_tensor = torch.from_numpy(cE).view(-1, 1).to(device)
            
            # Sample from GMM in batches
            batch_size_gmm = 512
            samples_list = []
            with torch.no_grad():
                for j in range(0, len(cE_tensor), batch_size_gmm):
                    c_chunk = cE_tensor[j:j+batch_size_gmm]
                    # Sample from GMM (no anchor needed for pure sampling)
                    s_chunk = gmm_model.sample_pure(
                        c_chunk,
                        n_per_cond=1,
                        x_anchor=None,
                        lam=0.0,  # No tethering during training
                        sigma=0.001,
                    )  # (B, data_dim) on device
                    samples_list.append(s_chunk.cpu())
            
            prior_flat = torch.cat(samples_list, dim=0).numpy()  # (N, data_dim)
            
            # Reshape to match detected voxel shape
            if len(detected_voxel_shape) == 2:
                prior_raw = prior_flat.reshape(n_events, detected_voxel_shape[0], detected_voxel_shape[1])
            elif len(detected_voxel_shape) == 3:
                prior_raw = prior_flat.reshape(n_events, *detected_voxel_shape)
            else:
                raise ValueError(f"Cannot reshape prior to detected_voxel_shape: {detected_voxel_shape}")
            
            # Convert from standardized logit-space to physical
            from calodiffusion.train.gmm_hgcal import invert_to_physical_logit
            prior_raw = invert_to_physical_logit(
                torch.from_numpy(prior_flat).view(n_events, -1),
                e_prior_GeV * 1000.0,  # Back to original units
                gmm_mean,
                gmm_std,
                detected_voxel_shape
            ).numpy()
            
            # Apply shower_scale if needed
            prior_raw = prior_raw * shower_scale
            
            print(f"Prior raw shape: {prior_raw.shape}", flush=True)
            print(f"Data shape after embedding: {data_.shape}", flush=True)
            
            # Check if prior is already in embedded format (matches data shape)
            expected_embedded_shape = data_.shape[1:] if data_.ndim > 1 else None
            prior_already_embedded = (prior_raw.ndim == len(data_.shape) and 
                                     prior_raw.shape[1:] == data_.shape[1:])
            
            # Process prior through same pipeline as data
            # Get energies for preprocessing (use same as data)
            e_prior = e_[:n_events] if len(e_) >= n_events else e_
            
            if prior_already_embedded:
                # Prior is already in embedded format, just preprocess
                print("Prior is already in embedded format, skipping embedding", flush=True)
                prior_preprocessed, _ = HGCal_utils.preprocess_hgcal_shower(
                    prior_raw,
                    e_prior,
                    dataset_config['SHAPE_PAD'],
                    dataset_config['SHOWERMAP'],
                    dataset_num=dataset_num,
                    orig_shape=orig_shape,
                    ecut=dataset_config.get('ECUT', 0),
                    max_deposit=dataset_config['MAXDEP'],
                )
            elif pre_embed and NN_embed is not None:
                # Prior needs to go through embedding
                # Check if prior is in (N, layers, spatial_2d) format - need to flatten to (N, layers, spatial_1d)
                if prior_raw.ndim == 4 and prior_raw.shape[1] == dataset_config['SHAPE_ORIG'][1]:
                    # Prior is in (N, 47, H, W) format - flatten spatial dimensions to (N, 47, H*W)
                    print(f"Reshaping prior from {prior_raw.shape} to raw format for embedding", flush=True)
                    n_events_prior, n_layers, h, w = prior_raw.shape
                    prior_raw = prior_raw.reshape(n_events_prior, n_layers, h * w)
                    print(f"Prior reshaped to: {prior_raw.shape}", flush=True)
                
                # Check if prior has compatible shape for embedding (N, 47, spatial_size)
                if prior_raw.ndim == 3 and prior_raw.shape[1] == dataset_config['SHAPE_ORIG'][1]:
                    # Use the detected expected spatial size (from actual data or config)
                    expected_spatial_size = expected_raw_spatial_size
                    actual_spatial_size = prior_raw.shape[2]
                    
                    # Intelligently handle size mismatch using actual input data size
                    if actual_spatial_size != expected_spatial_size:
                        print(f"[INFO] Prior spatial size mismatch: {actual_spatial_size} vs expected {expected_spatial_size}", flush=True)
                        print(f"[INFO] Intelligently converting prior to match actual data format...", flush=True)
                        
                        # Smart conversion: pad with zeros if smaller, crop if larger
                        # This preserves the existing data and pads/crops appropriately
                        if actual_spatial_size < expected_spatial_size:
                            # Pad with zeros at the end (low-energy cells typically at the end)
                            pad_size = expected_spatial_size - actual_spatial_size
                            prior_raw = np.pad(prior_raw, ((0, 0), (0, 0), (0, pad_size)), 
                                             mode='constant', constant_values=0)
                            print(f"[INFO] Padded prior from {actual_spatial_size} to {expected_spatial_size} cells (added {pad_size} zero cells)", flush=True)
                        else:
                            # Crop to expected size (keep the first cells, which are typically higher energy)
                            prior_raw = prior_raw[:, :, :expected_spatial_size]
                            print(f"[INFO] Cropped prior from {actual_spatial_size} to {expected_spatial_size} cells (removed {actual_spatial_size - expected_spatial_size} cells)", flush=True)
                        
                        print(f"[NOTE] For best results, regenerate GMM prior with spatial size matching your data ({expected_spatial_size} cells)", flush=True)
                    
                    # Prior now has correct shape, apply embedding
                    print(f"Applying embedding to prior: {prior_raw.shape} -> embedded", flush=True)
                    # Convert numpy array to torch tensor and apply embedding
                    # enc_batches already returns numpy array (does .cpu().numpy() internally)
                    prior_embedded = NN_embed.enc_batches(torch.Tensor(prior_raw))
                    # Preprocess after embedding
                    prior_preprocessed, _ = HGCal_utils.preprocess_hgcal_shower(
                        prior_embedded,
                        e_prior,
                        dataset_config['SHAPE_PAD'],
                        dataset_config['SHOWERMAP'],
                        dataset_num=dataset_num,
                        orig_shape=orig_shape,
                        ecut=dataset_config.get('ECUT', 0),
                        max_deposit=dataset_config['MAXDEP'],
                    )
                else:
                    # Prior shape doesn't match - might be from different geometry
                    raise ValueError(
                        f"Prior shape {prior_raw.shape} is incompatible. "
                        f"Expected either embedded shape {expected_embedded_shape} or "
                        f"raw shape (N, {dataset_config['SHAPE_ORIG'][1]}, {dataset_config.get('MAX_CELLS', 'spatial_size')}). "
                        f"Prior may be from a different geometry configuration."
                    )
            else:
                # No embedding, preprocess directly
                prior_preprocessed, _ = HGCal_utils.preprocess_hgcal_shower(
                    prior_raw,
                    e_prior,
                    dataset_config['SHAPE_PAD'],
                    dataset_config['SHOWERMAP'],
                    dataset_num=dataset_num,
                    orig_shape=orig_shape,
                    ecut=dataset_config.get('ECUT', 0),
                    max_deposit=dataset_config['MAXDEP'],
                )
            
            prior_ = prior_preprocessed.astype(np.float32)

        if(i ==0): 
            data = data_
            energies = e_
            if use_gmm:
                prior = prior_
        else:
            data = np.concatenate((data, data_))
            energies = np.concatenate((energies, e_))
            if use_gmm:
                prior = np.concatenate((prior, prior_))
        
    avg_showers = std_showers = E_bins = None
    # NN_embed already initialized above if pre_embed is True

    energies = np.reshape(energies,(-1))    
    
    # Check current data shape
    print(f"Data shape before reshape: {data.shape}")
    print(f"Data size: {data.size}")
    
    dshape = dataset_config['SHAPE_PAD'].copy() if isinstance(dataset_config['SHAPE_PAD'], list) else list(dataset_config['SHAPE_PAD'])
    
    if(not orig_shape): 
        # Calculate expected elements per sample (all dimensions except first)
        expected_elements_per_sample = np.prod(dshape[1:])  # 1 * 47 * 12 * 21 = 11844
        
        # Calculate number of samples from total size
        num_samples = data.size // expected_elements_per_sample
        
        print(f"Expected elements per sample: {expected_elements_per_sample}")
        print(f"Calculated number of samples: {num_samples}")
        print(f"Original target shape: {dshape}")
        
        # Replace -1 with calculated number of samples
        if dshape[0] == -1:
            dshape[0] = num_samples
        
        print(f"Final reshape target: {tuple(dshape)}")
        print(f"Expected total size: {np.prod(dshape)}")
        print(f"Actual data size: {data.size}")
        
        # Verify the reshape is possible
        if data.size % expected_elements_per_sample != 0:
            raise ValueError(
                f"Cannot reshape data: array size {data.size} is not divisible by "
                f"expected elements per sample {expected_elements_per_sample}. "
                f"Data shape: {data.shape}, Target shape per sample: {tuple(dshape[1:])}"
            )
        
        if data.size != np.prod(dshape):
            # Try to reshape with -1 to let numpy figure it out
            print(f"Warning: Size mismatch. Attempting reshape with -1 for first dimension...")
            dshape_auto = [-1] + dshape[1:]
            data = np.reshape(data, tuple(dshape_auto))
            print(f"Reshaped to: {data.shape}")
        else:
            data = np.reshape(data, tuple(dshape))
        
        if use_gmm:
            if prior.size != np.prod(dshape):
                dshape_prior = [-1] + dshape[1:]
                prior = np.reshape(prior, tuple(dshape_prior))
            else:
                prior = np.reshape(prior, tuple(dshape))
    else: 
        data = np.reshape(data, (data.shape[0], -1))
        if use_gmm:
            prior = np.reshape(prior, (prior.shape[0], -1))

    num_data = data.shape[0]
    print("Data Shape " + str(data.shape))
    data_size = data.shape[0]

    # Prepare torch tensors
    torch_data_tensor = torch.from_numpy(data)
    torch_E_tensor = torch.from_numpy(energies)
    
    if use_gmm:
        prior_flat_all = torch.from_numpy(prior).cpu().view(torch_data_tensor.shape)
        torch_data_tensor = torch_data_tensor.cpu()
        torch_E_tensor = torch_E_tensor.cpu()
        print("DATA mean, sum, std, shape", torch.mean(torch_data_tensor), torch.sum(prior_flat_all),torch.std(torch_data_tensor), torch_data_tensor.shape)
        print("PRIOR mean, sum, std, shape", torch.mean(prior_flat_all),torch.sum(prior_flat_all), torch.std(prior_flat_all), prior_flat_all.shape)
        print(f'prior shape {prior_flat_all.shape}')
    
    del data
    if use_gmm:
        del prior

    # Create dataset
    if use_gmm:
        torch_dataset = torchdata.TensorDataset(torch_data_tensor, torch_E_tensor, prior_flat_all)
        del prior_flat_all
    else:
        torch_dataset = torchdata.TensorDataset(torch_E_tensor, torch_data_tensor)
    
    nTrain = int(round(flags.frac * num_data))
    nVal = num_data - nTrain
    train_dataset, val_dataset = torch.utils.data.random_split(torch_dataset, [nTrain, nVal])

    loader_train = torchdata.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    loader_val = torchdata.DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

    del torch_data_tensor, torch_E_tensor, train_dataset, val_dataset
    
    # Use provided checkpoint folder or default to ../models/{CHECKPOINT_NAME}_{model}/
    if flags.checkpoint is not None:
        checkpoint_folder = flags.checkpoint
    else:
        checkpoint_folder = '../models/{}_{}/'.format(dataset_config['CHECKPOINT_NAME'],flags.model)
    
    # Ensure checkpoint folder ends with / for consistency
    if not checkpoint_folder.endswith('/'):
        checkpoint_folder += '/'
    
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint = dict()
    checkpoint_path = os.path.join(checkpoint_folder, "checkpoint.pth")
    if(flags.load and os.path.exists(checkpoint_path)): 
        print("Loading training checkpoint from %s" % checkpoint_path, flush = True)
        try:
            checkpoint = torch.load(checkpoint_path, map_location = device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location = device)
        print(checkpoint.keys())

    if(flags.model == "Diffu"):
        shape = dataset_config['SHAPE_PAD'][1:] if (not orig_shape) else dataset_config['SHAPE_ORIG'][1:]
        model = CaloDiffu(shape, config=dataset_config, training_obj = training_obj, NN_embed = NN_embed, nsteps = dataset_config['NSTEPS'],
                cold_diffu = cold_diffu, avg_showers = avg_showers, std_showers = std_showers, E_bins = E_bins ).to(device = device)

        #sometimes save only weights, sometimes save other info
        if('model_state_dict' in checkpoint.keys()): model.load_state_dict(checkpoint['model_state_dict'])
        elif(len(checkpoint.keys()) > 1): model.load_state_dict(checkpoint)
    else:
        print("Model %s not supported!" % flags.model)
        exit(1)

    # Enable multi-GPU training if multiple GPUs are available
    # DISABLE DataParallel for MeanFlow: jvp (used in loss computation) doesn't work with DataParallel
    # See: https://github.com/pytorch/pytorch/issues/102197
    # However, we use manual multi-GPU splitting in _parallel_jvp to utilize multiple GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 1:
        print(f"[INFO] {num_gpus} GPUs detected for MeanFlow training", flush=True)
        print(f"[INFO] DataParallel DISABLED (jvp incompatible), but using manual multi-GPU batch splitting", flush=True)
        print(f"[INFO] Batches will be split across {num_gpus} GPUs during jvp computation", flush=True)
        print(f"[INFO] Model will run on primary GPU: {device}", flush=True)
    else:
        print(f"[INFO] Using single GPU/CPU: {device}", flush=True)
    
    # Don't wrap model in DataParallel - jvp doesn't work with it
    # But _parallel_jvp will manually split batches across GPUs
    
    # Helper function to get the actual model (handles DataParallel)
    def get_model(model):
        if isinstance(model, nn.DataParallel):
            return model.module
        return model
    
    # Helper function to get model state dict (handles DataParallel)
    def get_model_state_dict(model):
        if isinstance(model, nn.DataParallel):
            return model.module.state_dict()
        return model.state_dict()

    # Backup config file
    os.system('cp {} {}'.format(flags.config,checkpoint_folder)) # bkp of config file

    early_stopper = utils.EarlyStopper(patience = dataset_config['EARLYSTOP'], mode = 'diff', min_delta = 1e-5)
    if('early_stop_dict' in checkpoint.keys() and not flags.reset_training): early_stopper.__dict__ = checkpoint['early_stop_dict']
    print(early_stopper.__dict__)
    

    criterion = nn.MSELoss().to(device = device)

    optimizer = optim.Adam(model.parameters(), lr = float(dataset_config["LR"]))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, factor = 0.1, patience = 15)
    if('optimizer_state_dict' in checkpoint.keys() and not flags.reset_training): optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if('scheduler_state_dict' in checkpoint.keys() and not flags.reset_training): scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    training_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    start_epoch = 0
    min_validation_loss = 99999.
    if('train_loss_hist' in checkpoint.keys() and not flags.reset_training): 
        training_losses = checkpoint['train_loss_hist']
        training_losses = np.concatenate((training_losses,np.zeros(num_epochs)))
        val_losses = checkpoint['val_loss_hist']
        val_losses = np.concatenate((val_losses,np.zeros(num_epochs)))
        start_epoch = checkpoint['epoch'] + 1
    
    MILESTONE_EPOCHS = {5, 10, 20, 30, 50, 80, 100, 200, 500}
    MAX_GRAD_NORM = 1.0
    
    #training loop
    for epoch in range(start_epoch, num_epochs):
        print("Beginning epoch %i" % epoch, flush=True)
        for i, param in enumerate(model.parameters()):
            break
        train_loss = 0

        model.train()
        train_pbar = tqdm(enumerate(loader_train, 0),
                          unit="batch",
                          total=len(loader_train))
        for i, batch in train_pbar:
            model.zero_grad()
            optimizer.zero_grad()

            if use_gmm:
                data, E, prior = batch
                data = data.to(device = device)
                E = E.to(device = device)
                prior = prior.to(device = device)
            else:
                E, data = batch
                data = data.to(device = device)
                E = E.to(device = device)

            noise = torch.randn_like(data)

            # Get the actual model (handles DataParallel)
            actual_model = get_model(model)
            loss_pidm_vec = None
            if use_gmm:
                # For GMM, check if there's a PIDM version
                if hasattr(actual_model, 'compute_loss_meanflow_gmm_pidm'):
                    loss_vec, loss_ref, loss_pidm_vec = actual_model.compute_loss_meanflow_gmm_pidm(data, E, gmm_prior=prior, noise = noise, energy_loss_scale = energy_loss_scale)
                    batch_loss = loss_vec.mean() + weight_pidm * loss_pidm_vec.mean()
                else:
                    loss_vec, loss_ref = actual_model.compute_loss_meanflow_gmm(data, E, gmm_prior=prior, noise = noise, energy_loss_scale = energy_loss_scale)
                    batch_loss = loss_vec.mean()
            else:
                # Use PIDM version which includes physics constraints
                loss_vec, loss_ref, loss_pidm_vec = actual_model.compute_loss_meanflow_pidm(data, E, noise = noise, energy_loss_scale = energy_loss_scale)
                batch_loss = loss_vec.mean() + weight_pidm * loss_pidm_vec.mean()
            
            batch_loss_ref = loss_ref.mean()
            
            # Save loss value before deleting tensors
            batch_loss_value = batch_loss.item()
            batch_loss_ref_value = batch_loss_ref.item()
            loss_pidm_value = loss_pidm_vec.mean().item() if loss_pidm_vec is not None else None
            
            batch_loss.backward()

            # --- grad clipping (after backward, before step) ---
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)

            optimizer.step()
            
            # Clear cache aggressively after each step (jvp is very memory-intensive)
            # Delete intermediate tensors to free memory
            del batch_loss, loss_vec, loss_ref, noise
            if loss_pidm_vec is not None:
                del loss_pidm_vec
            if use_gmm:
                del prior
            del data, E
            
            # Clear cache every step (not just every 10) - jvp needs aggressive cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Synchronize to ensure memory is freed before next iteration
                torch.cuda.synchronize()
            train_loss += batch_loss_value

            # progress bar
            if loss_pidm_value is not None:
                train_pbar.set_postfix(
                    loss=f"{batch_loss_value:.4f}",
                    ref_loss=f"{batch_loss_ref_value:.4f}",
                    pidm_loss=f"{loss_pidm_value:.4f}",
                )
            else:
                train_pbar.set_postfix(
                    loss=f"{batch_loss_value:.4f}",
                    ref_loss=f"{batch_loss_ref_value:.4f}",
                )

        train_loss = train_loss/len(loader_train)
        training_losses[epoch] = train_loss
        print("loss: "+ str(train_loss))

        val_loss = 0
        model.eval()
        val_pbar = tqdm(enumerate(loader_val, 0),
                        unit="batch",
                        total=len(loader_val))
        for i, batch in val_pbar:
            if use_gmm:
                vdata, vE, vprior = batch
                vdata = vdata.to(device=device)
                vE = vE.to(device = device)
                vprior = vprior.to(device = device)
            else:
                vE, vdata = batch
                vdata = vdata.to(device=device)
                vE = vE.to(device = device)

            noise = torch.randn_like(vdata)
            # Get the actual model (handles DataParallel)
            actual_model = get_model(model)
            if(cold_diffu): noise = actual_model.gen_cold_image(vE, cold_noise_scale, noise)

            if use_gmm:
                # For GMM, check if there's a PIDM version
                if hasattr(actual_model, 'compute_loss_meanflow_gmm_pidm'):
                    loss_vec, loss_ref, loss_pidm_vec = actual_model.compute_loss_meanflow_gmm_pidm(vdata, vE, gmm_prior=vprior, noise = noise, energy_loss_scale = energy_loss_scale)
                    batch_loss = loss_vec.mean() + weight_pidm * loss_pidm_vec.mean()
                else:
                    loss_vec, loss_ref = actual_model.compute_loss_meanflow_gmm(vdata, vE, gmm_prior=vprior, noise = noise, energy_loss_scale = energy_loss_scale)
                    batch_loss = loss_vec.mean()
            else:
                # Use PIDM version which includes physics constraints
                loss_vec, loss_ref, loss_pidm_vec = actual_model.compute_loss_meanflow_pidm(vdata, vE, noise = noise, energy_loss_scale = energy_loss_scale)
                batch_loss = loss_vec.mean() + weight_pidm * loss_pidm_vec.mean()
            
            batch_loss_ref = loss_ref.mean()

            # Save loss values before deleting tensors
            batch_loss_value = batch_loss.item()
            batch_loss_ref_value = batch_loss_ref.item()
            
            val_loss += batch_loss_value
            
            val_pbar.set_postfix(
                loss=f"{batch_loss_value:.4f}",
                ref_loss=f"{batch_loss_ref_value:.4f}",
            )
            
            if use_gmm:
                del vdata, vE, vprior, noise, batch_loss, batch_loss_ref
            else:
                del vdata, vE, noise, batch_loss, batch_loss_ref

        val_loss = val_loss/len(loader_val)
        val_losses[epoch] = val_loss
        print("val_loss: "+ str(val_loss), flush = True)

        scheduler.step(torch.tensor([train_loss]))

        if(val_loss < min_validation_loss):
            torch.save(get_model_state_dict(model), os.path.join(checkpoint_folder, 'best_val.pth'))
            min_validation_loss = val_loss

        if(early_stopper.early_stop(val_loss - train_loss)):
            print("Early stopping!")
            break

        # save the model
        model.eval()
        print("SAVING")
        
        #save full training state so can be resumed
        torch.save({
            'epoch': epoch,
            'model_state_dict': get_model_state_dict(model),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss_hist': training_losses,
            'val_loss_hist': val_losses,
            'early_stop_dict': early_stopper.__dict__,
            }, checkpoint_path)
        
        # --- NEW: milestone checkpoints at specific epochs ---
        epoch_1based = epoch + 1
        if epoch_1based in MILESTONE_EPOCHS:
            tag = f"epoch_{epoch_1based:03d}"
            print(f"SAVING in {tag}")
            # full state (resume-able)
            torch.save({
                'epoch': epoch,
                'model_state_dict': get_model_state_dict(model),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss_hist': training_losses,
                'val_loss_hist': val_losses,
                'early_stop_dict': early_stopper.__dict__,
            }, os.path.join(checkpoint_folder, f"checkpoint_{tag}.pt"))

        with open(checkpoint_folder + "/training_losses.txt","w") as tfileout:
            tfileout.write("\n".join("{}".format(tl) for tl in training_losses)+"\n")
        with open(checkpoint_folder + "/validation_losses.txt","w") as vfileout:
            vfileout.write("\n".join("{}".format(vl) for vl in val_losses)+"\n")

    print("Saving to %s" % checkpoint_folder, flush=True)
    torch.save(get_model_state_dict(model), os.path.join(checkpoint_folder, 'final.pth'))

    with open(checkpoint_folder + "/training_losses.txt","w") as tfileout:
        tfileout.write("\n".join("{}".format(tl) for tl in training_losses)+"\n")
    with open(checkpoint_folder + "/validation_losses.txt","w") as vfileout:
        vfileout.write("\n".join("{}".format(vl) for vl in val_losses)+"\n")
