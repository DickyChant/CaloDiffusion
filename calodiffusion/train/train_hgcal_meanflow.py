import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from calodiffusion.utils import utils
from calodiffusion.models.CaloDiffu import CaloDiffu


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
        print(f"Using GMM prior file: {flags.gmm_prior}", flush=True)

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

    batch_size = dataset_config['BATCH']
    num_epochs = dataset_config['MAXEPOCH']
    early_stop = dataset_config['EARLYSTOP']
    training_obj = dataset_config.get('TRAINING_OBJ', 'noise_pred')
    loss_type = dataset_config.get("LOSS_TYPE", "l2")
    dataset_num = dataset_config.get('DATASET_NUM', 5)
    shower_embed = dataset_config.get('SHOWER_EMBED', '')
    orig_shape = ('orig' in shower_embed)
    energy_loss_scale = dataset_config.get('ENERGY_LOSS_SCALE', 0.0)

    data = []
    energies = []
    prior = [] if use_gmm else None

    for i, dataset in enumerate(dataset_config['FILES']):
        if use_gmm:
            data_, prior_, e_ = utils.DataLoader_GMM(
                os.path.join(flags.data_folder,dataset),
                flags.gmm_prior,
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
            )
        else:
            data_, e_ = utils.DataLoader(
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
            )

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
    NN_embed = None

    dshape = dataset_config['SHAPE_PAD']
    energies = np.reshape(energies,(-1))    
    if(not orig_shape): 
        data = np.reshape(data,dshape)
        if use_gmm:
            prior = np.reshape(prior,dshape)
    else: 
        data = np.reshape(data, (len(data), -1))
        if use_gmm:
            prior = np.reshape(prior, (len(data), -1))

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

    os.system('cp CaloDiffu.py {}'.format(checkpoint_folder)) # bkp of model def
    os.system('cp models.py {}'.format(checkpoint_folder)) # bkp of model def
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

            if use_gmm:
                loss_vec, loss_ref = model.compute_loss_meanflow_gmm(data, E, gmm_prior=prior, noise = noise, energy_loss_scale = energy_loss_scale)
            else:
                loss_vec, loss_ref = model.compute_loss_meanflow(data, E, noise = noise, energy_loss_scale = energy_loss_scale)
            
            batch_loss = loss_vec.mean()
            batch_loss_ref = loss_ref.mean()
            
            batch_loss.backward()

            # --- grad clipping (after backward, before step) ---
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)

            optimizer.step()
            train_loss += batch_loss.item()

            # progress bar
            train_pbar.set_postfix(
                loss=f"{batch_loss.item():.4f}",
                ref_loss=f"{batch_loss_ref.item():.4f}",
            )

            if use_gmm:
                del data, E, prior, noise, batch_loss, batch_loss_ref
            else:
                del data, E, noise, batch_loss, batch_loss_ref

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
            if(cold_diffu): noise = model.gen_cold_image(vE, cold_noise_scale, noise)

            if use_gmm:
                loss_vec, loss_ref = model.compute_loss_meanflow_gmm(vdata, vE, gmm_prior=vprior, noise = noise, energy_loss_scale = energy_loss_scale)
            else:
                loss_vec, loss_ref = model.compute_loss_meanflow(vdata, vE, noise = noise, energy_loss_scale = energy_loss_scale)
            
            batch_loss = loss_vec.mean()
            batch_loss_ref = loss_ref.mean()

            val_loss+=batch_loss.item()
            
            val_pbar.set_postfix(
                loss=f"{batch_loss.item():.4f}",
                ref_loss=f"{batch_loss_ref.item():.4f}",
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
            torch.save(model.state_dict(), os.path.join(checkpoint_folder, 'best_val.pth'))
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
            'model_state_dict': model.state_dict(),
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
                'model_state_dict': model.state_dict(),
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
    torch.save(model.state_dict(), os.path.join(checkpoint_folder, 'final.pth'))

    with open(checkpoint_folder + "/training_losses.txt","w") as tfileout:
        tfileout.write("\n".join("{}".format(tl) for tl in training_losses)+"\n")
    with open(checkpoint_folder + "/validation_losses.txt","w") as vfileout:
        vfileout.write("\n".join("{}".format(vl) for vl in val_losses)+"\n")
