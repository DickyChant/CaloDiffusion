import os
import math
import h5py as h5
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import glob

from tqdm import trange



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =========================
# 0) DATA LOADING + PREPROC
# =========================
def load_showers_from_h5(paths, nevts=-1, evt_start=0,
                         dataset_key='showers', energy_key='incident_energies',
                         gen_info_key='gen_info', gen_info_energy_col=0,
                         target_spatial_shape=(40, 40), auto_reshape=True):
    """
    paths: str or list[str] of HDF5 files; concatenates until reaching nevts.
    Returns (showers, energies) before preprocessing.
    
    Supports multiple H5 file formats:
    1. Standard: 'showers' (N, 47, 40, 40) and 'incident_energies' (N,)
    2. Alternative: 'showers' (N, 47, 2200) and 'gen_info' (N, 3) with energy in column 0
    3. Flattened: 'showers' (N, 75200) and 'incident_energies' (N,)
    
    Args:
        paths: file path(s)
        nevts: number of events to load (-1 = all)
        evt_start: starting event index
        dataset_key: key for shower data (default: 'showers')
        energy_key: key for energy data (default: 'incident_energies')
        gen_info_key: key for gen_info if using alternative format (default: 'gen_info')
        gen_info_energy_col: column index in gen_info for energy (default: 0)
        target_spatial_shape: target spatial dimensions (default: (40, 40))
        auto_reshape: if True, automatically reshape spatial dimensions (default: True)
    """
    if isinstance(paths, str): paths = [paths]
    showers_list, energies_list = [], []
    need = None if nevts is None or nevts < 0 else int(nevts)
    got = 0
    file_stats = []  # Track stats per file
    
    for file_idx, p in enumerate(paths, 1):
        with h5.File(p, "r") as f:
            # Check available keys
            available_keys = list(f.keys())
            
            # Load showers
            if dataset_key in available_keys:
                sh = f[dataset_key][:]
            else:
                raise KeyError(f"Dataset key '{dataset_key}' not found in {p}. Available keys: {available_keys}")
            
            # Load energies - try multiple formats
            # If energy_key is set to gen_info_key, extract from gen_info array
            if energy_key == gen_info_key and gen_info_key in available_keys:
                # Extract energy column from gen_info array
                gen_info = f[gen_info_key][:]
                if gen_info.ndim == 2 and gen_info.shape[1] > gen_info_energy_col:
                    en = gen_info[:, gen_info_energy_col]
                    if file_idx == 1:  # Only print once
                        print(f"[INFO] Using energy from {gen_info_key} column {gen_info_energy_col}")
                else:
                    raise ValueError(f"gen_info shape {gen_info.shape} incompatible with energy_col={gen_info_energy_col}")
            elif energy_key in available_keys:
                en = f[energy_key][:]
            elif gen_info_key in available_keys:
                # Fallback: use gen_info if energy_key not found
                gen_info = f[gen_info_key][:]
                if gen_info.ndim == 2 and gen_info.shape[1] > gen_info_energy_col:
                    en = gen_info[:, gen_info_energy_col]
                    if file_idx == 1:  # Only print once
                        print(f"[INFO] '{energy_key}' not found, using energy from {gen_info_key} column {gen_info_energy_col}")
                else:
                    raise ValueError(f"gen_info shape {gen_info.shape} incompatible with energy_col={gen_info_energy_col}")
            else:
                raise KeyError(f"Neither '{energy_key}' nor '{gen_info_key}' found in {p}. Available keys: {available_keys}")
            
            original_size = sh.shape[0]
            
            # Handle event slicing
            if evt_start > 0:
                sh = sh[evt_start:]; en = en[evt_start:]; evt_start = 0
            if need is not None:
                take = min(need - got, sh.shape[0])
                sh = sh[:take]; en = en[:take]
            
            # Reshape showers if needed
            if auto_reshape and sh.ndim == 3:
                # Shape is (N, 47, spatial_size)
                N, n_layers, spatial_size = sh.shape
                target_size = np.prod(target_spatial_shape)
                
                if spatial_size != target_size:
                    # Try to reshape to target spatial dimensions
                    if spatial_size == 2200:  # Common case: 40x55 or 44x50
                        # Try reshaping to (40, 55) first, then crop/pad to (40, 40)
                        try:
                            sh_reshaped = sh.reshape(N, n_layers, 40, 55)
                            # Crop the last dimension from 55 to 40
                            sh = sh_reshaped[:, :, :, :40]
                            print(f"[INFO] Reshaped showers from (N, {n_layers}, {spatial_size}) to (N, {n_layers}, 40, 40)")
                        except:
                            # If reshape fails, try padding or other methods
                            print(f"[WARNING] Could not auto-reshape (N, {n_layers}, {spatial_size}) to (N, {n_layers}, {target_spatial_shape[0]}, {target_spatial_shape[1]})")
                            print("[WARNING] You may need to set auto_reshape=False and handle reshaping manually")
                    elif spatial_size == target_size:
                        # Already correct size, just reshape to 2D spatial
                        sh = sh.reshape(N, n_layers, *target_spatial_shape)
                    else:
                        print(f"[WARNING] Unexpected spatial size {spatial_size}, expected {target_size} or 2200")
                        print("[WARNING] Attempting to reshape assuming spatial_size can be factorized...")
                        # Try to find factors
                        factors = [i for i in range(1, int(np.sqrt(spatial_size)) + 1) if spatial_size % i == 0]
                        if factors:
                            h, w = factors[-1], spatial_size // factors[-1]
                            sh = sh.reshape(N, n_layers, h, w)
                            # Crop/pad to target
                            if h > target_spatial_shape[0] or w > target_spatial_shape[1]:
                                sh = sh[:, :, :target_spatial_shape[0], :target_spatial_shape[1]]
                            elif h < target_spatial_shape[0] or w < target_spatial_shape[1]:
                                # Pad with zeros
                                pad_h = target_spatial_shape[0] - h
                                pad_w = target_spatial_shape[1] - w
                                sh = np.pad(sh, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode='constant')
                            print(f"[INFO] Reshaped from (N, {n_layers}, {spatial_size}) via (N, {n_layers}, {h}, {w}) to (N, {n_layers}, {target_spatial_shape[0]}, {target_spatial_shape[1]})")
            
            loaded_size = sh.shape[0]
            showers_list.append(sh); energies_list.append(en)
            got += loaded_size
            
            # Track statistics
            file_stats.append({
                'file': os.path.basename(p),
                'original_size': original_size,
                'loaded_size': loaded_size,
                'energy_range': (float(en.min()), float(en.max()))
            })
            
            if need is not None and got >= need: break
    
    if not showers_list:
        raise RuntimeError("No data loaded; check paths/keys.")
    
    # Print per-file statistics
    if len(file_stats) > 1:
        print("\nPer-file loading statistics:")
        for stat in file_stats:
            print(f"  {stat['file']}: {stat['loaded_size']:,} / {stat['original_size']:,} events "
                  f"(E: {stat['energy_range'][0]:.2f}-{stat['energy_range'][1]:.2f} GeV)")
    
    showers = np.concatenate(showers_list, 0)
    energies = np.concatenate(energies_list, 0)
    
    return showers, energies


def reverse_logit_np(x, alpha=1e-6):
    exp = np.exp(x)
    o = exp / (1.0 + exp)
    o = (o - alpha) / (1.0 - 2.0 * alpha)
    return o

def reverse_logit_torch(x, alpha=1e-6):
    exp = torch.exp(x)
    o = exp / (1.0 + exp)
    o = (o - alpha) / (1.0 - 2.0 * alpha)
    return o



def logit(x, alpha = 1e-6):
    o = alpha + (1 - 2*alpha)*x
    o = np.ma.log(o/(1-o)).filled(0)    
    return o

def preprocess_showers_and_energies_logit(
    data, true_energies, eps=1e-8,
    #voxel_shape=(28,40,40),
    voxel_shape=(47,40,40),
    energy_scale_div=1000.0,
    #energy_log_base_range=(50.0, 100.0)
    energy_log_base_range=(1.0, 1000.0)
):
    import numpy as np

    # Convert to GeV (if your HDF5 is in MeV)
    X = np.asarray(data, dtype=np.float64) / energy_scale_div
    E = np.asarray(true_energies, dtype=np.float64).reshape(-1) / energy_scale_div  # (N,)

    # Ensure voxel shape
    if X.ndim == 2 and X.shape[1] == np.prod(voxel_shape):
        X = X.reshape(-1, *voxel_shape)  # (N,30,30,30)
    elif not (X.ndim == 4 and X.shape[1:] == voxel_shape):
        raise ValueError(f"Unexpected data shape {X.shape}; expected (N,30,30,30) or flat D={np.prod(voxel_shape)}.")

    # --- FIX: broadcast energy over voxels ---
    denom = 5.0 * E.reshape(-1, 1, 1, 1)   # (N,1,1,1)
    scaled = X / (denom + eps)             # per-event scaling

    # Log + standardize
    logv = logit(scaled)
    #mean =  -13.746323
    #std = 0.67718875
    
    mean =  -13.780706
    std = 0.47605845

    X_std = ((logv - mean) / (std)).astype(np.float32)

    # Continuous energy condition in ~[0,1]
    E_min, E_max = energy_log_base_range  # (100, 1000) GeV, adjust if you changed units upstream
    # Note: we already divided by energy_scale_div → E is in GeV here.
    cE = (np.log10(E) - np.log10(E_min / energy_scale_div)) / (
          np.log10(E_max / energy_scale_div) - np.log10(E_min / energy_scale_div)
    )
    cE = cE.astype(np.float32)

    return X_std, cE, (mean, std)


# =====================
# 1) SHARED UTILITIES
# =====================
def logsumexp(x, dim=-1, keepdim=False):
    m = x.max(dim=dim, keepdim=True).values
    out = (x - m).exp().sum(dim=dim, keepdim=True).log() + m
    return out if keepdim else out.squeeze(dim)

def kmeanspp_init(X, K, seed=0):
    if seed is not None: torch.manual_seed(seed)
    N, D = X.shape
    idx = torch.randint(N, (1,), device=X.device)
    centers = [X[idx]]
    for _ in range(1, K):
        d2 = torch.stack([((X - c)**2).sum(-1) for c in centers], 0).min(0).values
        probs = (d2 / (d2.sum() + 1e-12)).clamp_min(1e-12)
        idx = torch.multinomial(probs, 1)
        centers.append(X[idx])
    return torch.cat(centers, dim=0)

class OriginalMDNConditionalGMM(nn.Module):
    """
    Conditional GMM with diagonal covariances:
      params(c) = {π_k(c), μ_k(c), logσ_k^2(c)} via a small MLP on c.
    Trained by MLE (negative log-likelihood).
    """
    def __init__(self, cond_dim, data_dim, K, x_max=math.log(1e3), hidden=256):
        super().__init__()
        self.cond_dim, self.data_dim, self.K = cond_dim, data_dim, K
        H = hidden
        self.net = nn.Sequential(
            nn.Linear(cond_dim, H), nn.GELU(),
            nn.Linear(H, H), nn.GELU(),
            nn.Linear(H, K*(1 + 2*data_dim))  # logits + means + log_vars
        )
        self.x_max = x_max



    def _split_params(self, raw):
        """
        raw: [N, K*(1+2D)] -> logits [N,K], mu [N,K,D], log_var [N,K,D]
        """
        N = raw.shape[0]; K, D = self.K, self.data_dim
        logits = raw[:, :K]
        mu = raw[:, K:K+K*D].view(N, K, D)
        log_var = raw[:, K+K*D:].view(N, K, D)
        # Stabilize variances
        log_var = torch.clamp(log_var, min=math.log(1e-6), max=math.log(1e3))
        return logits, mu, log_var

    def forward_params(self, c):
        raw = self.net(c)
        logits, mu, log_var = self._split_params(raw)
        pi = F.softmax(logits, dim=-1)  # [N,K]
        return pi, mu, log_var

    def comp_logpdf(self, x, c):
        """
        x: [N,D], c: [N,cond_dim]
        returns log p(x|c) mixture log-likelihood per sample: [N]
        """
        pi, mu, log_var = self.forward_params(c)      # [N,K], [N,K,D], [N,K,D]
        inv_var = torch.exp(-log_var)
        # [N,K]: sum over D
        quad = 0.5 * ((x[:,None,:] - mu)**2 * inv_var).sum(-1)
        log_det = 0.5 * log_var.sum(-1)               # [N,K]
        const = 0.5 * self.data_dim * math.log(2*math.pi)
        comp = -(quad + log_det + const)              # [N,K]
        log_mix = torch.logsumexp(torch.log(pi + 1e-12) + comp, dim=-1)  # [N]
        return log_mix

    def nll(self, x, c):
        return -self.comp_logpdf(x, c).mean()
    
    def nll_with_aux(self, x, c,
                     anchor_weight=0.0,      # λ₁: mean-matching
                     var_weight=1e-4,        # λ₂: variance shrink
                     moment2_weight=0.0,     # λ₃: 2nd-moment match (optional)
                     energy_weight=0.0,      # λ₄: total-energy match (optional)
                     voxel_shape=None):
        """
        Total loss = NLL + λ₁*||E[x|c]-x||² + λ₂*E[σ²] + λ₃*||E[x²|c]-x²||² + λ₄*||sum(E[x|c]) - sum(x)||²
        All terms are differentiable and keep the *model parameters* (π, μ, σ²) close to data.
        """
        # NLL (exact)
        nll = -self.comp_logpdf(x, c).mean()

        # Forward params (no extra compute if you refactor comp_logpdf to return these)
        pi, mu, log_var = self.forward_params(c)           # [N,K], [N,K,D], [N,K,D]
        var = log_var.exp()

        # 1) Mean of mixture: E[x|c] = Σ π μ
        m_hat = (pi.unsqueeze(-1) * mu).sum(dim=1)         # [N,D]
        L_mean = F.mse_loss(m_hat, x)

        # 2) (Optional) second moment match: E[x²|c] = Σ π (μ² + σ²)
        if moment2_weight > 0.0:
            second_hat = (pi.unsqueeze(-1) * (mu**2 + var)).sum(dim=1)  # [N,D]
            L_m2 = F.mse_loss(second_hat, x**2)
        else:
            L_m2 = x.new_tensor(0.0)

        # 3) Variance shrink (keeps σ² sane; avoids huge samples)
        L_var = var.mean()

        # 4) (Optional) total energy match in voxel space
        if (energy_weight > 0.0) and (voxel_shape is not None):
            N, D = x.shape
            Z = m_hat.view(N, *voxel_shape)
            Zx = x.view(N, *voxel_shape)
            E_hat = Z.sum(dim=(1,2,3))
            E_true= Zx.sum(dim=(1,2,3))
            L_E = F.mse_loss(E_hat, E_true)
        else:
            L_E = x.new_tensor(0.0)

        return nll + anchor_weight*L_mean + var_weight*L_var + moment2_weight*L_m2 + energy_weight*L_E

    @torch.no_grad()
    def sample(self, c, n_per_cond=1, x_anchor=None, lam=0.3, sigma=0.001):
        """
        Sample per provided condition c: [N,cond_dim] -> [N*n_per_cond, D]
        """
        c = c.to(next(self.parameters()).device)
        pi, mu, log_var = self.forward_params(c)       # [N,K], [N,K,D], [N,K,D]
        N, K, D = mu.shape
        out = []
        for i in range(N):
            ks = torch.multinomial(pi[i], n_per_cond, replacement=True) # [n]
            mu_i = mu[i, ks]                     # [n,D]
            var_i = log_var[i, ks].exp()         # [n,D]
            z = mu_i + torch.randn_like(mu_i) * var_i.sqrt()
            #z = torch.clamp(z, min=1e-8, max=self.x_max)
            #print(sum(z))
            if (x_anchor is not None) and (lam > 0.0):
                xa = x_anchor[i].view(1, -1).to(z.device)                     # (1,D)
                if n_per_cond > 1:
                    xa = xa.repeat(n_per_cond, 1).to(z.device)               # (n,D)
                z  = torch.lerp(z, xa + sigma*torch.randn_like(z).to(z.device), float(lam))
                #print(sum(z))
            out.append(z)
        return torch.cat(out, 0)  
    
    @torch.no_grad()
    def sample_pure(self, c, n_per_cond=1, x_anchor=None, lam=0.3, sigma=0.001):
        """
        Sample per provided condition c: [N,cond_dim] -> [N*n_per_cond, D]
        """
        c = c.to(next(self.parameters()).device)
        pi, mu, log_var = self.forward_params(c)       # [N,K], [N,K,D], [N,K,D]
        N, K, D = mu.shape
        out = []
        for i in range(N):
            ks = torch.multinomial(pi[i], n_per_cond, replacement=True) # [n]
            mu_i = mu[i, ks]                     # [n,D]
            var_i = log_var[i, ks].exp()         # [n,D]
            z = mu_i + torch.randn_like(mu_i) * var_i.sqrt()
            out.append(z)
        return torch.cat(out, 0)  



# ==================================
# 4) PCA / FID / MMD (unchanged)
# ==================================

def pca_proj(X, mu, V): return (X - mu) @ V

@torch.no_grad()
def frechet_distance(F_real, F_fake, eps=1e-6):
    mu_r = F_real.mean(0); mu_f = F_fake.mean(0)
    Cr = torch.cov(F_real.T) + eps*torch.eye(F_real.shape[1], device=F_real.device)
    Cf = torch.cov(F_fake.T) + eps*torch.eye(F_fake.shape[1], device=F_fake.device)
    diff = (mu_r - mu_f)
    evals, evecs = torch.linalg.eigh(Cr @ Cf)
    evals = evals.clamp_min(0).sqrt()
    Tr = torch.trace(Cr + Cf - 2 * (evecs @ torch.diag(evals) @ evecs.T))
    return (diff @ diff) + Tr

@torch.no_grad()
def mmd_rbf(X, Y):
    Z = torch.cat([X, Y], 0)
    d2 = torch.cdist(Z, Z, p=2)**2
    med = torch.median(d2[d2>0])
    gamma = 1.0 / (2 * med.clamp_min(1e-12))
    def k(a,b): return torch.exp(-gamma * torch.cdist(a,b, p=2)**2)
    Kxx = k(X,X); Kyy = k(Y,Y); Kxy = k(X,Y)
    n = X.shape[0]; m = Y.shape[0]
    return Kxx.sum()/(n*n) + Kyy.sum()/(m*m) - 2*Kxy.sum()/(n*m)

def pca_fit(X, r, max_samples=100):
    """
    X: tensor of shape (N, D) or (N,45,50,18) in standardized space
    r: number of principal components
    max_samples: optional cap on N used to fit PCA (for speed/memory)
    Returns:
      mu: (D,) mean vector (CPU, float32)
      V:  (D, r) principal directions (CPU, float32)
    """
    with torch.no_grad():
        Xf = X.detach()

        # Flatten if needed
        if Xf.ndim > 2:
            N = Xf.shape[0]
            Xf = Xf.view(N, -1)

        # Optional subsampling for stability
        N = Xf.shape[0]
        if N > max_samples:
            idx = torch.randperm(N, device=Xf.device)[:max_samples]
            Xf = Xf[idx]

        # Move to float32 on CPU to avoid GPU SVD issues
        Xf = Xf.to(torch.float32).cpu()

        # Center
        mu = Xf.mean(dim=0, keepdim=True)          # (1, D)
        X0 = Xf - mu                               # (Ns, D)

        # SVD on CPU
        U, S, Vt = torch.linalg.svd(X0, full_matrices=False)  # Vt: (D, D)
        V = Vt[:r].T.contiguous()                 # (D, r)

        # Return mu as (D,)
        mu = mu.view(-1).contiguous()             # (D,)
        return mu, V


# ==========================================
# 5) MAIN EVAL (DISCRETE or CONTINUOUS CPD)
# ==========================================
def eval_gmm_prior(
    X_train, cond_train, X_val, cond_val,
    K=8, use_pca_feat=True, pca_dim=64, use_dino=None,
    continuous=False, mdn_hidden=256, mdn_epochs=50, mdn_lr=1e-3, mdn_batch=512, mdn_wd=0.0
):
    """
    X_*: [N, ...] tensors -> flattened internally
    cond_*:
      - if continuous=False: integer labels (LongTensor) for discrete conditions
      - if continuous=True:  float Tensor of shape [N, Cc] (e.g., energies in [0,1])
    """
    # flatten data
    Xtr = X_train.view(X_train.shape[0], -1).to(device).float()
    Xva = X_val.view(X_val.shape[0], -1).to(device).float()


    # ===== CONTINUOUS: MDN conditional GMM =====
    ctr = cond_train.to(device).float().view(cond_train.shape[0], -1)
    cva = cond_val.to(device).float().view(cond_val.shape[0], -1)
    D   = Xtr.shape[1]; Cc = ctr.shape[1]

    #mdn = MDNConditionalGMM(cond_dim=Cc, data_dim=D, K=K, hidden=mdn_hidden).to(device)
    mdn = OriginalMDNConditionalGMM(cond_dim=Cc, data_dim=D, K=K,hidden=mdn_hidden).to(device)
    opt = torch.optim.Adam(mdn.parameters(), lr=mdn_lr, weight_decay=mdn_wd)

    ds = TensorDataset(Xtr, ctr)
    dl = DataLoader(ds, batch_size=mdn_batch, shuffle=True, drop_last=False)
    mdn.train()
    for ep in trange(mdn_epochs):
        losses = []
        for xb, cb in dl:
            loss = mdn.nll(xb, cb)
            #loss = mdn.nll_with_aux(xb, cb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        # (optional) print(f"[MDN] ep {ep} nll={np.mean(losses):.4f}")

    #with torch.no_grad():
    #    x_min = Xtr.min(dim=0).values
    #    x_max = Xtr.max(dim=0).values
    #    mdn.set_data_bounds(x_min, x_max)
    # Held-out NLL (batched over validation set)
    mdn.eval()
    val_ds = TensorDataset(Xva, cva)
    val_dl = DataLoader(val_ds, batch_size=mdn_batch, shuffle=False, drop_last=False)

    with torch.no_grad():
        total_nll = 0.0
        total_n = 0
        for xb, cb in val_dl:
            loss_b = mdn.nll(xb, cb)              # average NLL over batch
            #loss_b = mdn.nll_with_aux(xb, cb)
            bs = xb.size(0)
            total_nll += loss_b.item() * bs       # accumulate weighted by batch size
            total_n += bs
        nll = float(total_nll / max(total_n, 1))

    # Feature space similarity + mean_abs_diff (batched)
    X_fake_chunks = []


    mu, V = pca_fit(Xtr, r=pca_dim)
    mu = mu.to(Xtr.device)
    V = V.to(Xtr.device)
    F_real_chunks = []
    F_fake_chunks = []
    with torch.no_grad():
        for xb, cb in val_dl:
            # real
            F_real_chunks.append(pca_proj(xb, mu, V))

            # fake
            x_fake = mdn.sample(cb, n_per_cond=1)
            F_fake_chunks.append(pca_proj(x_fake, mu, V))
            X_fake_chunks.append(x_fake)

    F_real = torch.cat(F_real_chunks, dim=0)
    F_fake = torch.cat(F_fake_chunks, dim=0)

    # FID / MMD on concatenated features
    fid = float(frechet_distance(F_real, F_fake))
    mmd = float(mmd_rbf(F_real, F_fake))

    # mean_abs_diff in raw space, also batched
    X_fake_all = torch.cat(X_fake_chunks, dim=0)        # same total N as Xva (if no drop_last)
    mean_abs_diff = float((Xva.mean(0) - X_fake_all.mean(0)).abs().mean())

    summary = {
        'heldout_NLL': nll,
        'FID_feat': fid,
        'MMD_feat': mmd,
        'mean_abs_diff_raw': mean_abs_diff,
        'mode': 'continuous'
    }
    return summary, mdn

def invert_to_physical_logit(
    X_std,
    E_GeV,
    mean,
    std,
    #voxel_shape=(28, 40, 40),
    voxel_shape=(47, 40, 40),
    alpha=1e-6,
):
    """
    Invert standardized *logit-space* back to physical voxels.

    Assumes forward transform:
        v_frac        = v_phys / (5 * E)
        v_frac_clamp  = (v_frac * (1 - 2α) + α)
        z             = log( v_frac_clamp / (1 - v_frac_clamp) )
        X_std         = (z - mean) / std

    This does:
        z      = X_std * std + mean
        v_frac = reverse_logit(z, alpha)
        v_phys = v_frac * (5 * E)
    """
    is_torch = torch.is_tensor(X_std)

    if is_torch:
        # --- Torch path ---
        Xs = X_std.to(torch.float32)
        dev = Xs.device

        if Xs.ndim == 2:
            Xs = Xs.view(Xs.shape[0], *voxel_shape)

        if torch.is_tensor(E_GeV):
            E = E_GeV.to(dev, dtype=torch.float32).view(-1)
        else:
            E = torch.as_tensor(E_GeV, dtype=torch.float32, device=dev).view(-1)

        # undo standardisation
        z = Xs * float(std) + float(mean)

        # reverse logit (torch version of your function)
        v_frac = reverse_logit_torch(z, alpha=alpha)
        v_frac = torch.clamp(v_frac, 0.0, 1.0)

        out = v_frac * (5.0 * E.view(-1, 1, 1, 1))
        return out  # torch.Tensor (N,45,50,18)

    else:
        # --- NumPy path ---
        Xs = np.asarray(X_std, dtype=np.float32)
        if Xs.ndim == 2:
            Xs = Xs.reshape(Xs.shape[0], *voxel_shape)

        E = np.asarray(E_GeV, dtype=np.float32).reshape(-1)

        # undo standardisation
        z = Xs * float(std) + float(mean)

        # reverse logit (your exact numpy version)
        v_frac = reverse_logit_np(z, alpha=alpha)
        np.clip(v_frac, 0.0, 1.0, out=v_frac)

        out = v_frac * (5.0 * E.reshape(-1, 1, 1, 1))
        return out.astype(np.float32)

    
    
@torch.no_grad()
def export_physical_real_and_prior_logit(
    X_val_std,              # (N,45,50,18) standardized log-space (same preprocessing as training)
    cond_val,               # (N,1) continuous c OR (N,) labels for discrete
    model_or_cpd,           # MDNConditionalGMM (continuous) OR ConditionalGMM (discrete)
    *,
    mode="continuous",      # "continuous" or "discrete"
    mean=None, std=None,    # dataset stats from preprocessing
    tether_lam=0.3,          # 0 = off; try 0.1–0.3
    tether_sigma=0.001,        # noise std in z-space; try 0.005–0.05
    energies_val_GeV=None,  # (N,) real energies in GeV (required)
    #voxel_shape=(28,40,40),
    voxel_shape=(47,40,40),
    out_prefix="phys_export",
    n_per_cond=1,           # samples per condition
    save = True,
    data_folder="../data"   # folder to save output files
):
    """
    Produces and saves:
      - {out_prefix}_real.h5  with dataset 'showers'  (N,45,50,18) physical
      - {out_prefix}_prior.h5 with dataset 'showers'  (Ns,45,50,18) physical
    Returns (real_phys, prior_phys) as torch tensors.
    """
    assert mean is not None and std is not None, "Provide mean/std used in preprocessing."
    assert energies_val_GeV is not None, "Provide per-event real energies (GeV)."

    # ensure shapes/tensors
    Xv = X_val_std.to(torch.float32)
    if Xv.ndim == 2:
        Xv = Xv.view(Xv.shape[0], *voxel_shape)
    c  = cond_val

    # your existing energy transform (unchanged)
    #energies_val_GeV = 50*(100/50)**energies_val_GeV
    energies_val_GeV = 1*(1000/1)**energies_val_GeV
    E_real = torch.as_tensor(energies_val_GeV, dtype=torch.float32, device=Xv.device).view(-1)

    # --- sample prior in standardized log-space, in batches ---
    samples_list = []
    batch_size = 512
    
    Xv_flat_all = Xv.view(Xv.shape[0], -1)  # z-space anchors

    c = c.to(torch.float32).view(c.shape[0], -1).to(Xv.device)
    for i in range(0, c.shape[0], batch_size):
        c_chunk = c[i:i + batch_size]
        X_anchor = Xv_flat_all[i:i + batch_size]
        s_chunk = model_or_cpd.sample(c_chunk, n_per_cond=n_per_cond,x_anchor=X_anchor,lam=float(tether_lam),sigma=float(tether_sigma))   # (B*n_per_cond, D)
        samples_list.append(s_chunk)
    prior_flat = torch.cat(samples_list, dim=0)   # (N*n_per_cond, D)



    prior_std = prior_flat.view(-1, *voxel_shape).to(Xv.device)

    # reverse normalization to physical (both real & prior)
    real_phys  = invert_to_physical_logit(Xv,        E_real,  mean, std, voxel_shape)   # (N,45,50,18)
    prior_phys = invert_to_physical_logit(prior_std, E_real,  mean, std, voxel_shape)   # (Ns,45,50,18)


    # --- extra stats: mean, std, sum for both standardized & physical ---
    def print_stats(name, arr):
        # assume torch; if it's numpy, wrap with torch.from_numpy
        if not torch.is_tensor(arr):
            arr = torch.from_numpy(arr)
        print(
            f"{name}: mean={arr.mean().item():.6g}, "
            f"std={arr.std().item():.6g}, "
            f"sum={arr.sum().item():.6g}"
        )
    def per_event_stats(name, arr):
        # arr shape: (N,45,50,18)
        x = arr.view(arr.shape[0], -1).float()
        per_ev_std  = x.std(dim=1)
        per_ev_mean = x.mean(dim=1)
        print(f"{name}:")
        print(f"  mean of per-ev std = {per_ev_std.mean().item():.4f}")
        print(f"  median per-ev std  = {per_ev_std.median().item():.4f}")
        print(f"  99% per-ev std     = {torch.quantile(per_ev_std, 0.99).item():.4f}")
        print()

    

    
    per_event_stats("real_phys", real_phys)
    per_event_stats("prior_phys", prior_phys)

    print_stats("Xv (std)",        Xv)
    print_stats("prior_std (std)", prior_std)
    print_stats("real_phys",       real_phys)
    print_stats("prior_phys",      prior_phys)
    
    #prior_phys = prior_phys * real_phys.sum()/prior_phys.sum()
    
    # ===== clamp prior_phys above 1.2 * max(real_phys) =====
    max_real = float(real_phys.max())
    threshold = 1.05 * max_real

    mask_exceed = prior_phys > threshold
    n_exceed = int(mask_exceed.sum())
    
    

    if n_exceed > 0:
        # sanity check – both arrays must line up elementwise
        assert real_phys.shape == prior_phys.shape, \
            f"Shape mismatch: real_phys {real_phys.shape}, prior_phys {prior_phys.shape}"

        # replace those prior voxels with the corresponding real voxels
        prior_phys0 = prior_phys              # avoid modifying any view in-place
        prior_phys0[mask_exceed] = real_phys[mask_exceed]
        prior_phys = prior_phys0


    if save:
        prior_path = os.path.join(data_folder, f"{out_prefix}_prior_hgcal2024.h5")
        os.makedirs(os.path.dirname(prior_path) if os.path.dirname(prior_path) else '.', exist_ok=True)
        with h5.File(prior_path, "w") as f:
            f.create_dataset("showers", data=prior_phys.cpu().numpy(), compression="gzip")

        print(f"[saved]  prior → {prior_path}")
    return real_phys, prior_phys


@torch.no_grad()
def sample_physical_once(
    X_val_std,              # (N,45,50,18) standardized log-space
    cond_val,               # (N,1) or (N,) conditions (energies)
    model_or_cpd,
    *,
    mean, std,
    energies_val_GeV,
    #voxel_shape=(28,40,40),
    voxel_shape=(47,40,40),
    tether_lam=0.0,
    tether_sigma=0.001,
    n_per_cond=1,
):
    """
    Single sampling pass: return (real_phys, prior_phys) as CPU tensors.
    All heavy logit/exp is done on CPU; GPU is used only for MDN sampling.
    """
    assert mean is not None and std is not None
    assert energies_val_GeV is not None

    # ---- X and energies on CPU ----
    Xv = X_val_std.to(torch.float32).cpu()
    if Xv.ndim == 2:
        Xv = Xv.view(Xv.shape[0], *voxel_shape)

    #energies_val_GeV = 50*(100/50)**energies_val_GeV
    energies_val_GeV = 1*(1000/1)**energies_val_GeV
    E_real = torch.as_tensor(energies_val_GeV, dtype=torch.float32, device="cpu").view(-1)

    # ---- conditions on model device (for MDN) ----
    dev = next(model_or_cpd.parameters()).device
    c_all = cond_val.to(torch.float32).view(cond_val.shape[0], -1).to(dev)

    batch_size = 512
    Xv_flat_all = Xv.view(Xv.shape[0], -1)  # CPU
    samples_list = []

    for i in range(0, c_all.shape[0], batch_size):
        c_chunk = c_all[i : i + batch_size]                      # GPU
        X_anchor = Xv_flat_all[i : i + batch_size].to(dev)       # move anchors to GPU

        s_chunk = model_or_cpd.sample_pure(
            c_chunk,
            n_per_cond=n_per_cond,
            x_anchor=X_anchor,
            lam=float(tether_lam),
            sigma=float(tether_sigma),
        )  # (B*n_per_cond, D) on GPU

        samples_list.append(s_chunk.detach().cpu())              # move back to CPU

        # cleanup GPU for this chunk
        del c_chunk, X_anchor, s_chunk
        torch.cuda.empty_cache()

    prior_flat = torch.cat(samples_list, dim=0)       # CPU
    prior_std = prior_flat.view(-1, *voxel_shape)     # CPU

    # invert to physical COMPLETELY on CPU (no CUDA OOM)
    real_phys  = invert_to_physical_logit(Xv,        E_real, mean, std, voxel_shape)
    prior_phys = invert_to_physical_logit(prior_std, E_real, mean, std, voxel_shape)

    
    # ===== clamp prior_phys above 1.2 * max(real_phys) =====
    max_real = float(real_phys.max())
    threshold = 1.05 * max_real

    mask_exceed = prior_phys > threshold
    n_exceed = int(mask_exceed.sum())


    if n_exceed > 0:
        # sanity check – both arrays must line up elementwise
        assert real_phys.shape == prior_phys.shape, \
            f"Shape mismatch: real_phys {real_phys.shape}, prior_phys {prior_phys.shape}"

        # replace those prior voxels with the corresponding real voxels
        prior_phys0 = prior_phys              # avoid modifying any view in-place
        prior_phys0[mask_exceed] = real_phys[mask_exceed]
        prior_phys = prior_phys0

    return real_phys, prior_phys

def sweep_prior_stats(
    model,
    X_all_std,      # (N,45,50,18) or (N,D)
    cond_all,       # (N,1) or (N,)
    E_all_GeV,      # (N,)
    mean, std,
    #voxel_shape=(28,40,40),
    voxel_shape=(47,40,40),
    tether_lam=0.3,          # 0 = off; try 0.1–0.3
    tether_sigma=0.001,        # noise std in z-space; try 0.005–0.05
    out_prefix="test_gmm",
    num_runs=100,
    n_inner=9,      # number of inner radial bins for "center" energy
    data_folder="../data"   # folder to save output files
):
    device = next(model.parameters()).device

    # ensure tensors on device
    X_all_std = X_all_std.to(device)
    cond_all  = cond_all.to(device)
    E_all_GeV = E_all_GeV.to(device)

    # paths
    base_dir = data_folder
    global_stats_path = os.path.join(base_dir, f"{out_prefix}_hgcal2024_global_stats.txt")
    layer_stats_path  = os.path.join(base_dir, f"{out_prefix}_hgcal2024_layer_stats.txt")
    center_stats_path = os.path.join(base_dir, f"{out_prefix}_hgcal2024_center_stats.txt")

    # headers
    with open(global_stats_path, "w") as f:
        f.write("# run  mean_real  std_real  sum_real  mean_prior  std_prior  sum_prior\n")
    with open(layer_stats_path, "w") as f:
        f.write("# run  type  layer0 ... layer44 (mean energy per layer)\n")
    with open(center_stats_path, "w") as f:
        f.write("# run  type  layer0 ... layer44 (mean center energy per layer)\n")

    for run in trange(num_runs):
        print(f"[sweep] run {run+1}/{num_runs}")

        real_phys, prior_phys = sample_physical_once(
            X_val_std=X_all_std,
            cond_val=cond_all,
            model_or_cpd=model,
            mean=mean,
            std=std,
            energies_val_GeV=E_all_GeV,
            voxel_shape=voxel_shape,
            tether_lam=float(tether_lam),
            tether_sigma=float(tether_sigma),
            n_per_cond=1,
        )

        # ---- global stats ----
        mr = real_phys.mean().item()
        sr = real_phys.std().item()
        Sr = real_phys.sum().item()

        mp = prior_phys.mean().item()
        sp = prior_phys.std().item()
        Sp = prior_phys.sum().item()

        with open(global_stats_path, "a") as f:
            f.write(f"{run} {mr:.6e} {sr:.6e} {Sr:.6e} {mp:.6e} {sp:.6e} {Sp:.6e}\n")

        # ---- mean layer energy (sum over last two dims, then mean over events) ----
        # real_phys / prior_phys: (N, 45, 50, 18)
        layer_real  = real_phys.sum(dim=(2, 3)).mean(dim=0)   # (45,)
        layer_prior = prior_phys.sum(dim=(2, 3)).mean(dim=0)  # (45,)

        with open(layer_stats_path, "a") as f:
            vals_r = " ".join(f"{x.item():.6e}" for x in layer_real)
            vals_p = " ".join(f"{x.item():.6e}" for x in layer_prior)
            f.write(f"{run} real  {vals_r}\n")
            f.write(f"{run} prior {vals_p}\n")

        # ---- mean center energy per layer ----
        # assume last dim is radial; take inner n_inner bins
        _, L, H, W = real_phys.shape

        block_h = 10
        block_w = 10
        y0 = (H - block_h) // 2
        y1 = y0 + block_h
        x0 = (W - block_w) // 2
        x1 = x0 + block_w

        # sum over the central 10x10 region, then mean over events -> (L,)
        center_real  = real_phys[:, :, y0:y1, x0:x1].sum(dim=(2, 3)).mean(dim=0)
        center_prior = prior_phys[:, :, y0:y1, x0:x1].sum(dim=(2, 3)).mean(dim=0)

        with open(center_stats_path, "a") as f:
            vals_cr = " ".join(f"{x.item():.6e}" for x in center_real)
            vals_cp = " ".join(f"{x.item():.6e}" for x in center_prior)
            f.write(f"{run} real  {vals_cr}\n")
            f.write(f"{run} prior {vals_cp}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train GMM prior for HGCAL')
    parser.add_argument('--data_folder', default='../data', help='Folder containing data files')
    parser.add_argument('--data_file', default='HGCal_2024_xy_10k.h5', 
                       help='Input HDF5 data file name(s). Can be: (1) single file, (2) comma-separated list (e.g., "file1.h5,file2.h5"), or (3) glob pattern (e.g., "HGCal_*.h5")')
    parser.add_argument('--data_files', nargs='+', default=None, 
                       help='Alternative: specify multiple data files as separate arguments (e.g., --data_files file1.h5 file2.h5). Overrides --data_file if provided.')
    parser.add_argument('--use_glob', action='store_true', default=False,
                       help='Treat --data_file as a glob pattern and expand it (e.g., "HGCal_*.h5" will match all matching files)')
    parser.add_argument('--dataset_key', default='showers', help='HDF5 dataset key for shower data (default: "showers")')
    parser.add_argument('--energy_key', default='incident_energies', help='HDF5 dataset key for energy data (default: "incident_energies"). Use "gen_info" if energy is in gen_info array.')
    parser.add_argument('--gen_info_key', default='gen_info', help='HDF5 dataset key for gen_info array (default: "gen_info")')
    parser.add_argument('--gen_info_energy_col', type=int, default=0, help='Column index in gen_info array for energy (default: 0)')
    parser.add_argument('--target_spatial_shape', nargs=2, type=int, default=[40, 40], help='Target spatial dimensions [height width] (default: 40 40)')
    parser.add_argument('--no_auto_reshape', action='store_true', help='Disable automatic reshaping of spatial dimensions')
    parser.add_argument('--prior_h5', default='', help='Path to existing prior H5 file (optional, for loading)')
    parser.add_argument('--ckpt_path', default='../data/gmm_prior_checkpoint.pt', help='Path to save/load GMM checkpoint')
    parser.add_argument('--out_prefix', default='ckpt_gmm_v2', help='Output prefix for prior H5 files')
    parser.add_argument('--nevts', type=int, default=10000, help='Number of events to load per file (use -1 for all events from all files)')
    parser.add_argument('--nevts_per_file', type=int, default=None, help='Number of events to load per file (overrides --nevts per-file behavior)')
    parser.add_argument('--mdn_epochs', type=int, default=50, help='Number of epochs for MDN training')
    parser.add_argument('--mdn_hidden', type=int, default=148, help='Hidden dimension for MDN')
    parser.add_argument('--K', type=int, default=2, help='Number of GMM components')
    parser.add_argument('--tether_lam', type=float, default=0.4, help='Tether lambda parameter')
    parser.add_argument('--tether_sigma', type=float, default=0.001, help='Tether sigma parameter')
    parser.add_argument('--sweep_runs', type=int, default=10, help='Number of runs for sweep_prior_stats')
    flags = parser.parse_args()

    #dataset_config = {'SHAPE': (-1,28,40,40,1)}
    dataset_config = {'SHAPE': (-1,47,40,40,1)}
    
    # Handle multiple data files with glob pattern support
    if flags.data_files:
        # Use --data_files if provided (space-separated)
        file_list = flags.data_files
        # Expand glob patterns in each file
        expanded_files = []
        for f in file_list:
            if '*' in f or '?' in f or '[' in f:
                # Contains glob pattern
                pattern = os.path.join(flags.data_folder, f) if not os.path.isabs(f) else f
                matches = sorted(glob.glob(pattern))
                if matches:
                    expanded_files.extend(matches)
                else:
                    print(f"[WARNING] No files matched pattern: {pattern}")
            else:
                expanded_files.append(f)
        h5_paths = [os.path.join(flags.data_folder, f) if not os.path.isabs(f) else f for f in expanded_files]
    else:
        # Parse --data_file (can be comma-separated or glob pattern)
        if flags.use_glob or ('*' in flags.data_file or '?' in flags.data_file or '[' in flags.data_file):
            # Treat as glob pattern
            pattern = os.path.join(flags.data_folder, flags.data_file) if not os.path.isabs(flags.data_file) else flags.data_file
            h5_paths = sorted(glob.glob(pattern))
            if not h5_paths:
                raise ValueError(f"No files matched glob pattern: {pattern}")
            print(f"[INFO] Glob pattern '{flags.data_file}' matched {len(h5_paths)} file(s)")
        else:
            # Comma-separated list
            data_files = [f.strip() for f in flags.data_file.split(',')]
            h5_paths = [os.path.join(flags.data_folder, f) if not os.path.isabs(f) else f for f in data_files]
    
    # Validate that all files exist
    missing_files = [p for p in h5_paths if not os.path.exists(p)]
    if missing_files:
        raise FileNotFoundError("The following files do not exist:\n  " + "\n  ".join(missing_files))
    
    # Remove duplicates while preserving order
    seen = set()
    h5_paths = [p for p in h5_paths if not (p in seen or seen.add(p))]
    
    print(f"\n{'='*60}")
    print(f"Loading data from {len(h5_paths)} file(s):")
    for i, p in enumerate(h5_paths, 1):
        file_size = os.path.getsize(p) / (1024**2)  # Size in MB
        print(f"  [{i}] {p} ({file_size:.1f} MB)")
    print(f"{'='*60}\n")
    
    target_spatial = tuple(flags.target_spatial_shape)
    auto_reshape = not flags.no_auto_reshape
    
    # Determine nevts per file if specified
    nevts_per_file = flags.nevts_per_file if flags.nevts_per_file is not None else flags.nevts
    
    # Load data from all files
    print("Loading data from files...")
    showers, energies = load_showers_from_h5(
        h5_paths, 
        nevts=flags.nevts,  # Total events across all files (or -1 for all)
        evt_start=0, 
        dataset_key=flags.dataset_key, 
        energy_key=flags.energy_key,
        gen_info_key=flags.gen_info_key,
        gen_info_energy_col=flags.gen_info_energy_col,
        target_spatial_shape=target_spatial,
        auto_reshape=auto_reshape
    )
    
    print(f"\n{'='*60}")
    print("Data Loading Summary:")
    print(f"  Total files processed: {len(h5_paths)}")
    print(f"  Loaded showers shape: {showers.shape}")
    print(f"  Loaded energies shape: {energies.shape}")
    print(f"  Energy range: {energies.min():.2f} - {energies.max():.2f} GeV")
    print(f"  Total events: {showers.shape[0]:,}")
    print(f"{'='*60}\n")
    X, energies_data, (mu, sd) = preprocess_showers_and_energies_logit(showers, energies)
    
    
    
    print((mu, sd))

    # # Split
    N = X.shape[0]   #; perm = torch.randperm(N)
    ntr = int(0.8*N)
    idx_tr = torch.arange(0, ntr)
    idx_va = torch.arange(ntr, N)
    Xall = torch.from_numpy(X); call = torch.from_numpy(energies_data)
    Xtr = torch.from_numpy(X[idx_tr]); Xva = torch.from_numpy(X[idx_va])
    ctr = torch.from_numpy(energies_data[idx_tr])[:, None]   # continuous 1D energy
    cva = torch.from_numpy(energies_data[idx_va])[:, None]
    
    
    E_val_GeV = cva.view(-1).to(torch.float32)
    E_all_GeV = call.view(-1).to(torch.float32)
    
    print(f'mean energy {E_val_GeV.mean()}')

    prior_h5 = flags.prior_h5 if flags.prior_h5 else ""
    # Construct ckpt_path from data_folder and out_prefix if using default
    # This ensures it respects the --data_folder argument passed from Condor
    if flags.ckpt_path == '../data/gmm_prior_checkpoint.pt' or (not os.path.isabs(flags.ckpt_path) and flags.ckpt_path.startswith('../')):
        # Use data_folder and out_prefix to construct the path
        ckpt_path = os.path.join(flags.data_folder, f"{flags.out_prefix}_gmm_prior_checkpoint.pt")
    else:
        ckpt_path = flags.ckpt_path
    
    # Normalize the path to handle relative paths correctly
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.normpath(ckpt_path)
    
    print(f"[ckpt_path] Using checkpoint path: {ckpt_path} (data_folder: {flags.data_folder})")

    # ---- fast path: if both H5s exist, load and skip recompute ----
    #if os.path.exists(real_h5) and os.path.exists(prior_h5):
    if prior_h5 and os.path.exists(prior_h5):
        print(f"[skip] found {prior_h5}, loading...")
        #with h5.File(real_h5, "r") as f:
        #    real_phys  = f["showers"][:]          # (N,30,30,30), physical
        with h5.File(prior_h5, "r") as f:
            prior_phys = f["showers"][:]          # (M,30,30,30), physical
        # (optional) convert to torch for later ops
        #real_phys_t  = torch.from_numpy(real_phys)
        prior_phys_t = torch.from_numpy(prior_phys)
        #print(f'real_phys_t {real_phys_t.shape}')
        print(f'prior_phys_t {prior_phys_t.shape}')
        
        mdn = OriginalMDNConditionalGMM(cond_dim=ctr.shape[1], data_dim=75200, K=flags.K, hidden=flags.mdn_hidden).to(device)
        
        
        
        if os.path.exists(ckpt_path):
            
            # Load the checkpoint dict
            ckpt = torch.load(ckpt_path, map_location="cuda")

            # Now load its model weights
            mdn.load_state_dict(ckpt["model_state"])
            mdn.eval()
            
            X_3d = Xall.reshape(-1,47,40,40).contiguous()

            real_phys_t, prior_phys_t = export_physical_real_and_prior_logit(
                X_val_std=X_3d,
                cond_val=call,
                model_or_cpd=mdn,
                mode="continuous",
                mean=mu, std=sd,
                tether_lam=flags.tether_lam, tether_sigma=flags.tether_sigma,
                energies_val_GeV=E_all_GeV,
                voxel_shape=(47,40,40),
                out_prefix=flags.out_prefix,
                n_per_cond=1,
                save=False,
                data_folder=flags.data_folder,
            )
            real_phys  = real_phys_t.cpu().numpy()
            prior_phys = prior_phys_t.cpu().numpy()
            
            sweep_prior_stats(
                model=mdn,
                X_all_std=X_3d,
                cond_all=call,          # all energies as conditions
                E_all_GeV=E_all_GeV,    # same energies as used in export
                mean=mu,
                std=sd,
                voxel_shape=(47,40,40),
                tether_lam=flags.tether_lam, tether_sigma=flags.tether_sigma,
                out_prefix=flags.out_prefix,
                num_runs=flags.sweep_runs,
                n_inner=9,              # or whatever inner radius you want
                data_folder=flags.data_folder,
            )
            


    else:
        # ===== no cache: train/eval MDN prior, export physical H5s =====
        summary, model = eval_gmm_prior(
            Xtr, ctr, Xva, cva,
            K=flags.K, continuous=True, mdn_epochs=flags.mdn_epochs, mdn_hidden=flags.mdn_hidden
        )
        print(summary)
        
        # Ensure checkpoint directory exists
        # Get the directory from ckpt_path, or use data_folder as fallback
        ckpt_dir = os.path.dirname(ckpt_path) if os.path.dirname(ckpt_path) else flags.data_folder
        # Normalize the path to handle relative paths correctly
        if not os.path.isabs(ckpt_dir):
            ckpt_dir = os.path.normpath(ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "cond_dim": model.cond_dim,
            "data_dim": model.data_dim,
            "K": model.K,
            "hidden": model.net[0].out_features,   # hidden dims
            "mean": mu,
            "std": sd,
        }, ckpt_path)
        print(f"[saved] GMM checkpoint → {ckpt_path}")

        X_3d = Xall.reshape(-1,47,40,40).contiguous()

        real_phys_t, prior_phys_t = export_physical_real_and_prior_logit(
            X_val_std=X_3d,
            cond_val=call,
            model_or_cpd=model,
            mode="continuous",
            mean=mu, std=sd,
            tether_lam=flags.tether_lam, tether_sigma=flags.tether_sigma,
            energies_val_GeV=E_all_GeV,
            voxel_shape=(47,40,40),
            out_prefix=flags.out_prefix,
            n_per_cond=1,
            data_folder=flags.data_folder,
        )
        real_phys  = real_phys_t.cpu().numpy()
        prior_phys = prior_phys_t.cpu().numpy()

