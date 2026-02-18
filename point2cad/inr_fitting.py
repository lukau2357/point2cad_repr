import torch
import numpy as np
import math
import tqdm
import time
import trimesh

from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader
from .primitive_fitting_utils import triangulate_and_mesh, grid_trimming

def encoder_to_uv(output, is_closed):
    # [B, 2] => [B, 1], depending on open/closed parameter configuration
    res =  torch.atan2(output[:, 0], output[:, 1]) / np.pi if is_closed else torch.nn.functional.tanh(output[:, 0])
    return res.unsqueeze(-1)

def uv_to_decoder(output, is_closed):
    # [B, 1] => [B, 2]
    if is_closed:
        output *= np.pi
        latent_1 = output.cos()
        latent_2 = output.sin()
        res = torch.cat([latent_1, latent_2], dim = -1)
    
    else:
        # Replication or zero padding. Implementation uses replication in function definition
        # but zero padding during training!!!
        # Function definition: https://github.com/prs-eth/point2cad/blob/81e15bfa952aee62cf06cdf4b0897c552fe4fb3a/point2cad/fitting_one_surface.py#L758
        # Training loop chunk: https://github.com/prs-eth/point2cad/blob/81e15bfa952aee62cf06cdf4b0897c552fe4fb3a/point2cad/fitting_one_surface.py#L592
        res = torch.cat([output, torch.zeros_like(output)], dim = -1)
    
    return res

def inr_recon_loss(X, Xhat):
    # Does not do a reduce operation, should be done manually if needed!
    return torch.abs(X - Xhat).sum(dim = -1)
    # return ((X - Xhat) ** 2).sum(dim = -1)

def inr_error(data_loader, model, cluster_mean, cluster_scale):
    # The loss that they are using is L1, but to compute the fitness error for INR they actually use L2
    # Explicit training loss definition: https://github.com/prs-eth/point2cad/blob/81e15bfa952aee62cf06cdf4b0897c552fe4fb3a/point2cad/fitting_one_surface.py#L319
    # Error inference: https://github.com/prs-eth/point2cad/blob/81e15bfa952aee62cf06cdf4b0897c552fe4fb3a/point2cad/fitting_one_surface.py#L835
    # We will try to use L1 for now in both places...
    batch_errors = []
    with torch.no_grad():
        for batch in data_loader:
            X = batch[0]
            Xhat, _ = model.forward(X)
            # Measure INR error on original data, consistent with error measuring for cannonical algorithms
            current_error = inr_recon_loss(X * cluster_scale + cluster_mean, Xhat * cluster_scale + cluster_mean)
            batch_errors.append(current_error.cpu().numpy())
    
    return np.array(batch_errors).mean()
    
'''
3N = (free * (1 - memory_margin))
N = floor((free * (1 - memory_margin)) / 3)
'''

def automatic_batch_size(N, max_memory_mb = 10):
    max_allowed = math.ceil((max_memory_mb * (2 ** 20)) / 3)
    return min(N, max_allowed)

class CustomLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        bound_weight = kwargs.pop("bound_weight", None)
        super().__init__(*args, **kwargs)

        with torch.no_grad():
            if bound_weight is not None:
                self.weight.uniform_(-bound_weight, bound_weight)

class SiLUBlock(torch.nn.Module):
    def __init__(self, dim_in, dim_out, use_shortcut = True):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.weight = torch.nn.Parameter(torch.ones((1,)))
        self.linear = torch.nn.Linear(dim_in, dim_out)
        self.norm = torch.nn.BatchNorm1d(dim_out)

        if use_shortcut:
            self.residual_map = torch.nn.Identity() if dim_in == dim_out else torch.nn.Linear(dim_in, dim_out)

    def forward(self, X):
        shortcut = X
        X = self.linear(X)
        X = self.norm(X)
        X = torch.nn.functional.silu(X)

        if self.use_shortcut:
            X = (self.weight * X + self.residual_map(shortcut)) / 2 ** 0.5
        
        return X

class SIRENBlock(torch.nn.Module):
    def __init__(self, dim_in, dim_out, angular_freq = 30):
        super().__init__()
        bound_weight = 1 / dim_in
        # bound_bias = 0.1 * bound_weight # https://github.com/prs-eth/point2cad/blob/main/point2cad/layers.py#L85

        # Important: Turn off bias for this linear map!
        # The original implementation of Point2CAD computes sin(w(Ax + b)), original SIREN does (sin(wAx + b))!
        # We manually implement the bias for the SIREN block.
        # https://github.com/prs-eth/point2cad/blob/main/point2cad/layers.py#L95
        self.linear = CustomLinear(dim_in, dim_out, bound_weight = bound_weight, bias = False)
        self.bias = torch.nn.Parameter(torch.zeros((dim_out,)))
        self.angular_freq = angular_freq

    def forward(self, X):
        X = self.linear(X)
        X = torch.sin(self.angular_freq * X + self.bias)
        return X

class INREncoder(torch.nn.Module):
    def __init__(self, hidden_dim, fraction_siren, is_u_closed, is_v_closed, use_shortcut = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fraction_siren = fraction_siren
        self.is_u_closed = is_u_closed
        self.is_v_closed = is_v_closed

        self.siren_dim = int(hidden_dim * fraction_siren)
        self.silu_dim = hidden_dim - self.siren_dim
        
        self.siren_layer = SIRENBlock(3, self.siren_dim)
        self.silu_layer = SiLUBlock(3, self.silu_dim, use_shortcut = use_shortcut)
        self.last_linear = torch.nn.Linear(hidden_dim, 4) # Maps to [u1, u2, v1, v2]

    def forward(self, X):
        # [B, 3] => [B, 2]
        X_siren = self.siren_layer(X) # [B, siren_dim]
        X_silu = self.silu_layer(X) # [B, silu_dim] siren_dim + silu_dim = hidden_dim

        X = torch.concat([X_siren, X_silu], dim = -1)
        X = self.last_linear(X)

        # Encoder to UV rules, not mentioned in the paper: 
        # https://github.com/prs-eth/point2cad/blob/81e15bfa952aee62cf06cdf4b0897c552fe4fb3a/point2cad/fitting_one_surface.py#L758
        X_u = X[:, :2]
        X_v = X[:, 2:]

        U = encoder_to_uv(X_u, self.is_u_closed) # [B, 1]
        V = encoder_to_uv(X_v, self.is_v_closed) # [B, 1]

        return torch.cat([U, V], dim = -1)

class INRDecoder(torch.nn.Module):
    def __init__(self, hidden_dim, fraction_siren, is_u_closed, is_v_closed, use_shortcut = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fraction_siren = fraction_siren
        self.is_u_closed = is_u_closed
        self.is_v_closed = is_v_closed

        self.siren_dim = int(hidden_dim * fraction_siren)
        self.silu_dim = hidden_dim - self.siren_dim

        self.siren_layer = SIRENBlock(4, self.siren_dim)
        self.silu_layer = SiLUBlock(4, self.silu_dim, use_shortcut = use_shortcut)
        self.last_linear = torch.nn.Linear(hidden_dim, 3)

    def forward(self, X):
        # [B, 2] => [B, 3]
        U_lifted = uv_to_decoder(X[:, [0]], self.is_u_closed) # [B, 2]
        V_lifted = uv_to_decoder(X[:, [1]], self.is_v_closed) # [B, 2]
        UV = torch.cat([U_lifted, V_lifted], dim = -1) # [B, 4]

        X_siren = self.siren_layer(UV) # [B, siren_dim]
        X_silu = self.silu_layer(UV) # [B, silu_dim], siren_dim + silu_dim = hidden_dim

        X = torch.cat([X_siren, X_silu], dim = -1) # [B, hidden_dim]
        return self.last_linear(X)

class INRNetwork(torch.nn.Module):
    def __init__(self, hidden_dim, fraction_siren, is_u_closed, is_v_closed, use_shortcut = False):
        super().__init__()
        self.is_u_closed = is_u_closed
        self.is_v_closed = is_v_closed
        self.encoder = INREncoder(hidden_dim, fraction_siren, is_u_closed, is_v_closed, use_shortcut = use_shortcut)
        self.decoder = INRDecoder(hidden_dim, fraction_siren, is_u_closed, is_v_closed, use_shortcut = use_shortcut)
    
    def forward(self, X, cluster_mean = None, cluster_scale = None):
        if cluster_mean is not None and cluster_scale is not None:
            cluster_mean = torch.tensor(cluster_mean, device = X.device)
            cluster_scale = torch.tensor(cluster_scale, device = X.device)
            X = (X - cluster_mean) / cluster_scale

        uv = self.encoder(X)
        Xhat = self.decoder(uv)

        if cluster_mean is not None and cluster_scale is not None:
            Xhat = Xhat * cluster_scale + cluster_mean

        return Xhat, uv
    
    def forward_encoder(self, X, cluster_mean = None, cluster_scale = None):        
        if cluster_mean is not None and cluster_scale is not None:
            cluster_mean = torch.tensor(cluster_mean, device = X.device)
            cluster_scale = torch.tensor(cluster_scale, device = X.device)
            X = (X - cluster_mean) / cluster_scale

        return self.encoder(X)

    def forward_decoder(self, uv):
        Xhat = self.decoder(uv)        
        return Xhat
    
    def sample_points(self, mesh_dim, uv_bb_min, uv_bb_max, cluster_mean, cluster_scale, uv_margin = 0):
        # Resulting samples will be of shape [mesh_dim^2, 3]
        uv_length = uv_bb_max - uv_bb_min
        uv_bb_min_extended = uv_bb_min - uv_length * uv_margin
        uv_bb_max_extended = uv_bb_max + uv_length * uv_margin

        # Should always do clipping regardless of closeness?
        if self.is_u_closed:
            uv_bb_min_extended[0] = max(uv_bb_min_extended[0], -1)
            uv_bb_max_extended[0] = min(uv_bb_max_extended[0], 1)

        if self.is_v_closed:
            uv_bb_min_extended[1] = max(uv_bb_min_extended[1], -1)
            uv_bb_max_extended[1] = min(uv_bb_max_extended[1], 1)

        device = next(self.parameters()).device

        u, v = torch.meshgrid(
            torch.linspace(uv_bb_min_extended[0], uv_bb_max_extended[0], mesh_dim, device = device),
            torch.linspace(uv_bb_min_extended[1], uv_bb_max_extended[1], mesh_dim, device = device),
            indexing = "ij"
        )

        # Cartesian product of two linspaces, where the first coordinate moves faster.
        uv = torch.stack((u, v), dim = 2).reshape(-1, 2)
        with torch.no_grad():
            X = self.forward_decoder(uv)
            cluster_mean = torch.tensor(cluster_mean, device = device)
            cluster_scale = torch.tensor(cluster_scale, device = device)
            X = X * cluster_scale + cluster_mean
        
        return X

    def sample_mesh(self, mesh_dim, uv_bb_min, uv_bb_max, cluster, cluster_mean, cluster_scale, uv_margin = 0.1, threshold_multiplier = 3):
        device = next(self.parameters()).device
        points = self.sample_points(mesh_dim, uv_bb_min, uv_bb_max, cluster_mean, cluster_scale, uv_margin = uv_margin)
        mask = grid_trimming(cluster, points.cpu().numpy(), mesh_dim, mesh_dim, device, threshold_multiplier = threshold_multiplier)
        mask = None
        meshes = triangulate_and_mesh(points.cpu().numpy(), mesh_dim, mesh_dim, "inr", mask = mask)
        # Contains Open3D mesh and Trimesh mesh for INR in particular
        return meshes
    
class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, eta_min = 0.0, last_epoch = -1):
        assert max_steps > warmup_steps, "max_steps must be greater than warmup_steps"

        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eta_min = eta_min # minimum achievable learning rate, will practically always be 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Warmup phase
        if self.last_epoch < self.warmup_steps:
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # Cosine annealing phase
        progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        cos_term = 0.5 * (1 + math.cos(math.pi * progress))

        return [
            self.eta_min + (base_lr - self.eta_min) * cos_term
            for base_lr in self.base_lrs
        ]

def fit_inr_single(network_parameters, device, dl, dl_generator, cluster_mean, cluster_scale,
            is_u_closed = False,
            is_v_closed = False,
            max_steps = 1000, 
            warmup_steps_ratio = 0.05, 
            initial_lr = 1e-2, 
            noise_magnitude_3d = 0.005,
            noise_magnitude_uv = 0.005):
    start = time.time()

    model = INRNetwork(**network_parameters, is_u_closed = is_u_closed, is_v_closed = is_v_closed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr  = initial_lr)
    warmup_steps = int(max_steps * warmup_steps_ratio)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_steps, max_steps)

    loop = tqdm.tqdm(range(max_steps), desc = "Training the INR network")

    for i, _ in enumerate(loop):
        noise_schedule = (max_steps - 1 - i) / (max_steps - 1)

        optimizer.zero_grad()

        X = next(dl_generator)
        X_original = X[0]
        if noise_magnitude_3d == 0:
            X_noised = X_original
        
        else:
            noise_x = torch.randn(size = X_original.shape, device = device)
            X_noised = X_original + noise_magnitude_3d * noise_schedule * noise_x

        uv = model.forward_encoder(X_noised)

        if noise_magnitude_uv != 0:
            noise_uv = torch.randn(size = uv.shape, device = device)
            uv = uv + noise_magnitude_uv * noise_schedule * noise_uv
        
        Xhat = model.forward_decoder(uv)
        recon_loss = inr_recon_loss(X_original, Xhat).mean()
        recon_loss.backward()
        optimizer.step()
        scheduler.step()
        # loop.set_postfix(recon_loss = f"{recon_loss:.4f}")
    
    torch.cuda.synchronize()
    end = time.time()

    error = inr_error(dl, model, cluster_mean, cluster_scale)

    result = {
        "surface_type": "inr",
        "error": error,
        "params": {
            "network_parameters": network_parameters,
            "model": model,
        },
        "metadata": {
            "fitting_time_seconds": end - start
        }
    }

    return result

def fit_inr(cluster, network_parameters, device = "cuda:0",
            max_steps = 1000, 
            warmup_steps_ratio = 0.05, 
            initial_lr = 1e-2, 
            max_memory_mb = 10,
            noise_magnitude_3d = 0.005,
            noise_magnitude_uv = 0.005):
    
    # TODO: Correct??
    if not torch.cuda.is_available():
        device = "cpu"

    N = cluster.shape[0]
    batch_size = automatic_batch_size(N, max_memory_mb = max_memory_mb)
    print(f"Using batch size of: {batch_size}")

    cluster_mean = cluster.mean(axis = 0)
    cluster_std = cluster.std(axis = 0)
    # Divide by maximum STD. This preserves the norm of vectors, up to maximum STD used for scaling, and therefore preserves aspect
    # ratios between the coordinates. Z-score standardization would not preserve aspect ratios.
    cluster_scale = cluster_std.max()
    # When passing points through the INR during inference/sampling, do not forget to account for 1e-6 factor for numerical
    # stability!
    cluster = (cluster - cluster_mean) / (cluster_scale + 1e-6)
    cluster_mean_torch = torch.tensor(cluster_mean, dtype = torch.float32).to(device)
    cluster_scale_torch = torch.tensor(cluster_scale, dtype = torch.float32).to(device)
    cluster = torch.tensor(cluster, device = device)
    dataset = TensorDataset(cluster)
    # TODO: Implement reproducibility for the DataLoader!
    dl = DataLoader(dataset, batch_size = batch_size, drop_last = False, shuffle = True)

    def get_next_item():
        while True:
            for X in dl:
                yield X

    best_model = None

    dl_generator = get_next_item()
    for u in [True, False]:
        for v in [True, False]:
            current_model = fit_inr_single(network_parameters, device, dl, dl_generator, cluster_mean_torch, cluster_scale_torch,
                                           is_u_closed = u, 
                                           is_v_closed = v,
                                           max_steps = max_steps,
                                           warmup_steps_ratio = warmup_steps_ratio,
                                           initial_lr = initial_lr,
                                           noise_magnitude_3d = noise_magnitude_3d,
                                           noise_magnitude_uv = noise_magnitude_uv)
            if best_model is None or best_model["error"] > current_model["error"]:
                best_model = current_model
    
    best_model["params"]["cluster_mean"] = cluster_mean
    best_model["params"]["cluster_scale"] = cluster_scale
    model = best_model["params"]["model"]

    # One more forward pass through the best model to obtain UV bounding boxes.
    uvs = []
    with torch.no_grad():
        for X in dl:
            X = X[0]
            uv = model.forward_encoder(X).cpu().numpy()
            uvs.append(uv)

    uvs = np.concatenate(uvs, axis = 0)
    uv_bb_min = uvs.min(axis = 0)
    uv_bb_max = uvs.max(axis = 0)

    best_model["params"]["uv_bb_min"] = uv_bb_min
    best_model["params"]["uv_bb_max"] = uv_bb_max

    return best_model