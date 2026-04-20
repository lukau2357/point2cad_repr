import copy
import torch
import numpy as np
import math
import sys
import tqdm
import time
import trimesh

from scipy.spatial import cKDTree
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader
from .primitive_fitting_utils import triangulate_and_mesh, grid_trimming, alpha_shape_trimming

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
    #
    # Old per-batch CPU-sync version (kept for reference):
    # batch_errors = []
    # with torch.no_grad():
    #     for batch in data_loader:
    #         X = batch[0]
    #         Xhat, _ = model.forward(X)
    #         X_orig = X * cluster_scale + cluster_mean
    #         Xhat_orig = Xhat * cluster_scale + cluster_mean
    #         # L1 (previous): per-point sum of absolute coordinate deltas — inconsistent
    #         # with the other primitives which report Euclidean point-to-surface distance.
    #         # current_error = inr_recon_loss(X_orig, Xhat_orig)
    #         # L2: per-point Euclidean distance — consistent with the other primitive errors.
    #         current_error = torch.linalg.norm(X_orig - Xhat_orig, dim=-1)
    #         batch_errors.append(current_error.cpu().numpy())
    # return np.array(batch_errors).mean()
    #
    # GPU-side accumulation: cluster_mean is a pure translation so it cancels in
    # (X + m) - (Xhat + m); cluster_scale is a positive scalar that factors out of
    # the per-point norm. So we can compute the norm in the normalized space and
    # rescale once at the end, avoiding per-batch device→host syncs.
    # Own the mode switch: BN must use running stats during eval, and training
    # forwards must not be polluted by eval-time batch-stat updates. Restore the
    # caller's prior mode on exit so this is safe to call mid-training.
    was_training = model.training
    model.eval()
    total_error = None
    total_points = 0
    try:
        with torch.no_grad():
            for batch in data_loader:
                X = batch[0]
                Xhat, _ = model.forward(X)
                per_point = torch.linalg.norm(X - Xhat, dim=-1)
                batch_sum = per_point.sum()
                total_error = batch_sum if total_error is None else total_error + batch_sum
                total_points += per_point.shape[0]
    finally:
        if was_training:
            model.train()

    return ((total_error / total_points) * cluster_scale).item()
    
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

        # Should always do clipping regardless of closedness?
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

    def sample_mesh(self, mesh_dim, uv_bb_min, uv_bb_max, cluster, cluster_mean, cluster_scale,
                    uv_margin = 0.1, threshold_multiplier = 3, spacing = None,
                    uv_points = None, alpha = 10.0):
        device = next(self.parameters()).device
        points = self.sample_points(mesh_dim, uv_bb_min, uv_bb_max, cluster_mean, cluster_scale, uv_margin = uv_margin)

        # Alpha shape trimming only for fully open surfaces — closed coordinates
        # have a seam at ±1 that creates artificial gaps in the alpha shape
        # if uv_points is not None and not self.is_u_closed and not self.is_v_closed:
        #     from .primitive_fitting_utils import alpha_shape_trimming
        #     uv_length = uv_bb_max - uv_bb_min
        #     uv_min_ext = uv_bb_min - uv_length * uv_margin
        #     uv_max_ext = uv_bb_max + uv_length * uv_margin
        #     mask = alpha_shape_trimming(uv_points, mesh_dim, mesh_dim, uv_min_ext, uv_max_ext, alpha=alpha)
        # else:
        #     mask = None

        mask = None
        # mask = grid_trimming(cluster, points.cpu().numpy(), mesh_dim, mesh_dim, device, threshold_multiplier = threshold_multiplier, spacing = spacing)
        meshes = triangulate_and_mesh(points.cpu().numpy(), mesh_dim, mesh_dim, "inr", mask = mask)
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
            steps_per_epoch,
            is_u_closed = False,
            is_v_closed = False,
            max_steps = 1000,
            warmup_steps_ratio = 0.05,
            initial_lr = 1e-2,
            noise_magnitude_3d = 0.005,
            noise_magnitude_uv = 0.005,
            eval_every = 5):
    start = time.time()

    model = INRNetwork(**network_parameters, is_u_closed = is_u_closed, is_v_closed = is_v_closed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr  = initial_lr)
    warmup_steps = int(max_steps * warmup_steps_ratio)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_steps, max_steps)

    # Dedicated eval loader. shuffle=False + no generator so mid-training eval
    # passes do not perturb `dl`'s shared torch.Generator (which drives the
    # infinite training iterator via dl_generator).
    eval_dl = DataLoader(dl.dataset, batch_size=dl.batch_size,
                         shuffle=False, drop_last=False)

    loop = tqdm.tqdm(range(max_steps), desc="Training the INR network",
                     disable=not sys.stderr.isatty())

    # Old epoch-mean L1-on-noised-inputs proxy — biased by the noise schedule
    # (anneals to 0 at the last step, which almost always wins). Replaced with
    # periodic L2-on-clean-inputs eval over the full cluster.
    # best_proxy = float("inf")
    # epoch_loss_sum = torch.zeros((), device=device)
    # epoch_step_count = 0

    best_error = float("inf")
    best_state_dict = None
    best_step = -1

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

        # Old proxy update (epoch-mean L1 on noised inputs):
        # epoch_loss_sum = epoch_loss_sum + recon_loss.detach()
        # epoch_step_count += 1
        # is_epoch_end = (epoch_step_count == steps_per_epoch)
        # is_last_step = (i == max_steps - 1)
        # if is_epoch_end or is_last_step:
        #     epoch_mean_loss = (epoch_loss_sum / epoch_step_count).item()
        #     if epoch_mean_loss < best_proxy:
        #         best_proxy = epoch_mean_loss
        #         best_step = i
        #         best_state_dict = copy.deepcopy(model.state_dict())
        #     epoch_loss_sum = torch.zeros((), device=device)
        #     epoch_step_count = 0

        if i % eval_every == 0 or i == max_steps - 1:
            step_error = inr_error(eval_dl, model, cluster_mean, cluster_scale)
            if step_error < best_error:
                best_error = step_error
                best_step = i
                best_state_dict = copy.deepcopy(model.state_dict())

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    torch.cuda.synchronize()
    end = time.time()

    # Sanity check: should match best_error exactly (same state, same eval path).
    error = inr_error(eval_dl, model, cluster_mean, cluster_scale)
    if best_state_dict is not None:
        tqdm.tqdm.write(f"  [inr] best_error={best_error:.6f}  final_error={error:.6f}  at step {best_step}/{max_steps - 1}")

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

def polish_inr(model, X, uv_points, uv_bb_min, uv_bb_max,
               cluster_scale, device,
               is_u_closed, is_v_closed,
               polish_steps,
               polish_lr,
               reg_peak=0.5,
               uv_margin=0.1,
               alpha=10.0,
               N_ext=256,
               reg_grid=101,
               seed=None):
    polish_t0 = time.time()
    np_rng = np.random.default_rng(seed)

    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.eval()
    model.decoder.eval()
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=polish_lr)

    uv_length = uv_bb_max - uv_bb_min
    outer_bb_min = (uv_bb_min - uv_margin * uv_length).astype(np.float64)
    outer_bb_max = (uv_bb_max + uv_margin * uv_length).astype(np.float64)
    if is_u_closed:
        outer_bb_min[0] = max(outer_bb_min[0], -1.0)
        outer_bb_max[0] = min(outer_bb_max[0], 1.0)
    if is_v_closed:
        outer_bb_min[1] = max(outer_bb_min[1], -1.0)
        outer_bb_max[1] = min(outer_bb_max[1], 1.0)

    u_edges = np.linspace(outer_bb_min[0], outer_bb_max[0], reg_grid)
    v_edges = np.linspace(outer_bb_min[1], outer_bb_max[1], reg_grid)
    u_width = float(u_edges[1] - u_edges[0])
    v_width = float(v_edges[1] - v_edges[0])

    # Reg-region cells: outside the alpha shape of encoded UVs (covers both margin
    # shell and interior holes). For closed axes, the alpha shape is unreliable
    # near the seam — fall back to "outside inner bbox only" on those.
    use_alpha = (not is_u_closed) and (not is_v_closed)
    if use_alpha:
        alpha_mask = alpha_shape_trimming(
            uv_points, reg_grid, reg_grid, outer_bb_min, outer_bb_max, alpha=alpha
        )
        reject_cells = np.argwhere((~alpha_mask).numpy())
    else:
        u_centers = (u_edges[:-1] + u_edges[1:]) / 2
        v_centers = (v_edges[:-1] + v_edges[1:]) / 2
        uu, vv = np.meshgrid(u_centers, v_centers, indexing="ij")
        inside_inner = (
            (uu >= uv_bb_min[0]) & (uu <= uv_bb_max[0]) &
            (vv >= uv_bb_min[1]) & (vv <= uv_bb_max[1])
        )
        reject_cells = np.argwhere(~inside_inner)

    K = len(reject_cells)
    tqdm.tqdm.write(f"  [polish] reg_cells={K}/{(reg_grid-1)**2}  use_alpha={use_alpha}")

    X_np = X.detach().cpu().numpy()
    kdtree = cKDTree(X_np)
    uv_points_t = torch.tensor(uv_points, dtype=torch.float32, device=device)

    with torch.no_grad():
        Xhat_pre, _ = model.forward(X)
        err_before = (torch.linalg.norm(X - Xhat_pre, dim=-1).mean() * cluster_scale).item()
    tqdm.tqdm.write(f"  [polish] pre: error={err_before:.6f}")

    loop = tqdm.tqdm(range(polish_steps), desc="Polishing INR",
                     disable=not sys.stderr.isatty())
    for i in loop:
        optimizer.zero_grad()

        uv = model.forward_encoder(X)
        Xhat = model.forward_decoder(uv)
        recon = inr_recon_loss(X, Xhat).mean()

        if K == 0:
            reg = torch.zeros((), device=device)
        else:
            sel = np_rng.choice(K, size=N_ext, replace=(K < N_ext))
            cells = reject_cells[sel]
            u_ext_np = np.stack([
                u_edges[cells[:, 0]] + np_rng.uniform(0.0, u_width, N_ext),
                v_edges[cells[:, 1]] + np_rng.uniform(0.0, v_width, N_ext),
            ], axis=1)
            u_ext = torch.tensor(u_ext_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                Xhat_ext_det = model.forward_decoder(u_ext)
            _, k_idx = kdtree.query(Xhat_ext_det.detach().cpu().numpy(), k=1)
            u_k = uv_points_t[k_idx]

            u_k_grad = u_k.detach().clone().requires_grad_(True)
            Xhat_k = model.forward_decoder(u_k_grad)
            delta = (u_ext - u_k).detach()

            J_delta = torch.zeros_like(Xhat_k)
            for j in range(3):
                grad_out = torch.zeros_like(Xhat_k)
                grad_out[:, j] = 1.0
                grad_u = torch.autograd.grad(
                    Xhat_k, u_k_grad,
                    grad_outputs=grad_out,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                J_delta[:, j] = (grad_u * delta).sum(dim=-1)

            Xhat_ext = model.forward_decoder(u_ext)
            residual = Xhat_ext - Xhat_k - J_delta
            reg = residual.abs().sum(dim=-1).mean()

        reg_weight = reg_peak * (i / max(polish_steps - 1, 1))
        loss = recon + reg_weight * reg

        recon_item = recon.item()
        reg_item = float(reg.item()) if isinstance(reg, torch.Tensor) else float(reg)
        if i % 50 == 0 or i == polish_steps - 1:
            tqdm.tqdm.write(
                f"  [polish] step {i:4d}: recon={recon_item:.6f}  "
                f"reg={reg_item:.6f}  reg_weight={reg_weight:.5f}"
            )

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        Xhat_post, _ = model.forward(X)
        err_after = (torch.linalg.norm(X - Xhat_post, dim=-1).mean() * cluster_scale).item()
    polish_time = time.time() - polish_t0
    tqdm.tqdm.write(
        f"  [polish] post: error={err_after:.6f} "
        f"(delta {err_after - err_before:+.6f})  time={polish_time:.2f}s"
    )

    return err_after

def fit_inr(cluster, network_parameters, device = "cuda:0",
            max_steps = 1000,
            warmup_steps_ratio = 0.05,
            initial_lr = 1e-2,
            max_memory_mb = 10,
            noise_magnitude_3d = 0.005,
            noise_magnitude_uv = 0.005,
            polish = False,
            polish_reg_peak = 0.01,
            seed = 42):
    
    # TODO: Correct??
    if not torch.cuda.is_available():
        device = "cpu"

    N = cluster.shape[0]
    batch_size = automatic_batch_size(N, max_memory_mb = max_memory_mb)
    steps_per_epoch = math.ceil(N / batch_size)
    tqdm.tqdm.write(f"  [inr] batch_size={batch_size}  steps_per_epoch={steps_per_epoch}")

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
    best_model = None

    torch.backends.cudnn.deterministic = True

    inr_t0 = time.time()
    for u in [True, False]:
        for v in [True, False]:
            # Global seed — same init, noise, and batch order for all combos;
            # only the (u, v) closedness configuration differs.
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed_all(seed)

            dl_gen = torch.Generator()
            dl_gen.manual_seed(seed)
            dl = DataLoader(dataset, batch_size = batch_size, drop_last = False, shuffle = True, generator = dl_gen)

            def get_next_item():
                while True:
                    for X in dl:
                        yield X

            dl_generator = get_next_item()
            current_model = fit_inr_single(network_parameters, device, dl, dl_generator, cluster_mean_torch, cluster_scale_torch,
                                           steps_per_epoch,
                                           is_u_closed = u,
                                           is_v_closed = v,
                                           max_steps = max_steps,
                                           warmup_steps_ratio = warmup_steps_ratio,
                                           initial_lr = initial_lr,
                                           noise_magnitude_3d = noise_magnitude_3d,
                                           noise_magnitude_uv = noise_magnitude_uv)
            fit_time = current_model["metadata"]["fitting_time_seconds"]
            tqdm.tqdm.write(f"  [inr] u_closed={u}  v_closed={v}  best_error={current_model['error']:.6f}  time={fit_time:.2f}s")
            if best_model is None or best_model["error"] > current_model["error"]:
                best_model = current_model  
    # best_model = fit_inr_single(network_parameters, device, dl, dl_generator, cluster_mean_torch, cluster_scale_torch,
    #                             is_u_closed = False, 
    #                             is_v_closed = False,
    #                             max_steps = max_steps,
    #                             warmup_steps_ratio = warmup_steps_ratio,
    #                             initial_lr = initial_lr,
    #                             noise_magnitude_3d = noise_magnitude_3d,
    #                             noise_magnitude_uv = noise_magnitude_uv)
    inr_total = time.time() - inr_t0
    tqdm.tqdm.write(f"  [inr] best: error={best_model['error']:.6f}  total_time={inr_total:.2f}s")
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
    best_model["params"]["uv_points"] = uvs

    if polish:
        polish_steps = max_steps // 2
        polish_lr = initial_lr * 0.1
        err_after = polish_inr(
            model=model,
            X=cluster,
            uv_points=uvs,
            uv_bb_min=uv_bb_min,
            uv_bb_max=uv_bb_max,
            cluster_scale=cluster_scale_torch,
            device=device,
            is_u_closed=model.is_u_closed,
            is_v_closed=model.is_v_closed,
            polish_steps=polish_steps,
            polish_lr=polish_lr,
            reg_peak=polish_reg_peak,
            seed=seed,
        )
        best_model["error"] = err_after

    return best_model