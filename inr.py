import torch
import numpy as np
import math
import tqdm

from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader

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
        self.encoder = INREncoder(hidden_dim, fraction_siren, is_u_closed, is_v_closed, use_shortcut = use_shortcut)
        self.decoder = INRDecoder(hidden_dim, fraction_siren, is_u_closed, is_v_closed, use_shortcut = use_shortcut)
    
    def forward(self, X):
        uv = self.encoder(X)
        X_hat = self.decoder(uv)
        return X_hat, uv

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

def train_inr(cluster, network_parameters, device, max_steps = 1000, warmup_steps_ratio = 0.05, initial_lr = 3e-4, batch_size = 32):
    cluster = torch.tensor(cluster, device = device)
    dataset = TensorDataset(cluster)
    # TODO: Implement reproducibility for the DataLoader!
    dl = DataLoader(dataset, batch_size = batch_size, drop_last = False, shuffle = True)

    def get_next_item():
        while True:
            for X in dl:
                yield X

    dl_generator = get_next_item()

    model = INRNetwork(**network_parameters)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr  = initial_lr)
    warmup_steps = int(max_steps * warmup_steps_ratio)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_steps, max_steps)

    N = cluster.shape[0]
    loop = tqdm.tqdm(range(max_steps), desc = "Training the INR network")

    for _ in loop:
        optimizer.zero_grad()

        X = next(dl_generator)
        X = X[0]

        X_hat, uv = model(X)
        recon_loss = ((X - X_hat) ** 2).sum(dim = -1).mean()
        recon_loss.backward()
        optimizer.step()
        scheduler.step()

        loop.set_postfix(recon_loss = f"{recon_loss:.4f}")
if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    network_parameters = {
        "hidden_dim": 64,
        "fraction_siren": 0.5,
        "is_u_closed": True,
        "is_v_closed": True,
        "use_shortcut": False
    }

    cluster = np.random.randn(320, 3).astype(np.float32)
    train_inr(cluster, network_parameters, device)