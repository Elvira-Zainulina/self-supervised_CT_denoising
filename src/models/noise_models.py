import torch
import torch.nn as nn
from torch.nn import functional as F


# simple noise model with constant parameters
class NoiseSimple(nn.Module):
    def __init__(self, shape=(1, 1), mode='poisson-gaussian'):
        super().__init__()

        assert mode in ['gaussian', 'poisson', 'poisson-gaussian']
        self.mode = mode
        self.a = nn.Parameter(torch.zeros(shape, dtype=torch.float32), requires_grad=True)

        if mode == 'poisson-gaussian':
            self.b = nn.Parameter(torch.zeros(shape, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        noise_est = F.softplus(self.a - 4.0) + 1e-3
        if self.mode == 'gaussian':
            return noise_est
        poisson_noise_var = x.clamp(1e-3, 1e9) * noise_est
        if self.mode == 'poisson':
            return poisson_noise_var ** 0.5
        else:
            noise_var = (poisson_noise_var + self.b).clamp(1e-3, 1e9)
            return noise_var ** 0.5


# parameters of the Poisson distribution depend on the detector column and the used tube current
class Noise(nn.Module):
    def __init__(self, num_of_cols=736):
        super().__init__()

        self.emb = nn.Embedding(num_of_cols, 1)
        self.map_acq_params = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(),
            nn.Linear(2, 2)
        )
        self.std = nn.Parameter(torch.empty((1, 1), dtype=torch.float32), requires_grad=True)
        nn.init.uniform_(self.std, a=1.0, b=10.0)

    def forward(self, x, iy, acq_params):
        iys = torch.arange(0, x.shape[-1], dtype=torch.long, device=x.device)
        iys = iy.unsqueeze(-1) + iys.unsqueeze(0)  # B x W
        iys = self.emb(iys)  # B x W x 1
        acq_params = self.map_acq_params(acq_params).unsqueeze(-1).unsqueeze(-1)  # B x 2 x 1 x 1
        iys = torch.relu(iys * acq_params[:, 0] + acq_params[:, 1]).squeeze(-1) + 1e-6  # B x W
        iys = iys.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x W
        #  x of shape B x C x H x W
        noise_var = x.clamp(1e-6, 1e9) / iys + (self.std / iys) ** 2
        return noise_var ** 0.5  # B x C x H x W
