import torch
import torch.nn as nn

class Diffusion(nn.Module):
    def __init__(self, time_steps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        super().__init__()
        self.betas = torch.linspace(beta_start, beta_end, time_steps).float().to(device)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(device)
        self.one_minus_sqrt_alphas_cumprod = 1. - torch.sqrt(self.alphas_cumprod).to(device)
        self.device = device

    @staticmethod
    def _extract(data, batch_t, shape):
        # data: [diffusion_steps], batch_t: [B], shape: [B, L, D]
        batch_size = batch_t.shape[0]
        out = torch.gather(data, -1, batch_t).to(batch_t.device) # FIXME
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))
    
    def q_sample(self, x, time, noise):
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, time, x.shape) # [B, 1, 1]
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, time, x.shape)
        x_noise = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
        return x_noise

if __name__ == "__main__":
    beta_start=0.0001
    beta_end=0.02
    time_steps = 100
    betas = torch.linspace(beta_start, beta_end, time_steps).float()
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0) # [time_steps]
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    B = 128
    time = torch.full((B,), 50) # [B]
    # print(time)
    out = torch.gather(sqrt_alphas_cumprod, -1, time)