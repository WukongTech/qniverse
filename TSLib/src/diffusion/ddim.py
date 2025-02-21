import torch
import torch.nn as nn

from .unet import Unet

# https://github.com/sunlin-ai/diffusion_tutorial/blob/main/diffusion_07_DDIM.ipynb

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a

class DDIM(nn.Module):
    def __init__(self, model, diffusion_steps=100, beta_start=0.0001, beta_end=0.04):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.beta = torch.linspace(beta_start, beta_end, self.diffusion_steps) # [steps]
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.model = model
        # self.model = Unet(dim=32)

    def q_xt_x0(self, x0, t):
        # forward diffusion: add noise to x_0 -> x_t
        # x: [B, D, L], t: [B]
        mean = gather(self.alpha_bar.to(x0.device), t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar.to(x0.device), t)
        eps = torch.randn_like(x0).to(x0.device)
        return mean + (var ** 0.5) * eps, eps # also returns noise(eps)
    
    def q_x0(self, xt, t, eps):
        mean = gather(self.alpha_bar.to(xt.device), t) ** -0.5
        var = 1 - gather(self.alpha_bar.to(xt.device), t)
        return mean * (xt - var ** 0.5 * eps)

    def p_xt(self, xt, noise, t, next_t):
        # backward diffusion: denoise
        eta = 0.0
        alpha_t = compute_alpha(self.beta.to(xt.device), t.long())
        alpha_t_next = compute_alpha(self.beta.to(xt.device), next_t.long())
        x0_t = (xt - noise * (1 - alpha_t).sqrt()) / alpha_t.sqrt()
        c1 = (eta * ((1 - alpha_t / alpha_t_next) * (1 - alpha_t_next) / (1 - alpha_t)).sqrt())
        c2 = ((1 - alpha_t_next) - c1 ** 2).sqrt()
        eps = torch.randn(xt.shape, device=xt.device) # noise
        xt_next = alpha_t_next.sqrt() * x0_t + c1 * eps + c2 * noise
        return xt_next

    def p_x0_xt(self, xt, cond=None, steps=50):
        # diffusion_steps: 1000 -> steps: 50
        skip = self.diffusion_steps // steps     
        seq = range(0, self.diffusion_steps, skip)
        seq_next = [-1] + list(seq[:-1])
        x = xt
        n = x.size(0) # bs
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            with torch.no_grad():
                pred_noise = self.model(x, t, cond)
                x = self.p_xt(x, pred_noise, t, next_t)
        return x # x0
