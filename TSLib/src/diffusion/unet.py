import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PosEmbed(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    
class Attention(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.to_qkv = nn.Conv1d(dim, dim*3, kernel_size=1, bias=False)
        self.scale = self.dim ** -0.5

    def forward(self, x):
        B, D, L = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1) # [B, D, L]
        # multi-head
        q = q.reshape(-1, self.n_heads, D // self.n_heads, L) # [B, H, D_, L] 
        k = k.reshape(-1, self.n_heads, D // self.n_heads, L) # [B, H, D_, L] 
        v = v.reshape(-1, self.n_heads, D // self.n_heads, L) # [B, H, D_, L] 
        q = q * self.scale
        attn = F.softmax(torch.einsum('bhdi,bhdj->bhij', q, k), dim=-1)
        out = torch.einsum('bhij,bhdj->bhid', attn, v) # [B, H, L, D_]
        out = out.permute(0, 1, 3, 2) # [B, H, D_, L]
        out = out.reshape(-1, D, L) # [B, D, L]
        # out_proj
        return out

def upsample(dim, dim_out = None):
    dim_out = dim_out or dim
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, dim_out, 3, padding = 1)
    )

def downsample(dim, dim_out = None):
    dim_out = dim_out or dim
    return nn.Conv1d(dim, dim_out, 4, 2, 1)

class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, kernel_size=3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        # x: [B, D?, L], scale_shift: tuple (scale, shift) [B, D/2, 1]
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift # [B, D/2, 1], [B, D/2, 1]
            x = x * (scale + 1) + shift
        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_embed_dim=None, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, dim_out * 2)
        ) if time_embed_dim is not None else None
        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_embed=None):
        # x: [B, D, L], time_embed: [B, 4D]
        if self.mlp is not None and time_embed is not None:
            time_embed = self.mlp(time_embed) # [B, 2D]
            time_embed = time_embed.unsqueeze(-1) # [B, D] -> [B, D, 1]
            scale_shift = time_embed.chunk(2, dim = 1) # tuple: ([B, D/2, 1], [B, D/2, 1])
        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class Unet(nn.Module):
    '''Unet in diffusion model'''
    # def __init__(self, dim, channels=3, n_heads=4, dropout=0.):
    def __init__(self, dim, n_heads=4, dropout=0.):
        super().__init__()
        self.dim = dim # D
        # self.channels = channels
        # dim
        scales = [1]
        dims = [dim]
        dims.extend([dim*scale for scale in scales]) # [D, D, 2D, 4D, 8D]
        in_out = list(zip(dims[:-1], dims[1:])) # [(D, D), (D, 2D), (2D, 4D), (4D, 8D)]
        # time embedding
        time_embed_dim = dim * 4
        pos_emb_theta = 10000
        pos_embed = PosEmbed(dim, pos_emb_theta)
        self.time_mlp = nn.Sequential(
            pos_embed,
            nn.Linear(dim, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        # in block
        self.in_block = nn.Conv1d(dim, dim, 7, padding=3)
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out) # 4
        # downs
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                # ResnetBlock(dim_in, dim_in, time_embed_dim, dropout),
                ResnetBlock(dim_in, dim_in, time_embed_dim, dropout),
                RMSNorm(dim_in),
                Residual(Attention(dim_in)),
                downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
            ]))
        # mids
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_embed_dim, dropout)
        self.mid_norm = RMSNorm(mid_dim)
        self.mid_attn = Residual(Attention(mid_dim, n_heads))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_embed_dim, dropout)
        # ups
        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = (i == (len(in_out) - 1))
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_embed_dim, dropout), # copy & concat
                RMSNorm(dim_out),
                Residual(Attention(dim_out)),
                upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding=1)
            ]))
        # out block
        self.out_block = nn.Conv1d(self.dim * 2, self.dim, 1)

    def forward(self, x, time, cond=None):
        # cond: [B, N, D]
        if cond is not None:
            x += cond
        # x: [B, D, L], time: denoise timestep [B]
        x = self.in_block(x)
        shortcut = x.clone() 
        # time embedding
        t = self.time_mlp(time) # [B, 4D] 
        hiddens = []
        # downsample
        for block, norm, attn, down in self.downs:
            x = block(x, t)
            x = norm(x)
            x = attn(x)
            hiddens.append(x)
            x = down(x)
        # mid
        x = self.mid_block1(x, t)
        x = self.mid_norm(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        # upsample
        for block, norm, attn, up in self.ups:
            x = torch.cat((x, hiddens.pop()), dim=1) # copy & concat
            x = block(x, t)
            x = norm(x)
            x = attn(x)
            x = up(x)
        x = torch.concat([x, shortcut], dim=1)
        x = self.out_block(x)
        return x # [B, D, L]
    
class GuidedUnet(nn.Module):
    def __init__(self, dim, n_heads=4, guidance_scale=3):
        super().__init__()
        self.dim = dim
        self.unet = Unet(self.dim, n_heads)
        self.guidance_scale = guidance_scale # classifier-free guidance
        self.cond_embed = nn.Conv1d()


    def forward(self, x, t, cond):
        # cond: [B, D, L]
        cond_embed = torch.zeros(cond.shape, device=cond.device) # FIXME temp implementation
        place_vector = torch.zeros(cond.shape, device=cond.device)
        cond_embed = self.cond_embed(cond_embed)
        place_embed = self.cond_embed(place_embed)
        cond_noise = self.unet(x, t, cond_embed)
        uncond_noise = self.unet(x, t, place_vector)
        # classierier-free guidance
        pred_noise = cond_noise + self.guidance_scale * (cond_noise - uncond_noise)
        return pred_noise