import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from math import sqrt

###################################
# PatchEmbed
###################################
class PatchEmbed(nn.Module):
    def __init__(self, args, num_p=1, d_model=None):
        super(PatchEmbed, self).__init__()
        self.num_p = num_p
        self.patch = args.seq_len // self.num_p
        self.d_model = args.d_model if d_model is None else d_model

        self.proj = nn.Linear(self.patch, self.d_model, bias=False)

    def forward(self, x, x_mark):
        x = torch.cat([x, x_mark], dim=-1).transpose(-1, -2)
        x = self.proj(x.reshape(*x.shape[:-1], self.num_p, self.patch))
        return x

###################################
# TSMixer
###################################
class TSMixer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(TSMixer, self).__init__()

        self.attention = attention
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, q, k, v, res=False, attn=None):
        B, L, _ = q.shape
        _, S, _ = k.shape
        H = self.n_heads

        q = self.q(q).view(B, L, H, -1)
        k = self.k(k).view(B, S, H, -1)
        v = self.v(v).view(B, S, H, -1)

        out, attn = self.attention(
            q, k, v,
            res=res, attn=attn
        )
        out = out.view(B, L, -1)

        return self.out(out), attn

###################################
# ResAttention
###################################
class ResAttention(nn.Module):
    def __init__(self, attention_dropout=0.1, scale=None):
        super(ResAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, res=False, attn=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous(), A

###################################
# TSEncoder
###################################
class TSEncoder(nn.Module):
    def __init__(self, attn_layers):
        super(TSEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        return x, attns

###################################
# t_layer
###################################
def PeriodNorm(x, period_len=24):
    if len(x.shape) == 3:
        x = x.unsqueeze(-2)
    b, c, n, t = x.shape
    x_patch = [x[..., period_len-1-i:-i+t] for i in range(0, period_len)]
    x_patch = torch.stack(x_patch, dim=-1)

    mean = x_patch.mean(4)
    mean = F.pad(mean.reshape(b * c, n, -1), mode='replicate', pad=(period_len-1, 0)).reshape(b, c, n, -1)
    out = x - mean

    out, mean = out.squeeze(-2), mean.squeeze(-2)
    return out, mean, 1

class t_layer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", stable=True, num_p=None):
        super(t_layer, self).__init__()

        self.stable = stable
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x = self.temporal_attn(x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))

        return self.norm2(x + y), None

    def temporal_attn(self, x):
        b, c, n, d = x.shape
        new_x = x.reshape(-1, n, d)

        qk = new_x
        if self.stable:
            with torch.no_grad():
                qk, _, _ = PeriodNorm(new_x, 6)
        new_x = self.attention(qk, qk, new_x)[0]
        new_x = new_x.reshape(b, c, n, d)
        return new_x

###################################
# d_layer
###################################
class d_layer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu",
                 in_p=30, out_p=4, stable=True):
        super(d_layer, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.in_p = in_p
        self.out_p = out_p

        self.attention = attention
        self.conv1 = nn.Conv1d(self.in_p, self.out_p, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv1d(self.out_p + 1, self.out_p, 1, 1, 0, bias=False)

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x = self.down_attn(x)
        y = x = self.norm1(new_x)

        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))

        return self.norm2(x + y), None

    def down_attn(self, x):
        b, c, n, d = x.shape
        x = x.reshape(-1, n, d)
        new_x = self.conv1(x)
        new_x = self.conv2(torch.cat([new_x, x.mean(-2, keepdim=True)], dim=-2)) + new_x
        new_x = self.attention(new_x, x, x)[0] + new_x
        new_x = self.dropout(new_x.reshape(b, c, -1, d))
        return new_x

###################################
# c_layer
###################################
class c_layer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu", stable=False, enc_in=None, axial=True):
        super(c_layer, self).__init__()

        self.stable = stable
        self.pad_rc = math.ceil((enc_in + 4) ** 0.5)
        self.pad_ch = nn.ConstantPad1d((0, self.pad_rc ** 2 - (enc_in + 4)), 0)

        d_ff = d_ff or 4 * d_model
        self.attention1 = attention
        self.attention2 = copy.deepcopy(attention)
        self.attn_ch = self.axial_attn if axial is True else self.full_attn

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x = self.attn_ch(x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))

        return self.norm2(x + y), None

    def axial_attn(self, x):
        b, c, n, d = x.shape
        new_x = rearrange(x, 'b c n d -> (b n) c d')

        new_x = self.pad_ch(new_x.transpose(-1, -2))\
            .transpose(-1, -2).reshape(-1, self.pad_rc, d)

        new_x = self.attention1(new_x, new_x, new_x)[0]
        new_x = rearrange(new_x, '(b r) c d -> (b c) r d', b=b*n)
        new_x = self.attention2(new_x, new_x, new_x)[0] + new_x
        new_x = rearrange(new_x, '(b n c) r d -> b (r c) n d', b=b, n=n)
        return new_x[:, :c, ...]

    def full_attn(self, x):
        b, c, n, d = x.shape

        new_x = rearrange(x, 'b c n d -> (b n) c d')
        new_x = self.attention1(new_x, new_x, new_x)[0]
        new_x = rearrange(new_x, '(b n) c d -> b c n d', b=b)
        return new_x




class TimeBridge(nn.Module):
    def __init__(self, configs):
        super(TimeBridge, self).__init__()

        self.c_in = configs.enc_in
        self.period = configs.period
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_p = self.seq_len // self.period
        configs.num_p = self.num_p

        # long-term and short-term embedding
        self.embedding = PatchEmbed(configs, num_p=self.num_p)
        # Encoder-only architecture

        layers = self.layers_init(configs)
        self.encoder = TSEncoder(layers)

        self.decoder = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(configs.num_p * configs.d_model, configs.pred_len, bias=False)
        )

        self.projection = nn.Linear(configs.enc_in, configs.c_out, bias=True)

    def layers_init(self, configs):

        layer_p1 = [t_layer(
            TSMixer(ResAttention(),configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, stable=True, num_p=self.num_p,
            dropout=configs.dropout, activation=configs.activation
        ) for i in range(configs.t_layers)]
        layer_p2 = t_layer(
            TSMixer(ResAttention(), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, stable=False, num_p=configs.num_p,
            dropout=configs.dropout, activation=configs.activation
        )
        layer_d = d_layer(
            TSMixer(ResAttention(), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, stable=False,
            in_p=self.num_p, out_p=configs.num_p,
            dropout=configs.dropout, activation=configs.activation
        )
        layer_c_full = [c_layer(
            TSMixer(ResAttention(), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, stable=False, enc_in=self.c_in,
            dropout=configs.dropout, activation=configs.activation, axial=False
        ) for i in range(configs.e_layers - 2)]

        layer_c_axial = [c_layer(
            TSMixer(ResAttention(), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, stable=False, enc_in=self.c_in,
            dropout=configs.dropout, activation=configs.activation
        ) for i in range(configs.e_layers - 2)]

        layer_c = layer_c_axial if self.c_in > 100 else layer_c_full
        return [*layer_p1, layer_d, layer_p2, *layer_c] if configs.num_p > 1 \
            else [*layer_p1, layer_d, *layer_c]

    def forecast(self, x):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        x_mark = torch.zeros((*x.shape[:-1], 4), device=x.device)
        x = self.embedding(x, x_mark)
        x = self.encoder(x)[0][:, :self.c_in, ...]
        x = self.decoder(x).transpose(-1, -2)

        x = self.projection(x)

        return x

    def forward(self, x):
        x = self.forecast(x)
        return x[:, -self.pred_len:, :]  # [B, L, D]