import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math

import ptwt
import numpy as np


###################################
# DataEmbedding
###################################
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


###################################
# Inception Block
###################################
class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res



def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


def Wavelet_for_Period(x, scale=16):
    scales = 2 ** np.linspace(-1, scale, 8)
    coeffs, freqs = ptwt.cwt(x, scales, "morl")
    return coeffs, freqs


class Wavelet(nn.Module):
    def __init__(self, configs):
        super(Wavelet, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.k = configs.top_k
        self.period_coeff = configs.period_coeff
        self.scale = configs.wavelet_scale

        self.period_conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels),
        )

        self.scale_conv = nn.Conv2d(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                kernel_size=(8, 1),
                stride=1,
                padding=(0, 0),
                groups=configs.d_model)
    
        self.projection = nn.Linear(self.seq_len + self.pred_len, self.pred_len, bias=True)

    def forward(self, x):
        # FFT
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.period_conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
            
        if len(res) > 0:
            res = torch.stack(res, dim=-1)
            # adaptive aggregation
            period_weight = F.softmax(period_weight, dim=1)
            period_weight = period_weight.unsqueeze(
                1).unsqueeze(1).repeat(1, T, N, 1)

        # Wavelet
        coeffs = Wavelet_for_Period(x.permute(0, 2, 1), self.scale)[0].permute(1, 2, 0, 3).float()

        wavelet_res = self.period_conv(coeffs)

        wavelet_res = self.scale_conv(wavelet_res).squeeze(2).permute(0, 2, 1)

        if len(res) > 0:
            
            res = (1 - self.period_coeff) * wavelet_res + (self.period_coeff) * torch.sum(res * period_weight, -1)

        else:
            res = wavelet_res
            
            res = res + x

            return self.projection(res.permute(0, 2, 1)).permute(0, 2, 1)
            
        res = res + x

        return res


class WFTNet(nn.Module):
    def __init__(self, configs):
        super(WFTNet, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.layer = configs.e_layers
        self.wavelet_model = nn.ModuleList([Wavelet(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.dropout,
        )
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)

        # c_out = 1   daily return
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forecast(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.wavelet_model[i](enc_out))

        # project back
        dec_out = self.projection(enc_out)
        return dec_out

    def forward(self, x):
        dec_out = self.forecast(x)
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
