
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F

from utils import dist


# this file only provides the VectorQuantizer2 used in VQVAE
__all__ = ['VectorQuantizer2',]


class VectorQuantizer2(nn.Module):
    # VQGAN originally use beta=1.0, never tried 0.25; SD seems using 0.25
    def __init__(
        self, vocab_size, Cvae, using_znorm, beta: float = 0.25,
        quant_resi=0.5):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.Cvae: int = Cvae
        self.using_znorm: bool = using_znorm
        
        self.quant_resi_ratio = quant_resi
        
        self.quant_resi = PhiShared(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        
        self.register_buffer('ema_vocab_hit_SV', torch.full((1, self.vocab_size), fill_value=0.0))
        self.record_hit = 0
        
        self.beta: float = beta
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)
        
        # only used for progressive training of VAR (not supported yet, will be tested and supported in the future)
        self.prog_si = -1   # progressive training: not supported yet, prog_si always -1
    
    def forward(self, f_BChw: torch.Tensor, ret_usages=False) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        dtype = f_BChw.dtype
        if dtype != torch.float32: f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_hat = torch.zeros_like(f_no_grad)

        mean_vq_loss: torch.Tensor = 0.0
        
        # find the nearest embedding
        if self.using_znorm:
            rest_NC = F.normalize(f_no_grad, dim=-1)
            idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
        else:
            rest_NC = f_no_grad.permute(0, 2, 3, 1).reshape(-1, C)
            d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
            d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
            idx_N = torch.argmin(d_no_grad, dim=1)
        
        hit_V = idx_N.bincount(minlength=self.vocab_size).float()
        if self.training:
            if dist.initialized(): handler = tdist.all_reduce(hit_V, async_op=True)
        
        # calc loss
        idx_Bhw = idx_N.view(B, H, W)
        h_BChw = self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
        h_BChw = self.quant_resi(h_BChw)
        f_hat = f_hat + h_BChw
        
        if self.training and dist.initialized():
            handler.wait()
            if self.record_hit == 0: self.ema_vocab_hit_SV.copy_(hit_V)
            elif self.record_hit < 100: self.ema_vocab_hit_SV.mul_(0.9).add_(hit_V.mul(0.1))
            else: self.ema_vocab_hit_SV.mul_(0.99).add_(hit_V.mul(0.01))
            self.record_hit += 1
        mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)

        f_hat = (f_hat.data - f_no_grad).add_(f_BChw)
        
        margin = tdist.get_world_size() * (f_BChw.numel() / f_BChw.shape[1]) / self.vocab_size * 0.08
        # margin = pn*pn / 100
        if ret_usages: usages = (self.ema_vocab_hit_SV >= margin).float().mean().item() * 100 
        else: usages = None
        return f_hat, usages, mean_vq_loss


class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw):
        return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi
    
    def __getitem__(self, _) -> Phi:
        return self.qresi
