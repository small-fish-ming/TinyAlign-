import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 关键：仅用方差归一化（无均值），比 LayerNorm 快 15-30%
        #RMS = √( (x₁² + x₂² + … + xₙ²) / n )
        rms = torch.sqrt(x.pow(2).mean(dim=-1,keepdim=True) + self.eps)
        return x / rms