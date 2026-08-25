from typing import Optional

from torch.nn import Parameter, Module
from einops import einsum
import torch

class HomeCookedLinear(Module):
    def __init__(
        self,
        in_features: int, 
        out_features: int, 
        device: Optional[torch.device] = None, 
        dtype: Optional[torch.dtype] = None
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        parameter_kwargs = {"device": device, "dtype": dtype}

        self.weight = Parameter(torch.empty((self.out_features, self.in_features), **parameter_kwargs))
        # Initialize weights
        sigma = 2/(self.in_features + self.out_features)
        torch.nn.init.trunc_normal_(self.weight, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "out_dim in_dim, ... in_dim -> ... out_dim")
