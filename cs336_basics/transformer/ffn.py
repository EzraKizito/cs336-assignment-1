from typing import Optional

import torch 

from torch import nn 
from einops import einsum

class FeedForwardNetwork(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        d_ff: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> None:  
        super().__init__()
        self.d_model = d_model

        # TODO Set d_ff to 8/3 size of d_model while keeping it multiple of 64
        self.d_ff = d_ff
        parameter_kwargs = {"device": device, "dtype": dtype}
        self.w1_weight = nn.Parameter(
            torch.empty((self.d_ff, self.d_model), **parameter_kwargs)
        )
        self.w2_weight = nn.Parameter(
            torch.empty((self.d_model, self.d_ff), **parameter_kwargs)
        )
        self.w3_weight = nn.Parameter(
            torch.empty((self.d_ff, self.d_model), **parameter_kwargs)
        )

        # Initialize weights
        sigma = 2/(self.d_ff + self.d_model)
        nn.init.trunc_normal_(self.w1_weight, std=sigma, a=-3*sigma, b=3*sigma)
        nn.init.trunc_normal_(self.w2_weight, std=sigma, a=-3*sigma, b=3*sigma)
        nn.init.trunc_normal_(self.w3_weight, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        in_dtype = x.dtype 
        x = x.to(torch.float32)
        w1_x = einsum(self.w1_weight, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu_output = w1_x * torch.sigmoid(w1_x) 
        w3_x = einsum(self.w3_weight, x, "d_ff d_model, ... d_model -> ... d_ff")

        l1_activations = silu_output * w3_x 
        result = einsum(
            self.w2_weight, l1_activations, "d_model d_ff, ... d_ff -> ... d_model"
        )

        return result.to(in_dtype)


        