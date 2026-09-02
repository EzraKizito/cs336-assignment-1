from typing import Optional

import torch 

from torch import nn 
from einops import einsum
from cs336_basics.transformer.linear import Linear

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
        self.w1 = Linear(in_features=self.d_model, out_features=self.d_ff, **parameter_kwargs)
        self.w2 = Linear(in_features=self.d_ff, out_features=self.d_model, **parameter_kwargs)
        self.w3 = Linear(in_features=self.d_model, out_features=self.d_ff, **parameter_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        in_dtype = x.dtype 
        x = x.to(torch.float32)
        w1_x = self.w1.forward(x)
        silu_output = w1_x * torch.sigmoid(w1_x) 
        w3_x = self.w3.forward(x)
        l1_activations = silu_output * w3_x 
        result = self.w2.forward(l1_activations)

        return result.to(in_dtype)


        