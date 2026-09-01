from typing import Optional

import torch

from torch import nn
from einops import reduce, einsum

class RMSNorm(nn.Module): 
    def __init__(
        self,
        d_model: int, 
        eps: float = 1e-5, 
        device: Optional[torch.device] = None, 
        dtype: Optional[torch.dtype] = None
    ) -> None:
        super().__init__()
        self.d_model = d_model 
        self.eps = eps 

        parameter_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.ones((self.d_model), **parameter_kwargs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # x is of shape `batch seq d_model`
        sum_of_squares = reduce(torch.square(x), '... d_model -> ...', 'sum')
        rms_a = torch.sqrt(sum_of_squares / self.d_model + self.eps)

        # rms_a is of shape `batch seq`
        result = einsum(
            torch.reciprocal(rms_a), x, self.weight, 
            "... , ... d_model, d_model -> ... d_model"
        )

        return result.to(in_dtype)