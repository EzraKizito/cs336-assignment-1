# Transformer block

# Input x is first normalized, then passes through the MHA block
# and then we add the skip connection to yield x1

# Input x1 is then normalized, passes through the position-wise
# feed-forward network, then we add the skip connection to yield y

from typing import Optional

from torch import nn 
import torch

from cs336_basics.transformer.ffn import FeedForwardNetwork
from cs336_basics.transformer.attention import CausalMultiHeadSelfAttention
from cs336_basics.transformer.rmsnorm import RMSNorm
from cs336_basics.transformer.rope import RotaryPositionalEmbedding

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        rope: RotaryPositionalEmbedding,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads

        parameter_kwargs = {"device": device, "dtype": dtype}
        self.ln1 = RMSNorm(
            d_model=self.d_model,
            **parameter_kwargs
        )
        self.ln2 = RMSNorm(
            d_model=self.d_model, 
            **parameter_kwargs
        )
        self.ffn = FeedForwardNetwork(
            d_model=self.d_model,
            d_ff=self.d_ff, 
            **parameter_kwargs
        )
        self.attn = CausalMultiHeadSelfAttention(
            rope=rope,
            d_model=self.d_model,
            num_heads=self.num_heads, 
            **parameter_kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        normed_x = self.ln1.forward(x)
        sublayer_1_output = x + self.attn.forward(normed_x)

        sublayer_2_input = self.ln2.forward(sublayer_1_output)
        sublayer_2_output = sublayer_1_output + self.ffn.forward(sublayer_2_input)

        return sublayer_2_output