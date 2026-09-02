# Transformer block

# Input x is first normalized, then passes through the MHA block
# and then we add the skip connection to yield x1

# Input x1 is then normalized, passes through the position-wise
# feed-forward network, then we add the skip connection to yield y

from typing import Any, Optional

from torch import nn 
import torch

from cs336_basics.transformer.ffn import FeedForwardNetwork
from cs336_basics.transformer.attention import CausalMultiHeadSelfAttention
from cs336_basics.transformer.rmsnorm import RMSNorm
from cs336_basics.transformer.rope import RotaryPositionalEmbedding
from cs336_basics.transformer.embedding import Embedding
from cs336_basics.transformer.linear import Linear

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
        normed_x = self.ln1(x)
        sublayer_1_output = x + self.attn(normed_x)

        sublayer_2_input = self.ln2(sublayer_1_output)
        sublayer_2_output = sublayer_1_output + self.ffn(sublayer_2_input)

        return sublayer_2_output

class TransformerLM(nn.Module): 
    def __init__(
        self, 
        d_model: int,
        num_heads: int, 
        d_ff: int, 
        theta: float,
        vocab_size: int, 
        context_length: int, 
        num_layers: int, 
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff # Hidden dim size
        self.num_layers = num_layers
        parameter_kwargs = {"device": device, "dtype": dtype}
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k = self.d_model // self.num_heads,
            max_seq_len=context_length, 
            device=device
        )
        self.layers = nn.ModuleList()

        for _ in range(num_layers): 
            self.layers.append(
                TransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads, 
                    d_ff=self.d_ff,
                    rope=self.rope,
                    **parameter_kwargs
                )
            )

        self.ln_final = RMSNorm(d_model=self.d_model)
        self.lm_head = Linear(in_features=self.d_model, out_features=vocab_size, **parameter_kwargs)


    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.embedding(x)
        for layer in self.layers: 
            x = layer(x)

        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
