from typing import Any, Optional

import torch 

from torch import nn 
from einops import einsum, rearrange

class RotaryPositionalEmbedding(nn.Module): 
    def __init__(
        self,
        theta: float, 
        d_k: int, 
        max_seq_len: int, 
        device: Optional[torch.device] = None, 
    ) -> None:
        super().__init__()
        self.theta = theta 
        self.d = d_k 
        self.max_seq_len = max_seq_len 


        self.device = device

        # Inefficient implementation: construct full d*d matrix, store it, and 
        # pass each key and query vector through it. But matrix is sparse.

        # Instead, we can compute once the cosine and sine values up to the max
        # sequence length, store them in a 2d buffer, and only apply the relevant
        # transformations
        indices = torch.arange(self.max_seq_len)
        d_range = torch.arange(start=1, end=self.d//2 + 1)
        transformed_theta = self.theta ** ((2*d_range - 2)/ self.d)
        angles = einsum(
            indices, torch.reciprocal(transformed_theta), 
            "seq_len, d_half -> seq_len d_half"
        )
        self.register_buffer('angles', angles, persistent=False)
        self.angles: torch.Tensor # satisfy linter

    def forward(
        self,
        x: torch.Tensor, 
        token_positions: Optional[torch.Tensor]
    ) -> torch.Tensor: 

        seq_len = x.shape[-2]
        if token_positions is None: 
            token_positions = torch.arange(end=seq_len, device=self.device)

        relevant_angles = self.angles[token_positions] # Shape is `batch seq_len d/2`
        c = torch.cos(relevant_angles)
        s = torch.sin(relevant_angles)

        x = rearrange(x, "... seq_len (d_half c) -> ... seq_len d_half c", c=2) # `batch seq-len d_half 2`
        rotation = rearrange([c, -s, s, c], "(h w) ... -> ... h w", h=2, w=2) # batch seq_len d_half 2 2
        rotated = einsum(rotation, x, "... h w, ... w -> ... h")
        rotated = rearrange(rotated, "... d_2 c -> ... (d_2 c)")
        return rotated