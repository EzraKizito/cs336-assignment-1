import torch

from torch import nn 
from typing import Optional

class HomeCookedEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int, 
        embedding_dim: int, 
        device: Optional[torch.device] = None, 
        dtype: Optional[torch.dtype] = None
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim 

        parameter_kwargs = {"device": device, "dtype": dtype}

        # Embedding matrix
        self.weight = nn.Parameter(
            torch.empty((self.num_embeddings, self.embedding_dim), **parameter_kwargs)
        )

        nn.init.trunc_normal_(self.weight, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Indices are of shape `(batch_size, sequence_length)`. 
        """
        return self.weight[token_ids]

