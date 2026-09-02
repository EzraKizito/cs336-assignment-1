import torch
from typing import Any, Optional
from einops import einsum, rearrange
from torch import nn

from cs336_basics.transformer.rope import RotaryPositionalEmbedding
from cs336_basics.transformer.linear import Linear

def softmax(x: torch.Tensor, dim: int = -1):
	maxes = torch.max(x, dim=dim, keepdim=True)[0]

	exponentiated = torch.exp(x - maxes)

	sum_exponentiated = torch.sum(exponentiated, dim=dim, keepdim=True)

	return exponentiated / sum_exponentiated

def scaled_dot_product_attention(
	Q: torch.Tensor, 
	K: torch.Tensor, 
	V: torch.Tensor, 
	mask: Optional[torch.Tensor] = None
): 
	d_k = torch.tensor(K.shape[-1], dtype=torch.float32)
	queries_keys = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / torch.sqrt(d_k)
	if mask is not None:
		additive_mask = torch.where(mask, 0, -torch.inf)
		queries_keys = queries_keys + additive_mask
	
	weights = softmax(queries_keys)

	return einsum(weights, V, "... q k, ... k d_v -> ... q d_v")


class CausalMultiHeadSelfAttention(nn.Module):
	def __init__(
		self,
		rope: Optional[RotaryPositionalEmbedding] = None,
		token_positions: Optional[torch.Tensor] = None,
		d_model: int = 64,
		num_heads: int = 8, 
		device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
	) -> None:
		super().__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		assert self.d_model % self.num_heads == 0

		self.rope = rope
		self.token_positions = token_positions
		self.device = device 
		self.dtype = dtype

		parameter_kwargs = {"device": self.device, "dtype": self.dtype}
		# Initialize the weight parameters for down-projecting from d_model to d_k
		self.q_proj = Linear(in_features=self.d_model, out_features=self.d_model, **parameter_kwargs)
		self.k_proj = Linear(in_features=self.d_model, out_features=self.d_model, **parameter_kwargs)
		self.v_proj = Linear(in_features=self.d_model, out_features=self.d_model, **parameter_kwargs)
		self.output_proj = Linear(in_features=self.d_model, out_features=self.d_model, **parameter_kwargs)

	def forward(self, x: torch.Tensor) -> torch.Tensor: 
		# Perform matrix operations from x through Q, K, V matrices

		# First, we construct a mask
		seq_len = x.shape[-2]
		pre_mask = torch.ones(seq_len, seq_len, device=self.device, dtype=self.dtype)
		mask = torch.tril(pre_mask).to(torch.bool)

		# Project all heads and reshape
		x_q = self.q_proj(x) # batch seq_len d_model
		x_k = self.k_proj(x)
		x_v = self.v_proj(x)

		# Reshape to h batch seq_len d_k
		x_q = rearrange(x_q, "... seq_len (h d_k) -> h ... seq_len d_k", h=self.num_heads)
		x_k = rearrange(x_k, "... seq_len (h d_k) -> h ... seq_len d_k", h=self.num_heads)
		x_v = rearrange(x_v, "... seq_len (h d_v) -> h ... seq_len d_v", h=self.num_heads)

		if self.rope:
			x_q = self.rope(x_q, self.token_positions)
			x_k = self.rope(x_k, self.token_positions)

		mha = scaled_dot_product_attention(x_q, x_k, x_v, mask) # h batch_size ... seq_len d_v
		up_project_mha = rearrange(mha, "h ... d_v -> ... (h d_v)") # batch_size ... seq_len d_model

		result = self.output_proj(up_project_mha)
		return result

