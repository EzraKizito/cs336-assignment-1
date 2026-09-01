import torch
from typing import Any, Optional
from einops import einsum, rearrange
from torch import nn

from cs336_basics.transformer.rope import RotaryPositionalEmbedding

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
		self.q_proj_weight = nn.Parameter(
			torch.empty((d_model, d_model), **parameter_kwargs)
		)
		self.k_proj_weight = nn.Parameter(
			torch.empty((d_model, d_model), **parameter_kwargs)
		)
		self.v_proj_weight = nn.Parameter(
			torch.empty((d_model, d_model), **parameter_kwargs)
		)
		self.o_proj_weight = nn.Parameter(
			torch.empty((d_model, d_model), **parameter_kwargs)
		)
		sigma = 2/(self.d_model + self.d_model)
		nn.init.trunc_normal_(self.q_proj_weight, std=sigma, a=-3*sigma, b=3*sigma)
		nn.init.trunc_normal_(self.k_proj_weight, std=sigma, a=-3*sigma, b=3*sigma)
		nn.init.trunc_normal_(self.v_proj_weight, std=sigma, a=-3*sigma, b=3*sigma)
		nn.init.trunc_normal_(self.o_proj_weight, std=sigma, a=-3*sigma, b=3*sigma)

	def forward(self, x: torch.Tensor) -> torch.Tensor: 
		# Perform matrix operations from x through Q, K, V matrices

		# First, we construct a mask
		seq_len = x.shape[-2]
		pre_mask = torch.ones(seq_len, seq_len, device=self.device, dtype=self.dtype)
		mask = torch.tril(pre_mask).to(torch.bool)

		# Now, we rearrange projection weights into projections
		q_proj_weight = rearrange(
			self.q_proj_weight, "(h d_q) d_model -> h d_q d_model", h=self.num_heads
		)
		k_proj_weight = rearrange(
			self.k_proj_weight, "(h d_k) d_model -> h d_k d_model", h=self.num_heads
		)
		v_proj_weight = rearrange(
			self.v_proj_weight, "(h d_v) d_model -> h d_v d_model", h=self.num_heads
		)

		x_q = einsum(
			q_proj_weight, x, 
			"h d_q d_model, ... seq_len d_model -> h ... seq_len d_q"
		)
		x_k = einsum(
			k_proj_weight, x, 
			"h d_k d_model, ... seq_len d_model -> h ... seq_len d_k"
		)
		x_v = einsum(
			v_proj_weight, x, 
			"h d_v d_model, ... seq_len d_model -> h ... seq_len d_v"
		)

		if self.rope:
			x_q = self.rope.forward(x_q, self.token_positions)
			x_k = self.rope.forward(x_k, self.token_positions)

		mha = scaled_dot_product_attention(x_q, x_k, x_v, mask) # h batch_size ... seq_len d_v
		up_project_mha = rearrange(mha, "h ... d_v -> ... (h d_v)") # batch_size ... seq_len d_model

		return einsum(self.o_proj_weight, up_project_mha, "d_out d_in, ... seq_len d_in -> ... seq_len d_out")

