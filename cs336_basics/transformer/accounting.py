import torch 

from torch import nn 
from cs336_basics.transformer.transformer import TransformerLM
from cs336_basics.tokenize.train_bpe import format_byte_size

with torch.device("meta"):
    model = TransformerLM(
        d_model=1600,
        num_heads=25,
        d_ff=4288,
        vocab_size=50257,
        context_length=1024,
        num_layers=48,
        theta=50.0
    )

trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print(f"Trainable params: {trainable_params:,}")

memory_bytes = trainable_params * 4 
print(f"\nMemory consumed by GPT-2 XL: {format_byte_size(memory_bytes)}")
