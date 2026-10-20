from turtle import back

import torch

from cs336_basics.tokenize.train_bpe import format_byte_size, format_time_duration
from cs336_basics.transformer.transformer import TransformerLM


def find_best_ff_dimension(number: float):
    x = number * 8/ 3
    return round(x / 64) * 64

def lm_accounting(d:int, num_layers: int, d_ff:int | None = None ,s=1024, v=50257):
    if not d_ff: 
        d_ff = find_best_ff_dimension(d)
    attn = num_layers * (8 * s * d**2 + 4 * s**2 * d)
    ffn = num_layers * (6 * s * d * d_ff)
    rope = 6 * s * d
    head = 2 * s * d * v

    total = attn + ffn + rope + head
    pct_dict: dict[str, list[float]] = {
        "attn": [attn, attn / total], 
        "ffn": [ffn, ffn / total], 
        "rope": [rope, rope / total],
        "head": [head, head / total]
    }
    return total, pct_dict

def format_lm_accounting(number: float):
    return f"{number:.3e}".replace("+", "")

def format_pct(number: float):
    return f"{number:.2f} %"

def calculate_optimize_memory_usage_gb(
        num_layers: int, 
        d_model: int,
        batch_size: int,
        vocab_size: int,
        seq_len: int,
):
    model_params = num_layers* (12 * d_model**2 + 2*d_model) + d_model*(vocab_size + 1)
    total_mem = 16*model_params + 8*batch_size*seq_len*vocab_size

    return total_mem / (1024**3)

def calculate_time_to_train(
        num_layers: int = 48, 
        d_model: int = 1600,
        batch_size: int = 1024,
        vocab_size: int = 50257,
        seq_len: int = 1024,
        training_steps = 400000,
        capacity=4.95e14,
        mfu=0.5
):
    params_, _ = lm_accounting(d=d_model, num_layers=num_layers)
    forward_pass = batch_size * params_
    backward_pass = 2 * forward_pass # - 6 * B * S * D?

    total_flops = (forward_pass + backward_pass) * training_steps
    print(f"{total_flops:.3e} FLOPs")
    return format_time_duration(total_flops / (mfu*capacity))
    

def main():
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

    print(f"{model}\n")
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Trainable params: {trainable_params:,}")

    memory_bytes = trainable_params * 4 
    print(f"\nMemory consumed by GPT-2 XL: {format_byte_size(memory_bytes)}")
    gpt_xl_accounting = lm_accounting(d=1600, d_ff=4288, num_layers=48)
    gpt_large_accounting = lm_accounting(d=1280, num_layers=36)
    gpt_medium_accounting = lm_accounting(d=1024, num_layers=24)
    gpt_small_accounting = lm_accounting(d=768, num_layers=12)

    accounts = [gpt_small_accounting, gpt_medium_accounting, gpt_large_accounting, gpt_xl_accounting]
    account_idx = {
        0: "small",
        1: "medium",
        2: "large",
        3: "xlarge"
    }

    print(f"\nGPT-2 XL FLOPs: {format_lm_accounting(gpt_xl_accounting[0])}")

    print(f"\nGPT-2 Large FLOPs: {format_lm_accounting(gpt_large_accounting[0])}")

    print(f"\nGPT-2 Medium FLOPs: {format_lm_accounting(gpt_medium_accounting[0])}")

    print(f"\nGPT-2 Small FLOPs: {format_lm_accounting(gpt_small_accounting[0])}")

    ### PROPORTIONS
    for idx, (_, pct_dict) in enumerate(accounts):
        print(f"\nProportions for GPT-{account_idx[idx]}")

        for part, proportion in pct_dict.items():
            print(f"{part} -- {format_lm_accounting(proportion[0])} FLOPs -- {format_pct(100 * proportion[1])}")

    ### Change context length
    new_gpt_xl_accounting = lm_accounting(d=1600, d_ff=4288, num_layers=48, s=16384)
    print(f"\nGPT XL with increased context length: {format_lm_accounting(new_gpt_xl_accounting[0])} FLOPs")

    for part, proportion in new_gpt_xl_accounting[1].items():
        print(f"{part} -- {format_lm_accounting(proportion[0])} FLOPs -- {format_pct(100 * proportion[1])}")

    optim_mem_usage = calculate_optimize_memory_usage_gb(
                num_layers=48,
                d_model=1600,
                vocab_size=50257,
                batch_size=148, 
                seq_len=1024
            )
    print(f"Total memory used by optimizer: {optim_mem_usage} GB")

    print(f"Time to train: {calculate_time_to_train()}")

if __name__ == "__main__":
    main()