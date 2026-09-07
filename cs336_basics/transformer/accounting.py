import torch 

from cs336_basics.transformer.transformer import TransformerLM
from cs336_basics.tokenize.train_bpe import format_byte_size

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

if __name__ == "__main__":
    main()