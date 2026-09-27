
import os
from typing import BinaryIO

import torch
import yaml

from cs336_basics.tokenize.tokenizer import BPETokenizer
from cs336_basics.training.loop import load_checkpoint
from cs336_basics.transformer.attention import softmax
from cs336_basics.transformer.transformer import TransformerLM


class LMDecoder:
    def __init__(
        self, 
        language_model: TransformerLM,
        eos_token_id: int = 256,
        pad_token_id: int = 256,
    ) -> None:
        self.lm = language_model 
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def decode(
        self,
        inputs: torch.Tensor,
        max_generated_tokens: int = 50, 
        temperature: float = 0.01, 
        top_p_threshold: float = 0.95,
        min_tokens_to_keep: int = 5
    ):
        generated_tokens = 0
        start_index = inputs.size(-1)

        # Track whether a sequence has finished decoding
        output_size = inputs.shape[:-1]
        finished = torch.zeros(output_size, dtype=torch.bool, device=inputs.device)
        pad_tensor = torch.tensor(self.pad_token_id, device=inputs.device)

        assert generated_tokens < max_generated_tokens, "Nothing to generate if max tokens are negative"
        # pass inputs through model
        while generated_tokens < max_generated_tokens:
            with torch.no_grad():
                outputs = self.lm.forward(inputs)
            
            logits = outputs[:, -1, :]
            
            probabilities = softmax(logits, temperature=temperature)

            # Top-p 
            sorted_probabilities, sorted_indices = torch.sort(probabilities, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probabilities, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p_threshold
            sorted_indices_to_remove[..., min_tokens_to_keep:] = sorted_indices_to_remove[..., :-min_tokens_to_keep].clone()
            sorted_indices_to_remove[..., 0:min_tokens_to_keep] = 0

            # Zero out the probabilities to be removed
            sorted_probabilities[sorted_indices_to_remove] = 0.0
            filtered_probabilities = torch.zeros_like(probabilities).scatter_(
                dim=-1, index=sorted_indices, src=sorted_probabilities
            )
            sum_filtered = torch.sum(filtered_probabilities, dim=-1, keepdim=True)
            scaled_probabilities = filtered_probabilities / (sum_filtered + 1e-9)

            # Sample token id from resulting distribution [ B,1 ]
            next_tokens = torch.multinomial(scaled_probabilities, num_samples=1)

            # append token to sequence
            next_tokens = next_tokens.squeeze(-1)

            next_tokens = torch.where(finished, pad_tensor, next_tokens)
            finished = finished | (next_tokens == self.eos_token_id)

            inputs = torch.cat((inputs, next_tokens.unsqueeze(-1)), dim=-1)
            generated_tokens += 1
            if finished.all():
                break

        return inputs[..., start_index:]

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | BinaryIO | os.PathLike, model: TransformerLM):
        load_checkpoint(src=checkpoint_path, model=model, load_optimizer=False)

        return LMDecoder(language_model=model)


def main():
    # TODO: make decoding modular (i.e via a CLI) to pass a config path (For model initialization)
    # and a checkpoint path for model loading (both required to decode)
    config_path = "/Users/ezrakizito/dev/learn/stanford-cs336/assignment1-basics/configs/lr_tuning/base_train_hp_tuning_mps.yaml"
    ckpt_path = "/Users/ezrakizito/dev/learn/stanford-cs336/assignment1-basics/ckpts/base_train_hp_tuning_mps/final.pt"
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)

    device = torch.device("cpu")
    lm = TransformerLM(
        d_model=cfg["model"]["d_model"], 
        num_heads=cfg["model"]["num_heads"],
        d_ff=cfg["model"]["d_ff"],
        theta=cfg["model"]["rope_theta"], 
        vocab_size=cfg["model"]["vocab_size"], 
        context_length=cfg["model"]["context_length"], 
        num_layers=cfg["model"]["num_layers"],
        device=device
        )
    decoder = LMDecoder.from_checkpoint(checkpoint_path=ckpt_path, model=lm)

    inputs = torch.tensor([[257, 280, 287, 265,]])

    outputs = decoder.decode(inputs, temperature=1.0)
    print(outputs)


if __name__ == "__main__":
    main()