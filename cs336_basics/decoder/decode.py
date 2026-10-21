
import os
import pickle
from typing import BinaryIO

import torch
import yaml
from torch.nn.utils.rnn import pad_sequence

from cs336_basics.tokenize.tokenizer import BPETokenizer
from cs336_basics.training.loop import load_checkpoint
from cs336_basics.transformer.attention import softmax
from cs336_basics.transformer.transformer import TransformerLM

SPECIAL_TOKENS = ["<|endoftext|>"]

class LMDecoder:
    def __init__(
        self, 
        language_model: TransformerLM,
        tokenizer: BPETokenizer,
        eos_token_id: int = 256,
        pad_token_id: int = 256,
    ) -> None:
        self.lm = language_model
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def decode(
        self,
        input_text: str | list[str],
        max_generated_tokens: int = 256, 
        temperature: float = 0.01, 
        top_p_threshold: float = 0.95,
        min_tokens_to_keep: int = 5
    ):

        inputs = self.text_to_ids(input_text)
        start_index = generated_tokens = inputs.size(-1)

        # Track whether a sequence has finished decoding
        output_size = inputs.shape[:-1]
        finished = torch.zeros(output_size, dtype=torch.bool, device=inputs.device)
        pad_tensor = torch.tensor(self.pad_token_id, device=inputs.device)

        assert generated_tokens < max_generated_tokens, "Number of tokens already beyond context"
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

        generated_outputs = self.ids_to_text(inputs[..., start_index:])
        return generated_outputs

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | BinaryIO | os.PathLike, model: TransformerLM, tokenizer: BPETokenizer):
        load_checkpoint(src=checkpoint_path, model=model, load_optimizer=False)

        return LMDecoder(language_model=model, tokenizer=tokenizer)

    def text_to_ids(self, text: str | list[str]) -> torch.Tensor:
        if isinstance(text, list):
            ids = [torch.tensor(self.tokenizer.encode(t), device=self.lm.device) for t in text]
            # Have to ensure that ids are of same length, so should pad with eos token
            inputs = pad_sequence(ids, batch_first=True, padding_value=self.pad_token_id)
        else:
            inputs = torch.tensor([self.tokenizer.encode(text)],  device=self.lm.device)
        return inputs
        
    def ids_to_text(self, outputs: torch.Tensor) -> list[str]:
        text_outputs = []
        outs = outputs.tolist()
        if isinstance(outs[0], list):
            for id_batch in outs:
                # Truncate each list to only one occurrence of end of token
                ids_to_decode = id_batch[:]
                if self.pad_token_id in id_batch:
                    index = id_batch.index(self.pad_token_id)
                    if index != -1:
                        ids_to_decode = id_batch[:index+1]
                text_outputs.append(self.tokenizer.decode(ids_to_decode))
            
        else:
            text_outputs.append(self.tokenizer.decode(outs))
        return text_outputs


def main():
    # TODO: make decoding modular (i.e via a CLI) to pass a config path (For model initialization)
    # and a checkpoint path for model loading (both required to decode)
    config_path = "/Users/ezrakizito/dev/learn/stanford-cs336/assignment1-basics/configs/lr_tuning/base_train_hp_tuning_mps.yaml"
    ckpt_path = "/Users/ezrakizito/dev/learn/stanford-cs336/assignment1-basics/ckpts/base_train_hp_tuning_mps/final.pt"
    ts_vocab_filepath = "/Users/ezrakizito/dev/learn/stanford-cs336/assignment1-basics/trained_tokenizer/tinystories/vocab.pkl"
    ts_merges_filepath = "/Users/ezrakizito/dev/learn/stanford-cs336/assignment1-basics/trained_tokenizer/tinystories/merges.pkl"
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=ts_vocab_filepath,
        merges_filepath=ts_merges_filepath,
        special_tokens=SPECIAL_TOKENS
    )
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
    decoder = LMDecoder.from_checkpoint(checkpoint_path=ckpt_path, model=lm, tokenizer=tokenizer)

    input_text = ["Hello there! ", "The princess kissed the ugly frog. ", "I don't want to play with you!"]
    print(f"Context length: {cfg["model"]["context_length"]}")
    outputs = decoder.decode(input_text, temperature=0.7, max_generated_tokens=cfg["model"]["context_length"])

    for i, out in enumerate(outputs):
        print(input_text[i] + out)


if __name__ == "__main__":
    main()