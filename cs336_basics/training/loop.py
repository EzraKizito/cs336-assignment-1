import os
import typing

import numpy as np
import torch
from torch import nn


def data_loading(x: np.typing.NDArray, batch_size: int, context_length: int, device: str):
    inputs = []
    targets = []

    # sample `batch_size` number of starting points without replacement, up to a maximum of N-C
    eligible_indices = np.arange(0, len(x) - context_length)
    starting_indices = np.random.choice(eligible_indices, size=batch_size, replace=False)

    for i in starting_indices:
        inputs.append(x[i:i+context_length])
        targets.append(x[i+1:i+context_length+1])

    input_seq = np.stack(inputs)
    target_seq = np.stack(targets)

    return torch.Tensor(input_seq, device=device), torch.Tensor(target_seq, device=device)

def save_checkpoint(
        model: nn.Module, 
        optimizer: torch.optim.Optimizer,
        iteration: int, 
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
): 
    to_be_saved = {
        "model": model.state_dict(), 
        "optimizer": optimizer.state_dict(), 
        "iteration": iteration
    }

    torch.save(to_be_saved, out)

def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
        model: nn.Module, 
        optimizer: typing.Optional[torch.optim.Optimizer] = None,
        load_optimizer: bool = True
): 
    entire_state = torch.load(src)

    model_dict, optim_dict = entire_state["model"], entire_state["optimizer"]
    model.load_state_dict(model_dict)
    if load_optimizer:
        assert optimizer is not None, "Optimizer must not be none"
        optimizer.load_state_dict(optim_dict)
    return entire_state["iteration"]