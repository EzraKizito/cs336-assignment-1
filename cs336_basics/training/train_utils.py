
import torch


def get_allocated_memory(device: torch.device) -> int:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    elif device.type == "mps":
        # Returns current allocated Metal memory in bytes
        return torch.mps.current_allocated_memory()
    return 0