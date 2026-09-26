
import math
import os
import time
from typing import Any, BinaryIO, Dict, Optional

import numpy as np
import torch
import typer
import yaml
from dotenv import load_dotenv

import wandb
from cs336_basics.training.loop import load_checkpoint, save_checkpoint
from cs336_basics.training.loss import cross_entropy_loss
from cs336_basics.training.optimizer import (
    AdamW,
    gradient_clipping,
)
from cs336_basics.transformer.transformer import TransformerLM

load_dotenv()

app = typer.Typer(help="CS336 Training Framework")

def get_batches(x: np.ndarray, batch_size: int, context_length: int, device, seed: int = 42):
    rng = np.random.default_rng(seed)
    seq_len = context_length + 1
    total_tokens = len(x)

    while True:
        # Random boundary shift (0 to context_length)
        starting_offset = rng.integers(0, context_length)
        
        # Calculate exactly how many valid non-overlapping sequences fit
        num_chunks = (total_tokens - starting_offset) // seq_len
        
        # Array of chunk start positions
        indices = np.arange(num_chunks) * seq_len + starting_offset
        rng.shuffle(indices)

        # Yield non-overlapping batches for entire epoch
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            batch_starts = indices[i : i + batch_size]
            
            batch_data = np.stack([x[start : start + seq_len] for start in batch_starts])

            batch_tensor = torch.from_numpy(batch_data).to(device=device, dtype=torch.long)
            
            inputs = batch_tensor[:, :-1]
            targets = batch_tensor[:, 1:]

            yield inputs, targets

def train_fn(
        config: Optional[Dict[str, Any]] = None, 
        resume_from: Optional[str | os.PathLike | BinaryIO] = None, 
        load_optimizer: bool = False
    ):
    """
    Core training function compatible with both standalone runs and wandb sweeps
    Supports resuming. 
    """
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"DEVICE: {device}")
    with wandb.init(
        project="cs336",
        notes="Training runs",
        config=config
    ) as run:
        # Load data
        cfg = run.config
        if cfg["training_dataset"] == "TinyStories":
            dataset_path = "data/encoded/tinystories.dat"
        else:
            dataset_path = "data/encoded/owt.dat"
        dataset = np.memmap(filename=dataset_path, dtype=np.int16, mode='r')

        # Initialize model
        language_model = TransformerLM(
            d_model=cfg["model"]["d_model"], 
            num_heads=cfg["model"]["num_heads"],
            d_ff=cfg["model"]["d_ff"],
            theta=cfg["model"]["rope_theta"], 
            vocab_size=cfg["model"]["vocab_size"], 
            context_length=cfg["model"]["context_length"], 
            num_layers=cfg["model"]["num_layers"],
            device=device
        )

        betas = (cfg["optimizer"]["beta_1"], cfg["optimizer"]["beta_2"])
        optimizer = AdamW(
            params=language_model.parameters(),
            lr = cfg["optimizer"]["lr"], 
            betas=betas, 
            weight_decay=cfg["optimizer"]["weight_decay"], 
            eps=cfg["optimizer"]["eps"],
            min_lr=cfg["optimizer"]["min_lr"], 
            total_steps=cfg["training"]["total_step_count"], 
            warmup_steps=cfg["training"]["warmup_steps"]
        )
        iteration = 0
        if resume_from:
            print(f"Loading checkpoint from {resume_from}...")
            iteration = load_checkpoint(
                src=resume_from,
                model=language_model,
                optimizer=optimizer, 
                load_optimizer=load_optimizer
            )
            print(f"Checkpoint and optimizer state loaded successfully! ")
            print(f"Beginning training from iteration {iteration}")
        else: 
            if load_optimizer:
                print("Load optimizer set to true but resume checkpoint not provided. Ignoring flag.")

        seed = cfg["training"]["seed"]
        batch_size = cfg["training"]["batch_size"]
        context_length = cfg["model"]["context_length"]
        total_training_steps = cfg["training"]["total_step_count"]
        max_norm = cfg["training"]["max_l2_norm"]
        dataloader = get_batches(
            x=dataset, 
            batch_size=batch_size,
            context_length=context_length,
            device=device,
            seed=seed
        )
        folder = cfg["model"]["checkpoint_folder"]
        save_every = cfg["training"]["save_every"]
        log_every = cfg["training"]["log_every"]

        save_location = os.path.join(folder, run.id)
        os.makedirs(save_location, exist_ok=True)
        cfg_file_path = f"{save_location}/config.yaml"
        with open(cfg_file_path, 'w') as file: 
            yaml.dump(cfg, file, sort_keys=False)

        for step in range(iteration, total_training_steps):
            LOG_STEP = (step + 1) % log_every == 0
            log_step_metrics = {}
            
            start_time = time.time()
            inputs, targets = next(dataloader)
            optimizer.zero_grad()
            preds = language_model(inputs)
            loss = cross_entropy_loss(preds, targets)
            
            loss.backward()
            assert loss.item() > 0, "Loss must be positive."

            # Calculate pre-clip norms
            if LOG_STEP:
                total_norm_sq = torch.zeros(1, device=next(language_model.parameters()).device)
                pre_clip_layer_norms = {} 
                for p_name, p in language_model.named_parameters():
                    if p.grad is not None:
                        p_norm = p.grad.detach().norm(2) 
                        pre_clip_layer_norms[f"{p_name}/pre-clip-weight-norm"] = p_norm
                        total_norm_sq += p_norm.square()
                        
                pre_clip_total_norm = total_norm_sq.sqrt().item()
                log_step_metrics["train/pre-clip norm"] = pre_clip_total_norm
                log_step_metrics["clipped"] = pre_clip_total_norm > max_norm
                log_step_metrics.update(pre_clip_layer_norms)   

            gradient_clipping(language_model.parameters(), max_norm)
            # Calculate post-clip norms
            if LOG_STEP:
                total_norm_sq = torch.zeros(1, device=next(language_model.parameters()).device)
                post_clip_layer_norms = {} 
                for p_name, p in language_model.named_parameters():
                    if p.grad is not None:
                        p_norm = p.grad.detach().norm(2) 
                        post_clip_layer_norms[f"{p_name}/post-clip-weight-norm"] = p_norm
                        total_norm_sq += p_norm.square()
                        
                post_clip_total_norm = total_norm_sq.sqrt().item()
                log_step_metrics["train/post-clip norm"] = post_clip_total_norm
                log_step_metrics.update(post_clip_layer_norms)
            optimizer.step()

            end_time = time.time()
            step_time = end_time - start_time
            tokens_in_batch = inputs.numel()

            # Safe memory check across backends
            step_memory = torch.accelerator.memory.max_memory_allocated()

            metrics = {
                "train/loss": loss.item(),
                "train/perplexity": math.exp(min(loss.item(), 20.0)),
                "optimizer/lr": optimizer.current_lr, 
                "perf/step_time": step_time,
                "perf/tokens_per_sec": tokens_in_batch / step_time, 
                "perf/step_memory": step_memory
            }

            if LOG_STEP:
                metrics.update(log_step_metrics)

            run.log(metrics)

            # Make sure save_every is a factor of total_training_steps
            if (step + 1) % save_every == 0:
                if (step + 1) == total_training_steps:
                    ckpt_path = f"{save_location}/final.pt"
                else: 
                    ckpt_path = f"{save_location}/ckpt_step_{step + 1}.pt"
                print(f"Saving model to {ckpt_path} after {step + 1} steps")
                save_checkpoint(
                    model=language_model,
                    optimizer=optimizer,
                    iteration=step+1,
                    out=ckpt_path
                )     
@app.command()
def train(
    config: str = typer.Option(..., "--config", "-c", help="Config file"),
    resume_from: Optional[str | os.PathLike | BinaryIO] = typer.Option(None, "--resume-from", "-r", help="Checkpoint from which to resume"),
    load_optimizer: bool = typer.Option(False, "--load-optimizer", "-o", help="Whether to load optimizer state when loading checkpoint")
):
    # Load configs
    with open(config, "r") as file:
        cfg = yaml.safe_load(file)
    train_fn(config=cfg, resume_from=resume_from, load_optimizer=load_optimizer)
            
if __name__ == "__main__": 
    app()