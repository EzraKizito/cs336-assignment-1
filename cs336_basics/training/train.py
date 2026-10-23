
import math
import os
import signal
import time
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, cast

import modal
import numpy as np
import torch
import typer
import yaml
from dotenv import load_dotenv

import wandb
from cs336_basics.training.loop import load_checkpoint, save_checkpoint
from cs336_basics.training.loss import cross_entropy_loss
from cs336_basics.training.modal_base import (
    MODAL_APP,
    MODAL_CKPT_VOLUME,
    MODAL_DATA_VOLUME,
    MODAL_IMAGE,
)
from cs336_basics.training.optimizer import (
    AdamW,
    gradient_clipping,
)
from cs336_basics.training.train_utils import get_allocated_memory
from cs336_basics.transformer.transformer import TransformerLM

load_dotenv()

class ProactiveTimeoutException(Exception):
    pass 

def alarm_handler(signum, frame):
    raise ProactiveTimeoutException("Approaching Modal timeout threshold. Initiating safe save!")

app = typer.Typer(help="CS336 Training Framework")

# Modal checkpoint volume
ckpt_volume = modal.Volume.from_name("cs336-checkpoints", create_if_missing=True)

def get_batches(x: np.ndarray, batch_size: int, context_length: int, device: torch.device, dtype: Optional[torch.dtype], seed: int = 42):
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
    # Set up such that best model is saved around 45 minutes in
    # We use an alarm to interrupt training with 2 minutes left to allow for 
    # sufficient time to save models
    interruptible = True
    save_location = ""
    current_step = 0
    language_model = None
    optimizer = None
    total_training_steps = 0

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(2580)
    try:  
        # Set device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.set_float32_matmul_precision('high')
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
            base_name = cfg["name"]
            lr = cfg["optimizer"]["lr"]
            batch_size = cfg["training"]["batch_size"]
            weight_decay = cfg["optimizer"]["weight_decay"]
            sweep = False
            if sweep:
                # run.name = f"{base_name}_LR-{lr}"
                # run.name = f"{base_name}_Weight-Decay-{weight_decay}"
                run.name = f"{base_name}_Batch Size-{batch_size}"
            else:
                run.name = f"{base_name}"
            dataset_path = cfg["data"]["dataset_path"]
            dataset = np.memmap(filename=dataset_path, dtype=np.int16, mode='r')
            eval_dataset_path = cfg["data"].get("eval_dataset_path", dataset_path)
            eval_dataset = np.memmap(filename=eval_dataset_path, dtype=np.int16, mode='r')

            lm_dtype = torch.bfloat16
            # Initialize model
            language_model = TransformerLM(
                d_model=cfg["model"]["d_model"], 
                num_heads=cfg["model"]["num_heads"],
                d_ff=cfg["model"]["d_ff"],
                theta=cfg["model"]["rope_theta"], 
                vocab_size=cfg["model"]["vocab_size"], 
                context_length=cfg["model"]["context_length"], 
                num_layers=cfg["model"]["num_layers"],
                device=device,
                dtype=lm_dtype
            )

            num_params = sum(p.numel() for p in language_model.parameters() if p.requires_grad)
            print(f"Trainable params: {num_params}")

            # Compilation to improve performance
            if device == torch.device("mps"):
                language_model = torch.compile(language_model, backend="aot_eager")
            else: 
                language_model = torch.compile(language_model)

            language_model = cast(TransformerLM, language_model)

            # Initialize optimizer
            betas = (cfg["optimizer"]["beta_1"], cfg["optimizer"]["beta_2"])
            optimizer = AdamW(
                params=language_model.parameters(),
                lr = cfg["optimizer"]["lr"], 
                betas=betas, 
                weight_decay=cfg["optimizer"]["weight_decay"],
                schedule_lr=True,
                eps=cfg["optimizer"]["eps"],
                min_lr=cfg["optimizer"]["min_lr"], 
                total_steps=cfg["training"]["total_step_count"], 
                warmup_steps=cfg["training"]["warmup_steps"]
            )

            # Resume logic if needed
            iteration = 0
            if resume_from is not None:
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
                    print("[WARN] Load optimizer set to true but resume checkpoint not provided. Ignoring flag.")

            # Load data
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
                seed=seed,
                dtype=lm_dtype
            )
            # Set up saving logic
            folder = cfg["checkpoint_folder"]
            save_every = cfg["training"]["save_every"]
            log_every = cfg["training"]["log_every"]
            eval_every = cfg["training"]["eval_every"]

            save_location = os.path.join(folder, run.id)
            os.makedirs(save_location, exist_ok=True)
            cfg_file_path = f"{save_location}/config.yaml"
            with open(cfg_file_path, 'w') as file: 
                yaml.dump(cfg, file, sort_keys=False)

            # Each training step
            for step in range(iteration, total_training_steps):
                current_step = step
                LOG_STEP = log_every > 0 and (step + 1) % log_every == 0
                EVAL_STEP = eval_every > 0 and ((step + 1) % eval_every == 0 or step == iteration or step == total_training_steps - 1)
                SAVE_STEP = save_every > 0 and (step + 1) % save_every == 0
                log_step_metrics = {}
                eval_metrics = {}

                start_time = time.time()
                inputs, targets = next(dataloader)
                if EVAL_STEP:
                    val_loss = run_eval(
                        model=language_model,
                        dataset=eval_dataset,
                        batch_size=batch_size,
                        context_length=context_length,
                        device=device,
                        dtype=lm_dtype,
                        seed=seed
                    )
                    eval_metrics["val/loss"] = val_loss
                
                optimizer.zero_grad()
                with torch.autocast(device_type=inputs.device.type, dtype=torch.bfloat16):
                    preds = language_model(inputs)
                    loss = cross_entropy_loss(preds, targets)
                
                loss.backward()
                if not torch.all(torch.isfinite(loss)):
                    print("Run Diverged, Moving on to next config")
                    break
                ### CALCULATING & LOGGING METRICS ########
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
                step_memory = get_allocated_memory(device)

                metrics = {
                    "train/loss": loss.item(),
                    "train/perplexity": math.exp(min(loss.item(), 20.0)),
                    "optimizer/lr": optimizer.current_lr, 
                    "perf/step_time (ms)": step_time * 1000.0,
                    "perf/tokens_per_sec": tokens_in_batch / step_time, 
                    "perf/step_memory": step_memory
                }

                if LOG_STEP:
                    metrics.update(log_step_metrics)
                if EVAL_STEP:
                    metrics.update(eval_metrics)

                run.log(metrics)

                # Make sure save_every is a factor of total_training_steps
                if SAVE_STEP:
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

            signal.alarm(0)
            interruptible = False
            print("Training completed successfully")
                
    except (ProactiveTimeoutException, modal.exception.TimeoutError) as exc:
        print(f"Stopping run early: {repr(exc)}")
    finally:
        if interruptible and language_model is not None and optimizer is not None:
            language_model = cast(TransformerLM, language_model)
            ckpt_path = f"{save_location}/final_interrupted_step_{current_step}.pt"
            save_checkpoint(
                model=language_model,
                optimizer=optimizer,
                iteration=current_step+1,
                out=ckpt_path
            )
            if not modal.is_local():
                ckpt_volume.commit()
            print(f"Training interrupted at step {current_step}/ {total_training_steps} :(, but model still saved successfully!")

@torch.no_grad()
def run_eval(
    model: torch.nn.Module, 
    dataset, 
    batch_size: int,
    context_length: int, 
    device,
    dtype, 
    seed: int,
    eval_iters: int = 50
):
    model.eval()
    eval_dataloader = get_batches(x=dataset, batch_size=batch_size, context_length=context_length, device=device, dtype=dtype, seed=seed)
    losses = torch.zeros(eval_iters, device=device)
    for k in range(eval_iters):
        inputs, targets = next(eval_dataloader)
        with torch.autocast(device_type=inputs.device.type, dtype=torch.bfloat16):
            preds = model(inputs)
            loss = cross_entropy_loss(preds, targets)
        losses[k] = loss.item()

    model.train()
    return losses.mean()

@MODAL_APP.function(
        image=MODAL_IMAGE,
        gpu="B200",
        timeout=60*45, 
        volumes={
            "/root/data": MODAL_DATA_VOLUME,
            "/root/checkpoints": MODAL_CKPT_VOLUME
        },
        secrets=[modal.Secret.from_name("wandb-secret")]
)
def train_remote(
    config: dict,
    resume_from: Optional[Path],
    load_optimizer: bool = False,
):
    train_fn(config=config, resume_from=resume_from, load_optimizer=load_optimizer)