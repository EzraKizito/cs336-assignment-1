from pathlib import Path
from typing import Optional

import typer
import yaml
from dotenv import load_dotenv

import wandb
from cs336_basics.training.modal_base import (
    MODAL_CKPT_VOLUME,
    MODAL_DATA_VOLUME,
    MODAL_IMAGE,
)
from cs336_basics.training.train import train_fn

load_dotenv()

CLI_APP = typer.Typer(help="CS336 Training Framework")

@CLI_APP.command()
def train(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True, help="Path to YAML config file"),
    resume_from: Optional[Path] = typer.Option(None, "--resume-from", "-r", help="Checkpoint from which to resume"),
    remote: bool = typer.Option(False, "--remote", "-m", help="Run remotely on Modal"),
    load_optimizer: bool = typer.Option(False, "--load-optimizer", "-o", help="Whether to load optimizer state when loading checkpoint"),
):
    # 1. Parse config locally
    with open(config, "r") as file:
        cfg = yaml.safe_load(file)

    if remote:
        # Import remote function & app dynamically to keep local CLI snappy
        from cs336_basics.training.modal_base import MODAL_APP
        from cs336_basics.training.train import train_remote

        # Resolve checkpoint name/path relative to the remote volume mount
        # e.g., if user passes "checkpoints/step_100.pt", container looks for "/root/checkpoints/step_100.pt"
        remote_resume_path = None
        if resume_from:
            remote_resume_path = (Path("/root/checkpoints") / resume_from.name)

        with MODAL_APP.run():
            train_remote.remote(
                config=cfg,
                resume_from=remote_resume_path,
                load_optimizer=load_optimizer,
            )
    else:
        train_fn(
            config=cfg,
            resume_from=str(resume_from) if resume_from else None,
            load_optimizer=load_optimizer,
        )

@CLI_APP.command()
def sweep(
    sweep_config: str = typer.Option(..., "--sweep-config", "-s", help="Sweep YAML config"),
    count: int = typer.Option(10, "--count", "-n", help="Number of runs to execute")
):
    with open(sweep_config, "r") as file:
        sweep_cfg = yaml.safe_load(file)

    sweep_id = wandb.sweep(sweep=sweep_cfg, project="cs336")
    try:
        wandb.agent(sweep_id, function=train_fn, count=count)
    finally:
        wandb.finish()

if __name__ == "__main__":
    CLI_APP()