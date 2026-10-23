import os

import modal
import yaml

from cs336_basics.training.modal_base import (
    MODAL_APP,
    MODAL_CKPT_VOLUME,
    MODAL_DATA_VOLUME,
    MODAL_IMAGE,
)


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
def run_agent(sweep_id: str, count: int = 1):
    import wandb
    from cs336_basics.training.train import train_fn

    MODAL_DATA_VOLUME.reload()
    os.chdir("/root")

    try:
        wandb.agent(
            sweep_id=sweep_id, 
            function=train_fn, 
            count=count, 
            project="cs336",
            entity="ezra-kizito"
        )
    finally:
        MODAL_CKPT_VOLUME.commit()
        wandb.finish()

@MODAL_APP.local_entrypoint()
def launch_sweep(
    sweep_config: str, 
    runs_per_gpu: int,
    num_parallel_gpus: int = 1,
): 
    import wandb 

    with open(sweep_config, 'r') as file: 
        sweep_cfg = yaml.safe_load(file)

    sweep_id = wandb.sweep(sweep=sweep_cfg, project="cs336")
    print(f"Spinning up {num_parallel_gpus} parallel GPU workers")

    args = [(sweep_id, runs_per_gpu)] * num_parallel_gpus
    list(run_agent.starmap(args))