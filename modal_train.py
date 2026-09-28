import os

import modal
import yaml

app = modal.App("cs336-sweeps")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("uv")
    .add_local_file("pyproject.toml", remote_path="/root/pyproject.toml", copy=True)
    .add_local_file("uv.lock", remote_path="/root/uv.lock", copy=True)
    .run_commands("uv pip install --system -r /root/pyproject.toml")
    .add_local_python_source("cs336_basics")
    .add_local_file("train.py", remote_path="/root/train.py")
)

data_volume = modal.Volume.from_name("cs336-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("cs336-checkpoints", create_if_missing=True)

@app.function(
        image=image,
        gpu="L40S",
        timeout=60*60*4, 
        volumes={
            "/root/data": data_volume,
            "/root/checkpoints": ckpt_volume
        },
        secrets=[modal.Secret.from_name("wandb-secret")]
)
def run_agent(sweep_id: str, count: int = 1):
    import wandb
    from train import train_fn

    data_volume.reload()
    os.chdir("/root")

    wandb.agent(
        sweep_id=sweep_id, 
        function=train_fn, 
        count=count, 
        project="cs336",
        entity="ezra-kizito"
    )

    ckpt_volume.commit()

    wandb.finish()

@app.local_entrypoint()
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