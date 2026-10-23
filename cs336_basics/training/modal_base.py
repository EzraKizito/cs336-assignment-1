import os

import modal

MODAL_APP = modal.App("cs336-sweeps")
MODAL_IMAGE = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("uv")
    .add_local_file("pyproject.toml", remote_path="/root/pyproject.toml", copy=True)
    .add_local_file("uv.lock", remote_path="/root/uv.lock", copy=True)
    .run_commands("uv pip install --system -r /root/pyproject.toml")
    .add_local_python_source("cs336_basics")
    .add_local_file("run_train.py", remote_path="/root/run_train.py")
)

MODAL_DATA_VOLUME = modal.Volume.from_name("cs336-data", create_if_missing=True)
MODAL_CKPT_VOLUME = modal.Volume.from_name("cs336-checkpoints", create_if_missing=True)


