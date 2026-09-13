"""Run one Surg-SegFormer dataset/task without mixing label spaces."""

import argparse
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from dataset_loader import DatasetLoader
from evaluation import Evaluation
from model import DualSegFormer
from trainer import Trainer


def load_config(path: str) -> dict:
    config_path = Path(path).resolve()
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    base = config_path.parent
    relative_paths = [
        (config["training"], "checkpoint_dir"),
        (config["data"]["tasks"][config["data"]["active_task"]], "root"),
        (config["data"]["tasks"][config["data"]["active_task"]], "masks_root"),
        (config["video"], "input"),
        (config["video"], "output"),
    ]
    for section, key in relative_paths:
        value = Path(section[key])
        if not value.is_absolute():
            section[key] = str((base / value).resolve())
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(config: dict, task_name: str) -> dict:
    try:
        task_config = config["data"]["tasks"][task_name]
    except KeyError as error:
        raise ValueError(f"unknown task: {task_name}") from error

    set_seed(config["training"]["seed"])
    train_loader, val_loader, test_loader = DatasetLoader(
        config, task_name
    ).prepare_datasets()
    model = DualSegFormer(config, task_config)
    Trainer(model, train_loader, val_loader, config, task_config).train()
    return Evaluation(model, test_loader, config, task_config).evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--task",
        help="dataset task (default: data.active_task)",
    )
    arguments = parser.parse_args()
    experiment_config = load_config(arguments.config)
    selected_task = arguments.task or experiment_config["data"]["active_task"]
    run(experiment_config, selected_task)
