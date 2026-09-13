"""Synthetic checks; no checkpoints or model downloads needed."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import albumentations as A
import cv2
import numpy as np
import torch
from torch import nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerModel
import yaml

from dataset_loader import DatasetLoader, SurgicalDataset
from evaluation import Evaluation
from fusion import _morph, fuse_predictions
from model import DenseSkipDecoder, DualSegFormer
from trainer import Trainer
import video_inference


def _tiny_segformer(num_labels: int) -> SegformerConfig:
    return SegformerConfig(
        num_labels=num_labels,
        depths=[1, 1, 1, 1],
        hidden_sizes=[8, 16, 32, 64],
        num_attention_heads=[1, 1, 2, 4],
        sr_ratios=[8, 4, 2, 1],
        decoder_hidden_size=16,
    )


def main() -> None:
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    task_config = config["data"]["tasks"]["sar_rarp50"]
    assert task_config["num_classes"] == 10
    assert task_config["class_groups"] == {
        "coarse": [0, 1, 2, 3],
        "fine": [0, 4, 5, 6, 7, 8, 9],
    }
    train_loader, val_loader, test_loader = DatasetLoader(config).prepare_datasets()
    assert [len(loader.dataset) for loader in (train_loader, val_loader, test_loader)] == [326, 81, 132]
    assert test_loader.dataset[0]["mask"].shape == (512, 896)

    decoder = DenseSkipDecoder([4, 8, 16, 32], 2, 3, 4, "dense_deep_to_shallow")
    features = [
        torch.randn(1, 4, 8, 8),
        torch.randn(1, 8, 4, 4),
        torch.randn(1, 16, 2, 2),
        torch.randn(1, 32, 1, 1),
    ]
    assert decoder(features).shape == (1, 3, 8, 8)

    coarse = torch.tensor([[[[3.0]], [[0.0]], [[0.0]]]])
    fine = torch.tensor([[[[0.0]], [[4.0]], [[0.0]]]])
    fusion_config = {
        "method": "priority_weighted_conditional_OR",
        "background_class": 0,
        "morphology": {"operations": [], "kernel_size": 1, "iterations": 0},
    }
    assert fuse_predictions(coarse, fine, fusion_config).item() == 1
    coarse_foreground = torch.tensor([[[[0.0]], [[0.0]], [[4.0]]]])
    fine_background = torch.tensor([[[[6.0]], [[0.0]], [[0.0]]]])
    assert fuse_predictions(coarse_foreground, fine_background, fusion_config).item() == 2
    speck = torch.zeros(1, 5, 5, dtype=torch.long)
    speck[:, 2, 2] = 1
    morphology = config["fusion"]["morphology"]
    assert not _morph(
        speck,
        2,
        morphology["operations"],
        morphology["kernel_size"],
        morphology["iterations"],
    ).any()

    tiny_config = deepcopy(config)
    tiny_config["model"]["fine"]["decoder"]["projection_channels"] = 2
    tiny_config["fusion"] = fusion_config
    task = {"num_classes": 3, "branches": ["coarse", "fine"]}
    coarse_model = SegformerForSemanticSegmentation(_tiny_segformer(3))
    fine_model = SegformerModel(_tiny_segformer(3))
    with (
        patch.object(
            SegformerForSemanticSegmentation,
            "from_pretrained",
            return_value=coarse_model,
        ),
        patch.object(SegformerModel, "from_pretrained", return_value=fine_model),
    ):
        model = DualSegFormer(tiny_config, task)
    assert model(torch.randn(1, 3, 64, 112)).shape == (1, 64, 112)

    with (
        patch.object(SegformerForSemanticSegmentation, "from_pretrained") as coarse_load,
        patch.object(SegformerModel, "from_pretrained", return_value=fine_model) as fine_load,
    ):
        single_branch = DualSegFormer(
            tiny_config, {"num_classes": 3, "branches": ["fine"]}
        )
    coarse_load.assert_not_called()
    fine_load.assert_called_once()
    assert single_branch.coarse is None

    with TemporaryDirectory() as directory:
        image_path = str(Path(directory, "frame.png"))
        mask_path = str(Path(directory, "mask.png"))
        cv2.imwrite(image_path, np.full((2, 2, 3), 127, dtype=np.uint8))
        cv2.imwrite(mask_path, np.full((2, 2, 3), 1, dtype=np.uint8))
        dataset = SurgicalDataset(
            [(image_path, mask_path)],
            A.Compose([A.Resize(2, 2)]),
            {0: 0, 1: 1, 2: 2},
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
            255,
        )
        sample = dataset[0]
        assert sample["image"].shape == (3, 2, 2)
        assert set(sample["mask"].flatten().tolist()) == {1}

    class BackgroundModel(nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return torch.zeros(images.shape[0], *images.shape[-2:], dtype=torch.long)

    evaluator = Evaluation(
        BackgroundModel(),
        [{
            "image": torch.zeros(1, 3, 2, 2),
            "mask": torch.tensor([[[0, 0], [0, 2]]]),
        }],
        {"loss": {"ignore_index": 255}},
        {
            "num_classes": 3,
            "class_names": ["seen", "absent-1", "absent-2"],
            "untrainable_class_ids": [2],
        },
    )
    metrics = evaluator.evaluate()
    assert metrics["mIoU_all_classes"] == 0.375
    assert metrics["mIoU_learnable_classes"] == 0.75
    assert metrics["per_class"]["absent-1"]["iou"] is None
    assert metrics["per_class"]["absent-2"]["test_frame_count"] == 1
    assert metrics["per_class"]["absent-2"]["test_pixel_share"] == 0.25
    assert "untrainable_by_split" in metrics["per_class"]["absent-2"]["flags"]

    class TinyBranches(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.coarse = nn.Conv2d(3, 2, 1)
            self.fine_encoder = nn.Conv2d(3, 2, 1)
            self.fine_decoder = nn.Identity()

        def parameters_for(self, branch: str):
            module = self.coarse if branch == "coarse" else self.fine_encoder
            return module.parameters()

        def branch_logits(self, images: torch.Tensor, branch: str) -> torch.Tensor:
            module = self.coarse if branch == "coarse" else self.fine_encoder
            return module(images)

    with TemporaryDirectory() as directory:
        train_config = deepcopy(config)
        train_config["training"]["epochs"] = 1
        train_config["training"]["checkpoint_dir"] = directory
        task = {
            "branches": ["coarse", "fine"],
            "class_groups": {"coarse": [0], "fine": [0, 1]},
        }
        masks = torch.tensor([[[0, 0], [1, 1]]])
        batches = [{"image": torch.randn(1, 3, 2, 2), "mask": masks}]
        trainer = Trainer(TinyBranches(), batches, batches, train_config, task)
        assert not trainer._targets_for(masks, "coarse").any()
        trainer.train()
        assert {path.name for path in Path(directory).glob("*.pth")} == {
            "coarse_best.pth", "coarse_last.pth",
            "fine_best.pth", "fine_last.pth",
        }

    with TemporaryDirectory() as directory:
        input_path = str(Path(directory, "input.avi"))
        output_path = str(Path(directory, "output.mp4"))
        writer = cv2.VideoWriter(
            input_path, cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 64)
        )
        assert writer.isOpened()
        writer.write(np.full((64, 64, 3), 80, dtype=np.uint8))
        writer.release()
        labels = np.zeros((64, 64), dtype=np.uint8)
        labels[24:40, 24:40] = 1
        with (
            patch.object(video_inference, "_load_model", return_value=object()),
            patch.object(video_inference, "_predict", return_value=labels),
        ):
            video_inference.render_video(
                config, input_path, output_path, "coarse.pth", "fine.pth", "cpu"
            )
        assert Path(output_path).stat().st_size > 0


if __name__ == "__main__":
    main()
    print("smoke test passed")
