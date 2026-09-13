"""Coarse/fine SegFormer branches and the Figure-2-inspired dense decoder."""

from collections.abc import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerModel

from fusion import fuse_predictions


_CHECKPOINTS = {"B2": "nvidia/mit-b2", "B5": "nvidia/mit-b5"}


class DenseSkipDecoder(nn.Module):
    """Project four encoder stages, decode deep-to-shallow with dense skips, and fuse."""

    def __init__(self, in_channels: Sequence[int], channels: int, num_classes: int,
                 dense_layers: int, skip_topology: str) -> None:
        super().__init__()
        if dense_layers != len(in_channels):
            raise ValueError("dense_layers must match the four encoder stages")
        if skip_topology != "dense_deep_to_shallow":
            raise ValueError(f"unsupported skip topology: {skip_topology}")

        self.projections = nn.ModuleList(
            nn.Conv2d(source, channels, 1) for source in in_channels
        )
        self.dense_layers = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels * (index + 1), channels, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            for index in range(dense_layers)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * dense_layers, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_classes, 1),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        projected = [
            F.interpolate(layer(feature), target_size, mode="bilinear", align_corners=False)
            for layer, feature in zip(self.projections, features)
        ]

        decoded: list[torch.Tensor] = []
        for feature, layer in zip(reversed(projected), self.dense_layers):
            decoded.append(layer(torch.cat([feature, *decoded], dim=1)))
        return self.fusion(torch.cat(decoded, dim=1))


class DualSegFormer(nn.Module):
    """Independent B2 coarse and B5 fine branches; fusion is inference-only."""

    def __init__(self, config: dict, task_config: dict) -> None:
        super().__init__()
        model_config = config["model"]
        self.fusion_config = config["fusion"]
        self.branches = tuple(task_config["branches"])
        self.num_classes = task_config["num_classes"]

        self.coarse = None
        if "coarse" in self.branches:
            coarse_variant = model_config["coarse"]["variant"].upper()
            if coarse_variant not in _CHECKPOINTS:
                raise ValueError(f"unsupported SegFormer variant: {coarse_variant}")
            self.coarse = SegformerForSemanticSegmentation.from_pretrained(
                _CHECKPOINTS[coarse_variant],
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True,
            )

        self.fine_encoder = None
        self.fine_decoder = None
        if "fine" in self.branches:
            fine_variant = model_config["fine"]["variant"].upper()
            if fine_variant not in _CHECKPOINTS:
                raise ValueError(f"unsupported SegFormer variant: {fine_variant}")
            self.fine_encoder = SegformerModel.from_pretrained(_CHECKPOINTS[fine_variant])
            decoder_config = model_config["fine"]["decoder"]
            self.fine_decoder = DenseSkipDecoder(
                self.fine_encoder.config.hidden_sizes,
                decoder_config["projection_channels"],
                self.num_classes,
                decoder_config["dense_layers"],
                decoder_config["skip_topology"],
            )

    def parameters_for(self, branch: str) -> Iterable[nn.Parameter]:
        if branch == "coarse":
            if self.coarse is None:
                raise ValueError("coarse is disabled for this task")
            return self.coarse.parameters()
        if branch == "fine":
            if self.fine_encoder is None or self.fine_decoder is None:
                raise ValueError("fine is disabled for this task")
            return (
                parameter
                for module in (self.fine_encoder, self.fine_decoder)
                for parameter in module.parameters()
            )
        raise ValueError(f"unknown branch: {branch}")

    def branch_logits(self, images: torch.Tensor, branch: str) -> torch.Tensor:
        size = images.shape[-2:]
        if branch == "coarse":
            if self.coarse is None:
                raise ValueError("coarse is disabled for this task")
            logits = self.coarse(pixel_values=images).logits
        elif branch == "fine":
            if self.fine_encoder is None or self.fine_decoder is None:
                raise ValueError("fine is disabled for this task")
            hidden_states = self.fine_encoder(
                pixel_values=images, output_hidden_states=True
            ).hidden_states
            logits = self.fine_decoder(hidden_states)
        else:
            raise ValueError(f"unknown branch: {branch}")
        return F.interpolate(logits, size, mode="bilinear", align_corners=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.branches == ("fine",):
            return self.branch_logits(images, "fine").argmax(dim=1)
        if self.branches == ("coarse",):
            return self.branch_logits(images, "coarse").argmax(dim=1)
        if set(self.branches) != {"coarse", "fine"}:
            raise ValueError(f"unsupported branches: {self.branches}")

        coarse_logits = self.branch_logits(images, "coarse")
        fine_logits = self.branch_logits(images, "fine")
        return fuse_predictions(coarse_logits, fine_logits, self.fusion_config)
