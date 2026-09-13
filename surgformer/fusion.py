"""Inference-only priority fusion from Surg-SegFormer Eq. 1."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def _morph(labels: torch.Tensor, num_classes: int, operations: Sequence[str],
           kernel_size: int, iterations: int) -> torch.Tensor:
    if kernel_size <= 1 or iterations <= 0:
        return labels
    if kernel_size % 2 == 0:
        raise ValueError("morphology kernel_size must be odd")

    masks = F.one_hot(labels, num_classes).permute(0, 3, 1, 2).float()
    pad = kernel_size // 2
    dilate = lambda x: F.max_pool2d(x, kernel_size, stride=1, padding=pad)
    erode = lambda x: -dilate(-x)
    for _ in range(iterations):
        for operation in operations:
            if operation == "closing":
                masks = erode(dilate(masks))
            elif operation == "opening":
                masks = dilate(erode(masks))
            else:
                raise ValueError(f"unsupported morphology operation: {operation}")
    return masks.argmax(dim=1)


def fuse_predictions(coarse_logits: torch.Tensor, fine_logits: torch.Tensor,
                     config: dict) -> torch.Tensor:
    """Apply Eq. 1, then configured morphology only where both branches overlap."""
    if coarse_logits.shape != fine_logits.shape:
        raise ValueError("fusion branches must have identical [B,C,H,W] shapes")
    if config["method"] != "priority_weighted_conditional_OR":
        raise ValueError(f"unsupported fusion method: {config['method']}")

    coarse_prob, coarse_mask = coarse_logits.softmax(1).max(1)
    fine_prob, fine_mask = fine_logits.softmax(1).max(1)
    background = config["background_class"]
    fused = torch.where(
        ((fine_mask != background) & (fine_prob > coarse_prob))
        | (coarse_mask == background),
        fine_mask,
        coarse_mask,
    )

    overlap = (coarse_mask != background) & (fine_mask != background)
    morphology = config["morphology"]
    refined = _morph(
        fused,
        coarse_logits.shape[1],
        morphology["operations"],
        morphology["kernel_size"],
        morphology["iterations"],
    )
    return torch.where(overlap, refined, fused)
