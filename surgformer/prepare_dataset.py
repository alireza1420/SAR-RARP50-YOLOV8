"""Prepare extracted SAR-RARP50 videos for Surg-SegFormer."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import shutil

import cv2
import numpy as np


VIDEO_NAME = re.compile(r"video_(\d{2})(?:_[12])?")


def _videos(source: Path) -> dict[str, tuple[int, Path, Path]]:
    videos = {}
    for video_path in source.rglob("video_left.avi"):
        directory = video_path.parent
        match = VIDEO_NAME.fullmatch(directory.name)
        masks = directory / "segmentation"
        if not match or not masks.is_dir():
            continue
        if directory.name in videos:
            raise ValueError(f"duplicate video directory: {directory.name}")
        videos[directory.name] = (int(match.group(1)), video_path, masks)
    if not videos:
        raise ValueError(f"no extracted SAR-RARP50 video directories found under {source}")
    return videos


def _split_operations(operation_ids: set[int], val_operations: int,
                      seed: int) -> tuple[set[int], set[int], set[int]]:
    development = sorted(operation_id for operation_id in operation_ids if operation_id <= 40)
    test = {operation_id for operation_id in operation_ids if operation_id >= 41}
    if not test:
        raise ValueError("official test operations 41-50 are missing")
    if not 0 < val_operations < len(development):
        raise ValueError("val_operations must leave at least one development operation for training")
    validation = set(random.Random(seed).sample(development, val_operations))
    return set(development) - validation, validation, test


def _mask_values(path: Path, num_classes: int) -> tuple[np.ndarray, int]:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"failed to read mask: {path}")
    if mask.ndim == 3:
        if not np.array_equal(mask[..., 0], mask[..., 1]) or not np.array_equal(
            mask[..., 0], mask[..., 2]
        ):
            raise ValueError(f"mask channels disagree: {path}")
        mask = mask[..., 0]
    unknown = set(np.unique(mask)) - set(range(num_classes)) - {255}
    if unknown:
        raise ValueError(f"unexpected mask values {sorted(unknown)} in {path}")
    valid = mask != 255
    return np.bincount(mask[valid], minlength=num_classes), int(valid.sum())


def prepare(source: Path, output: Path, val_operations: int = 4,
            seed: int = 42, num_classes: int = 10) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")

    videos = _videos(source)
    train_ops, val_ops, test_ops = _split_operations(
        {operation_id for operation_id, _, _ in videos.values()}, val_operations, seed
    )
    operation_splits = {operation_id: "train" for operation_id in train_ops}
    operation_splits.update({operation_id: "val" for operation_id in val_ops})
    operation_splits.update({operation_id: "test" for operation_id in test_ops})

    frame_counts = Counter()
    pixel_counts = {split: np.zeros(num_classes, dtype=np.int64)
                    for split in ("train", "val", "test")}
    valid_pixels = Counter()

    for video_name, (operation_id, video_path, masks_dir) in sorted(videos.items()):
        split = operation_splits[operation_id]
        image_output = output / "images" / split
        mask_output = output / "masks" / video_name / "segmentation"
        image_output.mkdir(parents=True, exist_ok=True)
        mask_output.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"failed to open video: {video_path}")
        try:
            masks = sorted(masks_dir.glob("*.png"), key=lambda path: int(path.stem))
            if not masks:
                raise ValueError(f"no segmentation masks found: {masks_dir}")
            for mask_path in masks:
                frame_id = int(mask_path.stem)
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                success, frame = capture.read()
                if not success:
                    raise ValueError(f"failed to read frame {frame_id} from {video_path}")
                image_path = image_output / f"{video_name}_{mask_path.stem}.jpg"
                if not cv2.imwrite(str(image_path), frame):
                    raise ValueError(f"failed to write image: {image_path}")
                shutil.copy2(mask_path, mask_output / mask_path.name)
                counts, valid = _mask_values(mask_path, num_classes)
                pixel_counts[split] += counts
                valid_pixels[split] += valid
                frame_counts[split] += 1
        finally:
            capture.release()
        print(f"{video_name}: {len(masks)} frames -> {split}")

    manifest = {
        "seed": seed,
        "operations": {
            "train": sorted(train_ops),
            "val": sorted(val_ops),
            "test": sorted(test_ops),
        },
        "frames": {split: frame_counts[split] for split in ("train", "val", "test")},
        "class_pixel_counts": {
            split: pixel_counts[split].tolist() for split in ("train", "val", "test")
        },
        "class_pixel_share": {
            split: (pixel_counts[split] / valid_pixels[split]).tolist()
            for split in ("train", "val", "test")
        },
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["frames"], indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="directory containing extracted video_* folders")
    parser.add_argument("output", type=Path, help="new prepared dataset directory")
    parser.add_argument("--val-operations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    prepare(arguments.source, arguments.output, arguments.val_operations, arguments.seed)
