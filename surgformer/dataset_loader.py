"""SAR-RARP50 loader using feature 001's split and the exact dense source masks."""

import glob
import os
import random
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _seed_worker(_: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class SurgicalDataset(Dataset):
    def __init__(self, file_pairs: list[tuple[str, str]], transform: A.Compose,
                 label_map: dict[int, int], mean: list[float], std: list[float],
                 ignore_index: int) -> None:
        self.file_pairs = file_pairs
        self.transform = transform
        self.label_map = {int(source): target for source, target in label_map.items()}
        self.mean = np.asarray(mean, dtype=np.float32)[:, None, None]
        self.std = np.asarray(std, dtype=np.float32)[:, None, None]
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.file_pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_path, mask_path = self.file_pairs[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if image is None or mask_image is None:
            raise ValueError(f"failed to read image/mask pair: {image_path}, {mask_path}")
        if mask_image.ndim == 3:
            if not np.array_equal(mask_image[..., 0], mask_image[..., 1]) or not np.array_equal(
                mask_image[..., 0], mask_image[..., 2]
            ):
                raise ValueError(f"mask channels disagree: {mask_path}")
            mask = mask_image[..., 0]
        else:
            mask = mask_image

        transformed = self.transform(
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mask=mask
        )
        image = transformed["image"].astype(np.float32).transpose(2, 0, 1) / 255.0
        image = (image - self.mean) / self.std
        raw_mask = transformed["mask"]
        unknown = set(np.unique(raw_mask)) - set(self.label_map) - {self.ignore_index}
        if unknown:
            raise ValueError(f"unmapped mask values {sorted(unknown)} in {mask_path}")
        remapped = np.full(raw_mask.shape, self.ignore_index, dtype=np.int64)
        for source, target in self.label_map.items():
            remapped[raw_mask == source] = target

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(remapped),
        }


class DatasetLoader:
    """Load SAR-RARP50 without changing the source videos or prepared split."""

    def __init__(self, config: dict, task_name: Optional[str] = None) -> None:
        self.config = config
        self.data_config = config["data"]
        self.task_name = task_name or self.data_config["active_task"]
        try:
            self.task_config = self.data_config["tasks"][self.task_name]
        except KeyError as error:
            raise ValueError(f"unknown task: {self.task_name}") from error
        self.seed = config["training"]["seed"]

    def _transform(self, training: bool) -> A.Compose:
        augmentation = self.data_config["augmentation"]
        width, height = self.data_config["image_size"]
        crop_width, crop_height = self.data_config["crop_size"]
        transforms: list[A.BasicTransform] = []
        if training and augmentation["flip"]:
            transforms.append(A.HorizontalFlip(p=0.5))
        if training and augmentation["crop"]:
            transforms.append(A.RandomCrop(crop_height, crop_width, p=0.5))
        if training and augmentation["rotation"]:
            transforms.append(A.Rotate(limit=15, p=0.5))
        transforms.append(A.Resize(height, width))
        return A.Compose(transforms)

    def _file_pairs(self, split: str) -> list[tuple[str, str]]:
        layout = self.data_config["split_layout"]
        values = {
            "root": self.task_config["root"],
            "split": self.data_config["splits"][split],
        }
        images_dir = layout.format(kind="images", **values)
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(
                f"{self.task_name}/{split} must follow configured split_layout: {images_dir}"
            )

        image_paths = sorted(
            path
            for extension in ("png", "jpg", "jpeg", "bmp", "tif", "tiff")
            for path in glob.glob(os.path.join(images_dir, f"*.{extension}"))
        )
        pairs = []
        for image_path in image_paths:
            stem = os.path.splitext(os.path.basename(image_path))[0]
            try:
                video_id, frame_id = stem.rsplit("_", 1)
            except ValueError as error:
                raise ValueError(f"unexpected SAR-RARP50 frame name: {stem}") from error
            mask_path = os.path.join(
                self.task_config["masks_root"], video_id, "segmentation", f"{frame_id}.png"
            )
            if not os.path.isfile(mask_path):
                raise FileNotFoundError(f"mask missing for {image_path}: {mask_path}")
            pairs.append((image_path, mask_path))
        if not pairs:
            raise ValueError(f"no {self.task_name}/{split} image-mask pairs found")
        return pairs

    def prepare_datasets(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        normalization = self.data_config["normalization"]
        common = {
            "label_map": self.task_config["label_map"],
            "mean": normalization["mean"],
            "std": normalization["std"],
            "ignore_index": self.config["loss"]["ignore_index"],
        }
        datasets = {
            split: SurgicalDataset(
                self._file_pairs(split), self._transform(split == "train"), **common
            )
            for split in ("train", "val", "test")
        }

        loader_args = {
            "batch_size": self.config["training"]["batch_size"],
            "num_workers": 4,
            "pin_memory": torch.cuda.is_available(),
            "worker_init_fn": _seed_worker,
        }
        loaders = []
        for offset, split in enumerate(("train", "val", "test")):
            generator = torch.Generator().manual_seed(self.seed + offset)
            loaders.append(DataLoader(
                datasets[split],
                shuffle=split == "train",
                generator=generator,
                **loader_args,
            ))
        return tuple(loaders)
