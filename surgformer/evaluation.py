"""Per-class and mean IoU/Dice evaluation for one configured task."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class Evaluation:
    def __init__(self, model: nn.Module, test_data: DataLoader, config: dict,
                 task_config: dict) -> None:
        self.model = model
        self.test_data = test_data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.num_classes = task_config["num_classes"]
        self.class_names = task_config["class_names"]
        self.untrainable_class_ids = set(task_config.get("untrainable_class_ids", []))
        self.ignore_index = config["loss"]["ignore_index"]

    def evaluate(self) -> dict:
        self.model.eval()
        confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        test_frame_count = np.zeros(self.num_classes, dtype=np.int64)
        total_valid_pixels = 0
        with torch.no_grad():
            for batch in self.test_data:
                labels = batch["mask"].numpy()
                predictions = self.model(batch["image"].to(self.device)).cpu().numpy()
                valid = labels != self.ignore_index
                total_valid_pixels += int(valid.sum())
                for class_id in range(self.num_classes):
                    test_frame_count[class_id] += np.any(
                        labels == class_id, axis=(1, 2)
                    ).sum()
                encoded = self.num_classes * labels[valid] + predictions[valid]
                confusion += np.bincount(
                    encoded, minlength=self.num_classes**2
                ).reshape(self.num_classes, self.num_classes)

        if not total_valid_pixels:
            raise ValueError("test data contains no evaluable pixels")

        intersection = np.diag(confusion)
        ground_truth = confusion.sum(axis=1)
        predicted = confusion.sum(axis=0)
        union = ground_truth + predicted - intersection
        present = union > 0
        iou = np.divide(
            intersection, union, out=np.full(self.num_classes, np.nan), where=present
        )
        dice_denominator = ground_truth + predicted
        dice = np.divide(
            2 * intersection,
            dice_denominator,
            out=np.full(self.num_classes, np.nan),
            where=dice_denominator > 0,
        )

        per_class = {}
        for class_id, (name, class_iou, class_dice) in enumerate(
            zip(self.class_names, iou, dice)
        ):
            flags = []
            if class_id in self.untrainable_class_ids:
                flags.append("untrainable_by_split")
            if test_frame_count[class_id] < 10:
                flags.append("indicative_only_under_10_test_frames")
            per_class[name] = {
                "iou": None if np.isnan(class_iou) else float(class_iou),
                "dice": None if np.isnan(class_dice) else float(class_dice),
                "test_frame_count": int(test_frame_count[class_id]),
                "test_pixel_share": float(ground_truth[class_id] / total_valid_pixels),
                "flags": flags,
            }
        learnable = np.array(
            [class_id not in self.untrainable_class_ids for class_id in range(self.num_classes)]
        )
        metrics = {
            "mIoU_all_classes": float(np.nanmean(iou)),
            "mIoU_learnable_classes": float(np.nanmean(iou[learnable])),
            "Dice": float(np.nanmean(dice)),
            "per_class": per_class,
            "paper_benchmark_comparable": False,
        }
        print(metrics)
        return metrics
