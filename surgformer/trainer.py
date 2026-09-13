"""Independent branch training for Surg-SegFormer."""

import os
from collections.abc import Iterable

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model: nn.Module, train_data: DataLoader, val_data: DataLoader,
                 config: dict, task_config: dict) -> None:
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.task_config = task_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        training = config["training"]
        if training["optimizer"].lower() != "adam":
            raise ValueError("Surg-SegFormer specifies Adam")
        if training["scheduler"]["name"].lower() != "cyclic":
            raise ValueError("Surg-SegFormer specifies a cyclic LR scheduler")
        self.training = training
        self.loss_config = config["loss"]
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=self.loss_config["ignore_index"]
        )

    def tversky_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Follow Eq. 2: alpha weights FP; this contradicts the paper's FN prose."""
        ignore_index = self.loss_config["ignore_index"]
        valid = targets != ignore_index
        safe_targets = targets.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe_targets, logits.shape[1]).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1)
        probabilities = logits.softmax(dim=1) * valid
        one_hot = one_hot * valid

        dims = (0, 2, 3)
        true_positive = (probabilities * one_hot).sum(dims)
        false_positive = (probabilities * (1 - one_hot) * valid).sum(dims)
        false_negative = ((1 - probabilities) * one_hot).sum(dims)
        parameters = self.loss_config["tversky"]
        score = (true_positive + 1e-7) / (
            true_positive
            + parameters["alpha"] * false_positive
            + parameters["beta"] * false_negative
            + 1e-7
        )
        return (1 - score).mean()

    def _targets_for(self, masks: torch.Tensor, branch: str) -> torch.Tensor:
        """Map the other branch's foreground labels to background per FR-005."""
        allowed = self.task_config["class_groups"][branch]
        keep = torch.zeros_like(masks, dtype=torch.bool)
        for class_id in allowed:
            keep |= masks == class_id
        ignored = masks == self.loss_config["ignore_index"]
        return masks.masked_fill(~keep & ~ignored, self.config["fusion"]["background_class"])

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tversky = self.tversky_loss(logits, targets)
        cross_entropy = self.cross_entropy(logits, targets)
        weight = self.loss_config["tversky_weight"]
        return weight * tversky + (1 - weight) * cross_entropy

    def _run_epoch(self, loader: DataLoader, branch: str,
                   optimizer: torch.optim.Optimizer | None = None,
                   scheduler: torch.optim.lr_scheduler.LRScheduler | None = None) -> float:
        training = optimizer is not None
        self.model.train(training)
        total = 0.0
        batches = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                images = batch["image"].to(self.device)
                targets = self._targets_for(batch["mask"].to(self.device), branch)
                if not (targets != self.loss_config["ignore_index"]).any():
                    continue
                if training:
                    optimizer.zero_grad()
                loss = self._loss(self.model.branch_logits(images, branch), targets)
                if training:
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                total += loss.item()
                batches += 1
        if not batches:
            raise ValueError(f"no labelled pixels available for {branch}")
        return total / batches

    def _branch_state(self, branch: str) -> dict:
        if branch == "coarse":
            return {"coarse": self.model.coarse.state_dict()}
        return {
            "fine_encoder": self.model.fine_encoder.state_dict(),
            "fine_decoder": self.model.fine_decoder.state_dict(),
        }

    def _save(self, branch: str, name: str, optimizer: torch.optim.Optimizer,
              scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int) -> None:
        directory = self.training["checkpoint_dir"]
        os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "branch_state": self._branch_state(branch),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            },
            os.path.join(directory, f"{branch}_{name}.pth"),
        )

    def _train_branch(self, branch: str) -> None:
        parameters: Iterable[nn.Parameter] = self.model.parameters_for(branch)
        optimizer = torch.optim.Adam(
            parameters,
            lr=self.training["learning_rate"],
            weight_decay=self.training["weight_decay"],
        )
        scheduler_config = self.training["scheduler"]
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.training["learning_rate"],
            max_lr=scheduler_config["max_learning_rate"],
            step_size_up=max(
                1, len(self.train_data) * scheduler_config["step_size_up_epochs"]
            ),
            cycle_momentum=False,
        )

        best_validation = float("inf")
        for epoch in range(1, self.training["epochs"] + 1):
            train_loss = self._run_epoch(self.train_data, branch, optimizer, scheduler)
            validation_loss = self._run_epoch(self.val_data, branch)
            self._save(branch, "last", optimizer, scheduler, epoch)
            if validation_loss < best_validation:
                best_validation = validation_loss
                self._save(branch, "best", optimizer, scheduler, epoch)
            print(
                f"{branch} epoch {epoch}/{self.training['epochs']}: "
                f"train={train_loss:.6f}, val={validation_loss:.6f}"
            )

    def train(self) -> None:
        """Train branch-specific losses; confidence fusion is never in this path."""
        for branch in self.task_config["branches"]:
            self._train_branch(branch)
