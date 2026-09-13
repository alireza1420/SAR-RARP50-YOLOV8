"""Render SAR-RARP50 segmentation overlays with measured inference throughput."""

import argparse
from collections import deque
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from main import load_config
from model import DualSegFormer


COLORS = [
    (0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (255, 0, 128), (0, 128, 255), (255, 128, 0),
]


def _load_model(config: dict, coarse_path: str, fine_path: str,
                device: torch.device) -> DualSegFormer:
    task = config["data"]["tasks"][config["data"]["active_task"]]
    model = DualSegFormer(config, task)
    coarse = torch.load(coarse_path, map_location="cpu", weights_only=True)["branch_state"]
    fine = torch.load(fine_path, map_location="cpu", weights_only=True)["branch_state"]
    model.coarse.load_state_dict(coarse["coarse"])
    model.fine_encoder.load_state_dict(fine["fine_encoder"])
    model.fine_decoder.load_state_dict(fine["fine_decoder"])
    return model.to(device).eval()


def _predict(frame: np.ndarray, model: DualSegFormer, config: dict,
             device: torch.device) -> np.ndarray:
    width, height = config["data"]["image_size"]
    image = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (width, height))
    image = image.astype(np.float32).transpose(2, 0, 1) / 255.0
    normalization = config["data"]["normalization"]
    mean = np.asarray(normalization["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(normalization["std"], dtype=np.float32)[:, None, None]
    tensor = torch.from_numpy((image - mean) / std).unsqueeze(0).to(device)
    with torch.inference_mode():
        labels = model(tensor)[0].cpu().numpy().astype(np.uint8)
    return cv2.resize(labels, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)


def annotate_frame(frame: np.ndarray, labels: np.ndarray, class_names: list[str],
                   alpha: float, fps: float) -> np.ndarray:
    """Reuse feature 001's colour blend and in-frame legend for dense predictions."""
    colors = np.asarray(COLORS[:len(class_names)], dtype=np.uint8)
    color_mask = colors[labels]
    blended = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)
    annotated = np.where((labels != 0)[..., None], blended, frame).copy()
    cv2.putText(
        annotated, f"Inference: {fps:.2f} FPS", (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
    )
    for class_id, name in enumerate(class_names[1:], 1):
        y = 48 + (class_id - 1) * 20
        cv2.rectangle(annotated, (10, y - 12), (24, y + 2), COLORS[class_id], -1)
        cv2.putText(
            annotated, name, (30, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (255, 255, 255), 1,
        )
    return annotated


def render_video(config: dict, input_path: str, output_path: str,
                 coarse_path: str, fine_path: str, device_name: str) -> None:
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    model = _load_model(config, coarse_path, fine_path, device)

    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open input video: {input_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"invalid video metadata: {width}x{height} at {fps} FPS")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open output video: {output_path}")

    latencies: list[float] = []
    rolling = deque(maxlen=config["video"]["fps_window"])
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = perf_counter()
            labels = _predict(frame, model, config, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency = perf_counter() - started
            latencies.append(latency)
            rolling.append(latency)
            live_fps = len(rolling) / sum(rolling)
            writer.write(annotate_frame(
                frame,
                labels,
                config["data"]["tasks"][config["data"]["active_task"]]["class_names"],
                config["video"]["overlay_alpha"],
                live_fps,
            ))
    finally:
        capture.release()
        writer.release()

    if not latencies:
        raise RuntimeError(f"no frames decoded from: {input_path}")
    print(f"Video saved: {output_path}")
    print(
        f"Throughput: {len(latencies) / sum(latencies):.3f} FPS mean, "
        f"p95 latency {np.percentile(latencies, 95) * 1000:.1f} ms, device {device}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--coarse-checkpoint")
    parser.add_argument("--fine-checkpoint")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args()
    resolved = load_config(arguments.config)
    checkpoint_dir = Path(resolved["training"]["checkpoint_dir"])
    render_video(
        resolved,
        arguments.input or resolved["video"]["input"],
        arguments.output or resolved["video"]["output"],
        arguments.coarse_checkpoint or str(checkpoint_dir / "coarse_best.pth"),
        arguments.fine_checkpoint or str(checkpoint_dir / "fine_best.pth"),
        arguments.device,
    )
