"""Small end-to-end check for full-dataset preparation."""

from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from prepare_dataset import prepare


def _video(root: Path, name: str) -> None:
    directory = root / name
    masks = directory / "segmentation"
    masks.mkdir(parents=True)
    writer = cv2.VideoWriter(
        str(directory / "video_left.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 32)
    )
    assert writer.isOpened()
    for value in range(4):
        writer.write(np.full((32, 32, 3), value * 20, dtype=np.uint8))
    writer.release()
    cv2.imwrite(str(masks / "000000000.png"), np.zeros((32, 32), dtype=np.uint8))
    cv2.imwrite(str(masks / "000000002.png"), np.full((32, 32), 2, dtype=np.uint8))


def main() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source, output = root / "raw", root / "prepared"
        for name in ("video_01", "video_02", "video_41"):
            _video(source, name)
        manifest = prepare(source, output, val_operations=1)
        assert manifest["frames"] == {"train": 2, "val": 2, "test": 2}
        assert manifest["operations"]["test"] == [41]
        assert sum(manifest["class_pixel_counts"]["train"]) == 2 * 32 * 32
        assert len(list(output.glob("images/*/*.jpg"))) == 6
        assert len(list(output.glob("masks/*/segmentation/*.png"))) == 6


if __name__ == "__main__":
    main()
    print("dataset preparation test passed")
