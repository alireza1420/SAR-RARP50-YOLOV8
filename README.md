# Surgery YOLOv8 Segmentation

Notebook-based YOLOv8 semantic segmentation workflow for a subset of the SAR-RARP50 robotic surgery dataset. The project extracts annotated frames from surgical AVI videos, converts class-ID PNG masks into YOLO polygon labels, fine-tunes a YOLOv8 nano segmentation model, evaluates it on a held-out video, and renders segmentation overlays on a full video.

## Project Structure

```text
surgery/
|-- notebooks/
|   |-- yolov8_seg_training.ipynb   # Dataset build, YOLOv8 training, evaluation
|   |-- video_mvp.ipynb             # Video inference and overlay export
|   |-- yolov8n-seg.pt              # COCO-pretrained YOLOv8 segmentation weights
|   |-- datasets/                   # Generated YOLO dataset when notebook is run
|   `-- results/                    # Generated checkpoints, metrics, videos
|-- videos/
|   |-- video_04/
|   |-- video_08/
|   |-- video_10/
|   `-- video_12/
|-- specs/001-yolov8-seg-training/  # Design notes and implementation plan
|-- requirements.txt
`-- CLAUDE.md
```

The `videos/`, generated `datasets/`, and generated `results/` directories are large/local artifacts and are ignored by Git.

## Dataset Layout

Each video folder is expected to contain a raw left-view video and segmentation masks named by frame number:

```text
videos/video_04/
|-- video_left.avi
|-- segmentation/
|   |-- 000000000.png
|   |-- 000000060.png
|   `-- ...
|-- action_discrete.txt
`-- action_continuous.txt
```

The current subset contains four videos and about 539 annotated frames. The default split holds out `video_12` for testing, then uses a seeded 80/20 train/validation split over the remaining videos.

## Requirements

- Python 3.10+
- pip
- Jupyter Notebook
- NVIDIA CUDA GPU recommended for training
- CPU inference fallback is supported by the video MVP notebook

Install dependencies from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Verify the Python environment:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Running the Training Notebook

The notebooks use relative paths. Run Jupyter from `notebooks/` so the default paths resolve correctly:

```powershell
cd notebooks
jupyter notebook yolov8_seg_training.ipynb
```

Run `yolov8_seg_training.ipynb` top to bottom. The notebook will:

1. Inspect `../videos/` and summarize the available frames and masks.
2. Extract annotated frames from `video_left.avi`.
3. Convert semantic PNG masks into YOLO segmentation polygon labels.
4. Write the generated dataset to `notebooks/datasets/sar_rarp50/`.
5. Fine-tune `yolov8n-seg.pt` for the configured number of epochs.
6. Evaluate on the held-out `video_12` test split.
7. Save metrics, plots, and model checkpoints.

Default training settings are in the first config cell:

```python
CONFIG = {
    'videos_dir': '../videos',
    'dataset_dir': 'datasets/sar_rarp50',
    'results_dir': 'results',
    'model': 'yolov8n-seg.pt',
    'epochs': 100,
    'imgsz': 640,
    'batch': 8,
    'device': 0 if torch.cuda.is_available() else 'cpu',
    'amp': True,
    'patience': 20,
    'seed': 42,
    'test_video': 'video_12',
    'val_fraction': 0.2,
}
```

If you start Jupyter from the repository root instead, update the config paths in the first cell before running.

## Training Outputs

Expected generated files when running from `notebooks/`:

```text
notebooks/datasets/sar_rarp50/
|-- dataset.yaml
|-- images/{train,val,test}/
`-- labels/{train,val,test}/

notebooks/results/train/
|-- weights/best.pt
|-- weights/last.pt
|-- results.csv
|-- confusion_matrix_test.png
|-- args.yaml
`-- evaluation_summary.json
```

The evaluation computes per-class IoU, mean IoU, background-excluded mean IoU, pixel accuracy, a confusion matrix, and qualitative overlay comparisons.

## Running the Video MVP

After training has produced `results/train/weights/best.pt`, open the video MVP notebook:

```powershell
cd notebooks
jupyter notebook video_mvp.ipynb
```

In the first cell, set paths for the current working directory. If running from `notebooks/`, use:

```python
VIDEO_PATH = '../videos/video_12/video_left.avi'
MODEL_PATH = 'results/train/weights/best.pt'
OUTPUT_PATH = 'results/video_mvp_output.mp4'
```

The notebook loads the trained checkpoint, runs frame-by-frame YOLOv8 segmentation, overlays class-colored masks and a legend, writes an MP4 output, and displays it inline when the file is small enough.

## Classes

| ID | Class |
| -- | ----- |
| 0 | background |
| 1 | bipolar_forceps |
| 2 | prograsp_forceps |
| 3 | large_needle_driver |
| 4 | vessel_sealer |
| 5 | grasping_retractor |
| 6 | monopolar_curved_scissors |
| 7 | ultrasound_probe |
| 8 | suction_instrument |
| 9 | suture_needle |

## Notes for Reproducibility

- The notebooks seed Python, NumPy, and PyTorch with `42`.
- `video_12` is the default held-out test video.
- Raw `videos/` data should not be modified by the notebooks.
- Generated datasets and results can be deleted and rebuilt by rerunning the training notebook.
- Training on CPU is possible but not recommended for the default 100-epoch run; CPU support is mainly intended for inference.
