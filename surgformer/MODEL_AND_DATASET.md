# Surg-SegFormer on SAR-RARP50

## Purpose

This implementation performs **semantic segmentation** of robotic-surgery frames.
For every input pixel, it predicts one of ten labels: background or one of nine
surgical instrument classes. It does not distinguish separate instances of the
same instrument and does not use information from earlier or later video frames.

The implementation adapts Surg-SegFormer's dual-branch idea to SAR-RARP50. A
smaller SegFormer branch handles the more common foreground classes, while a
larger branch focuses on rarer classes. Their predictions are combined only
during inference.

```mermaid
flowchart LR
    A[RGB surgical frame] --> B[Resize and normalize]
    B --> C[MiT-B2 coarse branch]
    B --> D[MiT-B5 fine branch]
    C --> E[Coarse-class logits]
    D --> F[Dense-skip decoder]
    F --> G[Fine-class logits]
    E --> H[Confidence-based fusion]
    G --> H
    H --> I[Morphological refinement]
    I --> J[10-class pixel mask]
```

## Model architecture

### Coarse branch

The coarse branch uses `nvidia/mit-b2` through
`SegformerForSemanticSegmentation`. Its pretrained encoder supplies general
visual features, while a new ten-class SegFormer decode head is initialized for
SAR-RARP50.

The base prototype configuration assigns this branch:

- `0`: background
- `1`: tool clasper
- `2`: tool wrist
- `3`: tool shaft

During a full-data Colab run, this list is replaced with the foreground classes
occupying at least 1% of training pixels. Labels assigned to the other branch are
mapped to background for this branch's loss.

### Fine branch

The fine branch uses the larger pretrained `nvidia/mit-b5` encoder. Its four
encoder stages have different spatial resolutions and channel widths. The custom
dense-skip decoder:

1. Projects every encoder stage to 256 channels with a `1x1` convolution.
2. Upsamples all four projected stages to the highest encoder resolution.
3. Decodes from the deepest stage toward the shallowest stage.
4. Concatenates every earlier decoded feature into the next decoder layer.
5. Fuses the four decoded outputs and produces ten-class logits.

The base prototype configuration trains this branch on background plus classes
4-9. During a full-data Colab run, it instead receives the foreground classes
below the 1% threshold. Labels assigned to the other branch are mapped to
background for its loss.

### Why pretrained-model warnings appear

The NVIDIA MiT checkpoints were not trained with this repository's ten-class
segmentation heads. Consequently, Transformers reports the old ImageNet
classifier weights as `UNEXPECTED` and the new segmentation decoder weights as
`MISSING`. This is intentional:

- The B2 and B5 encoder weights are reused.
- The original classification heads are discarded.
- The B2 segmentation head and custom B5 decoder start from new weights.
- Training adapts those weights to the ten SAR-RARP50 labels.

## SAR-RARP50 adaptation

### Full-data Colab split

The Colab notebook downloads all official archives: 12,998 annotated frames
from operations 1-40 and 3,252 test frames from operations 41-50. It holds out
four complete development operations for validation using seed 42, so frames
from one operation never cross splits. Thus 36 development operations provide
gradient updates, four provide validation, and the ten official test operations
remain evaluation-only.

The repository's small local prototype split remains available for quick checks:

| Split | Frames | Source |
| --- | ---: | --- |
| Train | 326 | Training videos, seeded split |
| Validation | 81 | Training videos, seeded split |
| Test | 132 | Held-out `video_12` |

Do not compare results from the 539-frame prototype with the full-data run.

### Image and mask pairing

Images are stored under:

```text
notebooks/datasets/sar_rarp50/images/{train,val,test}/
```

An image name such as `video_04_000000120.jpg` is split into `video_04` and
`000000120`. The loader then reads its dense mask from:

```text
videos/video_04/segmentation/000000120.png
```

The loader uses these original dense class-ID PNG masks. The YOLO polygon label
files elsewhere in the generated dataset are not used by Surg-SegFormer.

The current label map is an identity map: source mask value `N` remains class
`N`. Unknown mask values cause an error instead of being silently converted.
Mask value `255` is reserved as the ignored label.

### Class grouping

The full-data notebook calculates class pixel shares from its 36-operation
training split. Foreground classes occupying at least 1% of its labelled pixels
go to the coarse branch; the remaining foreground classes go to the fine branch.
Background is learned by both branches. This avoids carrying the prototype
subset's frequency split into the full experiment.

| ID | Class | Branch | Training status |
| ---: | --- | --- | --- |
| 0 | background | both | learnable |
| 1 | tool clasper | data-derived | learnable if present in train |
| 2 | tool wrist | data-derived | learnable if present in train |
| 3 | tool shaft | data-derived | learnable if present in train |
| 4 | suturing needle | data-derived | learnable if present in train |
| 5 | thread | data-derived | learnable if present in train |
| 6 | suction tool | data-derived | learnable if present in train |
| 7 | needle holder | data-derived | learnable if present in train |
| 8 | clamps | data-derived | learnable if present in train |
| 9 | catheter | data-derived | learnable if present in train |

Any class absent from the selected training operations remains in the ten-class
output and is automatically marked `untrainable_by_split` during evaluation.

### Preprocessing and augmentation

Every frame is converted from OpenCV's BGR order to RGB, scaled to `[0, 1]`, and
normalized with ImageNet mean and standard deviation. The final tensor size is
`3 x 512 x 896`.

Training augmentation consists of:

- horizontal flip with probability 0.5;
- random `448 x 784` crop with probability 0.5;
- rotation of up to 15 degrees with probability 0.5;
- resize to `512 x 896`.

Validation and test samples are only resized; random augmentation is disabled.
The same geometric operation is applied to each image and its mask.

## Training behavior

The branches are trained independently and sequentially: all coarse epochs,
followed by all fine epochs. The full-data Colab notebook allows at most 10
epochs per branch and stops a branch after three consecutive validation epochs
without improvement. It restores that branch's best validation checkpoint
before proceeding. Fusion is not part of the training loss.

For each branch, the objective is:

```text
loss = 0.7 * Tversky loss + 0.3 * cross-entropy loss
```

The Tversky loss weights false positives by `0.7` and false negatives by `0.3`.
Optimization uses Adam with a learning rate of `5e-6`, weight decay of `1e-4`,
and a cyclic learning-rate schedule rising to `5e-5`. Python, NumPy, PyTorch,
data-loader workers, and split shuffling are seeded from `42`.

Each epoch writes a `last` checkpoint. A `best` checkpoint is replaced whenever
validation loss improves:

```text
coarse_best.pth
coarse_last.pth
fine_best.pth
fine_last.pth
```

The checkpoints include model, optimizer, scheduler, and epoch state. The
current trainer saves these states but does not implement resume-on-restart.

## Inference fusion

At inference time, both branches return ten-class logits at the input image
resolution. Fusion works independently at every pixel:

1. Convert each branch's logits to probabilities and its most likely class.
2. Use the fine prediction whenever the coarse branch predicts background.
3. If both predict foreground, use the fine prediction only when its confidence
   is greater than the coarse confidence; otherwise retain the coarse result.
4. Apply one `3x3` morphological closing and opening pass.
5. Use the morphologically refined result only where both branches predicted
   foreground.

This gives the common-class branch priority while still allowing a confident
fine-class prediction to replace it.

## Evaluation and outputs

After training, `main.py` evaluates the in-memory final-epoch model on the 132
held-out test frames. It reports:

- IoU and Dice for every class;
- mean IoU over all classes;
- mean IoU over the nine potentially learnable classes;
- mean Dice;
- test-frame count and test-pixel share for every class;
- warnings for untrainable classes and classes appearing in fewer than ten test
  frames.

The training entry point restores each branch's `best` checkpoint before final
evaluation. `video_inference.py` also loads `coarse_best.pth` and
`fine_best.pth` by default and renders colored masks over the held-out video.

## What this adaptation does not claim

- It is a paper-derived reconstruction, not official author code.
- Configuration fields marked `ASSUMPTION` were not fully specified by the
  paper.
- The SAR-RARP50 split and label distribution differ from the paper's EndoVis
  evaluation, so the paper's reported scores are not reproduction targets.
- Frames are processed independently; there is no temporal model or tracking.
- The model performs semantic, not instance, segmentation.
- Class 9 performance cannot be representative until that class appears in
  training data.

## Relevant files

- [`config.yaml`](config.yaml): task, classes, model variants, and hyperparameters
- [`prepare_dataset.py`](prepare_dataset.py): full-video split and annotated-frame extraction
- [`dataset_loader.py`](dataset_loader.py): frame-mask pairing and preprocessing
- [`model.py`](model.py): coarse/fine branches and dense decoder
- [`trainer.py`](trainer.py): branch targets, losses, optimization, and checkpoints
- [`fusion.py`](fusion.py): inference-time prediction fusion
- [`evaluation.py`](evaluation.py): per-class IoU and Dice metrics
- [`video_inference.py`](video_inference.py): full-video overlay rendering
- [`PROVENANCE.md`](PROVENANCE.md): reconstruction provenance and comparability note
