# Provenance

This is the human-reviewed Codex implementation derived from PaperCoder's
2026-09-11 11:36–11:40 output and rewritten against the lead-engineer ledger.
It is intentionally outside `outputs/`, which remains owned by reproducible
PaperCoder runs. `smoke_test.py` is the local acceptance check.

This corrected deliverable targets the local SAR-RARP50 split required by feature 002;
the genuine PaperCoder output remains preserved for comparison. After training, render
the held-out clip (with automatic CPU fallback) using:

```powershell
python video_inference.py
```

The paper's EndoVis scores, including its 0.80 mIoU, are not comparable to this
SAR-RARP50 retarget and are not reproduction targets. The coarse/fine split was derived
from training data only: classes 1-3 exceed 1% pixel share and classes 4-9 do not.
Class 9 has no train/validation examples, so evaluation reports it as
`untrainable_by_split` and publishes both ten-class and nine-learnable-class mIoU.
