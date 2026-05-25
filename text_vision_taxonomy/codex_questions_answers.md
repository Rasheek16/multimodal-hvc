# Codex Answers For Text-Vision Extension Path

Generated from inspection of `E:\text_vision`.

## Short Answer

At inference time, the current frame text-vision model can run from frames alone.

It does not strictly need transcript/text input. If `--transcript` is empty, `inference_frame_text_vision.py` passes neutral text features:

- `text_embedding`: zero vector
- `text_binary_prob`: `0.5`
- `text_severity_probs`: uniform `[0.25, 0.25, 0.25, 0.25]`
- `has_text`: `0`
- `transcript_source_id`: `missing`

If a transcript is provided, inference runs the frozen text teacher and injects the resulting text embedding, hate probability, and severity distribution into the fusion model.

Important nuance: `--source-video-id` loads frames and metadata from the manifest, but it does not automatically use the manifest transcript. Text is used at inference only when passed through `--transcript`.

Recommended next system:

```text
video frames -> current frame/fusion classifier for hate probability + severity
ASR/OCR/VLM description -> separate or auxiliary hate-type classifier
optional: feed ASR/OCR/description text into the existing fusion path as transcript-like text
```

## 1. Current Data Structure

### Files in `data/`

- `data/multimodal_manifest.csv`
- `data/asr_transcripts.csv`
- `data/text_teacher_cache.csv`
- `data/text_teacher_cache.pt`
- smoke cache files: `text_teacher_cache_smoke.csv`, `text_teacher_cache_smoke.pt`

### `data/multimodal_manifest.csv`

This is the main manifest. It has 9,845 clip rows and 3,281 unique source videos.

Columns:

```text
sample_id
source_video_id
label
split
clip_idx
start
stride
length
segment
num_frames
clip_path
video_file_name
label_source
source
transcription
video_path
has_video_file
frame_dir
num_frames_video
has_text
transcript_source
transcript_confidence
fallback_severity
clip_path_exists
video_path_exists
```

Meaning:

- `sample_id`: clip-row id.
- `source_video_id`: video-level id shared by all clips from the same video.
- `label`: binary label, `0 = non-hate`, `1 = hate`.
- `split`: `train`, `val`, or `test`.
- `clip_idx`, `start`, `stride`, `length`, `segment`, `num_frames`: clip sampling metadata.
- `clip_path`: path to pre-materialized clip tensor.
- `video_file_name`, `video_path`, `has_video_file`: source video metadata.
- `frame_dir`, `num_frames_video`: extracted frame directory and frame count.
- `label_source`, `source`: provenance metadata.
- `transcription`: human or ASR transcript text when available.
- `has_text`: whether usable transcript text exists.
- `transcript_source`: `human`, `asr`, or `missing`.
- `transcript_confidence`: confidence used to downweight ASR/teacher signals.
- `fallback_severity`: fallback pseudo-severity, usually `0` for non-hate and `2` for hate.
- `clip_path_exists`, `video_path_exists`: path validation flags.

Counts observed:

```text
clip rows: 9,845
unique videos: 3,281

video split:
train 2,295
val     493
test    493

clip split:
train 6,894
val   1,480
test  1,471

video labels:
0 non-hate 1,948
1 hate     1,333

video transcript source:
human     806
asr     1,967
missing   508
```

### `data/asr_transcripts.csv`

Rows: 2,475.

Columns:

```text
source_video_id
video_path
transcript
confidence
language
source
status
error
```

This is optional ASR output used by `prepare_manifest.py` to fill `transcription`, `transcript_source=asr`, and `transcript_confidence`.

### `data/text_teacher_cache.csv`

Rows: 2,773.

Columns:

```text
source_video_id
label
transcript_source
binary_prob
predicted_severity
pseudo_severity
teacher_confidence
```

This CSV is only a summary. The full tensor cache is in `data/text_teacher_cache.pt`.

### `data/text_teacher_cache.pt`

Built by `cache_text_teacher.py`. Per source video, it contains:

- `source_video_id`
- `transcript_source`
- `transcript_confidence`
- `text`
- `embedding`
- `binary_logit`
- `binary_prob`
- `severity_logits`
- `severity_probs`
- `predicted_severity`
- `pseudo_severity`
- `teacher_confidence`
- `label`

So the cache contains logits, probabilities, embeddings, pseudo-severity, and confidence, not only labels.

### Files Used By Training And Inference

`train_frame_text_vision.py` uses:

- manifest: `cfg.manifest_path`, default `text_vision/data/multimodal_manifest.csv`
- teacher cache: `cfg.text_teacher_cache_path`, default `text_vision/data/text_teacher_cache.pt`
- frame directories from manifest column `frame_dir`
- labels from manifest column `label`
- split from manifest column `split`
- checkpoint output: default `checkpoints/best_frame_text_vision.pt`
- last checkpoint output: default `checkpoints/last_frame_text_vision.pt`

`inference_frame_text_vision.py` uses:

- checkpoint: default `checkpoints/best_frame_text_vision.pt`, or `--checkpoint`
- manifest: used only for `--source-video-id` lookup
- frame source: `--frame-dir` or `--source-video-id`
- text model checkpoint: `--text-model`, used only if `--transcript` is provided
- transcript argument: `--transcript`, optional

`--teacher-cache` exists as an argument in inference but is not actually used by `classify()`.

## 2. Existing Model Inputs And Outputs

### Training Inputs

The frame training path collapses the clip-level manifest to one row per video using `video_level_manifest()`.

`VideoTextFrameDataset` returns:

```text
pixel_values             [T, C, H, W], sampled frames from frame_dir
label                    float binary label
label_long               long binary label
severity_target          pseudo severity class 0..3
text_embedding           teacher embedding or zeros
text_binary_prob         teacher hate probability or 0.5
text_severity_probs      teacher severity distribution or fallback
has_text                 1 only when manifest has text and teacher cache exists
has_teacher              1 when teacher cache entry exists
teacher_confidence       teacher confidence, 0 if missing
transcript_source_id     missing/human/asr id
source_video_id
frame_dir
num_source_frames
split
transcript_source
```

### Inference Inputs

`inference_frame_text_vision.py` accepts:

```text
--source-video-id
--frame-dir
--transcript
--checkpoint
--manifest
--text-model
--model-name
--num-frames
--threshold
--device
--local-files-only
```

Actual model batch at inference:

```text
pixel_values
text_embedding
text_binary_prob
text_severity_probs
has_text
transcript_source_id
```

### Model Outputs

The model returns internal tensors:

```text
binary_logits
severity_logits
vision_binary_logits
vision_severity_logits
projected_text_embedding
visual_embedding
fusion_embedding
```

The inference JSON currently outputs:

```text
source_video_id or frame_dir metadata
is_hate
binary_label
prob_hate
confidence
threshold
severity
severity_probs
num_frames
has_transcript_input
```

It outputs both hate/non-hate and severity.

It does not output hate type, target group, OCR evidence, ASR evidence, or description evidence yet.

### Does It Use Transcript Embeddings During Inference?

Yes, but only when `--transcript` is supplied.

If transcript text is supplied, `FrozenTextTeacher.predict_one()` creates:

```text
text_embedding
text_binary_prob
text_severity_probs
has_text = 1
transcript_source_id = human
```

If no transcript is supplied, the model still runs with neutral text defaults. That means the vision path has learned to make predictions from frames alone, with text used as an optional fusion signal.

## 3. Can We Add Hate-Type Labels Easily?

Yes, structurally this is straightforward.

The cleanest place is in `models/fusion_model.py`, after the fusion trunk creates `hidden`, because `hidden` is already the shared fused representation used for binary and severity outputs.

Current heads:

```text
self.binary_head = nn.Linear(hidden_dim // 2, 1)
self.severity_head = nn.Linear(hidden_dim // 2, 4)
```

Add:

```text
self.hate_type_head = nn.Linear(hidden_dim // 2, num_hate_types)
```

Then return:

```text
hate_type_logits
```

Use `BCEWithLogitsLoss` because hate type is naturally multi-label.

### Exact Tensors Available Before Final Classifier

From `FrameTextVisionModel`:

- `visual_embedding`
- `vision_binary_logits`
- `vision_severity_logits`
- `projected_text_embedding`

From batch/text side:

- `text_embedding`
- `text_binary_prob`
- `text_severity_probs`
- `has_text`
- `transcript_source_id`

Inside `FusionClassifier`:

- `source_features`
- `visual_severity_probs`
- concatenated `features`
- `hidden`

Inside `GatedFusionClassifier`:

- `visual_severity_probs`
- `source_features`
- `metadata`
- `visual_features`
- `text_features`
- `metadata_features`
- `gate`
- `fused`
- `hidden`

From `FrameTextGuidedClassifier.forward()`:

- final `binary_logits`
- final `severity_logits`
- `vision_binary_logits`
- `vision_severity_logits`
- `projected_text_embedding`
- `visual_embedding`
- `fusion_embedding`

Best target for hate type:

```text
fusion_embedding -> hate_type_head -> multi-label hate_type_logits
```

Optional secondary target:

```text
visual_embedding -> vision_hate_type_head
```

That optional vision-only head would help frame-only inference and text-dropout robustness, but it is not the smallest change.

## 4. Best Place To Add Video Description

Insert description generation in `inference_frame_text_vision.py` inside `classify()`.

Current order:

```text
load cfg/checkpoint
load model
threshold = ...
text = teacher_features(args, cfg, device, text_embedding_dim)
frames, metadata = frame_tensor_from_args(args, cfg)
batch = {"pixel_values": frames.unsqueeze(0), **text}
outputs = model(batch)
```

For description generation, change the order to resolve frames first:

```text
frames, metadata = frame_tensor_from_args(args, cfg)
description = build_video_description(...)
text = teacher_features_from_text(description.text_for_classifier, ...)
batch = {"pixel_values": frames.unsqueeze(0), **text}
```

Do not use normalized tensors for OCR/VLM. The current `frames` tensor is normalized for the image backbone. OCR/VLM should use raw sampled frame paths from `frame_dir`.

### Suggested New Module

New file:

```text
video_description.py
```

Suggested signatures:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

@dataclass
class EvidenceItem:
    source: str              # "asr", "ocr", "vlm", "frame"
    text: str
    confidence: float | None = None
    frame_path: str | None = None
    timestamp: float | None = None

@dataclass
class VideoDescriptionResult:
    transcript: str
    ocr_text: str
    visual_description: str
    text_for_classifier: str
    evidence: list[EvidenceItem]

def sample_description_frame_paths(
    frame_dir: str | Path,
    num_frames: int = 8,
) -> list[Path]:
    ...

def extract_ocr_evidence(
    frame_paths: Sequence[str | Path],
) -> list[EvidenceItem]:
    ...

def transcribe_video_audio(
    video_path: str | Path | None,
    existing_transcript: str = "",
) -> EvidenceItem | None:
    ...

def generate_visual_description(
    frame_paths: Sequence[str | Path],
    transcript: str = "",
    ocr_text: str = "",
) -> EvidenceItem:
    ...

def build_video_description(
    frame_dir: str | Path,
    video_path: str | Path | None = None,
    transcript: str = "",
    num_frames: int = 8,
) -> VideoDescriptionResult:
    ...
```

Then in `inference_frame_text_vision.py`, add a flag such as:

```text
--describe-video
--description-frames 8
--use-description-as-transcript
```

## 5. Existing Artifacts And Best Checkpoint

### Data Artifacts

- `data/multimodal_manifest.csv`
- `data/asr_transcripts.csv`
- `data/text_teacher_cache.csv`
- `data/text_teacher_cache.pt`
- smoke teacher cache files

### Checkpoints

Observed checkpoint files:

```text
best_text_guided_vision.pt
last_text_guided_vision.pt
smoke_text_guided_vision.pt
smoke_last_text_guided_vision.pt
best_frame_text_vision.pt
last_frame_text_vision.pt
resume_from_best_frame_text_vision.pt
best_frame_text_vision_finetuned.pt
last_frame_text_vision_finetuned.pt
best_frame_text_vision_relabel.pt
last_frame_text_vision_relabel.pt
best_frame_text_vision_relabel_all_splits.pt
last_frame_text_vision_relabel_all_splits.pt
```

### Reports

Main report folders:

```text
reports/frame_text_vision
reports/frame_text_vision_relabel
reports/frame_text_vision_relabel_all_splits
reports/label_audit
reports/label_audit_all_splits
reports/smoke
reports/smoke_eval
```

`reports/frame_text_vision_relabel_all_splits/metrics.csv` has the best observed test result:

```text
threshold: 0.31
test accuracy: 0.8438133874
test F1: 0.8188235294
test precision: 0.7981651376
test recall: 0.8405797101
test ROC-AUC: 0.9303570825
test PR-AUC: 0.9089611064
pseudo-severity accuracy: 0.7058823529
pseudo-severity macro F1: 0.6637554475
```

Earlier `reports/frame_text_vision/metrics.csv`:

```text
threshold: 0.32
test accuracy: 0.7545638945
test F1: 0.7084337349
test ROC-AUC: 0.8122696246
```

Best candidate for extension:

```text
checkpoints/best_frame_text_vision_relabel_all_splits.pt
```

Reason: it corresponds to the strongest reported binary metrics.

Caveat: the relabel-all-splits path appears to have used cleaned labels across train, val, and test. That is fine for extending the model, but future evaluation should be explicit about whether it is measuring against original labels or cleaned labels.

For a conservative production baseline against original labels, keep `best_frame_text_vision_finetuned.pt` as a comparison point because it was used by the label-audit process.

## 6. Label-Noise And Pseudo-Label Support

`audit_label_noise.py` works like this:

1. Load the manifest.
2. Collapse to video-level rows.
3. Load predictions from `reports/frame_text_vision/predictions.csv`, or generate predictions from a checkpoint if missing.
4. Load `text_teacher_cache.csv` as teacher summary.
5. For selected splits, compare current binary label against model `prob_hate`.
6. Mark likely false positives:

```text
label == 1 and prob_hate < threshold
```

7. Mark likely false negatives:

```text
label == 0 and prob_hate >= threshold
```

8. Use teacher binary probability and teacher confidence as supporting evidence.
9. Split issues into automatic relabels and manual review queue.
10. Write a new manifest with auto-overridden `label` values.

Generated outputs:

```text
suspected_label_issues.csv
auto_relabel_overrides.csv
review_queue.csv
multimodal_manifest_auto_relabel.csv
audit_summary.json
```

In `reports/label_audit_all_splits/audit_summary.json`:

```text
num_suspicious: 462
num_auto_relabel: 160
num_review: 302
auto false negatives: 74
auto false positives: 86
```

Can this support hate-type pseudo-labels?

Yes, but not with the current binary-only implementation directly.

The mechanism can be extended by adding new columns rather than only changing `label`. For example:

```text
hate_type_identity
hate_type_religion
hate_type_gender
hate_type_race
hate_type_nationality
hate_type_political
hate_type_disability
hate_type_other
target_group
hate_type_source
hate_type_confidence
hate_type_evidence
```

The current audit code already has useful pieces:

- per-video aggregation
- teacher/model confidence thresholds
- auto vs review split
- generated manifest output
- summary JSON output

Needed changes:

- Add a hate-type pseudo-label generator, likely from transcript/OCR/VLM description.
- Add multi-label columns to manifest.
- Add confidence and evidence columns.
- Add review/auto thresholds per hate type.
- Avoid mutating binary `label` when only adding type pseudo-labels.

## 7. Synthetic-Data Integration

Current frame training requires real frame paths for every training row.

`VideoTextFrameDataset.__getitem__()` always resolves `row["frame_dir"]`, lists image files, samples frames, and raises if no frames exist.

So pure text-only synthetic rows cannot be inserted into the current `train_frame_text_vision.py` dataloader without changes.

Possible integration paths:

### Option A: Separate Text Hate-Type Classifier

Train a text-only hate-type classifier on synthetic text plus real transcripts/descriptions.

Inputs:

```text
ASR transcript
OCR text
VLM video description
```

Outputs:

```text
hate type multi-labels
target group
evidence spans
```

This is the easiest synthetic-data path.

### Option B: Add Hate-Type Labels To Real Video Rows

Generate pseudo hate-type labels for real videos using:

```text
transcript + OCR + VLM description
```

Then train the existing frame/fusion model with the real video rows and new hate-type columns.

This works with the current dataloader shape because every row still has `frame_dir`.

### Option C: Mixed Text-Only And Video Training

Support synthetic text-only rows inside the same model by changing the dataloader/model to allow:

```text
pixel_values = missing
has_video = 0
```

Then the fusion model would need a text-only branch. This is larger and riskier than Options A or B.

### Option D: Cache Real Video Embeddings

Run the frame model over real videos once and save:

```text
visual_embedding
fusion_embedding
prob_hate
severity_probs
```

Then train a small hate-type classifier on cached embeddings plus pseudo labels.

Synthetic text-only examples could help a text branch, but they cannot directly create video embeddings without a video/frame source.

## 8. Minimal Implementation Plan

Goal:

1. Add video description generation.
2. Add hate-type multi-label prediction.
3. Return JSON containing hate probability, severity, hate types, target group, and evidence.

### Smallest Practical Path

Do not first retrain the whole frame model.

Add a separate hate-type text/description classifier path and keep the current frame model for binary hate + severity.

This matches the current code because frame inference already works without text, and hate type needs semantic evidence from text/OCR/description more than another binary visual head.

### File-By-File Changes

#### New: `video_description.py`

Responsibilities:

- sample raw frames from `frame_dir`
- run OCR on sampled frames
- optionally run ASR or consume provided transcript
- optionally run VLM/image-caption model on sampled frames
- produce `text_for_classifier`
- return structured evidence

Main API:

```python
def build_video_description(
    frame_dir: str | Path,
    video_path: str | Path | None = None,
    transcript: str = "",
    num_frames: int = 8,
) -> VideoDescriptionResult:
    ...
```

#### New: `hate_type_classifier.py`

Responsibilities:

- load hate-type label schema
- run rule/model based multi-label prediction from description text
- return target group and evidence spans

Start with a lightweight interface even if the first implementation is model-backed later:

```python
def classify_hate_types(
    text: str,
    prob_hate: float | None = None,
    severity: int | None = None,
) -> HateTypeResult:
    ...
```

Output:

```text
hate_type_probs
hate_types
target_group
evidence
```

#### Update: `inference_frame_text_vision.py`

Changes:

- Add CLI flags:

```text
--describe-video
--description-frames
--use-description-as-transcript
--hate-type-threshold
```

- Resolve frames first.
- Build optional description from raw frames.
- Combine transcript + OCR + description into classifier text.
- If `--use-description-as-transcript`, feed combined text through `FrozenTextTeacher` as the existing text fusion input.
- Run current frame/fusion model as before.
- Run hate-type classifier on description text.
- Extend JSON output.

Target JSON shape:

```json
{
  "is_hate": true,
  "binary_label": 1,
  "prob_hate": 0.91,
  "confidence": 0.91,
  "severity": 2,
  "severity_probs": [0.01, 0.12, 0.72, 0.15],
  "hate_types": ["gender", "identity_attack"],
  "hate_type_probs": {
    "gender": 0.83,
    "identity_attack": 0.74
  },
  "target_group": "women",
  "evidence": [
    {
      "source": "asr",
      "text": "...",
      "confidence": 0.82
    },
    {
      "source": "ocr",
      "text": "...",
      "frame_path": "..."
    },
    {
      "source": "vlm",
      "text": "..."
    }
  ],
  "description": {
    "transcript": "...",
    "ocr_text": "...",
    "visual_description": "..."
  },
  "has_transcript_input": true,
  "has_description_input": true
}
```

#### Optional Update Later: `models/fusion_model.py`

If hate-type prediction should be part of the neural frame/fusion checkpoint:

- add `num_hate_types`
- add `hate_type_head`
- return `hate_type_logits`

#### Optional Update Later: `models/frame_text_vision.py`

If vision-only hate-type supervision is wanted:

- add `vision_hate_type_head` from `visual_embedding`
- return `vision_hate_type_logits`

#### Optional Update Later: `data_utils.py`

If training hate type inside the frame model:

- parse hate-type multi-hot columns from manifest
- return `hate_type_targets`
- return `has_hate_type_labels`

#### Optional Update Later: `train_frame_text_vision.py`

If training hate type inside the frame model:

- add `--hate-type-labels` or config label list
- add BCE multi-label loss
- mask hate-type loss when labels are missing
- report hate-type metrics
- save hate-type label list in checkpoint

#### Optional Update Later: `audit_label_noise.py`

For pseudo hate-type labels:

- add pseudo-label generation mode
- write hate-type columns into a new manifest
- keep binary relabeling separate from hate-type enrichment

## Final Decision

Because inference can run frame-only and text is optional, the best next design is:

```text
video frames -> current frame/fusion classifier -> hate probability + severity
ASR/OCR/VLM description -> hate-type classifier -> hate types + target group + evidence
```

Then optionally feed:

```text
ASR + OCR + VLM description
```

back into the existing `--transcript`/teacher path to improve binary and severity predictions when text evidence is available.

