# Codex Handoff: Text Vision Taxonomy System

Date: 2026-05-20

This file is for future Codex sessions working in `D:\text_vision_taxonomy`. Read this first before changing the model, training scripts, inference path, or documentation.

## Current User Goal

The user wants a strong multi-label taxonomy moderation system for video/content moderation.

Base models already exist:

1. Text base model:
   - input: transcript/text
   - output: hate/non-hate probability + severity
   - checkpoint: `D:\text_vision_taxonomy\roberta_base_finetuned_dualhead.pt`

2. Vision base model:
   - input: video frames
   - output: hate/non-hate probability + severity
   - checkpoint: `checkpoints\best_frame_text_vision_relabel_all_splits.pt`

The taxonomy model is separate. It predicts these nine labels:

```text
hate_speech
discrimination
contextual_hate
threat
violence
fear
sexual
illegal
online_harm
```

`neutral` is not a taxonomy output. Neutral means all nine taxonomy labels are zero.

## Most Important Current Decision

Use the text transformer taxonomy model as the production taxonomy labeler.

Current recommended architecture:

```text
raw video
  -> frame sampling
  -> ASR transcript
  -> OCR text
  -> metadata text
  -> RoBERTa text taxonomy model
  -> calibrated thresholds
  -> rare-label rules
  -> base-model gating
  -> final taxonomy labels

raw video frames
  -> vision base model
  -> hate/severity signal only

transcript/metadata text
  -> text base model
  -> hate/severity signal only
```

Do not use the video taxonomy head as a final taxonomy source yet. It is experimental.

Reason:

- text taxonomy metrics are strong and stable
- video taxonomy frame coverage is incomplete
- video taxonomy validation predicted too many labels under weak thresholds
- taxonomy labels are mostly semantic and benefit from transcript/OCR/metadata text

## Key Files Created For Handoff And Explanation

Read these:

```text
MODEL_ARCHITECTURE_METHODOLOGY.md
ARCHITECTURE_DIAGRAM.md
CODEX_TAXONOMY_HANDOFF.md
TAXONOMY_RUN_COMMANDS.md
```

`MODEL_ARCHITECTURE_METHODOLOGY.md` is the big explanation document. It includes:

- architecture
- model inputs
- how inputs are generated
- training methodology
- metric formulas
- current numbers
- why text taxonomy is preferred

`ARCHITECTURE_DIAGRAM.md` contains Mermaid diagrams for:

- runtime architecture
- training/calibration flow

## Best Current Taxonomy Model

Text transformer taxonomy model:

```text
checkpoints\transformer_multilabel_classifier_feedback\best
```

Report directories:

```text
reports\transformer_multilabel_classifier_feedback
reports\transformer_multilabel_feedback_calibration
reports\transformer_multilabel_feedback_eval
```

Production thresholds:

```text
reports\transformer_multilabel_feedback_calibration\thresholds.json
```

Current production thresholds:

| Label | Threshold |
|---|---:|
| hate_speech | 0.7644 |
| discrimination | 0.7842 |
| contextual_hate | 0.8324 |
| threat | 0.9900 |
| violence | 0.3850 |
| fear | 0.7500 |
| sexual | 0.9358 |
| illegal | 0.9900 |
| online_harm | 0.9000 |

## Current Best Metrics

Best current production-style result is calibrated RoBERTa text taxonomy on the test split.

Test rows: `154`

| Metric | Value |
|---|---:|
| micro precision | 0.9290 |
| micro recall | 0.8276 |
| micro F1 | 0.8754 |
| macro precision | 0.6247 |
| macro recall | 0.5633 |
| macro F1 | 0.5885 |
| neutral false positive rate | 0.0625 |
| rare false positive count | 0 |
| rare predicted positive count | 0 |
| rare true positive count | 5 |

Per-label calibrated test metrics:

| Label | True | Predicted | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| hate_speech | 37 | 35 | 0.9714 | 0.9189 | 0.9444 |
| discrimination | 35 | 27 | 1.0000 | 0.7714 | 0.8710 |
| contextual_hate | 25 | 19 | 1.0000 | 0.7600 | 0.8636 |
| threat | 3 | 0 | 0.0000 | 0.0000 | 0.0000 |
| violence | 31 | 37 | 0.7838 | 0.9355 | 0.8529 |
| fear | 19 | 15 | 0.8667 | 0.6842 | 0.7647 |
| sexual | 22 | 22 | 1.0000 | 1.0000 | 1.0000 |
| illegal | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| online_harm | 2 | 0 | 0.0000 | 0.0000 | 0.0000 |

Interpretation:

- strong: `hate_speech`, `discrimination`, `contextual_hate`, `violence`, `sexual`
- decent but recall-limited: `fear`
- intentionally conservative: `threat`, `illegal`, `online_harm`

Do not over-optimize rare label recall without adding more data.

## Dataset

Primary dataset:

```text
D:\hvc\datasets\action\archieve\final_dataset.csv
```

Loaded taxonomy split:

| Split | Rows |
|---|---:|
| train | 750 |
| val | 156 |
| test | 154 |
| total | 1060 |

Label counts:

| Label | Count |
|---|---:|
| hate_speech | 210 |
| discrimination | 208 |
| contextual_hate | 151 |
| threat | 17 |
| violence | 251 |
| fear | 115 |
| sexual | 101 |
| illegal | 8 |
| online_harm | 14 |

Zero-label neutral rows: `386`.

Rare labels are the biggest weakness:

```text
threat = 17
illegal = 8
online_harm = 14
```

## Runtime Command To Use Now

For raw video inference, use this:

```powershell
python inference_moderation_system.py `
  --video "D:\hvc\datasets\old-dataset\video\hate_video_428.mp4" `
  --vision-checkpoint checkpoints\best_frame_text_vision_relabel_all_splits.pt `
  --taxonomy-model-type transformer `
  --transformer-taxonomy-model checkpoints\transformer_multilabel_classifier_feedback\best `
  --taxonomy-thresholds reports\transformer_multilabel_feedback_calibration\thresholds.json `
  --no-use-video-taxonomy `
  --output reports\single_video_result.json
```

This uses:

- transformer text taxonomy for final taxonomy labels
- text base model for hate/severity
- vision base model for hate/severity
- no video taxonomy head

## Training Commands

Train/retrain the text transformer taxonomy model:

```powershell
python retrain_taxonomy_with_feedback.py `
  --dataset D:\hvc\datasets\action\archieve\final_dataset.csv `
  --trainer transformer `
  --model-names roberta-base `
  --output-dir checkpoints\transformer_multilabel_classifier_feedback `
  --report-dir reports\transformer_multilabel_classifier_feedback `
  --epochs 8 `
  --batch-size 4 `
  --loss focal `
  --use-pos-weight `
  --max-pos-weight 5 `
  --local-files-only
```

Calibrate thresholds:

```powershell
python calibrate_multilabel_thresholds.py `
  --predictions reports\transformer_multilabel_classifier_feedback\predictions.csv `
  --output-dir reports\transformer_multilabel_feedback_calibration `
  --split val
```

Evaluate:

```powershell
python evaluate_multilabel_system.py `
  --tfidf-predictions missing.csv `
  --ensemble-predictions missing.csv `
  --transformer-predictions reports\transformer_multilabel_classifier_feedback\predictions.csv `
  --transformer-thresholds reports\transformer_multilabel_feedback_calibration\thresholds.json `
  --output-dir reports\transformer_multilabel_feedback_eval `
  --eval-split test `
  --table-system transformer_calibrated
```

## Runtime Input Creation

`video_metadata_generator.py` handles raw video preprocessing:

- samples frames from video using OpenCV
- runs ASR with `faster_whisper` if available/enabled
- runs OCR on sampled frames if OCR stack is available
- VLM captioning is optional and disabled by default
- rule-derived metadata is disabled by default

Important: the taxonomy model does not directly classify raw pixels. It classifies generated/provided text metadata, transcript, and OCR text.

## Rare Label Rules

File:

```text
rare_label_rules.py
```

Rare labels:

```text
threat
illegal
online_harm
```

Rules:

- `threat` requires explicit threat text, high violence probability, high severity, or extremely high probability.
- `illegal` requires explicit illegal/crime/weapon/drug/abuse/fraud evidence.
- `online_harm` requires online/social/chat/comment/post/harassment/doxxing context.

If evidence is missing, rare labels become candidates or suppressed labels, not final labels.

## Base Model Gating

Implemented in:

```text
inference_moderation_system.py
```

Rules:

- If both text and vision hate probabilities are available and both are below `0.35`, suppress hate-specific labels.
- Do not suppress `violence`, `sexual`, or `fear` only because hate probability is low.
- If text or vision hate probability is high enough, near-threshold hate labels can be boosted.
- If severity is high, violence/threat can become candidates; threat still needs evidence.

## Video Taxonomy Status

Video taxonomy training was patched to be a deeper model path:

- temporal video encoder + fusion taxonomy layers can be trained
- many frames supported with `--num-frames`
- AMP and gradient accumulation supported
- positional embeddings are interpolated when frame count changes

But it is not recommended for production now.

Current video manifest coverage:

| Item | Count |
|---|---:|
| manifest rows | 1045 |
| usable frame rows | 490 |
| dropped rows without usable frames | 555 |
| train usable frame rows | 331 |
| val usable frame rows | 99 |
| test usable frame rows | 60 |

The user ran a 64-frame video taxonomy training run and saw bad threshold behavior: many labels predicted on almost every validation row. We patched thresholding to be more conservative, but the production decision remains text-first.

## Recent Code Changes To Remember

1. `config.py`
   - defaults text model to local `D:\text_vision_taxonomy\roberta_base_finetuned_dualhead.pt` when present.

2. `inference_moderation_system.py`
   - defaults taxonomy model type to transformer.
   - can run from raw `--video`.
   - supports `--no-use-video-taxonomy`.
   - combines metadata generation, text taxonomy, text base, vision base, rare rules, and gating.

3. `video_metadata_generator.py`
   - added metadata dataclass and generation flow.
   - rule-derived metadata disabled by default.

4. `train_transformer_multilabel_classifier.py`
   - fixed JSON serialization issue.
   - trains RoBERTa multi-label taxonomy model.

5. `retrain_taxonomy_with_feedback.py`
   - defaults to transformer trainer.
   - supports feedback dataset.

6. `train_taxonomy_head.py`
   - patched for deeper video taxonomy training.
   - supports `--train-scope temporal_fusion`.
   - supports AMP, accumulation, frame chunk size, many frames.
   - conservative thresholding added for rare labels.

7. `train_frame_text_vision.py`
   - checkpoint loader interpolates frame positional embeddings for changed frame counts.

8. Documentation added:
   - `MODEL_ARCHITECTURE_METHODOLOGY.md`
   - `ARCHITECTURE_DIAGRAM.md`
   - `CODEX_TAXONOMY_HANDOFF.md`

## What To Do In A New Session

Start by reading:

```text
CODEX_TAXONOMY_HANDOFF.md
MODEL_ARCHITECTURE_METHODOLOGY.md
ARCHITECTURE_DIAGRAM.md
```

If the user asks "what should we use now?", answer:

```text
Use text transformer taxonomy as the final taxonomy model.
Use text and vision base models only as hate/severity support signals.
Disable video taxonomy for production until frame coverage and metrics improve.
```

If the user asks for runtime:

```powershell
python inference_moderation_system.py `
  --video "path\to\video.mp4" `
  --vision-checkpoint checkpoints\best_frame_text_vision_relabel_all_splits.pt `
  --taxonomy-model-type transformer `
  --transformer-taxonomy-model checkpoints\transformer_multilabel_classifier_feedback\best `
  --taxonomy-thresholds reports\transformer_multilabel_feedback_calibration\thresholds.json `
  --no-use-video-taxonomy `
  --output reports\single_video_result.json
```

If the user wants better video taxonomy:

1. Fix missing frame coverage first.
2. Ensure every dataset row has valid frame directories or raw video paths.
3. Retrain video taxonomy.
4. Evaluate against text transformer on the same split.
5. Only enable video taxonomy if it improves calibrated precision/F1 and keeps rare false positives controlled.

