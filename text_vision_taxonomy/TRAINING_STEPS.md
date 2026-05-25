# Training Steps For `text_vision`

Run all commands from:

```powershell
cd E:\m-hvc\multimodal-hvc
```

## Current Status

Already done:

- `text_vision` package has been created.
- Manifest has been prepared:

```text
text_vision\data\multimodal_manifest.csv
```

- Frozen RoBERTa teacher cache has been created for the available human transcripts:

```text
text_vision\data\text_teacher_cache.pt
text_vision\data\text_teacher_cache.csv
```

- Basic script verification has been run:

```text
python -m compileall text_vision
```

- A one-batch smoke training run was completed only to verify wiring:

```text
text_vision\checkpoints\smoke_text_guided_vision.pt
text_vision\checkpoints\smoke_last_text_guided_vision.pt
text_vision\reports\smoke\
text_vision\reports\smoke_eval\
```

Not done yet:

- Full base VideoMAE training.
- Severity-boosted fusion fine-tuning.
- Full final evaluation from the real trained checkpoint.
- Optional ASR transcript expansion.

Important: the smoke checkpoint is not a final model. It only proves the scripts run.

## Data Being Used

The training data comes from the diversified video dataset under:

```text
E:\m-hvc\datasets\diversified
```

Main input files:

```text
E:\m-hvc\datasets\diversified\splits.csv
E:\m-hvc\datasets\diversified\materialized_clip_index.csv
E:\m-hvc\datasets\diversified\clips
```

Generated multimodal files:

```text
text_vision\data\multimodal_manifest.csv
text_vision\data\text_teacher_cache.pt
text_vision\data\text_teacher_cache.csv
```

### Source File Roles

`splits.csv` is the video-level metadata file. It contains:

- `source_video_id`
- `video_file_name`
- binary `label`
- original `source`
- human `transcription` when available
- original `video_path`
- frame directory metadata
- split assignment: `train`, `val`, or `test`

`materialized_clip_index.csv` is the clip-level index. It contains:

- `source_video_id`
- binary `label`
- `split`
- `clip_idx`
- clip temporal sampling info: `start`, `stride`, `length`, `segment`
- `num_frames`
- `clip_path`

`clips` contains pre-materialized PyTorch clip tensors:

```text
E:\m-hvc\datasets\diversified\clips\train\...\*.pt
E:\m-hvc\datasets\diversified\clips\val\...\*.pt
E:\m-hvc\datasets\diversified\clips\test\...\*.pt
```

Each `.pt` clip payload contains a video tensor and metadata. The clip tensor is loaded as visual input for VideoMAE.

### Dataset Size

Current manifest:

```text
Total videos: 3,281
Total clips:  9,845
```

Video-level split:

```text
train: 2,295 videos
val:     493 videos
test:    493 videos
```

Clip-level split:

```text
train: 6,894 clips
val:   1,480 clips
test:  1,471 clips
```

### Binary Labels

The main real supervised target is binary hate/non-hate:

```text
0 = non-hate
1 = hate
```

Video-level label distribution:

```text
train: non-hate 1,362 | hate 933
val:   non-hate   293 | hate 200
test:  non-hate   293 | hate 200
```

Clip-level label distribution:

```text
train: non-hate 4,095 | hate 2,799
val:   non-hate   886 | hate   594
test:  non-hate   863 | hate   608
```

The same binary label is assigned to every materialized clip from a video.

### Transcript Coverage

Human transcript coverage is limited:

```text
Human transcript videos: 806 / 3,281
Missing transcript videos: 2,475 / 3,281
```

Video-level transcript coverage by split:

```text
train: human 574 | missing 1,721
val:   human 122 | missing   371
test:  human 110 | missing   383
```

Clip-level transcript coverage by split:

```text
train: human 1,731 | missing 5,163
val:   human   371 | missing 1,109
test:  human   335 | missing 1,136
```

Current transcript sources:

```text
human
missing
```

ASR has not been generated yet, so there are currently no `asr` transcript rows.

### Clip Tensor Type

The visual data is loaded from `.pt` files. The existing clips are stored as tensors like:

```text
[C, T, H, W]
```

Example shape:

```text
[3, 24, 224, 224]
```

The training dataset converts every clip to VideoMAE input:

```text
[T, C, H, W]
```

Final model input shape per sample:

```text
[16, 3, 224, 224]
```

How this happens:

- Clips with `length=16`, `24`, or `32` are temporally resampled to `16` frames.
- Pixel values are converted to float.
- Pixel values are normalized using ImageNet-style mean/std.
- The batch shape becomes:

```text
[batch_size, 16, 3, 224, 224]
```

Current clip temporal lengths:

```text
length 16: 3,358 clips
length 24: 3,203 clips
length 32: 3,284 clips
```

Current temporal strides:

```text
stride 1: 2,693 clips
stride 2: 2,453 clips
stride 3: 2,328 clips
stride 4: 2,371 clips
```

## How The Data Is Used

### Manifest Creation

`prepare_manifest.py` joins:

```text
splits.csv
materialized_clip_index.csv
optional asr_transcripts.csv
```

It writes:

```text
text_vision\data\multimodal_manifest.csv
```

The manifest is clip-level. Each row represents one visual clip, plus video-level metadata joined onto it.

Important manifest fields:

- `sample_id`
- `source_video_id`
- `label`
- `split`
- `clip_idx`
- `clip_path`
- `transcription`
- `has_text`
- `transcript_source`
- `transcript_confidence`
- `fallback_severity`
- `clip_path_exists`

### Text Teacher Cache

`cache_text_teacher.py` uses the frozen RoBERTa model:

```text
text_classification\roberta_base_finetuned_dualhead.pt
```

It only runs on videos with usable transcript text.

For each transcript video, it caches:

- text embedding
- binary hate logit
- binary hate probability
- severity logits
- severity probabilities
- predicted severity
- pseudo severity
- teacher confidence

Output:

```text
text_vision\data\text_teacher_cache.pt
text_vision\data\text_teacher_cache.csv
```

The text teacher stays frozen. It is not trained again.

Important clarification:

For videos with transcripts, the text model does predict from the transcript and those predictions are used to teach the vision model. This is just done once before training and saved in `text_teacher_cache.pt`, instead of running RoBERTa again inside every training batch.

So conceptually the flow is:

```text
transcript
-> frozen RoBERTa teacher
-> text hate probability + text severity probabilities + text embedding
-> cached teacher outputs
-> VideoMAE/fusion training losses
```

The reason for caching is efficiency and reproducibility. RoBERTa outputs do not change because the text teacher is frozen.

During training, each clip from a transcript-backed video loads the cached teacher outputs and uses them as auxiliary guidance.

The binary video label is still the real dataset label. The text teacher's binary prediction is not used to replace the real label; it is used as a soft distillation signal.

Severity is different because there are no real video severity labels. For severity, the text teacher prediction becomes the pseudo-label or soft pseudo-target.

### Severity Labels

There are no true video severity labels in the local dataset.

Severity is pseudo-labeled like this:

```text
non-hate video -> severity 0
hate video with transcript -> severity from frozen text teacher
hate video without transcript -> fallback severity 2
```

Severity classes:

```text
0 = non-hate / no severity
1 = lower severity hate
2 = medium severity hate fallback or prediction
3 = high severity hate
```

Severity metrics are therefore pseudo-label metrics, not true ground-truth severity metrics.

### Samples With Transcripts

For clips from videos with transcripts:

- visual clip is used
- binary video label is used
- text teacher embedding is used
- text binary probability is used
- text severity probabilities are used
- text-vision embedding alignment loss is active
- text distillation losses are active
- transcript source is encoded as `human`

### Samples Without Transcripts

For clips from videos without transcripts:

- visual clip is used
- binary video label is still used
- text teacher losses are masked out
- embedding alignment loss is masked out
- text features are replaced with neutral defaults
- fallback severity is used only as weak pseudo supervision when configured
- transcript source is encoded as `missing`

This means all videos still train the visual binary classifier, even when text is missing.

### Optional ASR Samples

If ASR is generated later:

```powershell
python text_vision\asr_transcribe.py `
  --model-size small `
  --device cuda
```

Then `prepare_manifest.py` can mark those rows as:

```text
transcript_source = asr
```

ASR text is used similarly to human transcript text, but with lower confidence weighting.

### Training Losses

The total loss is:

```text
total_loss =
  1.00 * video_binary_label_loss
+ 0.35 * text_binary_distillation_loss
+ 0.45 * text_severity_distillation_loss
+ 0.25 * text_vision_embedding_alignment_loss
+ 0.20 * severity_consistency_loss
+ optional severity_ce_loss
```

How each loss uses data:

`video_binary_label_loss`

- Uses real binary labels.
- Applies to all clips.
- This is the main supervised training signal.

`text_binary_distillation_loss`

- Uses frozen text teacher binary probability.
- Applies only when `has_teacher = 1`.
- Masked out for missing-text videos.

`text_severity_distillation_loss`

- Uses frozen text teacher severity probability distribution.
- Applies only when `has_teacher = 1`.
- Masked out for missing-text videos.

`text_vision_embedding_alignment_loss`

- Aligns visual projection to cached text embedding.
- Applies only when `has_teacher = 1`.
- Masked out for missing-text videos.

`severity_consistency_loss`

- Encourages severity probability above `0` to agree with hate probability.
- Applies to all clips.

`severity_ce_loss`

- Optional pseudo-severity cross entropy.
- Used mainly during fusion fine-tuning.

### Model Inputs During Training

Each training batch contains:

```text
pixel_values             visual clip tensor
label                    binary hate/non-hate label
severity_target          pseudo severity label
text_embedding           cached teacher embedding or zero vector
text_binary_prob         cached teacher hate probability or 0.5
text_severity_probs      cached teacher severity probabilities or uniform/fallback
has_text                 whether transcript exists
has_teacher              whether cached teacher output exists
teacher_confidence       confidence weight for text losses
transcript_source_id     missing/human/asr encoded as an integer
source_video_id          video id
clip_path                clip file path
```

### Model Outputs

The model predicts:

```text
binary hate/non-hate logit
binary hate probability
severity logits for classes 0,1,2,3
severity probabilities
confidence score
```

### Evaluation Data Use

Evaluation runs on clip rows, then averages predictions by `source_video_id` to produce video-level predictions.

Validation split:

- used for threshold tuning
- used for model selection

Test split:

- used for final held-out evaluation

Train split:

- used for optimization
- can be reported for debugging but is not the main performance number

Evaluation reports:

```text
text_vision\reports\metrics.csv
text_vision\reports\model_comparison.csv
text_vision\reports\predictions.csv
text_vision\reports\confusion_binary.png
text_vision\reports\confusion_severity.png
text_vision\reports\hard_examples.csv
```

Metrics are reported for:

- all videos
- human transcript videos
- ASR transcript videos, if ASR exists
- missing transcript videos

## Accuracy And F1 Measurement

Yes, accuracy and F1 are measured, but there are two different stages.

### During Training

During each training epoch, the script records:

- train loss components
- validation accuracy
- validation F1
- validation precision
- validation recall
- validation ROC-AUC
- validation PR-AUC
- validation pseudo-severity accuracy
- validation pseudo-severity macro F1

These are written to:

```text
text_vision\reports\history.csv
```

Important detail:

```text
Per-epoch train accuracy and train F1 are not computed by default.
```

Reason:

- train accuracy/F1 would require an additional prediction pass over the full training set every epoch
- the training set has `6,894` clips
- running full train evaluation every epoch slows VideoMAE training significantly

So during training, the main train-side metric is loss, while validation accuracy/F1 are used for model selection.

### During Final Evaluation

When this command is run:

```powershell
python text_vision\evaluate.py --local-files-only
```

the evaluator measures accuracy and F1 for:

```text
train
val
test
```

It writes the full metrics to:

```text
text_vision\reports\metrics.csv
```

The most important metrics are:

```text
test accuracy
test F1
test precision
test recall
test ROC-AUC
test PR-AUC
```

Validation metrics are used to tune the binary threshold. Test metrics should be treated as the final held-out result.

## 1. Install Dependencies

```powershell
pip install -r text_vision\requirements.txt
```

Status: should already be satisfied in the current environment, but run this if using a fresh environment.

## 2. Prepare Manifest

```powershell
python text_vision\prepare_manifest.py
```

Status: already done.

Expected output is approximately:

```text
clip rows: 9,845
video rows: 3,281
text coverage by video: 806 / 3,281
```

## 3. Cache Frozen Text Teacher

```powershell
python text_vision\cache_text_teacher.py --batch-size 16 --local-files-only
```

Status: already done for the current manifest and human transcripts.

If the RoBERTa tokenizer/config is not cached locally, run once without `--local-files-only`:

```powershell
python text_vision\cache_text_teacher.py --batch-size 16
```

This writes:

```text
text_vision\data\text_teacher_cache.pt
text_vision\data\text_teacher_cache.csv
```

## 4. Train Base Text-Guided VideoMAE

Status: not done yet. This is the next main step.

```powershell
python text_vision\train_text_guided_vision.py `
  --epochs 8 `
  --batch-size 4 `
  --accum-steps 8 `
  --local-files-only
```

If CUDA runs out of memory, reduce the batch size and increase accumulation:

```powershell
python text_vision\train_text_guided_vision.py `
  --epochs 8 `
  --batch-size 2 `
  --accum-steps 16 `
  --local-files-only
```

Outputs:

```text
text_vision\checkpoints\best_text_guided_vision.pt
text_vision\checkpoints\last_text_guided_vision.pt
text_vision\reports\history.csv
```

## 5. Severity-Boosted Fusion Fine-Tuning

Status: not done yet. Run this only after step 4 finishes.

```powershell
python text_vision\finetune_fusion.py `
  --epochs 3 `
  --batch-size 4 `
  --accum-steps 8 `
  --resume `
  --local-files-only
```

To fine-tune from the best base checkpoint:

```powershell
python text_vision\finetune_fusion.py `
  --epochs 3 `
  --batch-size 4 `
  --accum-steps 8 `
  --resume `
  --last-checkpoint text_vision\checkpoints\best_text_guided_vision.pt `
  --local-files-only
```

## 6. Evaluate

Status: only smoke evaluation has been done. Full evaluation is not done yet.

```powershell
python text_vision\evaluate.py --local-files-only
```

Reports are saved to:

```text
text_vision\reports\metrics.csv
text_vision\reports\model_comparison.csv
text_vision\reports\predictions.csv
text_vision\reports\confusion_binary.png
text_vision\reports\confusion_severity.png
text_vision\reports\hard_examples.csv
```

## 7. Run Inference

Status: smoke inference has been verified. Use this after full training for real predictions.

Without transcript:

```powershell
python text_vision\inference.py `
  --source-video-id R_hate_video_100 `
  --max-clips 5 `
  --local-files-only
```

With transcript:

```powershell
python text_vision\inference.py `
  --source-video-id R_hate_video_100 `
  --transcript "example transcript here" `
  --max-clips 5 `
  --local-files-only
```

## Optional: Add ASR For Better Coverage

Status: partially started. The first ASR run processed `138` videos before hitting a video with a bad or missing audio stream. The ASR script has been updated so future runs continue past bad videos and record them with `status=failed`.

Run ASR for videos without transcripts:

```powershell
python text_vision\asr_transcribe.py `
  --model-size small `
  --device cuda
```

If the run stops or your terminal closes, run the same command again. It resumes from:

```text
text_vision\data\asr_transcripts.csv
```

The ASR CSV is video-level, not clip-level. One ASR transcript is generated per original video and then joined onto all clips from that video.

ASR output columns:

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

Status values:

```text
ok      transcript was produced
empty   ASR ran but no speech text was produced
failed  video/audio decode failed; the error is recorded and the run continues
```

The warning about Hugging Face symlinks on Windows is harmless. It only means the model cache may use more disk space.

Then rebuild the manifest and teacher cache:

```powershell
python text_vision\prepare_manifest.py --asr-cache text_vision\data\asr_transcripts.csv

python text_vision\cache_text_teacher.py --batch-size 16 --local-files-only
```

Then train again from step 4.

## Notes

- CUDA is strongly recommended.
- CPU training will be very slow.
- Severity labels are pseudo-labels from the frozen text teacher, not true video severity labels.
- The smoke checkpoint is only for script verification, not final performance.
