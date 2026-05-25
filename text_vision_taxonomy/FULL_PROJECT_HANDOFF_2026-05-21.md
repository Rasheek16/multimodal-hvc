# Full Project Handoff - Text Vision Taxonomy

Date written: 2026-05-21

Workspace: `D:\text_vision_taxonomy`

Primary dataset: `D:\hvc\datasets\action\archieve\final_dataset.csv`

This file is intended to let a new session resume the project without needing the chat history.

## Current One-Line State

The best usable taxonomy system is the trained RoBERTa transformer multi-label taxonomy classifier with calibrated thresholds, combined at runtime with the existing text and vision hate/severity base models for gating and final moderation output. Video taxonomy is implemented but still experimental and should not be trusted as production taxonomy yet.

## Project Goal

Build a strong multi-label taxonomy moderation system for video/text content.

The final runtime target is:

```text
raw video
  -> frame sampler
  -> ASR transcript
  -> OCR text
  -> optional VLM/video captioner
  -> generated metadata:
       Scene
       Action
       action_clean
       Subcategories
       ParentLabels
  -> trained taxonomy classifier
  -> rare-label and base-model gating
  -> final moderation JSON
```

The taxonomy classifier does not directly invent raw-video labels by itself. It consumes generated or manually provided metadata text.

## Existing Base Models

There are two base moderation models that already existed before the taxonomy work:

1. Text base model:
   - Input: transcript/text
   - Output: hate/non-hate probability and severity
   - Current local checkpoint:
     `D:\text_vision_taxonomy\roberta_base_finetuned_dualhead.pt`
   - Runtime field names:
     `text_hate_prob`, `text_severity`

2. Vision base model:
   - Input: video frames/frame directory
   - Output: hate/non-hate probability and severity
   - Current checkpoint:
     `checkpoints\best_frame_text_vision_relabel_all_splits.pt`
   - Runtime field names:
     `vision_hate_prob`, `vision_severity`

These base models are not replacements for the taxonomy classifier. They are used as gating, calibration, and final hate/severity signals.

## Taxonomy Labels

The multi-label taxonomy uses 9 labels:

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

`neutral` is not a classifier label. A neutral row means all 9 taxonomy labels are zero.

Rare labels:

```text
threat
illegal
online_harm
```

These labels are extremely sparse and must be handled conservatively.

## Dataset Counts

Dataset: `D:\hvc\datasets\action\archieve\final_dataset.csv`

Current known total rows: 1060.

Label counts:

| label | count |
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

Approximate zero-label neutral rows from the current split/report state: 386.

Main source-video split:

| split | rows |
|---|---:|
| train | 750 |
| val | 156 |
| test | 154 |
| total | 1060 |

Splitting is source-video based using `File Name`, not random row-only splitting. This avoids leaking clips from the same source video across train/val/test.

## Text Fields Used For Taxonomy

Defined in `multilabel_taxonomy_utils.py`:

```python
TEXT_FIELDS = ["Scene", "Action", "Subcategories", "ParentLabels", "action_clean"]
```

The text block is formatted as:

```text
[SCENE]
{Scene}

[ACTION]
{Action}

[SUBCATEGORIES]
{Subcategories}

[PARENTLABELS]
{ParentLabels}

[ACTION_CLEAN]
{action_clean}
```

Important: the trained taxonomy classifiers consume these fields. They do not generate them.

## Main Production Recommendation

Use this as the current production taxonomy path:

```text
RoBERTa transformer taxonomy classifier
  + calibrated production thresholds
  + rare-label conservative handling
  + base model hate/severity gating
  + optional generated metadata from video
```

Do not use the video taxonomy head as the production taxonomy source yet. It is useful for experiments and future multimodal work, but the current metrics show severe overprediction.

## Best Current Taxonomy Model

Model type: deep-learning transformer multi-label classifier.

Base transformer: `roberta-base`

Training command used through feedback wrapper:

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

Best checkpoint:

```text
checkpoints\transformer_multilabel_classifier_feedback\best
```

Best raw report directory:

```text
reports\transformer_multilabel_classifier_feedback
```

Best calibrated eval directory:

```text
reports\transformer_multilabel_feedback_eval
```

Best thresholds:

```text
reports\transformer_multilabel_feedback_calibration\thresholds.json
```

## Best Transformer Metrics

Evaluation split: test.

Rows: 154.

Two useful variants are recorded:

1. `transformer_only`
   - Uses default 0.5 threshold.
   - Higher recall and slightly higher micro F1.
   - But rare false positives remain.

2. `transformer_calibrated`
   - Uses calibrated production thresholds.
   - Lower recall.
   - Better precision and rare-label control.
   - This is the preferred production setting.

Overall metrics:

| system | rows | micro_precision | micro_recall | micro_f1 | macro_precision | macro_recall | macro_f1 | neutral_fp_rate | rare_fp_count | rare_pred_count | rare_true_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| transformer_only | 154 | 0.8703 | 0.9253 | 0.8969 | 0.7758 | 0.7503 | 0.7531 | 0.1042 | 4 | 7 | 5 |
| transformer_calibrated | 154 | 0.9290 | 0.8276 | 0.8754 | 0.6247 | 0.5633 | 0.5885 | 0.0625 | 0 | 0 | 5 |

Preferred model for paper/prod discussion:

```text
transformer_calibrated
micro precision: 0.9290
micro recall:    0.8276
micro F1:        0.8754
rare FP count:   0
```

Per-label calibrated metrics:

| label | threshold | true_count | predicted_count | TP | FP | FN | precision | recall | f1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hate_speech | 0.7644 | 37 | 35 | 34 | 1 | 3 | 0.9714 | 0.9189 | 0.9444 |
| discrimination | 0.7842 | 35 | 27 | 27 | 0 | 8 | 1.0000 | 0.7714 | 0.8710 |
| contextual_hate | 0.8324 | 25 | 19 | 19 | 0 | 6 | 1.0000 | 0.7600 | 0.8636 |
| threat | 0.9900 | 3 | 0 | 0 | 0 | 3 | 0.0000 | 0.0000 | 0.0000 |
| violence | 0.3850 | 31 | 37 | 29 | 8 | 2 | 0.7838 | 0.9355 | 0.8529 |
| fear | 0.7500 | 19 | 15 | 13 | 2 | 6 | 0.8667 | 0.6842 | 0.7647 |
| sexual | 0.9358 | 22 | 22 | 22 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| illegal | 0.9900 | 0 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| online_harm | 0.9000 | 2 | 0 | 0 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 |

Interpretation:

- The calibrated transformer is very strong for common labels.
- Rare labels are intentionally suppressed unless confidence is very high.
- This sacrifices rare-label recall to avoid many false positives.
- This matches the project requirement: precision matters more than recall for final taxonomy labels.

## TF-IDF Baseline

Script:

```text
train_text_multilabel_classifier.py
```

Best baseline:

```text
tfidf_word_char_linearsvc_calibrated_ovr
```

Checkpoint/report:

```text
checkpoints\text_multilabel_classifier\best_model.joblib
reports\text_multilabel_classifier
```

Test metrics from `reports\text_multilabel_classifier\metrics.json`:

| system | rows | micro_precision | micro_recall | micro_f1 | macro_precision | macro_recall | macro_f1 | neutral_fp_rate | rare_fp_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tfidf_word_char_linearsvc_calibrated_ovr | 154 | 0.8929 | 0.7184 | 0.7962 | 0.6140 | 0.4783 | 0.5171 | 0.1042 | 0 |

Older `tfidf_only` evaluation with default threshold showed:

```text
micro precision: 0.9054
micro recall:    0.7701
micro F1:        0.8323
rare FP count:   1
```

The TF-IDF model is a useful baseline, not the best final model.

## Zero-Shot LLM Evaluation

Script:

```text
evaluate_zero_shot_llm_taxonomy.py
```

The zero-shot evaluator supports:

```text
openai
groq
hf
ollama
fixture
```

The Groq implementation uses LangChain `ChatGroq`:

```text
langchain-groq
langchain-core
```

The evaluator sends one API call per evaluated row. On the test split, that means 154 API calls per complete model run.

### Llama 3.1 8B Instant

Command that was run:

```powershell
python evaluate_zero_shot_llm_taxonomy.py `
  --provider groq `
  --model llama-3.1-8b-instant `
  --system-name zero_shot_groq_llama31_8b_instant `
  --dataset D:\hvc\datasets\action\archieve\final_dataset.csv `
  --eval-split test `
  --output-dir reports\zero_shot_llm_eval\groq_llama31_8b_instant `
  --resume
```

Report folder:

```text
reports\zero_shot_llm_eval\groq_llama31_8b_instant
```

Summary file:

```text
reports\zero_shot_llm_eval\groq_llama31_8b_instant\LLAMA_RESULTS.md
```

Metrics:

| model | rows | micro_precision | micro_recall | micro_f1 | macro_precision | macro_recall | macro_f1 | neutral_fp_rate | rare_fp_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama-3.1-8b-instant | 154 | 0.5924 | 0.5345 | 0.5619 | 0.4193 | 0.3725 | 0.3851 | 0.0417 | 11 |

### Llama 3.3 70B Versatile

Command:

```powershell
python evaluate_zero_shot_llm_taxonomy.py `
  --provider groq `
  --model llama-3.3-70b-versatile `
  --system-name zero_shot_groq_llama33_70b_versatile `
  --dataset D:\hvc\datasets\action\archieve\final_dataset.csv `
  --eval-split test `
  --output-dir reports\zero_shot_llm_eval\groq_llama33_70b_versatile `
  --resume
```

Report folder:

```text
reports\zero_shot_llm_eval\groq_llama33_70b_versatile
```

Summary file:

```text
reports\zero_shot_llm_eval\groq_llama33_70b_versatile\LLAMA_70B_RESULTS.md
```

Metrics:

| model | rows | micro_precision | micro_recall | micro_f1 | macro_precision | macro_recall | macro_f1 | neutral_fp_rate | rare_fp_count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama-3.3-70b-versatile | 154 | 0.6291 | 0.5460 | 0.5846 | 0.4591 | 0.3342 | 0.3534 | 0.0208 | 4 |

70B per-label summary:

| label | true_count | predicted_count | TP | FP | FN | precision | recall | f1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hate_speech | 37 | 55 | 31 | 24 | 6 | 0.5636 | 0.8378 | 0.6739 |
| discrimination | 35 | 40 | 26 | 14 | 9 | 0.6500 | 0.7429 | 0.6933 |
| contextual_hate | 25 | 15 | 7 | 8 | 18 | 0.4667 | 0.2800 | 0.3500 |
| threat | 3 | 3 | 0 | 3 | 3 | 0.0000 | 0.0000 | 0.0000 |
| violence | 31 | 25 | 22 | 3 | 9 | 0.8800 | 0.7097 | 0.7857 |
| fear | 19 | 7 | 4 | 3 | 15 | 0.5714 | 0.2105 | 0.3077 |
| sexual | 22 | 5 | 5 | 0 | 17 | 1.0000 | 0.2273 | 0.3704 |
| illegal | 0 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| online_harm | 2 | 1 | 0 | 1 | 2 | 0.0000 | 0.0000 | 0.0000 |

### Why Llama Performs Poorly

The LLMs perform poorly because they are zero-shot classifiers that have not learned this dataset's annotation policy. Labels such as `hate_speech`, `discrimination`, `contextual_hate`, and `fear` overlap semantically. The LLM guesses broad semantic labels instead of following the exact dataset distribution and threshold policy.

Main reasons:

- No supervised training on this taxonomy.
- No source-video split training distribution.
- No calibrated thresholds.
- No base-model gating.
- No learned rarity prior for `threat`, `illegal`, and `online_harm`.
- Multi-label exact evaluation is strict.

The zero-shot LLMs are good comparison baselines for a paper, but should not be the main classifier.

## Model Comparison Snapshot

| system | test rows | micro_precision | micro_recall | micro_f1 | rare_fp_count | status |
|---|---:|---:|---:|---:|---:|---|
| RoBERTa taxonomy calibrated | 154 | 0.9290 | 0.8276 | 0.8754 | 0 | best production candidate |
| RoBERTa taxonomy default 0.5 | 154 | 0.8703 | 0.9253 | 0.8969 | 4 | high recall, more risky |
| TF-IDF calibrated baseline | 154 | 0.8929 | 0.7184 | 0.7962 | 0 | strong classical baseline |
| Llama 3.3 70B zero-shot | 154 | 0.6291 | 0.5460 | 0.5846 | 4 | paper comparison only |
| Llama 3.1 8B zero-shot | 154 | 0.5924 | 0.5345 | 0.5619 | 11 | paper comparison only |

## Video Taxonomy Head

Script:

```text
train_taxonomy_head.py
```

Purpose:

Train a video/frame-based taxonomy head using the existing frame-text-vision backbone and many sampled frames.

Most recent training command:

```powershell
python train_taxonomy_head.py `
  --manifest data\multilabel_manifest.csv `
  --checkpoint checkpoints\best_frame_text_vision_relabel_all_splits.pt `
  --output-dir checkpoints `
  --report-dir reports\taxonomy_head `
  --freeze-backbone `
  --train-scope temporal_fusion `
  --loss focal `
  --focal-gamma 2 `
  --pos-weight-mode capped `
  --pos-weight-cap 5 `
  --num-epochs 8 `
  --batch-size 1 `
  --accum-steps 4 `
  --num-frames 64 `
  --frame-chunk-size 8 `
  --amp `
  --local-files-only
```

Observed issue:

```text
filtered 555 manifest rows without usable frames
```

Usable video taxonomy data:

```text
data\multilabel_manifest.csv
total valid rows: 1045
usable rows with frames: about 490
train usable: 331
val usable: 99
test usable: 60
```

Current video taxonomy checkpoint:

```text
checkpoints\best_taxonomy_head.pt
```

Reports:

```text
reports\taxonomy_head
```

Video taxonomy validation behavior is currently poor. It overpredicts many labels. Example from latest `reports\taxonomy_head\per_label_metrics.csv`:

| label | support | predicted_count | precision | recall | f1 |
|---|---:|---:|---:|---:|---:|
| hate_speech | 5 | 96 | 0.0521 | 1.0000 | 0.0990 |
| discrimination | 4 | 99 | 0.0404 | 1.0000 | 0.0777 |
| contextual_hate | 2 | 99 | 0.0202 | 1.0000 | 0.0396 |
| violence | 35 | 79 | 0.4304 | 0.9714 | 0.5965 |
| fear | 10 | 79 | 0.1266 | 1.0000 | 0.2247 |
| sexual | 1 | 99 | 0.0101 | 1.0000 | 0.0200 |
| illegal | 1 | 68 | 0.0147 | 1.0000 | 0.0290 |

Conclusion:

The video taxonomy model is not ready. It can remain an optional candidate signal later, but production inference should use:

```text
--no-use-video-taxonomy
```

## Runtime Inference

Main runtime script:

```text
inference_moderation_system.py
```

Default taxonomy model type:

```text
transformer
```

Important default paths in the current code:

```text
--transformer-taxonomy-model checkpoints/transformer_multilabel_classifier_feedback/best
--taxonomy-thresholds reports/transformer_multilabel_feedback_calibration/thresholds.json
--taxonomy-classifier checkpoints/text_multilabel_classifier/best_model.joblib
--video-taxonomy-checkpoint checkpoints/best_taxonomy_head.pt
--video-taxonomy-thresholds reports/taxonomy_head/thresholds.json
--vision-checkpoint checkpoints/best_frame_text_vision_relabel_all_splits.pt
--text-model D:\text_vision_taxonomy\roberta_base_finetuned_dualhead.pt
```

Important parser flags:

```text
--video
--frame-dir
--generate-metadata
--disable-asr
--disable-ocr
--enable-vlm
--disable-vlm
--enable-rule-metadata
--scene
--action
--action-clean
--subcategories
--parent-labels
--transcript
--ocr-text
--taxonomy-model-type {transformer,tfidf}
--use-video-taxonomy / --no-use-video-taxonomy
--output
```

Recommended production raw-video command:

```powershell
python inference_moderation_system.py `
  --video "D:\path\to\video.mp4" `
  --taxonomy-model-type transformer `
  --transformer-taxonomy-model checkpoints\transformer_multilabel_classifier_feedback\best `
  --taxonomy-thresholds reports\transformer_multilabel_feedback_calibration\thresholds.json `
  --vision-checkpoint checkpoints\best_frame_text_vision_relabel_all_splits.pt `
  --text-model D:\text_vision_taxonomy\roberta_base_finetuned_dualhead.pt `
  --generate-metadata `
  --no-use-video-taxonomy `
  --output reports\single_video_result.json
```

Manual metadata inference command:

```powershell
python inference_moderation_system.py `
  --scene "A man gives an aggressive speech targeting a group" `
  --action "verbal abuse" `
  --subcategories "hate speech, discrimination" `
  --transcript "They are using insulting hateful language toward a group." `
  --taxonomy-model-type transformer `
  --transformer-taxonomy-model checkpoints\transformer_multilabel_classifier_feedback\best `
  --taxonomy-thresholds reports\transformer_multilabel_feedback_calibration\thresholds.json `
  --no-use-video-taxonomy `
  --local-files-only
```

If manually provided metadata fields are present, the system should use them directly and not regenerate them.

## Runtime Output Contract

The final JSON output should include:

```json
{
  "metadata": {
    "scene": "...",
    "action": "...",
    "action_clean": "...",
    "subcategories": "...",
    "parent_labels": "...",
    "transcript": "...",
    "ocr_text": "..."
  },
  "base_model_signals": {
    "text_hate_prob": 0.0,
    "vision_hate_prob": 0.0,
    "text_severity": 0,
    "vision_severity": 0
  },
  "vision": {
    "available": true,
    "prob_hate": 0.0,
    "severity": 0
  },
  "text": {
    "available": true,
    "prob_hate": 0.0,
    "severity": 0
  },
  "taxonomy": {
    "available": true,
    "taxonomy_raw_labels": [],
    "taxonomy_final_labels": [],
    "suppressed_labels": [],
    "candidate_labels": [],
    "probs": {},
    "thresholds_used": {},
    "model_sources": {}
  },
  "final": {
    "is_hate": false,
    "prob_hate": 0.0,
    "severity": 0,
    "taxonomy_labels": []
  },
  "evidence": []
}
```

Important fields:

- `taxonomy_raw_labels`: labels from taxonomy model before final gating.
- `taxonomy_final_labels`: final labels after thresholds, rare-label rules, and base-model gating.
- `suppressed_labels`: labels suppressed because confidence/evidence/gating failed.
- `candidate_labels`: labels that may be reviewed but are not high-confidence final outputs.
- `base_model_signals`: text and vision hate/severity signals used for gating.

## Base-Model Gating Rules

The intended behavior:

1. `hate_speech`, `discrimination`, `contextual_hate`, `threat`, `illegal`, and `online_harm` are hate-specific taxonomy labels.

2. Hate-specific taxonomy labels require either:
   - taxonomy confidence above threshold, or
   - supporting high hate probability from text/vision base model.

3. If both `text_hate_prob` and `vision_hate_prob` are below 0.35:
   - suppress hate-specific taxonomy labels.

4. Do not suppress `violence`, `sexual`, or `fear` just because hate probability is low.

5. If severity is 3 or higher:
   - allow `violence` and `threat` candidates.
   - `threat` still needs text evidence or very high taxonomy probability.

6. Rare labels should be conservative:
   - `threat`: needs explicit threat language, severity/violence support, or very high taxonomy probability.
   - `illegal`: needs explicit illegal/weapon/drug/crime/abuse evidence.
   - `online_harm`: needs online/social/chat/comment/meme/post/harassment context.

## Metadata Generation

Main file:

```text
video_metadata_generator.py
```

Data class:

```python
@dataclass
class VideoMetadata:
    scene: str
    action: str
    action_clean: str
    subcategories: str
    parent_labels: str
    transcript: str
    ocr_text: str
    evidence: list[dict]
```

Key functions:

```text
sample_frames(video_path or frame_dir, num_frames=8)
run_asr(video_path) -> transcript
run_ocr(frame_paths) -> ocr_text
generate_scene_description(frame_paths, transcript="", ocr_text="") -> scene
generate_action_description(frame_paths, transcript="", ocr_text="") -> action/action_clean
derive_subcategories_and_parent_labels(scene, action, transcript, ocr_text)
build_video_metadata(video_path=None, frame_dir=None, transcript="") -> VideoMetadata
```

Important current behavior:

- ASR can generate transcript from raw video when available.
- OCR uses `pytesseract` plus the external Tesseract binary.
- VLM/captioning is optional and has been flaky in this environment.
- If VLM/OCR/ASR is unavailable, the system should not crash. It should return partial metadata and evidence explaining what was skipped.
- Rule-derived metadata exists as a fallback, but the user prefers model-based/deep-learning taxonomy classification, not only rules.

Observed raw-video result example:

The system successfully processed:

```text
D:\hvc\datasets\old-dataset\video\hate_video_428.mp4
```

It generated a long transcript, used text base model and vision base model, and produced final JSON.

Observed signals:

```text
text_hate_prob:   0.9998679161
vision_hate_prob: 0.1384477764
text_severity:    2
vision_severity:  0
taxonomy label:   violence
```

The output was saved when `--output reports\single_video_result.json` was used.

Known issue from that run:

- OCR failed because Tesseract was not installed or not in PATH.
- VLM captioning failed due a Transformers task mismatch: `image-to-text` was not available in that installed version.

## Feedback Loop

Runtime feedback log:

```text
reports\runtime_feedback\inference_logs.csv
```

Current observed count:

```text
8 rows
```

Mining script:

```text
mine_taxonomy_training_cases.py
```

Command:

```powershell
python mine_taxonomy_training_cases.py `
  --inference-log reports\runtime_feedback\inference_logs.csv `
  --review-output data\taxonomy_review_queue.csv `
  --pseudo-output data\taxonomy_pseudo_labeled.csv
```

Earlier output after 3 log rows:

```text
review_rows: 2
pseudo_rows: 1
```

Feedback training script:

```text
retrain_taxonomy_with_feedback.py
```

Current feedback dataset:

```text
data\taxonomy_feedback_training.csv
```

Earlier output:

```text
rows: 1061
source_counts:
  original: 1060
  pseudo_label: 1
synthetic_rows: 1
```

The feedback loop is implemented but not yet rich. It needs many more reviewed or pseudo-labeled runtime cases before it materially changes training.

## Important Scripts

Core taxonomy:

```text
multilabel_taxonomy_utils.py
train_text_multilabel_classifier.py
train_transformer_multilabel_classifier.py
calibrate_multilabel_thresholds.py
evaluate_multilabel_system.py
inference_multilabel_taxonomy.py
inference_moderation_system.py
rare_label_rules.py
taxonomy_postprocessing.py
ensemble_multilabel_classifier.py
```

Video/frame:

```text
prepare_multilabel_manifest.py
train_taxonomy_head.py
inference_frame_text_vision.py
models\frame_text_vision.py
segment_frame_utils.py
video_metadata_generator.py
video_description.py
asr_transcribe.py
```

Feedback and augmentation:

```text
augment_multilabel_dataset.py
mine_taxonomy_training_cases.py
retrain_taxonomy_with_feedback.py
generate_synthetic_taxonomy_data.py
generate_taxonomy_pseudo_labels.py
generate_taxonomy_review_queue.py
audit_label_noise.py
```

LLM evaluation:

```text
evaluate_zero_shot_llm_taxonomy.py
```

Smoke/check scripts:

```text
smoke_moderation_checks.py
evaluate_final_taxonomy_system.py
evaluate_moderation_system.py
```

## Existing Documentation Files

Already created in repo root:

```text
MODEL_ARCHITECTURE_METHODOLOGY.md
ARCHITECTURE_DIAGRAM.md
CODEX_TAXONOMY_HANDOFF.md
RESEARCH_ABSTRACT.md
TAXONOMY_RUN_COMMANDS.md
```

LLM result docs:

```text
reports\zero_shot_llm_eval\groq_llama31_8b_instant\LLAMA_RESULTS.md
reports\zero_shot_llm_eval\groq_llama33_70b_versatile\LLAMA_70B_RESULTS.md
```

This file is the newest and most complete handoff snapshot.

## Dependencies

Current `requirements.txt`:

```text
torch
torchvision
transformers
pandas
numpy
scikit-learn
matplotlib
tqdm
faster-whisper
openai
accelerate
langchain-groq
```

External tools likely needed:

```text
ffmpeg       - needed for video/audio extraction and ASR workflows
tesseract    - needed for OCR through pytesseract
```

Python packages that may also be needed depending on the path:

```text
pytesseract
Pillow
opencv-python
```

Groq LLM eval needs:

```powershell
$env:GROQ_API_KEY = "YOUR_KEY"
```

Security note:

The user pasted a Groq key in a prior command. It should be treated as exposed and rotated/revoked. Do not write API keys into files or commands. Prefer environment variables.

## Exact Commands To Resume

### 1. Train TF-IDF baseline

```powershell
python train_text_multilabel_classifier.py `
  --dataset D:\hvc\datasets\action\archieve\final_dataset.csv `
  --output-dir checkpoints\text_multilabel_classifier `
  --report-dir reports\text_multilabel_classifier
```

### 2. Train deep-learning transformer taxonomy

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

### 3. Calibrate transformer thresholds

Use the transformer predictions from:

```text
reports\transformer_multilabel_classifier_feedback\predictions.csv
```

Command pattern:

```powershell
python calibrate_multilabel_thresholds.py `
  --predictions reports\transformer_multilabel_classifier_feedback\predictions.csv `
  --output-dir reports\transformer_multilabel_feedback_calibration `
  --split val
```

### 4. Evaluate transformer system

```powershell
python evaluate_multilabel_system.py `
  --transformer-predictions reports\transformer_multilabel_classifier_feedback\predictions.csv `
  --transformer-thresholds reports\transformer_multilabel_feedback_calibration\thresholds.json `
  --output-dir reports\transformer_multilabel_feedback_eval `
  --eval-split test
```

### 5. Run raw-video production inference

```powershell
python inference_moderation_system.py `
  --video "D:\path\to\video.mp4" `
  --taxonomy-model-type transformer `
  --transformer-taxonomy-model checkpoints\transformer_multilabel_classifier_feedback\best `
  --taxonomy-thresholds reports\transformer_multilabel_feedback_calibration\thresholds.json `
  --vision-checkpoint checkpoints\best_frame_text_vision_relabel_all_splits.pt `
  --text-model D:\text_vision_taxonomy\roberta_base_finetuned_dualhead.pt `
  --generate-metadata `
  --no-use-video-taxonomy `
  --output reports\single_video_result.json
```

### 6. Run Llama 70B zero-shot comparison

```powershell
$env:GROQ_API_KEY = "YOUR_ROTATED_KEY"

python evaluate_zero_shot_llm_taxonomy.py `
  --provider groq `
  --model llama-3.3-70b-versatile `
  --system-name zero_shot_groq_llama33_70b_versatile `
  --dataset D:\hvc\datasets\action\archieve\final_dataset.csv `
  --eval-split test `
  --output-dir reports\zero_shot_llm_eval\groq_llama33_70b_versatile `
  --resume
```

## Known Warnings And Fixes

1. `enable_nested_tensor` warning:
   - Occurred from `models\frame_text_vision.py`.
   - It was addressed by disabling nested tensor behavior in the transformer encoder path.

2. Repeated FPS warning:
   - Occurred from `segment_frame_utils.py`.
   - Frames with unknown FPS are sampled uniformly.
   - Warning spam was reduced to a startup summary style.

3. Tesseract OCR missing:
   - Install Tesseract and add it to PATH.
   - Python package alone is not enough.

4. VLM captioning task mismatch:
   - `image-to-text` task failed with current Transformers.
   - Possible future fix: use a supported task/model combo, for example `image-text-to-text` if compatible, or call BLIP model/processor directly.

5. Video taxonomy instability:
   - Due limited usable frame rows and class imbalance.
   - Do not use as final taxonomy until it is retrained and calibrated with enough video examples.

6. Transformer load reports:
   - Some `UNEXPECTED` and `MISSING` keys appeared when loading RoBERTa from a checkpoint with different heads.
   - This is expected when initializing a sequence classification head from a base/masked-LM style checkpoint, as long as training follows.

## What Is Done

Completed:

- TF-IDF word/char baseline training.
- Transformer multi-label taxonomy training.
- Focal loss and capped positive weights.
- Threshold calibration.
- Rare-label conservative defaults.
- Runtime metadata generation interface.
- Raw-video inference pipeline.
- Base text model integration.
- Base vision model integration.
- Base-signal taxonomy gating.
- Runtime feedback logging.
- Feedback mining and retraining wrapper.
- Zero-shot LLM evaluator with Groq ChatGroq.
- Llama 3.1 8B and Llama 3.3 70B comparison reports.
- Architecture/methodology/abstract/run-command docs.

Partially done:

- Video taxonomy deep-learning model.
- VLM captioning layer.
- OCR layer, because external Tesseract was missing.
- Automated feedback loop, because only a few runtime examples exist.

Not production-ready:

- Video taxonomy head as a final labeler.
- Zero-shot LLM taxonomy as a final labeler.
- Rare-label recall.

## Next Best Work

1. Keep production taxonomy on calibrated RoBERTa transformer.

2. Use raw-video inference with:

```text
--no-use-video-taxonomy
```

3. Improve metadata generation:
   - Fix OCR by installing Tesseract.
   - Fix VLM captioning with a supported caption model path.
   - Consider using a stronger VLM to generate `Scene` and `Action`.

4. Build a real feedback dataset:
   - Run inference on many raw videos.
   - Review `data\taxonomy_review_queue.csv`.
   - Mark reviewed labels.
   - Retrain transformer with reviewed examples.

5. Improve rare labels:
   - Add more true `threat`, `illegal`, and `online_harm` examples.
   - Add hard negatives that contain similar wording but should not fire the rare label.
   - Keep rare production thresholds high.

6. Revisit video taxonomy only after frame coverage is fixed:
   - The manifest currently drops many rows without usable frames.
   - Need more frame-aligned rows per label.
   - Especially rare labels need real video examples.

## Paper/Research Summary

The best result is a supervised RoBERTa multi-label classifier trained on metadata text fields from the dataset. It significantly outperforms zero-shot LLM baselines.

Core research comparison:

```text
Supervised calibrated RoBERTa:
  micro F1 = 0.8754
  micro precision = 0.9290
  rare false positives = 0

Zero-shot Llama 3.3 70B:
  micro F1 = 0.5846
  micro precision = 0.6291
  rare false positives = 4

Zero-shot Llama 3.1 8B:
  micro F1 = 0.5619
  micro precision = 0.5924
  rare false positives = 11
```

The main conclusion:

```text
Dataset-specific supervised taxonomy training plus calibrated thresholds is much stronger than zero-shot LLM classification for this multi-label moderation taxonomy.
```

## Final Reminder For Future Sessions

Do not claim the taxonomy model classifies raw video directly. The runtime pipeline first creates or receives metadata text, then the taxonomy classifier predicts labels from that metadata. The vision model contributes hate/severity signals, and the experimental video taxonomy head may be used only as a candidate signal until its overprediction is fixed.

If the user asks for the current best model, answer:

```text
Use checkpoints\transformer_multilabel_classifier_feedback\best with
reports\transformer_multilabel_feedback_calibration\thresholds.json,
and run inference through inference_moderation_system.py with --no-use-video-taxonomy.
```
