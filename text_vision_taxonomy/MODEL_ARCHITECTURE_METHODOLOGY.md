# Multi-Label Taxonomy Moderation System

Generated: 2026-05-20

This document explains the current taxonomy system in `D:\text_vision_taxonomy`: what the model is, what it takes as input, how those inputs are created, how the model is trained, how thresholds and metrics are calculated, and what the current measured numbers are.

Architecture diagrams:

```text
ARCHITECTURE_DIAGRAM.md
```

The practical conclusion is:

- The best current taxonomy model is the text-based RoBERTa multi-label transformer.
- The video model should currently be used for hate/non-hate and severity signals, not as the main taxonomy labeler.
- Video taxonomy training exists, but it is not production-ready because frame coverage is incomplete and validation behavior is unstable.

## 1. System Goal

The system predicts multiple taxonomy labels for a moderation sample.

Labels:

| Label | Meaning |
|---|---|
| `hate_speech` | Explicit hateful or abusive language targeting a group/person |
| `discrimination` | Protected-class or identity-based discriminatory content |
| `contextual_hate` | Contextual, coded, propaganda-like, or indirect hate |
| `threat` | Direct threat or threat-like intent |
| `violence` | Violent action, violent intent, or violent scene/content |
| `fear` | Fear, intimidation, panic, terror, fear-mongering |
| `sexual` | Sexual or explicit content |
| `illegal` | Illegal, weapon, drug, crime, abuse, fraud, trafficking, etc. |
| `online_harm` | Online harassment, social media abuse, doxxing, cyberbullying |

`neutral` is not trained as a model output. In this system, neutral means all nine taxonomy labels are zero.

## 2. Current Recommended Production Architecture

The production path should be:

```text
raw video
  -> frame sampling
  -> ASR transcript
  -> OCR text
  -> optional metadata text
  -> text transformer taxonomy classifier
  -> calibrated taxonomy thresholds
  -> rare-label rules
  -> base-model gating
  -> final labels / candidates / suppressed labels

raw video frames
  -> vision base model
  -> hate/non-hate probability + severity

transcript / OCR / metadata text
  -> text base model
  -> hate/non-hate probability + severity

final output
  -> hate decision
  -> severity
  -> taxonomy labels
  -> probabilities
  -> evidence
```

The taxonomy labels should currently come from the text transformer taxonomy model. The vision model is still useful, but mainly as a supporting base signal:

- `vision_hate_prob`
- `vision_severity`

The text base model is also a supporting signal:

- `text_hate_prob`
- `text_severity`

These base models do not replace the taxonomy classifier. They help suppress or boost taxonomy outputs.

## 3. Current Runtime Command

For a raw video input, the recommended current command is:

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

`--no-use-video-taxonomy` is intentional right now. It keeps the taxonomy labels text-based while still allowing the vision base model to contribute hate/severity signals.

## 4. Inputs

### 4.1 Training Dataset

Primary dataset:

```text
D:\hvc\datasets\action\archieve\final_dataset.csv
```

Current loaded row count:

| Split | Rows |
|---|---:|
| train | 750 |
| val | 156 |
| test | 154 |
| total | 1060 |

The split is source-video based using `File Name`. This avoids leaking clips from the same source video into multiple splits.

### 4.2 Training Text Fields

The text taxonomy model consumes these fields:

| Field | Used for taxonomy text |
|---|---|
| `Scene` | yes |
| `Action` | yes |
| `Subcategories` | yes |
| `ParentLabels` | yes |
| `action_clean` | yes |

The text input is built like this:

```text
[SCENE]
{Scene}

[ACTION]
{Action}

[SUBCATEGORIES]
{Subcategories}

[PARENT_LABELS]
{ParentLabels}

[ACTION_CLEAN]
{action_clean}
```

At runtime, extra generated fields can also be appended:

```text
[TRANSCRIPT]
{transcript}

[OCR_TEXT]
{ocr_text}
```

### 4.3 Runtime Inputs

The runtime can accept either:

1. Manual metadata:

```text
scene
action
action_clean
subcategories
parent_labels
transcript
ocr_text
```

2. Raw video:

```text
--video path\to\video.mp4
```

3. Frame directory:

```text
--frame-dir path\to\sampled_frames
```

For raw video, the system creates intermediate text and frame inputs automatically.

## 5. How Runtime Inputs Are Created

The file `video_metadata_generator.py` creates runtime metadata.

### 5.1 Frame Sampling

If a raw video is provided:

- OpenCV samples frames from the video.
- Frames are saved under `reports/runtime_metadata/frames/...`.
- Default metadata frame count is `8`.

If a frame directory is provided:

- The system samples frames from the existing directory.

These sampled frames are used for:

- OCR
- optional visual captioning
- vision base model inference

### 5.2 ASR Transcript

If a raw video is provided and ASR is enabled or auto-enabled:

- `faster_whisper` is used if available.
- The current ASR model path is the local runtime dependency, not a trained taxonomy model.
- If ASR is unavailable, the system does not crash. It returns partial metadata.

The transcript becomes one of the strongest taxonomy inputs because many taxonomy labels are semantic.

### 5.3 OCR Text

OCR runs on sampled frames if OCR dependencies are available.

If Tesseract/Pillow/pytesseract are missing or OCR returns nothing:

- runtime continues
- `ocr_text` is empty
- evidence explains OCR was unavailable or empty

### 5.4 VLM / Captioning

VLM captioning is optional and disabled by default in the current production path.

Reason:

- The text taxonomy classifier does not need a VLM to run.
- Local VLM availability was unstable.
- ASR and OCR are usually more reliable for this dataset.

If `--enable-vlm` is passed and the model is available, the VLM can create a visual scene description.

### 5.5 Rule-Derived Metadata

Rule-derived metadata is disabled by default.

When disabled, the generator does not create `Action`, `Subcategories`, or `ParentLabels` from keyword rules. This is deliberate: taxonomy labels should come from the trained model, not from hardcoded label guessing.

Rules can be enabled with:

```powershell
--enable-rule-metadata
```

But that should be treated as a fallback/helper mode, not the main model.

## 6. Model Components

### 6.1 Text Taxonomy Model - Main Model

Current best taxonomy model:

```text
checkpoints\transformer_multilabel_classifier_feedback\best
```

Architecture:

- HuggingFace `AutoModelForSequenceClassification`
- base model: `roberta-base`
- `num_labels = 9`
- `problem_type = multi_label_classification`
- output: 9 independent logits
- activation: sigmoid per label

Output probabilities:

```text
prob_hate_speech
prob_discrimination
prob_contextual_hate
prob_threat
prob_violence
prob_fear
prob_sexual
prob_illegal
prob_online_harm
```

This is not a softmax classifier. Multiple labels can be true at the same time.

### 6.2 Text Base Model - Hate/Severity Signal

Checkpoint:

```text
D:\text_vision_taxonomy\roberta_base_finetuned_dualhead.pt
```

Input:

```text
transcript + OCR + scene/action metadata
```

Output:

```json
{
  "prob_hate": 0.0,
  "severity": 0,
  "severity_probs": [...]
}
```

This model answers:

- is this hateful?
- how severe is it?

It does not output the nine taxonomy labels.

### 6.3 Vision Base Model - Hate/Severity Signal

Checkpoint:

```text
checkpoints\best_frame_text_vision_relabel_all_splits.pt
```

Input:

```text
sampled video frames
```

Output:

```json
{
  "prob_hate": 0.0,
  "severity": 0,
  "severity_probs": [...]
}
```

This model is useful for visual hate/severity evidence. It does not currently beat the text taxonomy model for taxonomy labels.

### 6.4 Video Taxonomy Model - Experimental

Checkpoint if trained:

```text
checkpoints\best_taxonomy_head.pt
```

This path trains a frame-based deep model with temporal aggregation. It is currently not recommended as the final taxonomy source.

Current issue:

| Video manifest item | Count |
|---|---:|
| manifest rows | 1045 |
| usable frame rows | 490 |
| dropped rows without usable frames | 555 |
| train usable frame rows | 331 |
| val usable frame rows | 99 |
| test usable frame rows | 60 |

Because over half the manifest rows do not have usable frame folders, the video taxonomy model sees much less training data. Rare labels become especially unstable.

## 7. Label Distribution

The nine trained taxonomy labels in `final_dataset.csv` currently load as:

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

Zero-label neutral rows:

| Split | Neutral rows |
|---|---:|
| train | 271 |
| val | 67 |
| test | 48 |
| total | 386 |

Note: neutral is not a trained output label. A row is neutral for this taxonomy task when all nine taxonomy labels are zero.

## 8. Training Methodology

### 8.1 Base Text Transformer Training

Training script:

```text
train_transformer_multilabel_classifier.py
```

Feedback retraining wrapper:

```text
retrain_taxonomy_with_feedback.py
```

The current best model was trained using:

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

Important training details:

| Setting | Value |
|---|---|
| model | `roberta-base` |
| max sequence length | 256 by default |
| labels | 9 |
| loss | focal loss |
| focal gamma | 2.0 |
| class imbalance | capped positive weight |
| max positive weight | 5 |
| optimizer | AdamW |
| learning rate | 2e-5 default |
| early stopping | patience 2 |
| split method | source-video split by `File Name` |

### 8.2 Feedback Data

Feedback retraining created:

```text
data\taxonomy_feedback_training.csv
```

Current feedback dataset:

| Source | Rows |
|---|---:|
| original | 1060 |
| pseudo_label | 1 |
| total | 1061 |

Synthetic/pseudo rows are used for training only. Evaluation excludes synthetic rows.

### 8.3 Loss Function

The model uses binary cross entropy per label, optionally converted into focal loss.

For BCE:

```text
BCE = -y * log(p) - (1 - y) * log(1 - p)
```

For focal loss:

```text
focal = (1 - p_t)^gamma * BCE
```

Where:

```text
p_t = p when y = 1
p_t = 1 - p when y = 0
gamma = 2
```

Focal loss reduces the effect of easy examples and focuses training on harder labels.

### 8.4 Class Imbalance

Rare labels are a major problem:

| Rare label | Count |
|---|---:|
| threat | 17 |
| illegal | 8 |
| online_harm | 14 |

The trainer supports positive class weights, but caps them:

```text
pos_weight = min(negative_count / positive_count, max_pos_weight)
max_pos_weight = 5
```

This avoids extreme weights that make the model predict rare labels everywhere.

## 9. Threshold Calibration Methodology

The model outputs probabilities. A probability becomes a final label only if it crosses the calibrated threshold for that label.

Calibration script:

```text
calibrate_multilabel_thresholds.py
```

Threshold report:

```text
reports\transformer_multilabel_feedback_calibration\threshold_report.csv
```

Production thresholds:

```text
reports\transformer_multilabel_feedback_calibration\thresholds.json
```

Calibration uses validation predictions and computes several threshold families:

| Threshold mode | Meaning |
|---|---|
| `max_f1` | threshold that maximizes F1 for the label |
| `top_k_by_prevalence` | threshold that makes predicted count close to true support |
| `precision_60` | threshold satisfying precision >= 0.60 if possible |
| `precision_70` | threshold satisfying precision >= 0.70 if possible |
| `precision_80` | threshold satisfying precision >= 0.80 if possible |
| `precision_90` | threshold satisfying precision >= 0.90 if possible |
| `production` | precision-oriented threshold used at runtime |

Production threshold rules:

- For normal labels, prefer precision-oriented thresholds and prevalence control.
- For rare labels, use very high thresholds.
- Apply minimum floors:

| Label | Floor |
|---|---:|
| threat | 0.90 |
| illegal | 0.95 |
| online_harm | 0.90 |
| sexual | 0.75 |
| fear | 0.75 |

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

## 10. Rare Label Logic

Rare labels are:

```text
threat, illegal, online_harm
```

The system does not trust rare-label probability alone unless it is very high or supported by evidence.

### 10.1 Threat

Threat requires at least one of:

- explicit threat wording
- high violence probability
- high severity
- extremely high threat probability

If not, it becomes a candidate or suppressed label instead of a final label.

### 10.2 Illegal

Illegal requires explicit illegal/crime/weapon/drug/abuse/fraud-style evidence in the text.

It should not be predicted purely from visual uncertainty.

### 10.3 Online Harm

Online harm requires online/social/chat/comment/post/harassment/doxxing context.

## 11. Base Model Gating

After taxonomy probabilities and rare-label rules, base hate/severity signals are applied.

Base signals:

```json
{
  "text_hate_prob": null,
  "vision_hate_prob": null,
  "text_severity": null,
  "vision_severity": null
}
```

Rules:

1. If both text and vision hate probabilities are available and both are below `0.35`, suppress hate-specific labels:

```text
hate_speech
discrimination
contextual_hate
threat
illegal
online_harm
```

2. Do not suppress these labels just because hate probability is low:

```text
violence
sexual
fear
```

3. If text or vision hate probability is high enough, currently `>= 0.70`, near-threshold hate labels can be boosted into final labels or candidates.

4. If severity is high, currently `>= 3`, violence and threat can be candidates. Threat still needs text evidence or very high taxonomy probability.

## 12. Metrics: How They Are Calculated

### 12.1 Per-Label Counts

For each label:

```text
TP = true positives
FP = false positives
FN = false negatives
```

Then:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

If denominator is zero, the metric is set to `0.0`.

### 12.2 Micro Metrics

Micro metrics pool all label decisions together before calculating precision/recall/F1.

This means every individual label decision across all rows is counted equally.

Micro precision:

```text
sum(TP_all_labels) / (sum(TP_all_labels) + sum(FP_all_labels))
```

Micro recall:

```text
sum(TP_all_labels) / (sum(TP_all_labels) + sum(FN_all_labels))
```

Micro F1:

```text
2 * micro_precision * micro_recall / (micro_precision + micro_recall)
```

Micro metrics are useful when you care about total correct label decisions.

### 12.3 Macro Metrics

Macro metrics calculate precision/recall/F1 per label, then average labels equally.

Macro F1 is hurt heavily by rare labels because `threat`, `illegal`, and `online_harm` often score zero under conservative thresholds.

### 12.4 Neutral False Positive Rate

Neutral rows are rows with no true taxonomy labels.

Neutral false positive rate:

```text
neutral rows where model predicted at least one label / total neutral rows
```

On the calibrated test split:

```text
0.0625 = 3 / 48 neutral test rows
```

### 12.5 Rare False Positive Count

Rare labels:

```text
threat, illegal, online_harm
```

Rare false positive count:

```text
number of false positive predictions across the rare labels
```

This is important because rare labels should not be predicted hundreds of times when true support is tiny.

## 13. Current Text Transformer Results

Best current model:

```text
checkpoints\transformer_multilabel_classifier_feedback\best
```

Reports:

```text
reports\transformer_multilabel_classifier_feedback
reports\transformer_multilabel_feedback_calibration
reports\transformer_multilabel_feedback_eval
```

### 13.1 Validation Calibration Result

Validation rows:

```text
156
```

Production-threshold validation metrics:

| Metric | Value |
|---|---:|
| micro precision | 0.9496 |
| micro recall | 0.8433 |
| micro F1 | 0.8933 |
| macro precision | 0.6308 |
| macro recall | 0.5768 |
| macro F1 | 0.5934 |
| neutral false positive rate | 0.0149 |
| rare false positive count | 0 |
| rare predicted positive count | 0 |
| rare true positive count | 5 |

### 13.2 Test Result - Transformer Only vs Calibrated

Test rows:

```text
154
```

| System | Micro precision | Micro recall | Micro F1 | Macro precision | Macro recall | Macro F1 | Neutral FP rate | Rare FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| transformer only, threshold 0.5 | 0.8703 | 0.9253 | 0.8969 | 0.7758 | 0.7503 | 0.7531 | 0.1042 | 4 |
| transformer calibrated | 0.9290 | 0.8276 | 0.8754 | 0.6247 | 0.5633 | 0.5885 | 0.0625 | 0 |

Interpretation:

- Raw threshold `0.5` has higher recall and F1, but creates rare-label false positives.
- Calibrated thresholds reduce recall, but improve precision and remove rare false positives.
- For production moderation labels, the calibrated model is safer.

### 13.3 Test Per-Label Metrics - Calibrated

| Label | Threshold | True count | Predicted count | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| hate_speech | 0.7644 | 37 | 35 | 0.9714 | 0.9189 | 0.9444 |
| discrimination | 0.7842 | 35 | 27 | 1.0000 | 0.7714 | 0.8710 |
| contextual_hate | 0.8324 | 25 | 19 | 1.0000 | 0.7600 | 0.8636 |
| threat | 0.9900 | 3 | 0 | 0.0000 | 0.0000 | 0.0000 |
| violence | 0.3850 | 31 | 37 | 0.7838 | 0.9355 | 0.8529 |
| fear | 0.7500 | 19 | 15 | 0.8667 | 0.6842 | 0.7647 |
| sexual | 0.9358 | 22 | 22 | 1.0000 | 1.0000 | 1.0000 |
| illegal | 0.9900 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| online_harm | 0.9000 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 |

Interpretation:

- Strong labels: `hate_speech`, `discrimination`, `contextual_hate`, `violence`, `sexual`.
- Good but recall-limited: `fear`.
- Intentionally conservative: `threat`, `illegal`, `online_harm`.

For `threat`, `illegal`, and `online_harm`, this is expected because the dataset has too few examples. The system currently chooses zero rare false positives over recall.

### 13.4 Test Per-Label Metrics - Raw 0.5 Threshold

| Label | True count | Predicted count | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| hate_speech | 37 | 39 | 0.8974 | 0.9459 | 0.9211 |
| discrimination | 35 | 35 | 0.9429 | 0.9429 | 0.9429 |
| contextual_hate | 25 | 26 | 0.8846 | 0.9200 | 0.9020 |
| threat | 3 | 3 | 0.6667 | 0.6667 | 0.6667 |
| violence | 31 | 35 | 0.8286 | 0.9355 | 0.8788 |
| fear | 19 | 21 | 0.7619 | 0.8421 | 0.8000 |
| sexual | 22 | 22 | 1.0000 | 1.0000 | 1.0000 |
| illegal | 0 | 3 | 0.0000 | 0.0000 | 0.0000 |
| online_harm | 2 | 1 | 1.0000 | 0.5000 | 0.6667 |

This looks stronger in F1, but it predicted `illegal` three times when there were zero true illegal rows in test. That is why production calibration is preferred.

## 14. Comparison With Older TF-IDF Text Baseline

Older classical text baseline:

```text
tfidf_word_char_linearsvc_calibrated_ovr
```

Test comparison:

| Model | Micro precision | Micro recall | Micro F1 | Macro F1 | Neutral FP rate | Rare FP |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF calibrated | 0.8929 | 0.7184 | 0.7962 | 0.5171 | 0.1042 | 0 |
| RoBERTa calibrated | 0.9290 | 0.8276 | 0.8754 | 0.5885 | 0.0625 | 0 |

RoBERTa is better on:

- precision
- recall
- F1
- macro F1
- neutral false positive rate

Therefore, the text transformer should be the main taxonomy model.

## 15. What Happens in Final Inference

The runtime output has this structure:

```json
{
  "metadata": {
    "scene": "...",
    "action": "...",
    "action_clean": "...",
    "subcategories": "...",
    "parent_labels": "...",
    "transcript": "...",
    "ocr_text": "...",
    "available_fields": {}
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
    "source": "transformer_multilabel_classifier"
  },
  "final": {
    "is_hate": true,
    "prob_hate": 0.0,
    "severity": 0,
    "taxonomy_labels": []
  },
  "evidence": []
}
```

Important output fields:

| Field | Meaning |
|---|---|
| `taxonomy_raw_labels` | labels whose probability crosses threshold before final gating |
| `taxonomy_final_labels` | final high-confidence labels |
| `candidate_labels` | near-threshold or evidence-limited labels for review |
| `suppressed_labels` | labels blocked by rare-label rules or base-model gating |
| `probs` | probability for every taxonomy label |
| `thresholds_used` | calibrated threshold for every label |
| `base_model_signals` | text/vision hate and severity signals |

## 16. Why Text-Based Taxonomy Is Better Right Now

The taxonomy labels are mostly semantic. Many examples are determined by spoken language, transcript content, OCR text, and metadata descriptions.

Examples:

- `hate_speech` depends heavily on what is said.
- `discrimination` depends on identity/group references.
- `contextual_hate` depends on semantic framing.
- `threat` depends on explicit threatening language.
- `online_harm` depends on online/chat/post context.

Vision alone often cannot infer these labels reliably.

The video taxonomy model also has a data coverage problem:

- only 490 usable frame rows in the current manifest
- 555 rows dropped because frame data is missing
- validation split for video has only 99 usable rows
- rare labels are almost absent in frame-backed validation

So the current production choice is:

```text
taxonomy labels = text transformer
hate/severity support = text base + vision base
```

## 17. Known Limitations

1. Rare labels have too little data.

`threat`, `illegal`, and `online_harm` cannot be expected to learn robustly from the current dataset without more labeled examples or carefully reviewed augmentation.

2. Current production thresholds favor precision over recall.

This means rare labels may be missed. That is intentional for now because false positives on rare serious labels are costly.

3. Raw video taxonomy depends on ASR/OCR quality.

If ASR fails and OCR is empty, taxonomy text may be weak. In that case, final taxonomy confidence should be treated as lower.

4. Video taxonomy is not final.

The deep video taxonomy model exists, but its training data coverage is currently insufficient.

5. The `neutral` CSV column is not used as a trained output.

Neutral is derived from all taxonomy labels being zero.

## 18. Recommended Next Steps

### 18.1 Production Now

Use:

```text
text transformer taxonomy + text base hate/severity + vision base hate/severity
```

Disable video taxonomy:

```powershell
--no-use-video-taxonomy
```

### 18.2 Improve Rare Labels

Add more true examples for:

```text
threat
illegal
online_harm
```

Target at least 50 to 100 high-quality examples per rare label before expecting stable recall.

### 18.3 Improve Raw Video Inputs

Improve:

- ASR reliability
- OCR/Tesseract setup
- optional VLM/captioning if cached and stable
- metadata quality

The text taxonomy model improves when the generated transcript/scene text improves.

### 18.4 Revisit Video Taxonomy Later

Before using video taxonomy in production:

1. Fix missing frame coverage.
2. Ensure every dataset row has a valid frame directory or video path.
3. Retrain video taxonomy.
4. Calibrate video taxonomy separately.
5. Compare video taxonomy against text transformer on the same test split.
6. Only include video taxonomy in final labels if it improves precision/F1 without creating rare-label false positives.

## 19. Source Artifacts

Main scripts:

| Purpose | File |
|---|---|
| train text transformer taxonomy | `train_transformer_multilabel_classifier.py` |
| feedback retraining | `retrain_taxonomy_with_feedback.py` |
| taxonomy metrics/calibration utilities | `multilabel_taxonomy_utils.py` |
| final runtime inference | `inference_moderation_system.py` |
| video metadata generation | `video_metadata_generator.py` |
| rare label rules | `rare_label_rules.py` |
| video taxonomy training | `train_taxonomy_head.py` |

Main checkpoints:

| Purpose | Path |
|---|---|
| text taxonomy transformer | `checkpoints\transformer_multilabel_classifier_feedback\best` |
| text hate/severity base | `D:\text_vision_taxonomy\roberta_base_finetuned_dualhead.pt` |
| vision hate/severity base | `checkpoints\best_frame_text_vision_relabel_all_splits.pt` |
| experimental video taxonomy | `checkpoints\best_taxonomy_head.pt` |

Main reports:

| Purpose | Path |
|---|---|
| transformer feedback training report | `reports\transformer_multilabel_classifier_feedback` |
| transformer calibration report | `reports\transformer_multilabel_feedback_calibration` |
| transformer test evaluation | `reports\transformer_multilabel_feedback_eval` |
| TF-IDF baseline report | `reports\text_multilabel_classifier` |

## 20. Final Summary

The current strongest system is a text-first deep learning taxonomy system.

It uses:

- RoBERTa multi-label taxonomy classifier for final taxonomy labels
- calibrated production thresholds
- rare-label conservative logic
- text hate/severity base model for support
- vision hate/severity base model for support
- raw video preprocessing through ASR/OCR/frame sampling

Current best test result:

```text
micro precision: 0.9290
micro recall:    0.8276
micro F1:        0.8754
rare FP count:   0
neutral FP rate: 0.0625
```

This is the model path that should be used for now.
