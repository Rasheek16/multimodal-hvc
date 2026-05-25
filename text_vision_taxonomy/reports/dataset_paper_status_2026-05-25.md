# Dataset And Paper Status Report

Date: 2026-05-25

Workspace: `D:\m-hvc`

Dataset root used for this report: `D:\hvc\datasets`

This report answers the dataset, paper-status, and technical questions for the
current `text_classification`, `text_vision`, and `text_vision_taxonomy`
artifacts. It separates the text-only base model, the vision+text binary base
model, and the taxonomy dataset because they do not all use the same data.

## Executive Summary

| Component | Main dataset | Unit | Verified size | Role |
| --- | --- | --- | ---: | --- |
| `text_classification` | `D:\hvc\datasets\text-dataset\ALL_TEXTS_LABELED.csv` | text row | 5,071,672 rows | Text-only base hate/severity model |
| `text_vision` | `D:\hvc\datasets\old-dataset` | source video / clips | 3,429 raw videos; 2,002 labeled rows; 3,281 videos in diversified manifest | Vision+text binary hate model guided by the text model |
| `text_vision_taxonomy` | `D:\hvc\datasets\action\archieve\final_dataset.csv` | timestamped segment row | 1,060 raw rows; 1,045 valid manifest rows | 9-label moderation taxonomy |

## 1. What Datasets Were Combined For The Final 1,060-Sample Dataset?

The 1,060-sample taxonomy dataset is not the same object as the old
HateMM/H-VENOM/MultiHateClip binary video dataset.

For the taxonomy work, the final 1,060 rows come from the action/scene
annotation files under `D:\hvc\datasets\action\archieve`:

| Taxonomy source file | Rows | Notes |
| --- | ---: | --- |
| `combined_output.xlsx` | 550 | Action, subcategory, scene, timestamp, and file-name annotations |
| `merged_hatevideos.xlsx` | 510 | Action, subcategory, scene, timestamp, and file-name annotations |
| `final_dataset.csv` | 1,060 | Merged taxonomy/action dataset used by the current pipeline |

The generated taxonomy manifest reports:

| Manifest value | Count |
| --- | ---: |
| Raw input rows | 1,060 |
| Valid manifest rows | 1,045 |
| Dropped rows | 15 |
| Unique source videos | 670 |
| Train rows | 735 |
| Validation rows | 156 |
| Test rows | 154 |

The base binary vision+text model uses a different dataset family:

| Old binary video source | Labeled rows |
| --- | ---: |
| HateMM-filename | 1,083 |
| MultiHateClip | 821 |
| H-VENOM | 98 |
| Total labeled rows | 2,002 |

The broader old video media pool contains 3,429 videos. The text-vision
training documentation reports a filtered/materialized diversified manifest of
3,281 videos and 9,845 clips.

## 2. "Hatem" Clarification

I found evidence for `HateMM`, not for a person named `Hatem`.

Current answer:

- `HateMM` is a named dataset/source used in the old binary video dataset.
- The old binary source column stores it as `HateMM-filename`.
- I did not find repo evidence that "Hatem" is a collaborator who collected or
  labeled 500 videos.
- If a collaborator named Hatem exists, that is outside the current repo and
  should not be claimed in the paper unless confirmed manually.

## 3. Are The 1,060 Samples Videos Or Frame-Level Rows?

The 1,060 taxonomy samples are timestamped segment/clip annotation rows. They
are not frame-level rows and they are not one-row-per-video records.

Each row contains fields such as:

- `Start Time`
- `End Time`
- `Action`
- `Subcategories`
- `Scene`
- `File Name`
- taxonomy label fields such as `final_labels`

Multiple rows can come from the same video. The current manifest summary gives
670 unique source videos after validation.

Frame data is derived later for model training/evaluation. In the taxonomy
manifest, only 1,045 of the 1,060 rows are valid after dropping 15 rows with
missing timestamps/file names or invalid time ranges.

## 4. Who Annotated The 9 Taxonomy Labels?

The 9 taxonomy labels are derived from the action/scene annotation dataset
fields, especially `Action`, `Subcategories`, `Scene`, `ParentLabels`, and
`final_labels`.

The 9 classifier labels are:

| Taxonomy label | Valid manifest positives |
| --- | ---: |
| `hate_speech` | 210 |
| `discrimination` | 208 |
| `contextual_hate` | 151 |
| `threat` | 17 |
| `violence` | 251 |
| `fear` | 115 |
| `sexual` | 101 |
| `illegal` | 8 |
| `online_harm` | 14 |

The repo evidence supports manual/human-authored annotation fields as the
ground-truth source. I did not find evidence that LLMs produced the ground-truth
taxonomy labels.

LLMs were used later as zero-shot comparison baselines. They should not be
described as the annotators for the training labels.

I did not find an inter-annotator agreement report or agreement statistic in
the current artifacts.

## 5. Paper Status

There is not a full paper manuscript in the repo.

Existing paper/report-style artifacts include:

- `text_vision_taxonomy\RESEARCH_ABSTRACT.md`
- `text_vision_taxonomy\MODEL_ARCHITECTURE_METHODOLOGY.md`
- `text_vision_taxonomy\ARCHITECTURE_DIAGRAM.md`
- `text_vision_taxonomy\CODEX_TAXONOMY_HANDOFF.md`
- `text_vision_taxonomy\FULL_PROJECT_HANDOFF_2026-05-21.md`
- metric reports under `text_vision_taxonomy\reports`

Current status: abstract, methodology, architecture, and experiment reports
exist; a complete draft paper was not found.

## 6. Co-Authors / Group Project Status

The repo does not contain a confirmed author list.

The dataset/source names mention Amber and Pratush in the action/scene
annotation context. They should be treated as dataset/source names unless the
project team confirms they are co-authors.

No repo evidence confirms a co-author named Hatem.

## 7. How Was `best_frame_text_vision_relabel_all_splits.pt` Trained?

`best_frame_text_vision_relabel_all_splits.pt` is the current vision+text binary
base checkpoint used by the taxonomy/moderation runtime for hate/severity
signals.

It was trained in the text-guided frame/vision pipeline:

- visual input: frame/clip tensors from the diversified video dataset
- binary target: real video-level hate/non-hate label
- text guidance: frozen RoBERTa text teacher outputs when transcripts exist
- text signals: cached text hate probability, severity probabilities, and text
  embedding
- transcript handling: human transcript, ASR transcript, or missing transcript
- evaluation: clip predictions are averaged by `source_video_id` to produce
  video-level metrics

The source dataset for this model is the old binary video dataset family:
HateMM, H-VENOM, and MultiHateClip, with the pretrained text model used as
guidance. The broader pool is about 3,500 videos by project description; the
verified local old video folder has 3,429 videos, and the filtered diversified
manifest reports 3,281 videos.

It is related to the earlier video-transformer work, but the current artifact is
the text-guided frame/vision checkpoint used by `text_vision` and
`text_vision_taxonomy`, not the standalone notebook-only checkpoint.

## 8. Best Binary Hate / Non-Hate Results

### Text-only base model

The text-only model uses `ALL_TEXTS_LABELED.csv`.

The notebook maps:

| Raw label | Severity | Binary |
| --- | ---: | --- |
| `label_0` | 0 | non-hate / appropriate |
| `label_1` | 1 | hate-side / inappropriate |
| `label_2` | 2 | hate-side / offensive |
| `label_3` | 3 | hate-side / violent |

The training notebook used an 80,000-row balanced subset with a 64,000/16,000
train/validation split. Best logged validation binary result:

| Metric | Value |
| --- | ---: |
| Binary accuracy | 0.9420 |
| Binary F1 | 0.9616 |

Best logged severity macro F1 appears later:

| Metric | Value |
| --- | ---: |
| Severity accuracy | 0.9079 |
| Severity macro F1 | 0.9078 |

### Vision+text binary base model

Final relabeled all-splits checkpoint, test split, all groups:

| Metric | Value |
| --- | ---: |
| Threshold | 0.31 |
| Accuracy | 0.8438 |
| F1 | 0.8188 |
| Precision | 0.7982 |
| Recall | 0.8406 |
| ROC-AUC | 0.9304 |
| PR-AUC | 0.9090 |
| Test videos | 493 |

Baseline comparison from the same report:

| Model | Accuracy | F1 | Precision | Recall | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Text-guided frame vision | 0.8438 | 0.8188 | 0.7982 | 0.8406 | 0.9304 | 0.9090 |
| Existing vision-only baseline | 0.7221 | 0.6730 | 0.6438 | 0.7050 | 0.7981 | 0.7206 |

## 9. Public Benchmark Testing

I did not find evidence of a separate held-out public benchmark evaluation such
as "train on our combined data and test only on HateMM".

The old binary video dataset includes HateMM-derived samples, so HateMM is part
of the combined training/evaluation data. The current reported results appear
to be from local train/validation/test splits of the combined datasets.

The taxonomy results are on the source-video split of `final_dataset.csv`, not a
standalone public benchmark split.

## Taxonomy Model Results For Paper Context

The current taxonomy recommendation is text-first: use the supervised RoBERTa
multi-label taxonomy classifier with calibrated thresholds. Vision is useful for
base hate/severity gating and calibration, while the experimental video taxonomy
head is not the main final taxonomy source.

Held-out test comparison:

| System | Rows | Micro precision | Micro recall | Micro F1 | Macro F1 | Rare false positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-shot Llama 3.3 70B | 154 | 0.6291 | 0.5460 | 0.5846 | 0.3534 | 4 |
| Transformer only | 154 | 0.8703 | 0.9253 | 0.8969 | 0.7531 | 4 |
| Transformer calibrated | 154 | 0.9290 | 0.8276 | 0.8754 | 0.5885 | 0 |

Validation/calibration report:

| Metric | Value |
| --- | ---: |
| Rows | 156 |
| Micro precision | 0.9512 |
| Micro recall | 0.8731 |
| Micro F1 | 0.9105 |
| Macro F1 | 0.6200 |
| Neutral false positive rate | 0.0149 |
| Rare false positive count | 0 |

## Evidence Files Used

Dataset and metadata files:

- `D:\hvc\datasets\text-dataset\ALL_TEXTS_LABELED.csv`
- `D:\hvc\datasets\old-dataset\final_video_labels.csv`
- `D:\hvc\datasets\old-dataset\text_dataset.csv`
- `D:\hvc\datasets\action\archieve\final_dataset.csv`
- `D:\hvc\DATASET_META.md`
- `D:\hvc\PROJECT_PLAN.md`

Code/report artifacts:

- `text_classification\model.ipynb`
- `text_classification\artifacts\text_multitask_training.log`
- `text_vision_taxonomy\reports\multilabel_manifest_summary.json`
- `text_vision_taxonomy\reports\frame_text_vision_relabel_all_splits\metrics.csv`
- `text_vision_taxonomy\reports\frame_text_vision_relabel_all_splits\model_comparison.csv`
- `text_vision_taxonomy\reports\transformer_multilabel_classifier\metrics.json`
- `text_vision_taxonomy\reports\zero_shot_llm_eval\groq_llama33_70b_versatile\comparison_metrics.csv`
- `text_vision_taxonomy\FULL_PROJECT_HANDOFF_2026-05-21.md`
- `text_vision_taxonomy\MODEL_ARCHITECTURE_METHODOLOGY.md`

## Paper Wording Recommendations

Use wording like this:

> The text-only base model was trained on a large text moderation corpus of
> 5.07M rows, using a balanced 80K-row training subset for the current
> checkpoint. The vision+text binary model was trained on a combined video
> corpus sourced from HateMM, H-VENOM, and MultiHateClip, with RoBERTa text
> teacher guidance where transcript or ASR text was available. The taxonomy
> classifier was trained separately on 1,060 timestamped action/scene segment
> annotations from the Amber + Pratush/action dataset, producing 1,045 valid
> manifest rows after validation.

Avoid wording that claims:

- "Hatem collected 500 videos" unless manually confirmed outside the repo.
- LLMs annotated the 9 ground-truth taxonomy labels.
- Inter-annotator agreement was measured.
- The reported numbers are from an external held-out HateMM benchmark.
