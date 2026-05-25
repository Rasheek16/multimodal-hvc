# Research Abstract

## Title

Multi-Label Taxonomy Classification for Video Moderation Using Text-Derived Semantic Signals and Base-Model Calibration

## Abstract

**Background:** Fine-grained video moderation requires more than binary hate/non-hate classification, because harmful content may involve overlapping categories such as hate speech, discrimination, contextual hate, violence, fear, sexual content, threats, illegal activity, and online harm. This work develops a multi-label taxonomy classifier while retaining existing text and vision base models for hate probability and severity estimation.

**Methods:** The system uses a text-first taxonomy architecture. Raw videos are converted into semantic inputs through frame sampling, automatic speech recognition, optical character recognition, and optional metadata fields. These signals are formatted into structured taxonomy text and classified using a RoBERTa-based multi-label transformer with nine independent sigmoid outputs. The model is trained on `1,060` rows from `final_dataset.csv` using a source-video split of `750` training, `156` validation, and `154` test rows. Per-label thresholds are calibrated on validation predictions to prioritize precision and control prevalence. Rare labels, specifically `threat`, `illegal`, and `online_harm`, use high thresholds and evidence-aware suppression. Existing text and vision hate/severity models are used only as gating and calibration signals.

**Results:** On the held-out test split, the calibrated RoBERTa taxonomy classifier achieved micro precision `0.9290`, micro recall `0.8276`, and micro F1 `0.8754`. Macro F1 was `0.5885`, reflecting rare-label difficulty. The neutral false positive rate was `0.0625`, and the rare-label false positive count was `0`. Strong per-label performance was observed for `hate_speech` with F1 `0.9444`, `discrimination` with F1 `0.8710`, `contextual_hate` with F1 `0.8636`, `violence` with F1 `0.8529`, and `sexual` with F1 `1.0000`.

**Conclusion:** Text-derived semantic modeling is currently the most reliable taxonomy source for this dataset. Vision remains useful for hate/severity calibration, but the experimental video taxonomy path is not production-ready due to incomplete frame coverage and unstable rare-label behavior.
