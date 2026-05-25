# Severity-Boosted Text-Guided Vision Classifier

This package trains a vision-first multimodal hate media classifier under `multimodal-hvc/text_vision`.
The frozen RoBERTa dual-head text classifier is used as a teacher for clips that have a transcript.
VideoMAE is the main learner and receives masked, confidence-weighted guidance from text embeddings,
binary hate probability, and pseudo severity predictions.

## Files

- `prepare_manifest.py` joins `datasets/diversified/splits.csv` with `materialized_clip_index.csv`.
- `asr_transcribe.py` optionally creates ASR transcripts for videos without human transcripts.
- `cache_text_teacher.py` caches frozen RoBERTa embeddings, binary logits, and severity logits.
- `train_text_guided_vision.py` trains VideoMAE plus fusion heads on mixed text/no-text batches.
- `finetune_fusion.py` fine-tunes fusion and heads with severity-boosted pseudo labels.
- `evaluate.py` writes metrics, predictions, confusion matrices, and model comparison reports.
- `inference.py` classifies a materialized clip, source video id, or raw video path plus optional transcript.

## Setup

```powershell
pip install -r text_vision\requirements.txt
```

The scripts expect the existing dataset and text model:

```text
E:\m-hvc\datasets\diversified\splits.csv
E:\m-hvc\datasets\diversified\materialized_clip_index.csv
E:\m-hvc\datasets\diversified\clips
E:\m-hvc\multimodal-hvc\text_classification\roberta_base_finetuned_dualhead.pt
```

## Pipeline

```powershell
python text_vision\prepare_manifest.py
python text_vision\cache_text_teacher.py --batch-size 16
python text_vision\train_text_guided_vision.py --epochs 8 --batch-size 4 --accum-steps 8
python text_vision\finetune_fusion.py --epochs 3 --batch-size 4 --accum-steps 8 --resume
python text_vision\evaluate.py
```

For a quick smoke run, add `--limit 8` to teacher caching and `--max-train-batches 1 --max-eval-batches 1 --epochs 1`
to training. Full performance requires full teacher caching, optional ASR expansion, and normal training.

## Optional ASR

```powershell
python text_vision\asr_transcribe.py --model-size small --device cuda
python text_vision\prepare_manifest.py --asr-cache text_vision\data\asr_transcripts.csv
python text_vision\cache_text_teacher.py --batch-size 16
```

Human transcripts are preferred over ASR. ASR rows are marked with `transcript_source=asr` and their confidence
downweights teacher losses.

## Outputs

Evaluation writes:

```text
text_vision\reports\metrics.csv
text_vision\reports\model_comparison.csv
text_vision\reports\predictions.csv
text_vision\reports\confusion_binary.png
text_vision\reports\confusion_severity.png
text_vision\reports\hard_examples.csv
```

Severity metrics are against pseudo labels, not true video severity annotations.

## Inference

```powershell
python text_vision\inference.py --source-video-id R_hate_video_100 --transcript "example transcript"
python text_vision\inference.py --clip-path E:\m-hvc\datasets\diversified\clips\test\R_hate_video_100\clip_000_s16_k3_t32.pt
```

The JSON output includes hate/non-hate, `prob_hate`, confidence, severity `0..3`, and severity probabilities.

## Taxonomy Moderation Extension

This isolated workspace also supports the segment-level moderation taxonomy:

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

Neutral is not a BCE label. Neutral rows have all taxonomy targets set to `0`.

### Commands

Prepare the segment manifest:

```powershell
python prepare_multilabel_manifest.py --input D:\hvc\datasets\action\archieve\final_dataset.csv --output data\multilabel_manifest.csv
```

Train only the taxonomy head:

```powershell
python train_taxonomy_head.py --manifest data\multilabel_manifest.csv --checkpoint checkpoints\best_frame_text_vision_relabel_all_splits.pt --output-dir checkpoints --report-dir reports\taxonomy_head --freeze-backbone --use-pos-weight
```

Fine-tune fusion layers plus taxonomy head:

```powershell
python train_taxonomy_fusion_finetune.py --manifest data\multilabel_manifest.csv --checkpoint checkpoints\best_taxonomy_head.pt --output-dir checkpoints --report-dir reports\taxonomy_fusion_finetune
```

Run inference the old way:

```powershell
python inference_frame_text_vision.py --frame-dir PATH --checkpoint checkpoints\best_frame_text_vision_relabel_all_splits.pt
```

Run inference with taxonomy:

```powershell
python inference_frame_text_vision.py --frame-dir PATH --checkpoint checkpoints\best_taxonomy_fusion_finetuned.pt --taxonomy-thresholds reports\taxonomy_head\thresholds.json
```

Run inference with description and evidence:

```powershell
python inference_frame_text_vision.py --frame-dir PATH --checkpoint checkpoints\best_taxonomy_fusion_finetuned.pt --describe-video --include-evidence
```

Generate pseudo-labels and a review queue:

```powershell
python generate_taxonomy_pseudo_labels.py --manifest data\multilabel_manifest.csv --checkpoint checkpoints\best_taxonomy_fusion_finetuned.pt
```

Train the text-only taxonomy baseline:

```powershell
python train_text_taxonomy_classifier.py --manifest data\multilabel_manifest.csv --synthetic-jsonl data\synthetic_taxonomy_examples.jsonl
```

Generate safe synthetic text examples:

```powershell
python generate_synthetic_taxonomy_data.py --output data\synthetic_taxonomy_examples.jsonl
```

Evaluate the full moderation system:

```powershell
python evaluate_moderation_system.py --manifest data\multilabel_manifest.csv --checkpoint checkpoints\best_taxonomy_fusion_finetuned.pt
```

Run light smoke checks:

```powershell
python smoke_moderation_checks.py --manifest data\multilabel_manifest.csv
```
