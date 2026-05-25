# Text Vision Taxonomy Run Commands

Run these from PowerShell.

```powershell
cd D:\text_vision_taxonomy
$py = "C:\Users\ASUS\.conda\envs\torch\python.exe"
```

## 1. Rebuild Manifest

```powershell
& $py prepare_multilabel_manifest.py `
  --input D:\hvc\datasets\action\archieve\final_dataset.csv `
  --output data\multilabel_manifest.csv `
  --summary reports\multilabel_manifest_summary.json
```

## 2. Smoke Check

```powershell
& $py smoke_moderation_checks.py `
  --manifest data\multilabel_manifest.csv
```

## Deep Learning Only Taxonomy Training

Train the transformer taxonomy model:

```powershell
& $py train_transformer_multilabel_classifier.py `
  --dataset D:\hvc\datasets\action\archieve\final_dataset.csv `
  --model-names roberta-base `
  --output-dir checkpoints\transformer_multilabel_classifier `
  --report-dir reports\transformer_multilabel_classifier `
  --epochs 8 `
  --patience 2 `
  --batch-size 4 `
  --max-length 256 `
  --lr 2e-5 `
  --loss focal `
  --gamma 2 `
  --use-pos-weight `
  --max-pos-weight 5 `
  --local-files-only
```

If `roberta-base` is not fully cached, remove `--local-files-only` for the first run.

Calibrate transformer thresholds:

```powershell
& $py calibrate_multilabel_thresholds.py `
  --predictions reports\transformer_multilabel_classifier\predictions.csv `
  --output-dir reports\transformer_multilabel_calibration `
  --split val
```

Evaluate transformer only:

```powershell
& $py evaluate_multilabel_system.py `
  --tfidf-predictions missing.csv `
  --ensemble-predictions missing.csv `
  --transformer-predictions reports\transformer_multilabel_classifier\predictions.csv `
  --transformer-thresholds reports\transformer_multilabel_calibration\thresholds.json `
  --output-dir reports\transformer_multilabel_eval `
  --eval-split test `
  --table-system transformer_calibrated
```

## Zero-Shot LLM Comparison

Use this to compare the trained taxonomy model against a zero-shot LLM judge on the same test split. The evaluator writes `predictions.csv`, `metrics.csv`, `per_label_metrics.csv`, `metrics.json`, and `comparison_metrics.csv`.

Install optional API/local-model dependencies if needed:

```powershell
pip install openai accelerate langchain-groq
```

GPT-style OpenAI run. Set `OPENAI_API_KEY` first:

```powershell
$env:OPENAI_API_KEY = "YOUR_KEY"

& $py evaluate_zero_shot_llm_taxonomy.py `
  --provider openai `
  --model gpt-4o `
  --system-name zero_shot_gpt4o `
  --dataset D:\hvc\datasets\action\archieve\final_dataset.csv `
  --eval-split test `
  --output-dir reports\zero_shot_llm_eval\gpt4o `
  --resume
```

If you specifically have access to an older `gpt-4` model ID, replace `gpt-4o` with `gpt-4`.

Groq ChatGroq Llama run. Set `GROQ_API_KEY` first. This uses LangChain's `ChatGroq` wrapper and Groq's `llama-3.1-8b-instant` model ID:

```powershell
$env:GROQ_API_KEY = "YOUR_GROQ_KEY"

& $py evaluate_zero_shot_llm_taxonomy.py `
  --provider groq `
  --model llama-3.1-8b-instant `
  --system-name zero_shot_groq_llama31_8b_instant `
  --dataset D:\hvc\datasets\action\archieve\final_dataset.csv `
  --eval-split test `
  --output-dir reports\zero_shot_llm_eval\groq_llama31_8b_instant `
  --resume
```

Ollama local Llama run:

```powershell
ollama pull llama3.1:8b

& $py evaluate_zero_shot_llm_taxonomy.py `
  --provider ollama `
  --model llama3.1:8b `
  --system-name zero_shot_llama31 `
  --dataset D:\hvc\datasets\action\archieve\final_dataset.csv `
  --eval-split test `
  --output-dir reports\zero_shot_llm_eval\llama31_ollama `
  --resume
```

HuggingFace/local Llama run. Use your actual local model folder or HuggingFace model ID:

```powershell
& $py evaluate_zero_shot_llm_taxonomy.py `
  --provider hf `
  --model meta-llama/Llama-3.1-8B-Instruct `
  --system-name zero_shot_llama31_hf `
  --dataset D:\hvc\datasets\action\archieve\final_dataset.csv `
  --eval-split test `
  --output-dir reports\zero_shot_llm_eval\llama31_hf `
  --resume
```

Note: official Llama 3.1 text models are commonly 8B, 70B, and 405B. If you mean an 11B Llama model, use the exact local model path or HuggingFace ID available on your machine.

## Vision Taxonomy Training

Prepare the frame/video manifest:

```powershell
& $py prepare_multilabel_manifest.py `
  --input D:\hvc\datasets\action\archieve\final_dataset.csv `
  --output data\multilabel_manifest.csv `
  --summary reports\multilabel_manifest_summary.json `
  --frame-root D:\hvc\datasets\action\timestamp_frames `
  --whole-frame-root D:\hvc\datasets\action\frames `
  --video-root D:\hvc\datasets\action\videos
```

Train the deep video taxonomy model on many sampled frames. This starts from the existing vision hate/severity model, freezes the ViT image backbone, and trains the temporal video encoder plus taxonomy fusion layers:

```powershell
& $py train_taxonomy_head.py `
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
  --num-frames 192 `
  --frame-chunk-size 8 `
  --amp `
  --local-files-only
```

If CUDA runs out of memory, keep `--batch-size 1` and reduce `--num-frames` to `128`, or reduce `--frame-chunk-size` to `4`.

## Final Runtime: Metadata-Guided Moderation

Manual metadata inference. This does not regenerate metadata:

```powershell
& $py inference_moderation_system.py `
  --scene "A man gives an aggressive speech targeting a group" `
  --action "verbal abuse" `
  --subcategories "hate speech, discrimination" `
  --taxonomy-classifier checkpoints\text_multilabel_classifier\best_model.joblib `
  --taxonomy-thresholds reports\multilabel_calibration\thresholds.json `
  --local-files-only
```

Raw video inference. This first generates metadata, then runs taxonomy:

```powershell
& $py inference_moderation_system.py `
  --video path\to\video.mp4 `
  --transformer-taxonomy-model checkpoints\transformer_multilabel_classifier_feedback\best `
  --taxonomy-thresholds reports\transformer_multilabel_feedback_calibration\thresholds.json `
  --video-taxonomy-checkpoint checkpoints\best_taxonomy_head.pt
```

For a fully offline raw-video run, only use this if the caption model is already cached locally:

```powershell
& $py inference_moderation_system.py `
  --video path\to\video.mp4 `
  --transformer-taxonomy-model checkpoints\transformer_multilabel_classifier_feedback\best `
  --taxonomy-thresholds reports\transformer_multilabel_feedback_calibration\thresholds.json `
  --video-taxonomy-checkpoint checkpoints\best_taxonomy_head.pt `
  --local-files-only
```

Frame directory inference:

```powershell
& $py inference_moderation_system.py `
  --frame-dir "D:\hvc\datasets\action\timestamp_frames\hate_video_1\merged_hatevideos_00002_0000018000_0000044000" `
  --vision-checkpoint checkpoints\best_frame_text_vision_relabel_all_splits.pt `
  --taxonomy-classifier checkpoints\text_multilabel_classifier\best_model.joblib `
  --taxonomy-thresholds reports\multilabel_calibration\thresholds.json `
  --generate-metadata `
  --local-files-only
```

If the text/vision hate classifiers already ran elsewhere, pass their signals directly:

```powershell
& $py inference_moderation_system.py `
  --scene "A man gives an aggressive speech targeting a group" `
  --action "verbal abuse" `
  --subcategories "hate speech, discrimination" `
  --text-hate-prob 0.91 `
  --vision-hate-prob 0.40 `
  --text-severity 3 `
  --vision-severity 2
```

Mine runtime logs into reinforcement queues:

```powershell
& $py mine_taxonomy_training_cases.py `
  --inference-log reports\runtime_feedback\inference_logs.csv `
  --review-output data\taxonomy_review_queue.csv `
  --pseudo-output data\taxonomy_pseudo_labeled.csv
```

Prepare feedback training data without training:

```powershell
& $py retrain_taxonomy_with_feedback.py `
  --dataset D:\hvc\datasets\action\archieve\final_dataset.csv `
  --prepare-only
```

Retrain with approved feedback and conservative pseudo-labels:

```powershell
& $py retrain_taxonomy_with_feedback.py `
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

## 3. Train Taxonomy Head

```powershell
& $py train_taxonomy_head.py `
  --manifest data\multilabel_manifest.csv `
  --checkpoint checkpoints\best_frame_text_vision_relabel_all_splits.pt `
  --output-dir checkpoints `
  --report-dir reports\taxonomy_head `
  --freeze-backbone `
  --use-pos-weight `
  --local-files-only
```

## 4. Fine-Tune Fusion

Run after `checkpoints\best_taxonomy_head.pt` exists.

```powershell
& $py train_taxonomy_fusion_finetune.py `
  --manifest data\multilabel_manifest.csv `
  --checkpoint checkpoints\best_taxonomy_head.pt `
  --output-dir checkpoints `
  --report-dir reports\taxonomy_fusion_finetune `
  --use-pos-weight `
  --local-files-only
```

## 5. Inference With Taxonomy

```powershell
& $py inference_frame_text_vision.py `
  --frame-dir "D:\hvc\datasets\action\timestamp_frames\hate_video_1\merged_hatevideos_00002_0000018000_0000044000" `
  --checkpoint checkpoints\best_taxonomy_head.pt `
  --taxonomy-thresholds reports\taxonomy_head\thresholds.json `
  --local-files-only
```

## 6. Inference With Description/Evidence

```powershell
& $py inference_frame_text_vision.py `
  --frame-dir "D:\hvc\datasets\action\timestamp_frames\hate_video_1\merged_hatevideos_00002_0000018000_0000044000" `
  --checkpoint checkpoints\best_taxonomy_head.pt `
  --taxonomy-thresholds reports\taxonomy_head\thresholds.json `
  --describe-video `
  --include-evidence `
  --local-files-only
```

## 7. Generate Pseudo Labels

```powershell
& $py generate_taxonomy_pseudo_labels.py `
  --manifest data\multilabel_manifest.csv `
  --checkpoint checkpoints\best_taxonomy_head.pt `
  --taxonomy-thresholds reports\taxonomy_head\thresholds.json `
  --output-dir reports\taxonomy_pseudo_labels `
  --enriched-output data\multilabel_manifest_enriched.csv `
  --local-files-only
```

## 8. Generate Synthetic Text Data

```powershell
& $py generate_synthetic_taxonomy_data.py `
  --output data\synthetic_taxonomy_examples.jsonl
```

## 9. Train Text-Only Taxonomy Classifier

```powershell
& $py train_text_taxonomy_classifier.py `
  --manifest data\multilabel_manifest.csv `
  --synthetic-jsonl data\synthetic_taxonomy_examples.jsonl `
  --output checkpoints\best_text_taxonomy_classifier.pt `
  --report-dir reports\text_taxonomy_classifier
```

## 10. Final Evaluation

```powershell
& $py evaluate_moderation_system.py `
  --manifest data\multilabel_manifest.csv `
  --checkpoint checkpoints\best_taxonomy_head.pt `
  --taxonomy-thresholds reports\taxonomy_head\thresholds.json `
  --output-dir reports\final_moderation_eval `
  --local-files-only
```

## Precision Fix: Calibrate Current Taxonomy Head

```powershell
& $py calibrate_taxonomy_thresholds.py `
  --predictions reports\taxonomy_head\predictions.csv `
  --output reports\taxonomy_head\calibrated_thresholds.json
```

Use calibrated thresholds in inference:

```powershell
& $py inference_frame_text_vision.py `
  --frame-dir "D:\hvc\datasets\action\timestamp_frames\hate_video_1\merged_hatevideos_00002_0000018000_0000044000" `
  --checkpoint checkpoints\best_taxonomy_head.pt `
  --taxonomy-thresholds reports\taxonomy_head\calibrated_thresholds.json `
  --local-files-only
```

## Precision Fix: Retrain Experiments

BCE without `pos_weight`:

```powershell
& $py train_taxonomy_head.py `
  --manifest data\multilabel_manifest.csv `
  --checkpoint checkpoints\best_frame_text_vision_relabel_all_splits.pt `
  --output-dir checkpoints\taxonomy_head_no_pos_weight `
  --report-dir reports\taxonomy_head_no_pos_weight `
  --loss bce `
  --pos-weight-mode none `
  --local-files-only
```

BCE with capped `pos_weight`:

```powershell
& $py train_taxonomy_head.py `
  --manifest data\multilabel_manifest.csv `
  --checkpoint checkpoints\best_frame_text_vision_relabel_all_splits.pt `
  --output-dir checkpoints\taxonomy_head_capped_pos_weight `
  --report-dir reports\taxonomy_head_capped_pos_weight `
  --loss bce `
  --pos-weight-mode capped `
  --pos-weight-cap 5 `
  --local-files-only
```

Focal loss:

```powershell
& $py train_taxonomy_head.py `
  --manifest data\multilabel_manifest.csv `
  --checkpoint checkpoints\best_frame_text_vision_relabel_all_splits.pt `
  --output-dir checkpoints\taxonomy_head_focal `
  --report-dir reports\taxonomy_head_focal `
  --loss focal `
  --focal-gamma 2 `
  --pos-weight-mode none `
  --local-files-only
```

Calibrate each experiment after it finishes:

```powershell
& $py calibrate_taxonomy_thresholds.py --predictions reports\taxonomy_head_no_pos_weight\predictions.csv --output reports\taxonomy_head_no_pos_weight\calibrated_thresholds.json
& $py calibrate_taxonomy_thresholds.py --predictions reports\taxonomy_head_capped_pos_weight\predictions.csv --output reports\taxonomy_head_capped_pos_weight\calibrated_thresholds.json
& $py calibrate_taxonomy_thresholds.py --predictions reports\taxonomy_head_focal\predictions.csv --output reports\taxonomy_head_focal\calibrated_thresholds.json
```

## Precision Fix: Text Classifier, Ensemble, Review Queue

```powershell
& $py train_text_taxonomy_classifier.py `
  --manifest data\multilabel_manifest.csv `
  --output checkpoints\text_taxonomy_classifier.joblib `
  --report-dir reports\text_taxonomy_classifier

& $py ensemble_taxonomy.py `
  --video-predictions reports\taxonomy_head\predictions.csv `
  --text-predictions reports\text_taxonomy_classifier\predictions.csv `
  --thresholds reports\taxonomy_head\calibrated_thresholds.json `
  --output reports\taxonomy_ensemble\predictions.csv

& $py calibrate_taxonomy_thresholds.py `
  --predictions reports\taxonomy_ensemble\predictions.csv `
  --output reports\taxonomy_ensemble\calibrated_thresholds.json

& $py generate_taxonomy_review_queue.py `
  --predictions reports\taxonomy_ensemble\predictions.csv `
  --thresholds reports\taxonomy_ensemble\calibrated_thresholds.json `
  --output-dir reports\taxonomy_review_queue

& $py evaluate_final_taxonomy_system.py `
  --output-dir reports\final_taxonomy_eval
```

## Recommended Run Order

```text
1 -> 2 -> 3 -> 5/6
then 4
then 7 -> 8 -> 9 -> 10
then precision calibration/retrain/text/ensemble/review/eval
```
