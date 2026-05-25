# Zero-Shot Llama 70B Taxonomy Evaluation Results

## Run

- System: `zero_shot_groq_llama33_70b_versatile`
- Provider: Groq
- Model: `llama-3.3-70b-versatile`
- Evaluation split: `test`
- Evaluated rows: 154
- API calls: 154
- Decision threshold: 0.5 for every taxonomy label
- Prediction mode: zero-shot LLM taxonomy classification

## Overall Metrics

| metric | value |
|---|---:|
| Micro precision | 0.6291 |
| Micro recall | 0.5460 |
| Micro F1 | 0.5846 |
| Macro precision | 0.4591 |
| Macro recall | 0.3342 |
| Macro F1 | 0.3534 |
| Neutral false positive rate | 0.0208 |
| Rare false positive count | 4 |
| Rare predicted positive count | 4 |
| Rare true positive count | 5 |

## Per-Label Metrics

| label | true_count | predicted_count | true_positives | false_positives | false_negatives | precision | recall | f1 |
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

## Interpretation

The 70B zero-shot Llama model performs slightly better than the 8B zero-shot Llama run on micro metrics, with micro F1 0.5846 compared with 0.5619 for the 8B run. It also reduces rare-label false positives from 11 to 4.

However, it is still much weaker than the trained transformer taxonomy classifier. The 70B model overpredicts `hate_speech` heavily, with 55 predictions for 37 true positives, and it fails to recover the rare labels: `threat`, `illegal`, and `online_harm` all have F1 0.0000.

The strongest 70B labels are `violence`, `hate_speech`, and `discrimination`. The rare-label behavior is more conservative than the 8B model, but recall is still poor.

## Output Files

- Predictions: `predictions.csv`
- Raw LLM responses: `raw_responses.jsonl`
- Overall metrics: `metrics.json`
- Per-label metrics: `per_label_metrics.csv`
- CSV comparison metrics: `comparison_metrics.csv`
