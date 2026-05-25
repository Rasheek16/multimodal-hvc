# Zero-Shot Llama Taxonomy Evaluation Results

## Run

- System: `zero_shot_groq_llama31_8b_instant`
- Provider: Groq
- Model: `llama-3.1-8b-instant`
- Evaluation split: `test`
- Evaluated rows: 154
- API calls: 154
- Decision threshold: 0.5 for every taxonomy label
- Prediction mode: zero-shot LLM taxonomy classification

## Overall Metrics

| metric | value |
|---|---:|
| Micro precision | 0.5924 |
| Micro recall | 0.5345 |
| Micro F1 | 0.5619 |
| Macro precision | 0.4193 |
| Macro recall | 0.3725 |
| Macro F1 | 0.3851 |
| Neutral false positive rate | 0.0417 |
| Rare false positive count | 11 |
| Rare predicted positive count | 12 |
| Rare true positive count | 5 |

## Per-Label Metrics

| label | true_count | predicted_count | true_positives | false_positives | false_negatives | precision | recall | f1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hate_speech | 37 | 32 | 19 | 13 | 18 | 0.5938 | 0.5135 | 0.5507 |
| discrimination | 35 | 38 | 27 | 11 | 8 | 0.7105 | 0.7714 | 0.7397 |
| contextual_hate | 25 | 17 | 8 | 9 | 17 | 0.4706 | 0.3200 | 0.3810 |
| threat | 3 | 4 | 1 | 3 | 2 | 0.2500 | 0.3333 | 0.2857 |
| violence | 31 | 33 | 27 | 6 | 4 | 0.8182 | 0.8710 | 0.8438 |
| fear | 19 | 16 | 6 | 10 | 13 | 0.3750 | 0.3158 | 0.3429 |
| sexual | 22 | 9 | 5 | 4 | 17 | 0.5556 | 0.2273 | 0.3226 |
| illegal | 0 | 3 | 0 | 3 | 0 | 0.0000 | 0.0000 | 0.0000 |
| online_harm | 2 | 5 | 0 | 5 | 2 | 0.0000 | 0.0000 | 0.0000 |

## Interpretation

The zero-shot Llama baseline is useful as a comparison system, but it is much weaker than the trained taxonomy classifier. Its main weakness is precision on rare labels: it produced 11 rare-label false positives, including false positives for `illegal` and `online_harm`.

The strongest Llama label was `violence`, with F1 0.8438. The weakest useful labels were `illegal` and `online_harm`, both with F1 0.0000 because the model predicted them but did not match any true positives on the test split.

## Output Files

- Predictions: `predictions.csv`
- Raw LLM responses: `raw_responses.jsonl`
- Overall metrics: `metrics.json`
- Per-label metrics: `per_label_metrics.csv`
- CSV comparison metrics: `comparison_metrics.csv`
