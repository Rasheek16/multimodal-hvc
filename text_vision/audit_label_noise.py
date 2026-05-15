from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.config import TextVisionConfig, as_path, get_config
from text_vision.data_utils import safe_torch_load, video_level_manifest
from text_vision.train_frame_text_vision import run_evaluation


OUTPUT_COLUMNS = [
    "source_video_id",
    "old_label",
    "suggested_label",
    "issue_type",
    "prob_hate",
    "model_confidence",
    "teacher_binary_prob",
    "teacher_confidence",
    "transcript_source",
    "frame_dir",
    "reason",
]


def _json_counts(series: pd.Series) -> Dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).sort_index().items()}


def _json_group_counts(frame: pd.DataFrame, columns: Iterable[str]) -> Dict[str, int]:
    if frame.empty:
        return {}
    grouped = frame.groupby(list(columns), dropna=False).size().sort_index()
    return {"|".join(str(part) for part in key if str(part) != ""): int(value) for key, value in grouped.items()}


def make_eval_config(args: argparse.Namespace) -> TextVisionConfig:
    cfg = get_config()
    cfg.manifest_path = as_path(args.manifest, cfg.manifest_path)
    cfg.text_teacher_cache_path = as_path(args.teacher_cache, cfg.text_teacher_cache_path)
    cfg.checkpoint_path = as_path(args.checkpoint, cfg.frame_checkpoint_path)
    cfg.last_checkpoint_path = as_path(args.last_checkpoint, cfg.frame_last_checkpoint_path)
    cfg.report_dir = as_path(args.report_dir, cfg.frame_report_dir)
    cfg.frame_vision_model_name = args.model_name
    cfg.vision_model_name = args.model_name
    cfg.num_frames = int(args.num_frames)
    cfg.batch_size = int(args.batch_size)
    cfg.accum_steps = max(1, int(args.accum_steps))
    cfg.epochs = int(args.epochs)
    cfg.num_workers = int(args.num_workers)
    cfg.seed = int(args.seed)
    cfg.head_lr = float(args.head_lr)
    cfg.backbone_lr = float(args.backbone_lr)
    cfg.weight_decay = float(args.weight_decay)
    cfg.grad_clip = float(args.grad_clip)
    cfg.freeze_epochs = int(args.freeze_epochs)
    cfg.unfreeze_last_n_blocks = int(args.unfreeze_last_n_blocks)
    cfg.head_dropout = float(args.head_dropout)
    cfg.text_dropout = float(args.text_dropout)
    cfg.class_weight_power = float(args.class_weight_power)
    cfg.temporal_layers = int(args.temporal_layers)
    cfg.temporal_heads = int(args.temporal_heads)
    cfg.temporal_dropout = float(args.temporal_dropout)
    cfg.temporal_pooling = str(args.temporal_pooling)
    cfg.frame_chunk_size = int(args.frame_chunk_size)
    cfg.fusion_architecture = str(args.fusion_architecture)
    cfg.selection_metric = str(args.selection_metric)
    cfg.loss_weights["severity_ce"] = float(args.severity_ce_weight)
    cfg.ensure_directories()
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def load_checkpoint_threshold(checkpoint_path: Path, explicit_threshold: float | None = None) -> Tuple[float, str]:
    if explicit_threshold is not None:
        return float(explicit_threshold), "cli"
    if checkpoint_path.exists():
        checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, Mapping) and "threshold" in checkpoint:
            return float(checkpoint["threshold"]), "checkpoint"
    return 0.5, "default"


def parse_requested_splits(value: str, manifest_videos: pd.DataFrame) -> list[str]:
    available = [str(split) for split in manifest_videos["split"].dropna().astype(str).unique().tolist()]
    preferred = [split for split in ["train", "val", "test"] if split in available]
    requested: list[str] = []
    for raw_part in str(value).replace(";", ",").split(","):
        part = raw_part.strip()
        if not part:
            continue
        lowered = part.lower()
        if lowered in {"all", "*"}:
            requested.extend(preferred or available)
        elif lowered == "eval":
            requested.extend([split for split in ["val", "test"] if split in available])
        else:
            requested.append(part)
    deduped = list(dict.fromkeys(requested))
    if not deduped:
        raise ValueError("At least one split must be selected")
    missing = [split for split in deduped if split not in available]
    if missing:
        raise ValueError(f"Requested split(s) not found in manifest: {missing}; available={available}")
    return deduped


def selected_splits(args: argparse.Namespace, manifest_videos: pd.DataFrame) -> list[str]:
    return parse_requested_splits(args.splits or args.split, manifest_videos)


def predictions_have_splits(
    predictions: pd.DataFrame,
    manifest_videos: pd.DataFrame,
    splits: Sequence[str],
) -> bool:
    if predictions.empty or "source_video_id" not in predictions.columns:
        return False
    for split in splits:
        if "split" in predictions.columns and predictions["split"].astype(str).eq(split).any():
            continue
        split_ids = set(manifest_videos.loc[manifest_videos["split"].astype(str).eq(split), "source_video_id"].astype(str))
        if not predictions["source_video_id"].astype(str).isin(split_ids).any():
            return False
    return True


def load_or_generate_predictions(
    args: argparse.Namespace,
    cfg: TextVisionConfig,
    manifest_videos: pd.DataFrame,
    splits: Sequence[str],
) -> Tuple[pd.DataFrame, Path, bool]:
    predictions_path = as_path(args.predictions, cfg.report_dir / "predictions.csv")
    generated = False
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        if predictions_have_splits(predictions, manifest_videos, splits):
            print(f"reusing predictions: {predictions_path}")
            return predictions, predictions_path, generated

    print(f"generating predictions from checkpoint: {cfg.checkpoint_path}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_evaluation(
        checkpoint_path=cfg.checkpoint_path,
        cfg=cfg,
        device=device,
        threshold=args.threshold,
        max_eval_batches=args.max_eval_batches,
        local_files_only=args.local_files_only,
    )
    generated = True
    generated_path = cfg.report_dir / "predictions.csv"
    if not generated_path.exists():
        raise FileNotFoundError(f"Prediction generation did not create {generated_path}")
    predictions = pd.read_csv(generated_path)
    if not predictions_have_splits(predictions, manifest_videos, splits):
        raise ValueError(f"Generated predictions do not contain requested splits: {list(splits)!r}")
    return predictions, generated_path, generated


def load_teacher_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["source_video_id", "teacher_binary_prob", "teacher_confidence"])
    teacher = pd.read_csv(path)
    if "source_video_id" not in teacher.columns:
        raise ValueError(f"Teacher summary is missing source_video_id: {path}")
    rename = {}
    if "binary_prob" in teacher.columns:
        rename["binary_prob"] = "teacher_binary_prob"
    if "label" in teacher.columns:
        rename["label"] = "teacher_label"
    if "transcript_source" in teacher.columns:
        rename["transcript_source"] = "teacher_transcript_source"
    teacher = teacher.rename(columns=rename).copy()
    for column in ["teacher_binary_prob", "teacher_confidence"]:
        if column not in teacher.columns:
            teacher[column] = np.nan
        teacher[column] = pd.to_numeric(teacher[column], errors="coerce")
    teacher["source_video_id"] = teacher["source_video_id"].astype(str)
    return teacher.drop_duplicates("source_video_id", keep="last")


def prepare_audit_frame(
    predictions: pd.DataFrame,
    manifest_videos: pd.DataFrame,
    teacher_summary: pd.DataFrame,
    splits: Sequence[str],
) -> pd.DataFrame:
    required_prediction_columns = {"source_video_id", "prob_hate"}
    missing = sorted(required_prediction_columns.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing required columns: {missing}")

    pred = predictions.copy()
    pred["source_video_id"] = pred["source_video_id"].astype(str)
    pred["prob_hate"] = pd.to_numeric(pred["prob_hate"], errors="coerce")
    pred_columns = ["source_video_id", "prob_hate"]
    for column in ["pred_binary", "binary_logit", "confidence"]:
        if column in pred.columns:
            pred_columns.append(column)
    pred = pred[pred_columns].dropna(subset=["prob_hate"]).drop_duplicates("source_video_id", keep="last")

    videos = manifest_videos.copy()
    videos["source_video_id"] = videos["source_video_id"].astype(str)
    videos = videos[videos["split"].astype(str).isin([str(split) for split in splits])].copy()
    if videos.empty:
        raise ValueError(f"Manifest contains no videos for selected splits: {list(splits)!r}")

    frame = videos.merge(pred, on="source_video_id", how="left", validate="one_to_one")
    frame = frame.dropna(subset=["prob_hate"]).copy()
    if frame.empty:
        raise ValueError(f"No predictions matched manifest videos for selected splits: {list(splits)!r}")

    frame = frame.merge(
        teacher_summary,
        on="source_video_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_teacher"),
    )
    if "teacher_binary_prob" not in frame.columns:
        frame["teacher_binary_prob"] = np.nan
    if "teacher_confidence" not in frame.columns:
        frame["teacher_confidence"] = np.nan
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce").astype("Int64")
    return frame.dropna(subset=["label"]).copy()


def teacher_agrees(
    issue_type: str,
    teacher_prob: float,
    teacher_confidence: float,
    args: argparse.Namespace,
) -> Tuple[bool, bool]:
    has_usable_teacher = bool(np.isfinite(teacher_prob) and np.isfinite(teacher_confidence))
    if not has_usable_teacher:
        return False, False
    strong_enough = teacher_confidence >= float(args.teacher_min_confidence)
    if not strong_enough:
        return False, False
    if issue_type == "suspected_false_positive_label":
        return True, teacher_prob <= float(args.teacher_nonhate_max_prob)
    if issue_type == "suspected_false_negative_label":
        return True, teacher_prob >= float(args.teacher_hate_min_prob)
    return True, False


def classify_issue(row: pd.Series, threshold: float, args: argparse.Namespace) -> Dict[str, Any] | None:
    label = int(row["label"])
    prob = float(row["prob_hate"])
    if label == 1 and prob < float(threshold):
        issue_type = "suspected_false_positive_label"
        suggested_label = 0
        model_confidence = 1.0 - prob
        weak_reason = f"model_prob<{threshold:.3f}"
        model_auto = prob <= float(args.auto_fp_max_prob)
        model_extreme = prob <= float(args.extreme_fp_max_prob)
        model_auto_reason = f"model_prob<={float(args.auto_fp_max_prob):.3f}"
        model_extreme_reason = f"model_prob<={float(args.extreme_fp_max_prob):.3f}"
    elif label == 0 and prob >= float(threshold):
        issue_type = "suspected_false_negative_label"
        suggested_label = 1
        model_confidence = prob
        weak_reason = f"model_prob>={threshold:.3f}"
        model_auto = prob >= float(args.auto_fn_min_prob)
        model_extreme = prob >= float(args.extreme_fn_min_prob)
        model_auto_reason = f"model_prob>={float(args.auto_fn_min_prob):.3f}"
        model_extreme_reason = f"model_prob>={float(args.extreme_fn_min_prob):.3f}"
    else:
        return None

    teacher_prob = float(row["teacher_binary_prob"]) if pd.notna(row.get("teacher_binary_prob")) else np.nan
    teacher_confidence = float(row["teacher_confidence"]) if pd.notna(row.get("teacher_confidence")) else np.nan
    has_usable_teacher, teacher_strong_agrees = teacher_agrees(issue_type, teacher_prob, teacher_confidence, args)

    if has_usable_teacher:
        teacher_agreement = "strong_agree" if teacher_strong_agrees else "disagree_or_weak"
    else:
        teacher_agreement = "unavailable"

    auto_relabel = False
    if has_usable_teacher and model_auto and teacher_strong_agrees:
        auto_relabel = True
        reason = (
            f"auto: {model_auto_reason}; teacher_agrees; "
            f"teacher_confidence>={float(args.teacher_min_confidence):.3f}"
        )
    elif not has_usable_teacher and model_extreme:
        auto_relabel = True
        reason = f"auto: no_usable_teacher; {model_extreme_reason}"
    elif has_usable_teacher and model_auto and not teacher_strong_agrees:
        reason = f"review: {model_auto_reason}; teacher_not_strongly_agreeing"
    elif has_usable_teacher and not model_auto:
        reason = "review: model_disagrees_with_label_but_not_auto_threshold"
    elif not has_usable_teacher and not model_extreme:
        reason = f"review: no_usable_teacher; not_extreme_model_probability; {weak_reason}"
    else:
        reason = f"review: {weak_reason}"

    return {
        "source_video_id": str(row["source_video_id"]),
        "split": str(row.get("split", "")),
        "old_label": label,
        "suggested_label": suggested_label,
        "issue_type": issue_type,
        "prob_hate": prob,
        "model_confidence": float(model_confidence),
        "teacher_binary_prob": teacher_prob,
        "teacher_confidence": teacher_confidence,
        "teacher_agreement": teacher_agreement,
        "auto_relabel": bool(auto_relabel),
        "transcript_source": str(row.get("transcript_source", "missing")),
        "frame_dir": str(row.get("frame_dir", "")),
        "reason": reason,
    }


def find_label_issues(audit_frame: pd.DataFrame, threshold: float, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for _, row in audit_frame.iterrows():
        issue = classify_issue(row, threshold=threshold, args=args)
        if issue is not None:
            rows.append(issue)
    issues = pd.DataFrame(rows)
    if issues.empty:
        return pd.DataFrame(
            columns=[
                *OUTPUT_COLUMNS,
                "split",
                "teacher_agreement",
                "auto_relabel",
            ]
        )
    return issues.sort_values(
        ["model_confidence", "issue_type", "source_video_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def write_relabel_manifest(
    manifest: pd.DataFrame,
    auto_overrides: pd.DataFrame,
    output_path: Path,
    splits: Sequence[str],
) -> pd.DataFrame:
    relabeled = manifest.copy()
    selected = {str(split) for split in splits}
    if not auto_overrides.empty:
        source_ids = relabeled["source_video_id"].astype(str)
        relabeled_splits = relabeled["split"].astype(str)
        for _, row in auto_overrides.iterrows():
            split = str(row.get("split", ""))
            if split not in selected:
                raise ValueError(f"Auto override has unselected split={split!r}")
            mask = source_ids.eq(str(row["source_video_id"])) & relabeled_splits.eq(split)
            relabeled.loc[mask, "label"] = int(row["suggested_label"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    relabeled.to_csv(output_path, index=False)
    return relabeled


def validate_relabel_manifest(
    original: pd.DataFrame,
    relabeled: pd.DataFrame,
    auto_overrides: pd.DataFrame,
    splits: Sequence[str],
) -> None:
    if auto_overrides.empty:
        if not original["label"].equals(relabeled["label"]):
            raise AssertionError("Labels changed even though no auto overrides were selected")
        return

    override_map = {
        (str(row["source_video_id"]), str(row["split"])): int(row["suggested_label"])
        for _, row in auto_overrides.iterrows()
    }
    for (source_video_id, split), new_label in override_map.items():
        rows = relabeled[
            relabeled["source_video_id"].astype(str).eq(source_video_id)
            & relabeled["split"].astype(str).eq(split)
        ]
        if rows.empty:
            raise AssertionError(f"Auto override video missing from relabeled manifest: {source_video_id} ({split})")
        labels = set(int(value) for value in rows["label"].tolist())
        if labels != {int(new_label)}:
            raise AssertionError(f"Inconsistent relabel rows for {source_video_id} ({split}): {sorted(labels)}")

    selected = {str(split) for split in splits}
    original_keys = pd.Series(
        list(zip(original["source_video_id"].astype(str), original["split"].astype(str))),
        index=original.index,
    )
    changed = original["label"].astype(int).ne(relabeled["label"].astype(int))
    changed_keys = set(original_keys.loc[changed].tolist())
    unexpected = changed_keys.difference(override_map)
    if unexpected:
        preview = sorted(unexpected)[:5]
        raise AssertionError(f"Unexpected label changes outside auto overrides: {preview}")
    unselected_changes = [key for key in changed_keys if key[1] not in selected]
    if unselected_changes:
        raise AssertionError(f"Unselected split labels changed: {sorted(unselected_changes)[:5]}")


def write_outputs(
    output_dir: Path,
    manifest: pd.DataFrame,
    issues: pd.DataFrame,
    summary: Mapping[str, Any],
    splits: Sequence[str],
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    issue_path = output_dir / "suspected_label_issues.csv"
    auto_path = output_dir / "auto_relabel_overrides.csv"
    review_path = output_dir / "review_queue.csv"
    relabel_path = output_dir / "multimodal_manifest_auto_relabel.csv"
    summary_path = output_dir / "audit_summary.json"

    auto_overrides = issues[issues["auto_relabel"].astype(bool)].copy() if not issues.empty else issues.copy()
    review = issues[~issues["auto_relabel"].astype(bool)].copy() if not issues.empty else issues.copy()

    for frame, path in [(issues, issue_path), (auto_overrides, auto_path), (review, review_path)]:
        ordered = frame.copy()
        extra_columns = [column for column in ordered.columns if column not in OUTPUT_COLUMNS]
        ordered = ordered[[column for column in OUTPUT_COLUMNS if column in ordered.columns] + extra_columns]
        ordered.to_csv(path, index=False)

    relabeled = write_relabel_manifest(manifest, auto_overrides, relabel_path, splits=splits)
    validate_relabel_manifest(manifest, relabeled, auto_overrides, splits=splits)

    full_summary = dict(summary)
    full_summary["output_files"] = {
        "suspected_label_issues": str(issue_path),
        "auto_relabel_overrides": str(auto_path),
        "review_queue": str(review_path),
        "multimodal_manifest_auto_relabel": str(relabel_path),
        "audit_summary": str(summary_path),
    }
    summary_path.write_text(json.dumps(full_summary, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "suspected_label_issues": issue_path,
        "auto_relabel_overrides": auto_path,
        "review_queue": review_path,
        "multimodal_manifest_auto_relabel": relabel_path,
        "audit_summary": summary_path,
    }


def build_summary(
    manifest: pd.DataFrame,
    audit_frame: pd.DataFrame,
    issues: pd.DataFrame,
    args: argparse.Namespace,
    splits: Sequence[str],
    threshold: float,
    threshold_source: str,
    predictions_path: Path,
    generated_predictions: bool,
) -> Dict[str, Any]:
    auto = issues[issues["auto_relabel"].astype(bool)].copy() if not issues.empty else issues.copy()
    review = issues[~issues["auto_relabel"].astype(bool)].copy() if not issues.empty else issues.copy()
    split_mask = (
        manifest["split"].astype(str).isin([str(split) for split in splits])
        if "split" in manifest.columns
        else pd.Series(False, index=manifest.index)
    )
    return {
        "manifest": str(as_path(args.manifest, get_config().manifest_path)),
        "predictions": str(predictions_path),
        "generated_predictions": bool(generated_predictions),
        "checkpoint": str(as_path(args.checkpoint, get_config().frame_checkpoint_path)),
        "teacher_summary": str(as_path(args.teacher_summary, get_config().text_teacher_summary_path)),
        "split": ",".join(str(split) for split in splits),
        "splits": [str(split) for split in splits],
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "num_manifest_rows": int(len(manifest)),
        "num_split_manifest_rows": int(split_mask.sum()),
        "num_split_videos_with_predictions": int(len(audit_frame)),
        "num_suspicious": int(len(issues)),
        "num_auto_relabel": int(len(auto)),
        "num_review": int(len(review)),
        "counts_by_issue_type": _json_counts(issues["issue_type"]) if not issues.empty else {},
        "counts_by_transcript_source": _json_counts(issues["transcript_source"]) if not issues.empty else {},
        "counts_by_old_new_label": _json_group_counts(issues, ["old_label", "suggested_label"]),
        "counts_by_teacher_agreement": _json_counts(issues["teacher_agreement"]) if not issues.empty else {},
        "counts_by_split": _json_counts(issues["split"]) if not issues.empty else {},
        "auto_counts_by_split": _json_counts(auto["split"]) if not auto.empty else {},
        "review_counts_by_split": _json_counts(review["split"]) if not review.empty else {},
        "auto_counts_by_issue_type": _json_counts(auto["issue_type"]) if not auto.empty else {},
        "review_counts_by_issue_type": _json_counts(review["issue_type"]) if not review.empty else {},
        "auto_thresholds": {
            "auto_fp_max_prob": float(args.auto_fp_max_prob),
            "auto_fn_min_prob": float(args.auto_fn_min_prob),
            "teacher_nonhate_max_prob": float(args.teacher_nonhate_max_prob),
            "teacher_hate_min_prob": float(args.teacher_hate_min_prob),
            "teacher_min_confidence": float(args.teacher_min_confidence),
            "extreme_fp_max_prob": float(args.extreme_fp_max_prob),
            "extreme_fn_min_prob": float(args.extreme_fn_min_prob),
        },
    }


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Audit selected splits for label noise using frame text-vision predictions.")
    parser.add_argument("--manifest", type=str, default=str(cfg.manifest_path))
    parser.add_argument("--teacher-summary", type=str, default=str(cfg.text_teacher_summary_path))
    parser.add_argument("--teacher-cache", type=str, default=str(cfg.text_teacher_cache_path))
    parser.add_argument("--checkpoint", type=str, default=str(cfg.frame_checkpoint_path))
    parser.add_argument("--last-checkpoint", type=str, default=str(cfg.frame_last_checkpoint_path))
    parser.add_argument("--predictions", type=str, default=str(cfg.frame_report_dir / "predictions.csv"))
    parser.add_argument("--output-dir", type=str, default=str(cfg.text_vision_root / "reports" / "label_audit"))
    parser.add_argument("--report-dir", type=str, default=str(cfg.frame_report_dir))
    parser.add_argument("--split", type=str, default="train", help="Single split to audit; accepts comma-separated values for compatibility.")
    parser.add_argument("--splits", type=str, default=None, help="Comma-separated splits to audit, or 'all'/'eval'. Overrides --split.")
    parser.add_argument("--threshold", type=float, default=None)

    parser.add_argument("--auto-fp-max-prob", type=float, default=0.10)
    parser.add_argument("--auto-fn-min-prob", type=float, default=0.90)
    parser.add_argument("--teacher-nonhate-max-prob", type=float, default=0.25)
    parser.add_argument("--teacher-hate-min-prob", type=float, default=0.75)
    parser.add_argument("--teacher-min-confidence", type=float, default=0.55)
    parser.add_argument("--extreme-fp-max-prob", type=float, default=0.05)
    parser.add_argument("--extreme-fn-min-prob", type=float, default=0.95)

    parser.add_argument("--model-name", type=str, default=cfg.frame_vision_model_name)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accum-steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--head-lr", type=float, default=cfg.head_lr)
    parser.add_argument("--backbone-lr", type=float, default=cfg.backbone_lr)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=cfg.grad_clip)
    parser.add_argument("--freeze-epochs", type=int, default=cfg.freeze_epochs)
    parser.add_argument("--unfreeze-last-n-blocks", type=int, default=cfg.unfreeze_last_n_blocks)
    parser.add_argument("--head-dropout", type=float, default=cfg.head_dropout)
    parser.add_argument("--text-dropout", type=float, default=cfg.text_dropout)
    parser.add_argument("--class-weight-power", type=float, default=cfg.class_weight_power)
    parser.add_argument("--temporal-layers", type=int, default=cfg.temporal_layers)
    parser.add_argument("--temporal-heads", type=int, default=cfg.temporal_heads)
    parser.add_argument("--temporal-dropout", type=float, default=cfg.temporal_dropout)
    parser.add_argument("--temporal-pooling", choices=["cls", "attn_mean_max"], default=cfg.temporal_pooling)
    parser.add_argument("--frame-chunk-size", type=int, default=cfg.frame_chunk_size)
    parser.add_argument("--fusion-architecture", choices=["concat", "gated"], default=cfg.fusion_architecture)
    parser.add_argument("--selection-metric", type=str, default=cfg.selection_metric)
    parser.add_argument("--severity-ce-weight", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = make_eval_config(args)
    output_dir = as_path(args.output_dir, cfg.text_vision_root / "reports" / "label_audit")

    manifest = pd.read_csv(cfg.manifest_path)
    manifest_videos = video_level_manifest(manifest)
    splits = selected_splits(args, manifest_videos)
    if any(split in {"val", "test"} for split in splits):
        print(
            "warning: relabeling validation/test splits changes the evaluation target; "
            "compare future metrics only against this cleaned manifest."
        )
    threshold, threshold_source = load_checkpoint_threshold(cfg.checkpoint_path, args.threshold)
    predictions, predictions_path, generated_predictions = load_or_generate_predictions(args, cfg, manifest_videos, splits=splits)
    teacher_summary = load_teacher_summary(as_path(args.teacher_summary, cfg.text_teacher_summary_path))
    audit_frame = prepare_audit_frame(predictions, manifest_videos, teacher_summary, splits=splits)
    issues = find_label_issues(audit_frame, threshold=threshold, args=args)

    summary = build_summary(
        manifest=manifest,
        audit_frame=audit_frame,
        issues=issues,
        args=args,
        splits=splits,
        threshold=threshold,
        threshold_source=threshold_source,
        predictions_path=predictions_path,
        generated_predictions=generated_predictions,
    )
    paths = write_outputs(output_dir, manifest, issues, summary, splits=splits)
    print(
        json.dumps(
            {
                "splits": summary["splits"],
                "num_suspicious": summary["num_suspicious"],
                "num_auto_relabel": summary["num_auto_relabel"],
                "num_review": summary["num_review"],
                "threshold": summary["threshold"],
                "outputs": {key: str(value) for key, value in paths.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
