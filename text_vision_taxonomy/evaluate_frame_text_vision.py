from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.config import as_path, get_config
from text_vision.train_frame_text_vision import cfg_from_args, run_evaluation


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Evaluate a frame-directory text-vision checkpoint.")
    parser.add_argument("--manifest", type=str, default=str(cfg.manifest_path))
    parser.add_argument("--teacher-cache", type=str, default=str(cfg.text_teacher_cache_path))
    parser.add_argument("--checkpoint", type=str, default=str(cfg.frame_checkpoint_path))
    parser.add_argument("--last-checkpoint", type=str, default=str(cfg.frame_last_checkpoint_path))
    parser.add_argument("--report-dir", type=str, default=str(cfg.frame_report_dir))
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
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = cfg_from_args(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    metrics = run_evaluation(
        checkpoint_path=as_path(args.checkpoint, cfg.frame_checkpoint_path),
        cfg=cfg,
        device=device,
        threshold=args.threshold,
        max_eval_batches=args.max_eval_batches,
        local_files_only=args.local_files_only,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
