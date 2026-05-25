from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class TextGuidedVisionModel(nn.Module):
    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        text_embedding_dim: int = 768,
        dropout: float = 0.35,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, VideoMAEModel

        config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
        self.backbone = VideoMAEModel.from_pretrained(
            model_name,
            config=config,
            local_files_only=local_files_only,
        )
        self.visual_dim = int(config.hidden_size)
        self.text_embedding_dim = int(text_embedding_dim)
        hidden = self.visual_dim
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(float(dropout))
        self.binary_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden // 2, 1),
        )
        self.severity_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden // 2, 4),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, self.text_embedding_dim),
        )

    def pooled_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.backbone(pixel_values=pixel_values)
        hidden = output.last_hidden_state
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            pooled = output.pooler_output
        else:
            pooled = hidden.mean(dim=1)
        return self.norm(pooled)

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        visual_embedding = self.pooled_features(pixel_values)
        visual_embedding = self.dropout(visual_embedding)
        binary_logits = self.binary_head(visual_embedding).squeeze(-1)
        severity_logits = self.severity_head(visual_embedding)
        projected_text_embedding = self.text_projection(visual_embedding)
        return {
            "visual_embedding": visual_embedding,
            "binary_logits": binary_logits,
            "severity_logits": severity_logits,
            "projected_text_embedding": projected_text_embedding,
        }

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(True)

    def unfreeze_last_encoder_blocks(self, last_n_blocks: int) -> int:
        self.freeze_backbone()
        encoder = getattr(getattr(self.backbone, "encoder", None), "layer", None)
        if encoder is None:
            encoder = getattr(getattr(getattr(self.backbone, "videomae", None), "encoder", None), "layer", None)
        if encoder is None:
            return 0
        last_n_blocks = max(0, min(int(last_n_blocks), len(encoder)))
        for block in encoder[-last_n_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        return last_n_blocks


def cosine_alignment_loss(projected: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    projected = F.normalize(projected, dim=-1)
    target = F.normalize(target, dim=-1)
    return 1.0 - (projected * target).sum(dim=-1)

