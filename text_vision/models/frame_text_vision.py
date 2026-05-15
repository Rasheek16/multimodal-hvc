from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F


class FrameTextVisionModel(nn.Module):
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224-in21k",
        text_embedding_dim: int = 768,
        num_frames: int = 64,
        temporal_layers: int = 2,
        temporal_heads: int = 8,
        temporal_dropout: float = 0.10,
        temporal_pooling: str = "attn_mean_max",
        frame_chunk_size: int = 16,
        dropout: float = 0.35,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
        self.backbone = AutoModel.from_pretrained(
            model_name,
            config=config,
            local_files_only=local_files_only,
        )
        self.visual_dim = int(
            getattr(config, "hidden_size", 0)
            or getattr(getattr(config, "vision_config", None), "hidden_size", 0)
        )
        if self.visual_dim <= 0:
            raise ValueError(f"Could not infer image backbone hidden size for {model_name}")

        self.num_frames = int(num_frames)
        self.text_embedding_dim = int(text_embedding_dim)
        self.frame_chunk_size = int(frame_chunk_size)
        self.temporal_pooling = str(temporal_pooling)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.visual_dim))
        self.frame_pos_embedding = nn.Parameter(torch.zeros(1, self.num_frames, self.visual_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.visual_dim,
            nhead=int(temporal_heads),
            dim_feedforward=self.visual_dim * 4,
            dropout=float(temporal_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(temporal_layers),
        )
        self.norm = nn.LayerNorm(self.visual_dim)
        attention_hidden = max(self.visual_dim // 2, 32)
        self.frame_attention = nn.Sequential(
            nn.LayerNorm(self.visual_dim),
            nn.Linear(self.visual_dim, attention_hidden),
            nn.GELU(),
            nn.Linear(attention_hidden, 1),
        )
        self.temporal_projection = nn.Sequential(
            nn.LayerNorm(self.visual_dim * 4),
            nn.Linear(self.visual_dim * 4, self.visual_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(self.visual_dim),
        )
        self.dropout = nn.Dropout(float(dropout))
        self.binary_head = nn.Sequential(
            nn.Linear(self.visual_dim, self.visual_dim // 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.visual_dim // 2, 1),
        )
        self.severity_head = nn.Sequential(
            nn.Linear(self.visual_dim, self.visual_dim // 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.visual_dim // 2, 4),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(self.visual_dim, self.visual_dim),
            nn.GELU(),
            nn.LayerNorm(self.visual_dim),
            nn.Linear(self.visual_dim, self.text_embedding_dim),
        )
        self._reset_temporal_parameters()

    def _reset_temporal_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.frame_pos_embedding, std=0.02)

    def _pool_backbone_output(self, output) -> torch.Tensor:
        vision_output = getattr(output, "vision_model_output", None)
        if vision_output is not None:
            return self._pool_backbone_output(vision_output)
        pooled = getattr(output, "pooler_output", None)
        if pooled is not None:
            return pooled
        image_embeds = getattr(output, "image_embeds", None)
        if image_embeds is not None:
            return image_embeds
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            if isinstance(output, (tuple, list)) and output:
                hidden = output[0]
            else:
                raise ValueError("Image backbone did not return hidden states")
        if hidden.ndim == 3 and hidden.shape[1] > 0:
            return hidden[:, 0]
        return hidden.mean(dim=1)

    def _position_embedding(self, time_steps: int) -> torch.Tensor:
        if int(time_steps) == self.frame_pos_embedding.shape[1]:
            return self.frame_pos_embedding
        pos = self.frame_pos_embedding.transpose(1, 2)
        pos = F.interpolate(pos, size=int(time_steps), mode="linear", align_corners=False)
        return pos.transpose(1, 2)

    def encode_frames(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W] pixel_values, got {tuple(pixel_values.shape)}")
        batch_size, time_steps, channels, height, width = pixel_values.shape
        flat = pixel_values.reshape(batch_size * time_steps, channels, height, width)
        chunk_size = self.frame_chunk_size if self.frame_chunk_size > 0 else flat.shape[0]
        frame_embeddings = []
        for start in range(0, flat.shape[0], chunk_size):
            output = self.backbone(pixel_values=flat[start : start + chunk_size])
            frame_embeddings.append(self._pool_backbone_output(output))
        frames = torch.cat(frame_embeddings, dim=0).reshape(batch_size, time_steps, self.visual_dim)
        return frames

    def pooled_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        frames = self.encode_frames(pixel_values)
        frames = frames + self._position_embedding(frames.shape[1]).to(dtype=frames.dtype, device=frames.device)
        cls = self.cls_token.to(dtype=frames.dtype, device=frames.device).expand(frames.shape[0], -1, -1)
        temporal = self.temporal_encoder(torch.cat([cls, frames], dim=1))
        cls_embedding = self.norm(temporal[:, 0])
        if self.temporal_pooling == "cls":
            return cls_embedding

        frame_tokens = temporal[:, 1:]
        attention_logits = self.frame_attention(frame_tokens).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=-1).unsqueeze(-1)
        attention_pool = (frame_tokens * attention_weights).sum(dim=1)
        mean_pool = frame_tokens.mean(dim=1)
        max_pool = frame_tokens.max(dim=1).values
        combined = torch.cat([cls_embedding, attention_pool, mean_pool, max_pool], dim=-1)
        return self.temporal_projection(combined)

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        visual_embedding = self.dropout(self.pooled_features(pixel_values))
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

    def _encoder_blocks(self) -> Optional[Iterable[nn.Module]]:
        candidates = [
            getattr(getattr(self.backbone, "encoder", None), "layer", None),
            getattr(getattr(self.backbone, "encoder", None), "layers", None),
            getattr(getattr(getattr(self.backbone, "vision_model", None), "encoder", None), "layers", None),
            getattr(getattr(getattr(self.backbone, "vit", None), "encoder", None), "layer", None),
        ]
        for blocks in candidates:
            if blocks is not None:
                return blocks
        return None

    def unfreeze_last_encoder_blocks(self, last_n_blocks: int) -> int:
        self.freeze_backbone()
        blocks = self._encoder_blocks()
        if blocks is None:
            return 0
        blocks = list(blocks)
        last_n_blocks = max(0, min(int(last_n_blocks), len(blocks)))
        for block in blocks[-last_n_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        return last_n_blocks


def cosine_alignment_loss(projected: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    projected = F.normalize(projected, dim=-1)
    target = F.normalize(target, dim=-1)
    return 1.0 - (projected * target).sum(dim=-1)
