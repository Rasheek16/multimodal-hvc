from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from ..config import source_vocab_size


class FusionClassifier(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        text_embedding_dim: int = 768,
        source_embedding_dim: int = 8,
        hidden_dim: int = 512,
        dropout: float = 0.35,
        text_dropout: float = 0.35,
        num_taxonomy_labels: int = 0,
    ) -> None:
        super().__init__()
        self.text_dropout = float(text_dropout)
        self.num_taxonomy_labels = int(num_taxonomy_labels)
        self.source_embedding = nn.Embedding(source_vocab_size(), int(source_embedding_dim))
        input_dim = (
            int(visual_dim)
            + int(text_embedding_dim)
            + int(source_embedding_dim)
            + 1  # has_text
            + 1  # text binary probability
            + 4  # text severity distribution
            + 4  # visual severity distribution
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.binary_head = nn.Linear(int(hidden_dim) // 2, 1)
        self.severity_head = nn.Linear(int(hidden_dim) // 2, 4)
        self.taxonomy_head = nn.Linear(int(hidden_dim) // 2, self.num_taxonomy_labels) if self.num_taxonomy_labels > 0 else None

    def forward(
        self,
        visual_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        text_binary_prob: torch.Tensor,
        text_severity_probs: torch.Tensor,
        has_text: torch.Tensor,
        transcript_source_id: torch.Tensor,
        visual_severity_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        has_text = has_text.float().view(-1, 1)
        keep_text = has_text
        if self.training and self.text_dropout > 0:
            random_keep = (torch.rand_like(has_text) >= self.text_dropout).float()
            keep_text = keep_text * random_keep

        text_embedding = text_embedding * keep_text
        text_binary_prob = text_binary_prob.float().view(-1, 1) * keep_text + 0.5 * (1.0 - keep_text)
        text_severity_probs = text_severity_probs.float() * keep_text + torch.tensor(
            [0.25, 0.25, 0.25, 0.25],
            device=text_severity_probs.device,
            dtype=text_severity_probs.dtype,
        ).view(1, 4) * (1.0 - keep_text)

        source_features = self.source_embedding(transcript_source_id.long())
        visual_severity_probs = F.softmax(visual_severity_logits, dim=-1)
        features = torch.cat(
            [
                visual_embedding,
                text_embedding,
                source_features,
                has_text,
                text_binary_prob,
                text_severity_probs,
                visual_severity_probs,
            ],
            dim=-1,
        )
        hidden = self.trunk(features)
        binary_logits = self.binary_head(hidden).squeeze(-1)
        severity_logits = self.severity_head(hidden)
        result = {
            "binary_logits": binary_logits,
            "severity_logits": severity_logits,
            "fusion_embedding": hidden,
        }
        if self.taxonomy_head is not None:
            result["taxonomy_logits"] = self.taxonomy_head(hidden)
        return result


class GatedFusionClassifier(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        text_embedding_dim: int = 768,
        source_embedding_dim: int = 8,
        hidden_dim: int = 512,
        dropout: float = 0.35,
        text_dropout: float = 0.35,
        num_taxonomy_labels: int = 0,
    ) -> None:
        super().__init__()
        self.text_dropout = float(text_dropout)
        self.num_taxonomy_labels = int(num_taxonomy_labels)
        self.source_embedding = nn.Embedding(source_vocab_size(), int(source_embedding_dim))
        self.visual_branch = nn.Sequential(
            nn.LayerNorm(int(visual_dim)),
            nn.Linear(int(visual_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.text_branch = nn.Sequential(
            nn.LayerNorm(int(text_embedding_dim)),
            nn.Linear(int(text_embedding_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        metadata_dim = int(source_embedding_dim) + 1 + 1 + 4 + 4
        self.metadata_branch = nn.Sequential(
            nn.LayerNorm(metadata_dim),
            nn.Linear(metadata_dim, int(hidden_dim) // 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.gate = nn.Sequential(
            nn.Linear(int(hidden_dim) * 2 + int(hidden_dim) // 2, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Sigmoid(),
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(int(hidden_dim) * 3 + int(hidden_dim) // 2),
            nn.Linear(int(hidden_dim) * 3 + int(hidden_dim) // 2, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.binary_head = nn.Linear(int(hidden_dim) // 2, 1)
        self.severity_head = nn.Linear(int(hidden_dim) // 2, 4)
        self.taxonomy_head = nn.Linear(int(hidden_dim) // 2, self.num_taxonomy_labels) if self.num_taxonomy_labels > 0 else None

    def forward(
        self,
        visual_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
        text_binary_prob: torch.Tensor,
        text_severity_probs: torch.Tensor,
        has_text: torch.Tensor,
        transcript_source_id: torch.Tensor,
        visual_severity_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        has_text = has_text.float().view(-1, 1)
        keep_text = has_text
        if self.training and self.text_dropout > 0:
            random_keep = (torch.rand_like(has_text) >= self.text_dropout).float()
            keep_text = keep_text * random_keep

        text_embedding = text_embedding * keep_text
        text_binary_prob = text_binary_prob.float().view(-1, 1) * keep_text + 0.5 * (1.0 - keep_text)
        neutral_severity = torch.tensor(
            [0.25, 0.25, 0.25, 0.25],
            device=text_severity_probs.device,
            dtype=text_severity_probs.dtype,
        ).view(1, 4)
        text_severity_probs = text_severity_probs.float() * keep_text + neutral_severity * (1.0 - keep_text)

        visual_severity_probs = F.softmax(visual_severity_logits, dim=-1)
        source_features = self.source_embedding(transcript_source_id.long())
        metadata = torch.cat(
            [
                source_features,
                has_text,
                text_binary_prob,
                text_severity_probs,
                visual_severity_probs,
            ],
            dim=-1,
        )

        visual_features = self.visual_branch(visual_embedding)
        text_features = self.text_branch(text_embedding)
        metadata_features = self.metadata_branch(metadata)
        gate = self.gate(torch.cat([visual_features, text_features, metadata_features], dim=-1))
        fused = gate * text_features + (1.0 - gate) * visual_features
        hidden = self.trunk(torch.cat([fused, visual_features, text_features, metadata_features], dim=-1))
        binary_logits = self.binary_head(hidden).squeeze(-1)
        severity_logits = self.severity_head(hidden)
        result = {
            "binary_logits": binary_logits,
            "severity_logits": severity_logits,
            "fusion_embedding": hidden,
        }
        if self.taxonomy_head is not None:
            result["taxonomy_logits"] = self.taxonomy_head(hidden)
        return result
