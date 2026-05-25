from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional
import logging

import torch
from torch import nn
import torch.nn.functional as F

logging.getLogger("transformers").setLevel(logging.ERROR)


class MultiTaskTextModel(nn.Module):
    """RoBERTa dual-head architecture used by text_classification/model.ipynb."""

    def __init__(self, model_name: str = "roberta-base", local_files_only: bool = False) -> None:
        super().__init__()
        from transformers.utils import logging as hf_logging
        from transformers import AutoModel

        hf_logging.set_verbosity_error()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        hidden = int(self.encoder.config.hidden_size)
        self.feature_layer = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.binary_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )
        self.severity_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 4),
        )

    def encode_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        cls = hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).float()
        mean_pool = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.feature_layer(torch.cat([cls, mean_pool], dim=1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_features: bool = False,
    ):
        features = self.encode_features(input_ids=input_ids, attention_mask=attention_mask)
        binary = self.binary_head(features).squeeze(-1)
        severity = self.severity_head(features)
        if return_features:
            return binary, severity, features
        return binary, severity


class FrozenTextTeacher:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        local_files_only: bool = False,
    ) -> None:
        from transformers.utils import logging as hf_logging
        from transformers import AutoTokenizer

        hf_logging.set_verbosity_error()
        checkpoint_path = Path(checkpoint_path)
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_name = checkpoint.get("model_name", "roberta-base")
        self.device = torch.device(device)
        self.model_name = str(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=local_files_only,
        )
        self.model = MultiTaskTextModel(self.model_name, local_files_only=local_files_only)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.embedding_dim = int(self.model.encoder.config.hidden_size)

    @torch.inference_mode()
    def predict_batch(self, texts: List[str], max_length: int = 160) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        binary_logits, severity_logits, embeddings = self.model(
            encoded["input_ids"],
            encoded["attention_mask"],
            return_features=True,
        )
        binary_probs = torch.sigmoid(binary_logits)
        severity_probs = F.softmax(severity_logits, dim=-1)
        binary_confidence = torch.maximum(binary_probs, 1.0 - binary_probs)
        severity_confidence = severity_probs.max(dim=-1).values
        teacher_confidence = 0.5 * binary_confidence + 0.5 * severity_confidence
        return {
            "embedding": embeddings.detach().cpu(),
            "binary_logit": binary_logits.detach().cpu(),
            "binary_prob": binary_probs.detach().cpu(),
            "severity_logits": severity_logits.detach().cpu(),
            "severity_probs": severity_probs.detach().cpu(),
            "predicted_severity": severity_probs.argmax(dim=-1).detach().cpu(),
            "teacher_confidence": teacher_confidence.detach().cpu(),
        }

    @torch.inference_mode()
    def predict_one(self, text: str, max_length: int = 160) -> Dict[str, torch.Tensor]:
        batch = self.predict_batch([text], max_length=max_length)
        return {key: value[0] for key, value in batch.items()}
