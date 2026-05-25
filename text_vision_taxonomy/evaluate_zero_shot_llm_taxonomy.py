from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from text_vision.multilabel_taxonomy_utils import (
    LABEL_COLUMNS,
    add_prediction_columns,
    evaluate_probabilities,
    load_multilabel_dataset,
    write_json,
)


LABEL_DEFINITIONS = {
    "hate_speech": "Explicit hateful, abusive, or dehumanizing language targeting a person or group.",
    "discrimination": "Protected-class or identity-based discriminatory content.",
    "contextual_hate": "Coded, indirect, propaganda-like, extremist, or contextual hate.",
    "threat": "Direct threat, intent to harm, or threat-like coercive language.",
    "violence": "Violent action, violent intent, physical attack, weapons, or violent scene/content.",
    "fear": "Fear, panic, intimidation, terror, or fear-mongering.",
    "sexual": "Sexual, nude, pornographic, or explicit content.",
    "illegal": "Illegal activity, weapons, drugs, crime, abuse, fraud, hacking, exploitation, or trafficking.",
    "online_harm": "Online harassment, social media abuse, doxxing, cyberbullying, chat/comment/post harm.",
}

LABEL_ALIASES = {
    "hate": "hate_speech",
    "hatespeech": "hate_speech",
    "hate_speech": "hate_speech",
    "discriminatory": "discrimination",
    "contextual": "contextual_hate",
    "contextualhate": "contextual_hate",
    "threats": "threat",
    "violent": "violence",
    "fear_mongering": "fear",
    "fearmongering": "fear",
    "sex": "sexual",
    "explicit": "sexual",
    "criminal": "illegal",
    "crime": "illegal",
    "online": "online_harm",
    "onlineharm": "online_harm",
    "online_harassment": "online_harm",
}


def json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def filter_eval_rows(frame: pd.DataFrame, eval_split: str) -> pd.DataFrame:
    out = frame.copy()
    out = out[pd.to_numeric(out["synthetic"], errors="coerce").fillna(0).astype(int).eq(0)].copy()
    if eval_split == "all":
        return out
    if eval_split == "non_train":
        return out[~out["split"].astype(str).eq("train")].copy()
    return out[out["split"].astype(str).eq(eval_split)].copy()


def normalize_label(value: Any) -> str | None:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9_ -]+", "", text)
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"_+", "_", text).strip("_")
    if text in LABEL_COLUMNS:
        return text
    return LABEL_ALIASES.get(text)


def extract_json_object(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw.strip()).strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def labels_from_response(blob: Mapping[str, Any]) -> List[str]:
    values = blob.get("labels", blob.get("taxonomy_labels", blob.get("final_labels", [])))
    if isinstance(values, str):
        values = re.split(r"[,|]", values)
    if not isinstance(values, Sequence):
        return []
    labels: List[str] = []
    for value in values:
        label = normalize_label(value)
        if label and label not in labels:
            labels.append(label)
    return labels


def score_map_from_response(blob: Mapping[str, Any]) -> Dict[str, float]:
    values = blob.get("scores", blob.get("probs", blob.get("probabilities", {})))
    if not isinstance(values, Mapping):
        return {}
    out: Dict[str, float] = {}
    for key, value in values.items():
        label = normalize_label(key)
        if not label:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        out[label] = max(0.0, min(1.0, score))
    return out


def build_messages(text: str, max_text_chars: int) -> List[Dict[str, str]]:
    label_lines = "\n".join(f"- {label}: {LABEL_DEFINITIONS[label]}" for label in LABEL_COLUMNS)
    clipped = str(text or "")[: int(max_text_chars)]
    system = (
        "You are a zero-shot content moderation taxonomy annotator. "
        "Return strict JSON only. Do not include markdown. "
        "Select only labels that are directly supported by the content. "
        "If no taxonomy label is supported, return an empty labels list."
    )
    user = (
        "Classify the content into zero or more of the allowed taxonomy labels.\n\n"
        f"Allowed labels:\n{label_lines}\n\n"
        "Return this JSON schema exactly:\n"
        "{\n"
        '  "labels": ["label_name"],\n'
        '  "scores": {"hate_speech": 0.0, "discrimination": 0.0, "contextual_hate": 0.0, "threat": 0.0, "violence": 0.0, "fear": 0.0, "sexual": 0.0, "illegal": 0.0, "online_harm": 0.0},\n'
        '  "rationale": "brief evidence-based reason"\n'
        "}\n\n"
        "Do not output a neutral label; use labels=[] for neutral. "
        "For threat, illegal, and online_harm, require explicit evidence.\n\n"
        f"Content:\n{clipped}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def messages_to_plain_prompt(messages: Sequence[Mapping[str, str]]) -> str:
    return "\n\n".join(f"{item['role'].upper()}:\n{item['content']}" for item in messages) + "\n\nASSISTANT:\n"


class ZeroShotBackend:
    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        raise NotImplementedError


class OpenAIBackend(ZeroShotBackend):
    def __init__(self, model: str, temperature: float, max_tokens: int, base_url: str = "", json_mode: bool = True) -> None:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("OpenAI SDK is not installed. Install it with: pip install openai") from exc
        kwargs: Dict[str, Any] = {}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.json_mode = bool(json_mode)

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


class ChatGroqBackend(ZeroShotBackend):
    def __init__(self, model: str, temperature: float, max_tokens: int, api_key: str = "") -> None:
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_groq import ChatGroq
        except Exception as exc:
            raise RuntimeError("LangChain Groq integration is not installed. Install it with: pip install langchain-groq") from exc
        self.HumanMessage = HumanMessage
        self.SystemMessage = SystemMessage
        self.llm = ChatGroq(
            model=model,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        langchain_messages = []
        for item in messages:
            role = str(item.get("role", "")).lower()
            content = str(item.get("content", ""))
            if role == "system":
                langchain_messages.append(self.SystemMessage(content=content))
            else:
                langchain_messages.append(self.HumanMessage(content=content))
        response = self.llm.invoke(langchain_messages)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return "\n".join(str(part.get("text", part)) if isinstance(part, Mapping) else str(part) for part in content)
        return str(content or "")


class OllamaBackend(ZeroShotBackend):
    def __init__(self, model: str, endpoint: str, temperature: float, max_tokens: int) -> None:
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        request = urllib.request.Request(
            f"{self.endpoint}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                blob = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        return str(blob.get("message", {}).get("content", ""))


class HFBackend(ZeroShotBackend):
    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        max_input_tokens: int,
        local_files_only: bool,
        trust_remote_code: bool,
        device_map: str,
        device: str,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            local_files_only=bool(local_files_only),
            trust_remote_code=bool(trust_remote_code),
        )
        kwargs: Dict[str, Any] = {
            "local_files_only": bool(local_files_only),
            "trust_remote_code": bool(trust_remote_code),
        }
        if device_map:
            kwargs["device_map"] = device_map
            kwargs["torch_dtype"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        self.device_map = device_map
        if not device_map:
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model.to(self.device)
        else:
            self.device = None
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.max_input_tokens = int(max_input_tokens)

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        import torch

        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
        else:
            prompt = messages_to_plain_prompt(messages)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        target_device = self.device if self.device is not None else getattr(self.model, "device", None)
        if target_device is not None:
            encoded = {key: value.to(target_device) for key, value in encoded.items()}
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
        with torch.inference_mode():
            output = self.model.generate(**encoded, **generation_kwargs)
        input_length = int(encoded["input_ids"].shape[-1])
        new_tokens = output[0][input_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


class FixtureBackend(ZeroShotBackend):
    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        text = " ".join(item.get("content", "") for item in messages).lower()
        labels: List[str] = []
        keyword_map = {
            "hate_speech": ["hate", "hateful", "slur", "dehuman"],
            "discrimination": ["race", "religion", "ethnic", "minority", "discrimin"],
            "contextual_hate": ["extremist", "supremacist", "propaganda"],
            "threat": ["threat", "kill", "attack", "shoot"],
            "violence": ["fight", "violence", "weapon", "blood", "assault"],
            "fear": ["fear", "panic", "terror", "scared"],
            "sexual": ["sexual", "nude", "porn"],
            "illegal": ["illegal", "drug", "crime", "fraud"],
            "online_harm": ["online", "comment", "post", "dox", "harass"],
        }
        for label, terms in keyword_map.items():
            if any(term in text for term in terms):
                labels.append(label)
        scores = {label: (1.0 if label in labels else 0.0) for label in LABEL_COLUMNS}
        return json.dumps({"labels": labels, "scores": scores, "rationale": "fixture backend for smoke testing"})


def build_backend(args: argparse.Namespace) -> ZeroShotBackend:
    if args.provider == "openai":
        return OpenAIBackend(
            args.model,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            base_url=args.openai_base_url,
            json_mode=not args.no_json_mode,
        )
    if args.provider == "groq":
        return ChatGroqBackend(
            args.model,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            api_key=args.groq_api_key,
        )
    if args.provider == "ollama":
        return OllamaBackend(
            args.model,
            endpoint=args.ollama_endpoint,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )
    if args.provider == "hf":
        return HFBackend(
            args.model,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
            device_map=args.device_map,
            device=args.device,
        )
    if args.provider == "fixture":
        return FixtureBackend()
    raise ValueError(f"Unsupported provider: {args.provider}")


def predictions_from_existing(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_progress(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.csv"
    raw_jsonl_path = output_dir / "raw_responses.jsonl"
    frame = load_multilabel_dataset(
        args.dataset,
        seed=args.seed,
        split_path=args.splits if args.splits else None,
    )
    frame = filter_eval_rows(frame, args.eval_split)
    if args.limit is not None:
        frame = frame.head(int(args.limit)).copy()
    if frame.empty:
        raise ValueError("No evaluation rows selected")

    existing = predictions_from_existing(prediction_path) if args.resume else pd.DataFrame()
    completed = set(existing["row_id"].astype(int).tolist()) if not existing.empty and "row_id" in existing.columns else set()
    rows: List[Dict[str, Any]] = existing.to_dict("records") if not existing.empty else []
    backend = build_backend(args)
    raw_handle = raw_jsonl_path.open("a", encoding="utf-8") if args.resume else raw_jsonl_path.open("w", encoding="utf-8")
    try:
        for processed, (_, row) in enumerate(frame.iterrows(), start=1):
            row_id = int(row["row_id"])
            if row_id in completed:
                continue
            messages = build_messages(str(row["text_input"]), max_text_chars=args.max_text_chars)
            error = ""
            raw = ""
            parsed: Dict[str, Any] = {}
            labels: List[str] = []
            scores: Dict[str, float] = {}
            try:
                raw = backend.generate(messages)
                parsed = extract_json_object(raw)
                labels = labels_from_response(parsed)
                scores = score_map_from_response(parsed)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            probabilities = []
            for label in LABEL_COLUMNS:
                if args.use_confidence_scores and label in scores:
                    value = float(scores[label])
                    if label in labels:
                        value = max(value, 0.5)
                else:
                    value = 1.0 if label in labels else 0.0
                probabilities.append(value)
            prediction = add_prediction_columns(
                pd.DataFrame([row.to_dict()]),
                np.array([probabilities], dtype=np.float32),
                {label: 0.5 for label in LABEL_COLUMNS},
            ).iloc[0].to_dict()
            prediction.update(
                {
                    "system": args.system_name or f"zero_shot_{args.provider}_{args.model}".replace("/", "__"),
                    "provider": args.provider,
                    "model": args.model,
                    "predicted_labels": "|".join(labels),
                    "scores_json": json.dumps(scores, ensure_ascii=False),
                    "parse_ok": int(bool(parsed) and not error),
                    "error": error,
                    "raw_response": raw[: int(args.max_raw_chars)],
                }
            )
            rows.append(prediction)
            raw_handle.write(
                json.dumps(
                    {
                        "row_id": row_id,
                        "file_name": row.get("File Name", ""),
                        "split": row.get("split", ""),
                        "messages": messages if args.save_prompts else None,
                        "raw_response": raw,
                        "parsed": parsed,
                        "labels": labels,
                        "error": error,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            raw_handle.flush()
            if processed % int(args.save_every) == 0:
                write_progress(prediction_path, rows)
            if args.sleep_seconds > 0:
                time.sleep(float(args.sleep_seconds))
    finally:
        raw_handle.close()
    predictions = pd.DataFrame(rows)
    predictions.to_csv(prediction_path, index=False)

    thresholds = {label: 0.5 for label in LABEL_COLUMNS}
    summary, per_label = evaluate_probabilities(predictions, thresholds, system=args.system_name or f"zero_shot_{args.provider}")
    metrics = pd.DataFrame([summary])
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    per_label.to_csv(output_dir / "per_label_metrics.csv", index=False)
    write_json(
        output_dir / "metrics.json",
        {
            "summary": json_safe(summary),
            "provider": args.provider,
            "model": args.model,
            "rows": int(len(predictions)),
            "eval_split": args.eval_split,
            "thresholds": thresholds,
        },
    )
    comparison_rows = [summary]
    transformer_metrics_path = Path(args.transformer_metrics)
    if transformer_metrics_path.exists():
        blob = json.loads(transformer_metrics_path.read_text(encoding="utf-8"))
        for item in blob.get("metrics", []):
            comparison_rows.append(item)
    pd.DataFrame(comparison_rows).to_csv(output_dir / "comparison_metrics.csv", index=False)
    print(metrics.to_string(index=False))
    print("\nlabel | true_count | predicted_count | precision | recall | f1")
    print(per_label[["label", "true_count", "predicted_count", "precision", "recall", "f1"]].to_string(index=False))
    print(json.dumps({"output_dir": str(output_dir), "predictions": str(prediction_path)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot LLM taxonomy evaluation for comparison against trained models.")
    parser.add_argument("--dataset", default=r"D:\hvc\datasets\action\archieve\final_dataset.csv")
    parser.add_argument("--splits", default="")
    parser.add_argument("--eval-split", choices=["test", "val", "non_train", "all"], default="test")
    parser.add_argument("--provider", choices=["openai", "groq", "hf", "ollama", "fixture"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-name", default="")
    parser.add_argument("--output-dir", default="reports/zero_shot_llm_eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument("--max-text-chars", type=int, default=6000)
    parser.add_argument("--max-raw-chars", type=int, default=4000)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--use-confidence-scores", action="store_true")
    parser.add_argument("--transformer-metrics", default="reports/transformer_multilabel_feedback_eval/metrics.json")
    parser.add_argument("--openai-base-url", default="")
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--groq-api-key", default="")
    parser.add_argument("--ollama-endpoint", default="http://localhost:11434")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
