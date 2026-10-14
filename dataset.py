"""JSONL dataset and batch collation for conditional embedding diffusion."""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


class WatermarkRecoveryDataset(Dataset):
    """Read records where ``vanilla`` is the condition and ``prompt`` is target."""

    def __init__(self, json_path, max_samples: Optional[int] = None):
        self.rows = []
        with Path(json_path).open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row.get("vanilla"), str) or not isinstance(row.get("prompt"), str):
                    raise ValueError(
                        f"{json_path}:{line_number}: expected string fields 'vanilla' and 'prompt'"
                    )
                self.rows.append({"attack": row["vanilla"], "target": row["prompt"]})
                if max_samples is not None and len(self.rows) >= max_samples:
                    break

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class BartDiffusionCollator:
    """Tokenize conditioning inputs and expected outputs into padded tensors."""

    def __init__(
        self,
        tokenizer,
        max_attack_length: int,
        max_target_length: int,
        target_length_margin: int = 0,
        pad_token_id: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_attack_length = max_attack_length
        self.max_target_length = max_target_length
        self.target_length_margin = target_length_margin
        self.pad_token_id = tokenizer.pad_token_id if pad_token_id is None else pad_token_id

    def __call__(self, examples):
        inputs = self.tokenizer(
            [example["attack"] for example in examples],
            padding=True,
            truncation=True,
            max_length=self.max_attack_length,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            [example["target"] for example in examples],
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )
        return {
            "attack_input_ids": inputs["input_ids"],
            "attack_attention_mask": inputs["attention_mask"],
            "target_input_ids": targets["input_ids"],
            "target_attention_mask": targets["attention_mask"],
        }
