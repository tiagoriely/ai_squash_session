#!/usr/bin/env python
"""
run_lora.py ▸ Minimal yet complete LoRA / QLoRA finetuning entry‑point.

Usage
-----
python finetuning/scripts/run_lora.py --config finetuning/configs/lora_qlora.yaml

* The YAML config drives everything; this script stays generic.
* Keys under **training_arguments:** in the YAML are mapped 1‑to‑1 onto
  🤗 `TrainingArguments`.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import yaml
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path | str) -> Dict[str, Any]:
    """Read a YAML file and return a plain dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def prepare_dataset(
    data_path: Path | str,
    tokenizer,
    prompt_key: str = "prompt",
    completion_key: str = "completion",
    max_length: int = 1024,
) -> Dataset:
    """Load a JSONL dataset and tokenize it for causal‑LM training."""
    LOGGER.info("Loading dataset from %s", data_path)

    ds = load_dataset("json", data_files=str(data_path), split="train")

    MAX_LEN = 512  # ↓ from 1024

    def _tokenize(ex):
        text = f"{ex[prompt_key]}\n\n{ex[completion_key]}"
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LEN,
            padding=False  # dynamic, not always-1024
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens


    return ds.map(_tokenize, remove_columns=ds.column_names)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---------- CLI & YAML ----------
    parser = argparse.ArgumentParser(description="LoRA / QLoRA finetuning runner")
    parser.add_argument("--config", required=True,
                        help="Path to YAML config (see sample in finetuning/configs/)")
    args = parser.parse_args()

    cfg      = load_yaml(args.config)
    ta_cfg   = cfg.get("training_arguments", {})
    lora_cfg = cfg.get("training", {})

    # ---------- Model & tokenizer ----------
    model_name   = cfg.get("model_name_or_path", "mistralai/Mistral-7B-v0.3")
    dataset_path = cfg["dataset_path"]
    LOGGER.info("Model: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token  # reasonable default

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        low_cpu_mem_usage=False
    )

    # ---------- Apply LoRA ----------
    lora_config = LoraConfig(
        r=lora_cfg.get("lora_r", 8),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------- Dataset ----------
    train_ds = prepare_dataset(dataset_path, tokenizer)

    # ---------- TrainingArguments ----------
    targs = TrainingArguments(
        output_dir                 = cfg.get("output_dir", "finetuning/checkpoints/lora"),
        per_device_train_batch_size= ta_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps= ta_cfg.get("gradient_accumulation_steps", 1),
        learning_rate           =float(ta_cfg.get("learning_rate", 2e-4)),
        num_train_epochs           = ta_cfg.get("num_train_epochs", 3),
        fp16                       = ta_cfg.get("fp16", False),
        bf16                       = ta_cfg.get("bf16", False),
        logging_steps              = ta_cfg.get("logging_steps", 50),
        save_strategy              = ta_cfg.get("save_strategy", "steps"),
        save_steps                 = ta_cfg.get("save_steps", 500),
        save_total_limit           = ta_cfg.get("save_total_limit", 3),
        optim                      = "adamw_torch",
        report_to                  = "none",
    )

    LOGGER.debug("fp16=%s  bf16=%s", targs.fp16, targs.bf16)

    # ---------- Trainer ----------
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=data_collator,
    )

    LOGGER.info("Starting training…")
    trainer.train()
    LOGGER.info("Training complete. Saving…")

    trainer.save_model(targs.output_dir)
    tokenizer.save_pretrained(targs.output_dir)
    LOGGER.info("All done — LoRA checkpoint at %s", targs.output_dir)


if __name__ == "__main__":
    main()
