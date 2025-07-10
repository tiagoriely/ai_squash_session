# Finetuning / LoRA workflow for *ai\_squash\_session*

> **TL;DR** — Run `python finetuning/scripts/run_lora.py --config finetuning/configs/lora_qlora.yaml` to train a TinyLlama‑1.1 B LoRA adapter on your squash‑session JSON pairs, then `python finetuning/run_lora_generation.py` (or the snippet below) to generate new drills.

---

## 1. Directory layout

```
finetuning/
├─ configs/            # YAML configs (training & inference)
│  └─ lora_qlora.yaml  # ← master config used by run_lora.py
├─ data/
│  ├─ finetune_splits/
│  │  ├─ train_instruct.jsonl    #  your "prompt ↔ completion" pairs
│  │  ├─ valid_instruct.jsonl    #  (currently empty – optional)
│  │  └─ test_instruct.jsonl     #  held‑out set
│  └─ template_finetuning.json   #  JSON schema reference
├─ checkpoints/         # saved LoRA adapters
│  └─ lora_crosslob/     # 1st run output (adapter_*.safetensors, etc.)
├─ scripts/              # data prep + training utilities
│  ├─ parse_docx_to_json.py   # DOCX → JSON extractor
│  ├─ make_pairs.py          # builds prompt/completion pairs
│  ├─ split_pairs.py         # train/valid/test splitter
│  └─ run_lora.py            # ***main training entry‑point***
└─ run_lora_generation.py    # minimal generation helper
```

---

## 2. Preparing data

```bash
# 1) Extract sessions from Word docs → raw JSON records
python finetuning/scripts/parse_docx_to_json.py

# 2) Wrap each record into a prompt/completion pair
python finetuning/scripts/make_pairs.py \
       --in  finetuning/data/train.jsonl \
       --out finetuning/data/finetune_splits/pairs.jsonl

# 3) Split pairs into train / valid / test (80‑10‑10)
python finetuning/scripts/split_pairs.py
```

*All three steps are idempotent – feel free to rerun after adding more DOCX files.*

---

## 3. Training

*Config:* `finetuning/configs/lora_qlora.yaml`

Key knobs already tuned for an Apple‑Silicon MacBook:

| parameter                     | value                    | note                                    |
| ----------------------------- | ------------------------ | --------------------------------------- |
| `model_name_or_path`          | TinyLlama‑1.1B‑Chat‑v1.0 | fits in 9 GB MPS pool                   |
| `per_device_train_batch_size` | **1**                    | smallest batch to avoid OOM             |
| `gradient_accumulation_steps` | 4                        | keeps effective batch = 4               |
| `fp16 / bf16`                 | `false`                  | CPU/MPS can’t do fp16                   |
| `num_train_epochs`            | 3 (demo)                 | bump to 6–10 once you have ≥ 50 samples |

```bash
python finetuning/scripts/run_lora.py \
       --config finetuning/configs/lora_qlora.yaml
```

*Runtime* on M‑series ≈ 1 h per 1 000 steps; console shows a tqdm bar.

Training artefacts land in `finetuning/checkpoints/<run‑name>`.

---

## 4. Generation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

CKPT = "finetuning/checkpoints/lora_crosslob"

cfg   = PeftConfig.from_pretrained(CKPT)
base  = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path,
                                             torch_dtype="auto")
model = PeftModel.from_pretrained(base, CKPT)
model.eval()

 tok = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path, use_fast=True)
 tok.pad_token = tok.eos_token

prompt = (
    "Design a 45‑minute squash drill session for an advanced player that focuses on: lobs."\
    "\n\nReturn **only** JSON in the exact format provided during fine‑tuning."
)
inputs  = tok(prompt, return_tensors="pt")
out_ids = model.generate(**inputs, max_new_tokens=400)
print(tok.decode(out_ids[0], skip_special_tokens=True))
```

Tip: add a *blank schema* to the prompt or use RAG to inject one of your gold JSON examples for more reliable structure.

---

## 5. Evaluation

```
python evaluation/eval_ragas.py \
       --pred finetuning/checkpoints/lora_crosslob/preds.jsonl \
       --gold finetuning/data/finetune_splits/test_instruct.jsonl
```

---

## 6. Troubleshooting cheatsheet

| symptom                                   | likely cause               | quick fix                                  |
| ----------------------------------------- | -------------------------- | ------------------------------------------ |
| **`MPS backend out of memory`**           | batch or sequence too big  | `batch_size=1`, `max_length=256`           |
| `unexpected keyword... TrainingArguments` | transformers version < arg | delete that field or update 🤗 libs        |
| Output not valid JSON                     | too few training pairs     | grow dataset to 70–100 or use RAG template |

---

## 7. Next steps

* **Add more DOCX drills** → rerun data pipeline → train 5–10 epochs.
* Attach the LoRA to the retrieval‑augmented pipeline in `rag/pipelines/`.
* Push adapter to HuggingFace Hub for easy sharing.

Questions / bugs? Open an issue or ping `@your‑name`. Happy drilling! 🎾
