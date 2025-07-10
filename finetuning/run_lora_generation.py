from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

ckpt = "finetuning/checkpoints/lora_crosslob"

cfg   = PeftConfig.from_pretrained(ckpt)
base  = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, torch_dtype="auto")
model = PeftModel.from_pretrained(base, ckpt)     # attaches LoRA

tok = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path, use_fast=True)
tok.pad_token = tok.eos_token

prompt = "Design a 45-minute squash drill session for an advanced player that focuses on: lobs.\n\nReturn **only** JSON in the exact format provided during fine-tuning."

inputs = tok(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=400)
print(tok.decode(outputs[0], skip_special_tokens=True))
