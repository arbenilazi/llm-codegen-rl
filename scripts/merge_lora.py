from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
lora_path = "results/online_grpo_mbpp"
output_path = "results/grpo_mbpp_merged"

print("Loading base model (bf16)...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, lora_path)

print("Merging LoRA weights...")
model = model.merge_and_unload()

print("Saving merged model (bf16)...")
model.save_pretrained(output_path)
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.save_pretrained(output_path)

print("Merged model saved to:", output_path)
