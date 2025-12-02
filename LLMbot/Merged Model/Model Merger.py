import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Now import everything else
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to adapter (one folder up from Merged Model)
ADAPTER_PATH = os.path.join(SCRIPT_DIR, "..", "medalpaca_lora_adapter")

# Output path (in the same folder as the script)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "MergedModel")

print(f"Adapter path: {os.path.abspath(ADAPTER_PATH)}")
print(f"Output path: {os.path.abspath(OUTPUT_PATH)}")

# Verify adapter exists
if not os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
    raise FileNotFoundError(f"adapter_config.json not found in {ADAPTER_PATH}")

print("Loading base model (this may take a while)...")
base_model = AutoModelForCausalLM.from_pretrained(
    "medalpaca/medalpaca-7b",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Merging weights...")
merged_model = model.merge_and_unload()

print("Saving merged model...")
merged_model.save_pretrained(OUTPUT_PATH)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"Done! Merged model saved to {OUTPUT_PATH}")