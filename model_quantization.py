import dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
from huggingface_hub import login
dotenv.load_dotenv()
login(token=os.getenv("HF_TOKEN"))

model_id = "meta-llama/Llama-3.2-3B-Instruct" # meta-llama/Llama-3.2-3B-Instruct, meta-llama/Llama-3.2-3B

if torch.cuda.get_device_capability()[0] >= 8:
    !pip install -qqq flash-attn
    torch_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
else:
  torch_dtype = torch.float16
  attn_implementation = "eager"

# Update bnb_config to use 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Enable 8-bit quantization
    bnb_8bit_compute_dtype=torch_dtype,  # Set the compute dtype for 8-bit (if needed)
)

# Load model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",  # Automatically map to available devices (CPU, GPU)
    attn_implementation=attn_implementation
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.push_to_hub(f"iamraymondlow/Llama-3.2-3B-Instruct-Q8_0", token = os.getenv("HF_TOKEN")) # Online saving
tokenizer.push_to_hub(f"iamraymondlow/Llama-3.2-3B-Instruct-Q8_0", token = os.getenv("HF_TOKEN")) # Online saving