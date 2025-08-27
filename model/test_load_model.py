from transformers import AutoModel, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
import torch

# Define quantization config with CPU offloading enabled
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,  # Use 4-bit quantization to save memory
#     bnb_4bit_compute_dtype=torch.float16,  # Compute in float16
#     bnb_4bit_use_double_quant=True,  # Additional quantization for efficiency
#     llm_int8_enable_fp32_cpu_offload=True  # Allow offloading to CPU
# )
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it", use_fast=True)

# Load model with CPU-GPU memory balancing
# model = AutoModelForImageTextToText.from_pretrained(
model = AutoModelForCausalLM.from_pretrained(
    # "google/gemma-3-12b-it",
    "Tower-Babel/Babel-9B-Chat",
    # "meta-llama/Llama-2-7b-chat-hf",
    # quantization_config=quantization_config,
    # device_map="auto"  # Distribute model automatically across available devices
    token = None
)

import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the current GPU device
    device = torch.device("cuda")
    
    # Print memory usage
    print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / 1e6} MB")
    print(f"Memory Reserved: {torch.cuda.memory_reserved(device) / 1e6} MB")
else:
    print("CUDA is not available. Running on CPU.")

print("Model loaded successfully!")
