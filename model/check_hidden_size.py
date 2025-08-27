import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForCausalLM
import numpy as np
import pandas as pd

token_1 = None

# Define a function to extract hidden states
def get_hidden_states(model, tokenizer, text, device="cuda"):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Get the hidden states (output from the model)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    return hidden_states

# Load models and tokenizers
models = [
    ("google/gemma-3-12b-it", token_1),
    ("Tower-Babel/Babel-9B-Chat", None),
    ("meta-llama/Llama-2-7b-chat-hf", token_1)
]

# Sample input text
text = "This is a sample sentence for testing hidden representations."

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Prepare for plotting
layer_sizes = {model_name: [] for model_name, _ in models}

# Loop over models and extract hidden states
for model_name, token in models:
    print(f"Processing {model_name}...")
    
    # Load the model and tokenizer
    if model_name == "google/gemma-3-12b-it":
        model = Gemma3ForCausalLM.from_pretrained(model_name, token=token).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=token)
    
    # Get hidden states
    hidden_states = get_hidden_states(model, tokenizer, text, device)
    
    # Record the size of hidden states for each layer
    for i, layer in enumerate(hidden_states):
        layer_size = layer.size()  # Shape of the tensor (batch_size, seq_len, hidden_dim)
        layer_sizes[model_name].append(layer_size[-1])  # Only take the last dimension (hidden_dim)

    del model, tokenizer, hidden_states
    torch.cuda.empty_cache()
pd.DataFrame(layer_sizes.items()).to_csv(f"{model_name.split("/")[1]}_layer_sizes.csv", index=False, header=["Model", "Hidden_Sizes"])

# Plot the results
plt.figure(figsize=(10, 6))

for model_name, sizes in layer_sizes.items():
    plt.plot(range(1, len(sizes) + 1), sizes, label=model_name)

plt.xlabel("Layer Number")
plt.ylabel("Hidden Representation Size")
plt.title("Hidden State Sizes of Different Models")
plt.legend()
plt.grid(True)
plt.show()
