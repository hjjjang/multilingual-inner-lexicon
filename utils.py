import json
import os
import sys
from typing import Dict, Any, Optional, Tuple, Union, List
from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma3ForConditionalGeneration, Qwen2ForCausalLM
import pandas as pd
import torch
from tqdm import tqdm

# Base directory for the project
BASE_DIR = "/home/hyujang/multilingual-inner-lexicon"

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load main configuration file."""
    if config_path is None:
        config_path = os.path.join(BASE_DIR, "RQ1/config.json")
    
    with open(config_path, "r") as f:
        return json.load(f)

def load_user_config(user_config_path: str = None) -> Dict[str, Any]:
    """Load user configuration file with tokens."""
    if user_config_path is None:
        user_config_path = os.path.join(BASE_DIR, "user_config.json")
    
    try:
        with open(user_config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: user_config.json not found")
        return {"huggingface_token": {}}

def get_token_value(tokenizer_name: str, config: Dict[str, Any] = None, user_config: Dict[str, Any] = None) -> Optional[str]:
    """Get token value for a specific tokenizer."""
    if config is None:
        config = load_config()
    if user_config is None:
        user_config = load_user_config()
    
    token_key = config.get("tokenizers", {}).get(tokenizer_name)
    if token_key:
        return user_config.get("huggingface_token", {}).get(token_key)
    return None

def setup_tokenizer(tokenizer_name: str, use_fast: bool = True) -> AutoTokenizer:
    """Setup tokenizer with proper token handling and special configurations."""
    token_value = get_token_value(tokenizer_name)
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, 
        use_fast=use_fast, 
        token=token_value
    )
    
    return tokenizer

def setup_model(tokenizer_name: str, device: Optional[torch.device] = None):
    """Setup model with appropriate dtype and device handling."""
    if device is None:
        device = get_device()
    
    token_value = get_token_value(tokenizer_name)
    dtype = get_model_dtype(tokenizer_name)
        
    # Load model based on tokenizer type
    if tokenizer_name == "google/gemma-3-12b-it":
        model = Gemma3ForConditionalGeneration.from_pretrained(
            tokenizer_name, 
            token=token_value, 
            torch_dtype=dtype
        )
        model.to(device)
    elif tokenizer_name == "meta-llama/Llama-2-7b-chat-hf":
        model = AutoModelForCausalLM.from_pretrained(
            tokenizer_name, 
            token=token_value, 
            torch_dtype=dtype
        )
        model.to(device)
    elif tokenizer_name == "Qwen/Qwen2.5-VL-7B-Instruct":
        model = Qwen2ForCausalLM.from_pretrained(
            tokenizer_name, 
            token=token_value, 
            torch_dtype=dtype
        )
        model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            tokenizer_name, 
            token=token_value, 
            torch_dtype=dtype
        )
    
    print(f"Model {tokenizer_name} with dtype {dtype} loaded successfully on {device}")
    return model

def extract_token_i_hidden_states(
    model, 
    tokenizer, 
    inputs: Union[str, List[str]], 
    tokenizer_name: str,
    device: torch.device,
    token_idx_to_extract: int = -1, 
    layers_to_extract: Optional[List[int]] = None
) -> Dict[int, torch.Tensor]:
    """
    Extract hidden states for tokenized words.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        inputs: String or list of strings to extract hidden states from
        tokenizer_name: Name of the tokenizer (for model-specific config)
        device: Device to run inference on
        token_idx_to_extract: Token position to extract (-1 for last token)
        layers_to_extract: List of layer indices to extract from
    
    Returns:
        Dictionary mapping layer indices to hidden states tensors
    """
    model.eval()

    if isinstance(inputs, str):
        inputs = [inputs]

    if layers_to_extract is None:
        if tokenizer_name == "google/gemma-3-12b-it":
            layers_to_extract = list(range(1, model.config.text_config.num_hidden_layers + 1))
        else:
            layers_to_extract = list(range(1, model.config.num_hidden_layers + 1))

    all_hidden_states = {layer: [] for layer in layers_to_extract}

    with torch.no_grad():
        for tokens in tqdm(inputs, desc="Extracting hidden states"):
            if isinstance(tokens, str):
                # Handle encoding errors by converting tokens back to string
                tokens = tokenizer.convert_tokens_to_string([tokens])
                input_ids = tokenizer(tokens, return_tensors="pt", return_attention_mask=False)['input_ids'].to(device)
            else:
                raise ValueError("Input should be a word not a list of tokenized tokens.")

            outputs = model(input_ids, output_hidden_states=True)
            for layer in layers_to_extract:
                hidden_states = outputs.hidden_states[layer]  # Shape: (1, seq_len, hidden_dim)
                all_hidden_states[layer].append(hidden_states[:, token_idx_to_extract, :].detach().cpu())

    for layer in all_hidden_states:
        all_hidden_states[layer] = torch.cat(all_hidden_states[layer], dim=0)

    return all_hidden_states



def setup_import_path():
    """Setup import path for cross-module imports."""
    if BASE_DIR not in sys.path:
        sys.path.append(BASE_DIR)

def ensure_output_dir(output_path: str):
    """Ensure output directory exists."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

def get_model_mappings() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Get model name mappings."""
    model_name_map = {
        "llama_2_7b": "Llama-2-7b-chat-hf",
        "babel_9b": "Babel-9B-Chat",
        "gemma_12b": "gemma-3-12b-it"
    }
    
    model_full_name_map = {
        "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
        "Babel-9B-Chat": "Tower-Babel/Babel-9B-Chat",
        "gemma-3-12b-it": "google/gemma-3-12b-it"
    }
    
    return model_name_map, model_full_name_map


def sample_by_frequency(df: pd.DataFrame, num_samples: int, num_quantiles: int = 5, 
                       freq_column: str = 'freq', seed: int = 2025) -> pd.DataFrame:
    """Sample dataframe by frequency quantiles."""
    df['freq_quantile'], bins = pd.qcut(
        df[freq_column], 
        num_quantiles, 
        labels=False, 
        duplicates='drop', 
        retbins=True
    )
    
    samples_per_quantile = num_samples // num_quantiles
    sampled = []
    
    for quantile in range(num_quantiles):
        quantile_df = df[df['freq_quantile'] == quantile]
        if len(quantile_df) > 0:
            sample_size = min(len(quantile_df), samples_per_quantile)
            sampled.append(
                quantile_df.sample(sample_size, replace=False, random_state=seed)
            )
    
    sampled_df = pd.concat(sampled, ignore_index=False).drop_duplicates(subset=['word'])
    
    # Fill remaining samples if needed
    if len(sampled_df) < num_samples:
        remaining = num_samples - len(sampled_df)
        other_df = df.drop(sampled_df.index, errors='ignore')
        if len(other_df) > 0:
            additional_samples = other_df.sample(
                min(len(other_df), remaining), 
                replace=False, 
                random_state=seed
            )
            sampled_df = pd.concat([sampled_df, additional_samples]).drop_duplicates(subset=['word'])
    
    return sampled_df.reset_index(drop=True)

def setup_model_dtype_mapping():
    """Get model-specific dtype configurations for memory optimization."""
    import torch
    
    return {
        "google/gemma-3-12b-it": torch.bfloat16,
        "meta-llama/Llama-2-7b-chat-hf": torch.float16,
        "google/gemma-2-9b-it": torch.bfloat16,
        "Qwen/Qwen2.5-VL-7B-Instruct": torch.bfloat16,
        "Tower-Babel/Babel-9B-Chat": torch.bfloat16,
        "default": torch.bfloat16
    }

def get_model_dtype(model_name: str):
    """Get appropriate dtype for a model."""
    dtype_mapping = setup_model_dtype_mapping()
    return dtype_mapping.get(model_name, dtype_mapping["default"])

def clean_memory():
    """Clean GPU memory and garbage collection."""
    import torch
    import gc
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def get_device():
    """Get appropriate device (CUDA or CPU)."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
