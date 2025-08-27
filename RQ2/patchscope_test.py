from collections import defaultdict
# from .word_retriever import PatchscopesRetriever
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import pandas as pd
from patchscope import PatchScope
from utils import clean_memory, FFNProbe, FFNProbeGemma3

os.environ["TORCHDYNAMO_DISABLE"] = "1"
import gc
from transformers.utils import logging
logging.set_verbosity_error()


def run_patchscope_job(model_name, source_lang, target_lang, prompt_lang):
    model_basename = model_name.split("/")[-1]
    
    csv_path = f"/home/hyujang/multilingual-inner-lexicon/data/RQ2/MUSE/{source_lang}_{target_lang}_1000.csv"
    words_list = pd.read_csv(csv_path)[source_lang].tolist()
    
    patchscope = PatchScope(prompt_lang, model_name, source_lang, target_lang, num_tokens_to_generate=20)
    
    output_path = f"/home/hyujang/multilingual-inner-lexicon/output/RQ2/PatchScope/num_token_20/{model_basename}_{source_lang}_to_{target_lang}_{prompt_lang}Prompt_withOriginalCode.csv"
    patchscope.run_patchscopes_on_list(words_list=words_list, output_csv_path=output_path)


if __name__ == "__main__":
    configs = [
        # Format: (MODEL_NAME, SOURCE_LANGUAGE, TARGET_LANGUAGE, PROMPT_LANGUAGE)
        ("Tower-Babel/Babel-9B-Chat", "Korean", "English", "Korean"),
        ("google/gemma-3-12b-it", "Korean", "English", "Korean"),
        ("meta-llama/Llama-2-7b-chat-hf", "Korean", "English", "Korean"),

        ("Tower-Babel/Babel-9B-Chat", "Korean", "English", "English"),
        ("google/gemma-3-12b-it", "Korean", "English", "English"),
        ("meta-llama/Llama-2-7b-chat-hf", "Korean", "English", "English"),

        ("Tower-Babel/Babel-9B-Chat", "English", "Korean", "English"),
        ("google/gemma-3-12b-it", "English", "Korean", "English"),
        ("meta-llama/Llama-2-7b-chat-hf", "English", "Korean", "English"),

        ("Tower-Babel/Babel-9B-Chat", "English", "Korean", "Korean"),
        ("google/gemma-3-12b-it", "English", "Korean", "Korean"),
        ("meta-llama/Llama-2-7b-chat-hf", "English", "Korean", "Korean"),

        ("Tower-Babel/Babel-9B-Chat", "English", "German", "English"),
        ("google/gemma-3-12b-it", "English", "German", "English"),
        ("meta-llama/Llama-2-7b-chat-hf", "English", "German", "English"),
    ]

    for model_name, src, tgt, prompt in configs:
        print(f"\n▶ Running: {model_name} | {src} → {tgt} | Prompt: {prompt}")
        run_patchscope_job(model_name, src, tgt, prompt)
        clean_memory()
