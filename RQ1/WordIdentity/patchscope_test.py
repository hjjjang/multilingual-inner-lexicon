from collections import defaultdict
# from .word_retriever import PatchscopesRetriever
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import pandas as pd
from patchscope import PatchScope
from utils import clean_memory

# os.environ["TORCHDYNAMO_DISABLE"] = ""
# import gc
from transformers.utils import logging
logging.set_verbosity_error()


def run_patchscope_job(model_name, language):
    model_basename = model_name.split("/")[-1]
    
    # csv_path = f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/WordIdentity/multi_token_{model_basename}_{language}.csv"
    # words_list = pd.read_csv(csv_path)['word'].tolist()
    csv_path = f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/WordNonword/wordnonword_{model_basename}_{language}.csv"
    df = pd.read_csv(csv_path)
    words_list = df[df['label']=='realword']["word"].tolist()
    # words_list = df[df['label']=='realword']["word"].sample(n=1000, random_state=2025).tolist()
    
    patchscope = PatchScope(language=language, 
                            tokenizer_name=model_name,
                            num_tokens_to_generate=10,
                            output_type="layer_hidden_states")

    # output_path = f"/home/hyujang/multilingual-inner-lexicon/output/RQ1/WordIdentity/layer_hidden_states/multi_token_{model_basename}_{language}.csv"
    output_path = f"/home/hyujang/multilingual-inner-lexicon/output/RQ1/WordIdentity/layer_hidden_states2/multi_token_{model_basename}_{language}.csv"

    patchscope.run_patchscopes_on_list(words_list=words_list, output_csv_path=output_path)


if __name__ == "__main__":
    configs = [
        ("Tower-Babel/Babel-9B-Chat", "English"),
        ("Tower-Babel/Babel-9B-Chat", "Korean"),
        ("Tower-Babel/Babel-9B-Chat", "German"),
        
        ("google/gemma-3-12b-it", "English"),
        ("google/gemma-3-12b-it", "Korean"),
        ("google/gemma-3-12b-it", "German"),
        
        # ("meta-llama/Llama-2-7b-chat-hf", "English"),
        # ("meta-llama/Llama-2-7b-chat-hf", "Korean"),
        # ("meta-llama/Llama-2-7b-chat-hf", "German"),    
    ]

    for model_name, language in configs:
        print(f"\n▶ Running: {model_name} | {language}")
        run_patchscope_job(model_name, language)
        clean_memory()
