import torch
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from RQ1.WordNonword.classification import WordNonwordClassifier

start_of_word_token = {
    "Tower-Babel/Babel-9B-Chat": "Ġ",
    "google/gemma-3-12b-it": "▁",
    "meta-llama/Llama-2-7b-chat-hf": "▁",
    }

class AttentionAnalyzer(WordNonwordClassifier):
    def __init__(self, language, tokenizer_name):
        super().__init__(language, tokenizer_name)
        self.model.eval()
        self.language = language
        self.tokenizer_name = tokenizer_name

    def get_attention_scores(self, df, target_len):
        if (self.tokenizer_name == "meta-llama/Llama-2-7b-chat-hf") and (self.language == "Korean"):
            target_len += 1 
        
        results = []
        cnt = 0
        for _, row in tqdm(df.iterrows(), total=len(df)):
            word = row["word"]
            context = row["selected_sentence"]
            
            encoding = self.tokenizer(context, return_tensors="pt", return_attention_mask=True, add_special_tokens=True)
            input_ids = encoding['input_ids'].to(self.device)
            
            if len(input_ids[0]) >= 100:
                # print(f"Input too long for word '{word}': {len(input_ids[0])} tokens. Skipping.")
                continue
                
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

            attention_mask = encoding['attention_mask'].to(self.device)
            
            word_tokens_ids = self.tokenizer(word, add_special_tokens=False)["input_ids"]
            word_tokens = self.tokenizer.convert_ids_to_tokens(word_tokens_ids)
            
            if (len(word_tokens_ids) != target_len):
                continue
            try:
                # First try: direct match of last token ID
                word_end_idx = input_ids[0].tolist().index(word_tokens_ids[-1])
            except ValueError:
                try:
                    # Second try: fallback using token strings
                    word_end_idx = tokens.index(start_of_word_token[self.tokenizer_name] + word_tokens[0])
                except ValueError:
                    # Final fallback if both methods fail
                    word_end_idx = None

            word_start_idx = word_end_idx - len(word_tokens_ids) + 1 if word_end_idx is not None else None
            
            if (word_end_idx == None):
                # print(f"Skipping word '{word}' due to invalid indices or token count: {word_start_idx}, {word_end_idx}, {len(word_tokens_ids)}")
                continue 
            
            
            with torch.no_grad():
                try:
                    outputs = self.model(input_ids, attention_mask=attention_mask, output_attentions=True, outputs_hidden_states=False)
                    attentions = outputs.attentions  # List of (batch, heads, seq_len, seq_len)
                # attentions = tuple(attn.to(dtype=torch.bfloat16) for attn in outputs.attentions)
                # attentions = [att.cpu() for att in outputs.attentions]
                except Exception as e:
                    print(f"Error processing word '{word}': {e}")
                    print("context:", context)
                    break

            layerwise_avg_attn_to_prefix = []

            for layer_idx, attn_layer in enumerate(attentions):
                # attn_to_prefix = attn_layer[0, :, word_end_idx, :word_start_idx]  # [heads, prefix_len]
                attn_to_prefix = attn_layer[0, :, word_end_idx, word_end_idx-1]
                avg_attention = attn_to_prefix.mean().item() 
                layerwise_avg_attn_to_prefix.append(avg_attention)

            cnt += 1
            results.append({
                "word": word,
                "context": context,
                "word_tokens": word_tokens,
                "query_token": tokens[word_end_idx],
                "key_token": tokens[word_end_idx-1],
                **{f"layer_{i+1}_attn": attn for i, attn in enumerate(layerwise_avg_attn_to_prefix)}
            })
            
            torch.cuda.empty_cache()

        print(f"Processed {cnt} words with valid attention scores.")
        return pd.DataFrame(results)

    def run(self, df_path, target_len=2, save_path=None):
        df = pd.read_csv(df_path)
        attn_df = self.get_attention_scores(df, target_len)

        if save_path:
            attn_df.to_csv(save_path, index=False)
        return attn_df
    

if __name__ == "__main__":
    # MODEL_NAME = "Tower-Babel/Babel-9B-Chat"
    # MODEL_NAME = "google/gemma-3-12b-it"
    MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
    LANGUAGE = "English"
    # LANGUAGE = "Korean"
    # LANGUAGE = "German"
    analyzer = AttentionAnalyzer(LANGUAGE, MODEL_NAME)

    df = analyzer.run(
        df_path=f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/ComponentAnalysis/{MODEL_NAME.split('/')[-1]}_{LANGUAGE}_wiki_noun_frequencies_context.csv",
        target_len=1, 
        save_path=f"/home/hyujang/multilingual-inner-lexicon/output/RQ1/ComponentAnalysis/attention_weights/{MODEL_NAME.split('/')[-1]}_{LANGUAGE}_single-token.csv"
    )
