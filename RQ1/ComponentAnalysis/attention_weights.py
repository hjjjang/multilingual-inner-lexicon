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


class AttentionAnalyzer(WordNonwordClassifier):
    def __init__(self, language, tokenizer_name):
        super().__init__(language, tokenizer_name)
        self.model.eval()
        self.model_name = tokenizer_name.split("/")[-1]

    def get_attention_scores(self, df):
        results = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            word = row["word"]
            context = row["context"]
            
            # encoding = self.tokenizer(word, return_tensors="pt", return_attention_mask=True)
            encoding = self.tokenizer(context, return_tensors="pt", return_attention_mask=True)
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            word_tokens = self.tokenizer(word, add_special_tokens=False)["input_ids"]
            word_start_idx = None
            for i in range(len(input_ids[0]) - len(word_tokens) + 1):
                if input_ids[0, i:i + len(word_tokens)].tolist() == word_tokens:
                    word_start_idx = i
                    break
            if word_start_idx is None:
                # Skip if the word is not found in the tokenized context
                print(f"Word '{word}' not found in context: {context}")
                continue
            
            word_end_idx = word_start_idx + len(word_tokens) - 1

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask, output_attentions=True)
                attentions = outputs.attentions  # List of (batch, heads, seq_len, seq_len)

            # layerwise_avg_attn_to_prefix = []
            
            # seq_len = input_ids.shape[1]
            # final_token_idx = seq_len - 1

            # for layer_idx, attn_layer in enumerate(attentions):
            #     attn_last_token = attn_layer[0, :, final_token_idx, :final_token_idx]  # [heads, prefix_len]
            #     avg_attention = attn_last_token.mean().item()
            #     layerwise_avg_attn_to_prefix.append(avg_attention)
            
            layerwise_avg_attn_to_prefix = []

            for layer_idx, attn_layer in enumerate(attentions):
                # Compute attention weights of the final token of the word to all prefix tokens
                attn_to_prefix = attn_layer[0, :, word_end_idx, :word_start_idx]  # [heads, prefix_len]
                avg_attention = attn_to_prefix.mean().item()
                layerwise_avg_attn_to_prefix.append(avg_attention)


            results.append({
                "word": word,
                **{f"layer_{i}_attn": attn for i, attn in enumerate(layerwise_avg_attn_to_prefix)}
            })

        return pd.DataFrame(results)

    def run(self, df_path, save_path=None):
        df = pd.read_csv(df_path)
        attn_df = self.get_attention_scores(df)

        if save_path:
            attn_df.to_csv(save_path, index=False)
        return attn_df


if __name__ == "__main__":
    MODEL_NAME = "Tower-Babel/Babel-9B-Chat"
    LANGUAGE = "English"
    analyzer = AttentionAnalyzer(LANGUAGE, MODEL_NAME)

    df_split, df_typo = analyzer.run(
        path=f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/WordNonword/r1_dataset_{MODEL_NAME.split('/')[-1]}_{LANGUAGE}-wiki-2token.csv",
        save_path=f"/home/hyujang/multilingual-inner-lexicon/output/RQ1/ComponentAnalysis/attention_weights/{MODEL_NAME.split('/')[-1]}_{LANGUAGE}.csv"
    )
