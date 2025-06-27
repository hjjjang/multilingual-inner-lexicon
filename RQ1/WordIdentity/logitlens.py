import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from transformers import Gemma3ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import re
# from ..WordNonword.classification import WordNonwordClassifier
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from RQ1.WordNonword.classification import WordNonwordClassifier

class LogitLens(WordNonwordClassifier):
    def __init__(self, language, tokenizer_name):
        super().__init__(language, tokenizer_name)  # Inherit token config
        self.setup_tokenizer()
        self.model.eval().to(self.device)
        self.embedding_matrix = self.model.get_input_embeddings().weight
        self.model_name = tokenizer_name.split("/")[-1]
        
    def setup_tokenizer(self):
        if self.tokenizer_name == "Tower-Babel/Babel-9B-Chat":
            self.tokenizer.add_special_tokens({'unk_token': 'UNK'})
            self.tokenizer.unk_token_id = self.tokenizer.convert_tokens_to_ids('UNK')

    def run_logit_lens(self, df, type, distance_metric="logits", k=3, save=False):
        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            word = row["word"]
            
            if type == "simple_split":
                split_tokens = literal_eval(row["splitted_tokens"])
            elif type == "typo_split":
                split_tokens = literal_eval(row['splitted_typo_tokens'])
            
            # input_ids = self.tokenizer.convert_tokens_to_ids(split_tokens)
            if self.tokenizer.bos_token:
                input_ids = [self.tokenizer.bos_token_ids] + self.tokenizer.convert_tokens_to_ids(split_tokens)
            else:
                input_ids = self.tokenizer.convert_tokens_to_ids(split_tokens)

            if self.tokenizer_name == "Tower-Babel/Babel-9B-Chat":
                input_ids = [31883 if id is None else id for id in input_ids]
            
            # inputs = {
                # "input_ids": torch.tensor([input_ids], dtype=torch.long, device=self.device)  # Shape: (1, 2)
            # }
            
            # input_ids = self.tokenizer(split_tokens, return_tensors="pt", return_attention_mask=False)['input_ids'].to(self.device)

            inputs_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)  # Shape: (1, seq_len)
            
            with torch.no_grad():
                # outputs = self.model(**inputs, output_hidden_states=True)
                outputs = self.model(inputs_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states

            per_layer_top_token_ids = []
            per_layer_top_token_strs = []
                
            for layer_idx, h in enumerate(hidden_states[1:], start=1):  # skip embedding layer
                hidden = h[0, -1] # last token's hidden state
                
                # step1: get the similarity
                if distance_metric == 'logits':
                    logits = torch.matmul(self.embedding_matrix, hidden)
                
                elif distance_metric == 'cosine':
                    logits = F.cosine_similarity(self.embedding_matrix, hidden, dim=-1)
                    # hidden_norm = F.normalize(hidden, p=2, dim=-1)
                    # embedding_matrix_norm = F.normalize(self.embedding_matrix, p=2, dim=-1)
                    # logits = torch.matmul(embedding_matrix_norm, hidden_norm)

                # step2: get the top token id
                # top1_token_id = torch.argmax(logits).item()
                # top1_token_str = tokenizer.convert_ids_to_tokens(top1_token_id)

                topk_values, topk_indices = torch.topk(logits, k=k)
                topk_token_strs = self.tokenizer.convert_ids_to_tokens(topk_indices.tolist())
                topk_token_strs = [self.tokenizer.convert_tokens_to_string([token]) for token in topk_token_strs]
                
                per_layer_top_token_ids.append(topk_indices.tolist())
                per_layer_top_token_strs.append(topk_token_strs)

            results.append({
                "word": word,
                "split_tokens": split_tokens,
                **{f"layer_{i}_top_token_id": tid for i, tid in enumerate(per_layer_top_token_ids, start=1)},
                **{f"layer_{i}_top_token_str": tstr for i, tstr in enumerate(per_layer_top_token_strs, start=1)},
            })
            
        results_df = pd.DataFrame(results)
        
        if save:
            results_df.to_csv(f"/home/hyujang/multilingual-inner-lexicon/output/RQ1/WordIdentity/single_token_{type}_{self.model_name}_{self.language}.csv", index=False)
        return results_df

    def normalize(self, word):
        return re.sub(r'\W+', '', word).lower()  # Remove non-alphanumeric chars and lowercase

    def get_retrieval_rates(self, results_df):
        token_str_cols = [col for col in results_df.columns if col.endswith("_top_token_str")]
        retrieval_rates = [] 
        layers = []

        for col in token_str_cols:
            pred_word = results_df[col].apply(lambda x: [self.normalize(word) for word in x])
            original_word = results_df["word"].apply(self.normalize)
            retrieval_match = pred_word.combine(original_word, lambda preds, orig: orig in preds)
            retrieval_rate = retrieval_match.mean()
            retrieval_rates.append(retrieval_rate)
            # Extract layer number from column name, e.g., "layer_1_top1_token_str" -> 1
            layer_num = int(col.split("_")[1])
            layers.append(layer_num)

        # Sort by layer number
        sorted_indices = sorted(range(len(layers)), key=lambda i: layers[i])
        layers = [layers[i] for i in sorted_indices]
        retrieval_rates = [retrieval_rates[i] for i in sorted_indices]
        
        return layers, retrieval_rates

    def run(self, path1, path2, vis=True, save=False):
        
        df = pd.read_csv(path1)
        # print(df)
        results_df_original = self.run_logit_lens(df, distance_metric='logits', type='simple_split', k=3, save=save)
        layers_original, rates_original = self.get_retrieval_rates(results_df_original)

        df = pd.read_csv(path2)
        results_df_typo = self.run_logit_lens(df, distance_metric='logits', type='typo_split', k=3, save=save)
        layers_typo, rates_typo = self.get_retrieval_rates(results_df_typo)
        
        # print(results_df_original)
        if vis:
            plt.figure(figsize=(8, 5))
            plt.plot(layers_original, rates_original, marker='o', label='Splitted')
            plt.plot(layers_typo, rates_typo, marker='s', label='Typo Splitted')
            plt.xlabel("Layer")
            plt.ylabel("Original Word Retrieval Rate (Top-3)")
            plt.title(f"Top-3 Retrieval Rate per Layer ({self.model_name}, {self.language})")
            plt.ylim(0, 1)
            plt.grid(True)
            plt.legend()
            plt.show()
        # return results_df_original, results_df_typo
        return layers_original, rates_original, layers_typo, rates_typo

if __name__ == "__main__":    
    MODEL_NAME = "Tower-Babel/Babel-9B-Chat"
    LANGUAGE = "English"
    logit_lens = LogitLens(LANGUAGE, MODEL_NAME)
    MODEL_NAME = MODEL_NAME.split("/")[-1]
    results_df_original, results_df_typo = logit_lens.run(
        path1 = f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/WordIdentity/single_token_splitted_{MODEL_NAME}_{LANGUAGE}.csv",
        path2 = f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/WordIdentity/single_token_typos_{MODEL_NAME}_{LANGUAGE}.csv",
        vis = False
    )