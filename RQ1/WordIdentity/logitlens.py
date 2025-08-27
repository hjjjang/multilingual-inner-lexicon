import os
import re
import sys
from ast import literal_eval

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils import setup_tokenizer, setup_model, get_device, extract_ffn_hidden_states

def batched_cosine_similarity(matrix, vector, chunk_size=10240):
    vector_norm = F.normalize(vector, dim=0)
    sims = []
    for i in range(0, matrix.size(0), chunk_size):
        chunk = matrix[i:i + chunk_size]
        chunk_norm = F.normalize(chunk, dim=1)
        sim = torch.matmul(chunk_norm, vector_norm)
        sims.append(sim)
    return torch.cat(sims)

def normalize(word):
    return re.sub(r'\W+', '', word).lower()  # Remove non-alphanumeric chars and lowercase

def get_retrieval_rates(results_df):
    token_str_cols = [col for col in results_df.columns if col.endswith("_top_token_str")]
    retrieval_rates = [] 
    layers = []

    for col in token_str_cols:
        pred_word = results_df[col].apply(lambda x: [normalize(word) for word in x])
        original_word = results_df["word"].apply(normalize)
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

def get_cumulative_retrieval_rates(results_df):
    layers = sorted([int(col.split("_")[1]) for col in results_df.columns if col.endswith("_top_token_str")])

    token_str_cols = [col for col in results_df.columns if col.endswith("_top_token_str")]
    token_str_cols = sorted(token_str_cols, key=lambda col: int(col.split("_")[1]))  # Ensure layers are ordered

    original_words = results_df["word"].apply(normalize)
    num_examples = len(original_words)

    cumulative_hits = [False] * num_examples
    cumulative_rates = []

    for col in token_str_cols:
        pred_words = results_df[col].apply(lambda x: [normalize(word) for word in x])
        retrieval_match = pred_words.combine(original_words, lambda preds, orig: orig in preds)
        # Update cumulative hits
        cumulative_hits = [prev or curr for prev, curr in zip(cumulative_hits, retrieval_match)]
        cumulative_rate = sum(cumulative_hits) / num_examples
        cumulative_rates.append(cumulative_rate)

    return layers, cumulative_rates

class LogitLens:
    """A class for performing logit lens analysis on language models."""
    
    def __init__(self, language, tokenizer_name, output_type="layer_hidden_states", retrieval_type="per_layer"):
        self.language = language
        self.tokenizer_name = tokenizer_name
        self.device = get_device()
        self.tokenizer = setup_tokenizer(tokenizer_name)
        self.model = setup_model(tokenizer_name, self.device)
        self.model.eval()
        self.embedding_matrix = self.model.get_input_embeddings().weight
        # self.embedding_matrix = self.model.lm_head.weight  # For Gemma3, use lm_head for logits
        self.model_name = tokenizer_name.split("/")[-1]
        self.output_type = output_type
        self.retrieval_type = retrieval_type
        

    def run_logit_lens(self, df, split_type, distance_metric="logits", k=3, save_dir=None, return_hidden_states=False):
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            word = row["word"]
            if split_type == "simple_split":
                split_tokens = literal_eval(row["splitted_tokens"])
            elif split_type == "typo_split":
                split_tokens = literal_eval(row['splitted_typo_tokens'])
            else:
                raise ValueError(f"Unknown split_type: {split_type}")
            
            input_ids = self.tokenizer.convert_tokens_to_ids(split_tokens)
            if self.tokenizer.bos_token:
                input_ids = [self.tokenizer.bos_token_id] + input_ids
            
            input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

            with torch.no_grad():
                if self.output_type == "layer_hidden_states":
                    outputs = self.model(input_ids, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
                elif self.output_type == "ffn_hidden_states":
                    hidden_states = extract_ffn_hidden_states(self.model, input_ids, self.tokenizer_name)

            if return_hidden_states:
                all_hidden_states = []
                for layer_idx, layer_output in enumerate(hidden_states[1:], start=1):  # skip embeddings
                    last_token_hidden = layer_output[0, -1].detach().cpu()
                    all_hidden_states.append(last_token_hidden)
                results.append({
                    "word": word,
                    "all_hidden_states": all_hidden_states
                })
                continue  # skip logit-lens part
            
            per_layer_top_token_ids = []
            per_layer_top_token_strs = []
            
            for layer_idx, h in enumerate(hidden_states[1:], start=1):  # skip embedding layer
                hidden = h[0, -1]  # last token's hidden state
                
                if distance_metric == 'logits':
                    logits = torch.matmul(self.embedding_matrix, hidden)
                elif distance_metric == 'cosine':
                    # logits = F.cosine_similarity(self.embedding_matrix, hidden, dim=-1)
                    logits = batched_cosine_similarity(self.embedding_matrix, hidden)
                else:
                    raise ValueError(f"Unknown distance_metric: {distance_metric}")

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
        
        if return_hidden_states:
            return results
        
        results_df = pd.DataFrame(results)
        if save_dir:
            results_df.to_csv(save_dir, index=False)

            # if self.output_type == "layer_hidden_states":
            #     results_df.to_csv(f"/home/hyujang/multilingual-inner-lexicon/output/RQ1/WordIdentity/single_token_{type}_{self.model_name}_{self.language}_normalized.csv", index=False)
            # elif self.output_type == "ffn_hidden_states":
            #     results_df.to_csv(f"/home/hyujang/multilingual-inner-lexicon/output/RQ1/ComponentAnalysis/ffn_hidden_states/single_token_{type}_{self.model_name}_{self.language}.csv", index=False)
        return results_df

    def run(self, path1, path2, vis=True, save=False):
        
        df = pd.read_csv(path1)
        results_df_original = self.run_logit_lens(
            df, distance_metric='logits', split_type='simple_split', k=3, save=save
        )
        
        if self.retrieval_type == "per_layer":
            layers_original, rates_original = get_retrieval_rates(results_df_original)
        elif self.retrieval_type == "cumulative":
            layers_original, rates_original = get_cumulative_retrieval_rates(results_df_original)
        else:
            raise ValueError(f"Unknown retrieval_type: {self.retrieval_type}")

        df = pd.read_csv(path2)
        results_df_typo = self.run_logit_lens(
            df, distance_metric='logits', split_type='typo_split', k=3, save=save
        )
        
        if self.retrieval_type == "per_layer":
            layers_typo, rates_typo = get_retrieval_rates(results_df_typo)
        elif self.retrieval_type == "cumulative":
            layers_typo, rates_typo = get_cumulative_retrieval_rates(results_df_typo)

        if vis:
            self._create_visualization(layers_original, rates_original, layers_typo, rates_typo)
        
        return layers_original, rates_original, layers_typo, rates_typo

    def _create_visualization(self, layers_original, rates_original, layers_typo, rates_typo):
        """Create visualization of retrieval rates."""
        plt.figure(figsize=(8, 5))
        plt.plot(layers_original, rates_original, marker='o', label='Splitted')
        plt.plot(layers_typo, rates_typo, marker='s', label='Typo Splitted')
        plt.xlabel("Layer")
        
        if self.retrieval_type == "per_layer":
            plt.ylabel("Word Retrieval Rate (Top-3)")
            plt.title(f"Top-3 Retrieval Rate per Layer ({self.model_name}, {self.language})")
        elif self.retrieval_type == "cumulative":
            plt.ylabel("Cumulative Retrieval Rate (Top-3)")
            plt.title(f"Top-3 Cumulative Retrieval Rate ({self.model_name}, {self.language})")
        
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    MODEL_NAME = "Tower-Babel/Babel-9B-Chat"
    LANGUAGE = "English"
    
    logit_lens = LogitLens(LANGUAGE, MODEL_NAME, output_type="ffn_hidden_states")
    model_name_short = MODEL_NAME.split("/")[-1]
    
    layers_original, rates_original, layers_typo, rates_typo = logit_lens.run(
        path1=f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/WordIdentity/single_token_splitted_{model_name_short}_{LANGUAGE}_v2.csv",
        path2=f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/WordIdentity/single_token_typos_{model_name_short}_{LANGUAGE}_v2.csv",
        vis=False,
        save=True
        # save_dir=f"/home/hyujang/multilingual-inner-lexicon/output/RQ1/WordIdentity/single_token_{type}_{model_name_short}_{self.language}_normalized.csv"
    )