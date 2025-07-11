import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import Gemma3ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import re
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from RQ1.WordNonword.classification import WordNonwordClassifier


class LogitLens(WordNonwordClassifier):
    def __init__(self, language, tokenizer_name, output_type="layer_hidden_states", retreival_type="per_layer"):
        super().__init__(language, tokenizer_name)  # Inherit token config
        self.setup_tokenizer()
        self.model.eval()
        self.embedding_matrix = self.model.get_input_embeddings().weight
        # self.embedding_matrix = self.model.lm_head.weight  # For Gemma3, use lm_head for logits
        self.model_name = tokenizer_name.split("/")[-1]
        self.output_type = output_type
        self.retreival_type = retreival_type
        
    def setup_tokenizer(self):
        # pass
        if self.tokenizer_name == "Tower-Babel/Babel-9B-Chat":
            self.tokenizer.add_special_tokens({'unk_token': 'UNK'})
            self.tokenizer.unk_token_id = self.tokenizer.convert_tokens_to_ids('UNK')

    def batched_cosine_similarity(self, matrix, vector, chunk_size=10240):
        """
        Compute cosine similarity between a single vector and rows of a matrix in chunks.
        Args:
            vector: Tensor of shape (hidden_dim,)
            matrix: Tensor of shape (vocab_size, hidden_dim)
            chunk_size: Number of rows to process at a time
        Returns:
            Tensor of shape (vocab_size,)
        """
        vector_norm = F.normalize(vector, dim=0)
        sims = []
        for i in range(0, matrix.size(0), chunk_size):
            chunk = matrix[i:i + chunk_size]
            chunk_norm = F.normalize(chunk, dim=1)
            sim = torch.matmul(chunk_norm, vector_norm)
            sims.append(sim)
        return torch.cat(sims)


    def run_logit_lens(self, df, type, distance_metric="logits", k=3, save=False):
        results = []
        for __name__, row in tqdm(df.iterrows(), total=len(df)):
            word = row["word"]
            if type == "simple_split":
                split_tokens = literal_eval(row["splitted_tokens"])
            elif type == "typo_split":
                split_tokens = literal_eval(row['splitted_typo_tokens'])
            
            # 1) Keep unknown token as it is without further tokenization
            # """ 
            input_ids = self.tokenizer.convert_tokens_to_ids(split_tokens)
            if self.tokenizer.bos_token:
                input_ids = [self.tokenizer.bos_token_id] + input_ids

            if self.tokenizer_name == "Tower-Babel/Babel-9B-Chat":
                input_ids = [31883 if id is None else id for id in input_ids]
            
            input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            # """
            # 2) 
            # tokens = self.tokenizer.convert_tokens_to_string(split_tokens) 
            # input_ids = self.tokenizer(tokens, return_tensors="pt", return_attention_mask=False)['input_ids'].to(self.device)

            # 3)
            # input_ids = self.tokenizer(split_tokens, return_tensors="pt", return_attention_mask=False)['input_ids'].to(self.device)

            with torch.no_grad():
                if self.output_type == "layer_hidden_states":
                    outputs = self.model(input_ids, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
                elif self.output_type == "ffn_hidden_states":
                    if "gemma-3" in self.model_name.lower():
                        ffn_probe = FFNProbeGemma3(self.model)
                    else:
                        ffn_probe = FFNProbe(self.model)
                    outputs = ffn_probe(input_ids, output_hidden_states=False) # final logits
                    hidden_states = ffn_probe.ffn_outputs
                    
                # outputs = self.model(input_ids, output_hidden_states=True)
                # hidden_states = outputs.hidden_states

            per_layer_top_token_ids = []
            per_layer_top_token_strs = []
            
            # assert self.embedding_matrix.shape[0] == self.model.config.vocab_size
            # assert self.embedding_matrix.device == hidden_states[0].device

            for layer_idx, h in enumerate(hidden_states[1:], start=1):  # skip embedding layer
                hidden = h[0, -1] # last token's hidden state
                
                # step1: get the similarity
                if distance_metric == 'logits':
                    # logits = torch.matmul(F.normalize(self.embedding_matrix, p=2, dim=-1), F.normalize(hidden, p=2, dim=-1))
                    logits = torch.matmul(self.embedding_matrix, hidden)
                
                elif distance_metric == 'cosine':
                    try:
                        logits = F.cosine_similarity(self.embedding_matrix, hidden, dim=-1)
                    except:
                        logits = self.batched_cosine_similarity(self.embedding_matrix, hidden)
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
            if self.output_type == "layer_hidden_states":
                results_df.to_csv(f"/home/hyujang/multilingual-inner-lexicon/output/RQ1/WordIdentity/single_token_{type}_{self.model_name}_{self.language}_normalized.csv", index=False)
            elif self.output_type == "ffn_hidden_states":
                results_df.to_csv(f"/home/hyujang/multilingual-inner-lexicon/output/RQ1/ComponentAnalysis/ffn_hidden_states/single_token_{type}_{self.model_name}_{self.language}.csv", index=False)
        return results_df

    def collect_layerwise_hidden_states(self, df, type='simple_split'):
        if "gemma-3" in self.model_name.lower():
            num_layers = self.model.config.text_config.num_hidden_layers
        else:
            num_layers = self.model.config.num_hidden_layers
        all_hidden_states = [[] for _ in range(num_layers)]  # layer-wise storage
        words = []

        for __name__, row in tqdm(df.iterrows(), total=len(df)):
            word = row["word"]
            words.append(word)

            if type == "simple_split":
                split_tokens = literal_eval(row["splitted_tokens"])
            elif type == "typo_split":
                split_tokens = literal_eval(row['splitted_typo_tokens'])

            input_ids = self.tokenizer.convert_tokens_to_ids(split_tokens)
            if self.tokenizer.bos_token:
                input_ids = [self.tokenizer.bos_token_id] + input_ids

            if self.tokenizer_name == "Tower-Babel/Babel-9B-Chat":
                input_ids = [31883 if id is None else id for id in input_ids]

            input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)

            with torch.no_grad():
                if self.output_type == "layer_hidden_states":
                    outputs = self.model(input_ids, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
                elif self.output_type == "ffn_hidden_states":
                    if "gemma-3" in self.model_name.lower():
                        ffn_probe = FFNProbeGemma3(self.model)
                    else:
                        ffn_probe = FFNProbe(self.model)
                    _ = ffn_probe(input_ids, output_hidden_states=False)
                    hidden_states = ffn_probe.ffn_outputs

            for layer_idx, layer_output in enumerate(hidden_states[1:], start=1):  # skip embeddings
                last_token_hidden = layer_output[0, -1].detach().cpu()
                all_hidden_states[layer_idx - 1].append(last_token_hidden)

        # Convert to tensors
        all_hidden_states = [torch.stack(layer_list) for layer_list in all_hidden_states]  # list of [N, D]

        return all_hidden_states, words  # optionally return words for traceability


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

    def get_cumulative_retrieval_rates(self, results_df):
        layers = sorted([int(col.split("_")[1]) for col in results_df.columns if col.endswith("_top_token_str")])

        token_str_cols = [col for col in results_df.columns if col.endswith("_top_token_str")]
        token_str_cols = sorted(token_str_cols, key=lambda col: int(col.split("_")[1]))  # Ensure layers are ordered

        original_words = results_df["word"].apply(self.normalize)
        num_examples = len(original_words)

        cumulative_hits = [False] * num_examples
        cumulative_rates = []

        for col in token_str_cols:
            pred_words = results_df[col].apply(lambda x: [self.normalize(word) for word in x])
            retrieval_match = pred_words.combine(original_words, lambda preds, orig: orig in preds)
            # Update cumulative hits
            cumulative_hits = [prev or curr for prev, curr in zip(cumulative_hits, retrieval_match)]
            cumulative_rate = sum(cumulative_hits) / num_examples
            cumulative_rates.append(cumulative_rate)

        return layers, cumulative_rates

    def run(self, path1, path2, vis=True, save=False):
        
        df = pd.read_csv(path1)
        # print(df)
        results_df_original = self.run_logit_lens(df, distance_metric='logits', type='simple_split', k=3, save=save)
        if self.retreival_type == "per_layer":
            layers_original, rates_original = self.get_retrieval_rates(results_df_original)
        elif self.retreival_type == "cumulative":
            layers_original, rates_original = self.get_cumulative_retrieval_rates(results_df_original)

        df = pd.read_csv(path2)
        results_df_typo = self.run_logit_lens(df, distance_metric='logits', type='typo_split', k=3, save=save)
        if self.retreival_type == "per_layer":
            layers_typo, rates_typo = self.get_retrieval_rates(results_df_typo)
        elif self.retreival_type == "cumulative":
            layers_typo, rates_typo = self.get_cumulative_retrieval_rates(results_df_typo)
        
        # print(results_df_original)
        if vis:
            plt.figure(figsize=(8, 5))
            plt.plot(layers_original, rates_original, marker='o', label='Splitted')
            plt.plot(layers_typo, rates_typo, marker='s', label='Typo Splitted')
            plt.xlabel("Layer")
            if self.retreival_type == "per_layer":
                plt.ylabel("Word Retrieval Rate (Top-3)")
                plt.title(f"Top-3 Retrieval Rate per Layer ({self.model_name}, {self.language})")
            elif self.retreival_type == "cumulative":
                plt.ylabel("Cumulative Retrieval Rate (Top-3)")
                plt.title(f"Top-3 Cumulative Retrieval Rate ({self.model_name}, {self.language})")
            plt.ylim(0, 1)
            plt.grid(True)
            plt.legend()
            plt.show()
        # return results_df_original, results_df_typo
        return layers_original, rates_original, layers_typo, rates_typo


class FFNProbe:
    def __init__(self, model):
        self.model = model
        self.ffn_outputs = []

        def hook_fn(module, input, output):
            self.ffn_outputs.append(output)

        # Register hooks to the FFN layers of each transformer block
        for block in self.model.model.layers:  # adjust depending on model architecture
            block.mlp.register_forward_hook(hook_fn)  # For LLaMA, GPT-J, Falcon, etc.

    def clear(self):
        self.ffn_outputs = []

    def __call__(self, *args, **kwargs):
        self.clear()
        return self.model(*args, **kwargs)


class FFNProbeGemma3:
    def __init__(self, model):
        self.model = model
        self.ffn_outputs = []

        def hook_fn(module, input, output):
            self.ffn_outputs.append(output)

        for block in self.model.model.language_model.layers:
            block.mlp.register_forward_hook(hook_fn)
    def clear(self):
        self.ffn_outputs = []

    def __call__(self, *args, **kwargs):
        self.clear()
        return self.model(*args, **kwargs)



if __name__ == "__main__":    
    MODEL_NAME = "Tower-Babel/Babel-9B-Chat"
    # MODEL_NAME = "google/gemma-3-12b-it"
    # LANGUAGE = "German"
    LANGUAGE = "English"
    # logit_lens = LogitLens(LANGUAGE, MODEL_NAME)
    logit_lens = LogitLens(LANGUAGE, MODEL_NAME, output_type="ffn_hidden_states")
    MODEL_NAME = MODEL_NAME.split("/")[-1]
    results_df_original, results_df_typo = logit_lens.run(
        path1 = f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/WordIdentity/single_token_splitted_{MODEL_NAME}_{LANGUAGE}_v2.csv",
        path2 = f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/WordIdentity/single_token_typos_{MODEL_NAME}_{LANGUAGE}_v2.csv",
        vis = False,
        save = True
    )