import torch
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM, Gemma3ForConditionalGeneration, Gemma3ForCausalLM, Qwen2ForCausalLM, Qwen2_5_VLForConditionalGeneration
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from data import WordNonwordData  # Import dataset generator
import os
import ast
import gc


class WordNonwordClassifier(WordNonwordData):
    def __init__(self, language, tokenizer_name):
        super().__init__(language, tokenizer_name)  # Inherit token config
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, token=self.token_value)
        # self.model = AutoModel.from_pretrained(tokenizer_name, device_map="auto")
        # self.model = AutoModelForImageTextToText.from_pretrained(tokenizer_name, 
        if self.tokenizer_name == "google/gemma-3-12b-it":
            # self.model = Gemma3ForCausalLM.from_pretrained(tokenizer_name, token=self.token_value)
            self.model = Gemma3ForConditionalGeneration.from_pretrained(tokenizer_name, token=self.token_value)
            self.model.to(self.device)
        elif self.tokenizer_name == "meta-llama/Llama-2-7b-chat-hf":
            self.model = AutoModelForCausalLM.from_pretrained(tokenizer_name, token=self.token_value)
            self.model.to(self.device)
        elif self.tokenizer_name == "Qwen/Qwen2.5-VL-7B-Instruct":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(tokenizer_name, token=self.token_value)
            self.model.to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(tokenizer_name, device_map="auto", token=self.token_value)
        # self.model = Gemma3ForCausalLM.from_pretrained(tokenizer_name,
        # self.model = AutoModelForCausalLM.from_pretrained(tokenizer_name,
                                                    #  device_map="auto",
                                                    #  torch_dtype=torch.bfloat16,
                                                    # token=self.token_value
        # )
            
        # self.model.to(self.device) # Not needed for device_map="auto"
        
    def extract_token_i_hidden_states(self, tokenized_inputs, token_idx_to_extract=-1, layers_to_extract=None):
        """Extract hidden states for tokenized words"""
        device = self.device
        self.model.eval()

        if isinstance(tokenized_inputs, str):
            tokenized_inputs = [tokenized_inputs]

        if layers_to_extract is None:
            if self.tokenizer_name in ["google/gemma-3-12b-it", "google/gemma-3-12b-pt"]:
                layers_to_extract = list(range(1, self.model.config.text_config.num_hidden_layers + 1))
            else:
                layers_to_extract = list(range(1, self.model.config.num_hidden_layers + 1))  # Exclude embedding layer

        all_hidden_states = {layer: [] for layer in layers_to_extract}

        with torch.no_grad():
            for tokens in tqdm(tokenized_inputs, desc="Extracting hidden states"):
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

                outputs = self.model(input_ids, output_hidden_states=True)
                for layer in layers_to_extract:
                    hidden_states = outputs.hidden_states[layer]  # Shape: (1, seq_len, hidden_dim)
                    all_hidden_states[layer].append(hidden_states[:, token_idx_to_extract, :].detach().cpu())

        for layer in all_hidden_states:
            all_hidden_states[layer] = torch.cat(all_hidden_states[layer], dim=0)

        return all_hidden_states

    def prepare_data(self, dataset):
        """Convert tokenized words into feature vectors"""
        print("Preparing data for KNN classification...")
        print(dataset)
        hidden_states_dict = self.extract_token_i_hidden_states(dataset['tokens'])
        
        label_encoder = LabelEncoder()
        dataset['label_encoded'] = label_encoder.fit_transform(dataset['label'])

        return hidden_states_dict, dataset['label_encoded']

    def train_and_evaluate(self, hidden_states, labels_encoded):
        """Train and evaluate KNN on multiple layers"""
        neighbors_range = range(1, 21)
        results = []

        for layer in tqdm(hidden_states, desc="Training on each layer"):
            X_train, X_test, y_train, y_test = train_test_split(
                # hidden_states[layer].numpy()
                hidden_states[layer].to(torch.float32).numpy(),  # Convert to float32
                labels_encoded, 
                test_size=self.config["test_split"], 
                random_state=self.seed,
                stratify=labels_encoded
            )

            for n in tqdm(neighbors_range, desc="Evaluating KNN"):
                knn = KNeighborsClassifier(n_neighbors=n)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)

                results.append({
                    'Layer': layer,
                    'n_neighbors': n,
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred),
                    'Recall': recall_score(y_test, y_pred),
                    'F1-Score': f1_score(y_test, y_pred),
                    'Confusion Matrix': confusion_matrix(y_test, y_pred)
                })

        results_df = pd.DataFrame(results)
        # results_df.to_csv(f'./output/RQ1/knn_eval_results_{self.tokenizer_name.split("/")[1]}_{self.language}-wordnet.csv', index=False)
        results_df.to_csv(f'./output/RQ1/WordNonword/knn_eval_results_{self.tokenizer_name.split("/")[1]}-PT_{self.language}-wiki.csv', index=False)
        # results_df.to_csv(f'./output/RQ1/two-token_words/knn_eval_results_{self.tokenizer_name.split("/")[1]}_{self.language}-wiki.csv', index=False)
        # print(results_df)
        return results_df


    def run(self):
        # dataset = self.main()  # Load and preprocess dataset
        # dataset_path = os.path.join(self.base_dir, f"data/RQ1/WordNonword/r1_dataset_{self.tokenizer_name.split('/')[1]}_{self.language}-wiki-2.csv")
        dataset_path = os.path.join(self.base_dir, f"data/RQ1/WordNonword/r1_dataset_gemma-3-12b-it_{self.language}-wiki-2.csv")
        # dataset_path = os.path.join(self.base_dir, f"data/r1_dataset_{self.tokenizer_name.split('/')[1]}_{self.language}-wiki-2token.csv")
        dataset = pd.read_csv(dataset_path)
        dataset['tokens'] = dataset['tokens'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)  # Convert string representation of list to actual list
        print("Data loaded successfull")
        X, y = self.prepare_data(dataset)
        results_df = self.train_and_evaluate(X, y)
        torch.cuda.empty_cache()
        gc.collect()
        return results_df
    
if __name__ == "__main__":
    # word_nonword_cls = WordNonwordClassifier("Korean", "Tower-Babel/Babel-9B-Chat")
    # results = word_nonword_cls.run()
    # word_nonword_cls = WordNonwordClassifier("English", "Tower-Babel/Babel-9B-Chat")
    # results = word_nonword_cls.run()
    # word_nonword_cls = WordNonwordClassifier("German", "Tower-Babel/Babel-9B-Chat")
    # results = word_nonword_cls.run()
    # word_nonword_cls = WordNonwordClassifier("Korean", "google/gemma-3-12b-it")
    # results = word_nonword_cls.run()
    # word_nonword_cls = WordNonwordClassifier("English", "google/gemma-3-12b-it")
    # results = word_nonword_cls.run()
    # word_nonword_cls = WordNonwordClassifier("German", "google/gemma-3-12b-it")
    # results = word_nonword_cls.run()
    # word_nonword_cls = WordNonwordClassifier("Korean", "meta-llama/Llama-2-7b-chat-hf")
    # results = word_nonword_cls.run()
    # word_nonword_cls = WordNonwordClassifier("English", "meta-llama/Llama-2-7b-chat-hf")
    # results = word_nonword_cls.run()
    # word_nonword_cls = WordNonwordClassifier("German", "meta-llama/Llama-2-7b-chat-hf")
    # results = word_nonword_cls.run()
    
    # word_nonword_cls = WordNonwordClassifier("Korean", "Qwen/Qwen2.5-VL-7B-Instruct")
    # results = word_nonword_cls.run()
    # word_nonword_cls = WordNonwordClassifier("English", "Qwen/Qwen2.5-VL-7B-Instruct")
    # results = word_nonword_cls.run()
    # word_nonword_cls = WordNonwordClassifier("German", "Qwen/Qwen2.5-VL-7B-Instruct")
    # results = word_nonword_cls.run()
    
    word_nonword_cls = WordNonwordClassifier("English", "google/gemma-3-12b-pt")
    results = word_nonword_cls.run()

    # print(dataset)
