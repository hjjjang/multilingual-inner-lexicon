import torch
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from data import WordNonwordData  # Import dataset generator
import os

class WordNonwordClassifier(WordNonwordData):
    def __init__(self, language, tokenizer_name):
        super().__init__(language, tokenizer_name)  # Inherit token config
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, token=self.token_value)
        # self.model = AutoModel.from_pretrained(tokenizer_name, device_map="auto")
        # self.model = AutoModelForImageTextToText.from_pretrained(tokenizer_name, device_map="auto")
        self.model = AutoModelForCausalLM.from_pretrained(tokenizer_name,
                                                     device_map="auto",
                                                    #  torch_dtype=torch.bfloat16
        )
            
        # self.model.to(self.device) # Not needed for device_map="auto"
        
    def extract_token_i_hidden_states(self, tokenized_inputs, token_idx_to_extract=-1, layers_to_extract=None):
        """Extract hidden states for tokenized words"""
        device = self.device
        self.model.eval()

        if isinstance(tokenized_inputs, str):
            tokenized_inputs = [tokenized_inputs]

        if layers_to_extract is None:
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
        # print(dataset['tokens'])
        # print("*********************************")
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
        # results_df.to_csv(f'./output/knn_eval_results_{self.tokenizer_name.split("/")[1]}_{self.language}-wordnet.csv', index=False)
        results_df.to_csv(f'./output/knn_eval_results_{self.tokenizer_name.split("/")[1]}_{self.language}-wiki.csv', index=False)
        # print(results_df)
        return results_df


    def run(self):
        # dataset = self.main()  # Load and preprocess dataset
        dataset_path = os.path.join(self.base_dir, f"data/r1_dataset_{self.tokenizer_name.split('/')[1]}_{self.language}-wiki.csv")
        dataset = pd.read_csv(dataset_path)
        print("Data loaded successfull")
        X, y = self.prepare_data(dataset)
        results_df = self.train_and_evaluate(X, y)
        return results_df
    
if __name__ == "__main__":
    # classifier = WordNonwordClassifier("English", "google/gemma-3-12b-it")
    classifier = WordNonwordClassifier("English", "Tower-Babel/Babel-9B-Chat")
    # classifier = WordNonwordClassifier("Korean", "Tower-Babel/Babel-9B-Chat")
    # classifier = WordNonwordClassifier("German", "Tower-Babel/Babel-9B-Chat")
    results = classifier.run()
    # print(dataset)
