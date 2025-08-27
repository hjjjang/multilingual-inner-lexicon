import torch
import pandas as pd
import numpy as np
import os
import ast
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Import utilities
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils import clean_memory, get_device, setup_tokenizer, setup_model, extract_token_hidden_states

# Import data class with fallback
try:
    from data import WordNonwordData
except ImportError:
    from RQ1.WordNonword.data import WordNonwordData


class WordNonwordClassifier(WordNonwordData):
    def __init__(self, language, tokenizer_name):
        super().__init__(language, tokenizer_name)
        
        clean_memory()

        self.device = get_device()
        print("Using device:", self.device)
        
        self.tokenizer = setup_tokenizer(tokenizer_name)
        self.model = setup_model(tokenizer_name, self.device)
        
        print(f"Model {self.tokenizer_name} for {self.language} loaded successfully.")
        
    def prepare_data(self, dataset):
        """Convert tokenized words into feature vectors"""
        print("Preparing data for classification...")

        # cache_dir = f"./cache/RQ1/{self.language}_{self.tokenizer_name.split('/')[1]}"
        # os.makedirs(cache_dir, exist_ok=True)
        # cache_path = os.path.join(cache_dir, "hidden_states.pt")

        # if os.path.exists(cache_path):
        #     print("Loading cached hidden states...")
        #     hidden_states_dict = torch.load(cache_path)
        # else:
        hidden_states_dict = extract_token_hidden_states(
            model=self.model,
            tokenizer=self.tokenizer,
            # inputs=dataset['word'],
            inputs=dataset['tokens'],  # Use the tokenized words directly
            tokenizer_name=self.tokenizer_name,
            device=self.device,
            token_idx_to_extract=-1,
            layers_to_extract=None
        )
            # print(f"Saving hidden states to {cache_path}")
            # torch.save(hidden_states_dict, cache_path)

        label_encoder = LabelEncoder()
        dataset['label_encoded'] = label_encoder.fit_transform(dataset['label'])

        return hidden_states_dict, dataset['label_encoded']

    def train_and_evaluate(self, hidden_states, labels_encoded):
        """Train and evaluate KNN on multiple layers"""
        neighbors_range = range(1, 11)
        results = []

        for layer in tqdm(hidden_states, desc="Training on each layer"):
            X_train, X_test, y_train, y_test = train_test_split(
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
        results_df.to_csv(f'./output/RQ1/WordNonword/two-token_words2/knn_eval_results_{self.tokenizer_name.split("/")[1]}_{self.language}_tokens.csv', index=False)
        # results_df.to_csv(f'./output/RQ1/WordNonword/two-token_words/knn_eval_results_{self.tokenizer_name.split("/")[1]}_{self.language}_wiki.csv', index=False)
        return results_df


    def run(self):
        # dataset = self.main()  # Load and preprocess dataset
        dataset_path = os.path.join(self.base_dir, f"data/RQ1/WordNonword/wordnonword_{self.tokenizer_name.split('/')[1]}_{self.language}-2token.csv")
        # dataset_path = os.path.join(self.base_dir, f"data/RQ1/WordNonword/r1_dataset_{self.tokenizer_name.split('/')[1]}_{self.language}-wiki-2token.csv")
        dataset = pd.read_csv(dataset_path)
        dataset['tokens'] = dataset['tokens'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)  # Convert string representation of list to actual list
        print("Data loaded successfully")

        X, y = self.prepare_data(dataset)
        results_df = self.train_and_evaluate(X, y)
        del self.model, self.tokenizer, X, y
        return results_df
    
if __name__ == "__main__":
    experiments = [
        # ("English", "Tower-Babel/Babel-9B-Chat"),
        # ("Korean", "Tower-Babel/Babel-9B-Chat"),
        # ("German", "Tower-Babel/Babel-9B-Chat"),
        # ("English", "google/gemma-3-12b-it"),
        # ("Korean", "google/gemma-3-12b-it"),
        ("German", "google/gemma-3-12b-it"),
        ("English", "meta-llama/Llama-2-7b-chat-hf"),
        ("Korean", "meta-llama/Llama-2-7b-chat-hf"),
        ("German", "meta-llama/Llama-2-7b-chat-hf"),
    ]
    
    for language, model_name in experiments:
        print(f"\n{'='*50}")
        print(f"Running experiment: {language} with {model_name}")
        print(f"{'='*50}")
        
        try:
            classifier = WordNonwordClassifier(language, model_name)
            results = classifier.run()
            print(f"Experiment completed successfully for {language} with {model_name}")
            
        except Exception as e:
            print(f"Error in experiment {language} with {model_name}: {e}")
            continue
