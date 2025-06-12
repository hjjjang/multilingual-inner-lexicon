import torch
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM, Gemma3ForConditionalGeneration, Gemma3ForCausalLM, Qwen2ForCausalLM
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from data import WordNonwordData  # Import dataset generator
import os
import ast
import gc

from scipy.stats import uniform, randint

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torch.cuda.amp import autocast, GradScaler


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
        if self.tokenizer_name == "google/gemma-3-12b-it" or self.tokenizer_name == "google/gemma-3-12b-pt":
            # self.model = Gemma3ForCausalLM.from_pretrained(tokenizer_name, token=self.token_value)
            self.model = Gemma3ForConditionalGeneration.from_pretrained(tokenizer_name, token=self.token_value)
            self.model.to(self.device)
        elif self.tokenizer_name == "meta-llama/Llama-2-7b-chat-hf":
            self.model = AutoModelForCausalLM.from_pretrained(tokenizer_name, token=self.token_value)
            self.model.to(self.device)
        elif self.tokenizer_name == "Qwen/Qwen2.5-VL-7B-Instruct":
            self.model = Qwen2ForCausalLM.from_pretrained(tokenizer_name, token=self.token_value)
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
            if self.tokenizer_name == "google/gemma-3-12b-it" or self.tokenizer_name == "google/gemma-3-12b-pt":
                layers_to_extract = list(range(1, self.model.config.text_config.num_hidden_layers + 1))  # Exclude embedding layer
            else:
                layers_to_extract = list(range(1, self.model.config.num_hidden_layers + 1))  # Exclude embedding layer

        all_hidden_states = {layer: [] for layer in layers_to_extract}

        with torch.no_grad():
            for tokens in tqdm(tokenized_inputs, desc="Extracting hidden states"):
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
                # input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
                try:
                    outputs = self.model(input_ids, output_hidden_states=True)
                except Exception as e:
                    print("input_ids shape:", input_ids.shape)
                    print("Error during model forward pass:", e)
                    break
                for layer in layers_to_extract:
                    hidden_states = outputs.hidden_states[layer]  # Shape: (1, seq_len, hidden_dim)
                    all_hidden_states[layer].append(hidden_states[:, token_idx_to_extract, :].detach().cpu())

        for layer in all_hidden_states:
            all_hidden_states[layer] = torch.cat(all_hidden_states[layer], dim=0)

        return all_hidden_states

    # def prepare_data(self, dataset):
    #     """Convert tokenized words into feature vectors"""
    #     print("Preparing data for classification...")
    #     print(dataset)
    #     # hidden_states_dict = self.extract_token_i_hidden_states(dataset['tokens'])
    #     hidden_states_dict = self.extract_token_i_hidden_states(dataset['tokens'], layers_to_extract=[1,2])
        
    #     label_encoder = LabelEncoder()
    #     dataset['label_encoded'] = label_encoder.fit_transform(dataset['label'])

    #     return hidden_states_dict, dataset['label_encoded']
    
    def prepare_data(self, dataset):
        """Convert tokenized words into feature vectors"""
        print("Preparing data for classification...")

        cache_dir = f"./cache/RQ1/{self.language}_{self.tokenizer_name.split('/')[1]}"
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "hidden_states.pt")

        if os.path.exists(cache_path):
            print("Loading cached hidden states...")
            hidden_states_dict = torch.load(cache_path)
        else:
            hidden_states_dict = self.extract_token_i_hidden_states(dataset['tokens'], layers_to_extract=None)
            # hidden_states_dict = self.extract_token_i_hidden_states(dataset['tokens'], layers_to_extract=[1,2])
            print(f"Saving hidden states to {cache_path}")
            torch.save(hidden_states_dict, cache_path)

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
        results_df.to_csv(f'./output/RQ1/WordNonword/knn_eval_results_{self.tokenizer_name.split("/")[1]}-CG_{self.language}-wiki.csv', index=False)
        # results_df.to_csv(f'./output/RQ1/two-token_words/knn_eval_results_{self.tokenizer_name.split("/")[1]}_{self.language}-wiki.csv', index=False)
        # print(results_df)
        return results_df

    def train_and_evaluate_lgbm(self, hidden_states, labels_encoded):
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

            # Define the LightGBM classifier
            lgbm = LGBMClassifier(
                device='gpu',        # Enable GPU support
                boosting_type='gbdt',
                objective='binary',
                verbosity=1
            )
            
            param_dist = {
                'learning_rate': uniform(0.01, 0.05),     # Range: 0.05 to 0.10
                'max_depth': randint(6, 12),              # Depth: 6–11
                'n_estimators': randint(100, 201)         # Fewer trees: 100–200
            }

            # Set up RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=lgbm,
                param_distributions=param_dist,
                n_iter=10,             # Number of random combinations to test
                scoring='f1',          # Use F1 score to evaluate the best models
                cv=3,                  # 3-fold cross-validation
                verbose=0,
                n_jobs=-1,             # Use all available cores
                random_state=self.seed
            )
            
            # Fit the model
            fit_params = {
                "eval_set": [(X_test, y_test)],
                "eval_metric": "binary_logloss",
                "early_stopping_rounds": 10,
                "verbose": False
            }

            # 5. Run the search
            # random_search.fit(X_train, y_train, **fit_params)
            random_search.fit(X_train, y_train)
            best_model = random_search.best_estimator_  # Get the best model from the random search
            y_pred = best_model.predict(X_test)        # Predict on the test set

            # Collect results
            results.append({
                'Layer': layer,
                'Best Params': random_search.best_params_,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'Confusion Matrix': confusion_matrix(y_test, y_pred)
            })

        results_df = pd.DataFrame(results)
        results_df.to_csv(f'./output/RQ1/WordNonword/two-token_words/lgbm_eval_results_{self.tokenizer_name.split("/")[1]}_{self.language}-wiki.csv', index=False)
        return results_df


    def train_and_evaluate_mlp_torch(self, hidden_states, labels_encoded):
        """Train and evaluate a simple MLP using PyTorch on GPU."""
        class SimpleAttentionMLP(nn.Module):
            def __init__(self, input_dim):
                super(SimpleAttentionMLP, self).__init__()
                self.fc1 = nn.Linear(input_dim, 512)
                self.attention = nn.MultiheadAttention(512, num_heads=8, dropout=0.2)
                self.fc2 = nn.Linear(512, 1)
                self.dropout = nn.Dropout(0.3)

            def forward(self, x):
                # x: [batch_size, seq_len, input_dim]
                x = self.fc1(x)  # [batch_size, seq_len, 512]
                x = x.unsqueeze(0)  # Add batch dimension for attention
                attn_output, _ = self.attention(x, x, x)
                x = attn_output.squeeze(0)
                x = self.fc2(x)
                return x

        class SimpleMLP(nn.Module):
            def __init__(self, input_dim):
                super(SimpleMLP, self).__init__()
                self.model = nn.Sequential( # mlp
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    # nn.Sigmoid() # mlp + e20
                )
                # self.model = nn.Sequential( # mpl2
                #     nn.Linear(input_dim, 512),
                #     nn.ReLU(),
                #     nn.Dropout(0.3),
                #     nn.Linear(512, 256),
                #     nn.ReLU(),
                #     nn.Dropout(0.2),
                #     nn.Linear(256, 1),
                #     # nn.Sigmoid() # mlp3
                # )
                # self.model = nn.Sequential( # mlp4
                #     nn.Linear(input_dim, 1024),
                #     nn.ReLU(),
                #     nn.Dropout(0.3),
                #     nn.Linear(1024, 512),
                #     nn.ReLU(),
                #     nn.Dropout(0.2),
                #     nn.Linear(512, 256),
                #     nn.ReLU(),
                #     nn.Dropout(0.2),
                #     nn.Linear(256, 128),
                #     nn.ReLU(),
                #     nn.Dropout(0.2),
                #     nn.Linear(128, 64),
                #     nn.ReLU(),
                #     nn.Linear(64, 1),
                # )

            def forward(self, x):
                return self.model(x)

        device = self.device
        results = []

        for layer in tqdm(hidden_states, desc="Training PyTorch MLP on each layer"):
            X = hidden_states[layer].to(torch.float32)
            y = torch.tensor(labels_encoded.values, dtype=torch.float32).unsqueeze(1)

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                # X.numpy(), y.numpy(), 
                X, y.numpy(),
                test_size=self.config["test_split"], 
                random_state=self.seed,
                stratify=labels_encoded
            )

            # Convert to PyTorch tensors and move to device
            X_train = torch.tensor(X_train.astype(np.float32)).to(device)
            y_train = torch.tensor(y_train.astype(np.float32)).to(device)
            X_test = torch.tensor(X_test.astype(np.float32)).to(device)
            y_test = torch.tensor(y_test.astype(np.float32)).to(device)

            model = SimpleMLP(input_dim=X_train.shape[1]).to(device)
            # model = SimpleAttentionMLP(input_dim=X_train.shape[1]).to(device)
            # criterion = nn.BCELoss() # with Sigmoid
            criterion = nn.BCEWithLogitsLoss() # without Sigmoid
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

            # Training loop
            for epoch in range(20):  # you can increase if needed
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                # y_pred = model(X_test).cpu().numpy() > 0.5 # With Sigmoid
                y_pred = (torch.sigmoid(model(X_test)) > 0.5).cpu().numpy() # Without Sigmoid
                y_true = y_test.cpu().numpy()

            results.append({
                'Layer': layer,
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred),
                'Recall': recall_score(y_true, y_pred),
                'F1-Score': f1_score(y_true, y_pred),
                'Confusion Matrix': confusion_matrix(y_true, y_pred)
            })
            


        results_df = pd.DataFrame(results)
        # results_df.to_csv(f'./output/RQ1/WordNonword/two-token_words/mlp4-standardscaler-e40_torch_eval_results_{self.tokenizer_name.split("/")[1]}_{self.language}-wiki.csv', index=False)
        # results_df.to_csv(f'./output/RQ1/WordNonword/two-token_words/mlp-attn-standardscaler-e40_torch_eval_results_{self.tokenizer_name.split("/")[1]}_{self.language}-wiki.csv', index=False)
        results_df.to_csv(f'./output/RQ1/WordNonword/two-token_words/mlp-standardscaler-e20_torch_eval_results_{self.tokenizer_name.split("/")[1]}_{self.language}-wiki.csv', index=False) 
        return results_df

    def train_and_evaluate_transformer_enc(self, hidden_states, labels_encoded):
        """Train and evaluate a Transformer Encoder using PyTorch on GPU."""

        class TransformerClassifier(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=input_dim,
                    nhead=4,
                    dim_feedforward=512,
                    dropout=0.1,
                    activation='relu',
                    batch_first=True  # Allows input shape [batch, seq_len, dim]
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
                self.classifier = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                # x: [batch_size, seq_len=1, input_dim]
                h = self.transformer(x)  # output: [batch_size, seq_len=1, dim]
                return self.classifier(h[:, 0])  # Take the output at position 0

        device = self.device
        results = []

        batch_size = 16  # Reduce batch size
        scaler = StandardScaler()

        for layer in tqdm(hidden_states, desc="Training Transformer on each layer"):
            X = hidden_states[layer].to(torch.float32)
            y = torch.tensor(labels_encoded.values, dtype=torch.float32).unsqueeze(1)

            X = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y.numpy(),
                test_size=self.config["test_split"],
                random_state=self.seed,
                stratify=labels_encoded
            )

            X_train = torch.tensor(X_train.astype(np.float32)).unsqueeze(1).to(device)
            X_test = torch.tensor(X_test.astype(np.float32)).unsqueeze(1).to(device)
            y_train = torch.tensor(y_train.astype(np.float32)).to(device)
            y_test = torch.tensor(y_test.astype(np.float32)).to(device)

            model = TransformerClassifier(input_dim=X_train.shape[-1]).to(device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            for epoch in range(20):
                model.train()
                optimizer.zero_grad()

                with autocast():  # Mixed precision training
                    outputs = model(X_train)
                    loss = criterion(outputs, y_train)

                loss.backward()
                optimizer.step()

                torch.cuda.empty_cache()  # Clear cache after each step

            model.eval()
            with torch.no_grad():
                y_pred = model(X_test).cpu().numpy() > 0.5
                y_true = y_test.cpu().numpy()

            results.append({
                'Layer': layer,
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred),
                'Recall': recall_score(y_true, y_pred),
                'F1-Score': f1_score(y_true, y_pred),
                'Confusion Matrix': confusion_matrix(y_true, y_pred)
            })

        results_df = pd.DataFrame(results)
        results_df.to_csv(f'./output/RQ1/WordNonword/two-token_words/transformer-standardscaler_torch_eval_results_{self.tokenizer_name.split("/")[1]}_{self.language}-wiki.csv', index=False)
        return results_df


    def run(self):
        # dataset = self.main()  # Load and preprocess dataset
        # dataset_path = os.path.join(self.base_dir, f"data/RQ1/WordNonword/r1_dataset_{self.tokenizer_name.split('/')[1]}_{self.language}-wiki-2.csv")
        dataset_path = os.path.join(self.base_dir, f"data/RQ1/WordNonword/r1_dataset_{self.tokenizer_name.split('/')[1]}_{self.language}-wiki-2token.csv")
        dataset = pd.read_csv(dataset_path)
        dataset['tokens'] = dataset['tokens'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)  # Convert string representation of list to actual list
        print("Data loaded successfull")
        X, y = self.prepare_data(dataset)
        # results_df = self.train_and_evaluate(X, y)
        # results_df = self.train_and_evaluate_lgbm(X, y)
        results_df = self.train_and_evaluate_mlp_torch(X, y)
        # results_df = self.train_and_evaluate_transformer_enc(X, y)
        del self.model, self.tokenizer, X, y
        torch.cuda.empty_cache()
        gc.collect()
        return results_df
    
if __name__ == "__main__":
    # word_nonword_cls = WordNonwordClassifier("Korean", "Tower-Babel/Babel-9B-Chat")
    # results = word_nonword_cls.run()
    # word_nonword_cls = WordNonwordClassifier("English", "Tower-Babel/Babel-9B-Chat")
    # results = word_nonword_cls.run()
    word_nonword_cls = WordNonwordClassifier("German", "Tower-Babel/Babel-9B-Chat")
    results = word_nonword_cls.run()
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
    
    # print(dataset)
