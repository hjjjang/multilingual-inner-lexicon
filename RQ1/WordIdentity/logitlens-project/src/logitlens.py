from tqdm import tqdm
import torch
import pandas as pd
from ast import literal_eval
from wordnonword_classifier import WordNonwordClassifier

MIN_WORD_LEN = 3
LANGUAGE = "en"
MODEL_NAME = "Tower-Babel/Babel-9B-Chat"

class LogitLens(WordNonwordClassifier):
    def __init__(self):
        super().__init__()
        self.setup_tokenizer()

    def setup_tokenizer(self, model_name=MODEL_NAME):
        if model_name == "Tower-Babel/Babel-9B-Chat":
            self.tokenizer.add_special_tokens({'unk_token': 'UNK'})
            self.tokenizer.unk_token_id = self.tokenizer.convert_tokens_to_ids('UNK')

    def run_logit_lens(self, df, distance_metric, k, type, model_name=MODEL_NAME):
        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            word = row["word"]        
            if type == "simple_split":
                split_tokens = literal_eval(row["splitted_tokens"])
            elif type == "typo_split":
                split_tokens = literal_eval(row['splitted_typo_tokens'])
            
            input_ids = self.tokenizer.convert_tokens_to_ids(split_tokens)

            if model_name == "Tower-Babel/Babel-9B-Chat":
                input_ids = [31883 if id is None else id for id in input_ids]
            
            inputs = {
                "input_ids": torch.tensor([input_ids], dtype=torch.long, device=self.device)
            }
            # Additional processing can be added here
            results.append(inputs)
        
        return results