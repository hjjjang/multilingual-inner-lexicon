import json
import pandas as pd
from transformers import AutoTokenizer
from nltk.corpus import wordnet
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import argparse
import os

class WordNonwordData:
    def __init__(self, language, tokenizer_name, base_dir="/home/hyujang/multilingual-inner-lexicon"):
        
        self.base_dir = base_dir
        # Load configuration
        with open(os.path.join(self.base_dir, "RQ1/config.json"), "r") as f:
            self.config = json.load(f)
            
        self.num_real_words = self.config["num_real_words"]
        self.num_non_words = self.config["num_non_words"]
        self.seed = self.config["seed"]
        self.language = language
        self.tokenizer_name = tokenizer_name
        token_key = self.config["tokenizers"][self.tokenizer_name]
        if token_key:
            with open(os.path.join(self.base_dir, "user_config.json"), "r") as f:
                user_config = json.load(f)
                self.token_value = user_config["huggingface_token"].get(token_key)
        else:
            self.token_value = None
            

    def load_dataset(self):
        if self.language == "English":
            # noun_synsets = list(wordnet.all_synsets(pos=wordnet.NOUN))
            # en_nouns = set(lemma.name() for synset in noun_synsets for lemma in synset.lemmas() if "_" not in lemma.name())
            # return list(en_nouns)
            en_nouns_df = pd.read_csv(os.path.join(self.base_dir, self.config['datasets']['English-wiki']))
            return en_nouns_df
        elif self.language == "Korean":
            # ko_nouns_df = pd.read_csv(os.path.join(self.base_dir, self.config['datasets']['Korean']))
            ko_nouns_df = pd.read_csv(os.path.join(self.base_dir, self.config['datasets']['Korean-wiki']))
            ko_nouns_df = ko_nouns_df[~ko_nouns_df['word'].str.contains(r'[^\uac00-\ud7a3]', na=False)]
            return ko_nouns_df
        elif self.language == "German":
            de_nouns_df = pd.read_csv(os.path.join(self.base_dir, self.config['datasets']['German-wiki']))
            return de_nouns_df

    def tokenize_and_analyze(self, input_data):
        """
        Tokenizes the input data and analyzes token distribution.
    
        Args:
        - input_data (list, pd.Series, or pd.DataFrame): Input data to tokenize.
    
        Returns:
        - pd.DataFrame: DataFrame containing tokenized words, their token counts, and other columns from the input DataFrame (if applicable).
        """
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True, tokens=self.token_value)
    
        # Handle different input types
        if isinstance(input_data, list):
            input_data = pd.Series(input_data, name="word")
            original_df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            if 'word' in input_data.columns: # for korean df
                original_df = input_data.copy()  # Keep all columns from the input DataFrame
                input_data = input_data['word']
            else:
                raise ValueError("The input DataFrame must contain a 'word' column.")
        else:
            raise ValueError("Input data must be a list, pd.Series, or pd.DataFrame.")
    
        word_to_tokens = {}
        for word in tqdm(input_data, desc="Tokenizing words"):
            try:
                tokens = tokenizer.tokenize(word)
                word_to_tokens[word] = tokens
            except Exception as e:
                print(f"Error tokenizing word '{word}': {e}")
                word_to_tokens[word] = None
    
        tokens_df = pd.DataFrame(word_to_tokens.items(), columns=['word', 'tokens'])
        tokens_df = tokens_df.dropna(subset=['word'])
        tokens_df = tokens_df[~tokens_df['word'].str.contains(" ")]  # Remove words with spaces
        tokens_df.reset_index(inplace=True, drop=True)
    
        # Add token count column
        tokens_df['token_num'] = tokens_df['tokens'].apply(lambda x: len(x) if x else 0)
    
        # Merge with original DataFrame to include all other columns
        if isinstance(original_df, pd.DataFrame):
            tokens_df = pd.merge(tokens_df, original_df, on='word', how='left')
    
        # Plot token distribution
        plt.figure(figsize=(10, 6))
        tokens_df['token_num'].hist(bins=range(1, tokens_df['token_num'].max() + 2), edgecolor='black', alpha=0.7)
        plt.xlabel("Number of Tokens")
        plt.ylabel("Frequency")
        plt.title(f"Token Distribution for {self.tokenizer_name} ({self.language})")
        if self.config.get("save_plots", False):
            plt.savefig(os.path.join(self.base_dir, f"output/image/token_dist_{self.tokenizer_name.split('/')[1]}_{self.language}.png"))
        plt.show()
    
        return tokens_df

    def sample_real_words(self, tokens_df, token_num, num_samples):
        """
        Samples real words based on token number and frequency quantiles, ensuring no duplicates.

        Args:
        - tokens_df (pd.DataFrame): DataFrame containing tokenized words.
        - token_num (int): Number of tokens per word.
        - num_samples (int): Number of samples to generate.

        Returns:
        - pd.DataFrame: Sampled real words.
        """
        sampled = []
        for quantile in range(self.config['num_quantiles']):
            quantile_df = tokens_df[(tokens_df['token_num'] == token_num) & (tokens_df['freq_quantile'] == quantile)]
            if len(quantile_df) > 0:
                sampled.append(quantile_df.sample(min(len(quantile_df), num_samples // self.config['num_quantiles']), 
                                                replace=False, random_state=self.seed))
        
        sampled_df = pd.concat(sampled, ignore_index=True).drop_duplicates(subset=['word']).reset_index(drop=True)

        # Handle cases where the sampled DataFrame has fewer rows than required
        if len(sampled_df) < num_samples:
            remaining = num_samples - len(sampled_df)
            other_df = tokens_df[tokens_df['token_num'] == token_num].drop(sampled_df.index, errors='ignore')
            additional_samples = other_df.sample(min(len(other_df), remaining), replace=False, random_state=self.seed)
            sampled_df = pd.concat([sampled_df, additional_samples]).drop_duplicates(subset=['word']).reset_index(drop=True)

        return sampled_df
    
    def generate_non_sensible_words(self, tokens_df, token_num, num_generated_words):
        if self.seed is not None:
            random.seed(self.seed)
        df_tokens = tokens_df[tokens_df['token_num'] == token_num]
        token_bags = [[] for _ in range(token_num)]
        for tokens in df_tokens['tokens']:
            for i in range(token_num):
                token_bags[i].append(tokens[i])
        original_words_set = set(tokens_df['word'])
        generated_words = set()  # Use a set to ensure uniqueness
        while len(generated_words) < num_generated_words:
            new_word_tokens = [random.choice(token_bags[i]) for i in range(token_num)]
            new_word = "".join(new_word_tokens)
            if new_word not in original_words_set and new_word not in generated_words:
                generated_words.add((new_word, tuple(new_word_tokens), token_num))
        generated_df = pd.DataFrame(list(generated_words), columns=["word", "tokens", "token_num"])
        generated_df = generated_df.sample(frac=1, random_state=self.seed).drop_duplicates(subset=['word']).reset_index(drop=True)
        return generated_df

    def generate_real_and_non_words(self, input_data):
        tokens_df = self.tokenize_and_analyze(input_data)
        
        if 'freq' in tokens_df.columns:
            print(f"Number of words before filtering: {len(tokens_df)}")
            tokens_df = tokens_df[tokens_df['freq'] >= self.config['min_freq']]
            print(f"Number of words after filtering out words with frequency less than {self.config['min_freq']}: {len(tokens_df)}")
            tokens_df['freq_quantile'], bins = pd.qcut(tokens_df['freq'], self.config['num_quantiles'], labels=False,  duplicates='drop', retbins=True)
            print("Quantile bins (ranges):", bins)
        else:
            tokens_df['freq_quantile'] = 0  # Default to a single level if no frequency column is provided
        
        print("Token count distribution (number of tokens per word):")
        print(tokens_df['token_num'].value_counts())
        
        real_words_2_tokens = int(self.num_real_words * 0.53)
        real_words_3_tokens = int(self.num_real_words * 0.373)
        real_words_4_tokens = self.num_real_words - real_words_2_tokens - real_words_3_tokens
        
        real_words_df = pd.concat([
            self.sample_real_words(tokens_df, 2, real_words_2_tokens),
            self.sample_real_words(tokens_df, 3, real_words_3_tokens),
            self.sample_real_words(tokens_df, 4, real_words_4_tokens)
        ]).reset_index(drop=True)
        
        if len(real_words_df) < self.num_real_words:
            more_needed = self.num_real_words - len(real_words_df)
            most_common_token_num = tokens_df['token_num'].value_counts().idxmax()
            print(f"Note: Only {len(real_words_df)} real words sampled. Need {more_needed} more from {most_common_token_num}-token words.")
            real_words_df = pd.concat([
                real_words_df,
                self.sample_real_words(tokens_df, most_common_token_num, more_needed)
            ]).reset_index(drop=True)

        non_words_2_tokens = int(self.num_non_words * 0.53)
        non_words_3_tokens = int(self.num_non_words * 0.373)
        non_words_4_tokens = self.num_non_words - non_words_2_tokens - non_words_3_tokens

        non_words_df = pd.concat([
            self.generate_non_sensible_words(tokens_df, 2, non_words_2_tokens), 
            self.generate_non_sensible_words(tokens_df, 3, non_words_3_tokens), 
            self.generate_non_sensible_words(tokens_df, 4, non_words_4_tokens)
        ]).reset_index(drop=True)
        
        
        
        real_words_df['label'] = "realword"
        non_words_df['label'] = "nonword"
        final_df = pd.concat([real_words_df, non_words_df]).reset_index(drop=True)
        return final_df
    
    def main(self):
        input_data = self.load_dataset()
        final_df = self.generate_real_and_non_words(input_data)
        final_df.to_csv(os.path.join(self.base_dir, f"data/r1_dataset_{self.tokenizer_name.split('/')[1]}_{self.language}-wiki.csv"), index=False)
        return final_df
    
    def tokenize_and_save(self):
        input_data = self.load_dataset()
        tokens_df = self.tokenize_and_analyze(input_data)
        # data_path = os.path.join(self.base_dir, f"data/ko_wiki_nooun_frequencies_kiwi_{self.tokenizer_name.split('/')[1]}_{self.language}-tokenized.csv")
        # tokens_df.to_csv(data_path, index=False)
        return tokens_df


def compare_tokenizers(lang):
    tokenizers = [
        ("Tower-Babel/Babel-9B-Chat", "babel_9b"),
        ("google/gemma-3-12b-it", "gemma_12b"),
        ("meta-llama/Llama-2-7b-chat-hf", "llama_2_7b"),
    ]
    
    # Initialize an empty list to store tokenized DataFrames
    tokenized_dfs = []
    
    for tokenizer_name, model_alias in tokenizers:
        word_nonword_cls = WordNonwordData(lang, tokenizer_name)
        tokens_df = word_nonword_cls.tokenize_and_save()
        
        # if tokenizer_name == "meta-llama/Llama-2-7b-chat-hf":
        #     tokens_df["tokens"] = tokens_df["tokens"].apply(lambda x: x[1:] if x else x)
        #     tokens_df["token_num"] = tokens_df["tokens"].apply(lambda x: len(x) if x else 0)

        # Rename columns to include the model alias, except for 'word' and 'freq'
        tokens_df = tokens_df.rename(columns={
            "tokens": f"tokens_{model_alias}",
            "token_num": f"token_num_{model_alias}"
        })
        
        tokenized_dfs.append(tokens_df)
        
    # Merge all tokenized DataFrames on the 'word' column
    final_df = tokenized_dfs[0]
    for df in tokenized_dfs[1:]:
        # Drop the 'freq' column from subsequent DataFrames before merging
        df = df.drop(columns=["freq"], errors="ignore")
        final_df = pd.merge(final_df, df, on="word", how="outer")
    
    # Calculate average token number and check if all tokenizers have the same token number
    token_num_columns = [f"token_num_{model_alias}" for _, model_alias in tokenizers]
    final_df["avg_token_num"] = final_df[token_num_columns].mean(axis=1)
    final_df["same_token_num"] = final_df[token_num_columns].nunique(axis=1) == 1

    # Move the 'freq' column to the last position
    if "freq" in final_df.columns:
        freq_column = final_df.pop("freq")
        final_df["freq"] = freq_column
    
    # Sort the DataFrame by 'freq' in descending order
    if "freq" in final_df.columns:
        final_df = final_df.sort_values(by="freq", ascending=False)
    
    # Save the final DataFrame to a CSV file
    final_df.to_csv(f"data/{lang}_tokenizers_comparison.csv", index=False)
    # final_df.to_csv("test.csv", index=False)
    
    print(f"Processing complete for {lang}.")
    
    
if __name__ == "__main__":
    # word_nonword_cls = WordNonwordData("Korean", "Tower-Babel/Babel-9B-Chat")
    # word_nonword_cls = WordNonwordData("English", "Tower-Babel/Babel-9B-Chat")
    # word_nonword_cls = WordNonwordData("German", "Tower-Babel/Babel-9B-Chat")
    # results = word_nonword_cls.main()
        
    compare_tokenizers("Korean")
    compare_tokenizers("English")
    compare_tokenizers("German")