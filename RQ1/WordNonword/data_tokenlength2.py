import pandas as pd
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import os
import ast

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils import load_config, setup_tokenizer

class WordNonwordData:
    def __init__(self, language: str, tokenizer_name: str, base_dir: str = "/home/hyujang/multilingual-inner-lexicon"):
        self.base_dir = base_dir
        self.language = language
        self.tokenizer_name = tokenizer_name
        
        self.config = load_config()
        self.num_real_words = self.config["num_real_words"]
        self.num_non_words = self.config["num_non_words"]
        self.seed = self.config["seed"]
        
        self.tokenizer = setup_tokenizer(tokenizer_name)

    def load_dataset(self) -> pd.DataFrame:
        """Load dataset based on language."""
        dataset_paths = {
            "English": self.config['datasets']['English-wiki'],
            "Korean": self.config['datasets']['Korean-wiki'],
            "German": self.config['datasets']['German-wiki']
        }
        
        if self.language not in dataset_paths:
            raise ValueError(f"Unsupported language: {self.language}")
        
        dataset_path = os.path.join(self.base_dir, dataset_paths[self.language])
        return pd.read_csv(dataset_path)

    def tokenize_and_analyze(self, input_data):
        """Tokenize input data and analyze token distribution."""
        # Handle different input types
        if isinstance(input_data, list):
            input_data = pd.Series(input_data, name="word")
            original_df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            if 'word' not in input_data.columns:
                raise ValueError("The input DataFrame must contain a 'word' column.")
            original_df = input_data.copy()
            input_data = input_data['word']
        else:
            raise ValueError("Input data must be a list, pd.Series, or pd.DataFrame.")
    
        # Tokenize words
        word_to_tokens = {}
        for word in tqdm(input_data, desc="Tokenizing words"):
            try:
                tokens = self.tokenizer.tokenize(word)
                word_to_tokens[word] = tokens
            except Exception as e:
                print(f"Error tokenizing word '{word}': {e}")
                word_to_tokens[word] = None
    
        # Create DataFrame and clean data
        tokens_df = pd.DataFrame(word_to_tokens.items(), columns=['word', 'tokens'])
        tokens_df = tokens_df.dropna(subset=['word'])
        tokens_df = tokens_df[~tokens_df['word'].str.contains(" ", na=False)]
        tokens_df.reset_index(inplace=True, drop=True)
    
        # Add token count column
        tokens_df['token_num'] = tokens_df['tokens'].apply(lambda x: len(x) if x else 0)
    
        # Merge with original DataFrame
        tokens_df = pd.merge(tokens_df, original_df, on='word', how='left')
    
        # Plot token distribution
        self._plot_token_distribution(tokens_df)
        
        return tokens_df

    def _plot_token_distribution(self, tokens_df: pd.DataFrame):
        """Plot token distribution."""
        plt.figure(figsize=(10, 6))
        max_tokens = tokens_df['token_num'].max()
        bins = range(1, max_tokens + 2)
        
        tokens_df['token_num'].hist(bins=bins, edgecolor='black', alpha=0.7)
        plt.xlabel("Number of Tokens")
        plt.ylabel("Frequency")
        plt.title(f"Token Distribution for {self.tokenizer_name.split('/')[-1]} ({self.language})")
        
        if self.config.get("save_plots", False):
            output_dir = os.path.join(self.base_dir, "output/image")
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"token_dist_{self.tokenizer_name.split('/')[-1]}_{self.language}.png"))
        
        plt.show()

    def sample_real_words(self, tokens_df: pd.DataFrame, token_num: int, num_samples: int):
        """Sample real words based on token number and frequency quantiles."""
        sampled = []
        for quantile in range(self.config['num_quantiles']):
            quantile_df = tokens_df[
                (tokens_df['token_num'] == token_num) & 
                (tokens_df['freq_quantile'] == quantile)
            ]
            if len(quantile_df) > 0:
                sample_size = min(len(quantile_df), num_samples // self.config['num_quantiles'])
                sampled.append(quantile_df.sample(sample_size, replace=False, random_state=self.seed))
        
        sampled_df = pd.concat(sampled, ignore_index=False).drop_duplicates(subset=['word'])
        sampled_indices = sampled_df.index.tolist()

        # Handle cases where we need more samples
        if len(sampled_df) < num_samples:
            print(f"Remaining before additional sampling for {token_num}-token words: {num_samples - len(sampled_df)}")
            remaining = num_samples - len(sampled_df)
            other_df = tokens_df[tokens_df['token_num'] == token_num].drop(sampled_df.index, errors='ignore')
            if len(other_df) > 0:
                additional_samples = other_df.sample(min(len(other_df), remaining), replace=False, random_state=self.seed)
                sampled_indices.extend(additional_samples.index.tolist())
                sampled_df = pd.concat([sampled_df, additional_samples]).drop_duplicates(subset=['word']).reset_index(drop=True)
        
        remaining = max(0, num_samples - len(sampled_df))
        return sampled_df, remaining, sampled_indices
    
    def sample_non_words(self, tokens_df: pd.DataFrame, token_num: int, num_generated_words: int):
        """Generate non-words by randomly combining tokens."""
        if self.seed is not None:
            random.seed(self.seed)
            
        df_tokens = tokens_df[tokens_df['token_num'] == token_num]
        token_bags = [[] for _ in range(token_num)]
        
        for tokens in df_tokens['tokens']:
            for i in range(token_num):
                token_bags[i].append(tokens[i])
                
        original_words_set = set(tokens_df['word'])
        generated_words = set()
        
        while len(generated_words) < num_generated_words:
            new_word_tokens = [random.choice(token_bags[i]) for i in range(token_num)]
            new_word = "".join(new_word_tokens)
            if new_word not in original_words_set and new_word not in generated_words:
                generated_words.add((new_word, tuple(new_word_tokens), token_num))
                
        generated_df = pd.DataFrame(list(generated_words), columns=["word", "tokens", "token_num"])
        generated_df = generated_df.sample(frac=1, random_state=self.seed).drop_duplicates(subset=['word']).reset_index(drop=True)
        return generated_df

    def generate_real_and_non_words(self, tokens_df: pd.DataFrame) -> pd.DataFrame:
        """Generate balanced dataset of real and non-words with focus on 2-token words."""
        # Process frequency data if available
        if 'freq' in tokens_df.columns:
            print(f"Words before frequency filtering: {len(tokens_df)}")
            tokens_df = tokens_df[tokens_df['freq'] >= self.config['min_freq']]
            print(f"Words after filtering (freq >= {self.config['min_freq']}): {len(tokens_df)}")
            tokens_df['freq_quantile'], bins = pd.qcut(
                tokens_df['freq'], 
                self.config['num_quantiles'], 
                labels=False, 
                duplicates='drop', 
                retbins=True
            )
            print("Quantile bins (ranges):", bins)
        else:
            tokens_df['freq_quantile'] = 0
        
        # Convert string representations to lists if needed
        if isinstance(tokens_df['tokens'].iloc[0], str):
            tokens_df['tokens'] = tokens_df['tokens'].apply(ast.literal_eval)
        
        # Handle special case for Korean + Llama
        if self.language == "Korean" and "llama" in self.tokenizer_name.lower():
            tokens_df['tokens'] = tokens_df['tokens'].apply(lambda x: x[1:] if x else x)
            tokens_df['token_num'] = tokens_df['tokens'].apply(len)
        
        print("Token distribution:")
        print(tokens_df['token_num'].value_counts().sort_index())

        ########## GENERATE REAL WORDS (2-TOKEN FOCUS) ##########
        real_words_2_tokens = 3000  # Focus on 2-token words
        
        real_words_2_df, remaining_2, sampled_indices_2 = self.sample_real_words(
            tokens_df, 2, real_words_2_tokens
        )
        
        real_words_df = real_words_2_df
        print(f"Total real words sampled: {len(real_words_df)}")

        ########## GENERATE NON-WORDS (2-TOKEN FOCUS) ##########
        non_words_2_tokens = 3000  # Focus on 2-token words
        
        non_words_df = self.sample_non_words(tokens_df, 2, non_words_2_tokens)
        
        # Add labels and combine
        real_words_df['label'] = "realword"
        non_words_df['label'] = "nonword"
        
        final_df = pd.concat([real_words_df, non_words_df]).reset_index(drop=True)
        return final_df
    
    def main(self) -> pd.DataFrame:
        """Main pipeline execution for 2-token word analysis."""
        print(f"Processing {self.language} with {self.tokenizer_name} (2-token focus)")
        
        input_data = self.load_dataset()
        tokens_df = self.tokenize_and_analyze(input_data)
        final_df = self.generate_real_and_non_words(tokens_df)
        
        # Save with specific naming for 2-token analysis
        output_path = os.path.join(
            self.base_dir, 
            f"data/RQ1/WordNonword/r1_dataset_{self.tokenizer_name.split('/')[1]}_{self.language}-wiki-2token.csv"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"Dataset saved to: {output_path}")
        
        return final_df
    
    def main2(self) -> pd.DataFrame:
        """Alternative main method using existing tokenized data."""
        print(f"Processing existing tokenized data for {self.language} with {self.tokenizer_name}")
        
        tokens_df = self.load_dataset()
        final_df = self.generate_real_and_non_words(tokens_df)
        
        output_path = os.path.join(
            self.base_dir, 
            f"data/r1_dataset_{self.tokenizer_name.split('/')[1]}_{self.language}-wiki-2token.csv"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"Dataset saved to: {output_path}")
        
        return final_df
        
    def tokenize_and_save(self) -> pd.DataFrame:
        """Tokenize data and save for later use."""
        input_data = self.load_dataset()
        tokens_df = self.tokenize_and_analyze(input_data)
        
        output_path = os.path.join(
            self.base_dir, 
            f"data/ko_wiki_nooun_frequencies_kiwi_{self.tokenizer_name.split('/')[1]}_{self.language}-tokenized.csv"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tokens_df.to_csv(output_path, index=False)
        print(f"Tokenized data saved to: {output_path}")
        
        return tokens_df


if __name__ == "__main__":
    
    # Define experiment configurations
    tokenizers = [
        "Tower-Babel/Babel-9B-Chat",
        "google/gemma-3-12b-it", 
        "meta-llama/Llama-2-7b-chat-hf"
    ]
    languages = ["English", "Korean", "German"]
    
    for tokenizer_name in tokenizers:
        for lang in languages:
            print(f"\n{'='*60}")
            print(f"Processing: {lang} with {tokenizer_name}")
            print(f"{'='*60}")
            
            try:
                word_nonword_cls = WordNonwordData(lang, tokenizer_name)
                result_df = word_nonword_cls.main()
                print(f"Successfully processed {lang} with {tokenizer_name}")
                print(f"Generated {len(result_df)} samples")
                
            except Exception as e:
                print(f"Error processing {lang} with {tokenizer_name}: {e}")
                continue
