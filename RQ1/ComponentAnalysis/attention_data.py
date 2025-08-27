from transformers import AutoTokenizer
import json
import pandas as pd
from tqdm import tqdm

def tokenize_text(path, tokenizer_name, language, save_path):
    """
    Tokenize words in the dataset using the specified tokenizer.
    """
    with open("/home/hyujang/multilingual-inner-lexicon/RQ1/config.json", "r") as f:
        config = json.load(f)
    
    token_key = config["tokenizers"].get(tokenizer_name)
    token_value = None
    
    if token_key:
        try:
            with open("/home/hyujang/multilingual-inner-lexicon/user_config.json", "r") as f:
                user_config = json.load(f)
                token_value = user_config["huggingface_token"].get(token_key)
        except FileNotFoundError:
            print("Warning: user_config.json not found")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, token=token_value)
    
    df = pd.read_csv(path)
    df.dropna(subset=["word"], inplace=True)
    
    tokens_list = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing words"):
        word = row["word"]
        try:
            tokens_list.append(tokenizer.tokenize(word))
        except Exception as e:
            print(f"Error tokenizing word '{word}': {e}")
            tokens_list.append([])
    
    df["tokens"] = tokens_list
    df["token_num"] = df["tokens"].apply(len)
    
    print(f"Number of 2-token words ({tokenizer_name} - {language}):", len(df[df["token_num"] == 2]))
    
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df

def sample_real_words(tokens_df, token_num, num_samples, num_quantiles=5, seed=2025):
    """
    Samples real words based on token number and frequency quantiles, ensuring no duplicates.

    Args:
    - tokens_df (pd.DataFrame): DataFrame containing tokenized words.
    - token_num (int): Number of tokens per word.
    - num_samples (int): Number of samples to generate.
    - num_quantiles (int): Number of frequency quantiles.
    - seed (int): Random seed for reproducibility.

    Returns:
    - pd.DataFrame: Sampled real words.
    - int: Remaining samples needed.
    - list: Indices of sampled words.
    """
    tokens_df['freq_quantile'], bins = pd.qcut(
        tokens_df['original_frequency'], 
        num_quantiles, 
        labels=False, 
        duplicates='drop', 
        retbins=True
    )
    
    sampled = []
    for quantile in range(num_quantiles):
        quantile_df = tokens_df[
            (tokens_df['token_num'] == token_num) & 
            (tokens_df['freq_quantile'] == quantile)
        ]
        if len(quantile_df) > 0:
            sample_size = min(len(quantile_df), num_samples // num_quantiles)
            sampled.append(
                quantile_df.sample(sample_size, replace=False, random_state=seed)
            )
    
    sampled_df = pd.concat(sampled, ignore_index=False).drop_duplicates(subset=['word'])
    sampled_indices = sampled_df.index.to_list()

    # Handle cases where the sampled DataFrame has fewer rows than required
    if len(sampled_df) < num_samples:
        print(f"Remaining before additional sampling for {token_num}-token words:", num_samples - len(sampled_df))
        remaining = num_samples - len(sampled_df)
        other_df = tokens_df[tokens_df['token_num'] == token_num].drop(sampled_df.index, errors='ignore')
        
        if len(other_df) > 0:
            additional_samples = other_df.sample(
                min(len(other_df), remaining), 
                replace=False, 
                random_state=seed
            )
            sampled_indices += additional_samples.index.to_list()
            sampled_df = pd.concat([sampled_df, additional_samples]).drop_duplicates(subset=['word']).reset_index(drop=True)
    
    remaining = num_samples - len(sampled_df)
    print(f"{remaining} remaining after sampling {len(sampled_df)} {token_num}-token words.")
    
    return sampled_df, remaining, sampled_indices


if __name__ == "__main__":
    # Configuration
    LANGUAGE = "German"
    MODEL_NAME = "google/gemma-3-12b-it"
    
    # Tokenize the dataset
    df = tokenize_text(
        f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/ComponentAnalysis/{LANGUAGE}_wiki_noun_frequencies_context.csv",
        tokenizer_name=MODEL_NAME,
        language=LANGUAGE,
        save_path=f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/ComponentAnalysis/{MODEL_NAME.split('/')[-1]}_{LANGUAGE}_wiki_noun_frequencies_context.csv"
    )
    
    # Sample 2-token words
    sampled_df, remaining, sampled_indices = sample_real_words(
        df, 
        token_num=2, 
        num_samples=2440, 
        num_quantiles=5, 
        seed=2025
    )
    
    # Save sampled results
    sampled_df.to_csv(
        f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/ComponentAnalysis/{MODEL_NAME.split('/')[-1]}_{LANGUAGE}_wiki_noun_frequencies_context_2token.csv", 
        index=False
    )
    
    print(f"Saved {len(sampled_df)} sampled 2-token words")