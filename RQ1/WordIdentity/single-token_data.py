import os
import json

import pandas as pd
import random


BASE_DIR = "/home/hyujang/multilingual-inner-lexicon"
with open(os.path.join(BASE_DIR, "RQ1/config.json"), "r") as f:
    CONFIG = json.load(f)

# Configuration variables
model_name_map = {
    "llama_2_7b": "Llama-2-7b-chat-hf",
    "babel_9b": "Babel-9B-Chat",
    "gemma_12b": "gemma-3-12b-it"
}

model_full_name_map = {
    "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "Babel-9B-Chat": "Tower-Babel/Babel-9B-Chat",
    "gemma-3-12b-it": "google/gemma-3-12b-it"
}

MIN_WORD_LEN = 3
MIN_WORD_FREQ = CONFIG["min_freq"]

NUM_SAMPLES = 700
NUM_QUANTILES = CONFIG["num_quantiles"]

RANDOM_SEED = CONFIG["seed"]
random.seed(RANDOM_SEED)

# --- ENGLISH & GERMAN FUNCTIONS ---

def random_split_valid(word, language, tokenizer, min_word_len=MIN_WORD_LEN):
    def is_valid_split(tokens):
        return all(tokenizer.convert_tokens_to_ids(token) != tokenizer.unk_token_id for token in tokens)

    max_attempts = 100
    attempts = 0

    if len(word) <= 1:
        return [word]
    while attempts < max_attempts:
        attempts += 1
        try:
            num_splits = random.randint(1, min(4, len(word) - min_word_len - 1))
        except:
            num_splits = 1
        split_points = sorted(random.sample(range(1, len(word)), num_splits))
        tokens = [word[i:j] for i, j in zip([0] + split_points, split_points + [None])]
        if is_valid_split(tokens):
            return tokens
        # print(f"word: {word}, tokens: {tokens}")
    print(f"Discarding word '{word}' after {max_attempts} attempts.")
    return None
    
def random_split(word, language, min_word_len=MIN_WORD_LEN):
    if len(word) <= 1:
        return [word]
    try:
        num_splits = random.randint(1, min(4, len(word) - min_word_len - 1))
    except:
        num_splits = 1
    split_points = sorted(random.sample(range(1, len(word)), num_splits))
    tokens = [word[i:j] for i, j in zip([0] + split_points, split_points + [None])]
    return tokens

def introduce_typo(word, language, typo_type=None):
    if language == "English":
        letters = 'abcdefghijklmnopqrstuvwxyz'
    elif language == "German":
        letters = 'abcdefghijklmnopqrstuvwxyzäöüß'
    
    if typo_type is None:
        typo_type = random.choice(["substitution", "deletion", "insertion"])
    if typo_type == "substitution":
        position = random.randint(1, len(word) - 1)
        original_char = word[position]
        typo_char = random.choice([c for c in letters if c != original_char])
        return word[:position] + typo_char + word[position + 1:], typo_type
    elif typo_type == "deletion":
        position = random.randint(1, len(word) - 1)
        return word[:position] + word[position + 1:], typo_type
    elif typo_type == "insertion":
        position = random.randint(1, len(word) - 1)
        typo_char = random.choice(letters)
        return word[:position] + typo_char + word[position:], typo_type
    else:
        return word, typo_type

# --- MAIN FUNCTIONS ---
def sample_by_freq(df):
    # df = df[df["splitted_tokens"].notnull()]  # Exclude rows with None in 'splitted_tokens'
    df['freq_quantile'], bins = pd.qcut(df['freq'], NUM_QUANTILES, labels=False, duplicates='drop', retbins=True)
    num_quantiles = df['freq_quantile'].nunique()
    samples_per_quantile = NUM_SAMPLES // num_quantiles
    
    sampled = []
    for quantile in range(num_quantiles):
        quantile_df = df[df['freq_quantile'] == quantile]
        if len(quantile_df) > 0:
            sampled.append(quantile_df.sample(min(len(quantile_df), samples_per_quantile), replace=False, random_state=RANDOM_SEED))
    sampled_df = pd.concat(sampled, ignore_index=False).drop_duplicates(subset=['word'])
    
    if len(sampled_df) < NUM_SAMPLES:
        remaining = NUM_SAMPLES - len(sampled_df)
        other_df = df.drop(sampled_df.index, errors='ignore')
        print(f"remaining: {remaining}, other_df: {len(other_df)}")
        additional_samples = other_df.sample(min(len(other_df), remaining), replace=False, random_state=RANDOM_SEED)
        print(f"additional_samples: {len(additional_samples)}")
        sampled_df = pd.concat([sampled_df, additional_samples]).drop_duplicates(subset=['word'])

    print(f"sampled_df: {len(sampled_df)}")
    
    return sampled_df.reset_index(drop=True)

def run_simple_split(LANGUAGE, TOKENIZER, MODEL_NAME):
    df = pd.read_csv(f"/home/hyujang/multilingual-inner-lexicon/data/{LANGUAGE}_tokenizers_comparison.csv")
    df.drop_duplicates(subset=["word"], keep="first", inplace=True)
    df["word_len"] = df["word"].apply(len)
    df = df[df['freq'] >= MIN_WORD_FREQ]
    df = df[(df[f"token_num_{TOKENIZER}"]==1) & (df["word_len"]>MIN_WORD_LEN)].reset_index(drop=True)
    
    print(f"Number of candidates words: {len(df)}")
    
    # df["splitted_tokens"] = df["word"].apply(lambda x: random_split(x, min_word_len=MIN_WORD_LEN, language=LANGUAGE))
    df["splitted_tokens"] = df["word"].apply(lambda x: random_split_valid(x, LANGUAGE, tokenizer, min_word_len=MIN_WORD_LEN)) # use tokenizer for validation
    df.dropna(subset=["splitted_tokens"], inplace=True)  # Remove rows where 'splitted_tokens' is None
    print(f"Splitted tokens length distribution before freq-based sampling: {df["splitted_tokens"].apply(len).value_counts()}")

    sampled_df = sample_by_freq(df)
    print(sampled_df["splitted_tokens"].apply(len).value_counts())
    
    sampled_df = sampled_df[['word', "splitted_tokens", "same_token_num", "same_token_num2", "freq", "freq_quantile", "word_len"]]
    # sampled_df.to_csv(f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/WordIdentity/single_token_splitted_{MODEL_NAME}_{LANGUAGE}.csv", index=False)
    sampled_df.to_csv(f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/WordIdentity/single_token_splitted_{MODEL_NAME}_{LANGUAGE}_v2.csv", index=False) # all tokens are valid, no UNK token


def run_typo_split(LANGUAGE, TOKENIZER, MODEL_NAME):
    df = pd.read_csv(f"/home/hyujang/multilingual-inner-lexicon/data/{LANGUAGE}_tokenizers_comparison.csv")
    df.drop_duplicates(subset=["word"], keep="first", inplace=True)
    df["word_len"] = df["word"].apply(lambda x: len(x))
    df = df[df['freq'] >= MIN_WORD_FREQ]
    df = df[(df[f"token_num_{TOKENIZER}"]==1) & (df["word_len"]>MIN_WORD_LEN-1)].reset_index(drop=True)
    print(f"Number of candidates words: {len(df)}")

    df[["typo_tokens", "typo_type"]] = df["word"].apply(lambda x: pd.Series(introduce_typo(x, typo_type=None, language=LANGUAGE)))
    df["typo_word_len"] = df["typo_tokens"].apply(lambda x: len(x))
    # df["splitted_typo_tokens"] = df["typo_tokens"].apply(lambda x: random_split(x, min_word_len=MIN_WORD_LEN, language=LANGUAGE))
    df["splitted_typo_tokens"] = df["typo_tokens"].apply(lambda x: random_split_valid(x, LANGUAGE, tokenizer, min_word_len=MIN_WORD_LEN)) # use tokenizer for validation
    df.dropna(subset=["splitted_typo_tokens"], inplace=True)
    
    print(df["typo_type"].value_counts())
    print(f"Splitted tokens length distribution before freq-based sampling: {df["splitted_typo_tokens"].apply(len).value_counts()}")
    
    sampled_df = sample_by_freq(df)
    print(sampled_df["splitted_typo_tokens"].apply(len).value_counts())

    sampled_df = sampled_df[['word', "typo_tokens", "splitted_typo_tokens", "typo_type", "same_token_num", "same_token_num2", "freq", "freq_quantile", "word_len", "typo_word_len"]]
    sampled_df.to_csv(f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/WordIdentity/single_token_typos_{MODEL_NAME}_{LANGUAGE}_v2.csv", index=False)
    
with open("/home/hyujang/multilingual-inner-lexicon/user_config.json", "r") as f:
    user_config = json.load(f)
    token_value = user_config["huggingface_token"].get("token_1", None)

def setup_tokenizer(self):
    # pass
    if self.tokenizer_name == "Tower-Babel/Babel-9B-Chat":
        self.tokenizer.add_special_tokens({'unk_token': 'UNK'})
        self.tokenizer.unk_token_id = self.tokenizer.convert_tokens_to_ids('UNK')

if __name__ == "__main__":
    LANGUAGE = "German"  # Options: "English", "German"
    TOKENIZER = "babel_9b"  # Options: "babel_9b", "gemma_12b", "llama_2_7b"
    MODEL_NAME = model_name_map[TOKENIZER]
    MODEL_FULL_NAME = model_full_name_map[MODEL_NAME]

    from transformers import AutoTokenizer

    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, token=token_value)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_FULL_NAME, use_fast=True)
    
    run_simple_split(LANGUAGE, TOKENIZER, MODEL_NAME)
    # run_typo_split(LANGUAGE, TOKENIZER, MODEL_NAME)
