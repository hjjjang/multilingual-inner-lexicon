import os
import json

import pandas as pd
import random

# For Korean
try:
    from jamo import h2j, j2h, hangul_to_jamo, jamo_to_hangul
except ImportError:
    pass


BASE_DIR = "/home/hyujang/multilingual-inner-lexicon"
with open(os.path.join(BASE_DIR, "RQ1/config.json"), "r") as f:
    CONFIG = json.load(f)

# Configuration variables
model_name_map = {
    "llama_2_7b": "Llama-2-7b-chat-hf",
    "babel_9b": "Babel-9B-Chat",
    "gemma_12b": "gemma-3-12b-it"
}

MIN_WORD_LEN = 3
MIN_JAMO_LEN = 2
MIN_WORD_FREQ = CONFIG["min_freq"]

NUM_SAMPLES = 700
NUM_QUANTILES = CONFIG["num_quantiles"]

RANDOM_SEED = CONFIG["seed"]
random.seed(RANDOM_SEED)

# --- ENGLISH & GERMAN FUNCTIONS ---

def random_split(word, language, min_word_len=MIN_WORD_LEN, min_jamo_len=MIN_JAMO_LEN):
    if language == "Korean":
        jamos = list(split_jamos(word))
        if len(jamos) <= 1:
            return [word]
        num_splits = random.randint(1, min(4, len(jamos) - min_jamo_len))
        split_points = sorted(random.sample(range(1, len(jamos)), num_splits))
        jamo_tokens = [jamos[i:j] for i, j in zip([0] + split_points, split_points + [None])]
        return [''.join(token) for token in jamo_tokens]
    else:
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
    elif typo_type == "transposition":
    # elif typo_type == "transposition" and len(word) >= 3:
        position = random.randint(1, len(word) - 2)
        return word[:position] + word[position + 1] + word[position] + word[position + 2:], typo_type
    else:
        return word, typo_type

# --- KOREAN FUNCTIONS ---

def count_jamos(word):
    return len(h2j(word))

def split_jamos(word):
    return list(h2j(word))

def join_jamos(jamos):
    return j2h(''.join(jamos))


CHO = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
JUN = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
JON = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

def decompose_syllable(s):
    code = ord(s) - 0xAC00
    cho = code // (21 * 28)
    jun = (code % (21 * 28)) // 28
    jon = code % 28
    return cho, jun, jon

def compose_syllable(cho, jun, jon):
    return chr(0xAC00 + cho * 21 * 28 + jun * 28 + jon)

def introduce_korean_syllable_typo(word, typo_type=None):
    if typo_type is None:
        typo_type = random.choice(["substitution", "deletion", "insertion", "transposition"])
    chars = list(word)
    if not chars:
        return word, typo_type
    idx = random.randint(0, len(chars) - 1)
    c = chars[idx]
    try:
        cho, jun, jon = decompose_syllable(c)
    except:
        return word, typo_type
    if typo_type == "substitution":
        part = random.choice(['cho', 'jun', 'jon'])
        if part == 'cho':
            cho = random.choice([i for i in range(len(CHO)) if i != cho])
        elif part == 'jun':
            jun = random.choice([i for i in range(len(JUN)) if i != jun])
        elif part == 'jon':
            jon = random.choice([i for i in range(len(JON)) if i != jon])
    elif typo_type == "deletion":
        part = random.choice(['cho', 'jun', 'jon'])
        if part == 'jon':
            jon = 0
    elif typo_type == "insertion":
        if jon == 0:
            jon = random.randint(1, len(JON) - 1)
    elif typo_type == "transposition":
        if jon != 0:
            cho, jon = jon % len(CHO), cho % len(JON)
    chars[idx] = compose_syllable(cho, jun, jon)
    return ''.join(chars), typo_type

# --- MAIN FUNCTIONS ---
def sample_by_freq(df):
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
    
    df["splitted_tokens"] = df["word"].apply(lambda x: random_split(x, min_word_len=MIN_WORD_LEN, language=LANGUAGE))
    print(f"Splitted tokens length distribution before freq-based sampling: {df["splitted_tokens"].apply(len).value_counts()}")

    sampled_df = sample_by_freq(df)
    print(sampled_df["splitted_tokens"].apply(len).value_counts())
    
    sampled_df = sampled_df[['word', "splitted_tokens", "same_token_num", "same_token_num2", "freq", "freq_quantile", "word_len"]]
    sampled_df.to_csv(f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/WordIdentity/single_token_splitted_{MODEL_NAME}_{LANGUAGE}.csv", index=False)


def run_typo_split(LANGUAGE, TOKENIZER, MODEL_NAME):
    df = pd.read_csv(f"/home/hyujang/multilingual-inner-lexicon/data/{LANGUAGE}_tokenizers_comparison.csv")
    df.drop_duplicates(subset=["word"], keep="first", inplace=True)
    df["word_len"] = df["word"].apply(lambda x: len(x))
    df = df[df['freq'] >= MIN_WORD_FREQ]
    df = df[(df[f"token_num_{TOKENIZER}"]==1) & (df["word_len"]>MIN_WORD_LEN-1)].reset_index(drop=True)
    print(f"Number of candidates words: {len(df)}")

    df[["typo_tokens", "typo_type"]] = df["word"].apply(lambda x: pd.Series(introduce_typo(x, typo_type=None, language=LANGUAGE)))
    df["typo_word_len"] = df["typo_tokens"].apply(lambda x: len(x))
    # df = df[(df["typo_word_len"]>MIN_WORD_LEN)].reset_index(drop=True)
    df["splitted_typo_tokens"] = df["typo_tokens"].apply(lambda x: random_split(x, min_word_len=MIN_WORD_LEN, language=LANGUAGE))
    print(df["typo_type"].value_counts())
    print(df["splitted_typo_tokens"].apply(len).value_counts())
    
    sampled_df = sample_by_freq(df)
    print(sampled_df["splitted_typo_tokens"].apply(len).value_counts())

    sampled_df = sampled_df[['word', "typo_tokens", "splitted_typo_tokens", "typo_type", "same_token_num", "same_token_num2", "freq", "freq_quantile", "word_len", "typo_word_len"]]
    sampled_df.to_csv(f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/WordIdentity/single_token_typos_{MODEL_NAME}_{LANGUAGE}.csv", index=False)


def run_simple_split_korean(LANGUAGE, TOKENIZER, MODEL_NAME):
    df = pd.read_csv(f"/home/hyujang/multilingual-inner-lexicon/data/{LANGUAGE}_tokenizers_comparison.csv")
    df["word_len"] = df["word"].apply(len)
    df['jamo_len'] = df['word'].apply(count_jamos)
    df = df[(df[f"token_num_{TOKENIZER}"]==1) & (df["jamo_len"]>MIN_JAMO_LEN)].reset_index(drop=True)
    df[f"splitted_tokens_{TOKENIZER}"] = df["word"].apply(lambda x: random_split(x, MIN_JAMO_LEN))
    print(df[f"splitted_tokens_{TOKENIZER}"].apply(len).value_counts())
    # Typo processing for Korean can be added similarly if needed
    

if __name__ == "__main__":
    LANGUAGE = "English"  # Options: "English", "German", "Korean"
    TOKENIZER = "llama_2_7b"  # Options: "babel_9b", "gemma_12b", "llama_2_7b"
    MODEL_NAME = model_name_map[TOKENIZER]

    run_simple_split(LANGUAGE, TOKENIZER, MODEL_NAME)
    run_typo_split(LANGUAGE, TOKENIZER, MODEL_NAME)

    LANGUAGE = "German"  # Options: "English", "German", "Korean"
    TOKENIZER = "llama_2_7b"  # Options: "babel_9b", "gemma_12b", "llama_2_7b"
    MODEL_NAME = model_name_map[TOKENIZER]

    run_simple_split(LANGUAGE, TOKENIZER, MODEL_NAME)
    run_typo_split(LANGUAGE, TOKENIZER, MODEL_NAME)
