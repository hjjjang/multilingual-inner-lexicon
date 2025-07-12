from datasets import load_dataset
import spacy
from konlpy.tag import Okt
from collections import Counter
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import os
from nltk.stem import WordNetLemmatizer
import stanza
from kiwipiepy import Kiwi 
from collections import defaultdict
import re

LANGUAGE_MAP = {
    "English": "en",
    "German": "de",
    "Korean": "ko"}

def load_wikipedia_data(lang, sample_size=20000):
    wiki = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", columns=["text"])
    return wiki.shuffle(seed=2025).select(range(sample_size)).to_pandas()

def clean_text_list(text_list):
    cleaned = []
    for item in text_list:
        # Split lines to handle \n cleanly
        # Remove extra spaces, tabs, etc.
        item = item.strip()                         # Strip leading/trailing space
        item = re.sub(r'\s+', ' ', item)            # Normalize all whitespace            line = re.sub(r'[^\S\r\n]+', ' ', line)     # Remove extra non-visible spaces
        if item:                                    # Remove empty strings
            cleaned.append(item)
    return cleaned


def extract_nouns_with_sentences_and_frequency(text, lang, lemmatizer=None, nlp=None, kiwi=None):
    """
    Extract nouns, their frequencies, and the sentences containing them from the given text.
    
    Args:
        text (str): The input text.
        lang (str): The language of the text ('en', 'de', 'ko').
        lemmatizer (WordNetLemmatizer, optional): Lemmatizer for English.
        nlp (stanza.Pipeline, optional): Stanza pipeline for German.
        kiwi (Kiwi, optional): Kiwi tokenizer for Korean.
    
    Returns:
        dict: A dictionary where keys are nouns and values are tuples of (frequency, list of sentences containing the noun).
    """
    noun_to_data = {}

    # Split the text into sentences
    # sentences = re.split(r'(?<=[.!?])\s+', text)

    if lang == "en":
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        nouns = [
            lemmatizer.lemmatize(word.lower()) 
            for word, tag in tagged 
            if tag in ['NN', 'NNS']
        ]
        sentences = sent_tokenize(text)
    elif lang == "de":
        doc = nlp(text)
        nouns = [
            word.lemma
            for sentence in doc.sentences
            for word in sentence.words
            if word.upos == "NOUN"
        ]
        sentences = [sentence.text for sentence in doc.sentences]
    elif lang == "ko":
        doc = kiwi.tokenize(text)
        nouns = [
            token.form
            for token in doc
            if token.tag == "NNG"
        ]
        sentences = kiwi.split_into_sents(text)
        sentences = [s.text for s in sentences]
    else:
        raise ValueError("Unsupported language")
    
    sentences = clean_text_list(sentences)

    # Count noun frequencies
    noun_frequencies = Counter(nouns)

    # Map nouns to sentences
    for sentence in sentences:
        for noun in noun_frequencies.keys():
            if re.search(rf'\b{re.escape(noun)}\b', sentence, re.IGNORECASE):
                if noun not in noun_to_data:
                    noun_to_data[noun] = {"frequency": noun_frequencies[noun], "sentences": []}
                noun_to_data[noun]["sentences"].append(sentence)

    return noun_to_data


def process_wikipedia_nouns(lang, save_path=None):
    
    cache_path = f"cache/{lang}_wiki_noun_frequencies_context.csv"
    if os.path.exists(cache_path):
        print(f"Output file '{cache_path}' already exists. Skipping processing.")
        noun_frequencies_df = pd.read_csv(cache_path)    
    
    else:
        df = load_wikipedia_data(lang)
        
        if lang == "en":
            print("Extracting nouns using nltk for English...")
            lemmatizer = WordNetLemmatizer()
            tqdm.pandas(desc="Processing text")
            df["noun_frequencies"] = df["text"].progress_apply(lambda text: extract_nouns_with_sentences_and_frequency(text, lang, lemmatizer=lemmatizer))

        elif lang == "de":
            print(f"Extracting nouns using Stanza model for {lang}...")
            nlp = stanza.Pipeline(lang="de", processors="tokenize,pos,lemma", use_gpu=True)  # Initialize Stanza pipeline
            tqdm.pandas(desc="Processing text")
            df["noun_frequencies"] = df["text"].progress_apply(lambda text: extract_nouns_with_sentences_and_frequency(text, lang, nlp=nlp))

        elif lang == "ko":
            print(f"Extracting nouns using Kiwi tokenizer for {lang}...")
            kiwi = Kiwi()
            tqdm.pandas(desc="Processing text")
            df["noun_frequencies"] = df["text"].progress_apply(lambda text: extract_nouns_with_sentences_and_frequency(text, lang, kiwi=kiwi))

        final_data = defaultdict(lambda: {"frequency": 0, "sentences": []})

        # Iterate over the rows of df["noun_frequencies"]
        for noun_data in df["noun_frequencies"]:
            for noun, data in noun_data.items():
                # Add the frequency of the noun
                final_data[noun]["frequency"] += data["frequency"]
                # Extend the list of sentences containing the noun
                final_data[noun]["sentences"].extend(data["sentences"])

        # Convert the defaultdict to a regular dictionary
        final_data = {noun: {"frequency": data["frequency"], "sentences": list(set(data["sentences"]))} 
                    for noun, data in final_data.items()}

        # Convert the final data into a DataFrame (optional)
        noun_frequencies_df = pd.DataFrame([
            {"noun": noun, "frequency": data["frequency"], "sentences": data["sentences"]}
            for noun, data in final_data.items()
        ])

        # Display the DataFrame
        noun_frequencies_df.sort_values(by="frequency", ascending=False, inplace=True)
        noun_frequencies_df.to_csv(cache_path, index=False)
        
    if lang == "en":
        noun_frequencies_df = noun_frequencies_df[~noun_frequencies_df['noun'].str.contains(r'[^a-zA-Z]', na=False)]
    if lang == "de":
        noun_frequencies_df = noun_frequencies_df[~noun_frequencies_df['noun'].str.contains(r'[^a-zA-ZäöüÄÖÜß]', na=False)]
    if lang == "ko":
        noun_frequencies_df = noun_frequencies_df[~noun_frequencies_df['noun'].str.contains(r'[^\uac00-\ud7a3]', na=False)]

    noun_frequencies_df = noun_frequencies_df[noun_frequencies_df["frequency"] >= 4]
    noun_frequencies_df = noun_frequencies_df.drop_duplicates(subset="noun").reset_index(drop=True)

    selected_sentences = {}
    for word, texts, frequency in tqdm(noun_frequencies_df[["noun", "sentences", "frequency"]].values):
        # Filter sentences with word length between 10 and 20
        valid_sentences = [text for text in texts if 10 <= len(text.split()) <= 20]
        
        if valid_sentences:
            # If there are valid sentences, choose the first one
            selected_sentence = valid_sentences[0]
        else:
            # If no valid sentences, choose the one closest to the range
            selected_sentence = min(texts, key=lambda text: abs(len(text.split()) - 15))  # Closest to the midpoint (15)

        # Save the selected sentence, its word length, and the original frequency
        selected_sentences[word] = {
            "selected_sentence": selected_sentence,
            "sentence_length": len(selected_sentence.split()),
            "original_frequency": frequency
        }

    # Convert the selected sentences into a DataFrame
    selected_sentences_df = pd.DataFrame(
        [
            {
                "word": word,
                "selected_sentence": data["selected_sentence"],
                "sentence_length": data["sentence_length"],
                "original_frequency": data["original_frequency"]
            }
            for word, data in selected_sentences.items()
        ]
    )
    
    if save_path:
        selected_sentences_df.to_csv(save_path, index=False)
    
    return selected_sentences_df


from transformers import AutoTokenizer
import json

def tokenize_text(path, tokenizer_name, language, save_path):
    with open("/home/hyujang/multilingual-inner-lexicon/RQ1/config.json", "r") as f:
            config = json.load(f)
    token_key = config["tokenizers"][tokenizer_name]
    if token_key:
        with open("/home/hyujang/multilingual-inner-lexicon/user_config.json", "r") as f:
            user_config = json.load(f)
            token_value = user_config["huggingface_token"].get(token_key)
    else:
        token_value = None

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, tokens=token_value)
    
    df = pd.read_csv(path)
    df.dropna(subset=["word"], inplace=True)
    tokens_list = []
    for index, row in df.iterrows():
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

    Returns:
    - pd.DataFrame: Sampled real words.
    """
    tokens_df['freq_quantile'], bins = pd.qcut(tokens_df['original_frequency'], num_quantiles, labels=False,  duplicates='drop', retbins=True)
    sampled = []
    for quantile in range(num_quantiles):
        quantile_df = tokens_df[(tokens_df['token_num'] == token_num) & (tokens_df['freq_quantile'] == quantile)]
        if len(quantile_df) > 0:
            sampled.append(quantile_df.sample(min(len(quantile_df), num_samples // num_quantiles), 
                                            replace=False, random_state=seed))
    
    sampled_df = pd.concat(sampled, ignore_index=False).drop_duplicates(subset=['word'])
    sampled_indices = sampled_df.index.to_list()

    # Handle cases where the sampled DataFrame has fewer rows than required
    if len(sampled_df) < num_samples:
        print(f"remaining before additional sampling for {token_num}-token words:", num_samples - len(sampled_df))
        remaining = num_samples - len(sampled_df)
        other_df = tokens_df[tokens_df['token_num'] == token_num].drop(sampled_df.index, errors='ignore')
        additional_samples = other_df.sample(min(len(other_df), remaining), replace=False, random_state=seed)
        sampled_indices += additional_samples.index.to_list()
        sampled_df = pd.concat([sampled_df, additional_samples]).drop_duplicates(subset=['word']).reset_index(drop=True)
    
    remaining = num_samples - len(sampled_df)
    print(f"{remaining} remaining after sampling {len(sampled_df)} {token_num}-token words.")
    return sampled_df, remaining, sampled_indices

if __name__ == "__main__":
    # LANGUAGE = "English"
    # LANGUAGE = "Korean"
    LANGUAGE = "German"
    # language_code = LANGUAGE_MAP.get(LANGUAGE)  
    # selected_sentences_df = process_wikipedia_nouns(language_code,
    #                                                 save_path=f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/ComponentAnalysis/{LANGUAGE}_wiki_noun_frequencies_context.csv")
    
    # MODEL_NAME = "Tower-Babel/Babel-9B-Chat"
    MODEL_NAME = "google/gemma-3-12b-it"
    # MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
    df = tokenize_text(f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/ComponentAnalysis/{LANGUAGE}_wiki_noun_frequencies_context.csv",
                  tokenizer_name=MODEL_NAME,
                  language=LANGUAGE,
                  save_path=f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/ComponentAnalysis/{MODEL_NAME.split("/")[-1]}_{LANGUAGE}_wiki_noun_frequencies_context.csv",
                #   save_path=None
                  )
    
    sampled_df, remaining, sampled_indices = sample_real_words(df, token_num=2, num_samples=2440, num_quantiles=5, seed=2025)
    sampled_df.to_csv(f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/ComponentAnalysis/{MODEL_NAME.split('/')[-1]}_{LANGUAGE}_wiki_noun_frequencies_context_2token.csv", index=False)
    