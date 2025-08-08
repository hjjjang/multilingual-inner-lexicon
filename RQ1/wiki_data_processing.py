from datasets import load_dataset
import spacy
from konlpy.tag import Okt
from collections import Counter
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import os
from nltk.stem import WordNetLemmatizer
import stanza
from kiwipiepy import Kiwi 

def load_wikipedia_data(lang, sample_size=20000):
    wiki = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", columns=["text"])
    return wiki.shuffle(seed=2025).select(range(sample_size)).to_pandas()

def extract_nouns_with_frequency(text, lang, lemmatizer=None, nlp=None, kiwi=None):
    if lang == "en":
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        nouns = [
            lemmatizer.lemmatize(word.lower()) 
            for word, tag in tagged 
            if tag in ['NN', 'NNS'] # NN: Noun, singular or mass / NNS: Noun, plural
        ]
    elif lang == "de":
        doc = nlp(text)
        nouns = [
            word.lemma  # Use lemma instead of the original word
            for sentence in doc.sentences
            for word in sentence.words 
            if word.upos == "NOUN"  
        ]
    elif lang == "ko":
        doc = kiwi.tokenize(text)
        nouns = [
            token.form 
            for token in doc 
            if token.tag == "NNG" # Common noun
        ]
    else:
        raise ValueError("Unsupported language")
    return Counter(nouns)

def process_wikipedia_nouns(lang):
    global df
    
    output_file = f"./data/_{lang}_wiki_noun_frequencies_lemmatized.csv"
    if os.path.exists(output_file):
        print(f"Output file '{output_file}' already exists. Skipping processing.")
        return  # Stop the function if the file exists

    print(f"Loading Wikipedia data for language: {lang}")
    df = load_wikipedia_data(lang)

    if lang == "en":
        print("Extracting nouns using nltk for English...")
        lemmatizer = WordNetLemmatizer()
        tqdm.pandas(desc="Processing text")
        df["noun_frequencies"] = df["text"].progress_apply(lambda text: extract_nouns_with_frequency(text, lang, lemmatizer=lemmatizer))

    elif lang == "de":
        print(f"Extracting nouns using Stanza model for {lang}...")
        nlp = stanza.Pipeline(lang="de", processors="tokenize,pos,lemma", use_gpu=True) 
        tqdm.pandas(desc="Processing text")
        df["noun_frequencies"] = df["text"].progress_apply(lambda text: extract_nouns_with_frequency(text, lang, nlp=nlp))

    elif lang == "ko":
        print(f"Extracting nouns using Kiwi tokenizer for {lang}...")
        kiwi = Kiwi()
        tqdm.pandas(desc="Processing text")
        df["noun_frequencies"] = df["text"].progress_apply(lambda text: extract_nouns_with_frequency(text, lang, kiwi=kiwi))
    
    print("Summing noun frequencies...")
    
    combined_noun_frequencies = sum((Counter(freq_dict) for freq_dict in tqdm(df["noun_frequencies"], desc="Summing")), Counter())
    noun_frequencies_df = pd.DataFrame.from_dict(combined_noun_frequencies, orient="index", columns=["frequency"])
    noun_frequencies_df.sort_values(by="frequency", ascending=False, inplace=True)
    noun_frequencies_df.reset_index(inplace=True)
    noun_frequencies_df.columns = ["word", "freq"]
    
    if lang == "en":
        noun_frequencies_df = noun_frequencies_df[~noun_frequencies_df['word'].str.contains(r'[^a-zA-Z]', na=False)]
    if lang == "de":
        noun_frequencies_df = noun_frequencies_df[~noun_frequencies_df['word'].str.contains(r'[^a-zA-ZäöüÄÖÜß]', na=False)]
    if lang == "ko":
        noun_frequencies_df = noun_frequencies_df[~noun_frequencies_df['word'].str.contains(r'[^\uac00-\ud7a3]', na=False)]
    
    noun_frequencies_df.drop_duplicates(subset=['word'], inplace=True)
    noun_frequencies_df.dropna(inplace=True)
    noun_frequencies_df.reset_index(inplace=True, drop=True)

    noun_frequencies_df.to_csv(output_file, index=False)
    print(f"Saved noun frequencies to {output_file}")

if __name__ == "__main__":
    for lang in ["en", "de", "ko"]:
        process_wikipedia_nouns(lang)
