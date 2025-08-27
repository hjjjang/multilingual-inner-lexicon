import pandas as pd
import numpy as np
from data import WordNonwordData


def compare_tokenizers(lang, tokenizers):
    """
    Compare multiple tokenizers by tokenizing the same dataset and merging results.
    
    Args:
        lang (str): Language to process (e.g., "Korean", "English", "German")
        tokenizers (list): List of tuples containing (tokenizer_name, model_alias)
    """
    # Initialize an empty list to store tokenized DataFrames
    tokenized_dfs = []
    
    for tokenizer_name, model_alias in tokenizers:
        word_nonword_cls = WordNonwordData(lang, tokenizer_name)
        tokens_df = word_nonword_cls.tokenize_and_save()
        
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
    final_df["avg_token_num_rounded"] = np.floor(final_df["avg_token_num"] + 0.5).astype(int)  # round half up
    
    token_num_main_columns = ["token_num_babel_9b", "token_num_gemma_12b"]
    final_df["avg_token_num2"] = final_df[token_num_main_columns].mean(axis=1)
    final_df['same_token_num2'] = final_df[token_num_main_columns].nunique(axis=1) == 1
    final_df["avg_token_num2_rounded"] = np.floor(final_df["avg_token_num2"] + 0.5).astype(int)  # round half up
    
    final_df["any_token_num_is_1"] = (final_df[token_num_columns] == 1).any(axis=1)
    
    # Move the 'freq' column to the last position
    if "freq" in final_df.columns:
        freq_column = final_df.pop("freq")
        final_df["freq"] = freq_column
    
    # Sort the DataFrame by 'freq' in descending order
    if "freq" in final_df.columns:
        final_df = final_df.sort_values(by="freq", ascending=False)
    
    # Save the final DataFrame to a CSV file
    final_df.to_csv(f"data/_{lang}_tokenizers_comparison.csv", index=False)
    
    print(f"Processing complete for {lang}.")
    return final_df


if __name__ == "__main__":
    tokenizers = [
        ("Tower-Babel/Babel-9B-Chat", "babel_9b"),
        ("google/gemma-3-12b-it", "gemma_12b"),
        ("meta-llama/Llama-2-7b-chat-hf", "llama_2_7b"),
    ]
    
    # Example usage
    compare_tokenizers("Korean", tokenizers)
    compare_tokenizers("English", tokenizers)
    compare_tokenizers("German", tokenizers)