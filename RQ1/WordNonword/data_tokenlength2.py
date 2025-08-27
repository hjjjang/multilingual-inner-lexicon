import pandas as pd
import os

# Import utilities
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the base class
from data import WordNonwordData


class WordNonwordData2Token(WordNonwordData):
    """
    Specialized version of WordNonwordData focused on 2-token words only.
    Inherits all functionality from the base class and overrides specific methods.
    """
    
    def __init__(self, language: str, tokenizer_name: str, base_dir: str = "/home/hyujang/multilingual-inner-lexicon"):
        # Call parent constructor to initialize everything
        super().__init__(language, tokenizer_name, base_dir)
        
        # Override sample sizes for 2-token focus
        self.real_words_2_tokens = 3000
        self.non_words_2_tokens = 3000

    def generate_real_and_non_words(self, tokens_df: pd.DataFrame) -> pd.DataFrame:
        """
        Override to generate only 2-token words instead of mixed token lengths.
        """
        # Process frequency data if available (same as parent)
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
        
        # Handle string token representations (same as parent)
        if isinstance(tokens_df['tokens'].iloc[0], str):
            import ast
            tokens_df['tokens'] = tokens_df['tokens'].apply(ast.literal_eval)
        
        # Handle special case for Korean + Llama (same as parent)
        if self.language == "Korean" and "llama" in self.tokenizer_name.lower():
            tokens_df['tokens'] = tokens_df['tokens'].apply(lambda x: x[1:] if x else x)
            tokens_df['token_num'] = tokens_df['tokens'].apply(len)
        
        print("Token distribution:")
        print(tokens_df['token_num'].value_counts().sort_index())

        ########## GENERATE REAL WORDS (2-TOKEN FOCUS ONLY) ##########
        real_words_2_df, remaining_2, sampled_indices_2 = self.sample_real_words(
            tokens_df, 2, self.real_words_2_tokens
        )
        
        real_words_df = real_words_2_df
        print(f"Total real words sampled: {len(real_words_df)}")

        ########## GENERATE NON-WORDS (2-TOKEN FOCUS ONLY) ##########
        non_words_df = self.sample_non_words(tokens_df, 2, self.non_words_2_tokens)
        
        # Add labels and combine
        real_words_df['label'] = "realword"
        non_words_df['label'] = "nonword"
        
        final_df = pd.concat([real_words_df, non_words_df]).reset_index(drop=True)
        return final_df

    def main(self) -> pd.DataFrame:
        """
        Override main method to use 2-token specific file naming.
        """
        print(f"Processing {self.language} with {self.tokenizer_name} (2-token focus)")
        
        input_data = self.load_dataset()
        tokens_df = self.tokenize_and_analyze(input_data)
        final_df = self.generate_real_and_non_words(tokens_df)
        
        # Save with specific naming for 2-token analysis
        output_path = os.path.join(
            self.base_dir, 
             f"data/RQ1/WordNonword/wordnonword_{self.tokenizer_name.split('/')[1]}_{self.language}-2token.csv"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"Dataset saved to: {output_path}")
        
        return final_df

    def main2(self) -> pd.DataFrame:
        """
        Override main2 method for 2-token specific processing of existing tokenized data.
        """
        print(f"Processing existing tokenized data for {self.language} with {self.tokenizer_name} (2-token focus)")
        
        tokens_df = self.load_dataset()
        final_df = self.generate_real_and_non_words(tokens_df)
        
        output_path = os.path.join(
            self.base_dir, 
            f"data/RQ1/WordNonword/wordnonword_{self.tokenizer_name.split('/')[1]}_{self.language}-2token.csv"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"Dataset saved to: {output_path}")
        
        return final_df


def run_experiments():
    """Run data generation experiments for 2-token focused analysis."""
    
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
            print(f"Processing: {lang} with {tokenizer_name} (2-token focus)")
            print(f"{'='*60}")
            
            try:
                # Use the specialized 2-token class
                word_nonword_cls = WordNonwordData2Token(lang, tokenizer_name)
                result_df = word_nonword_cls.main()
                print(f"Successfully processed {lang} with {tokenizer_name}")
                print(f"Generated {len(result_df)} samples")
                
            except Exception as e:
                print(f"Error processing {lang} with {tokenizer_name}: {e}")
                continue


if __name__ == "__main__":
    run_experiments()