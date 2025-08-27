from collections import defaultdict
# from .word_retriever import PatchscopesRetriever
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from RQ1.WordNonword.classification import WordNonwordClassifier
import torch
import pandas as pd
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import gc
from transformers.utils import logging
logging.set_verbosity_error()

class PatchScope(WordNonwordClassifier):
    def __init__(self, 
                 language, 
                 tokenizer_name,
                 source_language=None,
                 target_language=None,
                representation_prompt: str = "{word}",
                patchscopes_prompt: str = "Next is the same word twice: 1) {word} 2)",
                prompt_target_placeholder: str = "{word}",
                num_tokens_to_generate: int = 20,
                output_type="layer_hidden_states"
                 ):
        super().__init__(language, tokenizer_name)  # Inherit token config
        self.setup_tokenizer()
        
        self.embedding_matrix = self.model.get_input_embeddings().weight
        self.model_name = tokenizer_name.split("/")[-1]
        
        print(f"Setting up PatchScope for {source_language} to {target_language} translation for {self.model_name}.")
        
        if self.language=="English":
            if source_language == "English":
                patchscopes_prompt = f"This English word '{{word}}' in {target_language} is:"
            elif target_language == "English":
                patchscopes_prompt = f"This {source_language} word '{{word}}' in English is:" 
            else:
                print("Warning: wrong language configuration for PatchScope. ")
        
        elif self.language=="Korean":
            korean_map = {
                "English": "영어",
                "German": "독일어",
                "Korean": "한국어"
            }
            if source_language == "Korean":
                patchscopes_prompt = f"이 한국어 단어 '{{word}}'는 {korean_map[target_language]}로:"
            elif target_language == "Korean":
                patchscopes_prompt = f"이 {korean_map[source_language]} 단어 '{{word}}'는 한국어로:"
                
        elif self.language=="German":
            german_map = {
                "English": "Englisch",
                "Korean": "Koreanisch",
                "German": "Deutsch"
            }

            german_adj_map = {
                "English": "englische",
                "Korean": "koreanische",
                "German": "deutsche"
            }

            if source_language == "German":
                patchscopes_prompt = f"Dieses deutsche Wort '{{word}}' auf {german_map[target_language]} ist:"
            elif target_language == "German":
                patchscopes_prompt = f"Dieses {german_adj_map[source_language]} Wort '{{word}}' auf Deutsch ist:"
        
        print(f"Patchscopes prompt: {patchscopes_prompt}")
        self.prompt_input_ids, self.prompt_target_idx = self._build_prompt_input_ids_template(patchscopes_prompt, prompt_target_placeholder)
        self._prepare_representation_prompt = self._build_representation_prompt_func(representation_prompt, prompt_target_placeholder)
        self.num_tokens_to_generate = num_tokens_to_generate
        self.output_type = output_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def setup_tokenizer(self):
        if self.tokenizer_name == "Tower-Babel/Babel-9B-Chat":
            self.tokenizer.add_special_tokens({'unk_token': 'UNK'})
            self.tokenizer.unk_token_id = self.tokenizer.convert_tokens_to_ids('UNK')
            
    def _build_prompt_input_ids_template(self, prompt, target_placeholder):
        prompt_input_ids = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
        target_idx = []

        if prompt:
            assert target_placeholder is not None, \
                "Trying to set a prompt for Patchscopes without defining the prompt's target placeholder string, e.g., [MASK]"

            prompt_parts = prompt.split(target_placeholder)
            for part_i, prompt_part in enumerate(prompt_parts):
                prompt_input_ids += self.tokenizer.encode(prompt_part, add_special_tokens=False)
                if part_i < len(prompt_parts)-1:
                    target_idx += [len(prompt_input_ids)]
                    prompt_input_ids += [0]
        else:
            prompt_input_ids += [0]
            target_idx = [len(prompt_input_ids)]

        prompt_input_ids = torch.tensor(prompt_input_ids, dtype=torch.long)
        target_idx = torch.tensor(target_idx, dtype=torch.long)
        return prompt_input_ids, target_idx

    def _build_representation_prompt_func(self, prompt, target_placeholder):
        return lambda word: prompt.replace(target_placeholder, word)


    def extract_token_i_hidden_states_original(self,
            inputs,
            # token_idx_to_extract: int = -1,
            layers_to_extract=None,
            return_dict: bool = False,
            verbose: bool = True) -> torch.Tensor:
        self.model.eval()

        if isinstance(inputs, str):
            inputs = [inputs]

        if layers_to_extract is None:
            if "gemma-3" in self.tokenizer_name:
                layers_to_extract = list(range(1, self.model.config.text_config.num_hidden_layers + 1))  # Exclude embedding layer
            else:
                layers_to_extract = list(range(1, self.model.config.num_hidden_layers + 1))  # extract all but initial embeddings
        if return_dict:
            layers_to_extract = [0] + layers_to_extract

        all_hidden_states = {layer: [] for layer in layers_to_extract}

        with torch.no_grad():
            for input_text in tqdm(inputs, desc="Extracting hidden states", disable=not verbose):
                # Clear cache between iterations
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

                encoding = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    return_attention_mask=False,
                    add_special_tokens=True
                )
                input_ids = encoding["input_ids"].to(self.device)

                if self.output_type == "layer_hidden_states":
                    outputs = self.model(input_ids, output_hidden_states=True)
                    hidden_states_seq = outputs.hidden_states  # Tuple of [1, seq_len, hidden_dim]

                    for layer in layers_to_extract:
                        token_hidden = hidden_states_seq[layer][0, -1, :].detach().cpu()
                        all_hidden_states[layer].append(token_hidden)

                elif self.output_type == "ffn_hidden_states":
                    if "gemma-3" in self.model_name.lower():
                        ffn_probe = FFNProbeGemma3(self.model)
                    else:
                        ffn_probe = FFNProbe(self.model)

                    _ = ffn_probe(input_ids, output_hidden_states=False)
                    ffn_outputs = ffn_probe.ffn_outputs  # List of [1, seq_len, hidden_dim]
                    ffn_probe.remove_hooks()

                    for layer in layers_to_extract:
                        ffn_hidden = ffn_outputs[layer-1][0, -1, :].detach().cpu() # len(ffn_outputs) -> # of decoder layers (no emb layer included) , so layer-1 is correct
                        all_hidden_states[layer].append(ffn_hidden)

                # Clear intermediate variables
                del input_ids
                if 'outputs' in locals():
                    del outputs
                if 'hidden_states_seq' in locals():
                    del hidden_states_seq

        for layer in all_hidden_states:
            all_hidden_states[layer] = torch.concat(all_hidden_states[layer], dim=0)

        if not return_dict:
            # all_hidden_states = torch.concat([all_hidden_states[layer] for layer in layers_to_extract], dim=0)
            all_hidden_states = torch.stack(list(all_hidden_states.values()), dim=0)
        
        return all_hidden_states

    def extract_hidden_states(self, word):
        representation_input = self._prepare_representation_prompt(word)
        last_token_hidden_states = self.extract_token_i_hidden_states_original(
            inputs=representation_input, 
            return_dict=False, 
            verbose=False)
        return last_token_hidden_states
            
    
    def retrieve_word(self, hidden_states):
        self.model.eval()

        # insert hidden states into patchscopes prompt
        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)

        inputs_embeds = self.model.get_input_embeddings()(self.prompt_input_ids.to(self.model.device)).unsqueeze(0)
        batched_patchscope_inputs = inputs_embeds.repeat(len(hidden_states), 1, 1).to(hidden_states.dtype)
        batched_patchscope_inputs[:, self.prompt_target_idx] = hidden_states.unsqueeze(1).to(self.model.device) # patches the hidden state into a specific location (prompt_target_idx) in the prompt by replacing target token embedding with the hidden state

        attention_mask = (self.prompt_input_ids != self.tokenizer.eos_token_id).long().unsqueeze(0).repeat(
            len(hidden_states), 1).to(self.model.device)

        with torch.no_grad():
            patchscope_outputs = self.model.generate(
                do_sample=False, # greedy, deterministic decoding
                num_beams=1, # greedy decoding
                top_p=1.0, # consider all tokens (ignored when do_sample=False)
                temperature=None, # None -> default 1.0 (ignored when do_sample=False) 
                inputs_embeds=batched_patchscope_inputs, # Instead of using token IDs as input, this provides pre-computed token embeddings. Required when manipulating internal representations (e.g., inserting hidden states into prompts). 
                attention_mask=attention_mask,
                max_new_tokens=self.num_tokens_to_generate, 
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=None,
                )

        decoded_patchscope_outputs = self.tokenizer.batch_decode(patchscope_outputs)
        return decoded_patchscope_outputs
    
    def get_hidden_states_and_retrieve_word(self, word):
        last_token_hidden_states = self.extract_hidden_states(word)
        patchscopes_description_by_layers = self.retrieve_word(
            last_token_hidden_states)

        return patchscopes_description_by_layers, last_token_hidden_states
    
    def run_patchscopes_on_list(self,
            words_list,
            output_csv_path=None,
    ):
        outputs = defaultdict(dict)
        rows = []
        for word in tqdm(words_list, total=len(words_list), desc="Running patchscopes..."):
            patchscopes_description_by_layers, _ = self.get_hidden_states_and_retrieve_word(word)
            for layer_i, patchscopes_result in enumerate(patchscopes_description_by_layers):
                outputs[word][layer_i] = patchscopes_result
                rows.append({"word": word, "layer": layer_i, "patchscope_result": patchscopes_result})

        if output_csv_path:
            df = pd.DataFrame(rows)
            df.to_csv(output_csv_path, index=False)

        return outputs


class FFNProbe:
    def __init__(self, model):
        self.model = model
        self.ffn_outputs = []
        self.hook_handles = []

        def hook_fn(module, input, output):
            self.ffn_outputs.append(output)

        # Register hooks to the FFN layers of each transformer block
        for block in self.model.model.layers:  # adjust depending on model architecture
            handle = block.mlp.register_forward_hook(hook_fn)
            self.hook_handles.append(handle)
            # block.mlp.register_forward_hook(hook_fn)  # For LLaMA, GPT-J, Falcon, etc.
            
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []


    def clear(self):
        self.ffn_outputs = []

    def __call__(self, *args, **kwargs):
        self.clear()
        return self.model(*args, **kwargs)


class FFNProbeGemma3:
    def __init__(self, model):
        self.model = model
        self.ffn_outputs = []
        self.hook_handles = []

        def hook_fn(module, input, output):
            self.ffn_outputs.append(output)

        for block in self.model.model.language_model.layers:
            # block.mlp.register_forward_hook(hook_fn)
            handle = block.mlp.register_forward_hook(hook_fn)
            self.hook_handles.append(handle)
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
            
    def clear(self):
        self.ffn_outputs = []

    def __call__(self, *args, **kwargs):
        self.clear()
        return self.model(*args, **kwargs)


def run_patchscope_job(model_name, source_lang, target_lang, prompt_lang):
    model_basename = model_name.split("/")[-1]
    
    csv_path = f"/home/hyujang/multilingual-inner-lexicon/data/RQ2/MUSE/{source_lang}_{target_lang}_1000.csv"
    words_list = pd.read_csv(csv_path)[source_lang].tolist()
    
    patchscope = PatchScope(prompt_lang, model_name, source_lang, target_lang)
    
    output_path = f"/home/hyujang/multilingual-inner-lexicon/output/RQ2/PatchScope/num_token_20/{model_basename}_{source_lang}_to_{target_lang}_{prompt_lang}Prompt_withOriginalCode.csv"
    patchscope.run_patchscopes_on_list(words_list=words_list, output_csv_path=output_path)


if __name__ == "__main__":
    configs = [
        # Format: (MODEL_NAME, SOURCE_LANGUAGE, TARGET_LANGUAGE, PROMPT_LANGUAGE)
        ("Tower-Babel/Babel-9B-Chat", "English", "German", "German"),
        ("google/gemma-3-12b-it", "English", "German", "German"),
        ("meta-llama/Llama-2-7b-chat-hf", "English", "German", "German"),

        ("Tower-Babel/Babel-9B-Chat", "German", "English", "German"),
        ("google/gemma-3-12b-it", "German", "English", "German"),
        ("meta-llama/Llama-2-7b-chat-hf", "German", "English", "German"),

        ("Tower-Babel/Babel-9B-Chat", "German", "English", "English"),
        ("google/gemma-3-12b-it", "German", "English", "English"),
        ("meta-llama/Llama-2-7b-chat-hf", "German", "English", "English"),
    ]

    for model_name, src, tgt, prompt in configs:
        print(f"\n▶ Running: {model_name} | {src} → {tgt} | Prompt: {prompt}")
        run_patchscope_job(model_name, src, tgt, prompt)
