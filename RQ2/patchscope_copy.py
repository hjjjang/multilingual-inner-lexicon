import torch
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import os
import gc
from transformers.utils import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma3ForConditionalGeneration
import json

logging.set_verbosity_error()

class PatchScope:
    def __init__(self, 
                 source_language: str,
                 target_language: str, 
                 tokenizer_name: str,
                 translation_prompt: str,
                 prompt_target_placeholder: str = "[word]",
                 num_tokens_to_generate: int = 20):
        
        self.source_language = source_language
        self.target_language = target_language
        self.tokenizer_name = tokenizer_name
        self.translation_prompt = translation_prompt
        self.prompt_target_placeholder = prompt_target_placeholder
        self.num_tokens_to_generate = num_tokens_to_generate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and tokenizer
        self._set_tokenizer_and_model(self.tokenizer_name)
        self.model.eval()

        self.prompt_input_ids, self.prompt_target_idx = self._build_prompt_input_ids_template(
            self.translation_prompt, prompt_target_placeholder
        )

        
    def _set_tokenizer_and_model(self, tokenizer_name):
        
        with open("/home/hyujang/multilingual-inner-lexicon/RQ1/config.json", "r") as f:
            config = json.load(f)
        token_key = config["tokenizers"][tokenizer_name]
        if token_key:
            with open("/home/hyujang/multilingual-inner-lexicon/user_config.json", "r") as f:
                user_config = json.load(f)
                token_value = user_config["huggingface_token"].get(token_key)
        else:
            token_value = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, token=token_value)
        
        if self.tokenizer_name == "google/gemma-3-12b-it":
            self.model = Gemma3ForConditionalGeneration.from_pretrained(tokenizer_name, token=token_value, torch_dtype=torch.bfloat16)
            self.model.to(self.device)
        elif self.tokenizer_name == "meta-llama/Llama-2-7b-chat-hf":
            self.model = AutoModelForCausalLM.from_pretrained(tokenizer_name, token=token_value, torch_dtype=torch.float16)
            self.model.to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(tokenizer_name, token=token_value, torch_dtype=torch.bfloat16)
            self.model.to(self.device)


    def _build_prompt_input_ids_template(self, prompt, target_placeholder):
        prompt_input_ids = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
        target_idx = []

        if prompt:
            assert target_placeholder is not None, \
                "Prompt target placeholder must be defined, e.g., {word}"

            prompt_parts = prompt.split(target_placeholder)
            for part_i, prompt_part in enumerate(prompt_parts): # 2 parts: before and after the target word
                prompt_input_ids += self.tokenizer.encode(prompt_part, add_special_tokens=False)
                if part_i < len(prompt_parts) - 1:
                    target_idx.append(len(prompt_input_ids))
                    prompt_input_ids.append(0)  # Placeholder for the word
        else:
            prompt_input_ids.append(0)
            target_idx = [len(prompt_input_ids)]

        prompt_input_ids = torch.tensor(prompt_input_ids, dtype=torch.long)
        target_idx = torch.tensor(target_idx, dtype=torch.long)
        return prompt_input_ids, target_idx

    def _prepare_representation_prompt(self, word):
        return self.translation_prompt.replace(self.prompt_target_placeholder, word)

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

                # if self.output_type == "layer_hidden_states":
                outputs = self.model(input_ids, output_hidden_states=True)
                hidden_states_seq = outputs.hidden_states  # Tuple of [1, seq_len, hidden_dim]

                for layer in layers_to_extract:
                    token_hidden = hidden_states_seq[layer][0, -1, :].detach().cpu()
                    all_hidden_states[layer].append(token_hidden)

                # elif self.output_type == "ffn_hidden_states":
                #     if "gemma-3" in self.model_name.lower():
                #         ffn_probe = FFNProbeGemma3(self.model)
                #     else:
                #         ffn_probe = FFNProbe(self.model)

                #     _ = ffn_probe(input_ids, output_hidden_states=False)
                #     ffn_outputs = ffn_probe.ffn_outputs  # List of [1, seq_len, hidden_dim]
                #     ffn_probe.remove_hooks()

                #     for layer in layers_to_extract:
                #         ffn_hidden = ffn_outputs[layer-1][0, -1, :].detach().cpu() # len(ffn_outputs) -> # of decoder layers (no emb layer included) , so layer-1 is correct
                #         all_hidden_states[layer].append(ffn_hidden)

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

    def retrieve_translation(self, hidden_states, num_tokens_to_generate=None):
        """Retrieve the translation by inserting hidden states into the prompt."""
        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)

        inputs_embeds = self.model.get_input_embeddings()(self.prompt_input_ids.to(self.device)).unsqueeze(0)
        layers_inputs = inputs_embeds.repeat(len(hidden_states), 1, 1).to(hidden_states.dtype)
        layers_inputs[:, self.prompt_target_idx] = hidden_states.unsqueeze(1).to(self.device)

        attention_mask = (self.prompt_input_ids != self.tokenizer.eos_token_id).long().unsqueeze(0).repeat(
            len(hidden_states), 1).to(self.device)

        num_tokens_to_generate = num_tokens_to_generate if num_tokens_to_generate else self.num_tokens_to_generate

        with torch.no_grad():
            outputs = self.model.generate(
                do_sample=False, # greedy, deterministic decoding
                num_beams=1, # greedy decoding
                top_p=1.0, # consider all tokens (ignored when do_sample=False)
                temperature=None, # None -> default 1.0 (ignored when do_sample=False) 
                inputs_embeds=layers_inputs, # Instead of using token IDs as input, this provides pre-computed token embeddings. Required when manipulating internal representations (e.g., inserting hidden states into prompts). 
                attention_mask=attention_mask,
                max_new_tokens=num_tokens_to_generate, 
                pad_token_id=self.tokenizer.eos_token_id,
                )

        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return decoded_outputs

    def run_translation(self, words_list, output_csv_path=None):
        """Run the translation experiment on a list of words."""
        
        outputs = defaultdict(dict)
        rows = []
        for word in tqdm(words_list, desc="Translating words"):
            hidden_states = self.extract_hidden_states(word)
            patchscopes_results_by_layers = self.retrieve_translation(hidden_states)
            for layer_i, patchscopes_result in enumerate(patchscopes_results_by_layers):
                outputs[word][layer_i] = patchscopes_result
                rows.append({"word": word, "layer": layer_i, "patchscope_result": patchscopes_result})
                
        if output_csv_path:
            df = pd.DataFrame(rows)
            df.to_csv(output_csv_path, index=False)

        return outputs


if __name__ == "__main__":
    SOURCE_LANGUAGE = "German"
    TARGET_LANGUAGE = "English"
    # MODEL_NAME = "Tower-Babel/Babel-9B-Chat"
    # MODEL_NAME = "google/gemma-3-12b-it"
    MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Use Llama-2 for better performance

    # Initialize PatchScope
    patchscope = PatchScope(
        source_language=SOURCE_LANGUAGE,
        target_language=TARGET_LANGUAGE,
        tokenizer_name=MODEL_NAME,
        # translation_prompt = f"Translate the {SOURCE_LANGUAGE} word [word] into {TARGET_LANGUAGE}. Only return a json object with the key 'translation' and the value as the translated word.",
        # translation_prompt="Translate the Korean word '[word]' into English word. Respond only with a valid JSON object in the following format: {'translation': '<translated word>'}. Do not include any additional text or explanation.",
        # translation_prompt="Translate the Korean word '[word]' into English word:",
        # translation_prompt="What is this Korean word '[word]' in English? Respond only with the translated English word.", #v1
        # translation_prompt=f"This {SOURCE_LANGUAGE} word '[word]' in {TARGET_LANGUAGE} is:", #v2
        translation_prompt="Dieses Deutsch-Wort '[word]' auf Englisch ist:",
        num_tokens_to_generate = 20
    )

    # Load words list
    df = pd.read_csv(f"/home/hyujang/multilingual-inner-lexicon/data/RQ2/MUSE/{SOURCE_LANGUAGE}_{TARGET_LANGUAGE}_1000.csv")
    # from ast import literal_eval
    # df["en"] = df["en"].apply(literal_eval)
    words_list = df[f"{SOURCE_LANGUAGE}"].tolist()
    
    # Run translation experiment
    output_path = f"/home/hyujang/multilingual-inner-lexicon/output/RQ2/PatchScope/{MODEL_NAME.split("/")[-1]}_{SOURCE_LANGUAGE}_to_{TARGET_LANGUAGE}_GermanPrompt.csv"
    patchscope.run_translation(words_list, output_csv_path=output_path)