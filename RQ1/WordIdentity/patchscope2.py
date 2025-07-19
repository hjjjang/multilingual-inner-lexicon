from collections import defaultdict
# from .word_retriever import PatchscopesRetriever
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from RQ1.WordNonword.classification import WordNonwordClassifier
import torch
import pandas as pd
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from transformers.utils import logging
logging.set_verbosity_error()

class PatchScope(WordNonwordClassifier):
    def __init__(self, 
                 language, 
                 tokenizer_name,
                representation_prompt: str = "{word}",
                patchscopes_prompt: str = "Next is the same word twice: 1) {word} 2)",
                
                prompt_target_placeholder: str = "{word}",
                representation_token_idx_to_extract: int = -1,
                num_tokens_to_generate: int = 10,
                # batch_size: int = 8,
                 ):
        super().__init__(language, tokenizer_name)  # Inherit token config
        self.setup_tokenizer()
        self.model.eval().to(self.device)
        self.embedding_matrix = self.model.get_input_embeddings().weight
        self.model_name = tokenizer_name.split("/")[-1]
        
        if self.language=="English":
            patchscopes_prompt = "Next is the same word twice: 1) {word} 2)"
        elif self.language=="Korean":
            # patchscopes_prompt = "같은 단어가 두 번 나옵니다: 1) {word} 2)" # v2
            patchscopes_prompt = "다음 단어를 반복하세요: 1) {word} 2)" # v3   
        elif self.language=="German":
            # patchscopes_prompt = "Dasselbe Wort erscheint zweimal: 1) {word} 2)"
            patchscopes_prompt = "Wiederholen Sie dasselbe Wort: 1) {word} 2)" # v3
        
        self.prompt_input_ids, self.prompt_target_idx = self._build_prompt_input_ids_template(patchscopes_prompt, prompt_target_placeholder)
        self._prepare_representation_prompt = self._build_representation_prompt_func(representation_prompt, prompt_target_placeholder)
        self.representation_token_idx = representation_token_idx_to_extract
        self.num_tokens_to_generate = num_tokens_to_generate
        
        
            
        
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
            token_idx_to_extract: int = -1,
            batch_size: int = 1,
            layers_to_extract = None,
            return_dict: bool = True,
            verbose: bool = True,
    ) -> torch.Tensor:
        device = self.model.device
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
            for i in tqdm(range(0, len(inputs), batch_size), desc="Extracting hidden states", unit="batch", disable=not verbose):
                input_ids = self.tokenizer(inputs[i:i+batch_size], return_tensors="pt", return_attention_mask=False, add_special_tokens=True)['input_ids']
                outputs = self.model(input_ids.to(device), output_hidden_states=True)
                for input_i in range(len(input_ids)):
                    for layer in layers_to_extract:
                        hidden_states = outputs.hidden_states[layer]
                        all_hidden_states[layer].append(hidden_states[:, token_idx_to_extract, :].detach().cpu())
        for layer in all_hidden_states:
            all_hidden_states[layer] = torch.concat(all_hidden_states[layer], dim=0)

        if not return_dict:
            all_hidden_states = torch.concat([all_hidden_states[layer] for layer in layers_to_extract], dim=0)

        return all_hidden_states

    def extract_hidden_states(self, word):
        representation_input = self._prepare_representation_prompt(word)
        last_token_hidden_states = self.extract_token_i_hidden_states_original(
            inputs=representation_input, 
            token_idx_to_extract=self.representation_token_idx, 
            return_dict=False, 
            verbose=False)
        return last_token_hidden_states

    def retrieve_word(self, hidden_states, layer_idx=None, num_tokens_to_generate=None):
        self.model.eval()

        # insert hidden states into patchscopes prompt
        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)

        inputs_embeds = self.model.get_input_embeddings()(self.prompt_input_ids.to(self.model.device)).unsqueeze(0)
        batched_patchscope_inputs = inputs_embeds.repeat(len(hidden_states), 1, 1).to(hidden_states.dtype)
        batched_patchscope_inputs[:, self.prompt_target_idx] = hidden_states.unsqueeze(1).to(self.model.device) # patches the hidden state into a specific location (prompt_target_idx) in the prompt by replacing target token embedding with the hidden state

        attention_mask = (self.prompt_input_ids != self.tokenizer.eos_token_id).long().unsqueeze(0).repeat(
            len(hidden_states), 1).to(self.model.device)

        num_tokens_to_generate = num_tokens_to_generate if num_tokens_to_generate else self.num_tokens_to_generate

        with torch.no_grad():
            patchscope_outputs = self.model.generate(
                do_sample=False, num_beams=1, top_p=1.0, temperature=None,
                inputs_embeds=batched_patchscope_inputs, attention_mask=attention_mask,
                max_new_tokens=num_tokens_to_generate, pad_token_id=self.tokenizer.eos_token_id, )

        decoded_patchscope_outputs = self.tokenizer.batch_decode(patchscope_outputs)
        return decoded_patchscope_outputs
    
    def get_hidden_states_and_retrieve_word(self, word, num_tokens_to_generate=None):
        last_token_hidden_states = self.extract_hidden_states(word)
        patchscopes_description_by_layers = self.retrieve_word(
            last_token_hidden_states, num_tokens_to_generate=num_tokens_to_generate)

        return patchscopes_description_by_layers, last_token_hidden_states
    
    def run_patchscopes_on_list(self,
            words_list,
            patchscopes_n_new_tokens=5,
            output_csv_path=None,
    ):
        outputs = defaultdict(dict)
        rows = []
        for word in tqdm(words_list, total=len(words_list), desc="Running patchscopes..."):
            patchscopes_description_by_layers, _ = self.get_hidden_states_and_retrieve_word(word, num_tokens_to_generate=patchscopes_n_new_tokens)
            for layer_i, patchscopes_result in enumerate(patchscopes_description_by_layers):
                outputs[word][layer_i] = patchscopes_result
                rows.append({"word": word, "layer": layer_i, "patchscope_result": patchscopes_result})

        if output_csv_path:
            df = pd.DataFrame(rows)
            df.to_csv(output_csv_path, index=False)

        return outputs



if __name__ == "__main__":
    MODEL_NAME = "Tower-Babel/Babel-9B-Chat"
    LANGUAGE = "German"
    patchscope = PatchScope(LANGUAGE, MODEL_NAME)
    MODEL_NAME = MODEL_NAME.split("/")[-1]  # Extract model name from the full path
    words_list = pd.read_csv(f"/home/hyujang/multilingual-inner-lexicon/data/RQ1/WordIdentity/multi_token_{MODEL_NAME}_{LANGUAGE}.csv")['word'].tolist()
    # words_list = words_list[:20]
    patchscope.run_patchscopes_on_list(words_list=words_list,
                                    #    output_csv_path=f"/home/hyujang/multilingual-inner-lexicon/output/RQ1/WordIdentity/multi_token_{MODEL_NAME}_{LANGUAGE}_v3.csv"
                                    )
