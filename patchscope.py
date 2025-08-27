from collections import defaultdict
from tqdm import tqdm
from utils import clean_memory, get_device, setup_tokenizer, setup_model, extract_token_hidden_states_2, ensure_output_dir
import torch
import pandas as pd

class PatchScope:
    def __init__(self, 
                 language, 
                 tokenizer_name,
                 patchscopes_prompt: str = None,
                 prompt_target_placeholder: str = "{word}",
                 num_tokens_to_generate: int = 10,
                 output_type="layer_hidden_states",
                 source_language=None,
                 target_language=None):
        self.language = language
        self.tokenizer_name = tokenizer_name
        self.device = get_device()
        self.tokenizer = setup_tokenizer(tokenizer_name)
        self.model = setup_model(tokenizer_name, self.device)

        self.embedding_matrix = self.model.get_input_embeddings().weight
        self.model_name = tokenizer_name.split("/")[-1]

        # Set default base prompts for RQ1 use case
        if patchscopes_prompt is None:
            if self.language == "English":
                patchscopes_prompt = "Next is the same word twice: 1) {word} 2)"
            elif self.language == "Korean":
                patchscopes_prompt = "같은 단어가 두 번 나옵니다: 1) {word} 2)"
            elif self.language == "German":
                patchscopes_prompt = "Dasselbe Wort erscheint zweimal: 1) {word} 2)"

        # Handle translation-specific prompts for RQ2 use case
        if source_language and target_language:
            if self.language == "English":
                if source_language == "English":
                    patchscopes_prompt = f"This English word '{{word}}' in {target_language} is:"
                elif target_language == "English":
                    patchscopes_prompt = f"This {source_language} word '{{word}}' in English is:"
            elif self.language == "Korean":
                korean_map = {"English": "영어", "German": "독일어", "Korean": "한국어"}
                if source_language == "Korean":
                    patchscopes_prompt = f"이 한국어 단어 '{{word}}'는 {korean_map[target_language]}로:"
                elif target_language == "Korean":
                    patchscopes_prompt = f"이 {korean_map[source_language]} 단어 '{{word}}'는 한국어로:"
            elif self.language == "German":
                german_map = {"English": "Englisch", "Korean": "Koreanisch", "German": "Deutsch"}
                if source_language == "German":
                    patchscopes_prompt = f"Dieses deutsche Wort '{{word}}' auf {german_map[target_language]} ist:"
                elif target_language == "German":
                    patchscopes_prompt = f"Dieses {source_language} Wort '{{word}}' auf Deutsch ist:"

        self.prompt_input_ids, self.prompt_target_idx = self._build_prompt_input_ids_template(
            patchscopes_prompt, prompt_target_placeholder
        )
        # self._prepare_representation_prompt = self._build_representation_prompt_func(
        #     patchscopes_prompt, prompt_target_placeholder
        # )
        
        self.num_tokens_to_generate = num_tokens_to_generate
        self.output_type = output_type
        
        # Print intermediate configurations
        print(f"Language: {self.language}")
        print(f"Tokenizer Name: {self.tokenizer_name}")
        print(f"PatchScopes Prompt: {patchscopes_prompt}")
        print(f"Output Type: {self.output_type}")

    def _build_prompt_input_ids_template(self, prompt, target_placeholder):
        prompt_input_ids = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
        target_idx = []

        if prompt:
            assert target_placeholder is not None, \
                "Prompt target placeholder must be defined."

            prompt_parts = prompt.split(target_placeholder)
            for part_i, prompt_part in enumerate(prompt_parts):
                prompt_input_ids += self.tokenizer.encode(prompt_part, add_special_tokens=False)
                if part_i < len(prompt_parts) - 1:
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

    def retrieve_word(self, hidden_states):
        self.model.eval()

        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)

        inputs_embeds = self.model.get_input_embeddings()(self.prompt_input_ids.to(self.model.device)).unsqueeze(0)
        batched_patchscope_inputs = inputs_embeds.repeat(len(hidden_states), 1, 1).to(hidden_states.dtype)
        batched_patchscope_inputs[:, self.prompt_target_idx] = hidden_states.unsqueeze(1).to(self.model.device)

        attention_mask = (self.prompt_input_ids != self.tokenizer.eos_token_id).long().unsqueeze(0).repeat(
            len(hidden_states), 1).to(self.model.device)

        with torch.no_grad():
            patchscope_outputs = self.model.generate(
                do_sample=False, # greedy decoding
                num_beams=1,
                top_p=1.0,
                temperature=None,
                inputs_embeds=batched_patchscope_inputs,
                attention_mask=attention_mask,
                max_new_tokens=self.num_tokens_to_generate,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.batch_decode(patchscope_outputs)

    def run_patchscopes_on_list(self, words_list, output_csv_path=None):
        outputs = defaultdict(dict)
        rows = []

        # Prepare representation inputs
        # rep_inputs = [self._prepare_representation_prompt(w) for w in words_list]

        # Extract hidden states for all words
        last_token_hidden_states_all = extract_token_hidden_states_2(
            model=self.model,
            tokenizer=self.tokenizer,
            inputs=words_list,
            tokenizer_name=self.tokenizer_name,
            device=self.device,
            output_type=self.output_type,
            return_dict=False
        ) # [layers, words, hidden_dim]

        # Retrieve words and organize results
        for idx, word in enumerate(tqdm(words_list, desc="Running PatchScopes")):
            hidden_states = last_token_hidden_states_all[:, idx, :]
            patchscopes_description_by_layers = self.retrieve_word(hidden_states)

            for layer_i, patchscopes_result in enumerate(patchscopes_description_by_layers):
                outputs[word][layer_i] = patchscopes_result
                rows.append({"word": word, "layer": layer_i, "patchscope_result": patchscopes_result})

        # Save results to CSV if output path is provided
        if output_csv_path:
            ensure_output_dir(output_csv_path)
            pd.DataFrame(rows).to_csv(output_csv_path, index=False)

        return outputs
