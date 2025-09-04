#!/bin/bash

# List of models
models=(
  "Tower-Babel/Babel-9B-Chat"
  "google/gemma-3-12b-it"
  "meta-llama/Llama-2-7b-chat-hf"
)

# Loop over models, languages, and tasks
for model in "${models[@]}"; do
  # extract a safe name for the output dir (remove slashes)
  model_name=$(echo "$model" | sed 's/\//__/g')

  task_name="hrm8k_gsm8k"
  # task_name="hrm8k_gsm8k_en"

  echo "Running $model_name on $task_name..."

  lm_eval \
    --model vllm \
    --model_args "pretrained=${model}" \
    --tasks "${task_name}" \
    --num_fewshot 5 \
    --batch_size auto \
    --apply_chat_template True \
    --gen_kwargs "do_sample=true,temperature=0.7,top_p=0.95,min_tokens=8,max_gen_toks=2048" \
    --log_samples \
    --output_path "/home/hyujang/multilingual-inner-lexicon/output/model_performance/gsm8k" \
    --show_config
done
