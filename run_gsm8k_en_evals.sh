#!/bin/bash

# List of models
models=(
  "Tower-Babel/Babel-9B-Chat"
  "google/gemma-3-12b-it"
  "meta-llama/Llama-2-7b-chat-hf"
)

# List of languages
# langs=("" "en")

# Loop over models, languages, and tasks
for model in "${models[@]}"; do
  # extract a safe name for the output dir (remove slashes)
  model_name=$(echo "$model" | sed 's/\//__/g')

  task_name="hrm8k_gsm8k_en"

  echo "Running $model_name on $task_name..."

  lm_eval \
    --model vllm \
    --model_args pretrained=$model \
    --tasks $task_name \
    --num_fewshot 5 \
    --batch_size auto \
    --log_samples \
    --output_path /home/hyujang/multilingual-inner-lexicon/output/model_performance/gsm8k \
    --show_config

done
