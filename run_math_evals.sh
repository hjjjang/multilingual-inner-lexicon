#!/bin/bash

# List of models
models=(
  "Tower-Babel/Babel-9B-Chat"
  "google/gemma-3-12b-it"
  "meta-llama/Llama-2-7b-chat-hf"
)

# List of languages
langs=("en" "de")

# Loop over models, languages, and tasks
for model in "${models[@]}"; do
  # extract a safe name for the output dir (remove slashes)
  model_name=$(echo "$model" | sed 's/\//__/g')

  for lang in "${langs[@]}"; do
      task_name="mgsm_direct_${lang}"

      echo "Running $model_name on $task_name..."

      lm_eval \
        --model vllm \
        --model_args pretrained=$model \
        --tasks $task_name \
        --gen_kwargs temperature=0 \
        --num_fewshot 6 \
        --batch_size auto \
        --log_samples \
        --output_path /home/hyujang/multilingual-inner-lexicon/output/model_performance/mgsm \
        --show_config
  done
done
