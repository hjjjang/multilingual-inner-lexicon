#!/bin/bash

# List of models
models=(
  # "Tower-Babel/Babel-9B-Chat"
  # "google/gemma-3-12b-it"
  "meta-llama/Llama-2-7b-chat-hf"
)

# List of languages
# langs=("en" "ko" "de")
langs=("de")

# List of tasks
tasks=(
  # "biology"
  # "business"
  # "chemistry"
  # "computer_science"
  # "economics"
  # "engineering"
  # "health"
  "history"
  "law"
  "math"
  "other"
  "philosophy"
  "physics"
  "psychology"
)

# Loop over models, languages, and tasks
for model in "${models[@]}"; do
  # extract a safe name for the output dir (remove slashes)
  model_name=$(echo "$model" | sed 's/\//__/g')

  for lang in "${langs[@]}"; do
    for task in "${tasks[@]}"; do
      task_name="mmlu_prox_${lang}_${task}"

      echo "Running $model_name on $task_name..."

      lm_eval \
        --model vllm \
        --model_args pretrained=$model \
        --tasks $task_name \
        --batch_size auto \
        --log_samples \
        --output_path /home/hyujang/lm-evaluation-harness/output
    done
  done
done
