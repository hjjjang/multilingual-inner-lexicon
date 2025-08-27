import json
from pathlib import Path
import pandas as pd

ROOT = Path("/home/hyujang/lm-evaluation-harness/output/model_performance/gsm8k")

rows = []

for result_file in ROOT.glob("**/results_*.json"):
    with open(result_file, "r") as f:
        data = json.load(f)

    config = data.get("config", {})
    model_args = config.get("model_args", "")
    model_name = model_args.split("pretrained=")[-1] if "pretrained=" in model_args else model_args

    n_samples_data = data.get("n-samples", {})

    for task, metrics in data.get("results", {}).items():
        task_n_samples = n_samples_data.get(task, {})
        original_samples = task_n_samples.get("original", None)
        effective_samples = task_n_samples.get("effective", None)

        rows.append({
            "Model": model_name,
            "Language": data.get("configs").get(task).get("dataset_name"),
            "Task": data.get("configs").get(task).get("task_alias"),
            "Exact Match": metrics.get("exact_match,custom-extract"),
            "Exact Match StdErr": metrics.get("exact_match_stderr,custom-extract"),
            "N-Samples Original": original_samples,
            "N-Samples Effective": effective_samples,
            # "File": result_file.name
        })

# Convert to DataFrame
df = pd.DataFrame(rows)

# Compute "all" rows (overall performance per model-language)
grouped = df.dropna(subset=["Exact Match", "N-Samples Original"])  # Only keep valid rows

aggregates = []
for (model, lang), group in grouped.groupby(["Model", "Language"]):
    total_weighted = (group["Exact Match"] * group["N-Samples Original"]).sum()
    total_samples = group["N-Samples Original"].sum()
    if total_samples > 0:
        overall_em = total_weighted / total_samples
        aggregates.append({
            "Model": model,
            "Language": lang,
            "Task": "all",
            "Exact Match": overall_em,
            "Exact Match StdErr": None,
            "N-Samples Original": total_samples,
            "N-Samples Effective": None,
            # "File": None
        })

# Add to original DataFrame
df = pd.concat([df, pd.DataFrame(aggregates)], ignore_index=True)

# Sort and save
df.sort_values(by=["Model", "Language", "Task"], inplace=True)

# Print preview
print(df.to_string(index=False))

# Save to CSV
df.to_csv("/home/hyujang/multilingual-inner-lexicon/output/summary_results.csv", index=False)




# import pandas as pd
# from itertools import product

# # Load your CSV
# df = pd.read_csv("/home/hyujang/multilingual-inner-lexicon/output/summary_results.csv")

# # Your lists
# models = [
#   "Tower-Babel/Babel-9B-Chat",
#   "google/gemma-3-12b-it",
#   "meta-llama/Llama-2-7b-chat-hf"
# ]

# langs = ["en", "ko", "de"]

# tasks = [
#   "biology", "business", "chemistry", "computer_science", "economics",
#   "engineering", "health", "history", "law", "math", "other",
#   "philosophy", "physics", "psychology"
# ]

# # Create full set of expected combinations
# expected = set()
# for model, lang, task in product(models, langs, tasks):
#     task_name = f"mmlu_prox_{lang}_{task}"
#     expected.add((model, task_name))

# # Extract actual runs from CSV, assuming columns named 'model', 'language', 'task'
# actual = set(zip(df['Model'], df['Task']))

# # Find missing runs
# missing = expected - actual

# print(f"Number of missing runs: {len(missing)}")
# print("Missing runs:")
# for m in sorted(missing):
#     print(m)
