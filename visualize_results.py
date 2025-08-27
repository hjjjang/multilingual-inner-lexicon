import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv("/home/hyujang/multilingual-inner-lexicon/output/summary_results.csv")

# Filter out 'all' task as it's an aggregate
df = df[df['Task'] != 'all']

# Clean model names for better display
df['Model_Clean'] = df['Model'].str.replace('Tower-Babel/', '').str.replace('meta-llama/', '').str.replace('google/', '')

# Define colors for models
model_colors = {
    'Babel-9B-Chat': '#1f77b4',  # blue
    'gemma-3-12b-it': '#ff7f0e',  # orange  
    'Llama-2-7b-chat-hf': '#2ca02c'  # green
}

# Define line styles for languages
language_styles = {
    'en': '-',      # solid
    'de': '--',     # dashed
    'ko': ':'       # dotted
}

# Set up the plot
plt.figure(figsize=(15, 10))

# Get unique tasks and sort them
tasks = sorted(df['Task'].unique())

# Create model*language combinations
for model in df['Model_Clean'].unique():
    for lang in df['Language'].unique():
        subset = df[(df['Model_Clean'] == model) & (df['Language'] == lang)]
        
        if len(subset) > 0:
            # Sort by task to ensure consistent ordering
            subset = subset.sort_values('Task')
            
            # Get accuracies for each task
            accuracies = []
            task_labels = []
            
            for task in tasks:
                task_data = subset[subset['Task'] == task]
                if len(task_data) > 0:
                    accuracies.append(task_data['Exact Match'].iloc[0])
                    task_labels.append(task)
                else:
                    # If task is missing for this model-language pair, skip
                    continue
            
            if len(accuracies) > 0:
                plt.plot(range(len(task_labels)), accuracies, 
                        color=model_colors[model], 
                        linestyle=language_styles[lang],
                        linewidth=2,
                        marker='o',
                        markersize=4,
                        label=f'{model} ({lang})')

# Customize the plot
plt.xlabel('Task', fontsize=12, fontweight='bold')
plt.ylabel('Exact Match Accuracy', fontsize=12, fontweight='bold')
plt.title('Model Performance Across Tasks by Language', fontsize=14, fontweight='bold')

# Set x-axis labels
plt.xticks(range(len(tasks)), tasks, rotation=45, ha='right')

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Customize legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('/home/hyujang/multilingual-inner-lexicon/output/model_performance_by_task.png', 
            dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print("=" * 50)
for model in df['Model_Clean'].unique():
    print(f"\n{model}:")
    for lang in df['Language'].unique():
        subset = df[(df['Model_Clean'] == model) & (df['Language'] == lang) & (df['Task'] != 'all')]
        if len(subset) > 0:
            mean_acc = subset['Exact Match'].mean()
            print(f"  {lang}: {mean_acc:.3f} (avg across {len(subset)} tasks)")

# Create a second plot showing overall performance by language
plt.figure(figsize=(12, 8))

# Calculate mean performance by model and language
summary_stats = df[df['Task'] != 'all'].groupby(['Model_Clean', 'Language'])['Exact Match'].agg(['mean', 'std']).reset_index()

# Create bar plot
x_pos = np.arange(len(df['Language'].unique()))
width = 0.25

for i, model in enumerate(df['Model_Clean'].unique()):
    model_data = summary_stats[summary_stats['Model_Clean'] == model]
    means = []
    stds = []
    
    for lang in df['Language'].unique():
        lang_data = model_data[model_data['Language'] == lang]
        if len(lang_data) > 0:
            means.append(lang_data['mean'].iloc[0])
            stds.append(lang_data['std'].iloc[0])
        else:
            means.append(0)
            stds.append(0)
    
    plt.bar(x_pos + i * width, means, width, 
            yerr=stds, capsize=5,
            label=model, 
            color=model_colors[model],
            alpha=0.8)

plt.xlabel('Language', fontsize=12, fontweight='bold')
plt.ylabel('Average Exact Match Accuracy', fontsize=12, fontweight='bold')
plt.title('Average Model Performance by Language', fontsize=14, fontweight='bold')
plt.xticks(x_pos + width, df['Language'].unique())
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/hyujang/multilingual-inner-lexicon/output/average_performance_by_language.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print("\nPlots saved to:")
print("- /home/hyujang/multilingual-inner-lexicon/output/model_performance_by_task.png")
print("- /home/hyujang/multilingual-inner-lexicon/output/average_performance_by_language.png")
