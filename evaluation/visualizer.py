import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tabulate import tabulate
import os
from config import CHARTS_DIR

def get_emoji(score: float) -> str:
    if score >= 0.85:
        return "✅"
    elif score >= 0.70:
        return "⚠️"
    else:
        return "❌"

def print_custom_mode_table(base_prompt: str, results: dict):
    """
    Prints a rich table for Custom Prompt Mode.
    results format:
    {
        "gpt-4o-mini": {"Original": 1.0, "TextBugger": 0.91, "TextFooler": 0.87, ...},
        "claude-haiku": {"Original": 1.0, "TextBugger": 0.88, ...},
        ...
    }
    """
    print(f'\nPrompt: "{base_prompt}"\n')
    
    models = list(results.keys())
    # Assuming all models have the same attacks listed
    if not models:
        print("No results to display.")
        return
        
    attacks = list(results[models[0]].keys())
    
    table_data = []
    
    # Original row
    orig_row = ["Original"]
    for m in models:
        orig_row.append("similarity:\n1.00 (base)")
    table_data.append(orig_row)
    
    # Attack rows
    model_avgs = {m: [] for m in models}
    
    for attack in attacks:
        if attack == "Original":
            continue
        row = [attack]
        for m in models:
            score = results[m].get(attack, 0.0)
            model_avgs[m].append(score)
            row.append(f"{score:.2f} {get_emoji(score)}")
        table_data.append(row)
        
    # Avg Robustness row
    avg_row = ["Avg Robustness"]
    for m in models:
        avg = sum(model_avgs[m]) / len(model_avgs[m]) if model_avgs[m] else 0.0
        avg_row.append(f"{avg * 100:.1f}%")
        results[m]['__avg__'] = avg
    table_data.append(avg_row)
    
    headers = ["Attack Type"] + models
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    
    # Winner / Loser
    if models:
        ranked = sorted(models, key=lambda m: results[m].get('__avg__', 0), reverse=True)
        print(f"\n🏆 Most Robust: {ranked[0]}")
        if len(ranked) > 1:
            print(f"⚠️  Least Robust: {ranked[-1]}")
    print("\n")


def plot_accuracy_comparison(df: pd.DataFrame):
    """
    Saves a bar chart comparing clean vs attacked accuracy.
    df needs: Model, Dataset, Clean Accuracy, Attacked Accuracy
    """
    if df.empty:
        return
        
    plt.figure(figsize=(10, 6))
    
    # This is a placeholder for actual grouped bar chart logic
    # depending on how the CSV is structured.
    # We will refine this once we have the exact df shape.
    
    # We expect a melt format for seaborn
    melted = df.melt(id_vars=['Model', 'Dataset'], value_vars=['Clean Accuracy', 'Attacked Accuracy'], var_name='Metric', value_name='Accuracy')
    
    sns.barplot(data=melted, x='Model', y='Accuracy', hue='Metric', errorbar=None)
    plt.title('Clean vs Attacked Accuracy by Model')
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'accuracy_comparison.png'))
    plt.close()


def plot_robustness_heatmap(df: pd.DataFrame):
    """
    Saves a heatmap of robustness scores: Model x Attack Type.
    df needs: Model, Attack Type, Robustness Score
    """
    if df.empty or 'Attack Type' not in df.columns:
        return
        
    # Aggregate across datasets before pivoting to avoid duplicate index error
    agg_df = df.groupby(['Model', 'Attack Type'], as_index=False)['Robustness Score'].mean()
    pivot_df = agg_df.pivot(index='Model', columns='Attack Type', values='Robustness Score')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', fmt=".1f", vmin=0, vmax=100)
    plt.title('Robustness Score (%) Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'robustness_heatmap.png'))
    plt.close()
