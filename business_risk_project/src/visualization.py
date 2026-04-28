import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_model_comparison(results_df, output_dir):
    """
    Plots a bar chart and a heatmap comparing all evaluation metrics for the models.
    Saves the plots to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    
    # --- 1. Grouped Bar Chart ---
    melted_df = results_df.melt(id_vars="Model", value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    chart = sns.barplot(data=melted_df, x="Model", y="Score", hue="Metric", palette="viridis")
    
    plt.title("Model Performance Comparison (All Metrics)", fontsize=16)
    plt.ylim(0, 1.15)  # Make room for the legend and labels
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    
    for container in chart.containers:
        chart.bar_label(container, fmt='%.3f', padding=3, rotation=90)

    plt.tight_layout()
    
    bar_output_path = os.path.join(output_dir, "model_comparison_bar.png")
    plt.savefig(bar_output_path)
    print(f"\nModel comparison bar chart saved to: {bar_output_path}")
    plt.close()
