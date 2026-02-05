import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_model_comparison(results_df, output_dir):
    """
    Plots a bar chart comparing Accuracy and ROC-AUC for all models.
    Saves the plot to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Melt DataFrame for plotting with Seaborn
    metrics_to_plot = ["Accuracy", "ROC-AUC"]
    melted_df = results_df.melt(id_vars="Model", value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Create Bar Plot
    chart = sns.barplot(data=melted_df, x="Model", y="Score", hue="Metric", palette="viridis")
    
    plt.title("Model Performance Comparison (Accuracy vs ROC-AUC)", fontsize=16)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add value labels
    for container in chart.containers:
        chart.bar_label(container, fmt='%.3f', padding=3)

    plt.tight_layout()
    
    # Save Plot
    output_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(output_path)
    print(f"\nCorrection visualization saved to: {output_path}")
    plt.close()
