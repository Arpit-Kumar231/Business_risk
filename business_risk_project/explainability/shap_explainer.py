import os
import shap
import matplotlib
import matplotlib.pyplot as plt

# Use Agg backend for non-interactive environments and to ensure it doesn't block
matplotlib.use('Agg')

def generate_shap_plots(model, X_test, feature_names, output_dir):
    """
    Generates and saves SHAP plots for the given model.

    Args:
        model: Trained model (e.g., XGBoost).
        X_test: Test features (numpy array or pandas DataFrame).
        feature_names: List of feature names.
        output_dir: Directory to save the generated plots.
    """
    print("\nGenerating SHAP explainability plots...")
    os.makedirs(output_dir, exist_ok=True)

    # Convert X_test to DataFrame if it's a numpy array to keep feature names
    import pandas as pd
    if not isinstance(X_test, pd.DataFrame):
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
    else:
        X_test_df = X_test
        X_test_df.columns = feature_names

    # Explainer for tree-based models
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test_df)
    except Exception as e:
        print(f"Error initializing TreeExplainer: {e}")
        return

    # 1. Global feature importance (summary plot)
    try:
        plt.figure()
        shap.summary_plot(shap_values, X_test_df, show=False)
        summary_path = os.path.join(output_dir, "shap_summary.png")
        plt.savefig(summary_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {summary_path}")
    except Exception as e:
        print(f"Error generating summary plot: {e}")

    # 2. Bar plot for feature ranking
    try:
        plt.figure()
        shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
        bar_path = os.path.join(output_dir, "shap_bar.png")
        plt.savefig(bar_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {bar_path}")
    except Exception as e:
        print(f"Error generating bar plot: {e}")

    # 3. Waterfall plot for a single prediction (e.g., first instance)
    try:
        plt.figure()
        # Different structure for waterfall: it expects a single Explanation object
        shap.plots.waterfall(shap_values[0], show=False)
        waterfall_path = os.path.join(output_dir, "shap_waterfall.png")
        plt.savefig(waterfall_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {waterfall_path}")
    except Exception as e:
        print(f"Error generating waterfall plot: {e}")

    # 4. Force plot for a single prediction
    try:
        # Check standard SHAP objects format vs older numpy array structure
        if hasattr(shap_values, "base_values"):
            # New Explanation object format
            plt.figure()
            shap.plots.force(
                shap_values[0].base_values, 
                shap_values[0].values, 
                X_test_df.iloc[0], 
                matplotlib=True, 
                show=False
            )
        else:
            # Older numpy array format
            plt.figure()
            shap.plots.force(
                explainer.expected_value, 
                shap_values[0], 
                X_test_df.iloc[0], 
                matplotlib=True, 
                show=False
            )
        force_path = os.path.join(output_dir, "shap_force.png")
        plt.savefig(force_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {force_path}")
    except Exception as e:
        print(f"Error generating force plot: {e}")

    # 5. Generate Text Explanation
    try:
        import numpy as np
        text_path = os.path.join(output_dir, "shap_explanation.txt")
        with open(text_path, "w") as f:
            f.write("=== SHAP Feature Importance Explanation ===\n\n")
            
            # Global Importance
            f.write("--- Global Feature Importance ---\n")
            f.write("This ranks features by their average absolute impact on the model's output across all testing samples.\n")
            
            if hasattr(shap_values, "values"):
                vals = np.abs(shap_values.values).mean(0)
            else:
                vals = np.abs(shap_values).mean(0)
            
            feature_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['Feature', 'Importance'])
            feature_importance.sort_values(by=['Importance'], ascending=False, inplace=True)
            
            for idx, row in feature_importance.iterrows():
                f.write(f"- {row['Feature']}: {row['Importance']:.4f}\n")
                
            # Local Importance for Single Prediction (Instance 0)
            f.write("\n--- Local Feature Importance (First Instance) ---\n")
            f.write("This explains the prediction for the first specific instance in the test set, corresponding to the Waterfall and Force plots.\n")
            
            if hasattr(shap_values, "values"):
                base_val = shap_values[0].base_values
                instance_vals = shap_values[0].values
                # In newer shap versions, base_values can be an array if multi-class, but typically scalar for binary logistic
                if isinstance(base_val, np.ndarray) and base_val.size == 1:
                    base_val = base_val[0]
            else:
                base_val = explainer.expected_value
                if isinstance(base_val, np.ndarray) and base_val.size == 1:
                    base_val = base_val[0]
                instance_vals = shap_values[0]
                
            pred_val = instance_vals.sum() + base_val
            
            f.write(f"Base Value (Average model output): {base_val:.4f}\n")
            f.write(f"Predicted Value (Instance output): {pred_val:.4f}\n\n")
            f.write("Feature Contributions for this instance:\n")
            
            instance_importance = pd.DataFrame(list(zip(feature_names, instance_vals)), columns=['Feature', 'Contribution'])
            # Sort by absolute contribution to show most impactful features first
            instance_importance['AbsContribution'] = instance_importance['Contribution'].abs()
            instance_importance.sort_values(by=['AbsContribution'], ascending=False, inplace=True)
            
            for idx, row in instance_importance.iterrows():
                f.write(f"- {row['Feature']}: {row['Contribution']:+.4f}\n")
                
        print(f"Saved: {text_path}")
    except Exception as e:
        print(f"Error generating text explanation: {e}")

    print("SHAP explainability plots and text explanation completed.")
