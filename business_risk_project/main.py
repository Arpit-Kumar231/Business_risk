import os
import sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data
from src.preprocessing import DataPreprocessor
from src.models import train_evaluate_models
from src.visualization import plot_model_comparison
from explainability import generate_shap_plots

def main():
    # 1. Configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "company_data3.csv")
    plots_dir = os.path.join(base_dir, "plots")
    
    print("=== SME Business Risk Assessment Pipeline ===\n")
    
    
    try:
        df = load_data(data_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

   
    preprocessor = DataPreprocessor()
    df = preprocessor.derive_risk(df)
    
    
    X_train_scaled, X_test_scaled, y_train, y_test = preprocessor.preprocess(df)
    
    print(f"\nTraining set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    # 4. Model Training & Evaluation
    print("\nStarting Model Training...")
    results_df, models_dict = train_evaluate_models(
        X_train_scaled, y_train, X_test_scaled, y_test, return_models=True
    )
    
    
    print("\nGenerating Visualizations...")
    plot_model_comparison(results_df, plots_dir)
    
    # 6. Display Results
    print("\n=== Model Performance Comparison (Sorted by ROC-AUC) ===")
    print(results_df.to_string(index=False))
    
    # 7. Model Explainability Integration (SHAP)
    # Get feature names from the data pre-scaling
    # Based on preprocessing.py, the final feature set is all columns EXCEPT Credit_score and Risk
    feature_names = df.drop(columns=['Credit_score', 'Risk']).columns.tolist()
    
    # XGBoost is our best performing model, so we explain it
    if "XGBoost" in models_dict:
        xgb_model = models_dict["XGBoost"]
        try:
            generate_shap_plots(xgb_model, X_test_scaled, feature_names, plots_dir)
        except Exception as e:
            print(f"Could not generate SHAP plots: {e}")
    else:
        print("\nXGBoost model not found for explainability.")

    # # Output analysis
    # print("\n--- Analysis Comments ---")
    # print("1. XGBoost is expected to outperform due to its gradient boosting mechanism handling complex, non-linear relationships in financial data.")
    # print("2. MLP performance depends heavily on scaling (handled here) and the dataset size. It may be competitive but often requires more tuning.")
    # print("3. Random Forest and Gradient Boosting provide strong baselines, but XGBoost's regularization often gives it an edge.")

if __name__ == "__main__":
    main()
