import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_data
from src.preprocessing import DataPreprocessor
from src.models import train_evaluate_models
from src.visualization import plot_model_comparison

def main():
    # 1. Configuration
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "company_data.csv")
    plots_dir = os.path.join(base_dir, "plots")
    
    print("=== SME Business Risk Assessment Pipeline ===\n")
    
    # 2. Load Data
    try:
        df = load_data(data_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 3. Preprocessing
    preprocessor = DataPreprocessor()
    df = preprocessor.derive_risk(df)
    
    # Split and Scale
    X_train_scaled, X_test_scaled, y_train, y_test = preprocessor.preprocess(df)
    
    print(f"\nTraining set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    # 4. Model Training & Evaluation
    print("\nStarting Model Training...")
    results_df = train_evaluate_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 5. Visualization
    print("\nGenerating Visualizations...")
    plot_model_comparison(results_df, plots_dir)
    
    # 6. Display Results
    print("\n=== Model Performance Comparison (Sorted by ROC-AUC) ===")
    print(results_df.to_string(index=False))
    
    # Output analysis
    print("\n--- Analysis Comments ---")
    print("1. XGBoost is expected to outperform due to its gradient boosting mechanism handling complex, non-linear relationships in financial data.")
    print("2. MLP performance depends heavily on scaling (handled here) and the dataset size. It may be competitive but often requires more tuning.")
    print("3. Random Forest and Gradient Boosting provide strong baselines, but XGBoost's regularization often gives it an edge.")

if __name__ == "__main__":
    main()
