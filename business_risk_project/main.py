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
    
    # 8. AI Agent-Based Model Explainability
    print("\n" + "="*80)
    print("AI AGENT-BASED CLASSIFIER - NOVEL REASONING APPROACH")
    print("="*80)
    
    if "AI Agent" in models_dict:
        agent_model = models_dict["AI Agent"]
        print("\nNOVELTY: This model uses intelligent reasoning combining 5 sub-agents:")
        print("  1. Anomaly Detection Agent - Identifies statistical outliers")
        print("  2. Pattern Recognition Agent - Recognizes risky feature combinations")
        print("  3. Weighted Scoring Agent - Uses feature importance for scoring")
        print("  4. Outlier Detection Agent - IQR-based outlier analysis")
        print("  5. Distribution Analysis Agent - Analyzes feature extremeness")
        
        print("\nThe AI Agent COMBINES multiple reasoning strategies instead of just")
        print("learning correlations, making it more interpretable and novel.")
        
        # Show reasoning for a few test samples
        print("\nExample: AI Agent Reasoning for Sample 0:")
        print("-" * 80)
        try:
            reasoning = agent_model.get_agent_reasoning(X_test_scaled, sample_idx=0)
            print(f"  Anomaly Detection Score: {reasoning['anomaly_score']:.4f}")
            print(f"  Pattern Recognition Score: {reasoning['pattern_score']:.4f}")
            print(f"  Weighted Scoring Score: {reasoning['weighted_score']:.4f}")
            print(f"  Outlier Detection Score: {reasoning['outlier_score']:.4f}")
            print(f"  Distribution Analysis Score: {reasoning['distribution_score']:.4f}")
            print(f"  ➜ Final Risk Prediction: {reasoning['final_prediction']:.4f}")
            print(f"  Reasoning Depth: {reasoning['reasoning_depth']}")
            print(f"  Adaptive Threshold: {reasoning['adaptive_threshold']:.4f}")
        except Exception as e:
            print(f"  Could not generate reasoning: {e}")
        
        # Compare AI Agent with best model
        ai_agent_perf = results_df[results_df['Model'] == 'AI Agent']
        best_model_perf = results_df.iloc[0]
        
        if not ai_agent_perf.empty:
            print("\n" + "-" * 80)
            print("AI Agent vs Best Model Comparison:")
            print(f"  Best Model: {best_model_perf['Model']} (ROC-AUC: {best_model_perf['ROC-AUC']:.4f})")
            print(f"  AI Agent: {ai_agent_perf['ROC-AUC'].values[0]:.4f}")
            print(f"  F1-Score AI Agent: {ai_agent_perf['F1-Score'].values[0]:.4f}")
            print("\n  ✓ AI Agent provides interpretable multi-strategy reasoning")
            print("  ✓ Useful for understanding WHY a company is high/low risk")
            print("  ✓ Novel approach combining traditional analytics with agent logic")

    # # Output analysis
    # print("\n--- Analysis Comments ---")
    # print("1. XGBoost is expected to outperform due to its gradient boosting mechanism handling complex, non-linear relationships in financial data.")
    # print("2. MLP performance depends heavily on scaling (handled here) and the dataset size. It may be competitive but often requires more tuning.")
    # print("3. Random Forest and Gradient Boosting provide strong baselines, but XGBoost's regularization often gives it an edge.")

if __name__ == "__main__":
    main()
