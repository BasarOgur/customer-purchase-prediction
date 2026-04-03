"""
Enhanced Main Pipeline Script
=============================
This is the ENHANCED version of main.py with additional features:
1. GridSearchCV hyperparameter tuning
2. Feature importance analysis
3. Prediction function demonstration

HOW TO RUN:
-----------
From the project root directory, run:
    python main_enhanced.py

DIFFERENCES FROM main.py:
-------------------------
- main.py: Quick run (~10 seconds), basic pipeline
- main_enhanced.py: Full run (~3-5 minutes), includes tuning and analysis

WHAT THIS SCRIPT DOES:
----------------------
1. Load and preprocess data (same as main.py)
2. Train models with DEFAULT hyperparameters
3. Run GridSearchCV to find OPTIMAL hyperparameters
4. Compare default vs optimized performance
5. Analyze feature importance
6. Demonstrate prediction function with sample customers
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Import standard modules
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.models import train_all_models
from src.evaluation import evaluate_all_models

# Import enhancement modules
from src.enhancements import (
    run_gridsearch_tuning,
    analyze_feature_importance,
    predict_purchase,
    create_sample_customers
)

# Import enhanced evaluation (saves _enhanced.png files)
from src.evaluation_enhanced import evaluate_tuned_models


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def print_subheader(text: str):
    """Print a formatted subheader."""
    print("\n" + "-" * 70)
    print(text)
    print("-" * 70)


def main():
    """
    Enhanced main execution function.
    
    FLOW:
    -----
    1. Setup and load data
    2. Preprocess data
    3. Train models (default hyperparameters)
    4. Evaluate default models
    5. Run GridSearchCV (find optimal hyperparameters)
    6. Evaluate optimized models
    7. Compare default vs optimized
    8. Feature importance analysis
    9. Demonstrate prediction function
    """
    
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 10 + "CUSTOMER PURCHASE PREDICTION - ENHANCED PIPELINE" + " " * 19 + "║")
    print("║" + " " * 10 + "GridSearchCV + Feature Importance + Prediction" + " " * 21 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # =========================================================================
    # STEP 1: SETUP
    # =========================================================================
    print_header("STEP 1: SETUP")
    
    # Handle both direct execution and exec() calls
    try:
        project_root = Path(__file__).parent.absolute()
    except NameError:
        project_root = Path.cwd()
    dataset_path = project_root / "data" / "online_shoppers_intention.csv"
    
    if not dataset_path.exists():
        print(f"\n❌ ERROR: Dataset not found at {dataset_path}")
        print("Please run: python download_data.py")
        sys.exit(1)
    
    print(f"✓ Project root: {project_root}")
    print(f"✓ Dataset found: {dataset_path}")
    
    # =========================================================================
    # STEP 2: LOAD DATA
    # =========================================================================
    print_header("STEP 2: LOADING DATA")
    
    df = load_data(str(dataset_path))
    print(f"\nDataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Target variable: 'Revenue' (0=No Purchase, 1=Purchase)")
    print(f"\nClass distribution:")
    print(f"  No Purchase: {(df['Revenue'] == False).sum():,} ({(df['Revenue'] == False).mean()*100:.1f}%)")
    print(f"  Purchase:    {(df['Revenue'] == True).sum():,} ({(df['Revenue'] == True).mean()*100:.1f}%)")
    
    # =========================================================================
    # STEP 3: PREPROCESS DATA
    # =========================================================================
    print_header("STEP 3: PREPROCESSING DATA")
    
    data = preprocess_data(df)
    
    print(f"\n✓ Training samples: {data['X_train_trees'].shape[0]:,}")
    print(f"✓ Test samples: {data['X_test_trees'].shape[0]:,}")
    print(f"✓ Features after encoding: {data['X_train_trees'].shape[1]}")
    
    # =========================================================================
    # STEP 4: TRAIN DEFAULT MODELS
    # =========================================================================
    print_header("STEP 4: TRAINING MODELS (Default Hyperparameters)")
    
    models = train_all_models(
        X_train_trees=data['X_train_trees'],
        X_train_nn=data['X_train_nn'],
        y_train=data['y_train']
    )
    
    # =========================================================================
    # STEP 5: EVALUATE DEFAULT MODELS
    # =========================================================================
    print_header("STEP 5: EVALUATING DEFAULT MODELS")
    
    default_results = evaluate_all_models(
        models=models,
        X_test_trees=data['X_test_trees'],
        X_test_nn=data['X_test_nn'],
        y_test=data['y_test']
    )
    
    print("\nDefault Model Results:")
    print(default_results.to_string(index=False))
    
    # =========================================================================
    # STEP 6: GRIDSEARCHCV HYPERPARAMETER TUNING
    # =========================================================================
    print_header("STEP 6: GRIDSEARCHCV HYPERPARAMETER TUNING")
    print("\n⚠️  This step takes 5-10 minutes (includes Neural Network). Finding optimal hyperparameters...\n")
    
    tuning_results = run_gridsearch_tuning(
        X_train=data['X_train_trees'],
        y_train=data['y_train'],
        X_train_nn=data['X_train_nn'],  # Scaled data for Neural Network
        cv_folds=5,
        verbose=True
    )
    
    # =========================================================================
    # STEP 7: EVALUATE TUNED MODELS ON TEST SET (with enhanced visualizations)
    # =========================================================================
    print_header("STEP 7: EVALUATING TUNED MODELS ON TEST SET")
    
    # Use the enhanced evaluation module (saves _enhanced.png files)
    tuned_df = evaluate_tuned_models(
        tuned_models=tuning_results,
        X_test=data['X_test_trees'],
        y_test=data['y_test'],
        X_test_nn=data['X_test_nn'],  # Scaled data for Neural Network
        default_results=default_results,
        show_plots=False  # Don't display, just save
    )
    
    print("\nTuned Model Results (Test Set):")
    print(tuned_df.to_string(index=False))
    
    # Note: The comparison charts are already created by evaluate_tuned_models()
    # - data/default_vs_tuned_comparison.png shows the side-by-side comparison
    
    # =========================================================================
    # STEP 8: FEATURE IMPORTANCE ANALYSIS
    # =========================================================================
    print_header("STEP 8: FEATURE IMPORTANCE ANALYSIS")
    
    # Use the best tuned model for feature importance
    best_model_name = max(tuning_results.keys(), 
                         key=lambda x: tuning_results[x]['best_score'])
    best_model = tuning_results[best_model_name]['best_estimator']
    
    print(f"\nAnalyzing feature importance using: {best_model_name}")
    
    save_path = project_root / "data" / "feature_importance_enhanced.png"
    
    importance_df = analyze_feature_importance(
        model=best_model,
        feature_names=data['feature_names'],
        top_n=15,
        save_path=str(save_path),
        verbose=True
    )
    
    # =========================================================================
    # STEP 9: PREDICTION FUNCTION DEMONSTRATION
    # =========================================================================
    print_header("STEP 9: PREDICTION FUNCTION DEMONSTRATION")
    
    print("\nDemonstrating prediction function with sample customers...")
    print("Using the best tuned model for predictions.\n")
    
    sample_customers = create_sample_customers()
    
    for customer_name, customer_data in sample_customers:
        print_subheader(f"Sample: {customer_name}")
        
        result = predict_purchase(
            customer_data=customer_data,
            model=best_model,
            preprocessor=data['preprocessor_trees'],
            verbose=True
        )
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_header("FINAL SUMMARY")
    
    # Find overall best
    all_results = pd.concat([default_results, tuned_df], ignore_index=True)
    best_idx = all_results['f1'].idxmax()
    best_overall = all_results.loc[best_idx]
    
    print(f"\n🏆 BEST OVERALL MODEL: {best_overall['model']}")
    print(f"   F1-Score: {best_overall['f1']:.4f}")
    print(f"   ROC-AUC:  {best_overall['roc_auc']:.4f}")
    print(f"   Precision: {best_overall['precision']:.4f}")
    print(f"   Recall: {best_overall['recall']:.4f}")
    
    print("\n📁 FILES CREATED:")
    print("   Default model visualizations:")
    print("   - data/confusion_matrices.png")
    print("   - data/roc_curves.png")
    print("   - data/metrics_comparison.png")
    print("\n   Enhanced (tuned) model visualizations:")
    print("   - data/confusion_matrices_enhanced.png")
    print("   - data/roc_curves_enhanced.png")
    print("   - data/metrics_comparison_enhanced.png")
    print("   - data/default_vs_tuned_comparison.png")
    print("   - data/feature_importance_enhanced.png")
    
    print("\n📊 BEST HYPERPARAMETERS FOUND:")
    for model_name, result in tuning_results.items():
        print(f"\n   {model_name}:")
        for param, value in result['best_params'].items():
            print(f"      {param}: {value}")
    
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "✓ ENHANCED PIPELINE COMPLETED SUCCESSFULLY!" + " " * 20 + "║")
    print("╚" + "═" * 78 + "╝")
    
    print("\n📌 WHAT YOU LEARNED:")
    print("   1. How to use GridSearchCV for hyperparameter tuning")
    print("   2. How to analyze feature importance")
    print("   3. How to create a prediction function for new data")
    print("   4. How tuned models compare to default models")
    
    print("\n📌 NEXT STEPS:")
    print("   1. Review the visualizations in data/ folder")
    print("   2. Try modifying the hyperparameter grids")
    print("   3. Use predict_purchase() for real predictions")
    print("   4. Update your GitHub repository!")
    
    return {
        'default_results': default_results,
        'tuned_results': tuned_df,
        'tuning_details': tuning_results,
        'feature_importance': importance_df,
        'best_model': best_model
    }


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    """
    Entry point - runs when you execute: python main_enhanced.py
    
    Expected runtime: 3-5 minutes (due to GridSearchCV)
    """
    results = main()
