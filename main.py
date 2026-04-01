"""
Main Pipeline Script
====================
This is the ENTRY POINT for the entire project.
Run this script to execute the complete ML pipeline from start to finish.

WHAT DOES THIS SCRIPT DO?
-------------------------
1. Load the raw dataset
2. Preprocess the data (split, encode, scale)
3. Train all 4 models
4. Evaluate models and compare them
5. Print results and save visualizations

HOW TO RUN:
-----------
From the project root directory, run:
    python main.py

This single command does EVERYTHING!

WHY HAVE A MAIN.PY?
-------------------
Instead of running 5 different Python files in order,
we have one entry point that orchestrates the entire workflow.
This is professional, reproducible, and easy to use.

PROJECT STRUCTURE REVIEW:
-------------------------
src/
├── __init__.py           # Makes src a Python package
├── data_loader.py        # Load and inspect data
├── preprocessing.py      # Prepare data for models (split, encode, scale)
├── models.py             # Train 4 different models
├── evaluation.py         # Evaluate and compare models
└── [main.py is here]     # THIS FILE - orchestrates everything

DEPENDENCY CHAIN:
-----------------
main.py
  ├─→ data_loader.py
  ├─→ preprocessing.py (uses data_loader)
  ├─→ models.py (uses preprocessing)
  └─→ evaluation.py (uses models)
"""

import os
import sys
from pathlib import Path

# Import our custom modules
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.models import train_all_models
from src.evaluation import evaluate_all_models


# =============================================================================
# SECTION 1: SETUP AND CONFIGURATION
# =============================================================================
# Set up paths and create output directories if they don't exist

def setup_project():
    """
    Initialize project directories and configurations.

    WHY DO THIS?
    -----------
    We need to ensure:
    1. The data directory exists
    2. The data file is present
    3. Output directories exist
    4. We know the absolute paths (to avoid file-not-found errors)
    """
    print("\n" + "=" * 80)
    print("INITIALIZATION")
    print("=" * 80)

    # Get the project root directory (one level up from main.py)
    project_root = Path(__file__).parent.absolute()
    print(f"Project root: {project_root}")

    # Define important paths
    data_dir = project_root / "data"
    dataset_path = data_dir / "online_shoppers_intention.csv"

    # Check if dataset exists
    if not dataset_path.exists():
        print(f"\n❌ ERROR: Dataset not found at {dataset_path}")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset")
        sys.exit(1)

    print(f"✓ Dataset found: {dataset_path}")

    # Optionally create data directory if it doesn't exist
    # (it should already exist, but this is defensive programming)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(project_root / "notebooks", exist_ok=True)

    print(f"✓ Directories verified")

    return str(dataset_path)


# =============================================================================
# SECTION 2: LOAD DATA
# =============================================================================
def load_and_inspect():
    """
    Load the dataset and show basic information.

    WHAT HAPPENS HERE:
    ------------------
    1. Load CSV file into pandas DataFrame
    2. Show basic info (shape, columns, types, missing values)
    3. Understand what we're working with

    WHY?
    ---
    Good practices: Always inspect data before processing.
    Catches issues early (wrong file, corrupted data, etc.)
    """
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)

    # Get dataset path
    project_root = Path(__file__).parent.absolute()
    dataset_path = project_root / "data" / "online_shoppers_intention.csv"

    # Load data using our data_loader module
    df = load_data(str(dataset_path))

    # Show basic information
    print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Target variable: 'Revenue' (0=No Purchase, 1=Purchase)")
    print(f"\nClass distribution:")
    print(f"  No Purchase (0): {(df['Revenue'] == False).sum():,}")
    print(f"  Purchase (1):    {(df['Revenue'] == True).sum():,}")
    print(f"\nMissing values: {df.isnull().sum().sum()}")

    return df


# =============================================================================
# SECTION 3: PREPROCESS DATA
# =============================================================================
def preprocess():
    """
    Load data and run the preprocessing pipeline.

    WHAT HAPPENS HERE:
    ------------------
    1. Load data
    2. Separate features (X) and target (y)
    3. Split into train/test sets (80/20)
    4. Handle categorical variables (one-hot encoding)
    5. Scale numerical features (for neural networks)

    WHY TWO VERSIONS?
    ----------------
    We create TWO versions of preprocessed data:
    - trees_version: No scaling (for tree-based models)
    - nn_version: Scaled (for neural networks)

    This is because:
    - Tree models: Don't need scaling (decisions based on thresholds)
    - Neural networks: Need scaling (gradient descent is sensitive to scale)
    """
    print("\n" + "=" * 80)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 80)

    # Load data
    project_root = Path(__file__).parent.absolute()
    dataset_path = project_root / "data" / "online_shoppers_intention.csv"
    df = load_data(str(dataset_path))

    # Run preprocessing
    data = preprocess_data(df)

    return data


# =============================================================================
# SECTION 4: TRAIN MODELS
# =============================================================================
def train_models(data):
    """
    Train all 4 models using preprocessed data.

    WHAT HAPPENS HERE:
    ------------------
    1. Train Decision Tree Classifier
    2. Train Random Forest Classifier
    3. Train XGBoost Classifier
    4. Train Neural Network (MLP)

    IMPORTANT DETAIL:
    ----------------
    We pass TWO versions of training data:
    - Trees: X_train_trees (no scaling)
    - Neural Network: X_train_nn (scaled)

    Each model gets the data it needs!

    EXPECTED TIME:
    --------------
    Total training should take 3-5 seconds on a modern computer
    Neural Network takes longest (it iterates through epochs)
    """
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING MODELS")
    print("=" * 80)

    models = train_all_models(
        X_train_trees=data['X_train_trees'],
        X_train_nn=data['X_train_nn'],
        y_train=data['y_train']
    )

    return models


# =============================================================================
# SECTION 5: EVALUATE MODELS
# =============================================================================
def evaluate_models(models, data):
    """
    Evaluate all models on the test set and compare performance.

    WHAT HAPPENS HERE:
    ------------------
    1. Make predictions for each model
    2. Calculate metrics (accuracy, precision, recall, F1, ROC-AUC)
    3. Build comparison table
    4. Create visualizations:
       - Confusion matrices
       - ROC curves
       - Metrics comparison bars
    5. Print detailed reports
    6. Identify best model by F1-score

    OUTPUT FILES:
    ---------------
    data/confusion_matrices.png   - 2x2 grid of confusion matrices
    data/roc_curves.png           - ROC curves for all models
    data/metrics_comparison.png   - Bar charts comparing metrics

    KEY METRICS:
    -----------
    For imbalanced data like ours, focus on:
    1. F1-Score (best for imbalanced data) - PRIMARY FOCUS
    2. ROC-AUC (threshold-independent) - SECONDARY FOCUS

    Ignore plain accuracy (misleading for imbalanced data).
    """
    print("\n" + "=" * 80)
    print("STEP 4: EVALUATING MODELS")
    print("=" * 80)

    results = evaluate_all_models(
        models=models,
        X_test_trees=data['X_test_trees'],
        X_test_nn=data['X_test_nn'],
        y_test=data['y_test']
    )

    return results


# =============================================================================
# SECTION 6: SUMMARY AND RECOMMENDATIONS
# =============================================================================
def print_final_summary(results):
    """
    Print a final summary and recommendations.

    WHAT THIS DOES:
    ---------------
    1. Show which model won by F1-score
    2. Compare all metrics
    3. Give business recommendations
    4. Explain results
    """
    print("\n" + "=" * 80)
    print("FINAL SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    # Find best model by F1
    best_idx = results['f1'].idxmax()
    best_model = results.loc[best_idx]

    print(f"\n🏆 BEST MODEL: {best_model['model']}")
    print(f"   F1-Score: {best_model['f1']:.4f}")
    print(f"   ROC-AUC:  {best_model['roc_auc']:.4f}")
    print(f"   Accuracy: {best_model['accuracy']:.4f}")
    print(f"   Precision: {best_model['precision']:.4f}")
    print(f"   Recall: {best_model['recall']:.4f}")

    # Rankings by different metrics
    print("\n" + "-" * 80)
    print("RANKINGS BY DIFFERENT METRICS:")
    print("-" * 80)

    print("\nBy F1-Score (★ BEST for imbalanced data):")
    for i, (idx, row) in enumerate(results.nlargest(4, 'f1').iterrows(), 1):
        print(f"  {i}. {row['model']:20s} F1={row['f1']:.4f}")

    print("\nBy ROC-AUC (★ GOOD for imbalanced data):")
    for i, (idx, row) in enumerate(results.nlargest(4, 'roc_auc').iterrows(), 1):
        print(f"  {i}. {row['model']:20s} ROC-AUC={row['roc_auc']:.4f}")

    print("\nBy Accuracy (⚠️  NOT reliable for imbalanced data):")
    for i, (idx, row) in enumerate(results.nlargest(4, 'accuracy').iterrows(), 1):
        print(f"  {i}. {row['model']:20s} Accuracy={row['accuracy']:.4f}")

    # Model comparison
    print("\n" + "-" * 80)
    print("MODEL COMPARISON TABLE:")
    print("-" * 80)
    print(results.to_string(index=False))

    # Key insights
    print("\n" + "-" * 80)
    print("KEY INSIGHTS:")
    print("-" * 80)

    # Check if tree models dominate
    tree_models = results[results['model'].isin(['Decision Tree', 'Random Forest', 'XGBoost'])]
    nn_model = results[results['model'] == 'Neural Network']

    avg_tree_f1 = tree_models['f1'].mean()
    avg_nn_f1 = nn_model['f1'].iloc[0]

    if avg_tree_f1 > avg_nn_f1:
        print(f"\n✓ Tree-based models outperform neural networks")
        print(f"  Average tree F1-Score: {avg_tree_f1:.4f}")
        print(f"  Neural Network F1-Score: {avg_nn_f1:.4f}")
        print(f"\n  Why? Tabular data has clear decision boundaries,")
        print(f"  trees exploit feature interactions better.")

    else:
        print(f"\n✓ Neural Network performs competitively or better")
        print(f"  This suggests complex nonlinear patterns in the data.")

    # Check class imbalance handling
    best_precision = results['precision'].max()
    best_recall = results['recall'].max()

    if best_precision > 0.75 or best_recall > 0.75:
        print(f"\n✓ Good class imbalance handling detected")
        print(f"  Best Precision: {best_precision:.4f}")
        print(f"  Best Recall: {best_recall:.4f}")
        print(f"\n  Models are not just predicting majority class.")

    print("\n" + "-" * 80)
    print("RECOMMENDATIONS FOR BUSINESS USE:")
    print("-" * 80)
    print(f"\n1. Deploy: {best_model['model']}")
    print(f"   This model balances precision and recall best.")

    print(f"\n2. Monitor: F1-score ({best_model['f1']:.4f})")
    print(f"   Track F1-score, not accuracy!")

    print(f"\n3. Use case: Identify potential buyers")
    print(f"   Precision: {best_model['precision']:.4f} ({int(best_model['precision']*100)}% of predictions correct)")
    print(f"   Recall: {best_model['recall']:.4f} ({int(best_model['recall']*100)}% of actual buyers caught)")

    print("\n" + "=" * 80)


# =============================================================================
# SECTION 7: MAIN EXECUTION FUNCTION
# =============================================================================
def main():
    """
    Main execution function - orchestrates the entire pipeline.

    FLOW:
    -----
    Setup → Load Data → Preprocess → Train → Evaluate → Summarize

    This function calls all the steps in order.
    """
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "CUSTOMER PURCHASE PREDICTION ML PIPELINE" + " " * 24 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        # Step 1: Setup
        setup_project()

        # Step 2: Load
        load_and_inspect()

        # Step 3: Preprocess
        data = preprocess()

        # Step 4: Train
        models = train_models(data)

        # Step 5: Evaluate
        results = evaluate_models(models, data)

        # Step 6: Summary
        print_final_summary(results)

        # Success message
        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + " " * 20 + "✓ PIPELINE COMPLETED SUCCESSFULLY!" + " " * 25 + "║")
        print("╚" + "=" * 78 + "╝")

        print("\nNext steps:")
        print("1. Review visualizations in the data/ folder")
        print("2. Check notebooks/01_eda.ipynb for exploratory analysis")
        print("3. Use the best model for predictions on new data")
        print("4. Upload this project to GitHub for your portfolio!")

        return results

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Check that data/online_shoppers_intention.csv exists")
        print("2. Check that all src/*.py files are present")
        print("3. Check that all dependencies are installed (pip install -r requirements.txt)")
        raise


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    """
    Entry point - runs when you execute: python main.py

    WHY THIS PATTERN?
    ----------------
    if __name__ == "__main__":
        This idiom means "if this file is run directly (not imported)"

    This allows:
    - Running: python main.py (executes main())
    - Importing: from main import main (doesn't auto-run)
    """
    results = main()
