"""
Evaluation Module
=================
This module evaluates all trained models on the test set.

WHY SEPARATE MODULE?
--------------------
- Keeps evaluation logic organized
- Easy to calculate different metrics
- Reusable for new data
- Generates consistent visualizations

WHAT METRICS DO WE CARE ABOUT?
------------------------------
Remember: our data is imbalanced (84% No Purchase, 16% Purchase).

If we only used ACCURACY, a model that predicts "No Purchase" always
would get 84% - seems good, but it's useless!

So we use multiple metrics to get the full picture:
- Accuracy: Overall correctness
- Precision: Of those we predicted as "Purchase", how many actually did?
- Recall: Of all people who actually purchased, how many did we catch?
- F1-Score: Balance between precision and recall
- ROC-AUC: How well does the model rank customers (best for imbalanced data)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    classification_report
)


# =============================================================================
# PATH HELPER
# =============================================================================
def get_output_path(filename: str) -> str:
    """
    Get absolute path to output file in data directory.

    This ensures plots are saved correctly regardless of where the script is run from.

    Parameters
    ----------
    filename : str
        Name of the file (e.g., 'confusion_matrices.png')

    Returns
    -------
    str
        Absolute path to the file
    """
    # Get the directory where this module lives (src/)
    module_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to project root, then into data/
    project_root = os.path.dirname(module_dir)
    data_dir = os.path.join(project_root, 'data')
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, filename)


def predict_all_models(models: dict, X_test_trees: np.ndarray,
                       X_test_nn: np.ndarray) -> dict:
    """
    Make predictions for all models on the test set.

    IMPORTANT DETAIL:
    -----------------
    Notice we use DIFFERENT test data for different models:
    - Tree models use X_test_trees (NOT scaled)
    - Neural Network uses X_test_nn (scaled)

    This is because we preprocessed them differently!

    Parameters
    ----------
    models : dict
        Dictionary of trained models
    X_test_trees : np.ndarray
        Test features WITHOUT scaling
    X_test_nn : np.ndarray
        Test features WITH scaling

    Returns
    -------
    dict
        Predictions from each model (both probability and class)
    """
    predictions = {}

    # Tree models - use unscaled test data
    for model_name in ['Decision Tree', 'Random Forest', 'XGBoost']:
        predictions[model_name] = {
            # predict() returns class labels (0 or 1)
            'class': models[model_name].predict(X_test_trees),
            # predict_proba() returns probability for each class [P(0), P(1)]
            'proba': models[model_name].predict_proba(X_test_trees)[:, 1]
        }

    # Neural Network - use scaled test data
    model_name = 'Neural Network'
    predictions[model_name] = {
        'class': models[model_name].predict(X_test_nn),
        'proba': models[model_name].predict_proba(X_test_nn)[:, 1]
    }

    return predictions


# =============================================================================
# STEP 2: CALCULATE METRICS
# =============================================================================
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      y_pred_proba: np.ndarray) -> dict:
    """
    Calculate comprehensive evaluation metrics for a model.

    ALL METRICS EXPLAINED:
    ----------------------

    1. ACCURACY
       Formula: (TP + TN) / (TP + TN + FP + FN)
       Meaning: Out of all predictions, how many were correct?
       Range: 0-1, higher is better
       ⚠️  Problem with imbalanced data: Useless if just predicts majority class

    2. PRECISION
       Formula: TP / (TP + FP)
       Meaning: When we predict "Purchase", how often are we right?
       Range: 0-1, higher is better
       ✓ Use when: False positives are expensive
       Example: If false alarm costs $100, precision matters more

    3. RECALL (aka Sensitivity)
       Formula: TP / (TP + FN)
       Meaning: Of all people who actually purchased, what % did we catch?
       Range: 0-1, higher is better
       ✓ Use when: False negatives are expensive
       Example: If missing a buyer costs $1000, recall matters more

    4. F1-SCORE
       Formula: 2 × (Precision × Recall) / (Precision + Recall)
       Meaning: Harmonic mean of precision and recall
       Range: 0-1, higher is better
       ✓ Use when: You want to balance precision and recall
       ✓ BEST metric for imbalanced data (use as PRIMARY metric)

    5. ROC-AUC
       Meaning: How well does the model rank customers?
       Range: 0-1, higher is better
       0.5 = random guessing
       1.0 = perfect ranking
       ✓ Use when: Comparing models on imbalanced data
       ROC-AUC is THRESHOLD-INDEPENDENT (doesn't care about our 0.5 cutoff)

    CONFUSION MATRIX TERMS:
    ─────────────────────────
    True Positive (TP):     Predicted Purchase, actually purchased ✓✓
    True Negative (TN):     Predicted No Purchase, no purchase ✓✓
    False Positive (FP):    Predicted Purchase, didn't purchase ✗✓
                            (Type I error - "false alarm")
    False Negative (FN):    Predicted No Purchase, but did purchase ✗✗
                            (Type II error - "missed opportunity")

    Parameters
    ----------
    y_true : np.ndarray
        True labels from test set
    y_pred : np.ndarray
        Class predictions (0 or 1)
    y_pred_proba : np.ndarray
        Probability predictions (0.0 to 1.0)

    Returns
    -------
    dict
        Dictionary with all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
    }

    return metrics


# =============================================================================
# STEP 3: BUILD RESULTS TABLE
# =============================================================================
def build_results_table(models: dict, predictions: dict,
                       y_test: np.ndarray) -> pd.DataFrame:
    """
    Calculate metrics for all models and create comparison table.

    Parameters
    ----------
    models : dict
        All trained models
    predictions : dict
        Predictions from each model
    y_test : np.ndarray
        True test labels

    Returns
    -------
    pd.DataFrame
        Results table with all metrics for each model
    """
    print("\n" + "=" * 80)
    print("CALCULATING METRICS FOR ALL MODELS")
    print("=" * 80)

    results = []

    for model_name in models.keys():
        print(f"\nEvaluating {model_name}...")

        # Get predictions
        y_pred = predictions[model_name]['class']
        y_pred_proba = predictions[model_name]['proba']

        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['model'] = model_name

        results.append(metrics)

    # Convert to DataFrame for nice display
    results_df = pd.DataFrame(results)

    # Reorder columns
    results_df = results_df[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']]

    return results_df


# =============================================================================
# STEP 4: VISUALIZE CONFUSION MATRICES
# =============================================================================
def plot_confusion_matrices(models: dict, predictions: dict,
                            y_test: np.ndarray) -> None:
    """
    Plot confusion matrices for all models.

    WHAT IS A CONFUSION MATRIX?
    ---------------------------
    A 2x2 table showing correct and incorrect predictions:

                    Predicted
                    No Buy  Buy
    Actual No Buy    TN     FP
           Buy       FN     TP

    WHAT TO LOOK FOR:
    - Diagonal (TN and TP): Correct predictions (should be high)
    - Off-diagonal (FP and FN): Wrong predictions (should be low)

    For imbalanced data:
    - TN will be very large (most are no-buys)
    - TP will be small (few are buys)
    - Focus on right column (TP and FN) - the interesting "Buy" cases
    """
    print("\nPlotting confusion matrices...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (model_name, model) in enumerate(models.items()):
        # Get predictions
        y_pred = predictions[model_name]['class']

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    cbar_kws={'label': 'Count'})
        axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xticklabels(['No Purchase', 'Purchase'])
        axes[idx].set_yticklabels(['No Purchase', 'Purchase'])

    plt.tight_layout()
    plt.savefig(get_output_path('confusion_matrices.png'), dpi=300, bbox_inches='tight')
    print("Saved: data/confusion_matrices.png")
    plt.show()


# =============================================================================
# STEP 5: VISUALIZE ROC CURVES
# =============================================================================
def plot_roc_curves(models: dict, predictions: dict, y_test: np.ndarray) -> None:
    """
    Plot ROC (Receiver Operating Characteristic) curves for all models.

    WHAT IS A ROC CURVE?
    --------------------
    Shows the trade-off between:
    - True Positive Rate (Recall): % of actual purchasers we catch
    - False Positive Rate: % of non-purchasers we incorrectly classify as purchasers

    HOW TO READ IT:
    - Top-left corner = perfect (TPR=1, FPR=0)
    - Diagonal line = random guessing (useless model)
    - Above diagonal = better than random
    - AUC = Area Under Curve = overall ranking ability

    EXAMPLE INTERPRETATION:
    If ROC-AUC = 0.85:
    - Take a random buyer and non-buyer
    - 85% chance model ranks the buyer higher

    WHY IT'S BETTER THAN ACCURACY FOR IMBALANCED DATA:
    - Independent of threshold (0.5 cutoff)
    - Considers all possible decision points
    - Shows full picture of model performance
    """
    print("\nPlotting ROC curves...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Colors for each model
    colors = {'Decision Tree': '#e74c3c', 'Random Forest': '#3498db',
              'XGBoost': '#2ecc71', 'Neural Network': '#f39c12'}

    for model_name, model in models.items():
        # Get predicted probabilities
        y_pred_proba = predictions[model_name]['proba']

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})',
                color=colors[model_name], linewidth=2)

    # Plot diagonal (random guessing)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Guessing (AUC = 0.5)')

    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=11)
    ax.set_title('ROC Curves - Model Comparison', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    plt.savefig(get_output_path('roc_curves.png'), dpi=300, bbox_inches='tight')
    print("Saved: data/roc_curves.png")
    plt.show()


# =============================================================================
# STEP 6: VISUALIZE METRICS COMPARISON
# =============================================================================
def plot_metrics_comparison(results_df: pd.DataFrame) -> None:
    """
    Create a visualization comparing all metrics across models.

    Shows each metric as a bar chart, making it easy to see:
    - Which model is best for each metric
    - Which metric differs most between models
    """
    print("\nPlotting metrics comparison...")

    # Metrics to compare (excluding 'model' column)
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    for idx, metric in enumerate(metrics):
        # Get metric values
        values = results_df[metric].values
        models = results_df['model'].values

        # Create bar chart
        colors_list = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        axes[idx].bar(models, values, color=colors_list, alpha=0.7, edgecolor='black')

        axes[idx].set_ylabel('Score', fontsize=10)
        axes[idx].set_title(metric.upper(), fontsize=11, fontweight='bold')
        axes[idx].set_ylim([0, 1])
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(get_output_path('metrics_comparison.png'), dpi=300, bbox_inches='tight')
    print("Saved: data/metrics_comparison.png")
    plt.show()


# =============================================================================
# STEP 7: PRINT DETAILED RESULTS
# =============================================================================
def print_results_summary(results_df: pd.DataFrame) -> None:
    """
    Print a nicely formatted results summary.
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION RESULTS")
    print("=" * 80)

    print("\nMetrics Table:")
    print("-" * 80)
    print(results_df.to_string(index=False))

    print("\n\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)

    print("""
For IMBALANCED DATA (like ours: 84% No Purchase, 16% Purchase):

• ACCURACY: Overall correctness ⚠️ Not reliable alone!
  - A useless model can get 84% by always predicting "No Purchase"

• PRECISION: When we predict someone will buy, how often are we right?
  - Higher = fewer false alarms
  - Use if false alarms are costly

• RECALL: Of people who actually buy, what % do we identify?
  - Higher = catch more buyers
  - Use if missing buyers is costly

• F1-SCORE: Balance between precision and recall
  - ✓ BEST metric for imbalanced data (PRIMARY FOCUS)
  - Harmonic mean - unlikely to be high unless both are good

• ROC-AUC: How well does the model rank customers?
  - ✓ EXCELLENT metric for imbalanced data (SECONDARY FOCUS)
  - Threshold-independent (considers all decision points)
  - 0.5 = random, 1.0 = perfect

RECOMMENDATION:
Focus on F1-SCORE and ROC-AUC, not accuracy!
    """)

    # Find best model by F1-score
    best_model_idx = results_df['f1'].idxmax()
    best_model = results_df.loc[best_model_idx]

    print("\n" + "=" * 80)
    print("BEST MODEL BY F1-SCORE (recommended metric for imbalanced data)")
    print("=" * 80)
    print(f"\nModel: {best_model['model']}")
    print(f"  F1-Score:   {best_model['f1']:.4f}")
    print(f"  ROC-AUC:    {best_model['roc_auc']:.4f}")
    print(f"  Precision:  {best_model['precision']:.4f}")
    print(f"  Recall:     {best_model['recall']:.4f}")
    print(f"  Accuracy:   {best_model['accuracy']:.4f}")


# =============================================================================
# STEP 8: DETAILED CLASSIFICATION REPORT
# =============================================================================
def print_classification_reports(models: dict, predictions: dict,
                                 y_test: np.ndarray) -> None:
    """
    Print sklearn's detailed classification report for each model.

    This includes:
    - Precision, recall, F1-score for each class
    - Support (number of samples in each class)
    - Weighted averages
    """
    print("\n" + "=" * 80)
    print("DETAILED CLASSIFICATION REPORTS")
    print("=" * 80)

    for model_name in models.keys():
        y_pred = predictions[model_name]['class']

        print(f"\n{model_name}:")
        print("-" * 80)
        print(classification_report(
            y_test, y_pred,
            target_names=['No Purchase', 'Purchase'],
            digits=4
        ))


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================
def evaluate_all_models(models: dict, X_test_trees: np.ndarray,
                        X_test_nn: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """
    Complete evaluation pipeline: predict → calculate → visualize → compare.

    Parameters
    ----------
    models : dict
        All trained models
    X_test_trees : np.ndarray
        Test features without scaling
    X_test_nn : np.ndarray
        Test features with scaling
    y_test : np.ndarray
        Test labels

    Returns
    -------
    pd.DataFrame
        Results table with all metrics
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION PIPELINE")
    print("=" * 80)

    # Step 1: Make predictions
    print("\n[1/5] Making predictions...")
    predictions = predict_all_models(models, X_test_trees, X_test_nn)
    print("✓ Predictions completed")

    # Step 2: Build results table
    print("\n[2/5] Calculating metrics...")
    results_df = build_results_table(models, predictions, y_test)
    print("✓ Metrics calculated")

    # Step 3: Print results
    print("\n[3/5] Generating reports...")
    print_results_summary(results_df)
    print_classification_reports(models, predictions, y_test)
    print("✓ Reports generated")

    # Step 4: Visualize results
    print("\n[4/5] Creating visualizations...")
    plot_confusion_matrices(models, predictions, y_test)
    plot_roc_curves(models, predictions, y_test)
    plot_metrics_comparison(results_df)
    print("✓ Visualizations created")

    print("\n[5/5] Evaluation complete!")
    print("=" * 80)

    return results_df


# =============================================================================
# TEST THE MODULE
# =============================================================================
if __name__ == "__main__":
    # Quick test when running this file directly
    from data_loader import load_data
    from preprocessing import preprocess_data
    from models import train_all_models

    print("Testing evaluation module...\n")

    # Load and preprocess data
    df = load_data("/data/online_shoppers_intention.csv")
    data = preprocess_data(df)

    # Train all models
    models = train_all_models(
        X_train_trees=data['X_train_trees'],
        X_train_nn=data['X_train_nn'],
        y_train=data['y_train']
    )

    # Evaluate all models
    results = evaluate_all_models(
        models=models,
        X_test_trees=data['X_test_trees'],
        X_test_nn=data['X_test_nn'],
        y_test=data['y_test']
    )

    print("\nEvaluation complete!")
    print(f"\nBest model by F1-Score: {results.loc[results['f1'].idxmax(), 'model']}")
