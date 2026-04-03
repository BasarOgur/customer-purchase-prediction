"""
Enhanced Evaluation Module
==========================
This module evaluates TUNED models from GridSearchCV on the test set.

DIFFERENCES FROM evaluation.py:
-------------------------------
- Evaluates tuned models (from GridSearchCV) instead of default models
- Saves visualizations with "_enhanced" suffix to avoid overwriting
- Compares tuned vs default performance side by side
- Includes feature importance in the comparison charts

OUTPUT FILES:
-------------
- confusion_matrices_enhanced.png
- roc_curves_enhanced.png
- metrics_comparison_enhanced.png
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
    """
    try:
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(module_dir)
    except NameError:
        project_root = os.getcwd()
    
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, filename)


# =============================================================================
# PREDICT WITH TUNED MODELS
# =============================================================================
def predict_tuned_models(tuned_models: dict, X_test: np.ndarray, 
                          X_test_nn: np.ndarray = None) -> dict:
    """
    Make predictions using tuned models from GridSearchCV.
    
    Parameters
    ----------
    tuned_models : dict
        Dictionary from run_gridsearch_tuning() containing best_estimator for each model
    X_test : np.ndarray
        Test features (unscaled - for tree models)
    X_test_nn : np.ndarray, optional
        Test features (scaled - for neural network)
    
    Returns
    -------
    dict
        Predictions from each tuned model
    """
    predictions = {}
    
    # Tree models use unscaled data
    for model_name in ['Decision Tree', 'Random Forest', 'XGBoost']:
        if model_name in tuned_models:
            best_model = tuned_models[model_name]['best_estimator']
            predictions[model_name + ' (Tuned)'] = {
                'class': best_model.predict(X_test),
                'proba': best_model.predict_proba(X_test)[:, 1]
            }
    
    # Neural Network uses scaled data
    if 'Neural Network' in tuned_models:
        best_model = tuned_models['Neural Network']['best_estimator']
        nn_data = X_test_nn if X_test_nn is not None else X_test
        predictions['Neural Network (Tuned)'] = {
            'class': best_model.predict(nn_data),
            'proba': best_model.predict_proba(nn_data)[:, 1]
        }
    
    return predictions


# =============================================================================
# CALCULATE METRICS
# =============================================================================
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      y_pred_proba: np.ndarray) -> dict:
    """
    Calculate comprehensive evaluation metrics.
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
# BUILD RESULTS TABLE
# =============================================================================
def build_tuned_results_table(tuned_models: dict, X_test: np.ndarray,
                               y_test: np.ndarray, X_test_nn: np.ndarray = None) -> pd.DataFrame:
    """
    Calculate metrics for all tuned models.
    """
    predictions = predict_tuned_models(tuned_models, X_test, X_test_nn)
    
    results = []
    for model_name, preds in predictions.items():
        metrics = calculate_metrics(y_test, preds['class'], preds['proba'])
        metrics['model'] = model_name
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    results_df = results_df[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']]
    
    return results_df, predictions


# =============================================================================
# PLOT CONFUSION MATRICES (ENHANCED)
# =============================================================================
def plot_confusion_matrices_enhanced(predictions: dict, y_test: np.ndarray,
                                      show_plot: bool = True) -> None:
    """
    Plot confusion matrices for tuned models.
    Saves as confusion_matrices_enhanced.png
    """
    print("\nPlotting confusion matrices (enhanced)...")
    
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, preds) in enumerate(predictions.items()):
        y_pred = preds['class']
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[idx],
                    cbar_kws={'label': 'Count'})
        axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xticklabels(['No Purchase', 'Purchase'])
        axes[idx].set_yticklabels(['No Purchase', 'Purchase'])
    
    plt.tight_layout()
    save_path = get_output_path('confusion_matrices_enhanced.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


# =============================================================================
# PLOT ROC CURVES (ENHANCED)
# =============================================================================
def plot_roc_curves_enhanced(predictions: dict, y_test: np.ndarray,
                              show_plot: bool = True) -> None:
    """
    Plot ROC curves for tuned models.
    Saves as roc_curves_enhanced.png
    """
    print("\nPlotting ROC curves (enhanced)...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Colors for tuned models (green shades to distinguish from default)
    colors = {
        'Decision Tree (Tuned)': '#27ae60',
        'Random Forest (Tuned)': '#2980b9', 
        'XGBoost (Tuned)': '#8e44ad'
    }
    
    for model_name, preds in predictions.items():
        y_pred_proba = preds['proba']
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        color = colors.get(model_name, '#34495e')
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})',
                color=color, linewidth=2.5)
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Guessing (AUC = 0.5)')
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=11)
    ax.set_title('ROC Curves - Tuned Models (GridSearchCV)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    save_path = get_output_path('roc_curves_enhanced.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


# =============================================================================
# PLOT METRICS COMPARISON (ENHANCED)
# =============================================================================
def plot_metrics_comparison_enhanced(results_df: pd.DataFrame,
                                      show_plot: bool = True) -> None:
    """
    Create metrics comparison chart for tuned models.
    Saves as metrics_comparison_enhanced.png
    """
    print("\nPlotting metrics comparison (enhanced)...")
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    
    for idx, metric in enumerate(metrics):
        values = results_df[metric].values
        models = results_df['model'].values
        
        # Green color palette for tuned models
        colors_list = ['#27ae60', '#2980b9', '#8e44ad'][:len(models)]
        axes[idx].bar(models, values, color=colors_list, alpha=0.8, edgecolor='black')
        
        axes[idx].set_ylabel('Score', fontsize=10)
        axes[idx].set_title(metric.upper() + '\n(Tuned)', fontsize=11, fontweight='bold')
        axes[idx].set_ylim([0, 1])
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    save_path = get_output_path('metrics_comparison_enhanced.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


# =============================================================================
# PLOT DEFAULT VS TUNED COMPARISON
# =============================================================================
def plot_default_vs_tuned(default_results: pd.DataFrame, 
                           tuned_results: pd.DataFrame,
                           show_plot: bool = True) -> None:
    """
    Create side-by-side comparison of default vs tuned models.
    Saves as default_vs_tuned_comparison.png
    
    This visualization shows the improvement (or not) from hyperparameter tuning.
    """
    print("\nPlotting default vs tuned comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data - include Neural Network if available
    model_names = ['Decision Tree', 'Random Forest', 'XGBoost', 'Neural Network']
    
    # Filter to only models that exist in both results
    model_names = [m for m in model_names 
                   if m in default_results['model'].values 
                   and m + ' (Tuned)' in tuned_results['model'].values]
    
    # F1-Score comparison
    ax1 = axes[0]
    x = np.arange(len(model_names))
    width = 0.35
    
    default_f1 = [default_results[default_results['model'] == m]['f1'].values[0] 
                  for m in model_names]
    tuned_f1 = [tuned_results[tuned_results['model'] == m + ' (Tuned)']['f1'].values[0]
                for m in model_names]
    
    bars1 = ax1.bar(x - width/2, default_f1, width, label='Default', color='#e74c3c', alpha=0.7)
    bars2 = ax1.bar(x + width/2, tuned_f1, width, label='Tuned (GridSearchCV)', color='#27ae60', alpha=0.7)
    
    ax1.set_ylabel('F1-Score', fontsize=11)
    ax1.set_title('F1-Score: Default vs Tuned', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=15)
    ax1.legend()
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, default_f1):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', 
                ha='center', fontsize=9)
    for bar, val in zip(bars2, tuned_f1):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', 
                ha='center', fontsize=9)
    
    # ROC-AUC comparison
    ax2 = axes[1]
    
    default_auc = [default_results[default_results['model'] == m]['roc_auc'].values[0] 
                   for m in model_names]
    tuned_auc = [tuned_results[tuned_results['model'] == m + ' (Tuned)']['roc_auc'].values[0] 
                 for m in model_names]
    
    bars3 = ax2.bar(x - width/2, default_auc, width, label='Default', color='#e74c3c', alpha=0.7)
    bars4 = ax2.bar(x + width/2, tuned_auc, width, label='Tuned (GridSearchCV)', color='#27ae60', alpha=0.7)
    
    ax2.set_ylabel('ROC-AUC', fontsize=11)
    ax2.set_title('ROC-AUC: Default vs Tuned', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=15)
    ax2.legend()
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars3, default_auc):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', 
                ha='center', fontsize=9)
    for bar, val in zip(bars4, tuned_auc):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', 
                ha='center', fontsize=9)
    
    plt.tight_layout()
    save_path = get_output_path('default_vs_tuned_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


# =============================================================================
# PRINT DETAILED COMPARISON
# =============================================================================
def print_comparison_summary(default_results: pd.DataFrame, 
                              tuned_results: pd.DataFrame) -> None:
    """
    Print detailed comparison between default and tuned models.
    """
    print("\n" + "=" * 80)
    print("DEFAULT VS TUNED MODEL COMPARISON")
    print("=" * 80)
    
    # Include all models that exist in both
    model_names = ['Decision Tree', 'Random Forest', 'XGBoost', 'Neural Network']
    model_names = [m for m in model_names 
                   if m in default_results['model'].values 
                   and m + ' (Tuned)' in tuned_results['model'].values]
    
    print("\n" + "-" * 80)
    print(f"{'Model':<20} {'Metric':<12} {'Default':<10} {'Tuned':<10} {'Change':<15}")
    print("-" * 80)
    
    for model in model_names:
        for metric in ['f1', 'roc_auc', 'precision', 'recall']:
            default_val = default_results[default_results['model'] == model][metric].values[0]
            tuned_val = tuned_results[tuned_results['model'] == model + ' (Tuned)'][metric].values[0]
            change = tuned_val - default_val
            pct_change = (change / default_val * 100) if default_val > 0 else 0
            
            if change > 0:
                change_str = f"↑ +{change:.4f} (+{pct_change:.1f}%)"
            elif change < 0:
                change_str = f"↓ {change:.4f} ({pct_change:.1f}%)"
            else:
                change_str = "→ No change"
            
            if metric == 'f1':
                print(f"{model:<20} {metric.upper():<12} {default_val:<10.4f} {tuned_val:<10.4f} {change_str}")
            else:
                print(f"{'':<20} {metric.upper():<12} {default_val:<10.4f} {tuned_val:<10.4f} {change_str}")
        print()


# =============================================================================
# MAIN ENHANCED EVALUATION FUNCTION
# =============================================================================
def evaluate_tuned_models(tuned_models: dict, X_test: np.ndarray, 
                           y_test: np.ndarray, X_test_nn: np.ndarray = None,
                           default_results: pd.DataFrame = None,
                           show_plots: bool = True) -> pd.DataFrame:
    """
    Complete evaluation pipeline for tuned models.
    
    Parameters
    ----------
    tuned_models : dict
        Results from run_gridsearch_tuning()
    X_test : np.ndarray
        Test features (unscaled - for tree models)
    y_test : np.ndarray
        Test labels
    X_test_nn : np.ndarray, optional
        Test features (scaled - for neural network)
    default_results : pd.DataFrame, optional
        Results from default models for comparison
    show_plots : bool
        If True, display plots (set False for batch processing)
    
    Returns
    -------
    pd.DataFrame
        Results table with all metrics for tuned models
    """
    print("\n" + "=" * 80)
    print("ENHANCED EVALUATION PIPELINE (TUNED MODELS)")
    print("=" * 80)
    
    # Build results table
    print("\n[1/4] Calculating metrics for tuned models...")
    results_df, predictions = build_tuned_results_table(tuned_models, X_test, y_test, X_test_nn)
    print("✓ Metrics calculated")
    
    # Print results
    print("\n[2/4] Results Summary:")
    print("-" * 60)
    print(results_df.to_string(index=False))
    
    # Create visualizations
    print("\n[3/4] Creating enhanced visualizations...")
    plot_confusion_matrices_enhanced(predictions, y_test, show_plots)
    plot_roc_curves_enhanced(predictions, y_test, show_plots)
    plot_metrics_comparison_enhanced(results_df, show_plots)
    
    # Compare with default if provided
    
    # Compare with default if provided
    if default_results is not None:
        print("\n[4/4] Comparing with default models...")
        plot_default_vs_tuned(default_results, results_df, show_plots)
        print_comparison_summary(default_results, results_df)
    else:
        print("\n[4/4] Skipped default comparison (no default results provided)")
    
    print("\n" + "=" * 80)
    print("ENHANCED EVALUATION COMPLETE!")
    print("=" * 80)
    
    print("\n📁 Files created:")
    print("   - data/confusion_matrices_enhanced.png")
    print("   - data/roc_curves_enhanced.png")
    print("   - data/metrics_comparison_enhanced.png")
    if default_results is not None:
        print("   - data/default_vs_tuned_comparison.png")
    
    return results_df


# =============================================================================
# TEST THE MODULE
# =============================================================================
if __name__ == "__main__":
    print("This module evaluates tuned models from GridSearchCV.")
    print("Run main_enhanced.py to use this module.")
    print("\nMain function: evaluate_tuned_models()")
