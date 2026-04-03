"""
Enhancements Module
===================
This module contains advanced features to enhance the ML pipeline.

FEATURES INCLUDED:
------------------
1. GridSearchCV - Automatically find the best hyperparameters
2. Feature Importance Analysis - See which features matter most
3. Prediction Function - Predict for new customer data

WHY THESE ENHANCEMENTS?
-----------------------
- GridSearchCV: Instead of manually trying hyperparameters, let the computer find the best ones
- Feature Importance: Understand WHAT the model learned (not just accuracy)
- Prediction Function: Make the model usable for real-world predictions

WHAT YOU'LL LEARN:
------------------
- How to use sklearn's GridSearchCV for hyperparameter tuning
- How to interpret feature importance from tree models
- How to create a production-ready prediction function
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import time
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# FEATURE 1: GRIDSEARCHCV FOR HYPERPARAMETER TUNING
# =============================================================================

def run_gridsearch_tuning(X_train: np.ndarray, y_train: np.ndarray, 
                          X_train_nn: np.ndarray = None,
                          cv_folds: int = 5, verbose: bool = True) -> dict:
    """
    Run GridSearchCV to find optimal hyperparameters for each model.
    
    WHAT IS GRIDSEARCHCV?
    ---------------------
    GridSearchCV automatically searches through a "grid" of hyperparameters
    to find the combination that gives the best performance.
    
    HOW IT WORKS:
    -------------
    1. You define a grid of hyperparameter values to try
    2. GridSearchCV tries EVERY combination
    3. For each combination, it uses K-Fold Cross-Validation to evaluate
    4. Returns the best combination found
    
    EXAMPLE:
    --------
    If you have:
        param_grid = {
            'max_depth': [5, 10, 15],      # 3 options
            'n_estimators': [50, 100]       # 2 options
        }
    GridSearchCV will try: 3 × 2 = 6 combinations
    
    With 5-fold CV, that's 6 × 5 = 30 model trainings!
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features (unscaled - for tree models)
    y_train : np.ndarray
        Training labels
    X_train_nn : np.ndarray, optional
        Training features (scaled - for neural network)
        If not provided, will use X_train for NN too
    cv_folds : int
        Number of cross-validation folds (default: 5)
    verbose : bool
        If True, print progress updates
    
    Returns
    -------
    dict
        Dictionary with best parameters and scores for each model
    """
    
    if verbose:
        print("=" * 70)
        print("GRIDSEARCHCV HYPERPARAMETER TUNING")
        print("=" * 70)
        print(f"\nUsing {cv_folds}-fold cross-validation")
        print("Scoring metric: F1-score (best for imbalanced data)")
        print("\n⚠️  This may take 2-5 minutes. Please wait...\n")
    
    results = {}
    
    # =========================================================================
    # MODEL 1: DECISION TREE
    # =========================================================================
    if verbose:
        print("-" * 70)
        print("1. DECISION TREE")
        print("-" * 70)
    
    dt_param_grid = {
        'max_depth': [5, 10, 15, 20, None],           # How deep the tree can grow
        'min_samples_split': [2, 5, 10, 20],          # Min samples to split a node
        'min_samples_leaf': [1, 2, 5, 10],            # Min samples in a leaf
        'class_weight': ['balanced', None]            # Handle imbalanced classes
    }
    
    if verbose:
        total_combinations = (len(dt_param_grid['max_depth']) * 
                             len(dt_param_grid['min_samples_split']) * 
                             len(dt_param_grid['min_samples_leaf']) *
                             len(dt_param_grid['class_weight']))
        print(f"   Parameter grid: {total_combinations} combinations to try")
    
    start_time = time.time()
    
    dt_grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=dt_param_grid,
        cv=cv_folds,
        scoring='f1',              # Optimize for F1-score
        n_jobs=-1,                 # Use all CPU cores
        verbose=0
    )
    
    dt_grid_search.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    
    results['Decision Tree'] = {
        'best_params': dt_grid_search.best_params_,
        'best_score': dt_grid_search.best_score_,
        'best_estimator': dt_grid_search.best_estimator_,
        'time': elapsed
    }
    
    if verbose:
        print(f"   ✓ Completed in {elapsed:.1f}s")
        print(f"   Best F1-Score: {dt_grid_search.best_score_:.4f}")
        print(f"   Best Parameters:")
        for param, value in dt_grid_search.best_params_.items():
            print(f"      - {param}: {value}")
    
    # =========================================================================
    # MODEL 2: RANDOM FOREST
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("2. RANDOM FOREST")
        print("-" * 70)
    
    rf_param_grid = {
        'n_estimators': [50, 100, 200],               # Number of trees
        'max_depth': [10, 15, 20, None],              # Max depth per tree
        'min_samples_split': [2, 5, 10],              # Min samples to split
        'min_samples_leaf': [1, 2, 4],                # Min samples in leaf
        'class_weight': ['balanced', 'balanced_subsample']  # Handle imbalance
    }
    
    if verbose:
        total_combinations = (len(rf_param_grid['n_estimators']) * 
                             len(rf_param_grid['max_depth']) * 
                             len(rf_param_grid['min_samples_split']) *
                             len(rf_param_grid['min_samples_leaf']) *
                             len(rf_param_grid['class_weight']))
        print(f"   Parameter grid: {total_combinations} combinations to try")
    
    start_time = time.time()
    
    rf_grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=rf_param_grid,
        cv=cv_folds,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    rf_grid_search.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    
    results['Random Forest'] = {
        'best_params': rf_grid_search.best_params_,
        'best_score': rf_grid_search.best_score_,
        'best_estimator': rf_grid_search.best_estimator_,
        'time': elapsed
    }
    
    if verbose:
        print(f"   ✓ Completed in {elapsed:.1f}s")
        print(f"   Best F1-Score: {rf_grid_search.best_score_:.4f}")
        print(f"   Best Parameters:")
        for param, value in rf_grid_search.best_params_.items():
            print(f"      - {param}: {value}")
    
    # =========================================================================
    # MODEL 3: XGBOOST
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("3. XGBOOST")
        print("-" * 70)
    
    # Calculate scale_pos_weight for imbalanced data
    n_negative = np.sum(y_train == 0)
    n_positive = np.sum(y_train == 1)
    scale_pos_weight = n_negative / n_positive
    
    xgb_param_grid = {
        'n_estimators': [50, 100, 200],               # Number of boosting rounds
        'max_depth': [3, 5, 7, 10],                   # Max depth per tree
        'learning_rate': [0.01, 0.05, 0.1, 0.2],      # Step size shrinkage
        'min_child_weight': [1, 3, 5],                # Min sum of instance weight
        'scale_pos_weight': [1, scale_pos_weight]     # Handle imbalance
    }
    
    if verbose:
        total_combinations = (len(xgb_param_grid['n_estimators']) * 
                             len(xgb_param_grid['max_depth']) * 
                             len(xgb_param_grid['learning_rate']) *
                             len(xgb_param_grid['min_child_weight']) *
                             len(xgb_param_grid['scale_pos_weight']))
        print(f"   Parameter grid: {total_combinations} combinations to try")
    
    start_time = time.time()
    
    xgb_grid_search = GridSearchCV(
        estimator=xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
        param_grid=xgb_param_grid,
        cv=cv_folds,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    xgb_grid_search.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    
    results['XGBoost'] = {
        'best_params': xgb_grid_search.best_params_,
        'best_score': xgb_grid_search.best_score_,
        'best_estimator': xgb_grid_search.best_estimator_,
        'time': elapsed
    }
    
    if verbose:
        print(f"   ✓ Completed in {elapsed:.1f}s")
        print(f"   Best F1-Score: {xgb_grid_search.best_score_:.4f}")
        print(f"   Best Parameters:")
        for param, value in xgb_grid_search.best_params_.items():
            print(f"      - {param}: {value}")
    
    # =========================================================================
    # MODEL 4: NEURAL NETWORK (MLPClassifier)
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("4. NEURAL NETWORK (MLPClassifier)")
        print("-" * 70)
        print("   ⚠️  Note: Neural Network tuning takes longer than tree models...")
    
    nn_param_grid = {
        'hidden_layer_sizes': [(32,), (64,), (64, 32), (128, 64)],  # Network architecture
        'alpha': [0.0001, 0.001, 0.01],                              # L2 regularization
        'learning_rate_init': [0.001, 0.01],                         # Initial learning rate
        'max_iter': [200, 300]                                       # Max epochs
    }
    
    if verbose:
        total_combinations = (len(nn_param_grid['hidden_layer_sizes']) * 
                             len(nn_param_grid['alpha']) * 
                             len(nn_param_grid['learning_rate_init']) *
                             len(nn_param_grid['max_iter']))
        print(f"   Parameter grid: {total_combinations} combinations to try")
    
    start_time = time.time()
    
    nn_grid_search = GridSearchCV(
        estimator=MLPClassifier(
            random_state=42,
            early_stopping=True,        # Stop if validation score doesn't improve
            validation_fraction=0.1,    # Use 10% of training for validation
            n_iter_no_change=10         # Stop after 10 iterations with no improvement
        ),
        param_grid=nn_param_grid,
        cv=cv_folds,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    # Neural Network needs SCALED data - use X_train_nn if provided
    nn_data = X_train_nn if X_train_nn is not None else X_train
    if verbose and X_train_nn is not None:
        print("   Using scaled data for Neural Network ✓")
    elif verbose:
        print("   ⚠️  Using unscaled data (X_train_nn not provided)")
    
    nn_grid_search.fit(nn_data, y_train)
    
    elapsed = time.time() - start_time
    
    results['Neural Network'] = {
        'best_params': nn_grid_search.best_params_,
        'best_score': nn_grid_search.best_score_,
        'best_estimator': nn_grid_search.best_estimator_,
        'time': elapsed
    }
    
    if verbose:
        print(f"   ✓ Completed in {elapsed:.1f}s")
        print(f"   Best F1-Score: {nn_grid_search.best_score_:.4f}")
        print(f"   Best Parameters:")
        for param, value in nn_grid_search.best_params_.items():
            print(f"      - {param}: {value}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("GRIDSEARCHCV SUMMARY")
        print("=" * 70)
        
        print("\nBest F1-Scores (Cross-Validation):")
        for model_name, model_results in results.items():
            print(f"   {model_name:20s}: {model_results['best_score']:.4f}")
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['best_score'])
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   F1-Score: {results[best_model_name]['best_score']:.4f}")
        
        total_time = sum(r['time'] for r in results.values())
        print(f"\n⏱️  Total tuning time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    return results


# =============================================================================
# FEATURE 2: FEATURE IMPORTANCE ANALYSIS
# =============================================================================

def analyze_feature_importance(model, feature_names: list, 
                               top_n: int = 15, save_path: str = None,
                               verbose: bool = True) -> pd.DataFrame:
    """
    Analyze and visualize which features are most important for predictions.
    
    WHAT IS FEATURE IMPORTANCE?
    ---------------------------
    Feature importance tells us how much each feature (column) contributes
    to the model's predictions.
    
    HOW IT'S CALCULATED (for tree models):
    --------------------------------------
    - Based on how much each feature reduces impurity (Gini or Entropy)
    - Features used in more splits and higher up in the tree are more important
    - Values are normalized to sum to 1.0 (100%)
    
    WHY IT MATTERS:
    ---------------
    1. UNDERSTANDING: Know what drives customer purchases
    2. FEATURE SELECTION: Remove unimportant features to simplify model
    3. BUSINESS INSIGHTS: Focus marketing on important factors
    4. DEBUGGING: Check if model is learning sensible patterns
    
    Parameters
    ----------
    model : fitted model
        A fitted tree-based model (DecisionTree, RandomForest, or XGBoost)
    feature_names : list
        List of feature names in order
    top_n : int
        Number of top features to display (default: 15)
    save_path : str
        Path to save the visualization (optional)
    verbose : bool
        If True, print detailed information
    
    Returns
    -------
    pd.DataFrame
        DataFrame with feature names and importance scores, sorted
    """
    
    if verbose:
        print("=" * 70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 70)
    
    # Get feature importances from the model
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        raise ValueError("Model doesn't have feature_importances_ attribute. "
                        "Use a tree-based model (DecisionTree, RandomForest, XGBoost).")
    
    # Create DataFrame with feature names and importances
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance (descending)
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df = importance_df.reset_index(drop=True)
    
    # Add rank and percentage columns
    importance_df['rank'] = range(1, len(importance_df) + 1)
    importance_df['percentage'] = importance_df['importance'] * 100
    
    if verbose:
        model_name = type(model).__name__
        print(f"\nModel: {model_name}")
        print(f"Total features: {len(feature_names)}")
        
        print(f"\n📊 TOP {top_n} MOST IMPORTANT FEATURES:")
        print("-" * 50)
        
        for idx, row in importance_df.head(top_n).iterrows():
            bar_length = int(row['percentage'] * 2)  # Scale for display
            bar = "█" * bar_length
            print(f"{row['rank']:2d}. {row['feature']:30s} {row['percentage']:5.1f}% {bar}")
        
        # Show cumulative importance of top features
        top_importance = importance_df.head(top_n)['importance'].sum() * 100
        print(f"\n📈 Top {top_n} features explain {top_importance:.1f}% of model's decisions")
        
        # Identify very low importance features
        low_importance = importance_df[importance_df['importance'] < 0.01]
        if len(low_importance) > 0:
            print(f"\n⚠️  {len(low_importance)} features have < 1% importance:")
            for _, row in low_importance.iterrows():
                print(f"   - {row['feature']} ({row['percentage']:.2f}%)")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Get top N features for plotting
    plot_df = importance_df.head(top_n).copy()
    
    # Create horizontal bar chart
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(plot_df)))
    
    plt.barh(range(len(plot_df)), plot_df['percentage'], color=colors)
    plt.yticks(range(len(plot_df)), plot_df['feature'])
    plt.xlabel('Importance (%)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features\n({type(model).__name__})', 
              fontsize=14, fontweight='bold')
    
    # Add percentage labels on bars
    for i, (_, row) in enumerate(plot_df.iterrows()):
        plt.text(row['percentage'] + 0.3, i, f"{row['percentage']:.1f}%", 
                va='center', fontsize=9)
    
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if verbose:
            print(f"\n✓ Visualization saved to: {save_path}")
    
    plt.close()
    
    return importance_df


# =============================================================================
# FEATURE 3: PREDICTION FUNCTION FOR NEW DATA
# =============================================================================

def predict_purchase(customer_data: dict, model, preprocessor,
                     verbose: bool = True) -> dict:
    """
    Predict whether a new customer will make a purchase.
    
    WHAT THIS FUNCTION DOES:
    ------------------------
    Takes raw customer data (same format as original CSV) and returns
    a prediction with confidence level.
    
    HOW TO USE:
    -----------
    1. Create a dictionary with customer data (all 17 features)
    2. Pass it to this function along with your trained model and preprocessor
    3. Get back: prediction (Buy/Not Buy) and probability
    
    EXAMPLE:
    --------
    customer = {
        'Administrative': 2,
        'Administrative_Duration': 50.5,
        'Informational': 0,
        'Informational_Duration': 0,
        'ProductRelated': 15,
        'ProductRelated_Duration': 500.0,
        'BounceRates': 0.02,
        'ExitRates': 0.04,
        'PageValues': 25.0,
        'SpecialDay': 0,
        'Month': 'Nov',
        'OperatingSystems': 2,
        'Browser': 2,
        'Region': 1,
        'TrafficType': 1,
        'VisitorType': 'Returning_Visitor',
        'Weekend': False
    }
    
    result = predict_purchase(customer, model, preprocessor)
    
    Parameters
    ----------
    customer_data : dict
        Dictionary containing all 17 features (same as CSV columns except Revenue)
    model : fitted model
        A trained classifier model
    preprocessor : fitted ColumnTransformer
        The preprocessor used during training (handles encoding and scaling)
    verbose : bool
        If True, print detailed prediction information
    
    Returns
    -------
    dict
        Dictionary with prediction, probability, and confidence level
    """
    
    # Required features (in expected order)
    required_features = [
        'Administrative', 'Administrative_Duration',
        'Informational', 'Informational_Duration',
        'ProductRelated', 'ProductRelated_Duration',
        'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
        'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType',
        'VisitorType', 'Weekend'
    ]
    
    # Validate input
    missing_features = [f for f in required_features if f not in customer_data]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Convert to DataFrame (preprocessor expects DataFrame)
    customer_df = pd.DataFrame([customer_data])
    
    # Convert Weekend to int (same as preprocessing)
    customer_df['Weekend'] = customer_df['Weekend'].astype(int)
    
    # Apply preprocessing (encoding, scaling)
    customer_processed = preprocessor.transform(customer_df)
    
    # Make prediction
    prediction = model.predict(customer_processed)[0]
    
    # Get probability (if model supports it)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(customer_processed)[0]
        prob_no_purchase = probabilities[0]
        prob_purchase = probabilities[1]
    else:
        # If model doesn't support probabilities, use prediction as 0 or 1
        prob_purchase = float(prediction)
        prob_no_purchase = 1.0 - prob_purchase
    
    # Determine confidence level
    max_prob = max(prob_purchase, prob_no_purchase)
    if max_prob >= 0.80:
        confidence = "High"
    elif max_prob >= 0.60:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    # Create result dictionary
    result = {
        'prediction': 'Will Purchase' if prediction == 1 else 'Will NOT Purchase',
        'prediction_numeric': int(prediction),
        'probability_purchase': prob_purchase,
        'probability_no_purchase': prob_no_purchase,
        'confidence': confidence
    }
    
    if verbose:
        print("=" * 60)
        print("PURCHASE PREDICTION")
        print("=" * 60)
        
        print("\n📋 CUSTOMER DATA:")
        print("-" * 40)
        for feature, value in customer_data.items():
            print(f"   {feature:30s}: {value}")
        
        print("\n" + "=" * 60)
        print("🎯 PREDICTION RESULT")
        print("=" * 60)
        
        if prediction == 1:
            print(f"\n   ✅ {result['prediction']}")
        else:
            print(f"\n   ❌ {result['prediction']}")
        
        print(f"\n   Confidence: {confidence}")
        print(f"   Probability of Purchase: {prob_purchase*100:.1f}%")
        print(f"   Probability of No Purchase: {prob_no_purchase*100:.1f}%")
        
        # Visual probability bar
        bar_length = 40
        purchase_bar = int(prob_purchase * bar_length)
        no_purchase_bar = bar_length - purchase_bar
        
        print(f"\n   Purchase:    [{'█' * purchase_bar}{'░' * no_purchase_bar}] {prob_purchase*100:.1f}%")
        print(f"   No Purchase: [{'█' * no_purchase_bar}{'░' * purchase_bar}] {prob_no_purchase*100:.1f}%")
        
        print("\n" + "=" * 60)
    
    return result


def create_sample_customers() -> list:
    """
    Create sample customer profiles for testing the prediction function.
    
    Returns
    -------
    list
        List of sample customer dictionaries
    """
    
    # Sample 1: Highly engaged customer (likely to buy)
    customer_likely_buyer = {
        'Administrative': 3,
        'Administrative_Duration': 120.0,
        'Informational': 2,
        'Informational_Duration': 80.0,
        'ProductRelated': 45,
        'ProductRelated_Duration': 1800.0,
        'BounceRates': 0.01,
        'ExitRates': 0.02,
        'PageValues': 35.0,          # High PageValue = likely to buy!
        'SpecialDay': 0,
        'Month': 'Nov',              # November (holiday shopping)
        'OperatingSystems': 2,
        'Browser': 2,
        'Region': 1,
        'TrafficType': 2,
        'VisitorType': 'Returning_Visitor',
        'Weekend': False
    }
    
    # Sample 2: Quick browser (unlikely to buy)
    customer_unlikely_buyer = {
        'Administrative': 0,
        'Administrative_Duration': 0,
        'Informational': 0,
        'Informational_Duration': 0,
        'ProductRelated': 5,
        'ProductRelated_Duration': 60.0,
        'BounceRates': 0.15,          # High bounce rate = leaving quickly
        'ExitRates': 0.12,
        'PageValues': 0.0,            # No PageValue = no intent to buy
        'SpecialDay': 0,
        'Month': 'Feb',
        'OperatingSystems': 1,
        'Browser': 1,
        'Region': 3,
        'TrafficType': 3,
        'VisitorType': 'New_Visitor',
        'Weekend': True
    }
    
    # Sample 3: Moderate engagement (uncertain)
    customer_maybe_buyer = {
        'Administrative': 1,
        'Administrative_Duration': 30.0,
        'Informational': 1,
        'Informational_Duration': 25.0,
        'ProductRelated': 20,
        'ProductRelated_Duration': 600.0,
        'BounceRates': 0.04,
        'ExitRates': 0.05,
        'PageValues': 8.0,            # Some PageValue
        'SpecialDay': 0.4,            # Near special day
        'Month': 'May',
        'OperatingSystems': 3,
        'Browser': 2,
        'Region': 2,
        'TrafficType': 1,
        'VisitorType': 'Returning_Visitor',
        'Weekend': False
    }
    
    return [
        ('Highly Engaged Customer (Likely Buyer)', customer_likely_buyer),
        ('Quick Browser (Unlikely Buyer)', customer_unlikely_buyer),
        ('Moderate Engagement (Uncertain)', customer_maybe_buyer)
    ]


# =============================================================================
# TEST THE MODULE
# =============================================================================
if __name__ == "__main__":
    print("This module contains enhancement functions.")
    print("Run main_enhanced.py to use these features.")
    print("\nAvailable functions:")
    print("  1. run_gridsearch_tuning() - Find best hyperparameters")
    print("  2. analyze_feature_importance() - See which features matter")
    print("  3. predict_purchase() - Predict for new customers")
