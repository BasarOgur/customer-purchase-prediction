"""
Models Module
=============
This module contains functions to train different ML classification models.

WE'RE COMPARING 4 MODELS:
-------------------------
1. Decision Tree    - Simple, interpretable, like a flowchart
2. Random Forest    - Many trees voting together (ensemble)
3. XGBoost          - Gradient boosting, often wins competitions
4. Neural Network   - Inspired by brain neurons (MLPClassifier)

WHY COMPARE MULTIPLE MODELS?
----------------------------
No single model is always best. Different models have different strengths:

| Model         | Strengths                      | Weaknesses                    |
|---------------|--------------------------------|-------------------------------|
| Decision Tree | Easy to interpret, fast        | Can overfit, unstable         |
| Random Forest | More accurate, handles noise   | Slower, less interpretable    |
| XGBoost       | Often best accuracy, fast      | More hyperparameters to tune  |
| Neural Net    | Can learn complex patterns     | Needs more data, slow to train|

For tabular data (like ours), tree-based models often perform best.
Neural networks shine more with images, text, and huge datasets.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import time


# =============================================================================
# MODEL 1: DECISION TREE CLASSIFIER
# =============================================================================
def train_decision_tree(X_train: np.ndarray, y_train: np.ndarray,
                        random_state: int = 42) -> DecisionTreeClassifier:
    """
    Train a Decision Tree Classifier.

    WHAT IS A DECISION TREE?
    ------------------------
    A decision tree is like a flowchart of yes/no questions.

    Example:
                    [Is PageValues > 10?]
                     /                \
                   Yes                No
                   /                    \
        [Is ExitRate < 0.05?]       [Revenue = No]
             /        \
           Yes        No
           /           \
    [Revenue = Yes]  [Revenue = No]

    HOW IT WORKS:
    1. Find the best feature and threshold to split data
    2. Split data into two groups
    3. Repeat for each group until stopping criteria

    PROS:
    - Easy to understand and visualize
    - No need to scale features
    - Fast to train and predict

    CONS:
    - Can easily overfit (memorize training data)
    - Unstable (small data changes → very different tree)
    - Not the most accurate

    HYPERPARAMETERS WE USE:
    -----------------------
    max_depth=10:
        Limits how deep the tree can grow.
        Too deep → overfitting (memorizes data)
        Too shallow → underfitting (too simple)

    min_samples_split=10:
        Minimum samples needed to split a node.
        Higher = more conservative, less overfitting

    min_samples_leaf=5:
        Minimum samples in a leaf node.
        Prevents very specific rules like "if this exact person..."

    class_weight='balanced':
        Automatically adjusts for class imbalance!
        Gives more importance to the minority class (Purchase).
        Without this, model might just predict "No Purchase" always.
    """
    print("\n" + "-" * 50)
    print("Training Decision Tree...")
    print("-" * 50)

    start_time = time.time()

    # Create the model with hyperparameters
    model = DecisionTreeClassifier(
        max_depth=10,              # Limit tree depth to prevent overfitting
        min_samples_split=10,      # Need at least 10 samples to split
        min_samples_leaf=5,        # Each leaf must have at least 5 samples
        class_weight='balanced',   # Handle class imbalance automatically
        random_state=random_state  # For reproducibility
    )

    # Train the model
    # fit() is where the model LEARNS patterns from training data
    model.fit(X_train, y_train)

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Tree depth: {model.get_depth()}")
    print(f"Number of leaves: {model.get_n_leaves()}")

    return model


# =============================================================================
# MODEL 2: RANDOM FOREST CLASSIFIER
# =============================================================================
def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        random_state: int = 42) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier.

    WHAT IS A RANDOM FOREST?
    ------------------------
    Random Forest = Many decision trees working together.

    Imagine asking 100 experts (trees) for their opinion:
    - Each expert sees only PART of the data (random sampling)
    - Each expert considers only SOME features (random selection)
    - Final answer = majority vote of all experts

    This is called "ensemble learning" - combining multiple models.

    WHY IS IT BETTER THAN A SINGLE TREE?
    ------------------------------------
    Single tree: Might make wrong decisions, unstable
    100 trees:   Errors cancel out, more stable and accurate

    It's like asking one person vs. polling 100 people.

    HYPERPARAMETERS WE USE:
    -----------------------
    n_estimators=100:
        Number of trees in the forest.
        More trees = more accurate, but slower.
        100 is a good balance.

    max_depth=15:
        Maximum depth of each tree.
        Can be deeper than single tree because randomness
        helps prevent overfitting.

    min_samples_split=5:
        Minimum samples to split a node.

    min_samples_leaf=2:
        Minimum samples in leaf nodes.

    class_weight='balanced':
        Handles our imbalanced classes.

    n_jobs=-1:
        Use all CPU cores for faster training.
        Training trees can be done in parallel!
    """
    print("\n" + "-" * 50)
    print("Training Random Forest...")
    print("-" * 50)

    start_time = time.time()

    model = RandomForestClassifier(
        n_estimators=100,          # 100 trees in the forest
        max_depth=15,              # Each tree can be up to 15 levels deep
        min_samples_split=5,       # Need 5 samples to split
        min_samples_leaf=2,        # Each leaf needs at least 2 samples
        class_weight='balanced',   # Handle imbalanced classes
        random_state=random_state,
        n_jobs=-1                  # Use all CPU cores (parallel training)
    )

    model.fit(X_train, y_train)

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Number of trees: {model.n_estimators}")

    return model


# =============================================================================
# MODEL 3: XGBOOST CLASSIFIER
# =============================================================================
def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  random_state: int = 42) -> xgb.XGBClassifier:
    """
    Train an XGBoost Classifier.

    WHAT IS XGBOOST?
    ----------------
    XGBoost = eXtreme Gradient Boosting

    Unlike Random Forest (trees work independently),
    XGBoost builds trees SEQUENTIALLY where each new tree
    tries to fix the mistakes of previous trees.

    Analogy:
    - Random Forest: 100 students take a test independently, majority vote
    - XGBoost: First student takes test, second student focuses on what
               first student got wrong, third focuses on remaining errors...

    This is called "boosting" - each model boosts the performance.

    WHY IS XGBOOST OFTEN THE BEST FOR TABULAR DATA?
    -----------------------------------------------
    1. Regularization: Prevents overfitting built-in
    2. Handles missing values automatically
    3. Very optimized and fast
    4. Wins most Kaggle competitions on tabular data!

    HYPERPARAMETERS WE USE:
    -----------------------
    n_estimators=100:
        Number of boosting rounds (trees to build).

    max_depth=6:
        Shallower than Random Forest because boosting
        can overfit more easily.

    learning_rate=0.1:
        How much each tree contributes (shrinkage).
        Lower = more trees needed, but often better.
        Think of it as "learning speed" - slower is often better.

    subsample=0.8:
        Use 80% of data for each tree (like Random Forest).
        Adds randomness to prevent overfitting.

    colsample_bytree=0.8:
        Use 80% of features for each tree.

    scale_pos_weight:
        Handles class imbalance by giving more weight
        to the positive (minority) class.
        Calculated as: negative_count / positive_count
    """
    print("\n" + "-" * 50)
    print("Training XGBoost...")
    print("-" * 50)

    start_time = time.time()

    # Calculate scale_pos_weight for imbalanced data
    # This tells XGBoost how much more important the minority class is
    negative_count = np.sum(y_train == 0)
    positive_count = np.sum(y_train == 1)
    scale_pos_weight = negative_count / positive_count
    print(f"Class imbalance ratio: {scale_pos_weight:.2f}:1")

    model = xgb.XGBClassifier(
        n_estimators=100,            # 100 boosting rounds
        max_depth=6,                 # Shallower trees for boosting
        learning_rate=0.1,           # Step size shrinkage
        subsample=0.8,               # Use 80% of data per tree
        colsample_bytree=0.8,        # Use 80% of features per tree
        scale_pos_weight=scale_pos_weight,  # Handle imbalance
        random_state=random_state,
        eval_metric='logloss',       # Metric to optimize
        use_label_encoder=False      # Avoid deprecation warning
    )

    model.fit(X_train, y_train)

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    return model


# =============================================================================
# MODEL 4: NEURAL NETWORK (MLP CLASSIFIER)
# =============================================================================
def train_neural_network(X_train: np.ndarray, y_train: np.ndarray,
                         random_state: int = 42) -> MLPClassifier:
    """
    Train a Neural Network (Multi-Layer Perceptron) Classifier.

    WHAT IS A NEURAL NETWORK?
    -------------------------
    A neural network is inspired by how the brain works.
    It consists of layers of "neurons" connected together.

    Structure:
        Input Layer → Hidden Layers → Output Layer
        (features)    (learning)      (prediction)

    Visual representation of our network:
        Input (27 features)
              ↓
        Hidden Layer 1 (64 neurons) ← ReLU activation
              ↓
        Hidden Layer 2 (32 neurons) ← ReLU activation
              ↓
        Output Layer (1 neuron) ← Sigmoid → 0 or 1

    HOW IT LEARNS:
    1. Forward pass: Input flows through network, makes prediction
    2. Calculate error: Compare prediction to actual answer
    3. Backward pass: Adjust weights to reduce error (backpropagation)
    4. Repeat many times (epochs)

    WHY USE MLPClassifier (sklearn) INSTEAD OF TensorFlow/Keras?
    ------------------------------------------------------------
    For this project, MLPClassifier is better because:
    - Simpler API, consistent with other sklearn models
    - Easier to compare with tree models
    - Good enough for tabular data
    - Faster for small datasets

    TensorFlow/Keras would be better for:
    - Deep networks (many layers)
    - Custom architectures
    - Large datasets
    - Computer vision or NLP

    IMPORTANT: NEURAL NETWORKS NEED SCALED DATA!
    ---------------------------------------------
    Unlike tree models, neural networks need scaled features.
    If one feature ranges 0-360 and another 0-0.2, the large
    values dominate learning. Scaling puts all features on
    similar scale (mean=0, std=1).

    HYPERPARAMETERS WE USE:
    -----------------------
    hidden_layer_sizes=(64, 32):
        Two hidden layers with 64 and 32 neurons.
        - First layer: learns basic patterns
        - Second layer: combines patterns into complex features

    activation='relu':
        ReLU = Rectified Linear Unit
        Simple function: max(0, x)
        If negative → 0, if positive → keep value
        Works well in practice, trains fast.

    alpha=0.001:
        L2 regularization to prevent overfitting.
        Penalizes large weights.

    learning_rate_init=0.001:
        Initial step size for weight updates.

    max_iter=500:
        Maximum training epochs.
        One epoch = one pass through all training data.

    early_stopping=True:
        Stop training if validation score stops improving.
        Prevents overfitting by not training too long.

    validation_fraction=0.1:
        Use 10% of training data for early stopping check.
    """
    print("\n" + "-" * 50)
    print("Training Neural Network (MLP)...")
    print("-" * 50)
    print("Note: Neural network uses SCALED data (preprocessor_nn)")

    start_time = time.time()

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),  # Two hidden layers
        activation='relu',             # ReLU activation function
        solver='adam',                 # Adam optimizer (adaptive learning)
        alpha=0.001,                   # L2 regularization
        learning_rate_init=0.001,      # Initial learning rate
        max_iter=500,                  # Maximum epochs
        early_stopping=True,           # Stop if no improvement
        validation_fraction=0.1,       # 10% for validation
        n_iter_no_change=20,           # Stop after 20 epochs no improvement
        random_state=random_state,
        verbose=False                  # Set True to see training progress
    )

    model.fit(X_train, y_train)

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Number of iterations (epochs): {model.n_iter_}")

    # Check if early stopping kicked in
    if model.n_iter_ < 500:
        print(f"Early stopping activated (stopped before max_iter)")

    return model


# =============================================================================
# CONVENIENCE FUNCTION: TRAIN ALL MODELS
# =============================================================================
def train_all_models(X_train_trees: np.ndarray, X_train_nn: np.ndarray,
                     y_train: np.ndarray, random_state: int = 42) -> dict:
    """
    Train all 4 models and return them in a dictionary.

    Parameters
    ----------
    X_train_trees : np.ndarray
        Training features WITHOUT scaling (for tree models)
    X_train_nn : np.ndarray
        Training features WITH scaling (for neural network)
    y_train : np.ndarray
        Training labels

    Returns
    -------
    dict
        Dictionary with model names as keys and trained models as values
    """
    print("\n" + "=" * 60)
    print("TRAINING ALL MODELS")
    print("=" * 60)

    models = {}

    # Train tree-based models (no scaling needed)
    models['Decision Tree'] = train_decision_tree(X_train_trees, y_train, random_state)
    models['Random Forest'] = train_random_forest(X_train_trees, y_train, random_state)
    models['XGBoost'] = train_xgboost(X_train_trees, y_train, random_state)

    # Train neural network (needs scaled data!)
    models['Neural Network'] = train_neural_network(X_train_nn, y_train, random_state)

    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("=" * 60)

    return models


# =============================================================================
# TEST THE MODULE
# =============================================================================
if __name__ == "__main__":
    # Quick test when running this file directly
    from data_loader import load_data
    from preprocessing import preprocess_data

    print("Testing models module...\n")

    # Load and preprocess data
    df = load_data("data/online_shoppers_intention.csv")
    data = preprocess_data(df)

    # Train all models
    models = train_all_models(
        X_train_trees=data['X_train_trees'],
        X_train_nn=data['X_train_nn'],
        y_train=data['y_train']
    )

    print("\nTrained models:")
    for name, model in models.items():
        print(f"  - {name}: {type(model).__name__}")
