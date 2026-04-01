"""
Preprocessing Module
====================
This module handles all data preprocessing steps before training ML models.

WHAT IS PREPROCESSING?
----------------------
Raw data isn't ready for ML models. We need to:
1. Separate features (X) from target (y)
2. Split data into train/test sets
3. Convert text categories to numbers
4. Scale numerical values (for neural networks)

WHY IS THIS A SEPARATE MODULE?
------------------------------
- Keeps preprocessing logic organized and reusable
- Makes it easy to apply the SAME transformations to new data
- Helps avoid data leakage (explained below)

WHAT IS DATA LEAKAGE?
---------------------
Data leakage happens when information from the test set "leaks" into training.

Bad example:
    scaler.fit(ALL_DATA)  # Learns from test data too!
    scaler.transform(train_data)
    scaler.transform(test_data)

Good example:
    scaler.fit(train_data)  # Only learns from training data
    scaler.transform(train_data)
    scaler.transform(test_data)  # Uses parameters from training only

We use sklearn Pipelines to handle this automatically and correctly.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# =============================================================================
# STEP 1: DEFINE COLUMN NAMES BY TYPE
# =============================================================================
# We need to know which columns are which type so we can process them correctly.
# These are based on what we learned in EDA.

# Columns with continuous numbers (durations, rates, counts)
NUMERICAL_COLUMNS = [
    'Administrative', 'Administrative_Duration',
    'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration',
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
    'OperatingSystems', 'Browser', 'Region', 'TrafficType'
]

# Columns with text categories that need encoding
CATEGORICAL_COLUMNS = ['Month', 'VisitorType']

# Columns with True/False values
BOOLEAN_COLUMNS = ['Weekend']

# The column we're trying to predict
TARGET_COLUMN = 'Revenue'


# =============================================================================
# STEP 2: FUNCTION TO SEPARATE FEATURES AND TARGET
# =============================================================================
def separate_features_target(df: pd.DataFrame) -> tuple:
    """
    Separate the dataset into features (X) and target (y).

    WHAT ARE FEATURES AND TARGET?
    -----------------------------
    - Features (X): The input data the model uses to make predictions
                    (all columns except Revenue)
    - Target (y):   The output we want to predict (Revenue: True/False)

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset

    Returns
    -------
    X : pd.DataFrame
        Features (everything except the target)
    y : pd.Series
        Target variable (Revenue)
    """
    # Make a copy to avoid modifying the original data
    df = df.copy()

    # Convert boolean 'Weekend' column to integers (0 and 1)
    # Why? Some ML models work better with numbers than True/False
    df['Weekend'] = df['Weekend'].astype(int)

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])  # Everything except Revenue
    y = df[TARGET_COLUMN].astype(int)     # Just Revenue (as 0/1)

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    return X, y


# =============================================================================
# STEP 3: FUNCTION TO SPLIT DATA INTO TRAIN AND TEST SETS
# =============================================================================
def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
               random_state: int = 42) -> tuple:
    """
    Split data into training and test sets.

    WHY DO WE SPLIT DATA?
    ---------------------
    - Training set: Model learns patterns from this data
    - Test set: We evaluate model performance on data it has NEVER seen

    This tells us how well the model will work on new, real-world data.
    If we tested on training data, the model might just memorize answers
    (this is called "overfitting").

    WHY STRATIFIED SPLIT?
    ---------------------
    Our data is imbalanced (84.5% No Purchase, 15.5% Purchase).

    Regular split might give us:
        Train: 86% No Purchase, 14% Purchase
        Test:  80% No Purchase, 20% Purchase

    Stratified split ensures:
        Train: 84.5% No Purchase, 15.5% Purchase  (same as original)
        Test:  84.5% No Purchase, 15.5% Purchase  (same as original)

    This gives us a fair evaluation.

    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    test_size : float
        Proportion of data for testing (0.2 = 20%)
    random_state : int
        Random seed for reproducibility (same split every time)

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        Split datasets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # THIS IS KEY: maintains class proportions
    )

    print(f"\nData split complete:")
    print(f"  Training set: {X_train.shape[0]} samples ({100*(1-test_size):.0f}%)")
    print(f"  Test set:     {X_test.shape[0]} samples ({100*test_size:.0f}%)")

    # Verify stratification worked
    train_positive_rate = y_train.mean() * 100
    test_positive_rate = y_test.mean() * 100
    print(f"\nClass balance verification:")
    print(f"  Training set - Purchase rate: {train_positive_rate:.1f}%")
    print(f"  Test set     - Purchase rate: {test_positive_rate:.1f}%")

    return X_train, X_test, y_train, y_test


# =============================================================================
# STEP 4: CREATE PREPROCESSING PIPELINES
# =============================================================================
def create_preprocessor(scale_numerical: bool = False) -> ColumnTransformer:
    """
    Create a preprocessing pipeline using sklearn's ColumnTransformer.

    WHAT IS A PIPELINE?
    -------------------
    A pipeline chains multiple preprocessing steps together.
    It ensures transformations are applied consistently and prevents data leakage.

    WHAT DOES ColumnTransformer DO?
    -------------------------------
    It applies different transformations to different column types:
    - Numerical columns: optionally scale (for neural networks)
    - Categorical columns: one-hot encode (convert text to numbers)

    WHY THE scale_numerical PARAMETER?
    ----------------------------------
    - Tree models (DecisionTree, RandomForest, XGBoost): DON'T need scaling
      They make decisions based on thresholds, so scale doesn't matter.
      Example: "Is age > 30?" works the same whether age is 30 or 0.5 (scaled)

    - Neural Networks: NEED scaling
      They use gradient descent which works better when values are similar scale.
      If one feature ranges 0-1 and another 0-10000, the large one dominates.

    ONE-HOT ENCODING EXPLAINED
    --------------------------
    Categorical data like 'Month' can't be used directly.

    Before (1 column):
        Month
        -----
        Feb
        Mar
        May

    After one-hot encoding (10 columns, one per month):
        Month_Feb  Month_Mar  Month_May  ...
        ---------  ---------  ---------
        1          0          0
        0          1          0
        0          0          1

    Parameters
    ----------
    scale_numerical : bool
        If True, scale numerical columns (use for neural networks)
        If False, keep numerical columns as-is (use for tree models)

    Returns
    -------
    ColumnTransformer
        Preprocessing pipeline ready to fit and transform data
    """
    # Define transformations for each column type
    transformers = []

    # Numerical columns transformation
    if scale_numerical:
        # StandardScaler transforms data to have mean=0 and std=1
        # Formula: z = (x - mean) / std
        # Example: if mean=50, std=10, then value 60 becomes (60-50)/10 = 1.0
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        print("Creating preprocessor WITH scaling (for neural networks)")
    else:
        # 'passthrough' means: don't change these columns at all
        numerical_transformer = 'passthrough'
        print("Creating preprocessor WITHOUT scaling (for tree models)")

    transformers.append(('numerical', numerical_transformer, NUMERICAL_COLUMNS))

    # Categorical columns transformation (always one-hot encode)
    # handle_unknown='ignore': if test data has a category not seen in training,
    #                          it will be encoded as all zeros instead of error
    # sparse_output=False: return regular array instead of sparse matrix
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    transformers.append(('categorical', categorical_transformer, CATEGORICAL_COLUMNS))

    # Boolean columns (Weekend) - already converted to 0/1, just pass through
    transformers.append(('boolean', 'passthrough', BOOLEAN_COLUMNS))

    # Combine all transformers
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop any columns not explicitly listed
    )

    return preprocessor


# =============================================================================
# STEP 5: MAIN PREPROCESSING FUNCTION
# =============================================================================
def preprocess_data(df: pd.DataFrame, scale_for_nn: bool = False) -> dict:
    """
    Complete preprocessing pipeline: load → split → transform.

    This is the main function you'll call from main.py.
    It returns everything you need for model training.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset
    scale_for_nn : bool
        If True, also return scaled data for neural networks

    Returns
    -------
    dict
        Dictionary containing all preprocessed data and fitted preprocessors
    """
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Separate features and target
    print("\n[1/4] Separating features and target...")
    X, y = separate_features_target(df)

    # Step 2: Split into train/test
    print("\n[2/4] Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 3: Create and fit preprocessor for TREE MODELS (no scaling)
    print("\n[3/4] Creating preprocessor for tree models...")
    preprocessor_trees = create_preprocessor(scale_numerical=False)

    # fit_transform on training data: learn parameters AND transform
    # transform on test data: apply same parameters (no learning)
    X_train_trees = preprocessor_trees.fit_transform(X_train)
    X_test_trees = preprocessor_trees.transform(X_test)

    print(f"  Transformed shape: {X_train_trees.shape}")

    # Step 4: Create and fit preprocessor for NEURAL NETWORK (with scaling)
    print("\n[4/4] Creating preprocessor for neural network...")
    preprocessor_nn = create_preprocessor(scale_numerical=True)

    X_train_nn = preprocessor_nn.fit_transform(X_train)
    X_test_nn = preprocessor_nn.transform(X_test)

    print(f"  Transformed shape: {X_train_nn.shape}")

    # Get feature names after transformation (useful for analysis)
    feature_names = get_feature_names(preprocessor_trees)
    print(f"\nTotal features after preprocessing: {len(feature_names)}")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)

    # Return everything in a dictionary for easy access
    return {
        # For tree models (no scaling)
        'X_train_trees': X_train_trees,
        'X_test_trees': X_test_trees,

        # For neural network (with scaling)
        'X_train_nn': X_train_nn,
        'X_test_nn': X_test_nn,

        # Labels (same for all models)
        'y_train': y_train,
        'y_test': y_test,

        # Original data (for reference)
        'X_train_original': X_train,
        'X_test_original': X_test,

        # Fitted preprocessors (can save for production use)
        'preprocessor_trees': preprocessor_trees,
        'preprocessor_nn': preprocessor_nn,

        # Feature names
        'feature_names': feature_names
    }


# =============================================================================
# HELPER FUNCTION: GET FEATURE NAMES
# =============================================================================
def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Extract feature names from a fitted ColumnTransformer.

    After one-hot encoding, we go from 17 features to ~27 features.
    This function tells us the name of each feature.

    Returns
    -------
    list
        List of feature names in order
    """
    feature_names = []

    # Get names from each transformer
    for name, transformer, columns in preprocessor.transformers_:
        if name == 'remainder':
            continue
        if transformer == 'passthrough':
            feature_names.extend(columns)
        elif hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out())
        elif hasattr(transformer, 'named_steps'):
            # Pipeline - get names from the last step
            last_step = list(transformer.named_steps.values())[-1]
            if hasattr(last_step, 'get_feature_names_out'):
                names = last_step.get_feature_names_out(columns)
                feature_names.extend(names)
            else:
                feature_names.extend(columns)
        else:
            feature_names.extend(columns)

    return feature_names


# =============================================================================
# TEST THE MODULE
# =============================================================================
if __name__ == "__main__":
    # Quick test when running this file directly
    from data_loader import load_data

    print("Testing preprocessing module...\n")

    # Load data
    df = load_data("../data/online_shoppers_intention.csv")

    # Run preprocessing
    data = preprocess_data(df)

    # Show summary
    print("\n" + "=" * 60)
    print("SUMMARY OF PREPROCESSED DATA")
    print("=" * 60)
    print(f"\nTraining samples: {data['X_train_trees'].shape[0]}")
    print(f"Test samples: {data['X_test_trees'].shape[0]}")
    print(f"Features (after encoding): {data['X_train_trees'].shape[1]}")

    print(f"\nFeature names ({len(data['feature_names'])}):")
    for i, name in enumerate(data['feature_names']):
        print(f"  {i+1:2d}. {name}")
