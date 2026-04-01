"""
Data Loading Module
====================
This module handles loading and initial inspection of the dataset.

Why a separate module?
- Keeps data loading logic reusable
- Easy to modify if data source changes
- Clean separation of concerns
"""

import pandas as pd
from pathlib import Path


def load_data(filepath: str = "data/online_shoppers_intention.csv") -> pd.DataFrame:
    """
    Load the Online Shoppers Intention dataset.

    Parameters
    ----------
    filepath : str
        Path to the CSV file

    Returns
    -------
    pd.DataFrame
        The loaded dataset
    """
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def get_basic_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.

    This function provides a quick overview without printing everything,
    making it useful for both notebooks and scripts.

    Returns
    -------
    dict
        Dictionary with dataset statistics
    """
    info = {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": df.isnull().sum().sum(),
    }
    return info


def identify_column_types(df: pd.DataFrame) -> dict:
    """
    Identify numerical, categorical, and boolean columns.

    Understanding column types is crucial for preprocessing:
    - Numerical: may need scaling for neural networks
    - Categorical: need encoding (label or one-hot)
    - Boolean: special case of categorical (True/False)

    Returns
    -------
    dict
        Dictionary with lists of column names by type
    """
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    boolean_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    return {
        "numerical": numerical_cols,
        "categorical": categorical_cols,
        "boolean": boolean_cols,
    }


if __name__ == "__main__":
    # Quick test when running this file directly
    df = load_data()
    info = get_basic_info(df)
    print(f"\nTotal missing values: {info['total_missing']}")

    col_types = identify_column_types(df)
    print(f"\nNumerical columns ({len(col_types['numerical'])}): {col_types['numerical']}")
    print(f"Categorical columns ({len(col_types['categorical'])}): {col_types['categorical']}")
    print(f"Boolean columns ({len(col_types['boolean'])}): {col_types['boolean']}")
