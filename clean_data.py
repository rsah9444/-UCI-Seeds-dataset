# ============================================
# clean_data.py
# Utility functions for cleaning datasets
# ============================================

import pandas as pd

def check_missing_values(df):
    """
    Returns the count of missing values per column.
    """
    if df is None:
        print("ERROR: The dataset is empty.")
        return None

    print("\n--- Missing Values Per Column ---")
    missing = df.isnull().sum()
    print(missing)
    return missing


def handle_missing_values(df, method="drop", fill_value=None):
    """
    Cleans missing values using a selected method.

    Parameters:
        df (pd.DataFrame): Input dataset
        method (str): "drop", "mean", "median", "mode", "fill"
        fill_value: Any constant value (used only if method="fill")

    Returns:
        pd.DataFrame: Cleaned dataset
    """

    if df is None:
        print("ERROR: No dataset provided.")
        return None

    print(f"\nApplying missing-value handling method: {method}")

    if method == "drop":
        df = df.dropna()

    elif method == "mean":
        df = df.fillna(df.mean())

    elif method == "median":
        df = df.fillna(df.median())

    elif method == "mode":
        df = df.fillna(df.mode().iloc[0])

    elif method == "fill":
        if fill_value is None:
            raise ValueError("You must provide fill_value when method='fill'")
        df = df.fillna(fill_value)

    else:
        raise ValueError("Invalid method. Use: drop, mean, median, mode, fill")

    print("Missing values handled successfully.")
    return df


def remove_duplicates(df):
    """
    Removes duplicate rows from DataFrame.
    """
    if df is None:
        print("ERROR: No dataset provided.")
        return None

    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]

    print(f"\nRemoved {before - after} duplicate rows.")
    return df
