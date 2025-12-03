# ============================================
# data_info.py
# Utility functions to inspect dataset
# ============================================

import pandas as pd

def dataset_info(df, show_head=True, show_shape=True, show_describe=True):
    """
    Prints basic information about the dataset:
    - head()
    - shape
    - describe()

    Parameters:
        df (pd.DataFrame): Input DataFrame
        show_head (bool): Print first 5 rows
        show_shape (bool): Print shape of dataset
        show_describe (bool): Print summary statistics
    """
    if df is None:
        print("ERROR: The dataset is empty or not loaded properly.")
        return

    print("====== DATASET INFORMATION ======")

    if show_head:
        print("\n--- First 5 Rows (head) ---")
        print(df.head())

    if show_shape:
        print("\n--- Dataset Shape ---")
        print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    if show_describe:
        print("\n--- Summary Statistics (describe) ---")
        print(df.describe(include='all'))

    print("\n=================================\n")
