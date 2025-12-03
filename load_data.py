# ============================================
# Step 1: Data Loading Function
# ============================================

import pandas as pd
import numpy as np

def load_dataset(url, columns, sep=r"\s+"):
    """
    Loads a dataset from a given URL with specified columns and separator.
    
    Parameters:
        url (str): URL of the dataset
        columns (list): List of column names
        sep (str): Separator used in the dataset (default: whitespace)
    
    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame
    """
    try:
        df = pd.read_csv(url, sep=sep, header=None, names=columns)
        print("Data loaded successfully!")
        return df
    except FileNotFoundError:
        print("Data could not be loaded (FileNotFoundError)")
    except Exception as e:
        print(f"Data could not be loaded. Error: {e}")
        return None


# ==========================
# Example usage
# ==========================

# Dataset URL (UCI Seeds)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"

columns = [
    "area",
    "perimeter",
    "compactness",
    "length_of_kernel",
    "width_of_kernel",
    "asymmetry_coefficient",
    "length_of_kernel_groove",
    "class"
]

# df = load_dataset(url, columns)

# # Show first 5 rows
# if df is not None:
#     print(df.head())
