# ============================================
# preprocess_data.py
# Train-test split + feature scaling
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(
    df,
    target_col="class",
    test_size=0.2,
    random_state=42,
    scale_features=True,
    return_scaler=False,
    as_numpy=True
):
    """
    Preprocesses the dataset for modeling:
    - Splits into features (X) and target (y)
    - Train-test split
    - Standardizes features (fit on train, transform train & test)

    Parameters:
        df (pd.DataFrame): Input dataset
        target_col (str): Name of target column
        test_size (float): Test set proportion (default: 0.2)
        random_state (int): Random seed for reproducibility
        scale_features (bool): Whether to apply StandardScaler
        return_scaler (bool): Whether to return the fitted scaler
        as_numpy (bool): Return X, y as numpy arrays if True; else as DataFrames/Series

    Returns:
        X_train, X_test, y_train, y_test  (and scaler if return_scaler=True)
    """

    if df is None:
        raise ValueError("DataFrame is None. Please provide a valid dataset.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # 1. Split into X and y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Train-test split (stratify to keep class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"\nTrain/Test Split Done:")
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}, y_test: {y_test.shape}")

    scaler = None

    # 3. Feature scaling
    if scale_features:
        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        print("\nFeature scaling applied using StandardScaler().")

        if not as_numpy:
            # Convert back to DataFrame with original column names
            X_train = pd.DataFrame(X_train, columns=X.columns)
            X_test = pd.DataFrame(X_test, columns=X.columns)

    else:
        # If not scaling and as_numpy is True, convert to numpy only
        if as_numpy:
            X_train = X_train.values
            X_test = X_test.values

    # y → numpy if requested
    if as_numpy:
        y_train = y_train.values
        y_test = y_test.values

    if return_scaler:
        return X_train, X_test, y_train, y_test, scaler
    else:
        return X_train, X_test, y_train, y_test
