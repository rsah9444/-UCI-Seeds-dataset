# ============================================
# eda.py
# Exploratory Data Analysis utilities
# Saves all visualizations
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Create folder for saving plots
def _make_dir(path="eda_plots"):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def basic_eda(df, target_col=None):
    """
    Prints basic EDA info:
    - head()
    - shape
    - dtypes
    - describe()
    - target distribution (if target_col is given)
    """
    if df is None:
        print("ERROR: DataFrame is None.")
        return

    print("====== BASIC EDA ======\n")

    print("--- First 5 Rows ---")
    print(df.head())

    print("\n--- Shape ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- Summary Statistics ---")
    print(df.describe())

    if target_col is not None and target_col in df.columns:
        print(f"\n--- Target Distribution ({target_col}) ---")
        print(df[target_col].value_counts())

    print("\n=======================\n")


def plot_histograms(df, cols=None, bins=20, save=True):
    """
    Plots histograms for numeric columns.
    Also saves them.
    """
    if df is None:
        print("ERROR: DataFrame is None.")
        return

    if cols is None:
        cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    folder = _make_dir()

    df[cols].hist(bins=bins, figsize=(15, 10))
    plt.suptitle("Feature Distributions (Histograms)")
    plt.tight_layout()

    if save:
        path = os.path.join(folder, "histograms.png")
        plt.savefig(path)
        print(f"Histogram saved at: {path}")

    plt.show()


def plot_boxplots(df, cols=None, save=True):
    """
    Plots boxplots for numeric columns (outlier detection).
    Saves output.
    """
    if df is None:
        print("ERROR: DataFrame is None.")
        return

    if cols is None:
        cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    folder = _make_dir()

    plt.figure(figsize=(12, 6))
    df[cols].boxplot()
    plt.title("Boxplots of Numeric Features")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        path = os.path.join(folder, "boxplots.png")
        plt.savefig(path)
        print(f"Boxplot saved at: {path}")

    plt.show()


def plot_correlation_heatmap(df, save=True):
    """
    Plots correlation heatmap for numeric features.
    Saves output.
    """
    if df is None:
        print("ERROR: DataFrame is None.")
        return

    folder = _make_dir()

    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()

    if save:
        path = os.path.join(folder, "correlation_heatmap.png")
        plt.savefig(path)
        print(f"Correlation heatmap saved at: {path}")

    plt.show()


def plot_pairplot(df, cols=None, hue=None, save=True):
    """
    Plots and saves pairplot.
    """
    if df is None:
        print("ERROR: DataFrame is None.")
        return

    folder = _make_dir()

    if cols is None:
        cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cols = cols[:6]  # To avoid huge grids

    g = sns.pairplot(df[cols + ([hue] if hue and hue in df.columns else [])],
                     hue=hue, diag_kind="hist")

    g.fig.suptitle("Pairplot of Features", y=1.02)

    if save:
        path = os.path.join(folder, "pairplot.png")
        g.savefig(path)
        print(f"Pairplot saved at: {path}")

    plt.show()
