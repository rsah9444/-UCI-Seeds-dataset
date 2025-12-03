import pandas as pd
import os

from load_data import load_dataset
from data_info import dataset_info
from eda import (
    basic_eda, plot_histograms, plot_boxplots,
    plot_correlation_heatmap, plot_pairplot
)
from clean_data import check_missing_values, handle_missing_values, remove_duplicates
from preprocess_data import preprocess_data

# scikit-learn ANN
from ann_model import build_ann, train_ann, evaluate_ann, summarize_variant

# Keras ANN
from keras_ann import (
    build_keras_ann,
    train_keras_ann,
    evaluate_on_test,
    plot_history_variant,
    plot_accuracy_two_variants
)

# Plots for scikit-learn models
from evaluation_plots import (
    plot_confusion_matrix,
    plot_learning_curve,
    plot_multiclass_roc
)

from sklearn.model_selection import train_test_split

# ============================================
# 1. Load dataset
# ============================================

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

df = load_dataset(url, columns)

# 1) Basic EDA
basic_eda(df, target_col="class")
plot_histograms(df)
plot_boxplots(df)
plot_correlation_heatmap(df)
plot_pairplot(df, hue="class")

# Check missing values
check_missing_values(df)

# Handle missing data using mean or drop
df = handle_missing_values(df, method="mean")
df = remove_duplicates(df)

# Data information
dataset_info(df)

# ============================================
# 2. Preprocess (train/test split + scaling)
# ============================================

X_train, X_test, y_train, y_test, scaler = preprocess_data(
    df,
    target_col="class",
    test_size=0.2,
    random_state=42,
    scale_features=True,
    return_scaler=True,
    as_numpy=True  # good for ANN
)

# ============================================
# 3. scikit-learn ANN Variant A & B (extra analysis)
# ============================================

variant_A_params = {
    "hidden_layer_sizes": (16, 8),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "learning_rate_init": 0.001,
    "max_iter": 500,
    "random_state": 42
}

sk_model_A = build_ann(**variant_A_params)
sk_model_A = train_ann(sk_model_A, X_train, y_train)
metrics_A_skl = evaluate_ann(sk_model_A, X_train, y_train, X_test, y_test)

variant_B_params = {
    "hidden_layer_sizes": (32, 16, 8),  # deeper network
    "activation": "tanh",
    "solver": "adam",
    "alpha": 0.001,                     # stronger regularization
    "learning_rate_init": 0.0005,       # smaller learning rate
    "max_iter": 700,
    "random_state": 42
}

sk_model_B = build_ann(**variant_B_params)
sk_model_B = train_ann(sk_model_B, X_train, y_train)
metrics_B_skl = evaluate_ann(sk_model_B, X_train, y_train, X_test, y_test)

# Hyperparameter & Performance comparison (scikit-learn)
row_A = summarize_variant("Variant A (Baseline, sklearn)", variant_A_params, metrics_A_skl)
row_B = summarize_variant("Variant B (Deeper + Tanh, sklearn)", variant_B_params, metrics_B_skl)

comparison_df = pd.DataFrame([row_A, row_B])
print("\n=== Hyperparameter & Performance Comparison (scikit-learn) ===")
print(comparison_df)

results_folder = "results"
os.makedirs(results_folder, exist_ok=True)
comparison_path = os.path.join(results_folder, "ann_variants_comparison_sklearn.csv")
comparison_df.to_csv(comparison_path, index=False)
print(f"\nComparison table saved at: {comparison_path}")

# ============================================
# 4. Keras ANN – main part for the assignment
# ============================================

# Keras: labels must be 0,1,2 → our classes are 1,2,3 so subtract 1
y_train_keras = y_train - 1
y_test_keras = y_test - 1

# Split training data into train/validation (for curves)
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train,
    y_train_keras,
    test_size=0.2,
    random_state=42,
    stratify=y_train_keras
)

# ----- Keras Variant A (Dropout 0.3) -----
keras_model_A = build_keras_ann(
    units1=32,
    units2=16,
    dropout_rate=0.3,     # lower dropout
    learning_rate=0.001
)

history_A = train_keras_ann(
    keras_model_A,
    X_train_sub,
    y_train_sub,
    X_val,
    y_val,
    epochs=80,
    batch_size=16
)

# ----- Keras Variant B (Dropout 0.6) -----
keras_model_B = build_keras_ann(
    units1=32,
    units2=16,
    dropout_rate=0.6,     # higher dropout, more regularization
    learning_rate=0.001
)

history_B = train_keras_ann(
    keras_model_B,
    X_train_sub,
    y_train_sub,
    X_val,
    y_val,
    epochs=80,
    batch_size=16
)

# Plot individual histories (optional)
plot_history_variant(history_A, title_prefix="Keras Variant A (Dropout 0.3)")
plot_history_variant(history_B, title_prefix="Keras Variant B (Dropout 0.6)")

# Shared training & validation accuracy curves – assignment requirement
plot_accuracy_two_variants(
    history_A,
    history_B,
    label_A="Keras Variant A (Dropout 0.3)",
    label_B="Keras Variant B (Dropout 0.6)"
)

# ============================================
# 5. Final Evaluation (Keras – for assignment)
# ============================================

metrics_A_keras, y_pred_A = evaluate_on_test(keras_model_A, X_test, y_test_keras)
metrics_B_keras, y_pred_B = evaluate_on_test(keras_model_B, X_test, y_test_keras)

# Decide best KERAS variant by test accuracy
best_keras_model = keras_model_A if metrics_A_keras["accuracy"] >= metrics_B_keras["accuracy"] else keras_model_B
best_keras_name = "Keras Variant A (Dropout 0.3)" if best_keras_model is keras_model_A else "Keras Variant B (Dropout 0.6)"
best_keras_metrics = metrics_A_keras if best_keras_model is keras_model_A else metrics_B_keras

print(f"\nBest Keras variant on test set: {best_keras_name}")
print("Confusion matrix for best Keras variant (labels 0,1,2):")
print(best_keras_metrics["confusion_matrix"])

# ============================================
# 6. Visual Evaluation for sklearn model (extra, optional)
# ============================================

# Choose better sklearn model for evaluation_plots
better_skl_model = sk_model_A if metrics_A_skl["test_accuracy"] >= metrics_B_skl["test_accuracy"] else sk_model_B
better_skl_name = "sklearn Variant A" if better_skl_model is sk_model_A else "sklearn Variant B"
print(f"\nUsing {better_skl_name} for sklearn visual evaluation plots.\n")

class_names = ["Class 1", "Class 2", "Class 3"]

# These functions expect sklearn-style predict / predict_proba
plot_confusion_matrix(better_skl_model, X_test, y_test, class_names=class_names)
plot_learning_curve(better_skl_model, df.drop(columns=["class"]).values, df["class"].values)
plot_multiclass_roc(better_skl_model, X_test, y_test, class_names=class_names)
