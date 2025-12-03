# ============================================
# evaluation_plots.py
# Visualization utilities for ANN performance
# ============================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve


def _make_dir(path="model_plots"):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def plot_confusion_matrix(model, X_test, y_test, class_names=None, save=True):
    """
    Plots and saves confusion matrix as heatmap.
    """
    folder = _make_dir()
    y_pred = model.predict(X_test)

    plt.figure(figsize=(6, 5))
    cm = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=class_names,
        cmap="Blues",
        colorbar=True
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save:
        path = os.path.join(folder, "confusion_matrix.png")
        plt.savefig(path)
        print(f"Confusion matrix saved at: {path}")

    plt.show()


def plot_learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.2, 1.0, 5), save=True):
    """
    Plots and saves learning curve (train & CV score vs. training size).
    """
    folder = _make_dir()

    train_sizes_abs, train_scores, test_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        train_sizes=train_sizes,
        scoring="accuracy",
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes_abs, train_mean, marker="o", label="Training score")
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2)

    plt.plot(train_sizes_abs, test_mean, marker="s", label="Cross-validation score")
    plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.2)

    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save:
        path = os.path.join(folder, "learning_curve.png")
        plt.savefig(path)
        print(f"Learning curve saved at: {path}")

    plt.show()


def plot_multiclass_roc(model, X_test, y_test, class_names=None, save=True):
    """
    Plots ROC curves for multiclass classification using one-vs-rest.
    """
    folder = _make_dir()

    # Binarize labels for ROC
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = y_test_bin.shape[1]

    # Predicted probabilities
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        # fallback: use decision_function if available
        if hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            raise ValueError("Model has neither predict_proba nor decision_function.")

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(7, 6))
    for i in range(n_classes):
        label = f"Class {classes[i]}" if class_names is None else f"{class_names[i]}"
        plt.plot(fpr[i], tpr[i], lw=2, label=f"{label} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    if save:
        path = os.path.join(folder, "roc_curves.png")
        plt.savefig(path)
        print(f"ROC curves saved at: {path}")

    plt.show()
