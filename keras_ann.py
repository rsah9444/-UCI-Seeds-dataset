# ============================================
# keras_ann.py
# Keras-based ANN for UCI Seeds dataset
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_keras_ann(units1=32, units2=16, dropout_rate=0.3, learning_rate=0.001):
    """
    Build a feed-forward ANN with:
    - Input: 7 features
    - 2 hidden layers (ReLU)
    - Dropout for regularization
    - Output: 3 classes with softmax
    """
    model = Sequential([
        Dense(units1, activation="relu", input_shape=(7,)),
        Dropout(dropout_rate),
        Dense(units2, activation="relu"),
        Dropout(dropout_rate),
        Dense(3, activation="softmax")
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_keras_ann(model, X_train, y_train, X_val, y_val, epochs=80, batch_size=16):
    """
    Train the Keras ANN and return the history object.
    """
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history


def evaluate_on_test(model, X_test, y_test):
    """
    Evaluate model on test set and return metrics dict.
    """
    # Get predicted class probabilities and labels
    y_proba = model.predict(X_test)
    y_pred = np.argmax(y_proba, axis=1)

    # Because we subtract 1 from labels (see main code), y_test is 0,1,2 already
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[0, 1, 2]
    )

    metrics = {
        "accuracy": acc,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    print("\n=== Test Set Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report (per class):")
    print(classification_report(y_test, y_pred, labels=[0, 1, 2]))

    return metrics, y_pred


def plot_history_variant(history, title_prefix="Variant"):
    """
    Plot training & validation accuracy/loss curves for a single variant.
    """
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Train Acc")
    plt.plot(epochs, val_acc, label="Val Acc")
    plt.title(f"{title_prefix} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.title(f"{title_prefix} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_accuracy_two_variants(history_A, history_B, label_A="Variant A", label_B="Variant B"):
    """
    Plot training & validation accuracy for two variants on shared axes
    (as required by the assignment).
    """
    acc_A = history_A.history["accuracy"]
    val_acc_A = history_A.history["val_accuracy"]
    acc_B = history_B.history["accuracy"]
    val_acc_B = history_B.history["val_accuracy"]

    epochs = range(1, len(acc_A) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, acc_A, "--", label=f"{label_A} Train Acc")
    plt.plot(epochs, val_acc_A, "-", label=f"{label_A} Val Acc")
    plt.plot(epochs, acc_B, "--", label=f"{label_B} Train Acc")
    plt.plot(epochs, val_acc_B, "-", label=f"{label_B} Val Acc")

    plt.title("Training & Validation Accuracy – Two Variants")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
