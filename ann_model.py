# ============================================
# ann_model.py
# Build, train, and evaluate an ANN (MLP)
# ============================================

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def build_ann(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42
):
    """
    Builds an MLP (feed-forward artificial neural network) classifier.

    Parameters:
        hidden_layer_sizes (tuple): Neurons in each hidden layer.
        activation (str): 'identity', 'logistic', 'tanh', 'relu'.
        solver (str): 'lbfgs', 'sgd', 'adam'.
        alpha (float): L2 penalty (regularization term).
        learning_rate_init (float): Initial learning rate.
        max_iter (int): Max number of training iterations.
        random_state (int): Seed for reproducibility.

    Returns:
        model (MLPClassifier): Untrained ANN model.
    """
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state
    )
    return model


def train_ann(model, X_train, y_train):
    """
    Trains the ANN model on the training data.

    Parameters:
        model (MLPClassifier): The ANN model.
        X_train (array-like): Training features.
        y_train (array-like): Training labels.

    Returns:
        model: Trained model.
    """
    model.fit(X_train, y_train)
    print("\nModel training completed.")
    return model


def evaluate_ann(model, X_train, y_train, X_test, y_test, show_train=True):
    """
    Evaluates the trained ANN model on train and test sets.

    Parameters:
        model (MLPClassifier): Trained model.
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        show_train (bool): If True, also prints training accuracy.

    Returns:
        metrics (dict): Dictionary with accuracy scores and confusion matrix.
                        Keys: 'train_accuracy', 'test_accuracy', 'confusion_matrix'
    """
    metrics = {}

    # Train accuracy (optional)
    if show_train:
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        metrics["train_accuracy"] = train_acc
        print(f"\nTraining Accuracy: {train_acc:.4f}")
    else:
        metrics["train_accuracy"] = None

    # Test accuracy
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    metrics["test_accuracy"] = test_acc
    print(f"Test Accuracy: {test_acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    metrics["confusion_matrix"] = cm

    print("\nConfusion Matrix:")
    print(cm)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    return metrics


def summarize_variant(name, params, metrics):
    """
    Creates a summary row (dict) for a model variant, combining
    hyperparameters and performance metrics.

    Parameters:
        name (str): Label for the variant (e.g., "Variant A").
        params (dict): Hyperparameters used to build the model.
        metrics (dict): Output from evaluate_ann().

    Returns:
        dict: Row for comparison table.
    """
    return {
        "Variant": name,
        "Hidden Layers": params.get("hidden_layer_sizes"),
        "Activation": params.get("activation"),
        "Solver": params.get("solver"),
        "Alpha": params.get("alpha"),
        "Learning Rate": params.get("learning_rate_init"),
        "Max Iter": params.get("max_iter"),
        "Train Accuracy": metrics.get("train_accuracy"),
        "Test Accuracy": metrics.get("test_accuracy"),
    }
