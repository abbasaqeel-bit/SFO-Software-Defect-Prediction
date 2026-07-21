import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error
)


def evaluate_fitness(binary_mask, X_train, X_test, y_train, y_test, classifier):
    selected_features = np.where(binary_mask == 1)[0]

    if len(selected_features) == 0:
        return 1.0, 0.0

    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    try:
        model = clone(classifier)
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)

        accuracy = accuracy_score(y_test, y_pred)
        fitness = 1.0 - accuracy

        return fitness, accuracy

    except Exception:
        return 1.0, 0.0


def calculate_metrics(binary_mask, X_train, X_test, y_train, y_test, classifier):
    selected_features = np.where(binary_mask == 1)[0]

    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    model = clone(classifier)
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1": f1 * 100,
        "RMSE": rmse,
        "Selected_Features": len(selected_features)
    }