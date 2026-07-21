import numpy as np

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    confusion_matrix
)


def calculate_metrics(
    classifier,
    X_train_selected,
    X_test_selected,
    y_train,
    y_test
):
    """
    Train the classifier on the outer training set and evaluate it
    once on the independent test set.
    """

    model = clone(classifier)

    model.fit(
        X_train_selected,
        y_train
    )

    y_pred = model.predict(
        X_test_selected
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # Metrics for the defective class only: class = 1
    precision_defective = precision_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    recall_defective = recall_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    f1_defective = f1_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    # Weighted metrics for overall dataset performance
    precision_weighted = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "Accuracy": accuracy * 100,

        "Precision_Defective": precision_defective * 100,
        "Recall_Defective": recall_defective * 100,
        "F1_Defective": f1_defective * 100,

        "Precision_Weighted": precision_weighted * 100,
        "Recall_Weighted": recall_weighted * 100,
        "F1_Weighted": f1_weighted * 100,

        "RMSE": rmse,

        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),

        "Confusion_Matrix": cm.tolist()
    }