import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score


def evaluate_feature_subset(
    binary_mask,
    X_train,
    X_test,
    y_train,
    y_test,
    classifier
):
    selected_features = np.where(binary_mask == 1)[0]

    if len(selected_features) == 0:
        return 1.0

    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    try:
        model = clone(classifier)
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)

        accuracy = accuracy_score(y_test, y_pred)

        # نفس الورقة: fitness = error = 1 - accuracy
        fitness = 1.0 - accuracy

        return fitness

    except Exception:
        return 1.0