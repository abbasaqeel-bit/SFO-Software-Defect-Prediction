import numpy as np

from algorithms.sfo import SFO
from algorithms.transfer import position_to_binary
from feature_selection.fitness import evaluate_feature_subset


class SFOFeatureSelectionWrapper:
    """
    Wrapper-based feature selection using the original SFO.

    SFO evaluates candidate feature subsets using:
        Inner training set
        Validation set

    The independent test set must not be passed to this class.
    """

    def __init__(
        self,
        classifier,
        population_size=10,
        max_iter=200,
        lb=-6,
        ub=6,
        random_state=None
    ):
        self.classifier = classifier
        self.population_size = population_size
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.random_state = random_state

        self.best_position = None
        self.best_mask = None
        self.best_fitness = None
        self.convergence_curve = None
        self.selected_features = None
        self.accuracy_curve = None

    def fit(
        self,
        X_inner_train,
        X_validation,
        y_inner_train,
        y_validation
    ):
        dim = X_inner_train.shape[1]

        def objective_function(position):
            binary_mask = position_to_binary(position)

            return evaluate_feature_subset(
                binary_mask=binary_mask,
                X_train=X_inner_train,
                X_test=X_validation,
                y_train=y_inner_train,
                y_test=y_validation,
                classifier=self.classifier
            )

        optimizer = SFO(
            objective_function=objective_function,
            dim=dim,
            population_size=self.population_size,
            max_iter=self.max_iter,
            lb=self.lb,
            ub=self.ub,
            random_state=self.random_state
        )

        result = optimizer.optimize()

        self.best_position = result["best_position"]
        self.best_mask = position_to_binary(self.best_position)
        self.best_fitness = result["best_fitness"]
        self.convergence_curve = result["convergence_curve"]
        self.accuracy_curve = result["accuracy_curve"]

        self.selected_features = np.flatnonzero(
            self.best_mask == 1
        )

        if len(self.selected_features) == 0:
            raise RuntimeError(
                "SFO returned an empty feature subset."
            )

        return self

    def transform(self, X):
        if self.selected_features is None:
            raise RuntimeError(
                "The wrapper must be fitted before calling transform()."
            )

        return X[:, self.selected_features]

    def fit_transform(
        self,
        X_inner_train,
        X_validation,
        y_inner_train,
        y_validation
    ):
        self.fit(
            X_inner_train,
            X_validation,
            y_inner_train,
            y_validation
        )

        return (
            self.transform(X_inner_train),
            self.transform(X_validation)
        )