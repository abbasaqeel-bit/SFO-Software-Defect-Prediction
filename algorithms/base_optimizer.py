from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """
    Common interface for all optimization algorithms.

    Every optimizer must return:
        best_position
        best_fitness
        convergence_curve
        accuracy_curve
    """

    def __init__(
        self,
        objective_function,
        dim,
        population_size=10,
        max_iter=200,
        lb=-6,
        ub=6,
        random_state=None
    ):
        self.objective_function = objective_function
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.random_state = random_state

    @abstractmethod
    def optimize(self):
        pass