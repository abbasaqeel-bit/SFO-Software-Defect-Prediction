import numpy as np
from mealpy import FloatVar


class MealpyOptimizerAdapter:
    """
    Adapter between MEALPY optimizers and the project's
    generic feature-selection wrapper.

    Expected output:
        best_position
        best_fitness
        convergence_curve
        accuracy_curve
    """

    def __init__(
        self,
        objective_function,
        dim,
        mealpy_optimizer_class,
        mealpy_parameters=None,
        population_size=10,
        max_iter=200,
        lb=-6,
        ub=6,
        random_state=None,
    ):
        self.objective_function = objective_function
        self.dim = int(dim)

        self.mealpy_optimizer_class = mealpy_optimizer_class
        self.mealpy_parameters = mealpy_parameters or {}

        self.population_size = int(population_size)
        self.max_iter = int(max_iter)

        self.lb = float(lb)
        self.ub = float(ub)
        self.random_state = random_state

    def optimize(self):
        problem = {
            "obj_func": self.objective_function,
            "bounds": FloatVar(
                lb=(self.lb,) * self.dim,
                ub=(self.ub,) * self.dim,
                name="feature_position",
            ),
            "minmax": "min",
            "log_to": None,
        }

        model = self.mealpy_optimizer_class(
            epoch=self.max_iter,
            pop_size=self.population_size,
            **self.mealpy_parameters,
        )

        best_agent = model.solve(
            problem=problem,
            seed=self.random_state,
        )

        best_position = np.asarray(
            best_agent.solution,
            dtype=float,
        )

        best_fitness = float(
            best_agent.target.fitness
        )

        convergence_curve = self._extract_convergence_curve(
            model=model,
            best_fitness=best_fitness,
        )

        accuracy_curve = [
            (1.0 - fitness) * 100.0
            for fitness in convergence_curve
        ]

        return {
            "best_position": best_position,
            "best_fitness": best_fitness,
            "best_accuracy": (1.0 - best_fitness) * 100.0,
            "convergence_curve": convergence_curve,
            "accuracy_curve": accuracy_curve,
        }

    def _extract_convergence_curve(self, model, best_fitness):
        """
        Extract global-best fitness history robustly across
        minor MEALPY history-format differences.
        """

        history = getattr(model, "history", None)

        if history is None:
            return [best_fitness] * self.max_iter

        curve = getattr(
            history,
            "list_global_best_fit",
            None,
        )

        if curve is not None and len(curve) > 0:
            curve = [
                float(value)
                for value in curve
            ]

        else:
            global_best_agents = getattr(
                history,
                "list_global_best",
                None,
            )

            if global_best_agents:
                curve = [
                    float(agent.target.fitness)
                    for agent in global_best_agents
                ]
            else:
                curve = [best_fitness] * self.max_iter

        # توحيد الطول مع عدد التكرارات
        if len(curve) > self.max_iter:
            curve = curve[-self.max_iter:]

        elif len(curve) < self.max_iter:
            last_value = (
                curve[-1]
                if curve
                else best_fitness
            )

            curve.extend(
                [last_value]
                * (self.max_iter - len(curve))
            )

        return curve