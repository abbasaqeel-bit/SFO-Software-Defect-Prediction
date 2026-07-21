import numpy as np
from math import gamma, sin, pi
from algorithms.base_optimizer import BaseOptimizer

class SFO(BaseOptimizer):
    """
    Original Swift Flight Optimizer (SFO)
    General-purpose continuous optimizer.

    It does not know anything about:
    - Feature selection
    - Classifiers
    - Accuracy
    - Binary masks
    """

    def __init__(
        self,
        objective_function,
        dim,
        population_size=10,
        max_iter=200,
        lb=-6,
        ub=6,
        w=0.5,
        alpha=0.5,
        beta=0.57,
        levy_beta=1.5,
        psi1=1.5,
        psi2=1.5,
        epsilon_scale=0.01,
        glide_fail_limit=4,
        target_fail_limit=10,
        gtc_limit=30,
        stagnation_limit=20,
        max_resets=5,
        random_state=None,
    ):
        super().__init__(
            objective_function=objective_function,
            dim=dim,
            population_size=population_size,
            max_iter=max_iter,
            lb=lb,
            ub=ub,
            random_state=random_state
        )
        

        # SFO parameters
        self.w = w
        self.alpha = alpha
        self.beta = beta
        self.levy_beta = levy_beta
        self.psi1 = psi1
        self.psi2 = psi2
        self.epsilon_scale = epsilon_scale

        # Switching counters
        self.glide_fail_limit = glide_fail_limit
        self.target_fail_limit = target_fail_limit
        self.gtc_limit = gtc_limit
        self.stagnation_limit = stagnation_limit
        self.max_resets = max_resets

        self.rng = np.random.default_rng(random_state)

    # =====================================================
    # Initialization
    # =====================================================
    def initialize_population(self):
        positions = self.rng.uniform(
            self.lb, self.ub, size=(self.population_size, self.dim)
        )

        velocities = self.rng.uniform(
            -1, 1, size=(self.population_size, self.dim)
        )

        return positions, velocities

    # =====================================================
    # Eq. (2) and Eq. (3): Levy flight using Mantegna method
    # =====================================================
    def levy_flight(self):
        beta = self.levy_beta

        sigma_u = (
            gamma(1 + beta) * sin(pi * beta / 2)
            / (gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))
        ) ** (1 / beta)

        u = self.rng.normal(0, sigma_u, self.dim)
        v = self.rng.normal(0, 1, self.dim)

        L = u / (np.abs(v) ** (1 / beta))

        return L

    # =====================================================
    # Eq. (1a): Glide Mode velocity update
    # v(t+1) = w*v(t) + alpha*N(0,I) + beta*L
    # Eq. (1b): x(t+1) = x(t) + v(t+1)
    # =====================================================
    def update_glide(self, position, velocity):
        gaussian_noise = self.rng.normal(0, 1, self.dim)
        levy_step = self.levy_flight()

        new_velocity = (
            self.w * velocity
            + self.alpha * gaussian_noise
            + self.beta * levy_step
        )

        new_position = position + new_velocity

        return new_position, new_velocity

    # =====================================================
    # Eq. (4a): Target Mode velocity update
    # v(t+1) = w*v(t)
    #        + psi1*r1*(pbest - x)
    #        + psi2*r2*(gbest - x)
    # Eq. (4b): x(t+1) = x(t) + v(t+1)
    # =====================================================
    def update_target(self, position, velocity, pbest_position, gbest_position):
        r1 = self.rng.random(self.dim)
        r2 = self.rng.random(self.dim)

        new_velocity = (
            self.w * velocity
            + self.psi1 * r1 * (pbest_position - position)
            + self.psi2 * r2 * (gbest_position - position)
        )

        new_position = position + new_velocity

        return new_position, new_velocity

    # =====================================================
    # Eq. (5a): Micro Mode velocity update
    # v(t+1) = w*v(t) + psi2*r2*(gbest - x) + epsilon
    # Eq. (5b): x(t+1) = x(t) + v(t+1)
    # =====================================================
    def update_micro(self, position, velocity, gbest_position):
        r2 = self.rng.random(self.dim)
        epsilon = self.epsilon_scale * self.rng.normal(0, 1, self.dim)

        new_velocity = (
            self.w * velocity
            + self.psi2 * r2 * (gbest_position - position)
            + epsilon
        )

        new_position = position + new_velocity

        return new_position, new_velocity

    # =====================================================
    # Boundary control
    # =====================================================
    def apply_bounds(self, position):
        return np.clip(position, self.lb, self.ub)

    # =====================================================
    # Main SFO optimization loop
    # =====================================================
    def optimize(self):
        positions, velocities = self.initialize_population()

        pbest_positions = positions.copy()
        pbest_fitness = np.full(self.population_size, np.inf)

        gbest_position = None
        gbest_fitness = np.inf

        modes = np.array(["glide"] * self.population_size)

        glide_fail_counter = np.zeros(self.population_size, dtype=int)
        target_fail_counter = np.zeros(self.population_size, dtype=int)

        glide_target_cycle_counter = 0
        stagnation_counter = 0
        reset_counter = 0

        convergence_curve = []
        accuracy_curve = []

        # Initial evaluation
        for i in range(self.population_size):
            fitness = self.objective_function(positions[i])

            pbest_fitness[i] = fitness
            pbest_positions[i] = positions[i].copy()

            if fitness < gbest_fitness:
                gbest_fitness = fitness
                gbest_position = positions[i].copy()

        # Main loop
        for iteration in range(self.max_iter):
            previous_global_best = gbest_fitness

            for i in range(self.population_size):
                old_pbest = pbest_fitness[i]

                if modes[i] == "glide":
                    new_position, new_velocity = self.update_glide(
                        positions[i], velocities[i]
                    )

                elif modes[i] == "target":
                    new_position, new_velocity = self.update_target(
                        positions[i],
                        velocities[i],
                        pbest_positions[i],
                        gbest_position,
                    )

                else:  # micro mode
                    new_position, new_velocity = self.update_micro(
                        positions[i],
                        velocities[i],
                        gbest_position,
                    )

                new_position = self.apply_bounds(new_position)

                fitness = self.objective_function(new_position)

                positions[i] = new_position
                velocities[i] = new_velocity

                # Update pbest
                if fitness < pbest_fitness[i]:
                    pbest_fitness[i] = fitness
                    pbest_positions[i] = new_position.copy()

                # Update gbest
                if fitness < gbest_fitness:
                    gbest_fitness = fitness
                    gbest_position = new_position.copy()

                improved = pbest_fitness[i] < old_pbest

                # =====================================================
                # Mode switching strategy
                # =====================================================
                if modes[i] == "glide":
                    if improved:
                        glide_fail_counter[i] = 0
                    else:
                        glide_fail_counter[i] += 1

                    if (
                        glide_fail_counter[i] > self.glide_fail_limit
                        and glide_target_cycle_counter < self.gtc_limit
                    ):
                        modes[i] = "target"
                        glide_fail_counter[i] = 0
                        glide_target_cycle_counter += 1

                elif modes[i] == "target":
                    if improved:
                        target_fail_counter[i] = 0
                    else:
                        target_fail_counter[i] += 1

                    if (
                        target_fail_counter[i] > self.target_fail_limit
                        and glide_target_cycle_counter < self.gtc_limit
                    ):
                        modes[i] = "glide"
                        target_fail_counter[i] = 0
                        glide_target_cycle_counter += 1

                if glide_target_cycle_counter >= self.gtc_limit:
                    modes[i] = "micro"

            # =====================================================
            # Stagnation-aware reinitialization
            # =====================================================
            if gbest_fitness < previous_global_best:
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if (
                stagnation_counter > self.stagnation_limit
                and reset_counter < self.max_resets
            ):
                positions, velocities = self.initialize_population()

                for i in range(self.population_size):
                    fitness = self.objective_function(positions[i])

                    pbest_fitness[i] = fitness
                    pbest_positions[i] = positions[i].copy()

                    if fitness < gbest_fitness:
                        gbest_fitness = fitness
                        gbest_position = positions[i].copy()

                modes[:] = "glide"
                glide_fail_counter[:] = 0
                target_fail_counter[:] = 0
                glide_target_cycle_counter = 0
                stagnation_counter = 0
                reset_counter += 1

            convergence_curve.append(gbest_fitness)
            accuracy_curve.append((1.0 - gbest_fitness) * 100.0)

        return {
            "best_position": gbest_position,
            "best_fitness": gbest_fitness,
            "best_accuracy": (1.0 - gbest_fitness) * 100.0,
            "convergence_curve": convergence_curve,
            "accuracy_curve": accuracy_curve,
        }