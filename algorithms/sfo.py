# -*- coding: utf-8 -*-

import numpy as np

from algorithms.base_optimizer import BaseOptimizer


class SFO(BaseOptimizer):
    """
    Swift Flight Optimizer adapted only to the common project interface.

    The search equations, numerical values, global mode logic, transition
    conditions, stagnation handling, reset mechanism, and micro-adjustment
    mechanism are preserved from the submitted original SFO implementation.

    The only interface adaptations are:
    - objective_function is exposed through BaseOptimizer;
    - population_size maps to n_agents;
    - lb and ub map to the original bounds tuple;
    - random_state controls NumPy's random seed at the start of optimize();
    - optimize() returns the dictionary expected by the current project.
    """

    def __init__(
        self,
        objective_function,
        dim,
        population_size=10,
        max_iter=200,
        lb=-6,
        ub=6,
        random_state=None,
    ):
        super().__init__(
            objective_function=objective_function,
            dim=dim,
            population_size=population_size,
            max_iter=max_iter,
            lb=lb,
            ub=ub,
            random_state=random_state,
        )

        # Aliases required by the original published implementation.
        self.func = objective_function
        self.bounds = np.array((lb, ub))
        self.n_agents = population_size
        self.S = abs(ub - lb)

        # Original initialization values.
        self.gbest_history = []
        self.glide_fail_count = 0
        self.target_fail_count = 0
        self.sleep_fail_count = 0
        self.mode_switch_cycles = 0
        self.max_mode_switch_cycles = 5
        self.w, self.alpha, self.beta = 0.6, 0.4, 0.8

        self.epsilon = 1e-10
        self.stagnation_counter = np.zeros(self.n_agents, dtype=int)
        self.iteration = 0
        self.agent_state = [{} for _ in range(self.n_agents)]
        self.stagnation_global_count = 0

        self.w = 0.7
        self.prev_global_gbest = None
        self.prev_global_gbest_val = None
        self.reset_count = 0
        self.max_resets = 5
        self.best_overall_gbest = None
        self.best_overall_val = float("inf")
        self.saved_states = []
        self.w = 0.5
        self.gamma = 1.4
        self.gamma_local = 0.8

    def initialize(self):
        self.positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.n_agents, self.dim))
        self.velocities = np.zeros_like(self.positions)
        self.pbest = self.positions.copy()
        self.pbest_val = np.array([self.func(x) for x in self.positions])
        best_idx = np.argmin(self.pbest_val)
        self.gbest = self.positions[best_idx].copy()
        self.gbest_val = self.pbest_val[best_idx]
        self.w, self.alpha, self.beta = 0.1, 0.1, 0.2
        self.mode = "glide"
        self.iteration_in_target = 0
        self.sleep_counter = 0
        self.no_improve_count = 0
        self.prev_gbest_val = self.gbest_val
        self.sgma = 0.001
        self.target_mode = False
        self.target_durations = [int(self.max_iter * 0.1), int(self.max_iter * 0.1), int(self.max_iter * 0.05)]
        self.switch_to_target = int(self.max_iter * 0.25)
        self.max_sleep = 10
        self.gbest_history.append(self.gbest.copy())
        self.stagnation_global_count = 0
        self.target_thr = 10
        self.red  = 0
        self.trajectory_history = []


    def levy_flight(self, beta=1.5, size=1):
        from math import gamma, sin, pi
        sigma_u = (gamma(1 + beta) * sin(pi * beta / 2) /
                  (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size)
        v = np.random.normal(0, 1, size)
        return u / (np.abs(v) ** (1 / beta) + 1e-10)

    def optimize(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.initialize()
        self.global_best_so_far_history = []
        best_so_far = float('inf')
        for t in range(1, self.max_iter + 1):
            for i in range(self.n_agents):
                if self.mode == "glide":
                    # decay_factor = 0.001
                    # alpha = 0.1
                    # beta = 0.1
                    # w = 0.07

                    # L = self.levy_flight(beta=1.5, size=self.dim)
                    # self.velocities[i] = (
                    #     w * self.velocities[i] +
                    #     alpha * np.random.randn(self.dim) +
                    #     beta * L
                    # )


                    w=0.1
                    c1=1.1
                    c2=1.1
                    phi=0.1
                    alpha=0.1
                    beta_min=0.1
                    beta_max=1.0
                    eta=0.01
                    # --- معاملات تقليدية ---
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)

                    cognitive = c1 * r1 * (self.pbest[i] - self.positions[i])
                    social = c2 * r2 * (self.gbest - self.positions[i])
                    inertia = w * self.velocities[i]

                    # --- حساب مسافة الانزلاق المتكيّفة ---
                    beta_t = beta_min + (beta_max - beta_min) * np.exp(-eta * (t / self.max_iter))

                    # --- اتجاه الابتعاد عن gbest ---
                    direction = np.sign(self.positions[i] - self.gbest)  # يعطي 1 أو -1 حسب البُعد
                    # لو كان x يساوي gbest في بُعد معيّن نختار اتجاه عشوائي ±1
                    zero_mask = (direction == 0)
                    direction[zero_mask] = np.random.choice([-1, 1], size=np.sum(zero_mask))

                    # --- قفزة Levy بسيطة ---
                    # توليد قفزة Lévy من توزيع مستقر
                    levy = self.levy_flight(beta=1.5, size=self.dim)# بديل بسيط
                    # للحد من القيم المتطرفة:
                    levy = np.clip(levy, -3, 3)

                    # --- حساب مركب الانزلاق ---
                    d = direction * beta_t + alpha * levy
                    glide_component = phi * d

                    # --- تحديث السرعة والموقع ---
                    self.velocities[i] = inertia + cognitive + social + glide_component
                    # self.velocities[i] = np.clip(self.velocities[i], -self.vmax, self.vmax)
                elif self.mode == "target":
                    noise = np.random.normal(0, self.sgma, self.dim)
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)
                    self.sgma = 0.0001
                    w = 0.7

                    self.velocities[i] = (
                        self.w * self.velocities[i]
                        + self.gamma * r1 * (self.gbest - self.positions[i])
                        + self.gamma_local * r2 * (self.pbest[i] - self.positions[i])
                        + noise
                    )

                elif self.mode == "sleep":
                      sleep_shift = np.random.normal(0, 0.01)
                      self.velocities[i] = 0
                      self.positions[i] = self.gbest + sleep_shift
                elif self.mode == "micro_adjust":
                    w = 0.6
                    c1 = 0.0
                    c2 = 1.0
                    self.sgma = 0.00001
                    r1 = np.random.uniform(0.0, 1.1)
                    r2 = np.random.uniform(0.01, 2.1)
                    noise = np.random.normal(-self.sgma, self.sgma, self.dim)
                    self.velocities[i] = (w * self.velocities[i]
                                          + c1 * r1 * (self.pbest[i] - self.positions[i])
                                          + c2 * r2 * (self.gbest - self.positions[i])
                                          )

                elif self.mode == "deep_micro_adjust":
                    delta = 3e-0
                    min_delta = 1e0
                    decay_factor = 0.05

                    best_fitness = self.func(self.positions[i])
                    old = best_fitness
                    current_gbest = self.positions[i].copy()
                    n_dims = len(current_gbest)

                    for attempt in range(100):
                        improved = False
                        trial_plus = current_gbest.copy()
                        trial_minus = current_gbest.copy()

                        for d in range(n_dims):


                            trial_plus[d] += delta
                            f_plus = self.func(trial_plus)


                            trial_minus[d] -= delta
                            f_minus = self.func(trial_minus)
                        if f_plus < best_fitness:
                            current_gbest = trial_plus
                            best_fitness = f_plus
                            improved = True
                            break

                        if f_minus < best_fitness:
                            current_gbest = trial_minus
                            best_fitness = f_minus
                            improved = True
                            break

                        if improved:
                            print("its improved")
                            break
                        else:


                            delta = max(delta * decay_factor, min_delta)
                    print(f"the old fitness is {old} and the current is {best_fitness} and the fplus is {f_plus} and fminus is {f_minus}")

                    self.positions[i] = current_gbest
                    self.velocities[i] = np.zeros(self.dim)

                # Update position
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])


                fitness = self.func(self.positions[i])
                if fitness < self.pbest_val[i]:
                    self.stagnation_counter[i] = 0
                    self.pbest_val[i] = fitness
                    self.pbest[i] = self.positions[i].copy()

                if fitness < self.gbest_val:
                    self.gbest_val = fitness
                    self.gbest = self.positions[i].copy()
                else:
                    self.stagnation_counter[i] += 1
            if self.gbest_val < best_so_far:
                  best_so_far = self.gbest_val
            # Append best so far to history
            self.global_best_so_far_history.append(best_so_far)
            self.gbest_history.append(self.gbest.copy())

            improvement = (self.prev_gbest_val - self.gbest_val) / (abs(self.prev_gbest_val) + self.epsilon)
            # print(f"iteration {t}: mode = {self.mode}, fitness= {fitness:.4e},  gbest value = {self.gbest_val:.4e}, improvement = {improvement}, reset count = {self.reset_count}")
            # Mode transitions
            improvement = (self.prev_gbest_val - self.gbest_val) / (abs(self.prev_gbest_val)+1e-10)
            if improvement < 1e-4:
                self.stagnation_global_count += 1
            else:
                self.stagnation_global_count = 0
            if self.stagnation_global_count >= 20 and self.reset_count < self.max_resets:
                state_snapshot = {
                      "gbest": self.gbest.copy(),
                      "gbest_val": self.gbest_val,
                      "positions": self.positions.copy(),
                      "velocities": self.velocities.copy(),
                      "pbest": self.pbest.copy(),
                      "pbest_val": self.pbest_val.copy(),
                      "iteration": t,
                      "reset_index": self.reset_count + 1,
                      "trajectory": self.trajectory_history.copy()
                  }
                self.saved_states.append(state_snapshot)
                self.trajectory_history = []
                if self.gbest_val < self.best_overall_val:
                    self.best_overall_val = self.gbest_val
                    self.best_overall_gbest = self.gbest.copy()
                self.prev_global_gbest = self.gbest.copy()
                self.prev_global_gbest_val = self.gbest_val
                self.positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.n_agents, self.dim))
                self.velocities = np.zeros_like(self.positions)
                self.pbest = self.positions.copy()
                self.pbest_val = np.array([self.func(x) for x in self.positions])
                best_idx = np.argmin(self.pbest_val)
                self.gbest = self.positions[best_idx].copy()
                self.gbest_val = self.pbest_val[best_idx]
                self.mode = "target"
                self.glide_fail_count = 0
                self.no_improve_count = 0
                self.glide_fail_count = 0
                self.target_fail_count = 0
                self.mode_switch_cycles = 0
                self.mode_no_improve_count = 0
                self.sleep_fail_count = 0
                self.stagnation_global_count = 0
                self.reset_count += 1

            if self.mode_switch_cycles < self.max_mode_switch_cycles:
                if self.mode == "glide":
                    if improvement < 0.15:
                        self.glide_fail_count += 1
                        self.no_improve_count += 1
                    else:
                        self.glide_fail_count = 0
                        self.no_improve_count = 0
                    if self.glide_fail_count >= 2:
                        self.mode = "target"
                        self.glide_fail_count = 0
                        self.target_fail_count = 0
                        self.mode_switch_cycles += 1
                elif self.mode == "target":
                    if improvement < 0.15:
                        self.target_fail_count += 1
                    else:
                        self.target_fail_count = 0
                    if self.target_fail_count >= self.target_thr:
                        self.mode = "glide"
                        self.target_fail_count = 0
                        self.glide_fail_count = 0
                        self.mode_switch_cycles += 1
            elif self.stagnation_global_count >= 20 and self.reset_count >= self.max_resets and self.red == 0:
                  state_snapshot = {
                      "gbest": self.gbest.copy(),
                      "gbest_val": self.gbest_val,
                      "positions": self.positions.copy(),
                      "velocities": self.velocities.copy(),
                      "pbest": self.pbest.copy(),
                      "pbest_val": self.pbest_val.copy(),
                      "iteration": t,
                      "reset_index": self.reset_count + 1,
                      "trajectory": self.trajectory_history.copy()
                  }
                  self.saved_states.append(state_snapshot)
                  if self.gbest_val < self.best_overall_val:
                      self.best_overall_val = self.gbest_val
                      self.best_overall_gbest = self.gbest.copy()
                  best_state = min(self.saved_states, key=lambda s: self.func(s['gbest']))
                  self.trajectory_history = best_state['trajectory']
                  center = best_state['gbest']
                  r = 0.0001 * (self.bounds[1] - self.bounds[0])
                  # self.positions =best_state['positions']
                  # self.velocities = best_state['velocities']
                  # self.pbest = best_state['pbest']
                  # self.pbest_val = best_state['pbest_val']
                  # self.gbest = best_state['gbest']
                  # self.gbest_val = self.func(self.gbest)
                  self.positions = center + np.random.uniform(-r, r, (self.n_agents, self.dim))
                  self.velocities = np.zeros_like(self.positions)
                  self.pbest = self.positions.copy()
                  self.pbest_val = np.array([self.func(x) for x in self.positions])
                  best_idx = np.argmin(self.pbest_val)
                  # self.gbest = self.positions[best_idx].copy()
                  # self.gbest_val = self.pbest_val[best_idx]
                  # for i in range(self.n_agents):
                  #     self.positions[i] = center + np.random.uniform(-r, r, size=self.dim)
                  #     self.velocities[i] = np.zeros(self.dim)
                  #     self.pbest[i] = self.positions[i].copy()
                  #     self.pbest_val[i] = self.func(self.positions[i])
                  # self.gbest = center + np.random.uniform(-r, r, size=self.dim)
                  # self.gbest_val = self.func(center)
                  self.mode_switch_cycles = 20
                  self.mode = "micro_adjust"
                  if self.red == 0:
                      self.w *= 1
                      self.gamma *= 1
                      self.gamma_local *= 1
                      self.red += 1
                  self.stagnation_global_count = 0
                  self.target_thr = 150
            # else:
            #     if self.mode != "sleep":
            #         self.mode = "sleep"
            #         self.mode_no_improve_count = 0
            #     if improvement < 0.01:
            #         self.sleep_fail_count += 1
            #     else:
            #       self.sleep_fail_count = 0
            #     if self.sleep_fail_count >= 10:
            #         self.sleep_fail_count = 0
            #         self.mode_no_improve_count = 0
            #         self.mode_switch_cycles = 0
            #         self.glide_fail_count = 0
            #         self.target_fail_count = 0
            self.prev_gbest_val = self.gbest_val
            self.trajectory_history.append(self.positions[:, :2].copy())
        if self.gbest_val < self.best_overall_val:
              self.best_overall_val = self.gbest_val
              self.best_overall_gbest = self.gbest.copy()
        state_snapshot = {
                      "gbest": self.gbest.copy(),
                      "gbest_val": self.gbest_val,
                      "positions": self.positions.copy(),
                      "velocities": self.velocities.copy(),
                      "pbest": self.pbest.copy(),
                      "pbest_val": self.pbest_val.copy(),
                      "iteration": t,
                      "reset_index": self.reset_count + 1,
                      "trajectory": self.trajectory_history.copy()
                  }
        self.saved_states.append(state_snapshot)
        best_state = min(self.saved_states, key=lambda s: self.func(s['gbest']))
        self.trajectory_history = best_state['trajectory']
        convergence_curve = list(self.global_best_so_far_history)
        accuracy_curve = [(1.0 - value) * 100.0 for value in convergence_curve]

        return {
            "best_position": self.best_overall_gbest.copy(),
            "best_fitness": float(self.best_overall_val),
            "best_accuracy": (1.0 - float(self.best_overall_val)) * 100.0,
            "convergence_curve": convergence_curve,
            "accuracy_curve": accuracy_curve,
        }