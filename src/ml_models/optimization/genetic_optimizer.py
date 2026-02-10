"""
AI/ML Optimization Algorithms for Satellite Power Systems

This module provides intelligent optimization capabilities:
1. Genetic Algorithm for optimal power allocation
2. Particle Swarm Optimization for MPPT
3. Bayesian Optimization for hyperparameter tuning

Author: H2Z Development Team
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional, Any
from abc import ABC, abstractmethod
import logging
from datetime import datetime
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithms."""
    population_size: int = 100
    generations: int = 200
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elitism_count: int = 5
    tournament_size: int = 3
    convergence_threshold: float = 1e-6
    max_stagnation: int = 20


class GeneticAlgorithmOptimizer:
    """
    Genetic Algorithm for Optimal Power Allocation.
    
    Solves multi-objective optimization for satellite power management:
    - Maximize power efficiency
    - Minimize subsystem stress
    - Balance battery usage
    - Ensure operational constraints
    
    Features:
    - Tournament selection
    - Simulated binary crossover (SBX)
    - Polynomial mutation
    - Elitism preservation
    - Pareto-optimal front (for multi-objective)
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.population = None
        self.fitness_history = []
        self.best_individual = None
        self.best_fitness = float('-inf')
        
        logger.info("GeneticAlgorithmOptimizer initialized")
    
    def _initialize_population(
        self,
        bounds: List[Tuple[float, float]],
        objective_func: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        """Initialize random population within bounds."""
        population = np.zeros((self.config.population_size, len(bounds)))
        
        for i, (low, high) in enumerate(bounds):
            population[:, i] = np.random.uniform(low, high, self.config.population_size)
        
        return population
    
    def _evaluate_population(
        self,
        population: np.ndarray,
        objective_func: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        """Evaluate fitness for all individuals."""
        fitness = np.array([objective_func(ind) for ind in population])
        return fitness
    
    def _tournament_selection(
        self,
        population: np.ndarray,
        fitness: np.ndarray
    ) -> np.ndarray:
        """Select individual using tournament selection."""
        selected = np.zeros(population.shape[1])
        
        for i in range(population.shape[0]):
            tournament_idx = np.random.choice(
                population.shape[0],
                size=self.config.tournament_size,
                replace=False
            )
            winner = tournament_idx[np.argmax(fitness[tournament_idx])]
            selected[i] = winner
        
        return selected.astype(int)
    
    def _crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX)."""
        if np.random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # SBX parameters
        eta_c = 10  # Crowding index
        
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                # Calculate beta
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta_c + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                
                child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
            else:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
        
        return child1, child2
    
    def _mutate(self, individual: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Polynomial mutation."""
        eta_m = 20  # Mutation index
        
        for i, (low, high) in enumerate(bounds):
            for j in range(len(individual)):
                if np.random.random() < self.config.mutation_rate:
                    delta1 = (individual[i] - low) / (high - low)
                    delta2 = (high - individual[i]) / (high - low)
                    
                    u = np.random.random()
                    
                    if u < 0.5:
                        delta = (2 * u + (1 - 2 * u) * (1 - delta1) ** (eta_m + 1)) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta = (1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta2) ** (eta_m + 1))) ** (1.0 / (eta_m + 1))
                    
                    individual[i] += delta * (high - low)
                    individual[i] = np.clip(individual[i], low, high)
        
        return individual
    
    def _preserve_elitism(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        new_population: np.ndarray
    ) -> np.ndarray:
        """Preserve best individuals (elitism)."""
        # Sort by fitness
        sorted_idx = np.argsort(fitness)[::-1]
        
        # Replace worst individuals with best from previous generation
        num_elite = min(self.config.elitism_count, len(sorted_idx))
        
        for i in range(num_elite):
            worst_idx = len(population) - 1 - i
            elite_idx = sorted_idx[i]
            new_population[worst_idx] = population[elite_idx].copy()
        
        return new_population
    
    def optimize(
        self,
        objective_func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        maximize: bool = True
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run genetic algorithm optimization.
        
        Args:
            objective_func: Function to optimize (takes array, returns scalar)
            bounds: List of (min, max) tuples for each variable
            maximize: If True, maximize; otherwise minimize
            
        Returns:
            Tuple of (best_solution, best_fitness, fitness_history)
        """
        logger.info("Starting Genetic Algorithm optimization...")
        
        # Initialize population
        self.population = self._initialize_population(bounds, objective_func)
        
        # Evaluate initial population
        fitness = self._evaluate_population(self.population, objective_func)
        
        if not maximize:
            fitness = -fitness
        
        # Track best
        best_idx = np.argmax(fitness)
        self.best_individual = self.population[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.fitness_history = [self.best_fitness]
        
        stagnation_counter = 0
        
        for gen in range(self.config.generations):
            # Selection
            selected_idx = self._tournament_selection(self.population, fitness)
            selected = self.population[selected_idx]
            
            # Create new population through crossover and mutation
            new_population = np.zeros_like(self.population)
            
            for i in range(0, self.config.population_size, 2):
                if i + 1 < self.config.population_size:
                    child1, child2 = self._crossover(selected[i], selected[i + 1])
                else:
                    child1 = selected[i].copy()
                    child2 = selected[i].copy()
                
                new_population[i] = self._mutate(child1, bounds)
                new_population[i + 1] = self._mutate(child2, bounds)
            
            # Elitism
            new_population = self._preserve_elitism(self.population, fitness, new_population)
            
            # Evaluate new population
            new_fitness = self._evaluate_population(new_population, objective_func)
            
            if not maximize:
                new_fitness = -new_fitness
            
            # Update tracking
            current_best_idx = np.argmax(new_fitness)
            if new_fitness[current_best_idx] > self.best_fitness:
                self.best_individual = new_population[current_best_idx].copy()
                self.best_fitness = new_fitness[current_best_idx]
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            self.fitness_history.append(self.best_fitness)
            self.population = new_population
            fitness = new_fitness
            
            if gen % 20 == 0:
                logger.info(f"Generation {gen}: Best Fitness = {self.best_fitness:.6f}")
            
            # Check convergence
            if stagnation_counter >= self.config.max_stagnation:
                logger.info(f"Converged at generation {gen} (stagnation limit reached)")
                break
            
            if abs(self.fitness_history[-1] - self.fitness_history[-2]) < self.config.convergence_threshold:
                logger.info(f"Converged at generation {gen} (threshold reached)")
                break
        
        logger.info(f"Optimization complete. Best fitness: {self.best_fitness:.6f}")
        
        return self.best_individual, self.best_fitness, self.fitness_history
    
    def get_pareto_front(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        objectives: int = 2
    ) -> np.ndarray:
        """Extract Pareto-optimal front for multi-objective optimization."""
        pareto_mask = np.ones(len(population), dtype=bool)
        
        for i in range(len(population)):
            for j in range(len(population)):
                if i != j:
                    # Check if j dominates i (all objectives better or equal, at least one strictly better)
                    dominates = True
                    for k in range(objectives):
                        if fitness[j, k] < fitness[i, k]:
                            dominates = False
                            break
                    
                    strictly_better = any(fitness[j, k] > fitness[i, k] for k in range(objectives))
                    
                    if dominates and strictly_better:
                        pareto_mask[i] = False
                        break
        
        return population[pareto_mask]


class PSOOptimizer:
    """
    Particle Swarm Optimization for MPPT (Maximum Power Point Tracking).
    
    Optimizes solar panel operating point for maximum power extraction.
    
    Features:
    - Inertia weight for exploration/exploitation balance
    - Cognitive and social components
    - Constriction factor for convergence
    - Velocity clamping
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.particles = None
        self.velocities = None
        self.best_positions = None
        self.best_fitness = None
        self.global_best = None
        self.global_best_fitness = float('-inf')
        self.fitness_history = []
        
        logger.info("PSOOptimizer initialized")
    
    def _initialize_swarm(
        self,
        bounds: List[Tuple[float, float]],
        objective_func: Callable[[np.ndarray], float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize particle swarm."""
        n_particles = self.config.population_size
        n_dims = len(bounds)
        
        # Initialize positions
        positions = np.zeros((n_particles, n_dims))
        for i, (low, high) in enumerate(bounds):
            positions[:, i] = np.random.uniform(low, high, n_particles)
        
        # Initialize velocities
        velocities = np.zeros((n_particles, n_dims))
        for i, (low, high) in enumerate(bounds):
            vel_range = (high - low) * 0.1
            velocities[:, i] = np.random.uniform(-vel_range, vel_range, n_particles)
        
        return positions, velocities
    
    def optimize(
        self,
        objective_func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        maximize: bool = True
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run PSO optimization.
        
        Args:
            objective_func: Function to optimize
            bounds: Variable bounds
            maximize: Optimization direction
            
        Returns:
            Tuple of (best_position, best_fitness, history)
        """
        logger.info("Starting Particle Swarm Optimization...")
        
        # Initialize
        self.particles, self.velocities = self._initialize_swarm(bounds, objective_func)
        
        # PSO parameters
        w = 0.729  # Inertia weight
        c1 = 1.49445  # Cognitive coefficient
        c2 = 1.49445  # Social coefficient
        
        # Initialize personal bests
        fitness = np.array([objective_func(p) for p in self.particles])
        
        if not maximize:
            fitness = -fitness
        
        self.best_positions = self.particles.copy()
        self.best_fitness = fitness.copy()
        
        # Initialize global best
        best_idx = np.argmax(fitness)
        self.global_best = self.particles[best_idx].copy()
        self.global_best_fitness = fitness[best_idx]
        
        self.fitness_history = [self.global_best_fitness]
        
        for gen in range(self.config.generations):
            # Update velocities and positions
            r1 = np.random.random(self.particles.shape)
            r2 = np.random.random(self.particles.shape)
            
            cognitive = c1 * r1 * (self.best_positions - self.particles)
            social = c2 * r2 * (self.global_best - self.particles)
            
            self.velocities = w * self.velocities + cognitive + social
            
            # Clamp velocities
            for i, (low, high) in enumerate(bounds):
                v_min = -0.25 * (high - low)
                v_max = 0.25 * (high - low)
                self.velocities[:, i] = np.clip(self.velocities[:, i], v_min, v_max)
            
            # Update positions
            self.particles += self.velocities
            
            # Enforce bounds
            for i, (low, high) in enumerate(bounds):
                self.particles[:, i] = np.clip(self.particles[:, i], low, high)
            
            # Evaluate fitness
            fitness = np.array([objective_func(p) for p in self.particles])
            
            if not maximize:
                fitness = -fitness
            
            # Update personal bests
            improved = fitness > self.best_fitness
            self.best_positions[improved] = self.particles[improved]
            self.best_fitness[improved] = fitness[improved]
            
            # Update global best
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > self.global_best_fitness:
                self.global_best = self.particles[current_best_idx].copy()
                self.global_best_fitness = fitness[current_best_idx]
            
            self.fitness_history.append(self.global_best_fitness)
            
            if gen % 20 == 0:
                logger.info(f"Generation {gen}: Best Fitness = {self.global_best_fitness:.6f}")
        
        logger.info(f"PSO optimization complete. Best fitness: {self.global_best_fitness:.6f}")
        
        return self.global_best, self.global_best_fitness, self.fitness_history


class MPPTOptimizer:
    """
    Specialized MPPT optimizer using PSO for maximum power point tracking.
    
    Optimizes duty cycle for DC-DC converter to maximize power extraction
    from solar panels under varying conditions.
    """
    
    def __init__(self):
        self.pso = PSOOptimizer()
        logger.info("MPPTOptimizer initialized")
    
    class SolarPanelModel:
        """Simplified solar panel I-V characteristic model."""
        
        def __init__(
            self,
            irradiance_w_m2: float = 1000.0,
            temperature_c: float = 25.0,
            Voc: float = 45.0,
            Isc: float = 5.5,
            n: float = 1.2,  # Ideality factor
            Rs: float = 0.1,  # Series resistance
            Rsh: float = 1000.0  # Shunt resistance
        ):
            self.irradiance = irradiance_w_m2
            self.temperature = temperature_c
            self.Voc = Voc
            self.Isc = Isc
            self.n = n
            self.Rs = Rs
            self.Rsh = Rsh
            
            # Temperature coefficients
            self.Voc_temp_coef = -0.003  # per °C
            self.Isc_temp_coef = 0.001   # per °C
        
        def current_at_voltage(self, voltage: float) -> float:
            """Calculate current at given voltage using single-diode model."""
            # Temperature correction
            temp_corr = self.temperature - 25.0
            Vtc = self.Voc * (1 + self.Voc_temp_coef * temp_corr)
            Itc = self.Isc * (1 + self.Isc_temp_coef * temp_corr)
            
            # Irradiance correction
            Ir = self.irradiance / 1000.0
            
            # Thermal voltage
            Vt = 0.0257 * self.n
            
            # Simplified calculation
            Iph = Itc * Ir
            I0 = 1e-12  # Reverse saturation current
            Id = I0 * (np.exp(voltage / (self.n * Vt)) - 1)
            Ish = voltage / self.Rsh
            
            return max(0, Iph - Id - Ish)
        
        def power_at_voltage(self, voltage: float) -> float:
            """Calculate power at given voltage."""
            current = self.current_at_voltage(voltage)
            return current * voltage
    
    def optimize_duty_cycle(
        self,
        irradiance: float,
        temperature: float,
        Voc: float = 45.0,
        Isc: float = 5.5
    ) -> Dict[str, float]:
        """
        Optimize duty cycle for maximum power point.
        
        Args:
            irradiance: Solar irradiance in W/m²
            temperature: Cell temperature in °C
            Voc: Open circuit voltage
            Isc: Short circuit current
            
        Returns:
            Dictionary with optimal duty cycle and power metrics
        """
        # Create solar panel model
        panel = self.SolarPanelModel(
            irradiance_w_m2=irradiance,
            temperature_c=temperature,
            Voc=Voc,
            Isc=Isc
        )
        
        # Define objective function (maximize power)
        def objective(duty_cycle: np.ndarray) -> float:
            # Duty cycle affects effective voltage
            duty = duty_cycle[0]
            duty = np.clip(duty, 0.1, 0.9)
            
            # Effective voltage at converter output
            V_eff = panel.Voc * (1 - duty * 0.5)
            
            # Calculate power
            power = panel.power_at_voltage(V_eff)
            
            return power
        
        # Run PSO
        bounds = [(0.1, 0.9)]  # Duty cycle bounds
        best_duty, best_power, history = self.pso.optimize(objective, bounds)
        
        # Calculate MPP for comparison
        mpp_voltage = Voc * 0.78  # Typical MPP fraction
        mpp_power = panel.power_at_voltage(mpp_voltage)
        
        return {
            'optimal_duty_cycle': best_duty[0],
            'optimized_power': best_power,
            'theoretical_mpp': mpp_power,
            'tracking_efficiency': best_power / mpp_power if mpp_power > 0 else 0,
            'convergence_history': history
        }


class BayesianOptimizer:
    """
    Bayesian Optimization for hyperparameter tuning.
    
    Uses Gaussian Process surrogate model to efficiently search
    high-dimensional hyperparameter spaces.
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.X_observed = []
        self.y_observed = []
        self.gp_model = None
        
        logger.info("BayesianOptimizer initialized")
    
    def _acquisition_function(
        self,
        X_candidate: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        xi: float = 0.01
    ) -> np.ndarray:
        """
        Expected Improvement acquisition function.
        
        Balances exploration (high uncertainty) and exploitation (high mean).
        """
        best_y = max(self.y_observed)
        Z = (mean - best_y - xi) / (std + 1e-10)
        ei = (mean - best_y - xi) * self._normal_cdf(Z) + std * self._normal_pdf(Z)
        return ei
    
    def _normal_cdf(self, x: np.ndarray) -> np.ndarray:
        """Normal cumulative distribution function."""
        return 0.5 * (1 + np.erf(x / np.sqrt(2)))
    
    def _normal_pdf(self, x: np.ndarray) -> np.ndarray:
        """Normal probability density function."""
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
    
    def _fit_gp(self):
        """Fit Gaussian Process model to observed data."""
        if len(self.X_observed) < 2:
            return
        
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        
        # Simple GP with RBF kernel (simplified implementation)
        n = len(X)
        sigma_f = 1.0
        sigma_n = 0.1
        
        # Compute kernel matrix
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = sigma_f ** 2 * np.exp(-0.5 * np.sum((X[i] - X[j]) ** 2))
        
        # Add noise
        K += sigma_n ** 2 * np.eye(n)
        
        self.gp_model = {
            'X': X,
            'y': y,
            'K': K,
            'K_inv': np.linalg.inv(K)
        }
    
    def _predict_gp(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance at test points."""
        if self.gp_model is None or len(self.X_observed) < 2:
            # Prior (no observations yet)
            return np.full(len(X_test), 0), np.ones(len(X_test))
        
        X = self.gp_model['X']
        K_inv = self.gp_model['K_inv']
        y = self.gp_model['y']
        
        # Compute kernel with test points
        K_s = np.zeros((len(X_test), len(X)))
        for i in range(len(X_test)):
            for j in range(len(X)):
                K_s[i, j] = np.exp(-0.5 * np.sum((X_test[i] - X[j]) ** 2))
        
        # Predictive mean
        mean = K_s @ K_inv @ y
        
        # Predictive variance
        k_xx = 1.0 + 0.1 ** 2  # Prior variance
        var = k_xx - np.sum(K_s @ K_inv * K_s, axis=1, keepdims=True).flatten()
        var = np.maximum(var, 1e-10)
        
        return mean, var
    
    def optimize(
        self,
        objective_func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        n_iterations: int = 50
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run Bayesian optimization.
        
        Args:
            objective_func: Function to optimize
            bounds: Variable bounds
            n_iterations: Number of iterations
            
        Returns:
            Tuple of (best_solution, best_value, history)
        """
        logger.info("Starting Bayesian Optimization...")
        
        dim = len(bounds)
        history = []
        
        # Initial random sampling
        n_initial = min(5 * dim, 20)
        
        for i in range(n_initial):
            x = np.array([np.random.uniform(low, high) for low, high in bounds])
            y = objective_func(x)
            
            self.X_observed.append(x)
            self.y_observed.append(y)
            history.append(y)
        
        # Bayesian optimization iterations
        for it in range(n_iterations):
            # Fit GP
            self._fit_gp()
            
            # Find next point to evaluate
            best_ei = -np.inf
            best_x = None
            
            n_candidates = 100
            for _ in range(n_candidates):
                x_candidate = np.array([np.random.uniform(low, high) for low, high in bounds])
                mean, std = self._predict_gp(np.array([x_candidate]))
                ei = self._acquisition_function(np.array([x_candidate]), mean, std)
                
                if ei > best_ei:
                    best_ei = ei
                    best_x = x_candidate
            
            # Evaluate
            y_new = objective_func(best_x)
            self.X_observed.append(best_x)
            self.y_observed.append(y_new)
            history.append(max(self.y_observed))
            
            if it % 10 == 0:
                logger.info(f"Iteration {it}: Best Value = {max(self.y_observed):.6f}")
        
        # Find best
        best_idx = np.argmax(self.y_observed)
        best_solution = np.array(self.X_observed[best_idx])
        best_value = self.y_observed[best_idx]
        
        logger.info(f"Bayesian optimization complete. Best value: {best_value:.6f}")
        
        return best_solution, best_value, history


def optimize_power_allocation(
    solar_power_available: float,
    battery_soc: float,
    subsystem_demands: Dict[str, float],
    priorities: Dict[str, int] = None
) -> Dict[str, float]:
    """
    Optimize power allocation across satellite subsystems.
    
    Uses greedy algorithm with priority-based allocation.
    
    Args:
        solar_power_available: Current solar power generation
        battery_soc: Battery state of charge (0-1)
        subsystem_demands: Power demand per subsystem
        priorities: Priority levels (higher = more important)
        
    Returns:
        Dictionary with allocated power per subsystem
    """
    if priorities is None:
        priorities = {
            'ADCS': 3,
            'TT&C': 3,
            'CDH': 2,
            'Propulsion': 2,
            'Communication': 2,
            'Payload': 1
        }
    
    # Calculate total demand
    total_demand = sum(subsystem_demands.values())
    
    # Calculate available power
    battery_power = battery_soc * 100  # Max 100W from battery
    total_available = solar_power_available + battery_power
    
    # Initialize allocations
    allocations = {k: 0.0 for k in subsystem_demands.keys()}
    
    if total_available >= total_demand:
        # Enough power - allocate full demands
        for k, v in subsystem_demands.items():
            allocations[k] = v
    else:
        # Sort by priority
        sorted_subsystems = sorted(
            subsystem_demands.keys(),
            key=lambda x: priorities.get(x, 0),
            reverse=True
        )
        
        remaining_power = total_available
        
        for subsystem in sorted_subsystems:
            demand = subsystem_demands[subsystem]
            priority = priorities.get(subsystem, 0)
            
            # Allocate proportionally to priority
            if remaining_power <= 0:
                allocations[subsystem] = 0
                continue
            
            # Minimum power for critical systems
            min_power = demand * 0.5 if priority >= 2 else 0
            
            allocated = min(demand, remaining_power, max(demand * priority / 3, min_power))
            allocations[subsystem] = allocated
            remaining_power -= allocated
    
    return allocations


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("AI/ML Optimization Demo")
    logger.info("=" * 60)
    
    # Demo 1: Genetic Algorithm
    logger.info("\n--- Genetic Algorithm Demo ---")
    
    def rosenbrock(x: np.ndarray) -> float:
        """Rosenbrock test function."""
        return sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
    
    bounds = [(-2, 2), (-2, 2), (-2, 2)]
    
    ga = GeneticAlgorithmOptimizer(
        OptimizationConfig(population_size=50, generations=100)
    )
    best_solution, best_fitness, history = ga.optimize(rosenbrock, bounds, maximize=False)
    
    logger.info(f"GA Best Solution: {best_solution}")
    logger.info(f"GA Best Fitness: {best_fitness:.6f}")
    
    # Demo 2: MPPT Optimization
    logger.info("\n--- MPPT Optimization Demo ---")
    
    mppt = MPPTOptimizer()
    result = mppt.optimize_duty_cycle(
        irradiance=1000.0,
        temperature=25.0,
        Voc=45.0,
        Isc=5.5
    )
    
    logger.info(f"Optimal Duty Cycle: {result['optimal_duty_cycle']:.4f}")
    logger.info(f"Optimized Power: {result['optimized_power']:.2f} W")
    logger.info(f"Tracking Efficiency: {result['tracking_efficiency']*100:.2f}%")
    
    # Demo 3: Power Allocation
    logger.info("\n--- Power Allocation Demo ---")
    
    demands = {
        'ADCS': 41.26,
        'TT&C': 20.32,
        'CDH': 13.71,
        'Propulsion': 96.60,
        'Communication': 28.19,
        'Payload': 13.00
    }
    
    allocations = optimize_power_allocation(
        solar_power_available=500.0,
        battery_soc=0.5,
        subsystem_demands=demands
    )
    
    logger.info("Power Allocations:")
    for subsystem, power in allocations.items():
        logger.info(f"  {subsystem}: {power:.2f} W")
    
    logger.info("\n" + "=" * 60)
    logger.info("Optimization demo completed successfully!")

