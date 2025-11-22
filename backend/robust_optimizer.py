"""
PERSON 2: Robust Optimization Module
Implements robust optimization under uncertainty using box, budget, and ellipsoidal
uncertainty sets (Bertsimas-Sim framework).
"""

import cvxpy as cp
import numpy as np
from typing import Dict, List, Tuple, Optional
import time


class RobustOptimizer:
    """
    Robust bandwidth allocation under demand uncertainty.
    Implements three uncertainty models:
    1. Box uncertainty
    2. Budget uncertainty (Bertsimas-Sim)
    3. Ellipsoidal uncertainty
    """
    
    def __init__(self, n_users: int, total_capacity: float):
        """
        Initialize robust optimizer.
        
        Args:
            n_users: Number of users
            total_capacity: Total available bandwidth (Mbps)
        """
        self.n_users = n_users
        self.total_capacity = total_capacity
    
    def optimize_box_uncertainty(self,
                                 nominal_demands: np.ndarray,
                                 demand_deviations: np.ndarray,
                                 priorities: np.ndarray,
                                 min_bandwidth: np.ndarray,
                                 max_bandwidth: np.ndarray) -> Dict:
        """
        Robust optimization with box uncertainty.
        Uncertainty set: U_i = [d_i - δ_i, d_i + δ_i]
        
        Ensures worst-case feasibility: x_i ≥ d_i + δ_i
        
        Args:
            nominal_demands: Nominal demand values
            demand_deviations: Maximum deviation from nominal (δ_i)
            priorities: User priority weights
            min_bandwidth: Minimum bandwidth per user
            max_bandwidth: Maximum bandwidth per user
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        # Decision variables
        x = cp.Variable(self.n_users)
        
        # Objective: Maximize utility
        objective = cp.Maximize(cp.sum(cp.multiply(priorities, cp.log(x + 1e-6))))
        
        # Worst-case demand = nominal + deviation
        worst_case_demands = nominal_demands + demand_deviations
        
        # Constraints
        constraints = [
            cp.sum(x) <= self.total_capacity,
            x >= worst_case_demands,  # Worst-case feasibility
            x >= min_bandwidth,
            x <= max_bandwidth,
            x >= 0.1  # For log stability
        ]
        
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                raise ValueError(f"Optimization failed: {problem.status}")
            
            allocation = x.value
            solve_time = time.time() - start_time
            
            # Calculate robustness metrics
            robustness_metrics = self._evaluate_robustness(
                allocation, nominal_demands, demand_deviations, 'box'
            )
            
            return {
                'status': 'optimal',
                'allocation': allocation,
                'objective_value': problem.value,
                'solve_time': solve_time,
                'uncertainty_type': 'box',
                'robustness_metrics': robustness_metrics,
                'price_of_robustness': self._calculate_price_of_robustness(allocation, nominal_demands)
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'solve_time': time.time() - start_time
            }
    
    def optimize_budget_uncertainty(self,
                                    nominal_demands: np.ndarray,
                                    demand_deviations: np.ndarray,
                                    priorities: np.ndarray,
                                    min_bandwidth: np.ndarray,
                                    max_bandwidth: np.ndarray,
                                    gamma: int = None) -> Dict:
        """
        Robust optimization with budget uncertainty (Bertsimas-Sim).
        
        At most Γ demands deviate from nominal.
        Constraint: Σx_i + max_{S⊆[n], |S|≤Γ} Σ_{i∈S} δ_i ≤ C
        
        Args:
            gamma: Budget parameter (number of demands that can deviate)
                  Default: Γ = ceil(n/3)
            
        Returns:
            Dictionary with optimization results
        """
        if gamma is None:
            gamma = max(1, self.n_users // 3)
        
        start_time = time.time()
        
        # Decision variables
        x = cp.Variable(self.n_users)
        
        # Objective
        objective = cp.Maximize(cp.sum(cp.multiply(priorities, cp.log(x + 1e-6))))
        
        # For budget uncertainty, we need to find the worst-case deviation
        # This is equivalent to selecting Γ largest deviations
        sorted_deviations = np.sort(demand_deviations)[::-1]  # Descending order
        worst_case_deviation = np.sum(sorted_deviations[:gamma])
        
        # Constraints
        constraints = [
            cp.sum(x) + worst_case_deviation <= self.total_capacity,
            x >= nominal_demands,
            x >= min_bandwidth,
            x <= max_bandwidth,
            x >= 0.1
        ]
        
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                raise ValueError(f"Optimization failed: {problem.status}")
            
            allocation = x.value
            solve_time = time.time() - start_time
            
            robustness_metrics = self._evaluate_robustness(
                allocation, nominal_demands, demand_deviations, 'budget', gamma
            )
            
            return {
                'status': 'optimal',
                'allocation': allocation,
                'objective_value': problem.value,
                'solve_time': solve_time,
                'uncertainty_type': 'budget',
                'gamma': gamma,
                'robustness_metrics': robustness_metrics,
                'price_of_robustness': self._calculate_price_of_robustness(allocation, nominal_demands)
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'solve_time': time.time() - start_time
            }
    
    def optimize_ellipsoidal_uncertainty(self,
                                         nominal_demands: np.ndarray,
                                         priorities: np.ndarray,
                                         min_bandwidth: np.ndarray,
                                         max_bandwidth: np.ndarray,
                                         omega: float = None) -> Dict:
        """
        Robust optimization with ellipsoidal uncertainty.
        
        Uncertainty set: U = {d | ||d - d_bar||_2 ≤ Ω}
        Constraint: Σx_i + Ω||1||_2 ≤ C
        
        Args:
            omega: Radius of uncertainty ellipsoid
                  Default: Ω = sqrt(n) * mean(d) * 0.1
            
        Returns:
            Dictionary with optimization results
        """
        if omega is None:
            omega = np.sqrt(self.n_users) * np.mean(nominal_demands) * 0.1
        
        start_time = time.time()
        
        # Decision variables
        x = cp.Variable(self.n_users)
        
        # Objective
        objective = cp.Maximize(cp.sum(cp.multiply(priorities, cp.log(x + 1e-6))))
        
        # Ellipsoidal uncertainty adds Ω * ||1||_2 to capacity constraint
        uncertainty_margin = omega * np.sqrt(self.n_users)
        
        # Constraints
        constraints = [
            cp.sum(x) + uncertainty_margin <= self.total_capacity,
            x >= nominal_demands,
            x >= min_bandwidth,
            x <= max_bandwidth,
            x >= 0.1
        ]
        
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                raise ValueError(f"Optimization failed: {problem.status}")
            
            allocation = x.value
            solve_time = time.time() - start_time
            
            robustness_metrics = self._evaluate_robustness(
                allocation, nominal_demands, None, 'ellipsoidal', omega
            )
            
            return {
                'status': 'optimal',
                'allocation': allocation,
                'objective_value': problem.value,
                'solve_time': solve_time,
                'uncertainty_type': 'ellipsoidal',
                'omega': omega,
                'uncertainty_margin': uncertainty_margin,
                'robustness_metrics': robustness_metrics,
                'price_of_robustness': self._calculate_price_of_robustness(allocation, nominal_demands)
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'solve_time': time.time() - start_time
            }
    
    def _evaluate_robustness(self, allocation: np.ndarray, nominal_demands: np.ndarray,
                            deviations: Optional[np.ndarray], uncertainty_type: str,
                            param: Optional[float] = None) -> Dict:
        """
        Evaluate robustness of the allocation through simulation.
        
        Generates random demand scenarios and checks feasibility.
        """
        n_scenarios = 1000
        feasible_count = 0
        violation_amounts = []
        
        for _ in range(n_scenarios):
            if uncertainty_type == 'box':
                # Random demands within box
                random_demands = nominal_demands + np.random.uniform(-1, 1, self.n_users) * deviations
            elif uncertainty_type == 'budget':
                # Select gamma users to deviate
                gamma = int(param) if param else self.n_users // 3
                deviating_users = np.random.choice(self.n_users, gamma, replace=False)
                random_demands = nominal_demands.copy()
                random_demands[deviating_users] += deviations[deviating_users] * np.random.uniform(0, 1, gamma)
            elif uncertainty_type == 'ellipsoidal':
                # Random point in ellipsoid
                direction = np.random.randn(self.n_users)
                direction = direction / (np.linalg.norm(direction) + 1e-6)
                radius = param * np.random.uniform(0, 1)
                random_demands = nominal_demands + radius * direction
            else:
                random_demands = nominal_demands
            
            # Check if allocation meets demands
            shortfall = np.maximum(random_demands - allocation, 0)
            total_shortfall = np.sum(shortfall)
            
            if total_shortfall < 1e-3:
                feasible_count += 1
            else:
                violation_amounts.append(total_shortfall)
        
        robustness_probability = feasible_count / n_scenarios
        avg_violation = np.mean(violation_amounts) if violation_amounts else 0
        max_violation = np.max(violation_amounts) if violation_amounts else 0
        
        return {
            'robustness_probability': robustness_probability,
            'feasible_scenarios': feasible_count,
            'total_scenarios': n_scenarios,
            'avg_violation': avg_violation,
            'max_violation': max_violation
        }
    
    def _calculate_price_of_robustness(self, robust_allocation: np.ndarray, nominal_demands: np.ndarray) -> float:
        """
        Calculate "price of robustness": performance loss compared to nominal solution.
        
        Price of Robustness = (Nominal Objective - Robust Objective) / Nominal Objective
        """
        # Solve nominal problem (no uncertainty)
        x_nominal = cp.Variable(self.n_users)
        objective_nominal = cp.Maximize(cp.sum(cp.log(x_nominal + 1e-6)))
        constraints_nominal = [
            cp.sum(x_nominal) <= self.total_capacity,
            x_nominal >= nominal_demands,
            x_nominal >= 0.1
        ]
        
        problem_nominal = cp.Problem(objective_nominal, constraints_nominal)
        
        try:
            problem_nominal.solve(solver=cp.ECOS, verbose=False)
            nominal_obj = problem_nominal.value
            robust_obj = np.sum(np.log(robust_allocation + 1e-6))
            
            if nominal_obj > 0:
                por = (nominal_obj - robust_obj) / nominal_obj
                return max(0, por)  # Ensure non-negative
            else:
                return 0.0
        except:
            return 0.0
    
    def compare_uncertainty_models(self,
                                   nominal_demands: np.ndarray,
                                   demand_deviations: np.ndarray,
                                   priorities: np.ndarray,
                                   min_bandwidth: np.ndarray,
                                   max_bandwidth: np.ndarray) -> Dict:
        """
        Compare all three uncertainty models.
        
        Returns:
            Dictionary with results for each model
        """
        results = {}
        
        # Box uncertainty
        results['box'] = self.optimize_box_uncertainty(
            nominal_demands, demand_deviations, priorities, min_bandwidth, max_bandwidth
        )
        
        # Budget uncertainty with different gamma values
        for gamma in [self.n_users // 4, self.n_users // 3, self.n_users // 2]:
            results[f'budget_gamma_{gamma}'] = self.optimize_budget_uncertainty(
                nominal_demands, demand_deviations, priorities, min_bandwidth, max_bandwidth, gamma
            )
        
        # Ellipsoidal uncertainty with different omega values
        base_omega = np.sqrt(self.n_users) * np.mean(nominal_demands) * 0.1
        for omega_mult in [0.5, 1.0, 1.5]:
            omega = base_omega * omega_mult
            results[f'ellipsoidal_omega_{omega_mult}'] = self.optimize_ellipsoidal_uncertainty(
                nominal_demands, priorities, min_bandwidth, max_bandwidth, omega
            )
        
        return results
    
    def sensitivity_analysis_gamma(self,
                                    nominal_demands: np.ndarray,
                                    demand_deviations: np.ndarray,
                                    priorities: np.ndarray,
                                    min_bandwidth: np.ndarray,
                                    max_bandwidth: np.ndarray,
                                    gamma_range: Tuple[int, int] = None) -> Dict:
        """
        Analyze sensitivity to budget parameter Γ.
        
        Args:
            gamma_range: (min_gamma, max_gamma) tuple
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        if gamma_range is None:
            gamma_range = (1, self.n_users)
        
        gamma_values = list(range(gamma_range[0], min(gamma_range[1] + 1, self.n_users + 1)))
        
        results = {
            'gamma_values': gamma_values,
            'objective_values': [],
            'robustness_probabilities': [],
            'prices_of_robustness': [],
            'solve_times': []
        }
        
        for gamma in gamma_values:
            result = self.optimize_budget_uncertainty(
                nominal_demands, demand_deviations, priorities, min_bandwidth, max_bandwidth, gamma
            )
            
            if result['status'] == 'optimal':
                results['objective_values'].append(result['objective_value'])
                results['robustness_probabilities'].append(result['robustness_metrics']['robustness_probability'])
                results['prices_of_robustness'].append(result['price_of_robustness'])
                results['solve_times'].append(result['solve_time'])
            else:
                results['objective_values'].append(None)
                results['robustness_probabilities'].append(None)
                results['prices_of_robustness'].append(None)
                results['solve_times'].append(None)
        
        return results


class UncertaintyGenerator:
    """
    Generate realistic uncertainty scenarios for robust optimization testing.
    """
    
    @staticmethod
    def generate_demand_uncertainty(base_demands: np.ndarray, 
                                     uncertainty_level: float = 0.2) -> np.ndarray:
        """
        Generate demand deviations based on base demands.
        
        Args:
            base_demands: Nominal demand values
            uncertainty_level: Fraction of base demand (default: 20%)
            
        Returns:
            Array of demand deviations
        """
        return base_demands * uncertainty_level
    
    @staticmethod
    def generate_correlated_uncertainty(n_users: int, correlation: float = 0.5) -> np.ndarray:
        """
        Generate correlated uncertainty scenarios.
        
        Users in same group tend to have correlated demand changes.
        
        Args:
            n_users: Number of users
            correlation: Correlation coefficient [0, 1]
            
        Returns:
            Covariance matrix for uncertainty
        """
        # Create correlation matrix
        corr_matrix = np.ones((n_users, n_users)) * correlation
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Convert to covariance (assume unit variance)
        cov_matrix = corr_matrix
        
        return cov_matrix
    
    @staticmethod
    def simulate_demand_scenarios(nominal_demands: np.ndarray,
                                   demand_deviations: np.ndarray,
                                   n_scenarios: int = 100) -> np.ndarray:
        """
        Generate multiple demand scenarios for Monte Carlo analysis.
        
        Returns:
            Array of shape (n_scenarios, n_users) with demand scenarios
        """
        n_users = len(nominal_demands)
        scenarios = np.zeros((n_scenarios, n_users))
        
        for i in range(n_scenarios):
            # Random demands within uncertainty bounds
            random_factors = np.random.uniform(-1, 1, n_users)
            scenarios[i] = nominal_demands + random_factors * demand_deviations
            scenarios[i] = np.maximum(scenarios[i], 0)  # Ensure non-negative
        
        return scenarios
