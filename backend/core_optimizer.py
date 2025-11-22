"""
PERSON 2: Core Optimization Engine
Implements the fundamental bandwidth allocation optimization using CVXPY
with multiple utility functions and fairness metrics.
"""

import cvxpy as cp
import numpy as np
from typing import Dict, List, Tuple, Optional
import time


class CoreOptimizer:
    """
    Core bandwidth allocation optimizer using convex optimization.
    Supports multiple utility functions: log, sqrt, linear, and alpha-fair.
    """
    
    def __init__(self, n_users: int, total_capacity: float):
        """
        Initialize the core optimizer.
        
        Args:
            n_users: Number of users
            total_capacity: Total available bandwidth (Mbps)
        """
        self.n_users = n_users
        self.total_capacity = total_capacity
        self.optimal_allocation = None
        self.optimal_value = None
        self.solve_time = None
        
    def optimize(self, 
                 demands: np.ndarray,
                 priorities: np.ndarray,
                 min_bandwidth: np.ndarray,
                 max_bandwidth: np.ndarray,
                 utility_type: str = 'log',
                 alpha: float = 0.5) -> Dict:
        """
        Solve the bandwidth allocation optimization problem.
        
        Args:
            demands: User bandwidth demands (Mbps)
            priorities: User priority weights
            min_bandwidth: Minimum bandwidth per user
            max_bandwidth: Maximum bandwidth per user
            utility_type: Type of utility function ('log', 'sqrt', 'linear', 'alpha-fair')
            alpha: Alpha parameter for alpha-fair utility
            
        Returns:
            Dictionary containing optimization results
        """
        start_time = time.time()
        
        # Decision variables
        x = cp.Variable(self.n_users)
        
        # Build objective based on utility type
        if utility_type == 'log':
            # Proportional fairness (RECOMMENDED)
            objective = cp.Maximize(cp.sum(cp.multiply(priorities, cp.log(x + 1e-6))))
        elif utility_type == 'sqrt':
            # Balanced fairness
            objective = cp.Maximize(cp.sum(cp.multiply(priorities, cp.sqrt(x))))
        elif utility_type == 'linear':
            # Pure efficiency, no fairness
            objective = cp.Maximize(cp.sum(cp.multiply(priorities, x)))
        elif utility_type == 'alpha-fair':
            # Alpha-fair utility
            if alpha == 1.0:
                objective = cp.Maximize(cp.sum(cp.multiply(priorities, cp.log(x + 1e-6))))
            else:
                objective = cp.Maximize(cp.sum(cp.multiply(priorities, 
                                                            cp.power(x, 1 - alpha) / (1 - alpha))))
        else:
            raise ValueError(f"Unknown utility type: {utility_type}")
        
        # Constraints
        constraints = [
            cp.sum(x) <= self.total_capacity,  # Total capacity constraint
            x >= min_bandwidth,                 # Minimum bandwidth per user
            x <= max_bandwidth,                 # Maximum bandwidth per user
            x >= 0                              # Non-negativity
        ]
        
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                raise ValueError(f"Optimization failed with status: {problem.status}")
            
            self.optimal_allocation = x.value
            self.optimal_value = problem.value
            self.solve_time = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_metrics(demands, priorities)
            
            return {
                'status': 'optimal',
                'allocation': self.optimal_allocation,
                'objective_value': self.optimal_value,
                'solve_time': self.solve_time,
                'metrics': metrics,
                'utilization': np.sum(self.optimal_allocation) / self.total_capacity * 100
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'solve_time': time.time() - start_time
            }
    
    def _calculate_metrics(self, demands: np.ndarray, priorities: np.ndarray) -> Dict:
        """Calculate performance metrics for the allocation."""
        x = self.optimal_allocation
        
        # Jain's Fairness Index
        jains_index = (np.sum(x) ** 2) / (self.n_users * np.sum(x ** 2))
        
        # Satisfaction ratio (allocated / demanded)
        satisfaction = np.minimum(x / (demands + 1e-6), 1.0)
        avg_satisfaction = np.mean(satisfaction)
        
        # Weighted satisfaction
        weighted_satisfaction = np.sum(priorities * satisfaction) / np.sum(priorities)
        
        # Variance metrics
        allocation_std = np.std(x)
        allocation_cv = allocation_std / np.mean(x) if np.mean(x) > 0 else 0
        
        # Utility per user
        utility_per_user = priorities * np.log(x + 1e-6)
        
        return {
            'jains_fairness_index': jains_index,
            'avg_satisfaction': avg_satisfaction,
            'weighted_satisfaction': weighted_satisfaction,
            'allocation_std': allocation_std,
            'allocation_cv': allocation_cv,
            'min_allocation': np.min(x),
            'max_allocation': np.max(x),
            'median_allocation': np.median(x),
            'total_allocated': np.sum(x),
            'utility_per_user': utility_per_user
        }
    
    def compare_utility_functions(self,
                                   demands: np.ndarray,
                                   priorities: np.ndarray,
                                   min_bandwidth: np.ndarray,
                                   max_bandwidth: np.ndarray) -> Dict:
        """
        Compare different utility functions and their allocations.
        
        Returns:
            Dictionary with results for each utility function
        """
        utility_types = ['log', 'sqrt', 'linear']
        results = {}
        
        for util_type in utility_types:
            result = self.optimize(demands, priorities, min_bandwidth, 
                                   max_bandwidth, utility_type=util_type)
            results[util_type] = result
        
        # Also test alpha-fair with different alpha values
        for alpha in [0.3, 0.5, 0.7, 2.0]:
            result = self.optimize(demands, priorities, min_bandwidth, 
                                   max_bandwidth, utility_type='alpha-fair', alpha=alpha)
            results[f'alpha-fair_{alpha}'] = result
        
        return results
    
    def sensitivity_analysis(self,
                            demands: np.ndarray,
                            priorities: np.ndarray,
                            min_bandwidth: np.ndarray,
                            max_bandwidth: np.ndarray,
                            capacity_range: Tuple[float, float],
                            n_points: int = 20) -> Dict:
        """
        Perform sensitivity analysis by varying total capacity.
        
        Args:
            capacity_range: (min_capacity, max_capacity) tuple
            n_points: Number of capacity points to test
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        capacities = np.linspace(capacity_range[0], capacity_range[1], n_points)
        results = {
            'capacities': capacities,
            'objective_values': [],
            'utilizations': [],
            'fairness_indices': [],
            'avg_satisfactions': []
        }
        
        original_capacity = self.total_capacity
        
        for capacity in capacities:
            self.total_capacity = capacity
            result = self.optimize(demands, priorities, min_bandwidth, max_bandwidth)
            
            if result['status'] == 'optimal':
                results['objective_values'].append(result['objective_value'])
                results['utilizations'].append(result['utilization'])
                results['fairness_indices'].append(result['metrics']['jains_fairness_index'])
                results['avg_satisfactions'].append(result['metrics']['avg_satisfaction'])
            else:
                results['objective_values'].append(None)
                results['utilizations'].append(None)
                results['fairness_indices'].append(None)
                results['avg_satisfactions'].append(None)
        
        # Restore original capacity
        self.total_capacity = original_capacity
        
        return results


class FairnessMetrics:
    """
    Comprehensive fairness metrics calculator for bandwidth allocations.
    """
    
    @staticmethod
    def jains_fairness_index(allocation: np.ndarray) -> float:
        """
        Calculate Jain's Fairness Index.
        J(x) = (sum(x))^2 / (n * sum(x^2))
        Returns value in [0, 1], where 1 is perfectly fair.
        """
        n = len(allocation)
        return (np.sum(allocation) ** 2) / (n * np.sum(allocation ** 2))
    
    @staticmethod
    def gini_coefficient(allocation: np.ndarray) -> float:
        """
        Calculate Gini coefficient (inequality measure).
        Returns value in [0, 1], where 0 is perfect equality.
        """
        sorted_alloc = np.sort(allocation)
        n = len(allocation)
        cumsum = np.cumsum(sorted_alloc)
        return (2 * np.sum((np.arange(1, n + 1)) * sorted_alloc)) / (n * np.sum(sorted_alloc)) - (n + 1) / n
    
    @staticmethod
    def max_min_fairness_ratio(allocation: np.ndarray) -> float:
        """
        Calculate ratio of maximum to minimum allocation.
        Lower values indicate better fairness.
        """
        return np.max(allocation) / (np.min(allocation) + 1e-6)
    
    @staticmethod
    def coefficient_of_variation(allocation: np.ndarray) -> float:
        """
        Calculate coefficient of variation (normalized standard deviation).
        """
        return np.std(allocation) / (np.mean(allocation) + 1e-6)
    
    @staticmethod
    def atkinson_index(allocation: np.ndarray, epsilon: float = 0.5) -> float:
        """
        Calculate Atkinson inequality index.
        epsilon is the inequality aversion parameter (0 to infinity).
        """
        n = len(allocation)
        mean_alloc = np.mean(allocation)
        
        if epsilon == 1:
            geometric_mean = np.exp(np.mean(np.log(allocation + 1e-6)))
            return 1 - geometric_mean / mean_alloc
        else:
            return 1 - (np.mean(allocation ** (1 - epsilon)) ** (1 / (1 - epsilon))) / mean_alloc
    
    @staticmethod
    def theil_index(allocation: np.ndarray) -> float:
        """
        Calculate Theil T inequality index.
        """
        mean_alloc = np.mean(allocation)
        return np.mean((allocation / mean_alloc) * np.log(allocation / (mean_alloc + 1e-6) + 1e-6))
    
    @staticmethod
    def calculate_all_metrics(allocation: np.ndarray) -> Dict:
        """Calculate all fairness metrics at once."""
        return {
            'jains_fairness_index': FairnessMetrics.jains_fairness_index(allocation),
            'gini_coefficient': FairnessMetrics.gini_coefficient(allocation),
            'max_min_ratio': FairnessMetrics.max_min_fairness_ratio(allocation),
            'coefficient_of_variation': FairnessMetrics.coefficient_of_variation(allocation),
            'atkinson_index': FairnessMetrics.atkinson_index(allocation),
            'theil_index': FairnessMetrics.theil_index(allocation)
        }
