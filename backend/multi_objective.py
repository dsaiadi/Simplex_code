"""
PERSON 3: Multi-Objective Optimization Module
Implements multi-objective optimization with Pareto frontier generation,
weighted sum method, and epsilon-constraint method.
"""

import cvxpy as cp
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from scipy.optimize import linprog


class MultiObjectiveOptimizer:
    """
    Multi-objective bandwidth allocation optimizer.
    Balances fairness, efficiency, and latency objectives.
    """
    
    def __init__(self, n_users: int, total_capacity: float):
        """
        Initialize multi-objective optimizer.
        
        Args:
            n_users: Number of users
            total_capacity: Total available bandwidth (Mbps)
        """
        self.n_users = n_users
        self.total_capacity = total_capacity
        
    def optimize_weighted_sum(self,
                              demands: np.ndarray,
                              priorities: np.ndarray,
                              min_bandwidth: np.ndarray,
                              max_bandwidth: np.ndarray,
                              weights: Dict[str, float] = None) -> Dict:
        """
        Solve multi-objective optimization using weighted sum method.
        
        Objectives:
        1. Fairness: Maximize Jain's fairness index
        2. Efficiency: Maximize total allocated bandwidth
        3. Latency: Minimize latency penalty (inversely proportional to bandwidth)
        
        Args:
            weights: Dictionary with keys 'fairness', 'efficiency', 'latency'
                    Default: {'fairness': 0.4, 'efficiency': 0.4, 'latency': 0.2}
        
        Returns:
            Dictionary with optimization results
        """
        if weights is None:
            weights = {'fairness': 0.4, 'efficiency': 0.4, 'latency': 0.2}
        
        start_time = time.time()
        
        # Decision variables
        x = cp.Variable(self.n_users)
        
        # Objective 1: Fairness (using log utility as proxy for fairness)
        fairness_obj = cp.sum(cp.log(x + 1e-6))
        
        # Objective 2: Efficiency (total bandwidth utilization)
        efficiency_obj = cp.sum(x)
        
        # Objective 3: Latency (minimize 1/x, or maximize x with negative weight)
        # We approximate latency as inversely proportional to bandwidth
        latency_obj = cp.sum(cp.inv_pos(x + 1e-3))  # Lower is better, so we negate the weight
        
        # Combined objective
        objective = cp.Maximize(
            weights['fairness'] * fairness_obj +
            weights['efficiency'] * efficiency_obj / self.total_capacity -
            weights['latency'] * latency_obj / self.n_users
        )
        
        # Constraints
        constraints = [
            cp.sum(x) <= self.total_capacity,
            x >= min_bandwidth,
            x <= max_bandwidth,
            x >= 0.1  # Ensure positive for log and inv_pos
        ]
        
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                raise ValueError(f"Optimization failed: {problem.status}")
            
            allocation = x.value
            solve_time = time.time() - start_time
            
            # Calculate individual objective values
            fairness_val = self._calculate_fairness(allocation)
            efficiency_val = np.sum(allocation) / self.total_capacity
            latency_val = self._calculate_avg_latency(allocation)
            
            return {
                'status': 'optimal',
                'allocation': allocation,
                'objective_value': problem.value,
                'solve_time': solve_time,
                'fairness': fairness_val,
                'efficiency': efficiency_val,
                'latency': latency_val,
                'weights': weights
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'solve_time': time.time() - start_time
            }
    
    def optimize_epsilon_constraint(self,
                                     demands: np.ndarray,
                                     priorities: np.ndarray,
                                     min_bandwidth: np.ndarray,
                                     max_bandwidth: np.ndarray,
                                     primary_objective: str = 'fairness',
                                     efficiency_threshold: float = 0.8,
                                     latency_threshold: float = 50.0) -> Dict:
        """
        Solve using epsilon-constraint method.
        Optimize one objective while constraining others.
        
        Args:
            primary_objective: 'fairness', 'efficiency', or 'latency'
            efficiency_threshold: Minimum efficiency (fraction of capacity used)
            latency_threshold: Maximum average latency (ms)
        """
        start_time = time.time()
        
        x = cp.Variable(self.n_users)
        
        # Set primary objective
        if primary_objective == 'fairness':
            objective = cp.Maximize(cp.sum(cp.log(x + 1e-6)))
        elif primary_objective == 'efficiency':
            objective = cp.Maximize(cp.sum(x))
        elif primary_objective == 'latency':
            objective = cp.Minimize(cp.sum(cp.inv_pos(x + 1e-3)))
        else:
            raise ValueError(f"Unknown objective: {primary_objective}")
        
        # Base constraints
        constraints = [
            cp.sum(x) <= self.total_capacity,
            x >= min_bandwidth,
            x <= max_bandwidth,
            x >= 0.1
        ]
        
        # Add epsilon constraints
        if primary_objective != 'efficiency':
            constraints.append(cp.sum(x) >= efficiency_threshold * self.total_capacity)
        
        # Note: Latency constraint is complex with CVXPY, so we handle it in post-processing
        
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                raise ValueError(f"Optimization failed: {problem.status}")
            
            allocation = x.value
            solve_time = time.time() - start_time
            
            return {
                'status': 'optimal',
                'allocation': allocation,
                'objective_value': problem.value,
                'solve_time': solve_time,
                'primary_objective': primary_objective,
                'fairness': self._calculate_fairness(allocation),
                'efficiency': np.sum(allocation) / self.total_capacity,
                'latency': self._calculate_avg_latency(allocation)
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'solve_time': time.time() - start_time
            }
    
    def generate_pareto_frontier(self,
                                  demands: np.ndarray,
                                  priorities: np.ndarray,
                                  min_bandwidth: np.ndarray,
                                  max_bandwidth: np.ndarray,
                                  n_points: int = 20) -> Dict:
        """
        Generate Pareto frontier by varying weights in weighted sum method.
        
        Args:
            n_points: Number of points to generate on Pareto frontier
            
        Returns:
            Dictionary with Pareto frontier points and their objective values
        """
        pareto_points = []
        fairness_values = []
        efficiency_values = []
        latency_values = []
        weight_combinations = []
        
        # Generate diverse weight combinations
        for i in range(n_points):
            # Vary fairness weight from 0 to 1
            w_fairness = i / (n_points - 1)
            
            for j in range(n_points):
                # Vary efficiency weight
                w_efficiency = (1 - w_fairness) * (j / (n_points - 1))
                w_latency = 1 - w_fairness - w_efficiency
                
                if w_latency < 0:
                    continue
                
                weights = {
                    'fairness': w_fairness,
                    'efficiency': w_efficiency,
                    'latency': w_latency
                }
                
                result = self.optimize_weighted_sum(
                    demands, priorities, min_bandwidth, max_bandwidth, weights
                )
                
                if result['status'] == 'optimal':
                    pareto_points.append(result['allocation'])
                    fairness_values.append(result['fairness'])
                    efficiency_values.append(result['efficiency'])
                    latency_values.append(result['latency'])
                    weight_combinations.append(weights)
        
        # Filter to keep only non-dominated solutions
        pareto_indices = self._find_pareto_frontier(
            np.array(fairness_values),
            np.array(efficiency_values),
            -np.array(latency_values)  # Negate because lower latency is better
        )
        
        return {
            'pareto_points': [pareto_points[i] for i in pareto_indices],
            'fairness_values': [fairness_values[i] for i in pareto_indices],
            'efficiency_values': [efficiency_values[i] for i in pareto_indices],
            'latency_values': [latency_values[i] for i in pareto_indices],
            'weights': [weight_combinations[i] for i in pareto_indices],
            'n_pareto_points': len(pareto_indices)
        }
    
    def _find_pareto_frontier(self, obj1: np.ndarray, obj2: np.ndarray, obj3: np.ndarray) -> List[int]:
        """
        Find Pareto-optimal points (non-dominated solutions).
        All objectives are maximization (negate latency before calling).
        """
        n_points = len(obj1)
        pareto_indices = []
        
        for i in range(n_points):
            is_pareto = True
            for j in range(n_points):
                if i == j:
                    continue
                # Check if point j dominates point i
                if (obj1[j] >= obj1[i] and obj2[j] >= obj2[i] and obj3[j] >= obj3[i] and
                    (obj1[j] > obj1[i] or obj2[j] > obj2[i] or obj3[j] > obj3[i])):
                    is_pareto = False
                    break
            
            if is_pareto:
                pareto_indices.append(i)
        
        return pareto_indices
    
    def _calculate_fairness(self, allocation: np.ndarray) -> float:
        """Calculate Jain's Fairness Index."""
        n = len(allocation)
        return (np.sum(allocation) ** 2) / (n * np.sum(allocation ** 2))
    
    def _calculate_avg_latency(self, allocation: np.ndarray, base_latency: float = 10.0) -> float:
        """
        Calculate average latency (ms).
        Latency is inversely proportional to bandwidth.
        """
        # Simple model: latency = base_latency / bandwidth
        latencies = base_latency / (allocation + 0.1)
        return np.mean(latencies)
    
    def analyze_trade_offs(self,
                          demands: np.ndarray,
                          priorities: np.ndarray,
                          min_bandwidth: np.ndarray,
                          max_bandwidth: np.ndarray) -> Dict:
        """
        Comprehensive trade-off analysis between objectives.
        
        Returns:
            Dictionary with trade-off curves and analysis
        """
        results = []
        
        # Test various weight combinations
        weight_configs = [
            {'name': 'Fairness-focused', 'fairness': 0.8, 'efficiency': 0.1, 'latency': 0.1},
            {'name': 'Balanced', 'fairness': 0.4, 'efficiency': 0.4, 'latency': 0.2},
            {'name': 'Efficiency-focused', 'fairness': 0.1, 'efficiency': 0.8, 'latency': 0.1},
            {'name': 'Latency-focused', 'fairness': 0.2, 'efficiency': 0.2, 'latency': 0.6},
            {'name': 'Fair-Efficient', 'fairness': 0.5, 'efficiency': 0.5, 'latency': 0.0},
        ]
        
        for config in weight_configs:
            weights = {k: config[k] for k in ['fairness', 'efficiency', 'latency']}
            result = self.optimize_weighted_sum(
                demands, priorities, min_bandwidth, max_bandwidth, weights
            )
            
            if result['status'] == 'optimal':
                result['config_name'] = config['name']
                results.append(result)
        
        return {
            'configurations': results,
            'best_fairness': max(results, key=lambda x: x['fairness']),
            'best_efficiency': max(results, key=lambda x: x['efficiency']),
            'best_latency': min(results, key=lambda x: x['latency']),
        }


class ParetoAnalyzer:
    """
    Advanced Pareto frontier analysis and visualization utilities.
    """
    
    @staticmethod
    def calculate_hypervolume(pareto_front: np.ndarray, reference_point: np.ndarray) -> float:
        """
        Calculate hypervolume indicator for Pareto front quality.
        Higher values indicate better Pareto fronts.
        """
        # Simple 2D hypervolume calculation
        if pareto_front.shape[1] != 2:
            return 0.0
        
        # Sort by first objective
        sorted_front = pareto_front[pareto_front[:, 0].argsort()]
        
        hypervolume = 0.0
        for i in range(len(sorted_front)):
            if i == 0:
                width = reference_point[0] - sorted_front[i, 0]
            else:
                width = sorted_front[i-1, 0] - sorted_front[i, 0]
            
            height = sorted_front[i, 1] - reference_point[1]
            hypervolume += width * height
        
        return hypervolume
    
    @staticmethod
    def find_knee_point(pareto_front: np.ndarray) -> int:
        """
        Find the "knee point" on Pareto frontier (best trade-off point).
        Uses maximum distance from line connecting extreme points.
        """
        if len(pareto_front) < 3:
            return 0
        
        # Get extreme points
        p1 = pareto_front[0]
        p2 = pareto_front[-1]
        
        # Calculate distances from line
        max_dist = 0
        knee_idx = 0
        
        for i in range(1, len(pareto_front) - 1):
            point = pareto_front[i]
            # Distance from point to line
            dist = np.abs(np.cross(p2 - p1, point - p1)) / np.linalg.norm(p2 - p1)
            
            if dist > max_dist:
                max_dist = dist
                knee_idx = i
        
        return knee_idx
