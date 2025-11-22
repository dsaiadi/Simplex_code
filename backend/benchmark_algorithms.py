"""
PERSON 2 & 3: Comprehensive Benchmarking Module
Implements 10+ bandwidth allocation algorithms for comparison:
1. Equal Share, 2. Proportional Share, 3. Max-Min Fairness,
4. Weighted Max-Min, 5. Nash Bargaining, 6. Greedy,
7. Round Robin, 8. Water-Filling, 9-10. Convex Optimization (Log/Sqrt)
"""

import numpy as np
import time
from typing import Dict, List, Tuple
from .core_optimizer import CoreOptimizer, FairnessMetrics


class BenchmarkAlgorithms:
    """
    Collection of bandwidth allocation algorithms for comprehensive benchmarking.
    Compares 10+ methods with metrics: utility, fairness, efficiency, solve time.
    """
    
    def __init__(self, total_capacity: float):
        """
        Initialize benchmark algorithms.
        
        Args:
            total_capacity: Total available bandwidth (Mbps)
        """
        self.total_capacity = total_capacity
    
    def equal_share(self, 
                    n_users: int,
                    min_bandwidth: np.ndarray,
                    max_bandwidth: np.ndarray) -> Dict:
        """
        Algorithm 1: Equal Share
        Allocate bandwidth equally: x_i = C/n
        
        Simple baseline - no priorities, pure equality.
        
        Args:
            n_users: Number of users
            min_bandwidth: Minimum bandwidth per user
            max_bandwidth: Maximum bandwidth per user
            
        Returns:
            Dictionary with allocation and metrics
        """
        start_time = time.time()
        
        # Equal allocation
        equal_share = self.total_capacity / n_users
        
        # Apply min/max constraints
        allocation = np.full(n_users, equal_share)
        allocation = np.clip(allocation, min_bandwidth, max_bandwidth)
        
        # Adjust if constraints violated capacity
        if np.sum(allocation) > self.total_capacity:
            # Scale down proportionally
            allocation = allocation * (self.total_capacity / np.sum(allocation))
        
        solve_time = time.time() - start_time
        
        return {
            'algorithm': 'Equal Share',
            'allocation': allocation,
            'solve_time': solve_time,
            'metrics': FairnessMetrics.calculate_all_metrics(allocation)
        }
    
    def proportional_share(self,
                          demands: np.ndarray,
                          priorities: np.ndarray,
                          min_bandwidth: np.ndarray,
                          max_bandwidth: np.ndarray) -> Dict:
        """
        Algorithm 2: Proportional Share
        Allocate proportionally to priorities: x_i ∝ w_i
        
        x_i = C * (w_i / Σw_j)
        
        Args:
            demands: User bandwidth demands
            priorities: User priority weights
            min_bandwidth: Minimum bandwidth per user
            max_bandwidth: Maximum bandwidth per user
            
        Returns:
            Dictionary with allocation and metrics
        """
        start_time = time.time()
        
        # Proportional to priorities
        total_priority = np.sum(priorities)
        allocation = self.total_capacity * (priorities / total_priority)
        
        # Apply min/max constraints
        allocation = np.clip(allocation, min_bandwidth, max_bandwidth)
        
        # Adjust if constraints violated capacity
        if np.sum(allocation) > self.total_capacity:
            allocation = allocation * (self.total_capacity / np.sum(allocation))
        
        solve_time = time.time() - start_time
        
        return {
            'algorithm': 'Proportional Share',
            'allocation': allocation,
            'solve_time': solve_time,
            'metrics': FairnessMetrics.calculate_all_metrics(allocation)
        }
    
    def max_min_fairness(self,
                         demands: np.ndarray,
                         min_bandwidth: np.ndarray,
                         max_bandwidth: np.ndarray) -> Dict:
        """
        Algorithm 3: Max-Min Fairness
        Lexicographically maximize minimum allocation.
        
        Iteratively increase allocations starting from minimum,
        giving priority to users with smallest current allocation.
        
        Args:
            demands: User bandwidth demands
            min_bandwidth: Minimum bandwidth per user
            max_bandwidth: Maximum bandwidth per user
            
        Returns:
            Dictionary with allocation and metrics
        """
        start_time = time.time()
        
        n_users = len(demands)
        allocation = np.copy(min_bandwidth)
        remaining_capacity = self.total_capacity - np.sum(allocation)
        
        # Track which users are satisfied (hit max or demand)
        satisfied = np.zeros(n_users, dtype=bool)
        
        while remaining_capacity > 1e-6 and not np.all(satisfied):
            # Find unsatisfied users
            unsatisfied_indices = np.where(~satisfied)[0]
            
            if len(unsatisfied_indices) == 0:
                break
            
            # Distribute remaining capacity equally among unsatisfied users
            n_unsatisfied = len(unsatisfied_indices)
            increment = remaining_capacity / n_unsatisfied
            
            # Try to give increment to each unsatisfied user
            for idx in unsatisfied_indices:
                available = min(increment, 
                              max_bandwidth[idx] - allocation[idx],
                              demands[idx] - allocation[idx])
                
                if available < 1e-6:
                    satisfied[idx] = True
                else:
                    allocation[idx] += available
                    remaining_capacity -= available
            
            # Check if we're making progress
            if increment < 1e-6:
                break
        
        solve_time = time.time() - start_time
        
        return {
            'algorithm': 'Max-Min Fairness',
            'allocation': allocation,
            'solve_time': solve_time,
            'metrics': FairnessMetrics.calculate_all_metrics(allocation)
        }
    
    def weighted_max_min_fairness(self,
                                   demands: np.ndarray,
                                   priorities: np.ndarray,
                                   min_bandwidth: np.ndarray,
                                   max_bandwidth: np.ndarray) -> Dict:
        """
        Algorithm 4: Weighted Max-Min Fairness
        Max-min fairness with priority weights.
        
        Maximize minimum of (x_i / w_i) - weighted fairness.
        
        Args:
            demands: User bandwidth demands
            priorities: User priority weights
            min_bandwidth: Minimum bandwidth per user
            max_bandwidth: Maximum bandwidth per user
            
        Returns:
            Dictionary with allocation and metrics
        """
        start_time = time.time()
        
        n_users = len(demands)
        allocation = np.copy(min_bandwidth)
        remaining_capacity = self.total_capacity - np.sum(allocation)
        
        satisfied = np.zeros(n_users, dtype=bool)
        
        while remaining_capacity > 1e-6 and not np.all(satisfied):
            unsatisfied_indices = np.where(~satisfied)[0]
            
            if len(unsatisfied_indices) == 0:
                break
            
            # Weighted allocation: distribute proportionally to priorities
            unsatisfied_priorities = priorities[unsatisfied_indices]
            total_unsatisfied_priority = np.sum(unsatisfied_priorities)
            
            if total_unsatisfied_priority < 1e-9:
                break
            
            # Allocate proportionally to priorities
            for i, idx in enumerate(unsatisfied_indices):
                share = (priorities[idx] / total_unsatisfied_priority) * remaining_capacity
                available = min(share,
                              max_bandwidth[idx] - allocation[idx],
                              demands[idx] - allocation[idx])
                
                if available < 1e-6:
                    satisfied[idx] = True
                else:
                    allocation[idx] += available
                    remaining_capacity -= available
        
        solve_time = time.time() - start_time
        
        return {
            'algorithm': 'Weighted Max-Min',
            'allocation': allocation,
            'solve_time': solve_time,
            'metrics': FairnessMetrics.calculate_all_metrics(allocation)
        }
    
    def nash_bargaining(self,
                       demands: np.ndarray,
                       priorities: np.ndarray,
                       min_bandwidth: np.ndarray,
                       max_bandwidth: np.ndarray) -> Dict:
        """
        Algorithm 5: Nash Bargaining Solution
        Maximize product: ∏_i (x_i - x_i,min)^w_i
        
        Game-theoretic fair allocation maximizing weighted product of utilities.
        Equivalent to maximizing: Σ_i w_i * log(x_i - x_i,min)
        
        Args:
            demands: User bandwidth demands
            priorities: User priority weights
            min_bandwidth: Minimum bandwidth per user
            max_bandwidth: Maximum bandwidth per user
            
        Returns:
            Dictionary with allocation and metrics
        """
        start_time = time.time()
        
        # Nash bargaining: maximize Σ w_i * log(x_i - x_i,min)
        # This is similar to log utility but with offset
        
        n_users = len(demands)
        
        # Start with minimum allocation
        allocation = np.copy(min_bandwidth)
        remaining_capacity = self.total_capacity - np.sum(allocation)
        
        # Iteratively allocate to maximize weighted log utility above minimum
        max_iterations = 1000
        tolerance = 1e-6
        
        for iteration in range(max_iterations):
            if remaining_capacity < tolerance:
                break
            
            # Gradient: w_i / (x_i - x_i,min)
            # Allocate to user with highest gradient (most benefit)
            surplus = allocation - min_bandwidth + 1e-6  # Add epsilon for stability
            gradients = priorities / surplus
            
            # Mask out users who hit max
            mask = allocation < (max_bandwidth - tolerance)
            gradients = gradients * mask
            
            if np.all(gradients < 1e-9):
                break
            
            # Give small increment to user with highest gradient
            best_user = np.argmax(gradients)
            increment = min(remaining_capacity * 0.01, 
                          max_bandwidth[best_user] - allocation[best_user])
            
            allocation[best_user] += increment
            remaining_capacity -= increment
        
        solve_time = time.time() - start_time
        
        return {
            'algorithm': 'Nash Bargaining',
            'allocation': allocation,
            'solve_time': solve_time,
            'metrics': FairnessMetrics.calculate_all_metrics(allocation)
        }
    
    def greedy(self,
               demands: np.ndarray,
               priorities: np.ndarray,
               min_bandwidth: np.ndarray,
               max_bandwidth: np.ndarray) -> Dict:
        """
        Algorithm 6: Greedy Allocation
        Allocate to highest priority users first until capacity exhausted.
        
        Sort by priority (descending), allocate max possible to each in order.
        
        Args:
            demands: User bandwidth demands
            priorities: User priority weights
            min_bandwidth: Minimum bandwidth per user
            max_bandwidth: Maximum bandwidth per user
            
        Returns:
            Dictionary with allocation and metrics
        """
        start_time = time.time()
        
        n_users = len(demands)
        allocation = np.zeros(n_users)
        remaining_capacity = self.total_capacity
        
        # Sort users by priority (descending)
        sorted_indices = np.argsort(-priorities)
        
        for idx in sorted_indices:
            # Allocate as much as possible to this user
            allocation_amount = min(
                remaining_capacity,
                max_bandwidth[idx],
                demands[idx]
            )
            
            allocation[idx] = allocation_amount
            remaining_capacity -= allocation_amount
            
            if remaining_capacity < 1e-6:
                break
        
        # Ensure minimum bandwidth for all users if possible
        for idx in range(n_users):
            if allocation[idx] < min_bandwidth[idx]:
                needed = min_bandwidth[idx] - allocation[idx]
                if needed <= remaining_capacity:
                    allocation[idx] += needed
                    remaining_capacity -= needed
        
        solve_time = time.time() - start_time
        
        return {
            'algorithm': 'Greedy',
            'allocation': allocation,
            'solve_time': solve_time,
            'metrics': FairnessMetrics.calculate_all_metrics(allocation)
        }
    
    def round_robin(self,
                   demands: np.ndarray,
                   min_bandwidth: np.ndarray,
                   max_bandwidth: np.ndarray,
                   quantum: float = 10.0) -> Dict:
        """
        Algorithm 7: Round Robin Allocation
        Iteratively give equal increments (quantum) to each user in turn.
        
        Fair time-sharing approach adapted to bandwidth allocation.
        
        Args:
            demands: User bandwidth demands
            min_bandwidth: Minimum bandwidth per user
            max_bandwidth: Maximum bandwidth per user
            quantum: Bandwidth increment per round (Mbps)
            
        Returns:
            Dictionary with allocation and metrics
        """
        start_time = time.time()
        
        n_users = len(demands)
        allocation = np.zeros(n_users)
        remaining_capacity = self.total_capacity
        
        # Round-robin allocation
        max_rounds = int(self.total_capacity / quantum) + n_users
        
        for round_num in range(max_rounds):
            if remaining_capacity < 1e-6:
                break
            
            # Give quantum to each user in turn
            for user_idx in range(n_users):
                if remaining_capacity < 1e-6:
                    break
                
                # How much can we give?
                increment = min(
                    quantum,
                    remaining_capacity,
                    max_bandwidth[user_idx] - allocation[user_idx],
                    demands[user_idx] - allocation[user_idx]
                )
                
                if increment > 1e-6:
                    allocation[user_idx] += increment
                    remaining_capacity -= increment
        
        solve_time = time.time() - start_time
        
        return {
            'algorithm': 'Round Robin',
            'allocation': allocation,
            'solve_time': solve_time,
            'metrics': FairnessMetrics.calculate_all_metrics(allocation)
        }
    
    def water_filling(self,
                     demands: np.ndarray,
                     priorities: np.ndarray,
                     min_bandwidth: np.ndarray,
                     max_bandwidth: np.ndarray,
                     channel_gains: np.ndarray = None) -> Dict:
        """
        Algorithm 8: Water-Filling Allocation
        Based on channel state information and water-filling principle.
        
        Allocate more bandwidth to users with better channel conditions.
        Power allocation: x_i = max(0, λ - 1/h_i) where h_i is channel gain.
        
        Args:
            demands: User bandwidth demands
            priorities: User priority weights
            min_bandwidth: Minimum bandwidth per user
            max_bandwidth: Maximum bandwidth per user
            channel_gains: Channel gain for each user (if None, use priorities)
            
        Returns:
            Dictionary with allocation and metrics
        """
        start_time = time.time()
        
        n_users = len(demands)
        
        # Use priorities as proxy for channel gains if not provided
        if channel_gains is None:
            channel_gains = priorities / np.mean(priorities)
        
        # Water-filling algorithm
        allocation = np.zeros(n_users)
        
        # Binary search for water level (lambda)
        lambda_min, lambda_max = 0.0, self.total_capacity
        tolerance = 1e-3
        
        for _ in range(100):  # Max iterations for binary search
            lambda_mid = (lambda_min + lambda_max) / 2.0
            
            # Water-filling: x_i = max(0, lambda - 1/h_i)
            temp_allocation = np.maximum(0, lambda_mid - 1.0 / (channel_gains + 1e-6))
            
            # Apply constraints
            temp_allocation = np.clip(temp_allocation, min_bandwidth, max_bandwidth)
            
            total_allocated = np.sum(temp_allocation)
            
            if abs(total_allocated - self.total_capacity) < tolerance:
                allocation = temp_allocation
                break
            elif total_allocated < self.total_capacity:
                lambda_min = lambda_mid
            else:
                lambda_max = lambda_mid
        
        # If binary search didn't converge, use final result
        if np.sum(allocation) == 0:
            allocation = temp_allocation
        
        # Normalize to capacity
        if np.sum(allocation) > self.total_capacity:
            allocation = allocation * (self.total_capacity / np.sum(allocation))
        
        solve_time = time.time() - start_time
        
        return {
            'algorithm': 'Water-Filling',
            'allocation': allocation,
            'solve_time': solve_time,
            'metrics': FairnessMetrics.calculate_all_metrics(allocation)
        }
    
    def convex_optimization_log(self,
                               demands: np.ndarray,
                               priorities: np.ndarray,
                               min_bandwidth: np.ndarray,
                               max_bandwidth: np.ndarray) -> Dict:
        """
        Algorithm 9: Convex Optimization (Log Utility)
        Our proposed method - proportional fairness.
        
        Uses CoreOptimizer with log utility function.
        
        Args:
            demands: User bandwidth demands
            priorities: User priority weights
            min_bandwidth: Minimum bandwidth per user
            max_bandwidth: Maximum bandwidth per user
            
        Returns:
            Dictionary with allocation and metrics
        """
        optimizer = CoreOptimizer(len(demands), self.total_capacity)
        result = optimizer.optimize(demands, priorities, min_bandwidth, 
                                   max_bandwidth, utility_type='log')
        
        return {
            'algorithm': 'Convex Opt (Log)',
            'allocation': result['allocation'],
            'solve_time': result['solve_time'],
            'metrics': result['metrics']
        }
    
    def convex_optimization_sqrt(self,
                                demands: np.ndarray,
                                priorities: np.ndarray,
                                min_bandwidth: np.ndarray,
                                max_bandwidth: np.ndarray) -> Dict:
        """
        Algorithm 10: Convex Optimization (Sqrt Utility)
        Alternative convex method - balanced fairness.
        
        Uses CoreOptimizer with sqrt utility function.
        
        Args:
            demands: User bandwidth demands
            priorities: User priority weights
            min_bandwidth: Minimum bandwidth per user
            max_bandwidth: Maximum bandwidth per user
            
        Returns:
            Dictionary with allocation and metrics
        """
        optimizer = CoreOptimizer(len(demands), self.total_capacity)
        result = optimizer.optimize(demands, priorities, min_bandwidth, 
                                   max_bandwidth, utility_type='sqrt')
        
        return {
            'algorithm': 'Convex Opt (Sqrt)',
            'allocation': result['allocation'],
            'solve_time': result['solve_time'],
            'metrics': result['metrics']
        }
    
    def run_all_benchmarks(self,
                          demands: np.ndarray,
                          priorities: np.ndarray,
                          min_bandwidth: np.ndarray,
                          max_bandwidth: np.ndarray) -> Dict:
        """
        Run all 10 benchmark algorithms and compare results.
        
        Returns comprehensive comparison with all metrics:
        - Total utility
        - Jain's fairness index
        - Efficiency (% capacity used)
        - Solve time
        - Scalability
        
        Args:
            demands: User bandwidth demands
            priorities: User priority weights
            min_bandwidth: Minimum bandwidth per user
            max_bandwidth: Maximum bandwidth per user
            
        Returns:
            Dictionary with results for all algorithms
        """
        algorithms = [
            self.equal_share,
            self.proportional_share,
            self.max_min_fairness,
            self.weighted_max_min_fairness,
            self.nash_bargaining,
            self.greedy,
            self.round_robin,
            self.water_filling,
            self.convex_optimization_log,
            self.convex_optimization_sqrt
        ]
        
        results = {}
        
        for algo_func in algorithms:
            try:
                # Call algorithm with appropriate parameters
                if algo_func == self.equal_share:
                    result = algo_func(len(demands), min_bandwidth, max_bandwidth)
                elif algo_func == self.max_min_fairness or algo_func == self.round_robin:
                    result = algo_func(demands, min_bandwidth, max_bandwidth)
                else:
                    result = algo_func(demands, priorities, min_bandwidth, max_bandwidth)
                
                # Calculate additional metrics
                allocation = result['allocation']
                result['total_utility'] = np.sum(np.log(allocation + 1e-6) * priorities)
                result['efficiency'] = np.sum(allocation) / self.total_capacity * 100
                
                results[result['algorithm']] = result
                
            except Exception as e:
                print(f"Error running {algo_func.__name__}: {e}")
                continue
        
        return results
    
    def compare_algorithms(self, results: Dict) -> Dict:
        """
        Generate comprehensive comparison table and rankings.
        
        Args:
            results: Dictionary of algorithm results from run_all_benchmarks()
            
        Returns:
            Dictionary with comparison metrics and rankings
        """
        comparison = {
            'algorithms': [],
            'fairness': [],
            'efficiency': [],
            'solve_time': [],
            'utility': []
        }
        
        for algo_name, result in results.items():
            comparison['algorithms'].append(algo_name)
            comparison['fairness'].append(result['metrics']['jains_fairness_index'])
            comparison['efficiency'].append(result['efficiency'])
            comparison['solve_time'].append(result['solve_time'])
            comparison['utility'].append(result.get('total_utility', 0))
        
        # Rankings
        fairness_rank = np.argsort(-np.array(comparison['fairness']))
        efficiency_rank = np.argsort(-np.array(comparison['efficiency']))
        speed_rank = np.argsort(np.array(comparison['solve_time']))
        
        comparison['rankings'] = {
            'fairness': [comparison['algorithms'][i] for i in fairness_rank],
            'efficiency': [comparison['algorithms'][i] for i in efficiency_rank],
            'speed': [comparison['algorithms'][i] for i in speed_rank]
        }
        
        return comparison
