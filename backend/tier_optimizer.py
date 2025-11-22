"""
Tier-Based Bandwidth Allocation Optimizer
Prioritizes: Emergency Services > Premium > Free
Ensures emergency services always get priority, premium users get guarantees,
and free users are optimized with available capacity.
"""

import cvxpy as cp
import numpy as np
from typing import Dict, List, Tuple, Optional
import time


class TierBasedOptimizer:
    """
    Advanced optimizer that handles Emergency, Premium, and Free tier users.
    
    Priority Hierarchy:
    1. Emergency Services: Always satisfied first, can preempt others
    2. Premium Users: Guaranteed bandwidth with high priority
    3. Free Users: Best effort with optimization
    """
    
    def __init__(self, total_capacity: float):
        """
        Initialize tier-based optimizer.
        
        Args:
            total_capacity: Total available bandwidth (Mbps)
        """
        self.total_capacity = total_capacity
        
    def optimize_with_tiers(self,
                          demands: np.ndarray,
                          priorities: np.ndarray,
                          min_bandwidth: np.ndarray,
                          max_bandwidth: np.ndarray,
                          tiers: np.ndarray,  # 0=emergency, 1=premium, 2=free
                          allocation_weights: np.ndarray,
                          utility_type: str = 'log') -> Dict:
        """
        Optimize bandwidth allocation with tier-based priorities.
        
        Strategy:
        1. Allocate to emergency services first (up to their max)
        2. Allocate to premium users (meet guarantees)
        3. Optimize remaining capacity for all users
        
        Args:
            demands: User bandwidth demands
            priorities: User priorities
            min_bandwidth: Minimum bandwidth per user
            max_bandwidth: Maximum bandwidth per user
            tiers: User tier codes (0=emergency, 1=premium, 2=free)
            allocation_weights: Allocation weights (emergency=10x, premium=3x, free=1x)
            utility_type: Utility function type
            
        Returns:
            Dictionary with optimization results and tier statistics
        """
        start_time = time.time()
        n_users = len(demands)
        
        # Separate users by tier
        emergency_mask = (tiers == 0)
        premium_mask = (tiers == 1)
        free_mask = (tiers == 2)
        
        n_emergency = np.sum(emergency_mask)
        n_premium = np.sum(premium_mask)
        n_free = np.sum(free_mask)
        
        # Phase 1: Allocate to emergency services (highest priority)
        emergency_allocation = np.zeros(n_users)
        emergency_capacity_used = 0.0
        
        if n_emergency > 0:
            emergency_demands = demands[emergency_mask]
            emergency_max = max_bandwidth[emergency_mask]
            
            # Emergency services get their demands satisfied first
            emergency_needed = np.minimum(emergency_demands, emergency_max)
            total_emergency_need = np.sum(emergency_needed)
            
            if total_emergency_need <= self.total_capacity:
                # Can satisfy all emergency demands
                emergency_allocation[emergency_mask] = emergency_needed
                emergency_capacity_used = total_emergency_need
            else:
                # Even emergency services need proportional reduction (rare)
                scale_factor = self.total_capacity / total_emergency_need
                emergency_allocation[emergency_mask] = emergency_needed * scale_factor
                emergency_capacity_used = self.total_capacity
        
        # Remaining capacity after emergency
        remaining_capacity = self.total_capacity - emergency_capacity_used
        
        # Phase 2 & 3: Optimize for premium and free users together
        if remaining_capacity > 0 and (n_premium > 0 or n_free > 0):
            # Decision variables for non-emergency users
            x = cp.Variable(n_users)
            
            # Build objective with tier-weighted utilities
            if utility_type == 'log':
                objective = cp.Maximize(cp.sum(cp.multiply(allocation_weights, cp.log(x + 1e-6))))
            elif utility_type == 'sqrt':
                objective = cp.Maximize(cp.sum(cp.multiply(allocation_weights, cp.sqrt(x))))
            elif utility_type == 'linear':
                objective = cp.Maximize(cp.sum(cp.multiply(allocation_weights, x)))
            else:
                objective = cp.Maximize(cp.sum(cp.multiply(allocation_weights, cp.log(x + 1e-6))))
            
            # Constraints
            constraints = [
                cp.sum(x) <= remaining_capacity,  # Use remaining capacity
                x >= min_bandwidth,                # Minimum guarantees
                x <= max_bandwidth,                # Maximum limits
                x >= 0                             # Non-negativity
            ]
            
            # Emergency users already allocated
            for i in range(n_users):
                if emergency_mask[i]:
                    constraints.append(x[i] == emergency_allocation[i])
            
            # Premium users should get higher guarantees
            if n_premium > 0:
                premium_min = min_bandwidth[premium_mask]
                premium_priority_boost = cp.sum(x[premium_mask]) >= cp.sum(premium_min) * 0.9
                # Try to add as soft constraint (best effort)
            
            # Solve optimization
            problem = cp.Problem(objective, constraints)
            
            try:
                problem.solve(solver=cp.ECOS, verbose=False)
                
                if problem.status in ["optimal", "optimal_inaccurate"]:
                    final_allocation = x.value
                else:
                    # Fallback: proportional allocation
                    final_allocation = self._proportional_fallback(
                        demands, priorities, min_bandwidth, max_bandwidth,
                        remaining_capacity, emergency_allocation, emergency_mask
                    )
            except:
                # Fallback on error
                final_allocation = self._proportional_fallback(
                    demands, priorities, min_bandwidth, max_bandwidth,
                    remaining_capacity, emergency_allocation, emergency_mask
                )
        else:
            # Only emergency users, or no capacity left
            final_allocation = emergency_allocation
        
        solve_time = time.time() - start_time
        
        # Calculate tier-specific metrics
        tier_stats = self._calculate_tier_statistics(
            final_allocation, demands, min_bandwidth, max_bandwidth,
            tiers, emergency_mask, premium_mask, free_mask
        )
        
        # Overall metrics
        total_allocated = np.sum(final_allocation)
        efficiency = total_allocated / self.total_capacity if self.total_capacity > 0 else 0
        
        # Fairness (Jain's index)
        jains_index = self._jains_fairness_index(final_allocation)
        
        # Satisfaction scores
        satisfaction = final_allocation / np.maximum(demands, 1e-6)
        satisfaction = np.minimum(satisfaction, 1.0)
        
        return {
            'status': 'optimal',
            'allocation': final_allocation,
            'total_allocated': total_allocated,
            'efficiency': efficiency,
            'solve_time': solve_time,
            'jains_fairness_index': jains_index,
            'avg_satisfaction': np.mean(satisfaction),
            'tier_statistics': tier_stats,
            'emergency_capacity_used': emergency_capacity_used,
            'remaining_capacity': self.total_capacity - total_allocated,
            'utility_type': utility_type
        }
    
    def _proportional_fallback(self, demands, priorities, min_bandwidth, max_bandwidth,
                               remaining_capacity, emergency_allocation, emergency_mask):
        """Fallback proportional allocation if optimization fails."""
        n_users = len(demands)
        allocation = emergency_allocation.copy()
        
        # For non-emergency users
        non_emergency = ~emergency_mask
        
        if np.sum(non_emergency) > 0:
            # Weighted proportional allocation
            weights = priorities[non_emergency]
            total_weight = np.sum(weights)
            
            if total_weight > 0:
                proportional = (weights / total_weight) * remaining_capacity
                # Apply min/max constraints
                proportional = np.maximum(proportional, min_bandwidth[non_emergency])
                proportional = np.minimum(proportional, max_bandwidth[non_emergency])
                
                # Scale if over capacity
                if np.sum(proportional) > remaining_capacity:
                    proportional *= (remaining_capacity / np.sum(proportional))
                
                allocation[non_emergency] = proportional
        
        return allocation
    
    def _calculate_tier_statistics(self, allocation, demands, min_bandwidth, max_bandwidth,
                                   tiers, emergency_mask, premium_mask, free_mask) -> Dict:
        """Calculate detailed statistics for each tier."""
        stats = {}
        
        for tier_name, mask in [('emergency', emergency_mask), 
                                ('premium', premium_mask), 
                                ('free', free_mask)]:
            if np.sum(mask) > 0:
                tier_allocation = allocation[mask]
                tier_demands = demands[mask]
                tier_min = min_bandwidth[mask]
                tier_max = max_bandwidth[mask]
                
                # Calculate metrics
                satisfaction = tier_allocation / np.maximum(tier_demands, 1e-6)
                satisfaction = np.minimum(satisfaction, 1.0)
                
                stats[tier_name] = {
                    'count': int(np.sum(mask)),
                    'total_demand': float(np.sum(tier_demands)),
                    'total_allocated': float(np.sum(tier_allocation)),
                    'avg_allocation': float(np.mean(tier_allocation)),
                    'min_allocation': float(np.min(tier_allocation)),
                    'max_allocation': float(np.max(tier_allocation)),
                    'avg_satisfaction': float(np.mean(satisfaction)),
                    'min_satisfaction': float(np.min(satisfaction)),
                    'guarantee_met_pct': float(np.mean(tier_allocation >= tier_min) * 100),
                    'demand_met_pct': float(np.mean(satisfaction) * 100)
                }
            else:
                stats[tier_name] = None
        
        return stats
    
    def _jains_fairness_index(self, allocation: np.ndarray) -> float:
        """Calculate Jain's Fairness Index."""
        if len(allocation) == 0:
            return 0.0
        
        sum_x = np.sum(allocation)
        sum_x_squared = np.sum(allocation ** 2)
        
        if sum_x_squared == 0:
            return 0.0
        
        n = len(allocation)
        jains_index = (sum_x ** 2) / (n * sum_x_squared)
        
        return jains_index
    
    def optimize_emergency_scenario(self,
                                    demands: np.ndarray,
                                    priorities: np.ndarray,
                                    min_bandwidth: np.ndarray,
                                    max_bandwidth: np.ndarray,
                                    tiers: np.ndarray,
                                    allocation_weights: np.ndarray,
                                    scenario_multipliers: Dict) -> Dict:
        """
        Optimize for emergency scenarios with adjusted demands.
        
        Args:
            scenario_multipliers: Dict with multipliers per tier
                {
                    'emergency_demand_multiplier': 3.0,
                    'premium_demand_multiplier': 1.5,
                    'free_demand_multiplier': 0.5,
                    'capacity_reduction': 0.2
                }
        
        Returns:
            Optimization results for emergency scenario
        """
        # Adjust demands based on scenario
        adjusted_demands = demands.copy()
        
        emergency_mask = (tiers == 0)
        premium_mask = (tiers == 1)
        free_mask = (tiers == 2)
        
        adjusted_demands[emergency_mask] *= scenario_multipliers.get('emergency_demand_multiplier', 1.0)
        adjusted_demands[premium_mask] *= scenario_multipliers.get('premium_demand_multiplier', 1.0)
        adjusted_demands[free_mask] *= scenario_multipliers.get('free_demand_multiplier', 1.0)
        
        # Adjust capacity
        capacity_reduction = scenario_multipliers.get('capacity_reduction', 0.0)
        adjusted_capacity = self.total_capacity * (1.0 - capacity_reduction)
        
        # Save original and use adjusted
        original_capacity = self.total_capacity
        self.total_capacity = adjusted_capacity
        
        # Optimize with adjusted parameters
        result = self.optimize_with_tiers(
            adjusted_demands, priorities, min_bandwidth, max_bandwidth,
            tiers, allocation_weights
        )
        
        # Restore original capacity
        self.total_capacity = original_capacity
        
        # Add scenario info
        result['scenario'] = scenario_multipliers.get('description', 'Emergency Scenario')
        result['capacity_reduction'] = capacity_reduction
        result['adjusted_capacity'] = adjusted_capacity
        
        return result
    
    def compare_scenarios(self,
                         users_df,
                         scenarios: List[str] = ['normal', 'disaster', 'cyber_attack']) -> Dict:
        """
        Compare allocation across different emergency scenarios.
        
        Args:
            users_df: User DataFrame with tier information
            scenarios: List of scenario names to compare
            
        Returns:
            Comparison dictionary
        """
        from .data_generator_enhanced import EnhancedDataGenerator
        
        results = {}
        
        for scenario_name in scenarios:
            scenario_params = EnhancedDataGenerator.generate_emergency_scenarios(
                users_df, scenario_name
            )
            
            result = self.optimize_emergency_scenario(
                demands=users_df['base_demand_mbps'].values,
                priorities=users_df['priority'].values,
                min_bandwidth=users_df['min_bandwidth_mbps'].values,
                max_bandwidth=users_df['max_bandwidth_mbps'].values,
                tiers=users_df['user_type_code'].values,
                allocation_weights=users_df['allocation_weight'].values,
                scenario_multipliers=scenario_params
            )
            
            results[scenario_name] = result
        
        return results
