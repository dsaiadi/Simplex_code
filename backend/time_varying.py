"""
PERSON 1: Time-Varying Optimization Module
Implements temporal bandwidth allocation with realistic demand patterns,
temporal fairness constraints, and dynamic scheduling.
"""

import cvxpy as cp
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime, timedelta


class TimeVaryingOptimizer:
    """
    Time-varying bandwidth allocation optimizer.
    Handles temporal dynamics with 24-hour optimization horizon.
    """
    
    def __init__(self, n_users: int, time_slots: int = 24):
        """
        Initialize time-varying optimizer.
        
        Args:
            n_users: Number of users
            time_slots: Number of time slots (default: 24 for hourly)
        """
        self.n_users = n_users
        self.time_slots = time_slots
        
    def generate_realistic_demand_pattern(self, base_demands: np.ndarray, user_types: np.ndarray = None) -> np.ndarray:
        """
        Generate realistic time-varying demand patterns based on user types.
        
        User types:
        - 0: Business users (high during work hours 9am-5pm)
        - 1: Residential users (high during evening 7pm-11pm)
        - 2: Night users (high during night 11pm-6am)
        - 3: Constant users (uniform demand)
        
        Returns:
            Array of shape (n_users, time_slots) with demand for each user at each time
        """
        if user_types is None:
            # Randomly assign user types
            user_types = np.random.randint(0, 4, self.n_users)
        
        demands = np.zeros((self.n_users, self.time_slots))
        
        for i in range(self.n_users):
            base_demand = base_demands[i]
            user_type = user_types[i]
            
            if user_type == 0:  # Business users
                pattern = self._business_pattern()
            elif user_type == 1:  # Residential users
                pattern = self._residential_pattern()
            elif user_type == 2:  # Night users
                pattern = self._night_pattern()
            else:  # Constant users
                pattern = np.ones(self.time_slots)
            
            # Add random variation
            noise = np.random.normal(1.0, 0.1, self.time_slots)
            pattern = pattern * noise
            pattern = np.maximum(pattern, 0.1)  # Ensure minimum demand
            
            demands[i, :] = base_demand * pattern
        
        return demands
    
    def _business_pattern(self) -> np.ndarray:
        """High demand during business hours (9am-5pm)."""
        pattern = np.ones(self.time_slots) * 0.3  # Base low demand
        pattern[9:17] = 1.0  # Peak during work hours
        pattern[7:9] = 0.6   # Morning ramp-up
        pattern[17:19] = 0.6 # Evening ramp-down
        return pattern
    
    def _residential_pattern(self) -> np.ndarray:
        """High demand during evening (7pm-11pm)."""
        pattern = np.ones(self.time_slots) * 0.4
        pattern[19:23] = 1.0  # Peak evening
        pattern[17:19] = 0.7  # Early evening
        pattern[23:24] = 0.7  # Late evening
        pattern[0:2] = 0.5    # Early night
        pattern[6:9] = 0.6    # Morning
        return pattern
    
    def _night_pattern(self) -> np.ndarray:
        """High demand during night (11pm-6am)."""
        pattern = np.ones(self.time_slots) * 0.3
        pattern[23:24] = 1.0  # Late night
        pattern[0:6] = 1.0    # Early morning
        pattern[6:8] = 0.6    # Morning transition
        return pattern
    
    def generate_time_varying_capacity(self, base_capacity: float, pattern_type: str = 'realistic') -> np.ndarray:
        """
        Generate time-varying total capacity.
        
        Args:
            base_capacity: Base capacity value
            pattern_type: 'constant', 'realistic', or 'dynamic'
            
        Returns:
            Array of capacity for each time slot
        """
        if pattern_type == 'constant':
            return np.ones(self.time_slots) * base_capacity
        
        elif pattern_type == 'realistic':
            # Slightly higher capacity during off-peak hours
            capacity = np.ones(self.time_slots) * base_capacity
            capacity[0:6] *= 1.1    # More capacity at night
            capacity[9:17] *= 0.95  # Slightly less during peak hours
            return capacity
        
        elif pattern_type == 'dynamic':
            # Variable capacity simulating network conditions
            capacity = base_capacity * (0.9 + 0.2 * np.random.rand(self.time_slots))
            return capacity
        
        else:
            return np.ones(self.time_slots) * base_capacity
    
    def optimize_temporal(self,
                         demands: np.ndarray,
                         priorities: np.ndarray,
                         capacities: np.ndarray,
                         min_bandwidth: np.ndarray,
                         max_bandwidth: np.ndarray,
                         temporal_fairness_threshold: float = 0.8) -> Dict:
        """
        Solve time-varying optimization problem.
        
        Args:
            demands: Array of shape (n_users, time_slots)
            priorities: User priorities (n_users,)
            capacities: Capacity for each time slot (time_slots,)
            min_bandwidth: Minimum bandwidth per user (n_users,)
            max_bandwidth: Maximum bandwidth per user (n_users,)
            temporal_fairness_threshold: Minimum average allocation ratio
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        # Decision variables: x[i, t] = bandwidth for user i at time t
        x = cp.Variable((self.n_users, self.time_slots))
        
        # Objective: Maximize total weighted utility over all time slots
        objective_terms = []
        for t in range(self.time_slots):
            for i in range(self.n_users):
                objective_terms.append(priorities[i] * cp.log(x[i, t] + 1e-6))
        
        objective = cp.Maximize(cp.sum(objective_terms))
        
        # Constraints
        constraints = []
        
        # 1. Capacity constraint at each time slot
        for t in range(self.time_slots):
            constraints.append(cp.sum(x[:, t]) <= capacities[t])
        
        # 2. Min/max bandwidth constraints
        for i in range(self.n_users):
            for t in range(self.time_slots):
                constraints.append(x[i, t] >= min_bandwidth[i])
                constraints.append(x[i, t] <= max_bandwidth[i])
        
        # 3. Temporal fairness: average allocation should meet threshold
        for i in range(self.n_users):
            avg_demand = np.mean(demands[i, :])
            if avg_demand > 0:
                constraints.append(
                    cp.sum(x[i, :]) / self.time_slots >= temporal_fairness_threshold * avg_demand
                )
        
        # 4. Non-negativity
        constraints.append(x >= 0)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                raise ValueError(f"Optimization failed: {problem.status}")
            
            allocation = x.value
            solve_time = time.time() - start_time
            
            # Calculate metrics
            metrics = self._calculate_temporal_metrics(allocation, demands, capacities)
            
            return {
                'status': 'optimal',
                'allocation': allocation,  # Shape: (n_users, time_slots)
                'objective_value': problem.value,
                'solve_time': solve_time,
                'metrics': metrics
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'solve_time': time.time() - start_time
            }
    
    def _calculate_temporal_metrics(self, allocation: np.ndarray, demands: np.ndarray, capacities: np.ndarray) -> Dict:
        """Calculate comprehensive temporal metrics."""
        
        # Utilization per time slot
        utilization = np.sum(allocation, axis=0) / capacities
        
        # Satisfaction per user per time slot
        satisfaction = np.minimum(allocation / (demands + 1e-6), 1.0)
        
        # Temporal fairness: variance in allocation over time for each user
        temporal_variance = np.var(allocation, axis=1)
        
        # Peak vs off-peak analysis
        peak_hours = [9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22]
        off_peak_hours = [0, 1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 23]
        
        peak_utilization = np.mean(utilization[peak_hours])
        off_peak_utilization = np.mean(utilization[off_peak_hours])
        
        # Jain's fairness index per time slot
        fairness_per_slot = []
        for t in range(self.time_slots):
            alloc_t = allocation[:, t]
            fairness = (np.sum(alloc_t) ** 2) / (self.n_users * np.sum(alloc_t ** 2))
            fairness_per_slot.append(fairness)
        
        return {
            'avg_utilization': np.mean(utilization),
            'utilization_per_slot': utilization,
            'peak_utilization': peak_utilization,
            'off_peak_utilization': off_peak_utilization,
            'avg_satisfaction': np.mean(satisfaction),
            'satisfaction_per_user': np.mean(satisfaction, axis=1),
            'temporal_variance': temporal_variance,
            'avg_temporal_variance': np.mean(temporal_variance),
            'fairness_per_slot': fairness_per_slot,
            'avg_fairness': np.mean(fairness_per_slot),
            'total_allocated': np.sum(allocation),
            'total_capacity': np.sum(capacities)
        }
    
    def analyze_peak_hours(self, allocation: np.ndarray, demands: np.ndarray) -> Dict:
        """
        Analyze allocation during peak vs off-peak hours.
        
        Returns:
            Detailed peak hour analysis
        """
        peak_hours = list(range(9, 17)) + list(range(19, 23))  # 9am-5pm, 7pm-11pm
        off_peak_hours = [h for h in range(24) if h not in peak_hours]
        
        peak_allocation = allocation[:, peak_hours]
        off_peak_allocation = allocation[:, off_peak_hours]
        
        peak_demands = demands[:, peak_hours]
        off_peak_demands = demands[:, off_peak_hours]
        
        return {
            'peak_hours': peak_hours,
            'off_peak_hours': off_peak_hours,
            'peak_avg_allocation': np.mean(peak_allocation, axis=1),
            'off_peak_avg_allocation': np.mean(off_peak_allocation, axis=1),
            'peak_total_allocation': np.sum(peak_allocation),
            'off_peak_total_allocation': np.sum(off_peak_allocation),
            'peak_satisfaction': np.mean(np.minimum(peak_allocation / (peak_demands + 1e-6), 1.0)),
            'off_peak_satisfaction': np.mean(np.minimum(off_peak_allocation / (off_peak_demands + 1e-6), 1.0))
        }
    
    def forecast_future_demands(self, historical_demands: np.ndarray, forecast_horizon: int = 24) -> np.ndarray:
        """
        Simple demand forecasting using moving average and trend analysis.
        
        Args:
            historical_demands: Past demands of shape (n_users, past_time_slots)
            forecast_horizon: Number of future time slots to forecast
            
        Returns:
            Forecasted demands of shape (n_users, forecast_horizon)
        """
        n_users = historical_demands.shape[0]
        forecasts = np.zeros((n_users, forecast_horizon))
        
        for i in range(n_users):
            hist = historical_demands[i, :]
            
            # Simple moving average
            window_size = min(7, len(hist))
            recent_avg = np.mean(hist[-window_size:])
            
            # Trend
            if len(hist) > 1:
                trend = (hist[-1] - hist[0]) / len(hist)
            else:
                trend = 0
            
            # Forecast with trend
            for t in range(forecast_horizon):
                forecast = recent_avg + trend * t
                # Add seasonality (daily pattern)
                hour = t % 24
                seasonality = self._get_seasonality_factor(hour)
                forecasts[i, t] = forecast * seasonality
        
        return forecasts
    
    def _get_seasonality_factor(self, hour: int) -> float:
        """Get seasonality multiplier for given hour."""
        # Peak hours get higher factor
        if 9 <= hour <= 17 or 19 <= hour <= 22:
            return 1.2
        elif 0 <= hour <= 5:
            return 0.6
        else:
            return 1.0


class TemporalAnalyzer:
    """
    Advanced temporal analysis tools for time-varying allocations.
    """
    
    @staticmethod
    def calculate_temporal_fairness_index(allocation: np.ndarray) -> float:
        """
        Calculate temporal fairness: how evenly is bandwidth distributed over time?
        
        Args:
            allocation: Shape (n_users, time_slots)
            
        Returns:
            Temporal fairness index [0, 1]
        """
        n_users, time_slots = allocation.shape
        
        # Calculate coefficient of variation for each user's allocation over time
        cv_per_user = []
        for i in range(n_users):
            user_alloc = allocation[i, :]
            mean_alloc = np.mean(user_alloc)
            std_alloc = np.std(user_alloc)
            cv = std_alloc / (mean_alloc + 1e-6)
            cv_per_user.append(cv)
        
        # Lower CV means better temporal fairness
        # Convert to [0, 1] scale where 1 is best
        avg_cv = np.mean(cv_per_user)
        temporal_fairness = 1.0 / (1.0 + avg_cv)
        
        return temporal_fairness
    
    @staticmethod
    def detect_congestion_periods(allocation: np.ndarray, capacities: np.ndarray, threshold: float = 0.9) -> List[int]:
        """
        Detect time slots with high congestion (utilization > threshold).
        
        Returns:
            List of congested time slot indices
        """
        utilization = np.sum(allocation, axis=0) / capacities
        congested_slots = np.where(utilization > threshold)[0].tolist()
        return congested_slots
    
    @staticmethod
    def calculate_load_balancing_score(allocation: np.ndarray, capacities: np.ndarray) -> float:
        """
        Calculate how well load is balanced across time slots.
        Returns value in [0, 1] where 1 is perfectly balanced.
        """
        utilization = np.sum(allocation, axis=0) / capacities
        
        # Use coefficient of variation
        mean_util = np.mean(utilization)
        std_util = np.std(utilization)
        cv = std_util / (mean_util + 1e-6)
        
        # Convert to score
        balance_score = 1.0 / (1.0 + cv)
        return balance_score
