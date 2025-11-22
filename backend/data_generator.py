"""
PERSON 3: Data Generation Module
Generates realistic datasets for bandwidth allocation testing with 10,000+ users.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import random


class DataGenerator:
    """
    Generate realistic bandwidth allocation datasets.
    """
    
    @staticmethod
    def generate_users(n_users: int = 10000) -> pd.DataFrame:
        """
        Generate user dataset with realistic attributes.
        
        User types:
        - 0: Business (30%)
        - 1: Residential (50%)
        - 2: Night users (15%)
        - 3: Constant/Always-on (5%)
        
        Returns:
            DataFrame with user information
        """
        # User type distribution
        type_probs = [0.30, 0.50, 0.15, 0.05]
        user_types = np.random.choice([0, 1, 2, 3], size=n_users, p=type_probs)
        
        # Priority levels (1-5, with weights)
        priority_mapping = {
            0: [3, 4, 5],  # Business: mostly high priority
            1: [2, 3, 4],  # Residential: medium priority
            2: [1, 2, 3],  # Night: lower priority
            3: [4, 5]      # Always-on: high priority
        }
        
        priorities = np.zeros(n_users)
        for i in range(n_users):
            priorities[i] = np.random.choice(priority_mapping[user_types[i]])
        
        # Base bandwidth demands (Mbps)
        # Business: 10-100 Mbps, Residential: 5-50 Mbps, Night: 5-30 Mbps, Always-on: 20-150 Mbps
        demands = np.zeros(n_users)
        for i in range(n_users):
            if user_types[i] == 0:  # Business
                demands[i] = np.random.uniform(10, 100)
            elif user_types[i] == 1:  # Residential
                demands[i] = np.random.uniform(5, 50)
            elif user_types[i] == 2:  # Night
                demands[i] = np.random.uniform(5, 30)
            else:  # Always-on
                demands[i] = np.random.uniform(20, 150)
        
        # Minimum bandwidth guarantees (10% of demand)
        min_bandwidth = demands * 0.1
        
        # Maximum bandwidth limits (150% of demand)
        max_bandwidth = demands * 1.5
        
        # Service level agreements (SLA tiers)
        sla_tiers = np.random.choice(['Basic', 'Standard', 'Premium', 'Enterprise'], 
                                      size=n_users, 
                                      p=[0.4, 0.3, 0.2, 0.1])
        
        # Geographic regions
        regions = np.random.choice(['North', 'South', 'East', 'West', 'Central'], 
                                    size=n_users,
                                    p=[0.2, 0.2, 0.25, 0.25, 0.1])
        
        # Application types
        app_types = []
        for user_type in user_types:
            if user_type == 0:  # Business
                app_types.append(np.random.choice(['Video Conferencing', 'Cloud Services', 'VPN', 'File Transfer']))
            elif user_type == 1:  # Residential
                app_types.append(np.random.choice(['Video Streaming', 'Gaming', 'Social Media', 'Web Browsing']))
            elif user_type == 2:  # Night
                app_types.append(np.random.choice(['Downloads', 'Backups', 'Gaming', 'Streaming']))
            else:  # Always-on
                app_types.append(np.random.choice(['Server', 'IoT', 'Monitoring', 'Database']))
        
        # Create DataFrame
        df = pd.DataFrame({
            'user_id': range(1, n_users + 1),
            'user_type': user_types,
            'user_type_name': ['Business' if t == 0 else 'Residential' if t == 1 else 'Night' if t == 2 else 'Always-on' 
                               for t in user_types],
            'priority': priorities,
            'base_demand_mbps': demands,
            'min_bandwidth_mbps': min_bandwidth,
            'max_bandwidth_mbps': max_bandwidth,
            'sla_tier': sla_tiers,
            'region': regions,
            'application_type': app_types,
            'subscription_cost': DataGenerator._calculate_subscription_cost(demands, priorities, sla_tiers),
            'latency_sensitivity': DataGenerator._calculate_latency_sensitivity(app_types),
            'qos_requirement': DataGenerator._calculate_qos_requirement(sla_tiers)
        })
        
        return df
    
    @staticmethod
    def _calculate_subscription_cost(demands: np.ndarray, priorities: np.ndarray, sla_tiers: np.ndarray) -> np.ndarray:
        """Calculate monthly subscription cost based on demand, priority, and SLA."""
        base_cost = demands * 0.5  # $0.50 per Mbps
        priority_multiplier = priorities / 3.0
        
        sla_multipliers = {
            'Basic': 1.0,
            'Standard': 1.3,
            'Premium': 1.6,
            'Enterprise': 2.0
        }
        
        costs = np.zeros(len(demands))
        for i in range(len(demands)):
            costs[i] = base_cost[i] * priority_multiplier[i] * sla_multipliers[sla_tiers[i]]
        
        return costs
    
    @staticmethod
    def _calculate_latency_sensitivity(app_types: List[str]) -> np.ndarray:
        """Calculate latency sensitivity score (1-10)."""
        sensitivity_map = {
            'Video Conferencing': 9,
            'Gaming': 10,
            'Video Streaming': 7,
            'Cloud Services': 8,
            'VPN': 7,
            'File Transfer': 4,
            'Social Media': 6,
            'Web Browsing': 6,
            'Downloads': 3,
            'Backups': 2,
            'Server': 9,
            'IoT': 8,
            'Monitoring': 7,
            'Database': 9
        }
        
        return np.array([sensitivity_map.get(app, 5) for app in app_types])
    
    @staticmethod
    def _calculate_qos_requirement(sla_tiers: np.ndarray) -> np.ndarray:
        """Calculate QoS requirement score (1-100)."""
        qos_map = {
            'Basic': np.random.uniform(50, 70),
            'Standard': np.random.uniform(70, 85),
            'Premium': np.random.uniform(85, 95),
            'Enterprise': np.random.uniform(95, 100)
        }
        
        return np.array([qos_map[tier] for tier in sla_tiers])
    
    @staticmethod
    def generate_temporal_demands(users_df: pd.DataFrame, time_slots: int = 24) -> np.ndarray:
        """
        Generate time-varying demands for all users.
        
        Returns:
            Array of shape (n_users, time_slots)
        """
        n_users = len(users_df)
        demands = np.zeros((n_users, time_slots))
        
        for i, row in users_df.iterrows():
            base_demand = row['base_demand_mbps']
            user_type = row['user_type']
            
            # Generate pattern based on user type
            if user_type == 0:  # Business
                pattern = DataGenerator._business_pattern(time_slots)
            elif user_type == 1:  # Residential
                pattern = DataGenerator._residential_pattern(time_slots)
            elif user_type == 2:  # Night
                pattern = DataGenerator._night_pattern(time_slots)
            else:  # Always-on
                pattern = DataGenerator._constant_pattern(time_slots)
            
            # Add realistic noise
            noise = np.random.normal(1.0, 0.15, time_slots)
            noise = np.maximum(noise, 0.3)  # Ensure minimum
            
            demands[i, :] = base_demand * pattern * noise
        
        return demands
    
    @staticmethod
    def _business_pattern(time_slots: int = 24) -> np.ndarray:
        """Business hours pattern (9am-5pm)."""
        pattern = np.ones(time_slots) * 0.2
        pattern[9:17] = 1.0
        pattern[7:9] = 0.6
        pattern[17:19] = 0.5
        return pattern
    
    @staticmethod
    def _residential_pattern(time_slots: int = 24) -> np.ndarray:
        """Residential pattern (evening peak)."""
        pattern = np.ones(time_slots) * 0.3
        pattern[19:23] = 1.0
        pattern[6:9] = 0.5
        pattern[12:14] = 0.6
        return pattern
    
    @staticmethod
    def _night_pattern(time_slots: int = 24) -> np.ndarray:
        """Night user pattern."""
        pattern = np.ones(time_slots) * 0.2
        pattern[22:24] = 1.0
        pattern[0:6] = 0.9
        return pattern
    
    @staticmethod
    def _constant_pattern(time_slots: int = 24) -> np.ndarray:
        """Always-on constant pattern."""
        return np.ones(time_slots) * 0.9 + np.random.uniform(-0.1, 0.1, time_slots)
    
    @staticmethod
    def generate_network_capacity(time_slots: int = 24, 
                                   base_capacity: float = 50000.0,
                                   pattern: str = 'dynamic') -> np.ndarray:
        """
        Generate time-varying network capacity.
        
        Args:
            base_capacity: Base total capacity (Mbps)
            pattern: 'constant', 'dynamic', or 'realistic'
            
        Returns:
            Array of capacity per time slot
        """
        if pattern == 'constant':
            return np.ones(time_slots) * base_capacity
        
        elif pattern == 'realistic':
            # Slightly higher capacity during off-peak
            capacity = np.ones(time_slots) * base_capacity
            capacity[0:6] *= 1.15
            capacity[9:17] *= 0.95
            capacity[19:23] *= 0.9
            return capacity
        
        elif pattern == 'dynamic':
            # Simulate variable network conditions
            capacity = base_capacity * (0.85 + 0.3 * np.random.rand(time_slots))
            return capacity
        
        else:
            return np.ones(time_slots) * base_capacity
    
    @staticmethod
    def generate_uncertainty_scenarios(users_df: pd.DataFrame, 
                                        uncertainty_level: float = 0.2) -> Dict:
        """
        Generate uncertainty scenarios for robust optimization.
        
        Args:
            uncertainty_level: Fraction of demand that can deviate (0-1)
            
        Returns:
            Dictionary with uncertainty parameters
        """
        n_users = len(users_df)
        base_demands = users_df['base_demand_mbps'].values
        
        # Demand deviations based on uncertainty level
        demand_deviations = base_demands * uncertainty_level
        
        # Add user-specific uncertainty factors
        # High priority users have lower uncertainty
        priority_factor = 1.0 / (users_df['priority'].values / 3.0)
        demand_deviations *= priority_factor
        
        # Application type affects uncertainty
        latency_sensitivity = users_df['latency_sensitivity'].values
        uncertainty_factor = 1.0 + (10 - latency_sensitivity) / 20.0
        demand_deviations *= uncertainty_factor
        
        return {
            'nominal_demands': base_demands,
            'demand_deviations': demand_deviations,
            'uncertainty_level': uncertainty_level,
            'correlation_matrix': DataGenerator._generate_correlation_matrix(n_users, users_df)
        }
    
    @staticmethod
    def _generate_correlation_matrix(n_users: int, users_df: pd.DataFrame) -> np.ndarray:
        """
        Generate correlation matrix for demand uncertainties.
        Users in same region/type have correlated demands.
        """
        corr_matrix = np.eye(n_users)
        
        # Add correlation based on regions
        for region in users_df['region'].unique():
            region_users = users_df[users_df['region'] == region].index.values
            for i in region_users:
                for j in region_users:
                    if i != j:
                        corr_matrix[i, j] = 0.3  # Moderate correlation
        
        # Add correlation based on user types
        for user_type in users_df['user_type'].unique():
            type_users = users_df[users_df['user_type'] == user_type].index.values
            for i in type_users:
                for j in type_users:
                    if i != j:
                        corr_matrix[i, j] = max(corr_matrix[i, j], 0.2)
        
        return corr_matrix
    
    @staticmethod
    def export_to_excel(users_df: pd.DataFrame, 
                        temporal_demands: np.ndarray,
                        filename: str = 'bandwidth_allocation_data.xlsx'):
        """
        Export generated data to Excel file.
        
        Args:
            users_df: User information DataFrame
            temporal_demands: Time-varying demands array
            filename: Output filename
        """
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # User information
            users_df.to_excel(writer, sheet_name='Users', index=False)
            
            # Temporal demands
            demand_columns = [f'Hour_{h:02d}' for h in range(temporal_demands.shape[1])]
            temporal_df = pd.DataFrame(temporal_demands, columns=demand_columns)
            temporal_df.insert(0, 'user_id', users_df['user_id'])
            temporal_df.to_excel(writer, sheet_name='Temporal_Demands', index=False)
            
            # Summary statistics
            summary_df = pd.DataFrame({
                'Metric': [
                    'Total Users',
                    'Total Base Demand (Mbps)',
                    'Avg Demand per User (Mbps)',
                    'Max Demand (Mbps)',
                    'Min Demand (Mbps)',
                    'Business Users',
                    'Residential Users',
                    'Night Users',
                    'Always-on Users'
                ],
                'Value': [
                    len(users_df),
                    users_df['base_demand_mbps'].sum(),
                    users_df['base_demand_mbps'].mean(),
                    users_df['base_demand_mbps'].max(),
                    users_df['base_demand_mbps'].min(),
                    len(users_df[users_df['user_type'] == 0]),
                    len(users_df[users_df['user_type'] == 1]),
                    len(users_df[users_df['user_type'] == 2]),
                    len(users_df[users_df['user_type'] == 3])
                ]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Priority distribution
            priority_dist = users_df.groupby('priority').size().reset_index(name='count')
            priority_dist.to_excel(writer, sheet_name='Priority_Distribution', index=False)
            
            # Regional distribution
            regional_dist = users_df.groupby('region').agg({
                'user_id': 'count',
                'base_demand_mbps': ['sum', 'mean']
            }).reset_index()
            regional_dist.columns = ['Region', 'User Count', 'Total Demand', 'Avg Demand']
            regional_dist.to_excel(writer, sheet_name='Regional_Distribution', index=False)
