"""
Enhanced Data Generation Module with User Tiers
Supports: Emergency Services, Premium, and Free user tiers with priority-based allocation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import random
from datetime import datetime


class EnhancedDataGenerator:
    """
    Generate realistic bandwidth allocation datasets with user tiers.
    
    User Tiers:
    - Emergency Services (911, hospitals, police, fire): Highest priority, unlimited bandwidth
    - Premium Users: High priority, guaranteed bandwidth, lower latency
    - Free Users: Best effort, lower priority, bandwidth optimization needed
    """
    
    # Tier configuration
    TIER_CONFIG = {
        'emergency': {
            'priority_range': (9, 10),
            'bandwidth_multiplier': 3.0,
            'min_guarantee': 0.9,  # 90% of demand guaranteed
            'latency_tolerance': 5,  # ms
            'cost': 0,  # Free for emergency services
            'qos': 99.9
        },
        'premium': {
            'priority_range': (6, 8),
            'bandwidth_multiplier': 1.5,
            'min_guarantee': 0.7,  # 70% of demand guaranteed
            'latency_tolerance': 20,  # ms
            'cost_per_mbps': 2.0,
            'qos': 99.0
        },
        'free': {
            'priority_range': (1, 5),
            'bandwidth_multiplier': 1.0,
            'min_guarantee': 0.3,  # 30% of demand guaranteed
            'latency_tolerance': 100,  # ms
            'cost_per_mbps': 0.5,
            'qos': 95.0
        }
    }
    
    @staticmethod
    def generate_users(n_users: int = 10000,
                      emergency_pct: float = 0.02,  # 2% emergency
                      premium_pct: float = 0.25,     # 25% premium
                      seed: int | None = None) -> pd.DataFrame:
        """
        Generate diverse user dataset with emergency, premium, and free tiers.
        
        Args:
            n_users: Total number of users
            emergency_pct: Percentage of emergency service users
            premium_pct: Percentage of premium users
            seed: Random seed for reproducibility (None for random)
            
        Returns:
            DataFrame with user information including tier classification
        """
        # Set random seed for variety (None = different each time)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Calculate tier distribution
        n_emergency = int(n_users * emergency_pct)
        n_premium = int(n_users * premium_pct)
        n_free = n_users - n_emergency - n_premium
        
        users_data = []
        user_id = 1
        
        # Generate Emergency Service Users
        for _ in range(n_emergency):
            users_data.append(EnhancedDataGenerator._generate_emergency_user(user_id))
            user_id += 1
        
        # Generate Premium Users
        for _ in range(n_premium):
            users_data.append(EnhancedDataGenerator._generate_premium_user(user_id))
            user_id += 1
        
        # Generate Free Users
        for _ in range(n_free):
            users_data.append(EnhancedDataGenerator._generate_free_user(user_id))
            user_id += 1
        
        # Create DataFrame
        df = pd.DataFrame(users_data)
        
        # Shuffle to mix tiers
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Add computed fields
        df['bandwidth_cost'] = df.apply(
            lambda row: EnhancedDataGenerator._calculate_cost(row), axis=1
        )
        
        df['allocation_weight'] = df.apply(
            lambda row: EnhancedDataGenerator._calculate_allocation_weight(row), axis=1
        )
        
        return df
    
    @staticmethod
    def _generate_emergency_user(user_id: int) -> Dict:
        """Generate emergency service user profile."""
        emergency_types = [
            '911 Emergency', 'Hospital', 'Police', 'Fire Department',
            'Ambulance', 'Emergency Medical', 'Disaster Response',
            'Critical Infrastructure', 'Military', 'Coast Guard'
        ]
        
        service_type = random.choice(emergency_types)
        
        # Emergency services need high, consistent bandwidth
        base_demand = np.random.uniform(50, 200)  # High bandwidth needs
        
        return {
            'user_id': user_id,
            'tier': 'emergency',
            'tier_name': 'Emergency Services',
            'service_type': service_type,
            'priority': np.random.uniform(9, 10),  # Highest priority
            'base_demand_mbps': base_demand,
            'min_bandwidth_mbps': base_demand * 0.9,  # 90% guaranteed
            'max_bandwidth_mbps': base_demand * 2.0,  # Can burst
            'latency_requirement_ms': np.random.uniform(5, 15),
            'qos_requirement': np.random.uniform(99.9, 100),
            'reliability_requirement': 'CRITICAL',
            'application_type': service_type,
            'region': random.choice(['North', 'South', 'East', 'West', 'Central']),
            'user_type': 'emergency',
            'user_type_code': 0,
            'sla_tier': 'Emergency',
            'latency_sensitivity': 10,
            'throttle_allowed': False,
            'preemption_allowed': True  # Can preempt other users
        }
    
    @staticmethod
    def _generate_premium_user(user_id: int) -> Dict:
        """Generate premium subscriber profile."""
        premium_profiles = {
            'Business Executive': (30, 150, 'Video Conferencing'),
            'Professional Gamer': (100, 300, 'Gaming'),
            'Content Creator': (50, 200, 'Video Streaming/Upload'),
            'Remote Worker': (20, 100, 'Cloud Services'),
            'Small Business': (40, 150, 'VPN/Cloud'),
            'Tech Company': (100, 500, 'Data Transfer'),
            'Media Studio': (200, 800, 'Video Production'),
            'Financial Services': (50, 250, 'Real-time Trading')
        }
        
        profile_name = random.choice(list(premium_profiles.keys()))
        demand_range, max_demand, app_type = premium_profiles[profile_name]
        
        base_demand = np.random.uniform(demand_range, max_demand)
        
        return {
            'user_id': user_id,
            'tier': 'premium',
            'tier_name': 'Premium Subscriber',
            'service_type': profile_name,
            'priority': np.random.uniform(6, 8),
            'base_demand_mbps': base_demand,
            'min_bandwidth_mbps': base_demand * 0.7,  # 70% guaranteed
            'max_bandwidth_mbps': base_demand * 1.5,
            'latency_requirement_ms': np.random.uniform(10, 30),
            'qos_requirement': np.random.uniform(98, 99.5),
            'reliability_requirement': 'HIGH',
            'application_type': app_type,
            'region': random.choice(['North', 'South', 'East', 'West', 'Central']),
            'user_type': 'premium',
            'user_type_code': 1,
            'sla_tier': random.choice(['Premium', 'Premium Plus', 'Business']),
            'latency_sensitivity': np.random.randint(7, 10),
            'throttle_allowed': False,
            'preemption_allowed': False
        }
    
    @staticmethod
    def _generate_free_user(user_id: int) -> Dict:
        """Generate free tier user profile."""
        free_profiles = {
            'Casual Browser': (2, 10, 'Web Browsing'),
            'Social Media User': (5, 20, 'Social Media'),
            'Email User': (1, 5, 'Email'),
            'Light Streaming': (5, 25, 'Video Streaming'),
            'Student': (10, 50, 'Education'),
            'Home User': (5, 30, 'Mixed Usage'),
            'Night Downloader': (10, 50, 'Downloads'),
            'IoT Devices': (1, 10, 'IoT'),
            'Basic Phone': (2, 15, 'VoIP'),
            'Residential': (5, 40, 'General')
        }
        
        profile_name = random.choice(list(free_profiles.keys()))
        demand_range, max_demand, app_type = free_profiles[profile_name]
        
        base_demand = np.random.uniform(demand_range, max_demand)
        
        return {
            'user_id': user_id,
            'tier': 'free',
            'tier_name': 'Free Tier',
            'service_type': profile_name,
            'priority': np.random.uniform(1, 5),
            'base_demand_mbps': base_demand,
            'min_bandwidth_mbps': base_demand * 0.3,  # 30% guaranteed
            'max_bandwidth_mbps': base_demand * 1.2,
            'latency_requirement_ms': np.random.uniform(50, 200),
            'qos_requirement': np.random.uniform(90, 96),
            'reliability_requirement': 'STANDARD',
            'application_type': app_type,
            'region': random.choice(['North', 'South', 'East', 'West', 'Central']),
            'user_type': 'free',
            'user_type_code': 2,
            'sla_tier': random.choice(['Basic', 'Standard', 'Free']),
            'latency_sensitivity': np.random.randint(3, 7),
            'throttle_allowed': True,
            'preemption_allowed': False
        }
    
    @staticmethod
    def _calculate_cost(row: pd.Series) -> float:
        """Calculate monthly cost for user."""
        if row['tier'] == 'emergency':
            return 0.0  # Emergency services don't pay
        elif row['tier'] == 'premium':
            return row['base_demand_mbps'] * EnhancedDataGenerator.TIER_CONFIG['premium']['cost_per_mbps']
        else:  # free
            return row['base_demand_mbps'] * EnhancedDataGenerator.TIER_CONFIG['free']['cost_per_mbps']
    
    @staticmethod
    def _calculate_allocation_weight(row: pd.Series) -> float:
        """
        Calculate allocation weight for optimization.
        Emergency > Premium > Free
        """
        if row['tier'] == 'emergency':
            return row['priority'] * 10.0  # 10x weight
        elif row['tier'] == 'premium':
            return row['priority'] * 3.0   # 3x weight
        else:  # free
            return row['priority'] * 1.0   # 1x weight
    
    @staticmethod
    def generate_dynamic_demands(users_df: pd.DataFrame, 
                                time_slots: int = 24,
                                congestion_level: str = 'moderate') -> np.ndarray:
        """
        Generate time-varying demands with realistic patterns.
        Different each time due to noise and variations.
        
        Args:
            users_df: User DataFrame
            time_slots: Number of time periods
            congestion_level: 'low', 'moderate', 'high', 'critical'
            
        Returns:
            Array of shape (n_users, time_slots)
        """
        n_users = len(users_df)
        demands = np.zeros((n_users, time_slots))
        
        # Congestion multipliers
        congestion_factors = {
            'low': (0.7, 0.1),      # (mean, std)
            'moderate': (1.0, 0.2),
            'high': (1.3, 0.3),
            'critical': (1.8, 0.4)
        }
        
        cong_mean, cong_std = congestion_factors.get(congestion_level, (1.0, 0.2))
        
        for i, row in users_df.iterrows():
            base_demand = row['base_demand_mbps']
            tier = row['tier']
            service_type = row['service_type']
            
            # Different patterns based on user type
            if tier == 'emergency':
                # Emergency: constant high demand with random spikes
                pattern = np.ones(time_slots) * 0.8
                # Random emergency spikes
                spike_hours = np.random.choice(time_slots, size=np.random.randint(2, 5), replace=False)
                pattern[spike_hours] = 1.5
                
            elif 'Business' in service_type or 'Professional' in service_type:
                # Business hours pattern (9am-5pm)
                pattern = EnhancedDataGenerator._business_pattern(time_slots)
                
            elif 'Gaming' in service_type or 'Streaming' in service_type:
                # Evening/night pattern
                pattern = EnhancedDataGenerator._entertainment_pattern(time_slots)
                
            elif 'Night' in service_type or 'Download' in service_type:
                # Night pattern
                pattern = EnhancedDataGenerator._night_pattern(time_slots)
                
            else:
                # Residential mixed pattern
                pattern = EnhancedDataGenerator._residential_pattern(time_slots)
            
            # Add realistic noise and congestion
            noise = np.random.normal(1.0, 0.15, time_slots)
            noise = np.maximum(noise, 0.2)  # Minimum 20% demand
            
            congestion = np.random.normal(cong_mean, cong_std, time_slots)
            congestion = np.maximum(congestion, 0.5)
            
            demands[i, :] = base_demand * pattern * noise * congestion
        
        return demands
    
    @staticmethod
    def _business_pattern(time_slots: int = 24) -> np.ndarray:
        """Business hours pattern with variations."""
        pattern = np.ones(time_slots) * 0.15
        pattern[6:9] = 0.5   # Morning ramp-up
        pattern[9:12] = 1.0  # Peak morning
        pattern[12:14] = 0.7 # Lunch dip
        pattern[14:17] = 1.0 # Peak afternoon
        pattern[17:19] = 0.6 # Evening decline
        # Add small random variations
        pattern += np.random.uniform(-0.05, 0.05, time_slots)
        return np.maximum(pattern, 0.1)
    
    @staticmethod
    def _entertainment_pattern(time_slots: int = 24) -> np.ndarray:
        """Gaming/Streaming evening pattern."""
        pattern = np.ones(time_slots) * 0.2
        pattern[18:24] = 1.0  # Evening peak
        pattern[0:2] = 0.8    # Late night
        pattern[14:18] = 0.5  # Afternoon
        pattern += np.random.uniform(-0.05, 0.05, time_slots)
        return np.maximum(pattern, 0.1)
    
    @staticmethod
    def _night_pattern(time_slots: int = 24) -> np.ndarray:
        """Night usage pattern."""
        pattern = np.ones(time_slots) * 0.15
        pattern[22:24] = 1.0
        pattern[0:6] = 0.9
        pattern += np.random.uniform(-0.05, 0.05, time_slots)
        return np.maximum(pattern, 0.1)
    
    @staticmethod
    def _residential_pattern(time_slots: int = 24) -> np.ndarray:
        """General residential pattern."""
        pattern = np.ones(time_slots) * 0.25
        pattern[7:9] = 0.6   # Morning
        pattern[12:14] = 0.5 # Lunch
        pattern[18:23] = 1.0 # Evening peak
        pattern += np.random.uniform(-0.05, 0.05, time_slots)
        return np.maximum(pattern, 0.1)
    
    @staticmethod
    def generate_emergency_scenarios(users_df: pd.DataFrame,
                                    scenario_type: str = 'normal') -> Dict:
        """
        Generate emergency scenarios for testing.
        
        Scenarios:
        - normal: Regular operation
        - disaster: Natural disaster (earthquakes, hurricanes)
        - cyber_attack: DDoS or cyber incident
        - mass_event: Large public event
        - infrastructure_failure: Major infrastructure failure
        
        Returns:
            Dictionary with scenario parameters
        """
        n_users = len(users_df)
        
        scenarios = {
            'normal': {
                'emergency_demand_multiplier': 1.0,
                'premium_demand_multiplier': 1.0,
                'free_demand_multiplier': 1.0,
                'capacity_reduction': 0.0,
                'description': 'Normal operating conditions'
            },
            'disaster': {
                'emergency_demand_multiplier': 3.0,  # 3x emergency traffic
                'premium_demand_multiplier': 1.5,
                'free_demand_multiplier': 0.5,      # Throttle free users
                'capacity_reduction': 0.2,           # 20% capacity loss
                'description': 'Natural disaster - Emergency services priority'
            },
            'cyber_attack': {
                'emergency_demand_multiplier': 2.0,
                'premium_demand_multiplier': 1.2,
                'free_demand_multiplier': 0.3,      # Heavy throttling
                'capacity_reduction': 0.4,           # 40% capacity loss
                'description': 'Cyber attack - Critical services only'
            },
            'mass_event': {
                'emergency_demand_multiplier': 1.5,
                'premium_demand_multiplier': 2.0,   # High premium demand
                'free_demand_multiplier': 1.8,      # High free demand
                'capacity_reduction': 0.1,
                'description': 'Mass event - High congestion'
            },
            'infrastructure_failure': {
                'emergency_demand_multiplier': 2.5,
                'premium_demand_multiplier': 0.8,
                'free_demand_multiplier': 0.2,      # Severe throttling
                'capacity_reduction': 0.5,           # 50% capacity loss
                'description': 'Infrastructure failure - Emergency only'
            }
        }
        
        return scenarios.get(scenario_type, scenarios['normal'])
    
    @staticmethod
    def export_enhanced_data(users_df: pd.DataFrame,
                           temporal_demands: np.ndarray,
                           filename: str = None):
        """Export enhanced dataset to Excel with timestamp."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'bandwidth_data_{timestamp}.xlsx'
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # User information
            users_df.to_excel(writer, sheet_name='Users', index=False)
            
            # Temporal demands
            demand_columns = [f'Hour_{h:02d}' for h in range(temporal_demands.shape[1])]
            temporal_df = pd.DataFrame(temporal_demands, columns=demand_columns)
            temporal_df.insert(0, 'user_id', users_df['user_id'])
            temporal_df.to_excel(writer, sheet_name='Temporal_Demands', index=False)
            
            # Tier summary
            tier_summary = users_df.groupby('tier').agg({
                'user_id': 'count',
                'base_demand_mbps': ['sum', 'mean', 'min', 'max'],
                'priority': 'mean',
                'bandwidth_cost': 'sum'
            }).reset_index()
            tier_summary.to_excel(writer, sheet_name='Tier_Summary', index=False)
            
            # Service type distribution
            service_dist = users_df.groupby(['tier', 'service_type']).size().reset_index(name='count')
            service_dist.to_excel(writer, sheet_name='Service_Distribution', index=False)
            
            # Regional analysis
            regional = users_df.groupby(['region', 'tier']).agg({
                'user_id': 'count',
                'base_demand_mbps': 'sum'
            }).reset_index()
            regional.to_excel(writer, sheet_name='Regional_Analysis', index=False)
            
        return filename
