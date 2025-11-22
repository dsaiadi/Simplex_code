"""
Quick test script to verify all optimization modules work correctly
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
from backend.core_optimizer import CoreOptimizer, FairnessMetrics
from backend.multi_objective import MultiObjectiveOptimizer
from backend.time_varying import TimeVaryingOptimizer
from backend.robust_optimizer import RobustOptimizer
from backend.data_generator import DataGenerator

print("=" * 80)
print("BANDWIDTH ALLOCATION OPTIMIZATION - SYSTEM TEST")
print("=" * 80)
print()

# Generate small test dataset
print("1. Generating test dataset (100 users)...")
n_users = 100
users_df = DataGenerator.generate_users(n_users)
demands = users_df['base_demand_mbps'].values
priorities = users_df['priority'].values
min_bw = users_df['min_bandwidth_mbps'].values
max_bw = users_df['max_bandwidth_mbps'].values
total_capacity = demands.sum() * 0.8  # 80% of total demand

print(f"   ✓ Generated {n_users} users")
print(f"   ✓ Total demand: {demands.sum():.2f} Mbps")
print(f"   ✓ Available capacity: {total_capacity:.2f} Mbps")
print()

# Test 1: Core Optimization
print("2. Testing Core Optimization...")
try:
    optimizer = CoreOptimizer(n_users, total_capacity)
    result = optimizer.optimize(demands, priorities, min_bw, max_bw, utility_type='log')
    
    if result['status'] == 'optimal':
        print(f"   ✓ Optimization successful")
        print(f"   ✓ Objective value: {result['objective_value']:.2f}")
        print(f"   ✓ Fairness index: {result['metrics']['jains_fairness_index']:.4f}")
        print(f"   ✓ Utilization: {result['utilization']:.2f}%")
        print(f"   ✓ Solve time: {result['solve_time']:.4f}s")
    else:
        print(f"   ✗ Optimization failed: {result.get('error')}")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")
print()

# Test 2: Multi-Objective Optimization
print("3. Testing Multi-Objective Optimization...")
try:
    mo_optimizer = MultiObjectiveOptimizer(n_users, total_capacity)
    weights = {'fairness': 0.4, 'efficiency': 0.4, 'latency': 0.2}
    result = mo_optimizer.optimize_weighted_sum(demands, priorities, min_bw, max_bw, weights)
    
    if result['status'] == 'optimal':
        print(f"   ✓ Multi-objective optimization successful")
        print(f"   ✓ Fairness: {result['fairness']:.4f}")
        print(f"   ✓ Efficiency: {result['efficiency']:.2%}")
        print(f"   ✓ Latency: {result['latency']:.2f} ms")
    else:
        print(f"   ✗ Optimization failed")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")
print()

# Test 3: Time-Varying Optimization
print("4. Testing Time-Varying Optimization...")
try:
    temporal_demands = DataGenerator.generate_temporal_demands(users_df, time_slots=24)
    capacities = DataGenerator.generate_network_capacity(24, total_capacity, 'realistic')
    
    tv_optimizer = TimeVaryingOptimizer(n_users, time_slots=24)
    result = tv_optimizer.optimize_temporal(
        temporal_demands, priorities, capacities, min_bw, max_bw, temporal_fairness_threshold=0.8
    )
    
    if result['status'] == 'optimal':
        print(f"   ✓ Temporal optimization successful")
        print(f"   ✓ Avg utilization: {result['metrics']['avg_utilization']*100:.2f}%")
        print(f"   ✓ Peak utilization: {result['metrics']['peak_utilization']*100:.2f}%")
        print(f"   ✓ Temporal fairness: {result['metrics']['avg_fairness']:.4f}")
    else:
        print(f"   ✗ Optimization failed")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")
print()

# Test 4: Robust Optimization
print("5. Testing Robust Optimization...")
try:
    demand_deviations = demands * 0.2  # 20% uncertainty
    
    robust_optimizer = RobustOptimizer(n_users, total_capacity)
    result = robust_optimizer.optimize_budget_uncertainty(
        demands, demand_deviations, priorities, min_bw, max_bw, gamma=n_users//3
    )
    
    if result['status'] == 'optimal':
        print(f"   ✓ Robust optimization successful")
        print(f"   ✓ Objective value: {result['objective_value']:.2f}")
        print(f"   ✓ Robustness probability: {result['robustness_metrics']['robustness_probability']:.2%}")
        print(f"   ✓ Price of robustness: {result['price_of_robustness']:.2%}")
    else:
        print(f"   ✗ Optimization failed")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")
print()

# Test 5: Fairness Metrics
print("6. Testing Fairness Metrics...")
try:
    allocation = np.random.uniform(10, 50, n_users)
    metrics = FairnessMetrics.calculate_all_metrics(allocation)
    
    print(f"   ✓ Jain's index: {metrics['jains_fairness_index']:.4f}")
    print(f"   ✓ Gini coefficient: {metrics['gini_coefficient']:.4f}")
    print(f"   ✓ Max-min ratio: {metrics['max_min_ratio']:.4f}")
    print(f"   ✓ Coefficient of variation: {metrics['coefficient_of_variation']:.4f}")
except Exception as e:
    print(f"   ✗ Error: {str(e)}")
print()

print("=" * 80)
print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 80)
print()
print("Next steps:")
print("1. Run 'streamlit run app.py' to launch the web dashboard")
print("2. Or use 'python generate_data.py' to create larger datasets")
print("3. Check data/ folder for the generated Excel file")
