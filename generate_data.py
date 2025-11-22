"""
Script to generate sample data files for the bandwidth allocation system.
Run this to create the 10,000 user dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from backend.data_generator import DataGenerator
import pandas as pd
import numpy as np

def main():
    print("=" * 80)
    print("BANDWIDTH ALLOCATION DATA GENERATOR")
    print("=" * 80)
    print()
    
    # Generate 10,000 users
    print("Generating 10,000 users...")
    users_df = DataGenerator.generate_users(n_users=10000)
    print(f"✓ Generated {len(users_df)} users")
    
    # Generate temporal demands (24 hours)
    print("Generating 24-hour temporal demand patterns...")
    temporal_demands = DataGenerator.generate_temporal_demands(users_df, time_slots=24)
    print(f"✓ Generated temporal demands: {temporal_demands.shape}")
    
    # Generate network capacity
    print("Generating network capacity profile...")
    capacities = DataGenerator.generate_network_capacity(
        time_slots=24,
        base_capacity=50000.0,
        pattern='realistic'
    )
    print(f"✓ Generated capacity profile")
    
    # Generate uncertainty scenarios
    print("Generating uncertainty scenarios...")
    uncertainty_data = DataGenerator.generate_uncertainty_scenarios(
        users_df,
        uncertainty_level=0.2
    )
    print(f"✓ Generated uncertainty data")
    
    # Export to Excel
    print("\nExporting data to Excel...")
    output_file = "data/bandwidth_allocation_10000_users.xlsx"
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    DataGenerator.export_to_excel(users_df, temporal_demands, output_file)
    print(f"✓ Exported to {output_file}")
    
    # Export additional data
    print("\nExporting additional datasets...")
    
    # Capacity data
    capacity_df = pd.DataFrame({
        'Hour': range(24),
        'Capacity_Mbps': capacities,
        'Time_Label': [f"{h:02d}:00" for h in range(24)]
    })
    
    # Uncertainty data
    uncertainty_df = pd.DataFrame({
        'user_id': users_df['user_id'],
        'nominal_demand': uncertainty_data['nominal_demands'],
        'demand_deviation': uncertainty_data['demand_deviations']
    })
    
    # Save additional sheets
    with pd.ExcelWriter(output_file, mode='a', engine='openpyxl') as writer:
        capacity_df.to_excel(writer, sheet_name='Capacity_Profile', index=False)
        uncertainty_df.to_excel(writer, sheet_name='Uncertainty_Data', index=False)
    
    print(f"✓ Added additional data sheets")
    
    # Generate summary statistics
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"Total Users: {len(users_df)}")
    print(f"Total Base Demand: {users_df['base_demand_mbps'].sum():.0f} Mbps")
    print(f"Average Demand per User: {users_df['base_demand_mbps'].mean():.2f} Mbps")
    print(f"Max Demand: {users_df['base_demand_mbps'].max():.2f} Mbps")
    print(f"Min Demand: {users_df['base_demand_mbps'].min():.2f} Mbps")
    print(f"\nUser Type Distribution:")
    print(users_df['user_type_name'].value_counts())
    print(f"\nPriority Distribution:")
    print(users_df['priority'].value_counts().sort_index())
    print(f"\nAverage Network Capacity: {capacities.mean():.0f} Mbps")
    print(f"Peak Capacity: {capacities.max():.0f} Mbps")
    print(f"Off-peak Capacity: {capacities.min():.0f} Mbps")
    print("\n" + "=" * 80)
    print("✓ Data generation completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
