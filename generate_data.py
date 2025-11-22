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
    
    # Generate uncertainty scenarios
    print("Generating uncertainty scenarios...")
    uncertainty_data = DataGenerator.generate_uncertainty_scenarios(
        users_df,
        uncertainty_level=0.2
    )
    print(f"✓ Generated uncertainty data")
    
    # Export to CSV
    print("\nExporting data to CSV...")
    output_file = "data/bandwidth_allocation_10000_users.csv"
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    users_df.to_csv(output_file, index=False)
    print(f"✓ Exported to {output_file}")
    
    # Export uncertainty data
    print("\nExporting uncertainty data...")
    uncertainty_df = pd.DataFrame({
        'user_id': users_df['user_id'],
        'nominal_demand': uncertainty_data['nominal_demands'],
        'demand_deviation': uncertainty_data['demand_deviations']
    })
    uncertainty_df.to_csv("data/uncertainty_data.csv", index=False)
    print(f"✓ Exported uncertainty data")
    
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
    print("\n" + "=" * 80)
    print("✓ Data generation completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
