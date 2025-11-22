"""
PERSON 1, 2, 3: Main Streamlit Application
Interactive dashboard for Internet Bandwidth Allocation Optimization
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.core_optimizer import CoreOptimizer, FairnessMetrics
from backend.multi_objective import MultiObjectiveOptimizer, ParetoAnalyzer
from backend.robust_optimizer import RobustOptimizer, UncertaintyGenerator
from backend.data_generator import DataGenerator
from backend.visualizer import BandwidthVisualizer, ReportGenerator
from backend.benchmark_algorithms import BenchmarkAlgorithms

# Page configuration
st.set_page_config(
    page_title="Bandwidth Allocation Optimizer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'users_df' not in st.session_state:
        st.session_state.users_df = None
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = {}
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = BandwidthVisualizer()


def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">📡 Internet Bandwidth Allocation Optimization System</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; font-size: 16px; color: #666; margin-bottom: 30px;'>
    Optimal bandwidth distribution using convex optimization, balancing fairness, efficiency, and robustness
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio(
        "Select Module:",
        [
            "🏠 Home & Data Generation",
            "⚙️ Core Optimization",
            "🎯 Multi-Objective Optimization",
            "🛡️ Robust Optimization",
            "🔬 Benchmarking & Comparison",
            "📊 Analysis & Comparison",
            "📈 Visualization Dashboard"
        ]
    )
    
    # Route to appropriate page
    if page == "🏠 Home & Data Generation":
        home_and_data_page()
    elif page == "⚙️ Core Optimization":
        core_optimization_page()
    elif page == "🎯 Multi-Objective Optimization":
        multi_objective_page()
    elif page == "🛡️ Robust Optimization":
        robust_optimization_page()
    elif page == "🔬 Benchmarking & Comparison":
        benchmarking_page()
    elif page == "📊 Analysis & Comparison":
        analysis_page()
    elif page == "📈 Visualization Dashboard":
        visualization_page()


def home_and_data_page():
    """Home page with data generation."""
    st.markdown('<p class="sub-header">Welcome to Bandwidth Allocation Optimizer</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎓 About This System
    
    This advanced optimization system solves the **Internet Bandwidth Allocation Problem** using:
    
    - **Convex Optimization (CVXPY)**: Guaranteed global optimal solutions
    - **Multiple Utility Functions**: Log (proportional fairness), sqrt, linear, alpha-fair
    - **Multi-Objective Optimization**: Balance fairness, efficiency, and latency
    - **Time-Varying Optimization**: Handle temporal demand patterns (24-hour horizon)
    - **Robust Optimization**: Handle demand uncertainty (Box, Budget, Ellipsoidal uncertainty sets)
    
    #### 📐 Mathematical Formulation
    
    **Objective:**
    """)
    
    st.latex(r'''
    \max \sum_{i=1}^{n} w_i \cdot U(x_i)
    ''')
    
    st.markdown("**Subject to:**")
    
    st.latex(r'''
    \begin{aligned}
    &\sum_{i=1}^{n} x_i \leq C & \text{(Capacity)} \\
    &x_{i,\min} \leq x_i \leq x_{i,\max} & \text{(Min/Max limits)} \\
    &x_i \geq 0 & \text{(Non-negativity)}
    \end{aligned}
    ''')
    
    st.markdown("---")
    
    # Data Generation Section
    st.markdown('<p class="sub-header">📊 Generate User Dataset</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_users = st.number_input(
            "Number of Users",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="Generate dataset with this many users"
        )
        
        time_slots = st.number_input(
            "Time Slots (hours)",
            min_value=12,
            max_value=48,
            value=24,
            help="Number of time slots for temporal analysis"
        )
    
    with col2:
        base_capacity = st.number_input(
            "Total Network Capacity (Mbps)",
            min_value=1000.0,
            max_value=100000.0,
            value=50000.0,
            step=1000.0,
            help="Total available bandwidth"
        )
        
        capacity_pattern = st.selectbox(
            "Capacity Pattern",
            ["constant", "realistic", "dynamic"],
            help="How capacity varies over time"
        )
    
    if st.button("🚀 Generate Dataset", use_container_width=True):
        with st.spinner("Generating realistic user data..."):
            # Generate users
            users_df = DataGenerator.generate_users(n_users)
            st.session_state.users_df = users_df
            
            # Generate temporal demands
            temporal_demands = DataGenerator.generate_temporal_demands(users_df, time_slots)
            st.session_state.temporal_demands = temporal_demands
            
            # Generate capacity
            capacities = DataGenerator.generate_network_capacity(
                time_slots, base_capacity, capacity_pattern
            )
            st.session_state.capacities = capacities
            
            st.success(f"✅ Successfully generated {n_users} users with {time_slots}-hour temporal demands!")
    
    # Display dataset if generated
    if st.session_state.users_df is not None:
        st.markdown("---")
        st.markdown('<p class="sub-header">📋 Generated Dataset Summary</p>', unsafe_allow_html=True)
        
        df = st.session_state.users_df
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Users", len(df))
        with col2:
            st.metric("Total Demand", f"{df['base_demand_mbps'].sum():.0f} Mbps")
        with col3:
            st.metric("Avg Demand", f"{df['base_demand_mbps'].mean():.1f} Mbps")
        with col4:
            st.metric("Total Capacity", f"{st.session_state.get('capacities', [base_capacity])[0]:.0f} Mbps")
        
        # User type distribution
        st.markdown("#### User Type Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            type_dist = df['user_type_name'].value_counts()
            st.bar_chart(type_dist)
        
        with col2:
            priority_dist = df['priority'].value_counts().sort_index()
            st.bar_chart(priority_dist)
        
        # Sample data
        st.markdown("#### Sample User Data")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Download options
        st.markdown("#### 💾 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="users_data.csv",
                mime="text/csv"
            )
        
        with col2:
            if st.button("Export to Excel", use_container_width=True):
                filename = f"bandwidth_data_{n_users}_users.xlsx"
                DataGenerator.export_to_excel(
                    df, 
                    st.session_state.temporal_demands,
                    filename
                )
                st.success(f"Exported to {filename}")


def core_optimization_page():
    """Core optimization module page."""
    st.markdown('<p class="sub-header">⚙️ Core Bandwidth Optimization</p>', unsafe_allow_html=True)
    
    if st.session_state.users_df is None:
        st.warning("⚠️ Please generate dataset first from the Home page!")
        return
    
    df = st.session_state.users_df
    
    st.markdown("""
    Solve the fundamental bandwidth allocation problem using different utility functions.
    
    **Utility Functions:**
    - **Log (Proportional Fairness)**: U(x) = log(x) - RECOMMENDED
    - **Sqrt (Balanced Fairness)**: U(x) = √x
    - **Linear (Pure Efficiency)**: U(x) = x
    - **Alpha-Fair**: U(x) = x^(1-α)/(1-α)
    """)
    
    # Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        n_users = st.slider("Number of Users to Optimize", 
                           min_value=100, 
                           max_value=min(5000, len(df)), 
                           value=min(1000, len(df)),
                           help="Subset of users for faster computation")
        
        utility_type = st.selectbox(
            "Utility Function",
            ["log", "sqrt", "linear", "alpha-fair"],
            help="Type of utility function for fairness"
        )
    
    with col2:
        total_capacity = st.number_input(
            "Total Capacity (Mbps)",
            min_value=100.0,
            max_value=100000.0,
            value=float(df['base_demand_mbps'].head(n_users).sum() * 0.8),
            help="Total available bandwidth"
        )
        
        alpha = 0.5
        if utility_type == "alpha-fair":
            alpha = st.slider("Alpha Parameter", 
                            min_value=0.1, 
                            max_value=2.0, 
                            value=0.5,
                            help="Alpha parameter for alpha-fair utility")
    
    if st.button("🔧 Run Optimization", use_container_width=True):
        with st.spinner("Solving optimization problem..."):
            # Get subset of users
            subset_df = df.head(n_users)
            
            demands = subset_df['base_demand_mbps'].values
            priorities = subset_df['priority'].values
            min_bw = subset_df['min_bandwidth_mbps'].values
            max_bw = subset_df['max_bandwidth_mbps'].values
            
            # Create optimizer
            optimizer = CoreOptimizer(n_users, total_capacity)
            
            # Solve
            result = optimizer.optimize(
                demands, priorities, min_bw, max_bw,
                utility_type=utility_type,
                alpha=alpha
            )
            
            st.session_state.optimization_results['core'] = result
            
            if result['status'] == 'optimal':
                st.success("✅ Optimization completed successfully!")
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Objective Value", f"{result['objective_value']:.2f}")
                with col2:
                    st.metric("Solve Time", f"{result['solve_time']:.4f}s")
                with col3:
                    st.metric("Utilization", f"{result['utilization']:.1f}%")
                with col4:
                    st.metric("Fairness Index", 
                            f"{result['metrics']['jains_fairness_index']:.4f}")
                
                # Metrics
                st.markdown("#### 📊 Performance Metrics")
                metrics = result['metrics']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Average Satisfaction:** {metrics['avg_satisfaction']:.4f}")
                    st.write(f"**Weighted Satisfaction:** {metrics['weighted_satisfaction']:.4f}")
                    st.write(f"**Allocation Std Dev:** {metrics['allocation_std']:.2f} Mbps")
                
                with col2:
                    st.write(f"**Min Allocation:** {metrics['min_allocation']:.2f} Mbps")
                    st.write(f"**Max Allocation:** {metrics['max_allocation']:.2f} Mbps")
                    st.write(f"**Median Allocation:** {metrics['median_allocation']:.2f} Mbps")
                
                # Visualization
                st.markdown("#### 📈 Allocation Visualization")
                
                visualizer = st.session_state.visualizer
                
                # Show allocation vs demand for sample users
                n_show = min(30, n_users)
                fig = visualizer.plot_allocation_comparison(
                    {'Allocated': result['allocation']},
                    demands,
                    n_users_to_show=n_show
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Satisfaction distribution
                fig2 = visualizer.plot_user_satisfaction(
                    result['allocation'], demands, priorities
                )
                st.plotly_chart(fig2, use_container_width=True)
                
            else:
                st.error(f"❌ Optimization failed: {result.get('error', 'Unknown error')}")
    
    # Compare utility functions
    st.markdown("---")
    st.markdown("#### 🔬 Compare Utility Functions")
    
    if st.button("Compare All Utility Functions", use_container_width=True):
        with st.spinner("Comparing different utility functions..."):
            subset_df = df.head(n_users)
            demands = subset_df['base_demand_mbps'].values
            priorities = subset_df['priority'].values
            min_bw = subset_df['min_bandwidth_mbps'].values
            max_bw = subset_df['max_bandwidth_mbps'].values
            
            optimizer = CoreOptimizer(n_users, total_capacity)
            comparison = optimizer.compare_utility_functions(
                demands, priorities, min_bw, max_bw
            )
            
            # Create comparison DataFrame
            comparison_data = []
            for util_name, res in comparison.items():
                if res['status'] == 'optimal':
                    comparison_data.append({
                        'Utility Function': util_name,
                        'Objective Value': res['objective_value'],
                        'Fairness Index': res['metrics']['jains_fairness_index'],
                        'Avg Satisfaction': res['metrics']['avg_satisfaction'],
                        'Utilization (%)': res['utilization'],
                        'Solve Time (s)': res['solve_time']
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visualize fairness metrics
            fairness_metrics = {
                name: FairnessMetrics.calculate_all_metrics(res['allocation'])
                for name, res in comparison.items() if res['status'] == 'optimal'
            }
            
            fig = st.session_state.visualizer.plot_fairness_metrics(fairness_metrics)
            st.plotly_chart(fig, use_container_width=True)


def multi_objective_page():
    """Multi-objective optimization page."""
    st.markdown('<p class="sub-header">🎯 Multi-Objective Optimization</p>', unsafe_allow_html=True)
    
    if st.session_state.users_df is None:
        st.warning("⚠️ Please generate dataset first from the Home page!")
        return
    
    df = st.session_state.users_df
    
    st.markdown("""
    Balance multiple competing objectives:
    1. **Fairness**: Maximize Jain's fairness index
    2. **Efficiency**: Maximize total bandwidth utilization
    3. **Latency**: Minimize average latency
    
    **Methods:**
    - Weighted Sum Method
    - Epsilon-Constraint Method
    - Pareto Frontier Generation
    """)
    
    method = st.radio(
        "Select Method:",
        ["Weighted Sum", "Epsilon-Constraint", "Pareto Frontier"]
    )
    
    # Common parameters
    n_users = st.slider("Number of Users", 100, min(3000, len(df)), min(1000, len(df)))
    total_capacity = st.number_input(
        "Total Capacity (Mbps)",
        min_value=100.0,
        value=float(df['base_demand_mbps'].head(n_users).sum() * 0.8)
    )
    
    subset_df = df.head(n_users)
    demands = subset_df['base_demand_mbps'].values
    priorities = subset_df['priority'].values
    min_bw = subset_df['min_bandwidth_mbps'].values
    max_bw = subset_df['max_bandwidth_mbps'].values
    
    optimizer = MultiObjectiveOptimizer(n_users, total_capacity)
    
    if method == "Weighted Sum":
        st.markdown("#### ⚖️ Set Objective Weights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            w_fairness = st.slider("Fairness Weight", 0.0, 1.0, 0.4, 0.1)
        with col2:
            w_efficiency = st.slider("Efficiency Weight", 0.0, 1.0, 0.4, 0.1)
        with col3:
            w_latency = st.slider("Latency Weight", 0.0, 1.0, 0.2, 0.1)
        
        # Normalize weights
        total_weight = w_fairness + w_efficiency + w_latency
        if total_weight > 0:
            weights = {
                'fairness': w_fairness / total_weight,
                'efficiency': w_efficiency / total_weight,
                'latency': w_latency / total_weight
            }
        else:
            weights = {'fairness': 0.4, 'efficiency': 0.4, 'latency': 0.2}
        
        if st.button("🎯 Optimize", use_container_width=True):
            with st.spinner("Solving multi-objective optimization..."):
                result = optimizer.optimize_weighted_sum(
                    demands, priorities, min_bw, max_bw, weights
                )
                
                if result['status'] == 'optimal':
                    st.success("✅ Optimization completed!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Fairness", f"{result['fairness']:.4f}")
                    with col2:
                        st.metric("Efficiency", f"{result['efficiency']:.2%}")
                    with col3:
                        st.metric("Latency", f"{result['latency']:.2f} ms")
                    with col4:
                        st.metric("Solve Time", f"{result['solve_time']:.4f}s")
                    
                    st.session_state.optimization_results['multi_objective'] = result
                else:
                    st.error(f"❌ Failed: {result.get('error')}")
    
    elif method == "Pareto Frontier":
        n_points = st.slider("Number of Pareto Points", 5, 30, 15)
        
        if st.button("🔍 Generate Pareto Frontier", use_container_width=True):
            with st.spinner(f"Generating {n_points} Pareto-optimal solutions..."):
                pareto_results = optimizer.generate_pareto_frontier(
                    demands, priorities, min_bw, max_bw, n_points
                )
                
                st.success(f"✅ Found {pareto_results['n_pareto_points']} Pareto-optimal solutions!")
                
                # Visualize Pareto frontier
                fig = st.session_state.visualizer.plot_pareto_frontier(
                    pareto_results['fairness_values'],
                    pareto_results['efficiency_values'],
                    pareto_results['latency_values']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show Pareto points data
                pareto_df = pd.DataFrame({
                    'Fairness': pareto_results['fairness_values'],
                    'Efficiency': pareto_results['efficiency_values'],
                    'Latency (ms)': pareto_results['latency_values']
                })
                st.dataframe(pareto_df, use_container_width=True)


def robust_optimization_page():
    """Robust optimization page."""
    st.markdown('<p class="sub-header">🛡️ Robust Optimization Under Uncertainty</p>', 
                unsafe_allow_html=True)
    
    if st.session_state.users_df is None:
        st.warning("⚠️ Please generate dataset first!")
        return
    
    df = st.session_state.users_df
    
    st.markdown("""
    Handle demand uncertainty using three robust optimization models:
    
    1. **Box Uncertainty**: U_i = [d_i - δ_i, d_i + δ_i]
    2. **Budget Uncertainty (Bertsimas-Sim)**: At most Γ demands deviate
    3. **Ellipsoidal Uncertainty**: ||d - d̄||₂ ≤ Ω
    """)
    
    # Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        n_users = st.slider("Number of Users", 100, min(2000, len(df)), min(500, len(df)))
        uncertainty_type = st.selectbox(
            "Uncertainty Model",
            ["box", "budget", "ellipsoidal"]
        )
    
    with col2:
        total_capacity = st.number_input(
            "Total Capacity (Mbps)",
            min_value=100.0,
            value=float(df['base_demand_mbps'].head(n_users).sum() * 0.75)
        )
        uncertainty_level = st.slider(
            "Uncertainty Level",
            0.1, 0.5, 0.2, 0.05,
            help="Fraction of demand that can deviate"
        )
    
    # Model-specific parameters
    if uncertainty_type == "budget":
        gamma = st.slider(
            "Gamma (Budget Parameter)",
            1, n_users, n_users // 3,
            help="Number of demands that can deviate"
        )
    elif uncertainty_type == "ellipsoidal":
        base_omega = np.sqrt(n_users) * df['base_demand_mbps'].head(n_users).mean() * 0.1
        omega = st.slider(
            "Omega (Ellipsoid Radius)",
            base_omega * 0.5, base_omega * 2.0,
            base_omega,
            help="Radius of uncertainty ellipsoid"
        )
    
    if st.button("🛡️ Run Robust Optimization", use_container_width=True):
        with st.spinner(f"Solving robust optimization with {uncertainty_type} uncertainty..."):
            subset_df = df.head(n_users)
            nominal_demands = subset_df['base_demand_mbps'].values
            demand_deviations = nominal_demands * uncertainty_level
            priorities = subset_df['priority'].values
            min_bw = subset_df['min_bandwidth_mbps'].values
            max_bw = subset_df['max_bandwidth_mbps'].values
            
            optimizer = RobustOptimizer(n_users, total_capacity)
            
            if uncertainty_type == "box":
                result = optimizer.optimize_box_uncertainty(
                    nominal_demands, demand_deviations, priorities, min_bw, max_bw
                )
            elif uncertainty_type == "budget":
                result = optimizer.optimize_budget_uncertainty(
                    nominal_demands, demand_deviations, priorities, min_bw, max_bw, gamma
                )
            else:  # ellipsoidal
                result = optimizer.optimize_ellipsoidal_uncertainty(
                    nominal_demands, priorities, min_bw, max_bw, omega
                )
            
            if result['status'] == 'optimal':
                st.success("✅ Robust optimization completed!")
                
                st.session_state.optimization_results['robust'] = result
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Objective Value", f"{result['objective_value']:.2f}")
                with col2:
                    st.metric("Price of Robustness", f"{result['price_of_robustness']:.2%}")
                with col3:
                    rob_prob = result['robustness_metrics']['robustness_probability']
                    st.metric("Robustness Probability", f"{rob_prob:.2%}")
                with col4:
                    st.metric("Solve Time", f"{result['solve_time']:.4f}s")
                
                # Robustness analysis
                st.markdown("#### 🔍 Robustness Analysis")
                rob_metrics = result['robustness_metrics']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Feasible Scenarios:** {rob_metrics['feasible_scenarios']}/{rob_metrics['total_scenarios']}")
                    st.write(f"**Average Violation:** {rob_metrics['avg_violation']:.2f} Mbps")
                with col2:
                    st.write(f"**Max Violation:** {rob_metrics['max_violation']:.2f} Mbps")
                    st.write(f"**Uncertainty Type:** {result['uncertainty_type']}")
                
            else:
                st.error(f"❌ Optimization failed: {result.get('error')}")
    
    # Compare uncertainty models
    st.markdown("---")
    st.markdown("#### 📊 Compare Uncertainty Models")
    
    if st.button("Compare All Models", use_container_width=True):
        with st.spinner("Comparing uncertainty models..."):
            subset_df = df.head(n_users)
            nominal_demands = subset_df['base_demand_mbps'].values
            demand_deviations = nominal_demands * uncertainty_level
            priorities = subset_df['priority'].values
            min_bw = subset_df['min_bandwidth_mbps'].values
            max_bw = subset_df['max_bandwidth_mbps'].values
            
            optimizer = RobustOptimizer(n_users, total_capacity)
            comparison = optimizer.compare_uncertainty_models(
                nominal_demands, demand_deviations, priorities, min_bw, max_bw
            )
            
            # Create comparison table
            comparison_data = []
            for model_name, res in comparison.items():
                if res['status'] == 'optimal':
                    comparison_data.append({
                        'Model': model_name,
                        'Objective': res['objective_value'],
                        'Price of Robustness': f"{res['price_of_robustness']:.2%}",
                        'Robustness Prob': f"{res['robustness_metrics']['robustness_probability']:.2%}",
                        'Solve Time': f"{res['solve_time']:.4f}s"
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)


def analysis_page():
    """Comprehensive analysis and comparison page."""
    st.markdown('<p class="sub-header">📊 Analysis & Comparison</p>', unsafe_allow_html=True)
    
    if not st.session_state.optimization_results:
        st.warning("⚠️ No optimization results available. Please run some optimizations first!")
        return
    
    st.markdown("Compare results from different optimization approaches.")
    
    results = st.session_state.optimization_results
    
    # Summary table
    st.markdown("#### 📋 Results Summary")
    
    summary_data = []
    for method, result in results.items():
        if result.get('status') == 'optimal':
            row = {
                'Method': method.replace('_', ' ').title(),
                'Objective': result.get('objective_value', 'N/A'),
                'Solve Time (s)': result.get('solve_time', 'N/A')
            }
            
            if 'metrics' in result:
                row['Fairness'] = result['metrics'].get('jains_fairness_index', 'N/A')
            if 'fairness' in result:
                row['Fairness'] = result.get('fairness', 'N/A')
            if 'utilization' in result:
                row['Utilization (%)'] = result.get('utilization', 'N/A')
            
            summary_data.append(row)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
    
    # Detailed reports
    st.markdown("#### 📄 Detailed Reports")
    
    selected_method = st.selectbox(
        "Select method for detailed report:",
        list(results.keys())
    )
    
    if selected_method in results:
        result = results[selected_method]
        report = ReportGenerator.generate_summary_report(result)
        st.text(report)


def visualization_page():
    """Main visualization dashboard."""
    st.markdown('<p class="sub-header">📈 Visualization Dashboard</p>', unsafe_allow_html=True)
    
    if not st.session_state.optimization_results:
        st.warning("⚠️ No results to visualize. Please run optimizations first!")
        return
    
    st.markdown("Interactive visualizations of optimization results.")
    
    # Select result to visualize
    method = st.selectbox(
        "Select optimization result:",
        list(st.session_state.optimization_results.keys())
    )
    
    result = st.session_state.optimization_results[method]
    
    if result.get('status') == 'optimal':
        visualizer = st.session_state.visualizer
        
        # Dashboard summary
        fig = visualizer.create_dashboard_summary(result)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Selected result is not optimal.")


def benchmarking_page():
    """Comprehensive benchmarking and algorithm comparison."""
    st.markdown('<p class="sub-header">🔬 Benchmarking & Algorithm Comparison</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Compare 10 different bandwidth allocation algorithms:
    - **Baseline Algorithms**: Equal Share, Proportional, Max-Min, Weighted Max-Min, Nash Bargaining, Greedy, Round Robin, Water-Filling
    - **Optimization Methods**: Convex Optimization (Log & Sqrt utilities)
    """)
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        num_users = st.slider("Number of Users", 10, 1000, 100, step=10)
    with col2:
        total_capacity = st.slider("Total Capacity (Mbps)", 100, 10000, 1000, step=100)
    
    # Generate test scenario
    if st.button("🎲 Generate Test Scenario", type="primary"):
        generator = DataGenerator()
        test_data = generator.generate_users(num_users)
        
        st.session_state['benchmark_data'] = {
            'users': test_data['users'],
            'demands': test_data['demands'],
            'priorities': test_data['priorities'],
            'total_capacity': total_capacity
        }
        
        st.success(f"✅ Generated test scenario with {num_users} users")
        
        # Show summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Users", num_users)
        with col2:
            st.metric("Total Demand", f"{sum(test_data['demands'].values()):.0f} Mbps")
        with col3:
            st.metric("Available Capacity", f"{total_capacity} Mbps")
    
    # Run benchmarks
    if st.button("🚀 Run All Benchmarks", type="primary") and 'benchmark_data' in st.session_state:
        data = st.session_state['benchmark_data']
        
        with st.spinner("Running 10 algorithms..."):
            import time
            start_time = time.time()
            
            benchmark = BenchmarkAlgorithms(
                demands=data['demands'],
                priorities=data['priorities'],
                total_capacity=data['total_capacity']
            )
            
            # Run all benchmarks
            results = benchmark.run_all_benchmarks()
            elapsed = time.time() - start_time
            
            st.session_state['benchmark_results'] = results
            
            st.success(f"✅ Completed all benchmarks in {elapsed:.2f}s")
    
    # Display results
    if 'benchmark_results' in st.session_state:
        results = st.session_state['benchmark_results']
        
        st.markdown("---")
        st.markdown("### 📊 Results Comparison")
        
        # Create comparison table
        import pandas as pd
        
        table_data = []
        for algo_name, result in results.items():
            table_data.append({
                'Algorithm': algo_name,
                'Fairness Index': f"{result['fairness_index']:.4f}",
                'Efficiency (%)': f"{result['efficiency']*100:.1f}",
                'Total Utility': f"{result['total_utility']:.2f}",
                'Solve Time (ms)': f"{result['solve_time']*1000:.2f}"
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        # Rankings
        st.markdown("---")
        st.markdown("### 🏆 Algorithm Rankings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🎯 Best Fairness**")
            sorted_fairness = sorted(results.items(), 
                                    key=lambda x: x[1]['fairness_index'], 
                                    reverse=True)
            for i, (name, res) in enumerate(sorted_fairness[:3], 1):
                st.write(f"{i}. {name}: {res['fairness_index']:.4f}")
        
        with col2:
            st.markdown("**⚡ Best Efficiency**")
            sorted_efficiency = sorted(results.items(), 
                                      key=lambda x: x[1]['efficiency'], 
                                      reverse=True)
            for i, (name, res) in enumerate(sorted_efficiency[:3], 1):
                st.write(f"{i}. {name}: {res['efficiency']*100:.1f}%")
        
        with col3:
            st.markdown("**🚀 Fastest**")
            sorted_speed = sorted(results.items(), 
                                 key=lambda x: x[1]['solve_time'])
            for i, (name, res) in enumerate(sorted_speed[:3], 1):
                st.write(f"{i}. {name}: {res['solve_time']*1000:.2f}ms")
        
        # Visualization
        st.markdown("---")
        st.markdown("### 📈 Visual Comparison")
        
        metric = st.selectbox(
            "Select Metric to Visualize:",
            ["Fairness Index", "Efficiency", "Total Utility", "Solve Time"]
        )
        
        import plotly.graph_objects as go
        
        algorithms = list(results.keys())
        
        if metric == "Fairness Index":
            values = [results[algo]['fairness_index'] for algo in algorithms]
            ylabel = "Jain's Fairness Index"
        elif metric == "Efficiency":
            values = [results[algo]['efficiency']*100 for algo in algorithms]
            ylabel = "Efficiency (%)"
        elif metric == "Total Utility":
            values = [results[algo]['total_utility'] for algo in algorithms]
            ylabel = "Total Utility"
        else:  # Solve Time
            values = [results[algo]['solve_time']*1000 for algo in algorithms]
            ylabel = "Solve Time (ms)"
        
        fig = go.Figure(data=[
            go.Bar(
                x=algorithms,
                y=values,
                text=[f"{v:.2f}" for v in values],
                textposition='auto',
                marker_color='rgb(55, 83, 109)'
            )
        ])
        
        fig.update_layout(
            title=f"Algorithm Comparison: {metric}",
            xaxis_title="Algorithm",
            yaxis_title=ylabel,
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.markdown("---")
        st.markdown("### 💡 Key Insights")
        
        best_fairness = max(results.items(), key=lambda x: x[1]['fairness_index'])
        best_efficiency = max(results.items(), key=lambda x: x[1]['efficiency'])
        fastest = min(results.items(), key=lambda x: x[1]['solve_time'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **🎯 Most Fair Algorithm**: {best_fairness[0]}
            - Fairness Index: {best_fairness[1]['fairness_index']:.4f}
            - Efficiency: {best_fairness[1]['efficiency']*100:.1f}%
            """)
            
            st.info(f"""
            **⚡ Most Efficient Algorithm**: {best_efficiency[0]}
            - Efficiency: {best_efficiency[1]['efficiency']*100:.1f}%
            - Fairness Index: {best_efficiency[1]['fairness_index']:.4f}
            """)
        
        with col2:
            st.info(f"""
            **🚀 Fastest Algorithm**: {fastest[0]}
            - Solve Time: {fastest[1]['solve_time']*1000:.2f}ms
            - Fairness Index: {fastest[1]['fairness_index']:.4f}
            """)
            
            # Recommendation
            convex_log = results.get('Convex Optimization (Log)', {})
            if convex_log:
                st.success(f"""
                **✨ Recommended**: Convex Optimization (Log)
                - Balances fairness ({convex_log['fairness_index']:.4f}) and efficiency ({convex_log['efficiency']*100:.1f}%)
                - Optimal solution with mathematical guarantees
                """)


if __name__ == "__main__":
    main()

