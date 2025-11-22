"""
Enhanced Streamlit Application with Tier-Based Bandwidth Allocation
Emergency Services > Premium > Free Users
Dynamic data generation, emergency scenarios, and amazing visualizations
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime
import time as time_module

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.data_generator_enhanced import EnhancedDataGenerator
from backend.tier_optimizer import TierBasedOptimizer
from backend.core_optimizer import FairnessMetrics

# Page configuration
st.set_page_config(
    page_title="🚀 Advanced Bandwidth Optimizer - Tier System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with wow factor
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 52px;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from {
            filter: drop-shadow(0 0 5px #667eea);
        }
        to {
            filter: drop-shadow(0 0 20px #764ba2);
        }
    }
    
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #555;
        margin-bottom: 30px;
        font-weight: 500;
    }
    
    .tier-emergency {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
    }
    
    .tier-premium {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
    }
    
    .tier-free {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(67, 233, 123, 0.4);
    }
    
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin: 10px 0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 15px 30px;
        border-radius: 25px;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .scenario-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
        font-size: 14px;
    }
    
    .scenario-normal {
        background: #4caf50;
        color: white;
    }
    
    .scenario-disaster {
        background: #ff5252;
        color: white;
    }
    
    .scenario-cyber {
        background: #ff9800;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


def create_tier_visualization(users_df):
    """Create beautiful tier distribution visualization."""
    tier_counts = users_df['tier'].value_counts()
    
    colors = {
        'emergency': '#f5576c',
        'premium': '#4facfe',
        'free': '#43e97b'
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=[f"{tier.title()} Users" for tier in tier_counts.index],
        values=tier_counts.values,
        hole=0.4,
        marker=dict(colors=[colors.get(tier, '#999') for tier in tier_counts.index]),
        textinfo='label+percent+value',
        textfont=dict(size=14, color='white', family='Arial Black'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '👥 User Tier Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'family': 'Orbitron', 'color': '#333'}
        },
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig


def create_allocation_comparison_chart(results_dict):
    """Create comparison chart across tiers."""
    tiers = ['emergency', 'premium', 'free']
    tier_names = ['🚨 Emergency', '⭐ Premium', '📱 Free']
    colors = ['#f5576c', '#4facfe', '#43e97b']
    
    allocated = []
    demanded = []
    satisfaction = []
    
    for tier in tiers:
        stats = results_dict['tier_statistics'][tier]
        if stats:
            allocated.append(stats['total_allocated'])
            demanded.append(stats['total_demand'])
            satisfaction.append(stats['avg_satisfaction'] * 100)
        else:
            allocated.append(0)
            demanded.append(0)
            satisfaction.append(0)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Bandwidth Allocation vs Demand', 'User Satisfaction'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Chart 1: Allocation vs Demand
    fig.add_trace(
        go.Bar(name='Allocated', x=tier_names, y=allocated, marker_color=colors,
               text=[f'{v:.0f} Mbps' for v in allocated], textposition='auto'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(name='Demanded', x=tier_names, y=demanded, 
               marker_color='lightgray', opacity=0.6,
               text=[f'{v:.0f} Mbps' for v in demanded], textposition='auto'),
        row=1, col=1
    )
    
    # Chart 2: Satisfaction
    fig.add_trace(
        go.Bar(x=tier_names, y=satisfaction, marker_color=colors,
               text=[f'{v:.1f}%' for v in satisfaction], textposition='auto',
               showlegend=False),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        title_text="📊 Tier-Based Performance Metrics",
        title_x=0.5,
        title_font=dict(size=20, family='Orbitron')
    )
    
    fig.update_yaxes(title_text="Bandwidth (Mbps)", row=1, col=1)
    fig.update_yaxes(title_text="Satisfaction (%)", row=1, col=2)
    
    return fig


def create_priority_heatmap(users_df, allocation):
    """Create priority vs allocation heatmap."""
    # Sample users for visualization
    sample_size = min(100, len(users_df))
    sample_indices = np.random.choice(len(users_df), sample_size, replace=False)
    
    sample_df = users_df.iloc[sample_indices].copy()
    sample_df['allocation'] = allocation[sample_indices]
    sample_df['satisfaction'] = sample_df['allocation'] / sample_df['base_demand_mbps']
    sample_df['satisfaction'] = sample_df['satisfaction'].clip(0, 1)
    
    # Create scatter plot with color
    fig = px.scatter(
        sample_df,
        x='priority',
        y='allocation',
        size='base_demand_mbps',
        color='tier',
        color_discrete_map={'emergency': '#f5576c', 'premium': '#4facfe', 'free': '#43e97b'},
        hover_data=['service_type', 'satisfaction'],
        title='🎯 Priority vs Bandwidth Allocation'
    )
    
    fig.update_layout(
        height=500,
        title_font=dict(size=20, family='Orbitron'),
        xaxis_title="Priority Level",
        yaxis_title="Allocated Bandwidth (Mbps)"
    )
    
    return fig


def tier_allocation_page():
    """Main page for tier-based bandwidth allocation."""
    st.markdown('<h1 class="main-title">🚀 TIER-BASED BANDWIDTH OPTIMIZER</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Emergency Services  > Premium Users > Free Users</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.title("⚙️ Configuration")
    
    # User distribution
    st.sidebar.markdown("### 👥 User Distribution")
    total_users = st.sidebar.slider("Total Users", 100, 10000, 1000, step=100)
    emergency_pct = st.sidebar.slider("Emergency Services (%)", 1, 10, 2) / 100
    premium_pct = st.sidebar.slider("Premium Users (%)", 10, 50, 25) / 100
    
    # Network capacity
    st.sidebar.markdown("### 🌐 Network Capacity")
    total_capacity = st.sidebar.slider("Total Bandwidth (Mbps)", 1000, 100000, 10000, step=1000)
    
    # Utility function
    st.sidebar.markdown("### 🎯 Optimization Method")
    utility_type = st.sidebar.selectbox(
        "Utility Function",
        ['log', 'sqrt', 'linear'],
        help="Log: Fair allocation, Sqrt: Balanced, Linear: Maximum throughput"
    )
    
    # Generate data button
    if st.sidebar.button("🎲 Generate New Dataset", type="primary"):
        with st.spinner("🔄 Generating diverse user dataset..."):
            # Generate with NO seed for different data each time
            users_df = EnhancedDataGenerator.generate_users(
                n_users=total_users,
                emergency_pct=emergency_pct,
                premium_pct=premium_pct,
                seed=None  # Different each time!
            )
            
            st.session_state['users_df'] = users_df
            st.session_state['total_capacity'] = total_capacity
            st.success(f"✅ Generated {len(users_df)} users with realistic profiles!")
            
            # Show generation timestamp
            st.info(f"📅 Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display dataset if available
    if 'users_df' in st.session_state:
        users_df = st.session_state['users_df']
        total_capacity = st.session_state['total_capacity']
        
        # Dataset overview
        st.markdown("## 📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Users", f"{len(users_df):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Demand", f"{users_df['base_demand_mbps'].sum():,.0f} Mbps")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Available Capacity", f"{total_capacity:,} Mbps")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            demand_ratio = users_df['base_demand_mbps'].sum() / total_capacity
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Demand/Capacity Ratio", f"{demand_ratio:.2f}x")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tier distribution
        st.markdown("### 🎯 Tier Breakdown")
        
        col1, col2, col3 = st.columns(3)
        
        emergency_count = len(users_df[users_df['tier'] == 'emergency'])
        premium_count = len(users_df[users_df['tier'] == 'premium'])
        free_count = len(users_df[users_df['tier'] == 'free'])
        
        with col1:
            st.markdown('<div class="tier-emergency">', unsafe_allow_html=True)
            st.markdown(f"### 🚨 Emergency Services")
            st.markdown(f"**{emergency_count:,} users** ({emergency_count/len(users_df)*100:.1f}%)")
            st.markdown(f"Demand: {users_df[users_df['tier']=='emergency']['base_demand_mbps'].sum():,.0f} Mbps")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="tier-premium">', unsafe_allow_html=True)
            st.markdown(f"### ⭐ Premium Users")
            st.markdown(f"**{premium_count:,} users** ({premium_count/len(users_df)*100:.1f}%)")
            st.markdown(f"Demand: {users_df[users_df['tier']=='premium']['base_demand_mbps'].sum():,.0f} Mbps")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="tier-free">', unsafe_allow_html=True)
            st.markdown(f"### 📱 Free Tier Users")
            st.markdown(f"**{free_count:,} users** ({free_count/len(users_df)*100:.1f}%)")
            st.markdown(f"Demand: {users_df[users_df['tier']=='free']['base_demand_mbps'].sum():,.0f} Mbps")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualization
        st.plotly_chart(create_tier_visualization(users_df), use_container_width=True)
        
        # Optimize button
        st.markdown("---")
        st.markdown("## 🚀 Run Optimization")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            optimize_button = st.button("⚡ OPTIMIZE ALLOCATION", type="primary")
        
        with col2:
            st.info("💡 **Smart Allocation**: Emergency services get priority, premium users receive guarantees, free users are optimized efficiently!")
        
        if optimize_button:
            with st.spinner("🔄 Running tier-based optimization..."):
                start_time = time_module.time()
                
                # Run optimization
                optimizer = TierBasedOptimizer(total_capacity=total_capacity)
                
                result = optimizer.optimize_with_tiers(
                    demands=users_df['base_demand_mbps'].values,
                    priorities=users_df['priority'].values,
                    min_bandwidth=users_df['min_bandwidth_mbps'].values,
                    max_bandwidth=users_df['max_bandwidth_mbps'].values,
                    tiers=users_df['user_type_code'].values,
                    allocation_weights=users_df['allocation_weight'].values,
                    utility_type=utility_type
                )
                
                elapsed = time_module.time() - start_time
                
                st.session_state['optimization_result'] = result
                st.session_state['optimize_time'] = elapsed
                
                st.success(f"✅ Optimization completed in {elapsed:.3f} seconds!")
        
        # Display results
        if 'optimization_result' in st.session_state:
            result = st.session_state['optimization_result']
            
            st.markdown("---")
            st.markdown("## 📈 Optimization Results")
            
            # Key metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Allocated", f"{result['total_allocated']:,.0f} Mbps")
            
            with col2:
                st.metric("Efficiency", f"{result['efficiency']*100:.1f}%")
            
            with col3:
                st.metric("Fairness Index", f"{result['jains_fairness_index']:.4f}")
            
            with col4:
                st.metric("Avg Satisfaction", f"{result['avg_satisfaction']*100:.1f}%")
            
            with col5:
                st.metric("Solve Time", f"{result['solve_time']:.3f}s")
            
            # Tier statistics
            st.markdown("### 📊 Per-Tier Performance")
            
            tier_stats = result['tier_statistics']
            
            # Create comparison table
            comparison_data = []
            for tier_name, tier_key in [('🚨 Emergency', 'emergency'), 
                                        ('⭐ Premium', 'premium'), 
                                        ('📱 Free', 'free')]:
                stats = tier_stats[tier_key]
                if stats:
                    comparison_data.append({
                        'Tier': tier_name,
                        'Users': f"{stats['count']:,}",
                        'Total Demand': f"{stats['total_demand']:,.0f} Mbps",
                        'Allocated': f"{stats['total_allocated']:,.0f} Mbps",
                        'Avg Allocation': f"{stats['avg_allocation']:.1f} Mbps",
                        'Satisfaction': f"{stats['avg_satisfaction']*100:.1f}%",
                        'Guarantee Met': f"{stats['guarantee_met_pct']:.1f}%"
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, height=150)
            
            # Visualizations
            st.plotly_chart(create_allocation_comparison_chart(result), use_container_width=True)
            
            st.plotly_chart(create_priority_heatmap(users_df, result['allocation']), use_container_width=True)
            
            # Download results
            st.markdown("### 💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Create results dataframe
                results_df = users_df.copy()
                results_df['allocated_mbps'] = result['allocation']
                results_df['satisfaction'] = results_df['allocated_mbps'] / results_df['base_demand_mbps']
                results_df['satisfaction'] = results_df['satisfaction'].clip(0, 1)
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name=f"allocation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export to Excel
                if st.button("📊 Export to Excel"):
                    filename = EnhancedDataGenerator.export_enhanced_data(
                        users_df, 
                        result['allocation'].reshape(-1, 1),
                        filename=f"allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    )
                    st.success(f"✅ Exported to {filename}")
    
    else:
        st.info("👈 Click 'Generate New Dataset' in the sidebar to begin!")
        
        # Show features
        st.markdown("## ✨ Amazing Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Tier-Based Priority System
            - **🚨 Emergency Services**: Hospitals, 911, police get highest priority
            - **⭐ Premium Users**: Business-grade service with guarantees
            - **📱 Free Users**: Optimized best-effort allocation
            
            ### 🔄 Dynamic Data Generation
            - Different dataset every time
            - Realistic user profiles
            - 10+ service types per tier
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Advanced Optimization
            - Mathematical fairness guarantees
            - Convex optimization (CVXPY)
            - Multiple utility functions
            
            ### 🎨 Beautiful Visualizations
            - Interactive Plotly charts
            - Real-time metrics
            - Export capabilities
            """)


def emergency_scenarios_page():
    """Emergency scenario simulation page."""
    st.markdown('<h1 class="main-title">🚨 EMERGENCY SCENARIO SIMULATOR</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Test system under extreme conditions</p>', unsafe_allow_html=True)
    
    if 'users_df' not in st.session_state:
        st.warning("⚠️ Please generate a dataset first from the Tier Allocation page!")
        return
    
    users_df = st.session_state['users_df']
    total_capacity = st.session_state['total_capacity']
    
    st.markdown("## 🎬 Select Emergency Scenario")
    
    scenarios = {
        'normal': {
            'name': '✅ Normal Operations',
            'description': 'Regular network conditions',
            'color': '#4caf50'
        },
        'disaster': {
            'name': '🌪️ Natural Disaster',
            'description': 'Earthquake/hurricane - Emergency services 3x demand',
            'color': '#ff5252'
        },
        'cyber_attack': {
            'name': '🔒 Cyber Attack',
            'description': 'DDoS attack - 40% capacity loss',
            'color': '#ff9800'
        },
        'mass_event': {
            'name': '🎉 Mass Event',
            'description': 'Concert/sports - All users high demand',
            'color': '#2196f3'
        },
        'infrastructure_failure': {
            'name': '⚡ Infrastructure Failure',
            'description': 'Major outage - 50% capacity loss',
            'color': '#9c27b0'
        }
    }
    
    selected_scenario = st.selectbox(
        "Choose Scenario",
        list(scenarios.keys()),
        format_func=lambda x: scenarios[x]['name']
    )
    
    scenario_info = scenarios[selected_scenario]
    st.markdown(f"""
    <div style='background-color: {scenario_info["color"]}; padding: 20px; border-radius: 10px; color: white; margin: 20px 0;'>
        <h3>{scenario_info['name']}</h3>
        <p>{scenario_info['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Run Scenario Simulation", type="primary"):
        with st.spinner(f"🔄 Simulating {scenario_info['name']}..."):
            optimizer = TierBasedOptimizer(total_capacity=total_capacity)
            
            scenario_params = EnhancedDataGenerator.generate_emergency_scenarios(
                users_df, selected_scenario
            )
            
            result = optimizer.optimize_emergency_scenario(
                demands=users_df['base_demand_mbps'].values,
                priorities=users_df['priority'].values,
                min_bandwidth=users_df['min_bandwidth_mbps'].values,
                max_bandwidth=users_df['max_bandwidth_mbps'].values,
                tiers=users_df['user_type_code'].values,
                allocation_weights=users_df['allocation_weight'].values,
                scenario_multipliers=scenario_params
            )
            
            st.session_state['scenario_result'] = result
            st.success("✅ Scenario simulation complete!")
    
    if 'scenario_result' in st.session_state:
        result = st.session_state['scenario_result']
        
        st.markdown("---")
        st.markdown("## 📊 Scenario Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Adjusted Capacity", f"{result['adjusted_capacity']:,.0f} Mbps")
        
        with col2:
            st.metric("Capacity Reduction", f"{result['capacity_reduction']*100:.0f}%")
        
        with col3:
            st.metric("Emergency Usage", f"{result['emergency_capacity_used']:,.0f} Mbps")
        
        with col4:
            st.metric("System Efficiency", f"{result['efficiency']*100:.1f}%")
        
        # Tier statistics
        st.markdown("### 📈 Tier Performance Under Stress")
        
        tier_stats = result['tier_statistics']
        
        for tier_name, tier_key, color in [('🚨 Emergency Services', 'emergency', '#f5576c'),
                                            ('⭐ Premium Users', 'premium', '#4facfe'),
                                            ('📱 Free Users', 'free', '#43e97b')]:
            stats = tier_stats[tier_key]
            if stats:
                st.markdown(f"""
                <div style='background: {color}; padding: 15px; border-radius: 10px; color: white; margin: 10px 0;'>
                    <h4>{tier_name}</h4>
                    <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;'>
                        <div>
                            <strong>Users:</strong> {stats['count']:,}<br>
                            <strong>Satisfaction:</strong> {stats['avg_satisfaction']*100:.1f}%
                        </div>
                        <div>
                            <strong>Allocated:</strong> {stats['total_allocated']:,.0f} Mbps<br>
                            <strong>Demanded:</strong> {stats['total_demand']:,.0f} Mbps
                        </div>
                        <div>
                            <strong>Min Guarantee:</strong> {stats['guarantee_met_pct']:.1f}%<br>
                            <strong>Demand Met:</strong> {stats['demand_met_pct']:.1f}%
                        </div>
                        <div>
                            <strong>Avg Alloc:</strong> {stats['avg_allocation']:.1f} Mbps<br>
                            <strong>Range:</strong> {stats['min_allocation']:.0f}-{stats['max_allocation']:.0f}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


def main():
    """Main application with navigation."""
    st.sidebar.title("🎯 Navigation")
    
    page = st.sidebar.radio(
        "Select Page",
        ["🚀 Tier-Based Allocation", "🚨 Emergency Scenarios", "📚 User Guide"]
    )
    
    if page == "🚀 Tier-Based Allocation":
        tier_allocation_page()
    elif page == "🚨 Emergency Scenarios":
        emergency_scenarios_page()
    elif page == "📚 User Guide":
        user_guide_page()


def user_guide_page():
    """Comprehensive user guide for the frontend."""
    st.markdown('<h1 class="main-title">📚 COMPLETE USER GUIDE</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Everything you need to know about the Bandwidth Optimizer</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Table of contents
    st.markdown("""
    ## 📑 Table of Contents
    1. [Getting Started](#getting-started)
    2. [Tier System Explained](#tier-system-explained)
    3. [How to Use Tier Allocation](#tier-allocation-usage)
    4. [Emergency Scenarios](#emergency-scenarios-usage)
    5. [Understanding Results](#understanding-results)
    6. [Advanced Features](#advanced-features)
    7. [FAQ](#faq)
    """)
    
    st.markdown("---")
    
    # Getting Started
    st.markdown("## 🚀 Getting Started")
    st.markdown("""
    ### Quick Start Guide
    
    1. **Navigate to Tier-Based Allocation page**
    2. **Configure your network** using the sidebar:
       - Total users (100-10,000)
       - Emergency services percentage (1-10%)
       - Premium users percentage (10-50%)
       - Total bandwidth capacity
    3. **Click "Generate New Dataset"** - Creates a unique dataset every time!
    4. **Review the dataset** - Check tier distribution and demand patterns
    5. **Click "OPTIMIZE ALLOCATION"** - Runs the smart optimizer
    6. **Analyze results** - View metrics, charts, and per-tier performance
    7. **Export data** - Download results as CSV or Excel
    
    ### System Requirements
    - Modern web browser (Chrome, Firefox, Safari, Edge)
    - Stable internet connection
    - Recommended: 1920x1080 or higher resolution
    """)
    
    st.markdown("---")
    
    # Tier System
    st.markdown("## 👥 Tier System Explained")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="tier-emergency">
            <h3>🚨 Emergency Services</h3>
            <ul>
                <li><strong>Priority:</strong> HIGHEST (9-10)</li>
                <li><strong>Examples:</strong> 911, hospitals, police, fire</li>
                <li><strong>Guarantee:</strong> 90% of demand</li>
                <li><strong>Bandwidth:</strong> 50-200 Mbps</li>
                <li><strong>Latency:</strong> < 15ms</li>
                <li><strong>QoS:</strong> 99.9%</li>
                <li><strong>Cost:</strong> FREE</li>
                <li><strong>Preemption:</strong> Can override others</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tier-premium">
            <h3>⭐ Premium Users</h3>
            <ul>
                <li><strong>Priority:</strong> HIGH (6-8)</li>
                <li><strong>Examples:</strong> Business, gamers, creators</li>
                <li><strong>Guarantee:</strong> 70% of demand</li>
                <li><strong>Bandwidth:</strong> 20-800 Mbps</li>
                <li><strong>Latency:</strong> < 30ms</li>
                <li><strong>QoS:</strong> 98-99%</li>
                <li><strong>Cost:</strong> $2/Mbps</li>
                <li><strong>Throttling:</strong> Not allowed</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="tier-free">
            <h3>📱 Free Tier</h3>
            <ul>
                <li><strong>Priority:</strong> STANDARD (1-5)</li>
                <li><strong>Examples:</strong> Home users, students, casual</li>
                <li><strong>Guarantee:</strong> 30% of demand</li>
                <li><strong>Bandwidth:</strong> 1-50 Mbps</li>
                <li><strong>Latency:</strong> < 200ms</li>
                <li><strong>QoS:</strong> 90-96%</li>
                <li><strong>Cost:</strong> $0.5/Mbps</li>
                <li><strong>Throttling:</strong> Allowed during congestion</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # How to Use
    st.markdown("## 🎯 How to Use Tier Allocation")
    
    with st.expander("📊 Step 1: Generate Dataset"):
        st.markdown("""
        ### Generating a New Dataset
        
        1. **Open sidebar** (click arrow on left if collapsed)
        2. **Adjust sliders:**
           - **Total Users:** How many users in the network
           - **Emergency %:** Percentage of emergency service users
           - **Premium %:** Percentage of premium subscribers
           - **Remaining:** Automatically assigned to free tier
        3. **Set capacity:** Total bandwidth available (in Mbps)
        4. **Click "Generate New Dataset"**
        
        **🎲 Every generation is unique!** The system creates different:
        - User profiles
        - Service types
        - Demand patterns
        - Geographic distribution
        - Application mixes
        
        **💡 Tip:** Try different percentages to see how tier balance affects optimization!
        """)
    
    with st.expander("⚡ Step 2: Run Optimization"):
        st.markdown("""
        ### Running the Optimizer
        
        1. **Review dataset** - Check the overview metrics
        2. **Choose utility function:**
           - **Log (Proportional Fair):** Best for fairness - RECOMMENDED
           - **Sqrt (Balanced):** Balance between fairness and throughput
           - **Linear (Max Throughput):** Maximum efficiency, less fair
        3. **Click "OPTIMIZE ALLOCATION"**
        4. **Wait for results** - Usually completes in < 1 second
        
        **🧠 What happens during optimization:**
        - Phase 1: Emergency services allocated first
        - Phase 2: Premium users get guaranteed bandwidth
        - Phase 3: Remaining capacity optimized for all users
        - Mathematical solver ensures optimal solution
        """)
    
    with st.expander("📈 Step 3: Analyze Results"):
        st.markdown("""
        ### Understanding Your Results
        
        **Key Metrics:**
        - **Total Allocated:** Bandwidth actually distributed
        - **Efficiency:** % of capacity used (higher = better)
        - **Fairness Index:** Jain's index 0-1 (1 = perfectly fair)
        - **Avg Satisfaction:** How well demands are met
        - **Solve Time:** Optimization duration
        
        **Per-Tier Statistics:**
        - See how each tier performed
        - Check guarantee compliance
        - Compare allocation vs demand
        
        **Visualizations:**
        - **Pie Chart:** User distribution by tier
        - **Bar Charts:** Allocation vs demand comparison
        - **Scatter Plot:** Priority vs allocation relationship
        """)
    
    with st.expander("💾 Step 4: Export Data"):
        st.markdown("""
        ### Exporting Your Results
        
        **CSV Export:**
        - Click "Download Results (CSV)"
        - Opens in Excel, Google Sheets, etc.
        - Contains all user data + allocations
        
        **Excel Export:**
        - Click "Export to Excel"
        - Multi-sheet workbook with:
          - User details
          - Allocations
          - Tier summaries
          - Regional analysis
        
        **Filename Format:** `allocation_YYYYMMDD_HHMMSS.xlsx`
        """)
    
    st.markdown("---")
    
    # Emergency Scenarios
    st.markdown("## 🚨 Emergency Scenarios Usage")
    
    st.markdown("""
    ### Simulating Crisis Conditions
    
    Test how your network handles extreme situations:
    
    #### Available Scenarios:
    
    1. **✅ Normal Operations**
       - Baseline performance
       - Regular demand patterns
       - Full capacity available
    
    2. **🌪️ Natural Disaster**
       - Emergency demand 3x higher
       - 20% capacity loss
       - Free users throttled 50%
       - *Example: Earthquake, hurricane*
    
    3. **🔒 Cyber Attack**
       - 40% capacity loss
       - Emergency demand 2x higher
       - Free users throttled 70%
       - *Example: DDoS attack*
    
    4. **🎉 Mass Event**
       - All users high demand
       - Premium users 2x demand
       - 10% capacity loss
       - *Example: Concert, sports event*
    
    5. **⚡ Infrastructure Failure**
       - 50% capacity loss
       - Emergency 2.5x demand
       - Free users throttled 80%
       - *Example: Power outage, fiber cut*
    
    ### How to Run:
    1. Generate dataset on Tier Allocation page first
    2. Navigate to Emergency Scenarios
    3. Select scenario from dropdown
    4. Click "Run Scenario Simulation"
    5. Compare results vs normal operation
    
    **💡 Use Cases:**
    - Network capacity planning
    - Disaster preparedness
    - SLA validation
    - Cost-benefit analysis of redundancy
    """)
    
    st.markdown("---")
    
    # Advanced Features
    st.markdown("## 🚀 Advanced Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎨 Visualization Features
        - **Interactive Charts:** Hover for details
        - **Zoom & Pan:** Click and drag
        - **Export Images:** Camera icon (top-right of charts)
        - **Responsive Design:** Works on all screen sizes
        
        ### 🔄 Dynamic Generation
        - **No Seed:** Different data each time
        - **Realistic Patterns:** Based on real-world usage
        - **10+ Service Types:** Per tier
        - **Geographic Diversity:** 5 regions
        
        ### 🎯 Smart Optimization
        - **Convex Optimization:** Guaranteed optimal
        - **Multiple Objectives:** Fairness + efficiency
        - **Constraint Handling:** Min/max guarantees
        - **Fast Solving:** Sub-second for 10K users
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Analytics
        - **Real-time Metrics:** Instant feedback
        - **Comparative Analysis:** Tier-by-tier breakdown
        - **Satisfaction Tracking:** User experience metrics
        - **Efficiency Monitoring:** Resource utilization
        
        ### 💡 Business Intelligence
        - **Revenue Estimation:** Cost calculations per tier
        - **SLA Compliance:** Guarantee tracking
        - **Capacity Planning:** Demand forecasting
        - **What-If Analysis:** Scenario comparison
        
        ### 🔒 Reliability
        - **Fallback Mechanisms:** Proportional allocation backup
        - **Error Handling:** Graceful degradation
        - **Status Messages:** Clear feedback
        - **Data Validation:** Input checking
        """)
    
    st.markdown("---")
    
    # FAQ
    st.markdown("## ❓ Frequently Asked Questions")
    
    with st.expander("Q: Why is the data different every time?"):
        st.markdown("""
        **A:** This is a feature! The system generates unique datasets each time to:
        - Demonstrate robustness of the algorithm
        - Allow testing different scenarios
        - Simulate real-world variability
        - Prevent overfitting to one dataset
        
        If you need consistent data for comparison, you can export and reload datasets.
        """)
    
    with st.expander("Q: What does 'Jain's Fairness Index' mean?"):
        st.markdown("""
        **A:** Jain's Fairness Index measures how fairly bandwidth is distributed:
        - **1.0 = Perfect fairness** (everyone gets same allocation)
        - **0.0 = Maximum unfairness** (one user gets everything)
        - **Typical good values:** 0.85-0.95
        
        Formula: `(Σx_i)² / (n × Σx_i²)`
        
        **Note:** Perfect fairness (1.0) isn't always desirable - emergency services 
        should get more than free users!
        """)
    
    with st.expander("Q: Why can't all demands be satisfied?"):
        st.markdown("""
        **A:** Total demand often exceeds capacity (oversubscription):
        - Common in networks (1.5-3x oversubscription typical)
        - Optimization distributes limited capacity fairly
        - Emergency services always prioritized
        - Premium users get guaranteed minimums
        
        **Solution:** Increase capacity or reduce demand percentages.
        """)
    
    with st.expander("Q: Which utility function should I use?"):
        st.markdown("""
        **A:** Depends on your goal:
        
        - **Log (Proportional Fair)** - RECOMMENDED
          - Best balance of fairness and efficiency
          - Industry standard (Kelly's criterion)
          - Use for: General-purpose networks
        
        - **Sqrt (Balanced)**
          - More throughput than log
          - Still maintains fairness
          - Use for: High-capacity networks
        
        - **Linear (Max Throughput)**
          - Maximum total bandwidth
          - Less fair to small users
          - Use for: Efficiency-critical applications
        """)
    
    with st.expander("Q: How do I interpret satisfaction percentages?"):
        st.markdown("""
        **A:** Satisfaction = (Allocated / Demanded) × 100%
        
        - **100%:** User got full demand (ideal but rare)
        - **70-100%:** Very good
        - **50-70%:** Acceptable
        - **< 50%:** May need capacity increase
        
        **Tier expectations:**
        - Emergency: Should be near 100%
        - Premium: 70-90% typical
        - Free: 30-60% acceptable
        """)
    
    with st.expander("Q: Can I use this for real networks?"):
        st.markdown("""
        **A:** This is a research/educational tool, but the algorithms are production-ready:
        
        **Yes, suitable for:**
        - Network capacity planning
        - What-if analysis
        - Algorithm comparison
        - Academic research
        - Proof of concept
        
        **Additional requirements for production:**
        - Real-time monitoring integration
        - Historical data import
        - Automated policy enforcement
        - Scalability testing (100K+ users)
        - Integration with network equipment APIs
        """)
    
    st.markdown("---")
    
    # Contact
    st.markdown("""
    ## 📧 Need Help?
    
    - 📚 **Documentation:** Check this guide thoroughly
    - 💡 **Tips:** Hover over '?' icons for quick help
    - 🐛 **Issues:** Report bugs with detailed steps to reproduce
    - ✨ **Features:** Suggest improvements
    
    ---
    
    **Built with:** Python, Streamlit, CVXPY, Plotly, NumPy, Pandas
    
    **Optimization:** Convex optimization with ECOS solver
    
    **Version:** 2.0 - Enhanced Tier System
    """)


if __name__ == "__main__":
    main()
