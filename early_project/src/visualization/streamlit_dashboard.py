"""
H2Z Satellite Power System - Enhanced Streamlit Dashboard

Professional interactive dashboard for satellite power system monitoring,
battery optimization, and RL training visualization.

Features:
- Real-time telemetry monitoring
- Battery degradation analytics
- MPPT efficiency analysis
- Orbital visualization
- RL training metrics

Author: H2Z Development Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os
import sys

# Configure Streamlit page
st.set_page_config(
    page_title="🛰️ H2Z Satellite Power Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aerospace theme
st.markdown("""
<style>
    .main {
        background-color: #0d1117;
    }
    .stApp {
        background-color: #0d1117;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
    }
    .metric-card {
        background-color: #161b22;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #21262d;
    }
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #58a6ff;
    }
    .metric-label {
        color: #8b949e;
        font-size: 0.9em;
    }
    .stMetric {
        background-color: #161b22;
        border-radius: 8px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


class H2ZDashboard:
    """Main dashboard class for H2Z Satellite Power System."""
    
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.sample_data = self._generate_sample_data()
    
    def _generate_sample_data(self) -> dict:
        """Generate sample telemetry data for demonstration."""
        base_time = datetime.now()
        n_points = 100
        
        timestamps = [base_time + timedelta(minutes=i*5) for i in range(n_points)]
        
        # Solar power (sinusoidal with noise)
        solar_power = 800 + 200 * np.sin(np.linspace(0, 10, n_points)) + np.random.randn(n_points) * 30
        
        # Battery SOC
        soc = 70 + 15 * np.sin(np.linspace(0, 5, n_points)) + np.random.randn(n_points) * 2
        
        # Power demand
        demand = {
            'ADCS': 35 + 5 * np.random.randn(n_points),
            'TT&C': 18 + 3 * np.random.randn(n_points),
            'CDH': 12 + 2 * np.random.randn(n_points),
            'Propulsion': 90 + 10 * np.random.randn(n_points),
            'Communication': 25 + 4 * np.random.randn(n_points),
            'Payload': 12 + 2 * np.random.randn(n_points)
        }
        
        # Temperature
        temperature = 25 + 5 * np.sin(np.linspace(0, 10, n_points)) + np.random.randn(n_points) * 2
        
        # Eclipse phase
        eclipse = (np.arange(n_points) % 20) > 15
        
        return {
            'timestamps': timestamps,
            'solar_power': solar_power,
            'battery_soc': soc,
            'demand': demand,
            'temperature': temperature,
            'eclipse': eclipse
        }
    
    def sidebar_navigation(self):
        """Sidebar navigation menu."""
        st.sidebar.title("🛰️ H2Z Navigation")
        
        pages = [
            "📊 Power System Monitor",
            "🔋 Battery Analytics",
            "☀️ MPPT Analysis",
            "🌍 3D Orbit View",
            "🤖 RL Training Dashboard",
            "📈 Mission Timeline"
        ]
        
        choice = st.sidebar.radio("Go to", pages)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚙️ Settings")
        
        show_metrics = st.sidebar.checkbox("Show Metrics", value=True)
        dark_mode = st.sidebar.checkbox("Dark Mode", value=True)
        auto_refresh = st.sidebar.slider("Auto Refresh (s)", 0, 30, 0)
        
        return choice, show_metrics, auto_refresh
    
    def display_header(self):
        """Display dashboard header."""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.title("🛰️ H2Z Satellite Power System")
            st.markdown("**AI-Enhanced Power & Communication Subsystem for LEO Space Tug**")
        
        with col2:
            st.markdown("### 📡 Status")
            st.success("🟢 ONLINE")
        
        with col3:
            st.markdown("### 🕐 Mission Time")
            st.markdown("**Day 127**")
            st.markdown("**Orbit: 500 km**")
    
    def display_power_monitor(self):
        """Display power system monitoring dashboard."""
        st.header("⚡ Power System Monitor")
        
        data = self.sample_data
        
        # Top metrics row
        cols = st.columns(6)
        
        with cols[0]:
            current_soc = data['battery_soc'][-1]
            st.metric(
                "Battery SOC",
                f"{current_soc:.1f}%",
                delta=f"{current_soc - data['battery_soc'][-2]:.1f}%"
            )
        
        with cols[1]:
            current_solar = data['solar_power'][-1]
            st.metric(
                "Solar Power",
                f"{current_solar:.0f} W",
                delta=f"{current_solar - data['solar_power'][-2]:.0f} W"
            )
        
        with cols[2]:
            total_demand = sum(d['demand'][d['eclipse'][-1]] if d['eclipse'][-1] else 0 for d in [data])
            st.metric(
                "Total Demand",
                f"{total_demand:.0f} W",
                delta="-2.5 W"
            )
        
        with cols[3]:
            temp = data['temperature'][-1]
            st.metric(
                "Temperature",
                f"{temp:.1f} °C",
                delta=f"{temp - data['temperature'][-2]:.1f} °C"
            )
        
        with cols[4]:
            orbit_phase = "☀️ Sunlight" if not data['eclipse'][-1] else "🌙 Eclipse"
            st.metric(
                "Orbit Phase",
                orbit_phase
            )
        
        with cols[5]:
            mppt_eff = 97.0 - np.random.rand() * 2
            st.metric(
                "MPPT Efficiency",
                f"{mppt_eff:.1f}%"
            )
        
        st.markdown("---")
        
        # Main charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔋 Battery State of Charge")
            fig_soc = go.Figure()
            fig_soc.add_trace(go.Scatter(
                x=data['timestamps'],
                y=data['battery_soc'],
                mode='lines+markers',
                name='SOC',
                line=dict(color='#58a6ff', width=2),
                fill='tozeroy',
                fillcolor='rgba(88, 166, 255, 0.2)'
            ))
            fig_soc.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Time",
                yaxis_title="SOC (%)"
            )
            st.plotly_chart(fig_soc, use_container_width=True)
        
        with col2:
            st.subheader("☀️ Solar Power Generation")
            fig_solar = go.Figure()
            fig_solar.add_trace(go.Scatter(
                x=data['timestamps'],
                y=data['solar_power'],
                mode='lines',
                name='Solar Power',
                line=dict(color='#2ea043', width=2)
            ))
            # Add eclipse shading
            for i, (ts, eclipse) in enumerate(zip(data['timestamps'][:-1], data['eclipse'][:-1])):
                if eclipse:
                    fig_solar.add_vrect(
                        x0=ts,
                        x1=data['timestamps'][i+1],
                        fillcolor="rgba(0,0,0,0.3)",
                        layer="below",
                        line_width=0
                    )
            fig_solar.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Time",
                yaxis_title="Power (W)"
            )
            st.plotly_chart(fig_solar, use_container_width=True)
        
        # Power demand breakdown
        st.subheader("🔌 Subsystem Power Demand")
        
        current_demands = {
            'ADCS': 41.26,
            'TT&C': 20.32,
            'CDH': 13.71,
            'Propulsion': 96.60,
            'Communication': 28.19,
            'Payload': 13.00
        }
        
        col_chart1, col_chart2 = st.columns([2, 1])
        
        with col_chart1:
            # Stacked area chart
            fig_demand = go.Figure()
            for name, color in zip(['ADCS', 'TT&C', 'CDH', 'Propulsion', 'Communication', 'Payload'],
                                   ['#58a6ff', '#2ea043', '#f78166', '#a371f7', '#d29922', '#f85149']):
                fig_demand.add_trace(go.Scatter(
                    x=data['timestamps'],
                    y=current_demands[name] + np.random.randn(len(data['timestamps'])) * 5,
                    mode='lines',
                    name=name,
                    stackgroup='demand',
                    line=dict(width=0.5),
                    marker=dict(size=4)
                ))
            fig_demand.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Time",
                yaxis_title="Power (W)",
                showlegend=True
            )
            st.plotly_chart(fig_demand, use_container_width=True)
        
        with col_chart2:
            # Pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(current_demands.keys()),
                values=list(current_demands.values()),
                hole=0.4,
                marker=dict(colors=px.colors.qualitative.Set3),
                textinfo='label+percent'
            )])
            fig_pie.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                height=350,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    def display_battery_analytics(self):
        """Display battery degradation analytics."""
        st.header("🔋 Battery Analytics & Degradation")
        
        # Mission timeline selector
        timeline = st.select_slider(
            "Mission Timeline",
            options=[0, 90, 180, 365, 730, 1095],
            format_func=lambda x: f"{x//365} years {x%365} days" if x > 0 else "Launch"
        )
        
        # Degradation projection
        st.subheader("📉 Battery Degradation Projection")
        
        # Generate SOH projection data
        days = np.linspace(0, 1095, 100)
        soh_baseline = 100 - 0.02 * days  # 2% per year baseline
        soh_optimal = 100 - 0.01 * days  # Optimal with RL
        soh_conservative = 100 - 0.005 * days  # Conservative approach
        
        fig_soh = go.Figure()
        fig_soh.add_trace(go.Scatter(
            x=days,
            y=soh_baseline,
            mode='lines',
            name='Baseline (2%/year)',
            line=dict(color='#f78166', width=2, dash='dash')
        ))
        fig_soh.add_trace(go.Scatter(
            x=days,
            y=soh_optimal,
            mode='lines',
            name='RL Optimized (1%/year)',
            line=dict(color='#2ea043', width=3)
        ))
        fig_soh.add_trace(go.Scatter(
            x=days,
            y=soh_conservative,
            mode='lines',
            name='Conservative (0.5%/year)',
            line=dict(color='#58a6ff', width=2, dash='dot')
        ))
        # Add threshold line
        fig_soh.add_hline(y=80, line_dash="dash", line_color="#f85149", 
                         annotation_text="EOL Threshold (80%)")
        fig_soh.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            xaxis_title="Mission Days",
            yaxis_title="State of Health (%)",
            xaxis=dict(range=[0, 1095]),
            yaxis=dict(range=[60, 100])
        )
        st.plotly_chart(fig_soh, use_container_width=True)
        
        # Degradation factors
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌡️ Temperature Effects")
            temps = np.linspace(-100, 80, 100)
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(
                x=temps,
                y=100 - 0.1 * np.exp((temps + 100) / 50) * 100,
                mode='lines',
                name='Degradation Rate',
                fill='tozeroy',
                fillcolor='rgba(247, 129, 102, 0.3)',
                line=dict(color='#f78166')
            ))
            fig_temp.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300,
                xaxis_title="Temperature (°C)",
                yaxis_title="Relative Degradation Rate"
            )
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with col2:
            st.subheader("⚡ C-Rate Effects")
            c_rates = np.linspace(0.1, 2.0, 50)
            fig_crate = go.Figure()
            fig_crate.add_trace(go.Scatter(
                x=c_rates,
                y=100 - 5 * c_rates ** 2,
                mode='lines',
                name='Capacity Fade',
                fill='tozeroy',
                fillcolor='rgba(88, 166, 255, 0.3)',
                line=dict(color='#58a6ff')
            ))
            fig_crate.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300,
                xaxis_title="C-Rate",
                yaxis_title="Capacity Retention (%)"
            )
            st.plotly_chart(fig_crate, use_container_width=True)
        
        # Key metrics
        st.subheader("📊 Battery Specifications")
        
        specs_cols = st.columns(4)
        
        with specs_cols[0]:
            st.metric("Nominal Capacity", "163.22 Wh")
        with specs_cols[1]:
            st.metric("Nominal Voltage", "28.0 V")
        with specs_cols[2]:
            st.metric("Max DOD", "80%")
        with specs_cols[3]:
            st.metric("Initial SOH", "100%")
    
    def display_mppt_analysis(self):
        """Display MPPT efficiency analysis."""
        st.header("☀️ MPPT Efficiency Analysis")
        
        # Time slider
        years = st.slider("Mission Years", 0, 3, 3)
        
        # Efficiency comparison
        st.subheader("📈 MPPT vs Fixed System")
        
        time_years = np.linspace(0, years, 100)
        mppt_efficiency = 0.97 * (1 - 0.005 * time_years)
        fixed_efficiency = np.full_like(time_years, 0.85)
        
        fig_mppt = go.Figure()
        fig_mppt.add_trace(go.Scatter(
            x=time_years,
            y=mppt_efficiency * 100,
            mode='lines',
            name='MPPT System',
            line=dict(color='#2ea043', width=3)
        ))
        fig_mppt.add_trace(go.Scatter(
            x=time_years,
            y=fixed_efficiency * 100,
            mode='lines',
            name='Fixed System (85%)',
            line=dict(color='#f78166', width=2, dash='dash')
        ))
        fig_mppt.add_trace(go.Scatter(
            x=time_years,
            y=(mppt_efficiency - fixed_efficiency) * 100,
            mode='lines',
            name='Power Advantage',
            fill='tozeroy',
            fillcolor='rgba(88, 166, 255, 0.2)',
            line=dict(color='#58a6ff', width=2),
            yaxis='y2'
        ))
        
        fig_mppt.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450,
            xaxis_title="Mission Time (Years)",
            yaxis=dict(title="Efficiency (%)", side='left'),
            yaxis2=dict(title="Power Advantage (%)", side='right', overlaying='y')
        )
        st.plotly_chart(fig_mppt, use_container_width=True)
        
        # Key metrics
        mppt_cols = st.columns(4)
        
        with mppt_cols[0]:
            st.metric("Initial MPPT Efficiency", "97.0%")
        with mppt_cols[1]:
            final_eff = mppt_efficiency[-1] * 100
            st.metric(f"Efficiency @ Year {years}", f"{final_eff:.1f}%")
        with mppt_cols[2]:
            advantage = (mppt_efficiency - fixed_efficiency[0]) * 100
            st.metric("Power Advantage", f"{advantage:.1f}%")
        with mppt_cols[3]:
            energy_gain = 12 * advantage  # kWh over mission
            st.metric(f"Energy Gain ({years}yr)", f"{energy_gain:.0f} kWh")
        
        # Temperature effects
        st.subheader("🌡️ Temperature Dependence")
        
        temps = np.linspace(0, 100, 101)
        temp_efficiency = 97 - 0.08 * temps  # -0.08% per °C
        
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=temps,
            y=temp_efficiency,
            mode='lines',
            name='Efficiency',
            line=dict(color='#a371f7', width=3),
            fill='tozeroy',
            fillcolor='rgba(163, 113, 246, 0.2)'
        ))
        fig_temp.add_vline(x=25, line_dash="dash", line_color="#2ea043",
                          annotation_text="Nominal (25°C)")
        fig_temp.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            xaxis_title="Temperature (°C)",
            yaxis_title="MPPT Efficiency (%)"
        )
        st.plotly_chart(fig_temp, use_container_width=True)
    
    def display_3d_orbit(self):
        """Display 3D orbital visualization."""
        st.header("🌍 3D Orbital Visualization")
        
        st.info("🪐 Interactive 3D orbital visualization would be displayed here.")
        st.info("🔧 Requires: pydeck, kepler-gl, or CesiumJS integration")
        
        # Placeholder for 3D visualization
        st.markdown("""
        ### 🚀 Orbital Parameters
        
        | Parameter | Value |
        |-----------|-------|
        | Altitude | 500 km |
        | Inclination | 97.4° |
        | Period | 98 minutes |
        | Beta Angle | 45° (typical) |
        | Eccentricity | 0.0 (circular) |
        """)
        
        # 2D orbit projection (placeholder for 3D)
        st.subheader("📍 Ground Track Projection")
        
        # Generate orbit ground track
        n_orbits = 3
        points_per_orbit = 100
        
        fig_orbit = go.Figure()
        
        # Add Earth outline
        fig_orbit.add_trace(go.Scatter(
            x=[-180, -180, 180, 180, -180],
            y=[-90, 90, 90, -90, -90],
            mode='lines',
            line=dict(color='#21262d', width=2),
            fill='toself',
            fillcolor='rgba(33, 38, 45, 0.3)',
            showlegend=False
        ))
        
        # Generate ground track
        for i in range(n_orbits):
            orbit_lon = np.linspace(-180, 180, points_per_orbit)
            orbit_lat = 97.4 * np.sin(np.linspace(0, 2*np.pi, points_per_orbit))
            
            fig_orbit.add_trace(go.Scatter(
                x=orbit_lon,
                y=orbit_lat,
                mode='lines',
                name=f'Orbit {i+1}',
                line=dict(width=2)
            ))
        
        # Add satellite position
        current_lon = 45
        current_lat = 30
        fig_orbit.add_trace(go.Scatter(
            x=[current_lon],
            y=[current_lat],
            mode='markers',
            marker=dict(size=15, color='#58a6ff'),
            name='H2Z Satellite'
        ))
        
        fig_orbit.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            xaxis_title="Longitude (°)",
            yaxis_title="Latitude (°)",
            xaxis=dict(range=[-180, 180]),
            yaxis=dict(range=[-90, 90]),
            showlegend=True
        )
        st.plotly_chart(fig_orbit, use_container_width=True)
    
    def display_rl_training(self):
        """Display RL training metrics."""
        st.header("🤖 RL Training Dashboard")
        
        # Training configuration
        st.subheader("⚙️ Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Algorithm**: Soft Actor-Critic (SAC)
            - State Space: 20 dimensions
            - Action Space: 5 continuous actions
            - Replay Buffer: 500,000 transitions
            - Target Update: Soft (τ=0.005)
            """)
        
        with col2:
            st.markdown("""
            **Environment**: H2Z Battery Life
            - Max Steps/Episode: 500
            - Reward Components:
              - Battery Lifespan (+5.0)
              - Mission Success (+3.0)
              - Thermal Stability (+2.0)
              - Efficiency (+1.0)
            """)
        
        st.markdown("---")
        
        # Simulated training data
        episodes = np.arange(1, 101)
        rewards = -24150 + 100 * np.random.randn(100) + 10 * np.arange(100)
        soh_values = 100 - 0.01 * np.arange(100)
        
        # Training curves
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("📈 Episode Rewards")
            fig_reward = go.Figure()
            fig_reward.add_trace(go.Scatter(
                x=episodes,
                y=rewards,
                mode='lines',
                name='Reward',
                line=dict(color='#58a6ff', width=2),
                fill='tozeroy',
                fillcolor='rgba(88, 166, 255, 0.2)'
            ))
            # Moving average
            ma_window = 10
            ma = np.convolve(rewards, np.ones(ma_window)/ma_window, mode='valid')
            fig_reward.add_trace(go.Scatter(
                x=episodes[ma_window-1:],
                y=ma,
                mode='lines',
                name=f'{ma_window}-Episode MA',
                line=dict(color='#2ea043', width=3)
            ))
            fig_reward.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350,
                xaxis_title="Episode",
                yaxis_title="Reward"
            )
            st.plotly_chart(fig_reward, use_container_width=True)
        
        with col_chart2:
            st.subheader("🔋 Battery SOH Over Training")
            fig_soh = go.Figure()
            fig_soh.add_trace(go.Scatter(
                x=episodes,
                y=soh_values,
                mode='lines',
                name='SOH',
                line=dict(color='#2ea043', width=2),
                fill='tozeroy',
                fillcolor='rgba(46, 160, 67, 0.2)'
            ))
            fig_soh.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350,
                xaxis_title="Episode",
                yaxis_title="State of Health (%)"
            )
            st.plotly_chart(fig_soh, use_container_width=True)
        
        # Baseline comparison
        st.subheader("🏆 Baseline Comparison")
        
        baselines = {
            'SAC RL Agent': -24151,
            'Simple Rule-Based': -538463,
            'Constant Current': -582048,
            'CC-CV Charging': -582065,
            'Temperature Aware': -546175,
            'Adaptive Charging': -542290
        }
        
        fig_baselines = go.Figure()
        fig_baselines.add_trace(go.Bar(
            x=list(baselines.keys()),
            y=list(baselines.values()),
            marker_color=['#2ea043' if k == 'SAC RL Agent' else '#58a6ff' for k in baselines.keys()]
        ))
        fig_baselines.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            xaxis_title="Strategy",
            yaxis_title="Mean Reward"
        )
        st.plotly_chart(fig_baselines, use_container_width=True)
    
    def display_mission_timeline(self):
        """Display mission timeline visualization."""
        st.header("📈 Mission Timeline")
        
        st.markdown("### 🎯 Mission Phases")
        
        phases = {
            'Phase 1: Commissioning': {'days': (0, 30), 'status': '✅ Complete'},
            'Phase 2: Early Operations': {'days': (30, 90), 'status': '✅ Complete'},
            'Phase 3: Full Operations': {'days': (90, 365), 'status': '🔄 In Progress'},
            'Phase 4: Extended Mission': {'days': (365, 730), 'status': '⏳ Planned'},
            'Phase 5: End of Life': {'days': (730, 1095), 'status': '⏳ Planned'}
        }
        
        for phase, info in phases.items():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"**{phase}**")
            with col2:
                st.markdown(f"Days {info['days'][0]}-{info['days'][1]}")
            with col3:
                status_color = {'✅': 'green', '🔄': 'blue', '⏳': 'gray'}[info['status'].split()[0]]
                st.markdown(info['status'])
        
        st.markdown("---")
        
        # Milestones
        st.subheader("🏆 Key Milestones")
        
        milestones = [
            {'day': 0, 'name': 'Launch', 'status': '✅'},
            {'day': 1, 'name': 'First Contact', 'status': '✅'},
            {'day': 7, 'name': 'Solar Array Deployment', 'status': '✅'},
            {'day': 14, 'name': 'Initial Power Generation', 'status': '✅'},
            {'day': 30, 'name': 'Commissioning Complete', 'status': '✅'},
            {'day': 90, 'name': 'First Debris Capture', 'status': '🔄'},
            {'day': 180, 'name': '100 Full Cycles', 'status': '⏳'},
            {'day': 365, 'name': '1 Year Anniversary', 'status': '⏳'},
            {'day': 730, 'name': '2 Year Anniversary', 'status': '⏳'},
            {'day': 1095, 'name': 'Mission Complete', 'status': '⏳'}
        ]
        
        milestone_df = pd.DataFrame(milestones)
        st.dataframe(milestone_df, use_container_width=True)
    
    def run(self):
        """Run the dashboard."""
        choice, show_metrics, auto_refresh = self.sidebar_navigation()
        
        self.display_header()
        
        if choice == "📊 Power System Monitor":
            self.display_power_monitor()
        elif choice == "🔋 Battery Analytics":
            self.display_battery_analytics()
        elif choice == "☀️ MPPT Analysis":
            self.display_mppt_analysis()
        elif choice == "🌍 3D Orbit View":
            self.display_3d_orbit()
        elif choice == "🤖 RL Training Dashboard":
            self.display_rl_training()
        elif choice == "📈 Mission Timeline":
            self.display_mission_timeline()


def main():
    """Main entry point."""
    dashboard = H2ZDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()

