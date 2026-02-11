"""
Interactive Dashboard for H2Z Satellite Power System

This module provides professional-grade visualizations using Plotly and Dash.
Features:
- Real-time power system monitoring
- Interactive orbital visualization
- MPPT analysis plots
- AI model performance metrics
- Anomaly detection visualization

Author: H2Z Development Team
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

# Configure Plotly for dark theme (professional aerospace look)
pio.templates["custom_dark"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font_color="#c9d1d9",
        xaxis=dict(
            gridcolor="#21262d",
            zerolinecolor="#21262d"
        ),
        yaxis=dict(
            gridcolor="#21262d",
            zerolinecolor="#21262d"
        )
    )
)
pio.templates.default = "plotly_dark+custom_dark"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for dashboard visualization."""
    refresh_rate: int = 1000  # ms
    max_history_points: int = 1000
    theme: str = "dark"
    orbit_altitude_km: float = 500


class PowerSystemDashboard:
    """
    Interactive dashboard for satellite power system monitoring.
    
    Provides comprehensive visualization of:
    - Power generation and consumption
    - Battery state
    - Thermal status
    - MPPT performance
    - Subsystem allocations
    """
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        self.history_data = {
            'timestamps': [],
            'solar_power': [],
            'battery_soc': [],
            'total_demand': [],
            'subsystems': {
                'ADCS': [],
                'TT&C': [],
                'CDH': [],
                'Propulsion': [],
                'Communication': [],
                'Payload': []
            },
            'temperature': [],
            'eclipse': []
        }
        
        logger.info("PowerSystemDashboard initialized")
    
    def update_data(
        self,
        solar_power: float,
        battery_soc: float,
        demands: Dict[str, float],
        temperature: float,
        is_eclipse: bool,
        timestamp: datetime = None
    ):
        """Update dashboard with new data point."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.history_data['timestamps'].append(timestamp)
        self.history_data['solar_power'].append(solar_power)
        self.history_data['battery_soc'].append(battery_soc * 100)  # Convert to %
        self.history_data['total_demand'].append(sum(demands.values()))
        self.history_data['temperature'].append(temperature)
        self.history_data['eclipse'].append(1.0 if is_eclipse else 0.0)
        
        for name, power in demands.items():
            if name in self.history_data['subsystems']:
                self.history_data['subsystems'][name].append(power)
        
        # Limit history size
        max_points = self.config.max_history_points
        
        if len(self.history_data['timestamps']) > max_points:
            for key in self.history_data:
                if isinstance(self.history_data[key], list):
                    self.history_data[key] = self.history_data[key][-max_points:]
    
    def create_power_monitor(self) -> go.Figure:
        """Create power system monitoring visualization."""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Solar Power Generation',
                'Battery State of Charge',
                'Total Power Demand',
                'Subsystem Power Distribution',
                'Temperature Profile',
                'Eclipse Phase'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "indicator"}],
                [{"type": "scatter"}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        # Solar power over time
        fig.add_trace(
            go.Scatter(
                x=self.history_data['timestamps'],
                y=self.history_data['solar_power'],
                mode='lines+markers',
                name='Solar Power',
                line=dict(color='#2ea043', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Battery SOC gauge
        current_soc = self.history_data['battery_soc'][-1] if self.history_data['battery_soc'] else 0
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=current_soc,
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#c9d1d9"),
                    bar=dict(color="#58a6ff"),
                    bgcolor="#0d1117",
                    borderwidth=2,
                    bordercolor="#21262d",
                    steps=[
                        dict(range=[0, 20], color="#f85149"),
                        dict(range=[20, 50], color="#d29922"),
                        dict(range=[50, 100], color="#2ea043")
                    ],
                    threshold=dict(
                        line=dict(color="#ffffff", width=2),
                        thickness=0.75,
                        value=50
                    )
                ),
                title=dict(text="Battery SOC (%)")
            ),
            row=1, col=2
        )
        
        # Total demand over time
        fig.add_trace(
            go.Scatter(
                x=self.history_data['timestamps'],
                y=self.history_data['total_demand'],
                mode='lines+markers',
                name='Total Demand',
                line=dict(color='#f78166', width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
        
        # Subsystem pie chart
        current_demands = {
            name: self.history_data['subsystems'][name][-1] 
            for name in self.history_data['subsystems']
            if self.history_data['subsystems'][name]
        }
        
        fig.add_trace(
            go.Pie(
                labels=list(current_demands.keys()),
                values=list(current_demands.values()),
                hole=0.4,
                marker=dict(
                    colors=px.colors.qualitative.Set3
                ),
                textinfo='label+percent',
                showlegend=True
            ),
            row=2, col=2
        )
        
        # Temperature over time
        fig.add_trace(
            go.Scatter(
                x=self.history_data['timestamps'],
                y=self.history_data['temperature'],
                mode='lines+markers',
                name='Temperature',
                line=dict(color='#a371f7', width=2),
                fill='tozeroy',
                fillcolor='rgba(163, 113, 246, 0.2)'
            ),
            row=3, col=1
        )
        
        # Eclipse phase bar
        eclipse_colors = ['#f78166' if e else '#2ea043' for e in self.history_data['eclipse']]
        
        fig.add_trace(
            go.Bar(
                x=self.history_data['timestamps'],
                y=self.history_data['eclipse'],
                name='Eclipse',
                marker_color=eclipse_colors,
                showlegend=False
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='🛰️ H2Z Satellite Power System Monitor',
                font=dict(size=20)
            ),
            height=900,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_mppt_analysis(self, mppt_data: Dict[str, np.ndarray]) -> go.Figure:
        """Create MPPT efficiency analysis visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'MPPT Efficiency Over Time',
                'Power Gain Comparison',
                'Tracking Accuracy',
                'Temperature Effect'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        time_years = mppt_data['time_years']
        mppt_efficiency = mppt_data['mppt_efficiency']
        fixed_efficiency = mppt_data['fixed_efficiency']
        advantage_percent = mppt_data['advantage_percent']
        
        # Efficiency comparison
        fig.add_trace(
            go.Scatter(
                x=time_years,
                y=mppt_efficiency * 100,
                mode='lines',
                name='MPPT System',
                line=dict(color='#2ea043', width=3)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=time_years,
                y=fixed_efficiency * 100,
                mode='lines',
                name='Fixed System',
                line=dict(color='#f78166', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Power gain
        fig.add_trace(
            go.Scatter(
                x=time_years,
                y=advantage_percent,
                mode='lines+markers',
                name='Power Gain',
                line=dict(color='#58a6ff', width=3),
                fill='tozeroy',
                fillcolor='rgba(88, 166, 255, 0.2)'
            ),
            row=1, col=2
        )
        
        # Tracking accuracy (simulated data)
        accuracy = 99.2 - 0.3 * time_years * 3
        fig.add_trace(
            go.Scatter(
                x=time_years,
                y=accuracy,
                mode='lines',
                name='Tracking Accuracy',
                line=dict(color='#a371f7', width=3)
            ),
            row=2, col=1
        )
        
        # Temperature effect
        temperatures = np.linspace(0, 100, 101)
        temp_efficiency = 97 - 0.08 * temperatures
        
        fig.add_trace(
            go.Scatter(
                x=temperatures,
                y=temp_efficiency,
                mode='lines',
                name='Efficiency',
                line=dict(color='#f78166', width=3)
            ),
            row=2, col=2
        )
        
        # Add minimum acceptable line
        fig.add_hline(y=95, line_dash="dash", row=2, col=1, 
                      line_color="#f85149", annotation_text="Minimum (95%)")
        
        fig.update_layout(
            title=dict(
                text='📊 MPPT Efficiency Analysis',
                font=dict(size=20)
            ),
            height=700,
            showlegend=True
        )
        
        return fig
    
    def create_orbit_visualization(
        self,
        orbit_data: Dict[str, Any]
    ) -> go.Figure:
        """Create 3D orbital visualization."""
        # Create orbit path
        altitude = orbit_data.get('altitude_km', self.config.orbit_altitude_km)
        radius = 6371 + altitude
        
        theta = np.linspace(0, 2 * np.pi, 100)
        
        # Earth sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_earth = 6371 * np.outer(np.cos(u), np.sin(v))
        y_earth = 6371 * np.outer(np.sin(u), np.sin(v))
        z_earth = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig = go.Figure()
        
        # Add Earth
        fig.add_trace(go.Surface(
            x=x_earth, y=y_earth, z=z_earth,
            colorscale='Blues',
            showscale=False,
            opacity=0.8,
            name='Earth'
        ))
        
        # Add orbit path
        orbit_x = radius * np.cos(theta)
        orbit_y = radius * np.sin(theta)
        orbit_z = np.zeros_like(theta) * altitude / 100  # Slight inclination
        
        fig.add_trace(go.Scatter3d(
            x=orbit_x, y=orbit_y, z=orbit_z,
            mode='lines',
            line=dict(color='#2ea043', width=4),
            name='Orbit'
        ))
        
        # Add satellite position
        current_theta = datetime.now().second / 60 * 2 * np.pi
        sat_x = radius * np.cos(current_theta)
        sat_y = radius * np.sin(current_theta)
        sat_z = 0
        
        fig.add_trace(go.Scatter3d(
            x=[sat_x], y=[sat_y], z=[sat_z],
            mode='markers',
            marker=dict(size=10, color='#58a6ff'),
            name='Satellite'
        ))
        
        # Add sun direction
        sun_x = [0, 20000]
        sun_y = [0, 0]
        sun_z = [0, 0]
        
        fig.add_trace(go.Scatter3d(
            x=sun_x, y=sun_y, z=sun_z,
            mode='lines',
            line=dict(color='#f78166', width=4),
            name='Sun Direction'
        ))
        
        fig.update_layout(
            title=dict(
                text='🪐 Satellite Orbit Visualization',
                font=dict(size=20)
            ),
            scene=dict(
                xaxis=dict(title='X (km)', showbackground=False),
                yaxis=dict(title='Y (km)', showbackground=False),
                zaxis=dict(title='Z (km)', showbackground=False),
                aspectmode='data'
            ),
            height=700,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        return fig
    
    def create_anomaly_dashboard(
        self,
        reconstruction_errors: np.ndarray,
        threshold: float,
        timestamps: pd.DatetimeIndex = None
    ) -> go.Figure:
        """Create anomaly detection dashboard."""
        if timestamps is None:
            timestamps = pd.date_range(
                start=datetime.now(),
                periods=len(reconstruction_errors),
                freq='5min'
            )
        
        anomalies = reconstruction_errors > threshold
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                'Reconstruction Error Over Time',
                'Anomaly Detection Summary'
            ),
            specs=[
                [{"type": "scatter"}],
                [{"type": "histogram"}]
            ]
        )
        
        # Reconstruction error
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=reconstruction_errors,
                mode='lines+markers',
                name='Reconstruction Error',
                line=dict(color='#58a6ff', width=2),
                marker=dict(
                    size=4,
                    color=['#f85149' if a else '#2ea043' for a in anomalies]
                )
            ),
            row=1, col=1
        )
        
        # Threshold line
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#f78166",
            row=1, col=1,
            annotation_text="Anomaly Threshold"
        )
        
        # Histogram
        normal_errors = reconstruction_errors[~anomalies]
        anomaly_errors = reconstruction_errors[anomalies]
        
        fig.add_trace(
            go.Histogram(
                x=normal_errors,
                name='Normal',
                marker_color='#2ea043',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=anomaly_errors,
                name='Anomaly',
                marker_color='#f85149',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=dict(
                text='🚨 Anomaly Detection Dashboard',
                font=dict(size=20)
            ),
            height=600,
            barmode='overlay',
            showlegend=True
        )
        
        return fig
    
    def create_ai_performance_dashboard(
        self,
        training_history: Dict[str, List],
        evaluation_metrics: Dict[str, float]
    ) -> go.Figure:
        """Create AI model performance dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Training Loss',
                'Episode Rewards',
                'Prediction Accuracy',
                'Model Metrics'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "indicator"}]
            ]
        )
        
        # Training loss
        if 'train_loss' in training_history:
            steps = range(len(training_history['train_loss']))
            fig.add_trace(
                go.Scatter(
                    x=list(steps),
                    y=training_history['train_loss'],
                    mode='lines',
                    name='Training Loss',
                    line=dict(color='#2ea043', width=2)
                ),
                row=1, col=1
            )
        
        # Episode rewards
        if 'episode_rewards' in training_history:
            episodes = range(len(training_history['episode_rewards']))
            fig.add_trace(
                go.Scatter(
                    x=list(episodes),
                    y=training_history['episode_rewards'],
                    mode='lines',
                    name='Episode Reward',
                    line=dict(color='#58a6ff', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(88, 166, 255, 0.2)'
                ),
                row=1, col=2
            )
        
        # Prediction accuracy bar chart
        if 'accuracy' in evaluation_metrics:
            fig.add_trace(
                go.Bar(
                    x=['MSE', 'MAE', 'R² Score'],
                    y=[
                        evaluation_metrics.get('mse', 0),
                        evaluation_metrics.get('mae', 0),
                        evaluation_metrics.get('r2', 0)
                    ],
                    marker_color=['#f78166', '#a371f7', '#2ea043'],
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Overall model indicator
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=evaluation_metrics.get('mean_reward', 0),
                gauge=dict(
                    axis=dict(range=[0, 100]),
                    bar=dict(color="#58a6ff"),
                    steps=[
                        dict(range=[0, 50], color="#f85149"),
                        dict(range=[50, 80], color="#d29922"),
                        dict(range=[80, 100], color="#2ea043")
                    ]
                ),
                title=dict(text="Model Score")
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=dict(
                text='🤖 AI Model Performance Dashboard',
                font=dict(size=20)
            ),
            height=700,
            showlegend=True
        )
        
        return fig


class ThermalVisualizer:
    """Thermal analysis visualization."""
    
    @staticmethod
    def create_thermal_map(
        temperature_data: np.ndarray,
        labels: List[str]
    ) -> go.Figure:
        """Create thermal distribution heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=temperature_data,
            x=labels,
            y=[f'Component {i+1}' for i in range(len(temperature_data))],
            colorscale='RdBu_r',
            colorbar=dict(title='Temperature (°C)')
        ))
        
        fig.update_layout(
            title=dict(text='🌡️ Subsystem Thermal Distribution', font=dict(size=20)),
            height=500,
            xaxis_title='Time',
            yaxis_title='Component'
        )
        
        return fig
    
    @staticmethod
    def create_thermal_profile(
        time_data: np.ndarray,
        surface_temp: np.ndarray,
        internal_temp: np.ndarray
    ) -> go.Figure:
        """Create temperature profile over time."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_data,
            y=surface_temp,
            mode='lines',
            name='Surface Temperature',
            line=dict(color='#f78166', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=time_data,
            y=internal_temp,
            mode='lines',
            name='Internal Temperature',
            line=dict(color='#58a6ff', width=3)
        ))
        
        fig.update_layout(
            title=dict(text='🌡️ Thermal Profile Over Time', font=dict(size=20)),
            xaxis_title='Time',
            yaxis_title='Temperature (°C)',
            height=500
        )
        
        return fig


def generate_report_html(
    dashboard: PowerSystemDashboard,
    output_path: str = "h2z_dashboard_report.html"
) -> str:
    """Generate comprehensive HTML report."""
    
    # Create all figures
    power_fig = dashboard.create_power_monitor()
    
    # Generate report content
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>H2Z Satellite Power System - Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
            color: #c9d1d9;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            padding: 40px 0;
            border-bottom: 2px solid #21262d;
            margin-bottom: 40px;
        }}
        h1 {{
            color: #58a6ff;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .subtitle {{
            color: #8b949e;
            font-size: 1.2em;
        }}
        .section {{
            background: #0d1117;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        .section-title {{
            color: #2ea043;
            font-size: 1.5em;
            margin-bottom: 16px;
            border-bottom: 1px solid #21262d;
            padding-bottom: 8px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-top: 20px;
        }}
        .metric-card {{
            background: #161b22;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #58a6ff;
        }}
        .metric-label {{
            color: #8b949e;
            font-size: 0.9em;
            margin-top: 4px;
        }}
        .plot-container {{
            margin: 20px 0;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #8b949e;
            border-top: 1px solid #21262d;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛰️ H2Z Satellite Power System</h1>
            <p class="subtitle">AI-Enhanced Power & Communication Subsystem Analysis Report</p>
            <p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2 class="section-title">📊 System Overview</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{sum(dashboard.history_data['solar_power'][-100:])/100:.1f} W</div>
                    <div class="metric-label">Avg Solar Power</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{dashboard.history_data['battery_soc'][-1] if dashboard.history_data['battery_soc'] else 0:.1f}%</div>
                    <div class="metric-label">Current Battery SOC</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{sum(dashboard.history_data['total_demand'][-100:])/100:.1f} W</div>
                    <div class="metric-label">Avg Power Demand</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{np.mean(dashboard.history_data['temperature'][-100:]):.1f}°C</div>
                    <div class="metric-label">Avg Temperature</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">⚡ Power System Monitor</h2>
            <div class="plot-container">
                {power_fig.to_html(full_html=False, include_plotlyjs='cdn')}
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">🤖 AI/ML Capabilities</h2>
            <ul>
                <li>LSTM-based Solar Irradiance Forecasting</li>
                <li>Physics-Informed Neural Networks for Battery Degradation</li>
                <li>Autoencoder-based Anomaly Detection</li>
                <li>Genetic Algorithm for Power Allocation Optimization</li>
                <li>PPO Reinforcement Learning Agent</li>
            </ul>
        </div>
        
        <div class="footer">
            <p>H2Z Satellite Power & Communication Subsystem</p>
            <p>© 2024 All Rights Reserved</p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Report generated: {output_path}")
    return output_path


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Dashboard Visualization Demo")
    logger.info("=" * 60)
    
    # Create dashboard
    dashboard = PowerSystemDashboard()
    
    # Generate sample data
    import numpy as np
    from datetime import datetime, timedelta
    
    base_time = datetime.now()
    
    for i in range(100):
        timestamp = base_time + timedelta(minutes=i*5)
        
        # Simulate sinusoidal solar power
        solar_power = 800 + 200 * np.sin(i * 0.1)
        
        # Battery SOC with some noise
        battery_soc = 0.7 + 0.1 * np.sin(i * 0.05)
        
        # Demands
        demands = {
            'ADCS': 35 + 5 * np.random.randn(),
            'TT&C': 18 + 3 * np.random.randn(),
            'CDH': 12 + 2 * np.random.randn(),
            'Propulsion': 90 + 10 * np.random.randn(),
            'Communication': 25 + 4 * np.random.randn(),
            'Payload': 12 + 2 * np.random.randn()
        }
        
        temperature = 25 + 5 * np.sin(i * 0.1)
        is_eclipse = (i % 20) > 15
        
        dashboard.update_data(solar_power, battery_soc, demands, temperature, is_eclipse, timestamp)
    
    # Create power monitor
    fig = dashboard.create_power_monitor()
    
    # Save interactive HTML
    output_path = "h2z_power_dashboard.html"
    fig.write_html(output_path)
    logger.info(f"Dashboard saved to: {output_path}")
    
    # Generate MPPT analysis
    mppt_data = {
        'time_years': np.linspace(0, 3, 100),
        'mppt_efficiency': 0.97 * (1 - 0.005 * np.linspace(0, 3, 100)),
        'fixed_efficiency': np.full(100, 0.85),
        'advantage_percent': (0.97 - 0.85) * 100 * (1 - 0.005 * np.linspace(0, 3, 100))
    }
    
    mppt_fig = dashboard.create_mppt_analysis(mppt_data)
    mppt_fig.write_html("h2z_mppt_analysis.html")
    logger.info("MPPT analysis saved")
    
    # Generate report
    report_path = generate_report_html(dashboard, "h2z_full_report.html")
    
    logger.info("=" * 60)
    logger.info("Dashboard demo completed successfully!")

