"""
Battery Life Optimization Visualization Module

Generates comprehensive visualizations for:
- Training metrics (TensorBoard-style plots)
- Battery degradation over time
- SOH projections
- Strategy comparison
- Action analysis

Author: H2Z Development Team
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
import json

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

# Configure Plotly
pio.templates.default = "plotly_dark"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    theme: str = "dark"
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 150
    color_scheme: Dict[str, str] = None


class BatteryResultsVisualizer:
    """Visualization tools for battery optimization results."""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        
        # Color scheme
        self.colors = {
            'sac': '#2ea043',       # Green
            'baseline': '#f78166',  # Orange
            'optimal': '#58a6ff',   # Blue
            'danger': '#f85149',    # Red
            'warning': '#d29922',   # Yellow
            'success': '#2ea043',   # Green
            'primary': '#a371f7',   # Purple
            'secondary': '#8b949e'  # Gray
        }
        
        logger.info("BatteryResultsVisualizer initialized")
    
    def plot_training_metrics(
        self,
        history: Dict[str, List],
        save_path: str = None
    ) -> go.Figure:
        """
        Plot training metrics over time.
        
        Includes:
        - Episode rewards
        - Critic/Actor losses
        - Entropy alpha
        - Mean Q values
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Episode Rewards',
                'Policy Loss',
                'Value Loss',
                'Training Metrics'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # Episode rewards
        if 'episode_rewards' in history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(history['episode_rewards']))),
                    y=history['episode_rewards'],
                    mode='lines',
                    name='Reward',
                    line=dict(color=self.colors['sac'], width=2),
                    fill='tozeroy',
                    fillcolor='rgba(46, 160, 67, 0.2)'
                ),
                row=1, col=1
            )
        
        # Moving average
        if 'episode_rewards' in history and len(history['episode_rewards']) > 10:
            window = 10
            ma = pd.Series(history['episode_rewards']).rolling(window).mean()
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(ma))),
                    y=ma,
                    mode='lines',
                    name=f'{window}-Episode MA',
                    line=dict(color=self.colors['primary'], width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        # Actor loss
        if 'actor_losses' in history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(history['actor_losses']))),
                    y=history['actor_losses'],
                    mode='lines',
                    name='Actor Loss',
                    line=dict(color=self.colors['primary'], width=1),
                    opacity=0.7
                ),
                row=1, col=2
            )
        
        # Critic loss
        if 'critic_losses' in history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(history['critic_losses']))),
                    y=history['critic_losses'],
                    mode='lines',
                    name='Critic Loss',
                    line=dict(color=self.colors['secondary'], width=1),
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Alpha (entropy temperature)
        if 'alphas' in history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(history['alphas']))),
                    y=history['alphas'],
                    mode='lines',
                    name='Alpha',
                    line=dict(color=self.colors['warning'], width=2)
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=dict(
                text='🎓 SAC Training Metrics',
                font=dict(size=20)
            ),
            height=700,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(title_text="Episode", row=1, col=1)
        fig.update_xaxes(title_text="Update Step", row=1, col=2)
        fig.update_xaxes(title_text="Update Step", row=2, col=1)
        fig.update_xaxes(title_text="Update Step", row=2, col=2)
        
        fig.update_yaxes(title_text="Reward", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=2, col=1)
        fig.update_yaxes(title_text="Alpha", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Training metrics saved to: {save_path}")
        
        return fig
    
    def plot_soh_projection(
        self,
        days: np.ndarray,
        soh_values: np.ndarray,
        strategy_name: str = "Strategy",
        save_path: str = None
    ) -> go.Figure:
        """Plot State of Health projection over mission lifetime."""
        fig = go.Figure()
        
        # SOH curve
        fig.add_trace(
            go.Scatter(
                x=days,
                y=soh_values * 100,
                mode='lines',
                name=f'{strategy_name} SOH',
                line=dict(color=self.colors['sac'], width=3),
                fill='tozeroy',
                fillcolor='rgba(46, 160, 67, 0.1)'
            )
        )
        
        # 80% EOL threshold
        fig.add_hline(
            y=80,
            line_dash="dash",
            line_color=self.colors['warning'],
            annotation_text="EOL Threshold (80%)"
        )
        
        # 3-year mark
        fig.add_vline(
            x=1095,
            line_dash="dot",
            line_color=self.colors['secondary'],
            annotation_text="3 Years"
        )
        
        fig.update_layout(
            title=dict(
                text=f'📉 Battery State of Health Projection - {strategy_name}',
                font=dict(size=20)
            ),
            xaxis_title="Mission Days",
            yaxis_title="State of Health (%)",
            height=500,
            xaxis=dict(range=[0, 1100]),
            yaxis=dict(range=[50, 105]),
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"SOH projection saved to: {save_path}")
        
        return fig
    
    def plot_soh_comparison(
        self,
        results: Dict[str, Dict[str, np.ndarray]],
        save_path: str = None
    ) -> go.Figure:
        """Compare SOH projections across strategies."""
        fig = go.Figure()
        
        colors = [self.colors['sac'], self.colors['baseline'], 
                  self.colors['primary'], self.colors['warning']]
        
        for i, (name, data) in enumerate(results.items()):
            days = data.get('days', np.arange(len(data['SOH'])))
            soh = data['SOH'] * 100
            
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=soh,
                    mode='lines',
                    name=name,
                    line=dict(color=colors[i % len(colors)], width=2),
                    opacity=0.8
                )
            )
        
        # EOL threshold
        fig.add_hline(
            y=80,
            line_dash="dash",
            line_color=self.colors['danger'],
            annotation_text="EOL (80%)",
            annotation_position="bottom right"
        )
        
        fig.update_layout(
            title=dict(
                text='📊 SOH Comparison Across Strategies',
                font=dict(size=20)
            ),
            xaxis_title="Mission Days",
            yaxis_title="State of Health (%)",
            height=600,
            xaxis=dict(range=[0, 1100]),
            yaxis=dict(range=[50, 105]),
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"SOH comparison saved to: {save_path}")
        
        return fig
    
    def plot_strategy_comparison(
        self,
        metrics: Dict[str, Dict[str, Any]],
        save_path: str = None
    ) -> go.Figure:
        """Create bar chart comparing strategies."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Final SOH (%)',
                'Mean Reward',
                'Safety Violations',
                'Lithium Plating Events'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        strategies = list(metrics.keys())
        colors = [self.colors['sac'] if 'SAC' in s else 
                  self.colors['primary'] for s in strategies]
        
        # Final SOH
        soh_values = [m['final_soh']['mean'] * 100 for m in metrics.values()]
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=soh_values,
                marker_color=colors,
                name='Final SOH'
            ),
            row=1, col=1
        )
        
        # Mean Reward
        reward_values = [m['reward']['mean'] for m in metrics.values()]
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=reward_values,
                marker_color=colors,
                name='Mean Reward'
            ),
            row=1, col=2
        )
        
        # Safety Violations
        violation_values = [m['safety_violations']['mean'] for m in metrics.values()]
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=violation_values,
                marker_color=colors,
                name='Safety Violations'
            ),
            row=2, col=1
        )
        
        # Plating Events
        plating_values = [m['lithium_plating_events']['mean'] for m in metrics.values()]
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=plating_values,
                marker_color=colors,
                name='Plating Events'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=dict(
                text='📊 Strategy Performance Comparison',
                font=dict(size=20)
            ),
            height=700,
            showlegend=False
        )
        
        fig.update_xaxes(tickangle=45)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Strategy comparison saved to: {save_path}")
        
        return fig
    
    def plot_degradation_analysis(
        self,
        degradation_history: Dict[str, List],
        save_path: str = None
    ) -> go.Figure:
        """Plot degradation metrics over time."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Capacity Fade',
                'Internal Resistance Growth',
                'SEI Layer Thickness',
                'Degradation Rate'
            )
        )
        
        days = degradation_history.get('time_days', [])
        
        # Capacity fade
        fade = degradation_history.get('capacity_fade', [])
        fig.add_trace(
            go.Scatter(
                x=days,
                y=[f * 100 for f in fade],
                mode='lines',
                name='Capacity Fade (%)',
                line=dict(color=self.colors['danger'], width=2)
            ),
            row=1, col=1
        )
        
        # Internal resistance
        r_int = degradation_history.get('R_int', [])
        fig.add_trace(
            go.Scatter(
                x=days,
                y=r_int,
                mode='lines',
                name='R_int (Ohms)',
                line=dict(color=self.colors['warning'], width=2)
            ),
            row=1, col=2
        )
        
        # SEI thickness
        sei = degradation_history.get('sei_thickness', [])
        fig.add_trace(
            go.Scatter(
                x=days,
                y=sei,
                mode='lines',
                name='SEI Thickness (nm)',
                line=dict(color=self.colors['primary'], width=2)
            ),
            row=2, col=1
        )
        
        # Cycles completed
        cycles = degradation_history.get('cycles', [])
        fig.add_trace(
            go.Scatter(
                x=days,
                y=cycles,
                mode='lines',
                name='Cycles',
                line=dict(color=self.colors['success'], width=2)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=dict(
                text='🔬 Battery Degradation Analysis',
                font=dict(size=20)
            ),
            height=700,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Degradation analysis saved to: {save_path}")
        
        return fig
    
    def plot_action_distribution(
        self,
        actions: np.ndarray,
        action_names: List[str] = None,
        save_path: str = None
    ) -> go.Figure:
        """Plot distribution of actions taken by agent."""
        if action_names is None:
            action_names = [
                'Charge Current (A)',
                'Discharge Limit (A)',
                'Voltage (V)',
                'Heater Power (W)',
                'MPPT Target'
            ]
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=action_names[:5],
            specs=[
                [{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
                [{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}]
            ]
        )
        
        colors = [self.colors['sac'], self.colors['primary'], self.colors['warning'],
                  self.colors['success'], self.colors['baseline']]
        
        for i in range(5):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            fig.add_trace(
                go.Histogram(
                    x=actions[:, i],
                    name=action_names[i],
                    marker_color=colors[i],
                    opacity=0.7
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=dict(
                text='🎮 Action Distribution (SAC Agent)',
                font=dict(size=20)
            ),
            height=600,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Action distribution saved to: {save_path}")
        
        return fig
    
    def create_summary_dashboard(
        self,
        training_history: Dict,
        evaluation_metrics: Dict[str, Dict],
        degradation_results: Dict[str, np.ndarray],
        save_path: str = None
    ) -> go.Figure:
        """Create comprehensive summary dashboard."""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Training Rewards',
                'SOH Comparison',
                'Strategy Performance',
                'Degradation Metrics',
                'Loss Curves',
                'Action Analysis',
                'Key Metrics',
                'Improvement Summary',
                'Recommendations'
            ),
            specs=[
                [{"type": "scatter", "colspan": 1}, {"type": "scatter", "colspan": 1}, {"type": "bar", "colspan": 1}],
                [{"type": "scatter", "colspan": 1}, {"type": "scatter", "colspan": 1}, {"type": "histogram", "colspan": 1}],
                [{"type": "table", "colspan": 3}, None, None]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        # Training rewards
        if 'episode_rewards' in training_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(training_history['episode_rewards']))),
                    y=training_history['episode_rewards'],
                    mode='lines',
                    name='Reward',
                    line=dict(color=self.colors['sac'], width=2),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # SOH comparison
        for i, (name, data) in enumerate(degradation_results.items()):
            if 'SOH' in data:
                days = data.get('days', np.arange(len(data['SOH'])))
                fig.add_trace(
                    go.Scatter(
                        x=days,
                        y=data['SOH'] * 100,
                        mode='lines',
                        name=name,
                        line=dict(width=2),
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # Performance bar chart
        strategies = list(evaluation_metrics.keys())
        soh_values = [m['final_soh']['mean'] * 100 for m in evaluation_metrics.values()]
        colors_bar = [self.colors['sac'] if 'SAC' in s else self.colors['baseline'] 
                     for s in strategies]
        
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=soh_values,
                marker_color=colors_bar,
                showlegend=False
            ),
            row=1, col=3
        )
        
        # Degradation metrics
        days = degradation_results.get('time_days', [])
        if 'capacity_fade' in degradation_results:
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=[f * 100 for f in degradation_results['capacity_fade']],
                    mode='lines',
                    name='Fade',
                    line=dict(color=self.colors['danger'], width=2),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Loss curves
        if 'critic_losses' in training_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(training_history['critic_losses']))),
                    y=training_history['critic_losses'],
                    mode='lines',
                    name='Critic Loss',
                    line=dict(color=self.colors['primary'], width=1),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Action histogram
        if 'actions' in training_history:
            actions = training_history['actions']
            fig.add_trace(
                go.Histogram(
                    x=actions[:, 0],
                    name='Charge Current',
                    marker_color=self.colors['sac'],
                    showlegend=False
                ),
                row=2, col=3
            )
        
        # Summary table
        table_data = []
        for name, metrics in evaluation_metrics.items():
            table_data.append([
                name,
                f"{metrics['final_soh']['mean']*100:.1f}%",
                f"{metrics['reward']['mean']:.1f}",
                f"{metrics['safety_violations']['mean']:.1f}"
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Strategy', 'Final SOH', 'Reward', 'Violations'],
                    fill_color=self.colors['primary'],
                    font=dict(color='white')
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    fill_color=[[self.colors['secondary']] * len(table_data)]
                )
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title=dict(
                text='🛰️ H2Z Battery Life Optimization - Summary Dashboard',
                font=dict(size=24)
            ),
            height=1200,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Summary dashboard saved to: {save_path}")
        
        return fig


def load_training_history(filepath: str) -> Dict:
    """Load training history from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_evaluation_results(filepath: str) -> Dict:
    """Load evaluation results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Battery Results Visualization Demo")
    logger.info("=" * 60)
    
    # Create visualizer
    viz = BatteryResultsVisualizer()
    
    # Generate sample training data
    training_history = {
        'episode_rewards': np.random.randn(100).cumsum() * 10 + 100,
        'actor_losses': np.abs(np.random.randn(1000)) * 0.1,
        'critic_losses': np.abs(np.random.randn(1000)) * 0.5,
        'alphas': np.random.uniform(0.1, 0.3, 1000)
    }
    
    # Plot training metrics
    fig = viz.plot_training_metrics(training_history, "training_metrics.html")
    logger.info("Training metrics plot created")
    
    # Sample SOH projection
    days = np.linspace(0, 1095, 100)
    soh_sac = 1.0 - 0.0001 * (days / 365) ** 1.5
    soh_baseline = 1.0 - 0.0002 * (days / 365) ** 1.5
    
    fig = viz.plot_soh_projection(days, soh_sac, "SAC Agent", "soh_sac.html")
    logger.info("SOH projection created")
    
    fig = viz.plot_soh_comparison({
        'SAC Agent': {'days': days, 'SOH': soh_sac},
        'Baseline': {'days': days, 'SOH': soh_baseline}
    }, "soh_comparison.html")
    logger.info("SOH comparison created")
    
    # Sample evaluation metrics
    evaluation_metrics = {
        'SAC Agent': {
            'reward': {'mean': 150.5, 'std': 20.3},
            'final_soh': {'mean': 0.92, 'std': 0.02},
            'safety_violations': {'mean': 0.5},
            'lithium_plating_events': {'mean': 0.1}
        },
        'Rule-Based': {
            'reward': {'mean': 80.2, 'std': 30.1},
            'final_soh': {'mean': 0.75, 'std': 0.05},
            'safety_violations': {'mean': 3.2},
            'lithium_plating_events': {'mean': 2.5}
        }
    }
    
    fig = viz.plot_strategy_comparison(evaluation_metrics, "strategy_comparison.html")
    logger.info("Strategy comparison created")
    
    logger.info("\n" + "=" * 60)
    logger.info("Visualization demo completed!")
    logger.info("Generated files:")
    logger.info("  - training_metrics.html")
    logger.info("  - soh_sac.html")
    logger.info("  - soh_comparison.html")
    logger.info("  - strategy_comparison.html")

