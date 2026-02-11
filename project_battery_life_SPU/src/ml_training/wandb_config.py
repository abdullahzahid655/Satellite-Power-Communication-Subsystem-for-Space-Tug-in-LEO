"""
H2Z Weights & Biases Integration

Integration with W&B for real-time training visualization,
experiment tracking, and collaboration.

Features:
- Real-time training metrics logging
- Hyperparameter sweeps
- Custom dashboards
- Team collaboration

Author: H2Z Development Team
"""

import wandb
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Optional, Any, List
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class H2ZWandBTracker:
    """W&B integration for H2Z RL training."""
    
    def __init__(
        self,
        project_name: str = "H2Z-Battery-Optimization",
        entity: Optional[str] = None,
        config: Optional[Dict] = None,
        settings: Optional[Dict] = None
    ):
        """
        Initialize W&B tracker.
        
        Args:
            project_name: W&B project name
            entity: W&B entity (username or team)
            config: Initial configuration/hyperparameters
            settings: W&B settings dictionary
        """
        self.project_name = project_name
        self.entity = entity
        self.config = config or {}
        self.settings = settings or {
            'offline': False,  # Set True for offline mode
            'sync_tensorboard': False,
            'save_code': True
        }
        
        self.run = None
        self.history_buffer = {
            'rewards': [],
            'soh_values': [],
            'violations': []
        }
    
    def init(
        self,
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None
    ):
        """
        Initialize a new W&B run.
        
        Args:
            run_name: Name for this run
            config: Hyperparameters and settings
            tags: List of tags
            notes: Notes for the run
        """
        run_name = run_name or f"H2Z_RL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        run_config = {**self.config, **(config or {})}
        
        wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=run_name,
            config=run_config,
            tags=tags,
            notes=notes,
            settings=self.settings
        )
        
        self.run = wandb.run
        logger.info(f"Initialized W&B run: {run_name}")
        
        return self.run
    
    def finish(self):
        """Finish the current W&B run."""
        if self.run:
            wandb.finish()
            logger.info("Finished W&B run")
    
    def log_metrics(self, metrics: Dict, step: Optional[int] = None):
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number
        """
        wandb.log(metrics, step=step)
    
    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        soh: float,
        violations: int,
        energy: float,
        info: Optional[Dict] = None
    ):
        """
        Log training episode summary.
        
        Args:
            episode: Episode number
            reward: Total episode reward
            length: Episode length (steps)
            soh: Final state of health
            violations: Safety violations count
            energy: Energy processed (Wh)
            info: Additional information
        """
        metrics = {
            'episode_reward': reward,
            'episode_length': length,
            'battery_soh': soh,
            'safety_violations': violations,
            'energy_processed_wh': energy
        }
        
        self.log_metrics(metrics, step=episode)
        
        # Add to history buffer
        self.history_buffer['rewards'].append(reward)
        self.history_buffer['soh_values'].append(soh)
        self.history_buffer['violations'].append(violations)
    
    def log_training_batch(
        self,
        step: int,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        learning_rate: float
    ):
        """
        Log SAC training batch metrics.
        
        Args:
            step: Global training step
            policy_loss: Policy network loss
            value_loss: Value network loss
            entropy: Policy entropy
            learning_rate: Current learning rate
        """
        metrics = {
            'train/policy_loss': policy_loss,
            'train/value_loss': value_loss,
            'train/entropy': entropy,
            'train/learning_rate': learning_rate
        }
        
        self.log_metrics(metrics, step=step)
    
    def log_eval_results(
        self,
        episode: int,
        mean_reward: float,
        std_reward: float,
        mean_soh: float,
        mean_violations: float,
        baseline_rewards: Dict[str, float]
    ):
        """
        Log evaluation results.
        
        Args:
            episode: Episode number at evaluation
            mean_reward: Mean evaluation reward
            std_reward: Standard deviation of rewards
            mean_soh: Mean final SOH
            mean_violations: Mean safety violations
            baseline_rewards: Dictionary of baseline strategy rewards
        """
        metrics = {
            'eval/mean_reward': mean_reward,
            'eval/std_reward': std_reward,
            'eval/mean_soh': mean_soh,
            'eval/mean_violations': mean_violations
        }
        
        # Add baseline comparisons
        for name, reward in baseline_rewards.items():
            metrics[f'eval/baseline_{name.replace(" ", "_")}'] = reward
        
        # Calculate improvement over best baseline
        best_baseline = min(baseline_rewards.values())
        improvement = (mean_reward - best_baseline) / abs(best_baseline) * 100
        metrics['eval/improvement_over_baseline_%'] = improvement
        
        self.log_metrics(metrics, step=episode)
    
    def log_table(self, table_name: str, data: List, columns: List[str]):
        """
        Log a table to W&B.
        
        Args:
            table_name: Name for the table
            data: List of rows
            columns: Column names
        """
        table = wandb.Table(columns=columns, data=data)
        wandb.log({table_name: table})
    
    def watch_model(self, model: torch.nn.Module):
        """
        Watch a PyTorch model for gradient tracking.
        
        Args:
            model: PyTorch model to watch
        """
        wandb.watch(model, log='all')
    
    def log_model_checkpoint(
        self,
        checkpoint_path: str,
        episode: int,
        metrics: Dict
    ):
        """
        Log a model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            episode: Episode number
            metrics: Associated metrics
        """
        artifact = wandb.Artifact(
            name=f"model_checkpoint_ep{episode}",
            type="model"
        )
        artifact.add_file(checkpoint_path)
        
        # Add metadata
        with artifact.new_json("metrics.json") as f:
            json.dump(metrics, f)
        
        wandb.log_artifact(artifact)
    
    def create_training_dashboard(self):
        """Create a W&B dashboard for training visualization."""
        # Define panels for the dashboard
        panels = [
            wandb.panel.Table(
                dataframe_name="training_history",
                fields=["episode", "reward", "soh", "violations"],
                filterable=["episode", "violations"],
                sorts=["episode"],
                name="training_history_table"
            ),
            wandb.panel.Media(
                matplotlib_fig_name="reward_curve",
                caption="Episode Rewards Over Training"
            ),
            wandb.panel.Media(
                matplotlib_fig_name="soh_curve",
                caption="Battery SOH Over Training"
            ),
            wandb.panel.Value(
                field_name="eval/mean_reward",
                name="Current Eval Reward"
            ),
            wandb.panel.Value(
                field_name="eval/improvement_over_baseline_%",
                name="Improvement %"
            )
        ]
        
        return panels
    
    def log_config_table(self, config: Dict):
        """Log configuration as a W&B table."""
        data = [[k, str(v)] for k, v in config.items()]
        table = wandb.Table(columns=["Parameter", "Value"], data=data)
        wandb.log({"configuration": table})
    
    def log_battery_sweep(self, soh_trajectory: List[float], time_steps: List[int]):
        """Log battery SOH trajectory over time."""
        # Create line plot
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.plot(time_steps, soh_trajectory)
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Battery SOH (%)")
        ax.set_title("Battery State of Health During Training")
        
        wandb.log({"soh_trajectory": wandb.Image(fig)})
        plt.close(fig)
    
    def start_sweep(
        self,
        sweep_config: Dict,
        method: str = "bayes",
        project: Optional[str] = None
    ):
        """
        Start a hyperparameter sweep.
        
        Args:
            sweep_config: W&B sweep configuration
            method: Sweep method (bayes, grid, random)
            project: Project name (defaults to self.project_name)
        """
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=project or self.project_name,
            entity=self.entity
        )
        
        logger.info(f"Started sweep: {sweep_id}")
        return sweep_id


def create_sweep_config() -> Dict:
    """
    Create a W&B sweep configuration for hyperparameter tuning.
    
    Returns:
        Dictionary containing sweep configuration
    """
    return {
        'method': 'bayes',
        'metric': {
            'name': 'eval/mean_reward',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform',
                'min': 1e-5,
                'max': 1e-3
            },
            'gamma': {
                'distribution': 'uniform',
                'min': 0.95,
                'max': 0.999
            },
            'tau': {
                'distribution': 'uniform',
                'min': 0.001,
                'max': 0.05
            },
            'alpha': {
                'distribution': 'uniform',
                'min': 0.05,
                'max': 0.5
            },
            'batch_size': {
                'distribution': 'categorical',
                'values': [128, 256, 512]
            },
            'buffer_size': {
                'distribution': 'categorical',
                'values': [100000, 500000, 1000000]
            },
            'hidden_dim': {
                'distribution': 'categorical',
                'values': [128, 256, 512]
            }
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 1000,
            'eta': 3
        }
    }


def login():
    """Helper function to login to W&B."""
    print("=" * 60)
    print("Weights & Biases (W&B) Login")
    print("=" * 60)
    print("\n1. Create account at: https://wandb.ai")
    print("2. Get your API key from: https://wandb.ai/settings")
    print("3. Run: wandb login")
    print("=" * 60)


if __name__ == "__main__":
    # Demo usage (without actual W&B login)
    print("=" * 60)
    print("H2Z W&B Integration Demo")
    print("=" * 60)
    
    # Note: This requires wandb to be installed and logged in
    print("\nTo use W&B tracking:")
    print("1. Install: pip install wandb")
    print("2. Login: wandb login")
    print("3. Initialize tracker in your training script")
    
    # Example sweep configuration
    sweep_config = create_sweep_config()
    print("\nExample Sweep Configuration:")
    print(json.dumps(sweep_config, indent=2))
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)

