"""
H2Z MLflow Experiment Tracker

Integration with MLflow for tracking RL training experiments,
model versioning, and experiment comparison.

Features:
- Automatic metrics logging
- Model artifact versioning
- Hyperparameter tracking
- Experiment comparison UI

Author: H2Z Development Team
"""

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Any
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class H2ZExperimentTracker:
    """MLflow experiment tracker for H2Z RL training."""
    
    def __init__(
        self,
        experiment_name: str = "H2Z_Battery_Optimization",
        tracking_uri: str = "http://localhost:5000",
        artifact_location: Optional[str] = None
    ):
        """
        Initialize MLflow experiment tracker.
        
        Args:
            experiment_name: Name of MLflow experiment
            tracking_uri: MLflow tracking server URI
            artifact_location: Custom artifact storage location
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
        # Set up MLflow
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    artifact_location=artifact_location
                )
                logger.info(f"Created new experiment: {experiment_name}")
            else:
                self.experiment_id = self.experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"Could not connect to MLflow server: {e}")
            logger.info("Using local MLflow tracking")
            mlflow.set_tracking_uri("file://./mlruns")
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
        
        mlflow.set_experiment(experiment_name)
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run
            nested: Whether this is a nested run
        """
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.active_run = mlflow.start_run(
            run_name=run_name,
            tags=tags,
            nested=nested
        )
        
        logger.info(f"Started MLflow run: {run_name}")
        return self.active_run
    
    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run."""
        if self.active_run:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run with status: {status}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} parameters")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log training metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number
        """
        mlflow.log_metrics(metrics, step=step)
        logger.info(f"Logged {len(metrics)} metrics at step {step}")
    
    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str = "model"
    ):
        """Log a PyTorch model."""
        mlflow.pytorch.log_model(model, artifact_path=artifact_path)
        logger.info(f"Logged model to artifact path: {artifact_path}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact (file or directory)."""
        mlflow.log_artifact(local_path, artifact_path=artifact_path)
        logger.info(f"Logged artifact: {local_path}")
    
    def log_dict(self, dictionary: Dict, artifact_file: str):
        """Log a dictionary as a JSON artifact."""
        with open(artifact_file, 'w') as f:
            json.dump(dictionary, f, indent=2)
        mlflow.log_artifact(artifact_file)
        logger.info(f"Logged dictionary to: {artifact_file}")
    
    def set_tag(self, key: str, value: str):
        """Set a tag on the current run."""
        mlflow.set_tag(key, value)
    
    def log_env_params(self, env_config):
        """Log environment configuration."""
        params = {
            'orbit_altitude_km': getattr(env_config, 'orbit_altitude_km', 500.0),
            'battery_capacity_wh': getattr(env_config, 'battery_capacity_wh', 163.22),
            'mission_duration_years': getattr(env_config, 'mission_duration_years', 3.0),
            'max_steps_per_episode': getattr(env_config, 'max_steps_per_episode', 500),
            'simulation_minutes_per_step': getattr(env_config, 'simulation_minutes_per_step', 12.0)
        }
        self.log_params(params)
    
    def log_sac_params(self, sac_config):
        """Log SAC algorithm parameters."""
        params = {
            'learning_rate': getattr(sac_config, 'learning_rate', 3e-4),
            'gamma': getattr(sac_config, 'gamma', 0.99),
            'tau': getattr(sac_config, 'tau', 0.005),
            'alpha': getattr(sac_config, 'alpha', 0.2),
            'buffer_size': getattr(sac_config, 'buffer_size', 500_000),
            'batch_size': getattr(sac_config, 'batch_size', 256),
            'hidden_dim': getattr(sac_config, 'hidden_dim', 256)
        }
        self.log_params(params)
    
    def log_training_episode(
        self,
        episode: int,
        reward: float,
        episode_length: int,
        soh_start: float,
        soh_end: float,
        violations: int,
        energy_processed: float,
        epsilon: float = 0.0
    ):
        """Log metrics for a single training episode."""
        metrics = {
            'episode_reward': reward,
            'episode_length': episode_length,
            'soh_start': soh_start,
            'soh_end': soh_end,
            'soh_change': soh_end - soh_start,
            'safety_violations': violations,
            'energy_processed_wh': energy_processed,
            'exploration_epsilon': epsilon
        }
        self.log_metrics(metrics, step=episode)
    
    def log_evaluation(
        self,
        eval_name: str,
        mean_reward: float,
        std_reward: float,
        mean_soh: float,
        mean_violations: float,
        energy_throughput: float,
        baseline_comparison: Optional[Dict] = None
    ):
        """Log evaluation results."""
        metrics = {
            f'{eval_name}/mean_reward': mean_reward,
            f'{eval_name}/std_reward': std_reward,
            f'{eval_name}/mean_soh': mean_soh,
            f'{eval_name}/mean_violations': mean_violations,
            f'{eval_name}/energy_throughput': energy_throughput
        }
        self.log_metrics(metrics)
        
        if baseline_comparison:
            self.log_dict(baseline_comparison, f'{eval_name}_baselines.json')
    
    def save_checkpoint(
        self,
        agent,
        episode: int,
        checkpoint_path: str
    ):
        """Save a model checkpoint and log to MLflow."""
        # Save locally
        agent.save(checkpoint_path)
        
        # Log to MLflow
        self.log_artifact(checkpoint_path, f"checkpoints/episode_{episode}")
        
        logger.info(f"Saved checkpoint at episode {episode}")
    
    def get_run_history(self, run_id: str) -> Dict:
        """Get the history of a run."""
        run = mlflow.get_run(run_id)
        return {
            'params': run.data.params,
            'metrics': run.data.metrics,
            'tags': run.data.tags
        }
    
    def search_experiments(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 10
    ) -> pd.DataFrame:
        """Search for experiments."""
        search_results = mlflow.search_runs(
            filter_string=filter_string,
            max_results=max_results
        )
        return search_results


def setup_mlflow_ui():
    """Helper function to set up MLflow UI."""
    import subprocess
    import sys
    
    print("=" * 60)
    print("Starting MLflow UI...")
    print("=" * 60)
    print("\nTo view experiments:")
    print("  1. Open browser to: http://localhost:5000")
    print("  2. Or run: mlflow ui --port 5000")
    print("\nTo start tracking server:")
    print("  mlflow server --host 0.0.0.0 --port 5000")
    print("=" * 60)


if __name__ == "__main__":
    # Demo usage
    tracker = H2ZExperimentTracker()
    
    # Start a demo run
    tracker.start_run(run_name="demo_run")
    
    # Log some dummy parameters
    tracker.log_params({
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'buffer_size': 500000
    })
    
    # Log dummy metrics
    for episode in range(10):
        tracker.log_metrics({
            'episode_reward': -24150 + np.random.randn() * 100,
            'episode_length': 500,
            'soh_end': 99.9 - episode * 0.01
        }, step=episode)
    
    tracker.end_run()
    
    print("\n" + "=" * 60)
    print("MLflow tracking demo completed!")
    print("=" * 60)
    print("\nTo view results:")
    print("  1. Run: mlflow ui")
    print("  2. Open: http://localhost:5000")
    print("=" * 60)

