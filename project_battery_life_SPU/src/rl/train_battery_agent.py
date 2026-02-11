#!/usr/bin/env python3
"""
H2Z Battery Life Optimization Training Script

Trains the SAC agent for battery charge/discharge optimization
using the H2ZBatteryLifeEnv environment.

Usage:
    python train_battery_agent.py --total_timesteps 500000 --eval_freq 10000

Author: H2Z Development Team
"""

import numpy as np
import torch
import argparse
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.battery_degradation import BatteryDegradationModel
from src.rl.environments.h2z_battery_env import H2ZBatteryLifeEnv, EnvironmentConfig
from src.rl.agents.sac_agent import SACAgent, SACConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingCallbacks:
    """Callbacks for training monitoring and checkpointing."""
    
    def __init__(self, agent, env, eval_env=None, log_dir="logs"):
        self.agent = agent
        self.env = env
        self.eval_env = eval_env
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_reward = -np.inf
        self.episode_count = 0
        self.timestep_count = 0
    
    def __call__(self, locals_, globals_):
        """Callback called at each timestep."""
        timestep = locals_.get('timestep', 0)
        
        # Log episode rewards
        if 'episode_reward' in locals_.get('info', {}):
            self.episode_count += 1
            reward = locals_['info']['episode_reward']
            
            # Log to file
            log_entry = {
                'episode': self.episode_count,
                'timestep': timestep,
                'reward': reward,
                'battery_soh': locals_.get('info', {}).get('battery_soh', 0),
                'safety_violations': locals_.get('info', {}).get('safety_violations', 0)
            }
            
            with open(self.log_dir / 'training_log.jsonl', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            # Check for best model
            if reward > self.best_reward:
                self.best_reward = reward
                self.agent.save(timestep)
                logger.info(f"New best model saved! Reward: {reward:.2f}")
        
        return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train H2Z Battery Life Agent')
    
    # Training settings
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                        help='Total number of environment timesteps')
    parser.add_argument('--buffer_size', type=int, default=500000,
                        help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--gradient_steps', type=int, default=1,
                        help='Gradient steps per update')
    
    # Agent settings
    parser.add_argument('--hidden_dim', type=int, default=400,
                        help='Hidden layer dimension')
    parser.add_argument('--gamma', type=float, default=0.995,
                        help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update coefficient')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    
    # Environment settings
    parser.add_argument('--simulation_minutes', type=float, default=12.0,
                        help='Minutes per simulation step')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Max steps per episode')
    
    # Evaluation
    parser.add_argument('--eval_freq', type=int, default=10000,
                        help='Evaluation frequency')
    parser.add_argument('--n_eval_episodes', type=int, default=5,
                        help='Number of evaluation episodes')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Log directory')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Model save directory')
    parser.add_argument('--save_freq', type=int, default=50000,
                        help='Checkpoint save frequency')
    
    # Hardware
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cuda, cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info("=" * 70)
    logger.info("H2Z Battery Life Optimization - SAC Training")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Total Timesteps: {args.total_timesteps}")
    logger.info(f"Seed: {args.seed}")
    
    # Create environment config
    env_config = EnvironmentConfig(
        simulation_minutes_per_step=args.simulation_minutes,
        max_steps_per_episode=args.max_steps
    )
    
    # Create training and evaluation environments
    logger.info("\nCreating environments...")
    env = H2ZBatteryLifeEnv(env_config)
    eval_env = H2ZBatteryLifeEnv(env_config)
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    logger.info(f"State Dimension: {state_dim}")
    logger.info(f"Action Dimension: {action_dim}")
    
    # Create SAC config
    sac_config = SACConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.learning_rate,
        critic_lr=args.learning_rate,
        gradient_steps=args.gradient_steps,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        seed=args.seed
    )
    
    # Create agent
    agent = SACAgent(sac_config, device)
    
    # Create callbacks
    callbacks = TrainingCallbacks(agent, env, eval_env, args.log_dir)
    
    logger.info("\nStarting training...")
    logger.info("-" * 70)
    
    # Train agent
    history = agent.train(
        env=env,
        total_timesteps=args.total_timesteps,
        eval_env=eval_env,
        callback=callbacks
    )
    
    # Final evaluation
    logger.info("\n" + "=" * 70)
    logger.info("Final Evaluation")
    logger.info("=" * 70)
    
    eval_results = agent.evaluate(eval_env, n_episodes=10)
    
    logger.info(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    logger.info(f"Mean Final SOH: {eval_results['mean_final_soh']*100:.2f}%")
    logger.info(f"Mean Violations: {eval_results['mean_violations']:.2f}")
    logger.info(f"Total Energy Processed: {eval_results['total_energy_wh']:.2f} Wh")
    
    # Save final model
    agent.save(args.total_timesteps)
    
    # Save training history
    history_path = Path(args.log_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        # Convert to serializable format
        serializable_history = {
            'episode_rewards': [float(r) for r in history['episode_rewards']],
            'critic_losses': [float(l) for l in history['critic_losses']],
            'actor_losses': [float(l) for l in history['actor_losses']],
            'alphas': [float(a) for a in history['alphas']],
            'eval_rewards': [float(r) for r in history['eval_rewards']]
        }
        json.dump(serializable_history, f, indent=2)
    
    logger.info(f"\nTraining history saved to: {history_path}")
    logger.info("\n" + "=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)
    
    return eval_results


if __name__ == "__main__":
    main()

