#!/usr/bin/env python3
"""
H2Z Battery Life Optimization Evaluation Script

Evaluates trained RL agent against baseline strategies and generates
comprehensive performance metrics and visualizations.

Usage:
    python evaluate_battery_agent.py --model models/sac_battery/final.pt
    python evaluate_battery_agent.py --compare_baselines

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
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.battery_degradation import BatteryDegradationModel
from src.rl.environments.h2z_battery_env import H2ZBatteryLifeEnv, EnvironmentConfig
from src.rl.agents.sac_agent import SACAgent, SACConfig
from src.rl.baselines.rule_based import (
    SimpleRuleBasedCharging,
    ConstantCurrentCharging,
    ConstantCurrentConstantVoltage,
    TemperatureAwareCharging,
    AdaptiveCharging,
    run_baseline_simulation
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Comprehensive evaluation metrics for battery optimization."""
    
    def __init__(self, name: str):
        self.name = name
        self.episodes = []
        self.reset()
    
    def reset(self):
        """Reset metrics."""
        self.episodes = []
    
    def add_episode(self, info: Dict, reward: float):
        """Add episode data."""
        self.episodes.append({
            'reward': reward,
            'final_soh': info.get('battery_soh', 1.0),
            'final_soc': info.get('battery_soc', 0.8),
            'final_temperature': info.get('battery_temperature', 210.18),
            'steps': info.get('step', 0),
            'safety_violations': info.get('safety_violations', 0),
            'power_deficit_events': info.get('power_deficit_events', 0),
            'lithium_plating_events': info.get('lithium_plating_events', 0),
            'total_energy_wh': info.get('total_energy_wh', 0.0),
            'reward_components': info.get('reward_components', {}),
            'penalties': info.get('penalties', {})
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics."""
        if not self.episodes:
            return {}
        
        rewards = [e['reward'] for e in self.episodes]
        sohs = [e['final_soh'] for e in self.episodes]
        
        return {
            'name': self.name,
            'n_episodes': len(self.episodes),
            'reward': {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards)
            },
            'final_soh': {
                'mean': np.mean(sohs),
                'std': np.std(sohs),
                'min': np.min(sohs),
                'max': np.max(sohs)
            },
            'safety_violations': {
                'total': sum(e['safety_violations'] for e in self.episodes),
                'mean': np.mean([e['safety_violations'] for e in self.episodes])
            },
            'power_deficit_events': {
                'total': sum(e['power_deficit_events'] for e in self.episodes),
                'mean': np.mean([e['power_deficit_events'] for e in self.episodes])
            },
            'lithium_plating_events': {
                'total': sum(e['lithium_plating_events'] for e in self.episodes),
                'mean': np.mean([e['lithium_plating_events'] for e in self.episodes])
            },
            'total_energy_wh': sum(e['total_energy_wh'] for e in self.episodes),
            'avg_steps': np.mean([e['steps'] for e in self.episodes])
        }


def evaluate_sac_agent(
    env,
    agent: SACAgent,
    n_episodes: int = 10
) -> EvaluationMetrics:
    """Evaluate SAC agent."""
    logger.info(f"Evaluating SAC agent ({n_episodes} episodes)...")
    
    metrics = EvaluationMetrics("SAC Agent")
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        metrics.add_episode(info, episode_reward)
        logger.info(f"  Episode {episode+1}: Reward={episode_reward:.2f}, "
                   f"SOH={info.get('battery_soh', 0)*100:.1f}%, "
                   f"Violations={info.get('safety_violations', 0)}")
    
    return metrics


def evaluate_baseline(
    env,
    strategy,
    n_episodes: int = 10
) -> EvaluationMetrics:
    """Evaluate baseline strategy."""
    logger.info(f"Evaluating {strategy.__class__.__name__} ({n_episodes} episodes)...")
    
    results = run_baseline_simulation(env, strategy, n_episodes)
    
    metrics = EvaluationMetrics(strategy.__class__.__name__)
    
    for episode_data in results['episodes']:
        info = {
            'battery_soh': episode_data.get('final_soh', 1.0),
            'battery_soc': episode_data.get('final_soc', 0.8),
            'safety_violations': episode_data.get('safety_violations', 0),
            'power_deficit_events': episode_data.get('power_deficit_events', 0),
            'lithium_plating_events': episode_data.get('lithium_plating_events', 0),
            'total_energy_wh': episode_data.get('energy_wh', 0.0),
            'reward_components': {},
            'penalties': {}
        }
        metrics.add_episode(info, episode_data.get('reward', 0))
    
    return metrics


def compare_strategies(
    env,
    agent: SACAgent = None,
    n_episodes: int = 10
) -> Dict[str, EvaluationMetrics]:
    """Compare all strategies including SAC agent."""
    results = {}
    
    # Define baseline strategies
    baselines = [
        SimpleRuleBasedCharging(),
        ConstantCurrentCharging(c_rate=0.3),
        ConstantCurrentCharging(c_rate=0.5),
        ConstantCurrentConstantVoltage(),
        TemperatureAwareCharging(),
        AdaptiveCharging()
    ]
    
    # Evaluate baselines
    for strategy in baselines:
        metrics = evaluate_baseline(env, strategy, n_episodes)
        results[strategy.__class__.__name__] = metrics
    
    # Evaluate SAC agent if provided
    if agent is not None:
        sac_metrics = evaluate_sac_agent(env, agent, n_episodes)
        results["SAC Agent"] = sac_metrics
    
    return results


def print_comparison_table(results: Dict[str, EvaluationMetrics]):
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print("BATTERY LIFE OPTIMIZATION - PERFORMANCE COMPARISON")
    print("=" * 100)
    print(f"{'Strategy':<30} {'Reward':>12} {'Final SOH':>12} {'Violations':>12} {'Plating':>10} {'Energy (kWh)':>14}")
    print("-" * 100)
    
    for name, metrics in results.items():
        summary = metrics.get_summary()
        reward = f"{summary['reward']['mean']:.1f} ± {summary['reward']['std']:.1f}"
        soh = f"{summary['final_soh']['mean']*100:.1f} ± {summary['final_soh']['std']*100:.1f}%"
        violations = f"{summary['safety_violations']['mean']:.1f}"
        plating = f"{summary['lithium_plating_events']['mean']:.1f}"
        energy = f"{summary['total_energy_wh']/1000:.1f}"
        
        print(f"{name:<30} {reward:>12} {soh:>12} {violations:>12} {plating:>10} {energy:>14}")
    
    print("=" * 100)


def calculate_improvement(
    sac_results: EvaluationMetrics,
    baseline_results: EvaluationMetrics
) -> Dict[str, float]:
    """Calculate improvement of SAC over baseline."""
    sac_summary = sac_results.get_summary()
    baseline_summary = baseline_results.get_summary()
    
    # SOH improvement
    soh_improvement = (
        (sac_summary['final_soh']['mean'] - baseline_summary['final_soh']['mean']) /
        baseline_summary['final_soh']['mean'] * 100
    )
    
    # Violation reduction
    violation_reduction = (
        (baseline_summary['safety_violations']['mean'] - 
         sac_summary['safety_violations']['mean']) /
        max(1, baseline_summary['safety_violations']['mean']) * 100
    )
    
    # Reward improvement
    reward_improvement = (
        (sac_summary['reward']['mean'] - baseline_summary['reward']['mean']) /
        abs(baseline_summary['reward']['mean']) * 100
    )
    
    return {
        'soh_improvement_percent': soh_improvement,
        'violation_reduction_percent': violation_reduction,
        'reward_improvement_percent': reward_improvement
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate H2Z Battery Life Agent')
    
    # Model loading
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--no_baselines', action='store_true',
                        help='Skip baseline comparison')
    
    # Evaluation settings
    parser.add_argument('--n_episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--simulation_minutes', type=float, default=12.0,
                        help='Minutes per simulation step')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Max steps per episode')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--save_json', action='store_true',
                        help='Save results to JSON')
    
    # Hardware
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cuda, cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info("=" * 70)
    logger.info("H2Z Battery Life Optimization - Evaluation")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Episodes: {args.n_episodes}")
    
    # Create environment
    env_config = EnvironmentConfig(
        simulation_minutes_per_step=args.simulation_minutes,
        max_steps_per_episode=args.max_steps
    )
    
    env = H2ZBatteryLifeEnv(env_config)
    
    # Initialize results
    results = {}
    sac_agent = None
    
    # Load SAC agent if model provided
    if args.model and os.path.exists(args.model):
        logger.info(f"\nLoading SAC agent from: {args.model}")
        
        sac_config = SACConfig(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            seed=args.seed
        )
        
        sac_agent = SACAgent(sac_config, device)
        sac_agent.load(args.model)
        
        logger.info("SAC agent loaded successfully!")
    
    # Compare with baselines
    if not args.no_baselines:
        logger.info("\n" + "-" * 70)
        logger.info("Running Baseline Comparison")
        logger.info("-" * 70)
        
        results = compare_strategies(env, sac_agent, args.n_episodes)
        
        # Print comparison table
        print_comparison_table(results)
        
        # Calculate improvements vs best baseline
        if sac_agent and "SAC Agent" in results:
            sac_metrics = results["SAC Agent"]
            
            # Find best baseline by SOH
            best_baseline = None
            best_soh = -1
            
            for name, metrics in results.items():
                if name != "SAC Agent":
                    if metrics.get_summary()['final_soh']['mean'] > best_soh:
                        best_soh = metrics.get_summary()['final_soh']['mean']
                        best_baseline = metrics
            
            if best_baseline:
                improvement = calculate_improvement(sac_metrics, best_baseline)
                
                logger.info("\n" + "=" * 70)
                logger.info("SAC AGENT IMPROVEMENT vs BASELINE")
                logger.info("=" * 70)
                logger.info(f"SOH Improvement: +{improvement['soh_improvement_percent']:.1f}%")
                logger.info(f"Violation Reduction: {improvement['violation_reduction_percent']:.1f}%")
                logger.info(f"Reward Improvement: {improvement['reward_improvement_percent']:.1f}%")
    
    # Save results
    if args.save_json:
        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'n_episodes': args.n_episodes,
                'simulation_minutes': args.simulation_minutes,
                'max_steps': args.max_steps
            },
            'metrics': {}
        }
        
        for name, metrics in results.items():
            results_dict['metrics'][name] = metrics.get_summary()
        
        output_file = output_dir / 'evaluation_results.json'
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"\nResults saved to: {output_file}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Evaluation Complete!")
    logger.info("=" * 70)
    
    return results


if __name__ == "__main__":
    main()

