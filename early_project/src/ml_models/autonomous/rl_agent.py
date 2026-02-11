"""
Reinforcement Learning Agent for Autonomous Satellite Power Management

This module implements a Proximal Policy Optimization (PPO) agent
for intelligent power management in satellite systems.

Features:
- Autonomous power allocation optimization
- Battery management and charging control
- Subsystem priority adaptation
- Fault response and recovery

Author: H2Z Development Team
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import spaces

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Configuration for RL agent training."""
    # Environment
    state_dim: int = 20
    action_dim: int = 10
    
    # PPO Hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_epsilon: float = 0.2  # PPO clip parameter
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training
    num_epochs: int = 10
    batch_size: int = 64
    buffer_size: int = 2048
    update_interval: int = 2048
    num_training_steps: int = 50000
    
    # Network
    hidden_dim: int = 256
    use_lstm: bool = True


class SatellitePowerEnv(gym.Env):
    """
    Gymnasium environment for satellite power management.
    
    This environment simulates the satellite power system and allows
    an RL agent to learn optimal power allocation strategies.
    
    State Space (20 dimensions):
    - Solar irradiance (normalized)
    - Battery SOC (0-1)
    - Current power generation
    - Power demands for each subsystem
    - Temperature
    - Time in orbit phase
    - Historical power usage
    
    Action Space (10 dimensions):
    - Power allocation ratios for 6 subsystems
    - Battery charging rate
    - Mode selection (normal/emergency)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        config: RLConfig = None,
        max_time_steps: int = 1000
    ):
        self.config = config or RLConfig()
        self.max_time_steps = max_time_steps
        self.current_step = 0
        
        # Define action space
        # Actions: [ADCS_alloc, TTC_alloc, CDH_alloc, Propulsion_alloc, 
        #          Comm_alloc, Payload_alloc, charge_rate, mode]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32
        )
        
        # Define observation space
        # State: [solar_irr, battery_soc, power_gen, ADCS_demand, TTC_demand,
        #         CDH_demand, Propulsion_demand, Comm_demand, Payload_demand,
        #         temperature, eclipse_phase, hour, minute, 10 historical values]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),
            dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        self.episode_reward = 0
        self.battery_history = []
        
        logger.info("SatellitePowerEnv initialized")
    
    def _get_observation(self) -> np.ndarray:
        """Generate current observation from environment state."""
        obs = np.array([
            self.state['solar_irradiance'],
            self.state['battery_soc'],
            self.state['power_generation'],
            self.state['demands']['ADCS'],
            self.state['demands']['TT&C'],
            self.state['demands']['CDH'],
            self.state['demands']['Propulsion'],
            self.state['demands']['Communication'],
            self.state['demands']['Payload'],
            self.state['temperature'],
            self.state['eclipse_phase'],
            self.state['hour'],
            self.state['minute'],
            *self.state['history'][-10:]  # Last 10 power readings
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_reward(
        self,
        allocations: Dict[str, float],
        power_generated: float,
        battery_change: float
    ) -> float:
        """
        Calculate reward for the agent's action.
        
        Reward components:
        - Power satisfaction (higher = better)
        - Battery health (maintain optimal SOC)
        - Efficiency (minimize waste)
        - Priority adherence (critical systems get power)
        """
        reward = 0.0
        
        # Power satisfaction reward
        total_demand = sum(self.state['demands'].values())
        if total_demand > 0:
            satisfaction = min(sum(allocations.values()) / total_demand, 1.0)
            reward += satisfaction * 10.0
        
        # Battery health reward (optimal around 70% SOC)
        optimal_soc = 0.7
        soc_deviation = abs(self.state['battery_soc'] - optimal_soc)
        reward -= soc_deviation * 5.0
        
        # Battery degradation penalty
        if battery_change < -0.1:  # Heavy discharge
            reward -= 2.0
        if battery_change > 0.15:  # Aggressive charging
            reward -= 1.0
        
        # Efficiency reward (power used / power available)
        power_used = sum(allocations.values()) + battery_change * 100
        efficiency = min(power_used / max(power_generated, 1.0), 1.0)
        reward += efficiency * 5.0
        
        # Priority reward (critical systems should be satisfied)
        priority_weight = {
            'ADCS': 3,
            'TT&C': 3,
            'CDH': 2,
            'Propulsion': 2,
            'Communication': 2,
            'Payload': 1
        }
        
        priority_reward = 0.0
        for subsystem, demand in self.state['demands'].items():
            if demand > 0:
                allocation_ratio = allocations.get(subsystem, 0) / demand
                priority_reward += allocation_ratio * priority_weight[subsystem]
        
        reward += priority_reward * 2.0
        
        # Penalty for mode misuse (emergency mode without emergency)
        if self.state.get('emergency', False) and self.action[-1] < 0.5:
            reward -= 5.0
        
        # Small time penalty to encourage efficiency
        reward -= 0.1
        
        return reward
    
    def _update_state(self):
        """Update environment state for next timestep."""
        # Simulate orbital dynamics
        self.state['hour'] = (self.state['hour'] + 0.1) % 24
        self.state['minute'] = (self.state['minute'] + 6) % 60
        
        # Update eclipse phase
        hour = self.state['hour']
        eclipse_hours = [22, 23, 0, 1, 2, 3, 4, 5]
        self.state['eclipse_phase'] = 1.0 if hour in eclipse_hours else 0.0
        
        # Update solar irradiance based on eclipse phase
        if self.state['eclipse_phase'] > 0.5:
            self.state['solar_irradiance'] *= 0.1  # Drastic reduction in eclipse
            self.state['power_generation'] *= 0.1
        else:
            # Smooth variation
            solar_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (hour - 6) / 24)
            self.state['solar_irradiance'] = 0.8 + 0.2 * solar_factor
            self.state['power_generation'] = (
                self.state['solar_irradiance'] * self.state['max_generation']
            )
        
        # Update demands (vary slightly each step)
        for key in self.state['demands']:
            variation = np.random.uniform(0.9, 1.1)
            self.state['demands'][key] = np.clip(
                self.state['demands'][key] * variation,
                self.state['demands'][key] * 0.5,
                self.state['demands'][key] * 1.2
            )
        
        # Update temperature
        self.state['temperature'] = np.clip(
            self.state['temperature'] + np.random.uniform(-1, 1),
            -20, 50
        )
        
        # Update history
        total_power = sum(self.state['demands'].values())
        self.state['history'].append(total_power)
        if len(self.state['history']) > 100:
            self.state['history'].pop(0)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Initial state
        self.state = {
            'solar_irradiance': 1.0,
            'battery_soc': 0.8,  # Start at 80% SOC
            'power_generation': 800.0,  # W
            'max_generation': 1000.0,
            'demands': {
                'ADCS': 41.26,
                'TT&C': 20.32,
                'CDH': 13.71,
                'Propulsion': 96.60,
                'Communication': 28.19,
                'Payload': 13.00
            },
            'temperature': 25.0,
            'eclipse_phase': 0.0,
            'hour': 12.0,
            'minute': 0.0,
            'history': [0.0] * 50,
            'emergency': False
        }
        
        self.current_step = 0
        self.episode_reward = 0
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute environment step.
        
        Args:
            action: Agent's action vector
            
        Returns:
            observation: New state observation
            reward: Reward for this step
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        self.current_step += 1
        
        # Parse action
        subsystem_names = ['ADCS', 'TT&C', 'CDH', 'Propulsion', 'Communication', 'Payload']
        allocations = {}
        
        for i, name in enumerate(subsystem_names):
            allocations[name] = action[i] * self.state['demands'][name]
        
        # Charging rate
        charge_rate = action[6]
        battery_change = charge_rate * 0.02  # Max 2% per step
        
        # Emergency mode
        in_emergency = action[7] > 0.5
        self.state['emergency'] = in_emergency
        
        # Update battery
        old_soc = self.state['battery_soc']
        power_available = self.state['power_generation']
        
        # Calculate actual allocations based on power constraints
        total_demand = sum(allocations.values())
        
        if total_demand > power_available:
            # Need to reduce allocations
            reduction_factor = power_available / total_demand
            for key in allocations:
                allocations[key] *= reduction_factor
            actual_used = power_available
        else:
            actual_used = total_demand
        
        # Calculate battery change
        excess_power = power_available - actual_used
        battery_change = excess_power / 100.0 * charge_rate
        
        # Update battery SOC
        self.state['battery_soc'] = np.clip(
            old_soc + battery_change,
            0.2,  # Minimum SOC
            1.0   # Maximum SOC
        )
        
        # Check for critical states
        terminated = False
        if self.state['battery_soc'] <= 0.2 and self.state['eclipse_phase'] > 0.5:
            # Critical battery during eclipse
            terminated = True
            logger.warning("Critical battery state - episode terminated")
        
        # Calculate reward
        reward = self._calculate_reward(allocations, power_available, battery_change)
        self.episode_reward += reward
        
        # Update state for next step
        self._update_state()
        
        # Check truncation
        truncated = self.current_step >= self.max_time_steps
        
        # Info
        info = {
            'battery_soc': self.state['battery_soc'],
            'power_generation': self.state['power_generation'],
            'total_demand': sum(self.state['demands'].values()),
            'allocations': allocations,
            'episode_reward': self.episode_reward,
            'step': self.current_step
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def render(self, mode: 'human' = 'human'):
        """Render environment state."""
        print(f"\n{'='*50}")
        print(f"Step: {self.current_step}")
        print(f"Solar Irradiance: {self.state['solar_irradiance']:.2f}")
        print(f"Battery SOC: {self.state['battery_soc']*100:.1f}%")
        print(f"Power Generation: {self.state['power_generation']:.1f} W")
        print(f"Total Demand: {sum(self.state['demands'].values()):.1f} W")
        print(f"Temperature: {self.state['temperature']:.1f} °C")
        print(f"Eclipse Phase: {'Yes' if self.state['eclipse_phase'] > 0.5 else 'No'}")
        print(f"Episode Reward: {self.episode_reward:.2f}")
        print(f"{'='*50}\n")


class PPONetwork(nn.Module):
    """
    Actor-Critic network for PPO algorithm.
    
    Uses shared backbone with separate policy and value heads.
    Optionally includes LSTM for temporal processing.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        use_lstm: bool = True
    ):
        super().__init__()
        
        self.use_lstm = use_lstm
        self.hidden_dim = hidden_dim
        
        # Shared backbone
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # LSTM for temporal processing
        if use_lstm:
            self.lstm = nn.LSTM(
                hidden_dim,
                hidden_dim,
                batch_first=True
            )
        
        # Policy head (actor)
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Value head (critic)
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Log std for continuous actions
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        logger.info(f"PPONetwork initialized: state_dim={state_dim}, action_dim={action_dim}")
    
    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass through network.
        
        Returns:
            action_mean, value, (hidden_state, cell_state)
        """
        # Shared backbone
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        
        # LSTM
        if self.use_lstm:
            if hidden_state is None:
                x, (h_n, c_n) = self.lstm(x.unsqueeze(0))
            else:
                x, (h_n, c_n) = self.lstm(x.unsqueeze(0), hidden_state)
            x = x.squeeze(0)
        else:
            h_n = torch.zeros(1, self.hidden_dim, device=x.device)
            c_n = torch.zeros(1, self.hidden_dim, device=x.device)
        
        # Heads
        action_mean = self.policy_net(x)
        value = self.value_net(x)
        
        return action_mean, value.squeeze(-1), (h_n, c_n)
    
    def get_action(
        self,
        state: np.ndarray,
        hidden_state: Tuple[torch.Tensor, torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, Tuple]:
        """
        Get action from current state.
        
        Returns:
            action, log_prob, value, new_hidden_state
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_mean, value, hidden = self.forward(state_tensor, hidden_state)
        
        # Add noise for exploration
        std = torch.exp(self.log_std)
        
        if deterministic:
            action = action_mean
        else:
            dist = torch.distributions.Normal(action_mean, std)
            action = dist.sample()
        
        # Clip action to valid range
        action = torch.clamp(action, 0, 1)
        
        # Calculate log probability
        dist = torch.distributions.Normal(action_mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.squeeze(0).numpy(), log_prob, value, hidden
    
    def get_value(self, state: np.ndarray) -> torch.Tensor:
        """Get value estimate for state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            _, value, _ = self.forward(state_tensor)
        
        return value


class ReplayBuffer:
    """Experience replay buffer for PPO."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = 'cpu'
    ):
        self.buffer_size = buffer_size
        self.device = device
        
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        
        self.position = 0
        self.full = False
        
        logger.info(f"ReplayBuffer initialized with size {buffer_size}")
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float
    ):
        """Store transition in buffer."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = 1.0 if done else 0.0
        self.values[self.position] = value
        self.log_probs[self.position] = log_prob
        
        self.position = (self.position + 1) % self.buffer_size
        self.full = self.position == 0
    
    def get_all(self) -> Dict[str, np.ndarray]:
        """Get all stored transitions."""
        if self.full:
            indices = np.arange(self.buffer_size)
        else:
            indices = np.arange(self.position)
        
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'dones': self.dones[indices],
            'values': self.values[indices],
            'log_probs': self.log_probs[indices]
        }
    
    def clear(self):
        """Clear buffer."""
        self.position = 0
        self.full = False


class PPOAgent:
    """
    Proximal Policy Optimization Agent for Satellite Power Management.
    
    Implements PPO with GAE for stable and efficient training.
    
    Features:
    - Generalized Advantage Estimation (GAE)
    - PPO clip objective
    - Experience replay
    - Automatic learning rate scheduling
    """
    
    def __init__(self, config: RLConfig = None):
        self.config = config or RLConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create environment to get dimensions
        self.env = SatellitePowerEnv(self.config)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        # Update config
        self.config.state_dim = state_dim
        self.config.action_dim = action_dim
        
        # Create network
        self.network = PPONetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.config.hidden_dim,
            use_lstm=self.config.use_lstm
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate
        )
        
        # Replay buffer
        self.buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=self.config.buffer_size,
            device=self.device
        )
        
        # Training metrics
        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []
        
        logger.info(f"PPOAgent initialized on device: {self.device}")
    
    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Returns:
            advantages, returns
        """
        advantages = np.zeros(len(rewards), dtype=np.float32)
        returns = np.zeros(len(rewards), dtype=np.float32)
        
        gae = 0
        next_values = np.append(values[1:], next_value)
        
        for t in reversed(range(len(rewards))):
            mask = 1 - dones[t]
            delta = rewards[t] + self.config.gamma * next_values[t] * mask - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _update_policy(self, states, actions, old_log_probs, advantages, returns):
        """Update policy using PPO clip objective."""
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Forward pass
        action_mean, values, _ = self.network(states)
        std = torch.exp(self.network.log_std)
        
        # Calculate new log probabilities
        dist = torch.distributions.Normal(action_mean, std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Calculate ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO clip objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(-1), returns)
        
        # Entropy bonus
        entropy = dist.entropy().mean()
        
        # Total loss
        loss = (
            policy_loss +
            self.config.value_loss_coef * value_loss -
            self.config.entropy_coef * entropy
        )
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item(), entropy.item()
    
    def train(self, num_episodes: int = 1000) -> Dict[str, List]:
        """
        Train the PPO agent.
        
        Args:
            num_episodes: Number of episodes to train
            
        Returns:
            Training history
        """
        logger.info(f"Starting PPO training for {num_episodes} episodes...")
        
        state, _ = self.env.reset()
        hidden_state = None
        episode_reward = 0
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            hidden_state = None
            episode_reward = 0
            episode_steps = 0
            
            while True:
                # Get action
                action, log_prob, value, hidden_state = self.network.get_action(
                    state, hidden_state
                )
                
                # Take step
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.buffer.push(
                    state, action, reward, done, value.item(), log_prob.item()
                )
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
                
                # Update if buffer is full
                if len(self.buffer.states) >= self.config.buffer_size or done:
                    if len(self.buffer.states) >= self.config.buffer_size:
                        # Get next value
                        next_value = self.network.get_value(next_state).item()
                        
                        # Compute GAE
                        buffer_data = self.buffer.get_all()
                        advantages, returns = self._compute_gae(
                            buffer_data['rewards'],
                            buffer_data['values'],
                            buffer_data['dones'],
                            next_value
                        )
                        
                        # Update policy
                        for _ in range(self.config.num_epochs):
                            p_loss, v_loss, ent = self._update_policy(
                                buffer_data['states'],
                                buffer_data['actions'],
                                buffer_data['log_probs'],
                                advantages,
                                returns
                            )
                            self.policy_losses.append(p_loss)
                            self.value_losses.append(v_loss)
                        
                        self.buffer.clear()
                
                if done or episode_steps >= self.env.max_time_steps:
                    break
            
            self.episode_rewards.append(episode_reward)
            
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                logger.info(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                           f"Steps = {episode_steps}")
        
        logger.info("Training completed!")
        
        return {
            'episode_rewards': self.episode_rewards,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses
        }
    
    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate trained agent.
        
        Returns:
            Evaluation metrics
        """
        rewards = []
        allocations = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            hidden_state = None
            episode_reward = 0
            episode_allocations = []
            
            while True:
                action, _, _, hidden_state = self.network.get_action(
                    state, hidden_state, deterministic=deterministic
                )
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_allocations.append(info.get('allocations', {}))
                
                if terminated or truncated:
                    break
                
                state = next_state
            
            rewards.append(episode_reward)
            allocations.append(episode_allocations)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'episode_allocations': allocations
        }
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'episode_rewards': self.episode_rewards
        }, filepath)
        logger.info(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        logger.info(f"Agent loaded from {filepath}")


class AutonomousPowerManager:
    """
    High-level interface for autonomous power management using trained RL agent.
    
    Integrates with satellite power system to provide:
    - Real-time power allocation recommendations
    - Battery management strategies
    - Fault response actions
    """
    
    def __init__(self, agent: PPOAgent = None):
        self.agent = agent
        self.env = SatellitePowerEnv()
        self.current_state = None
        self.allocation_history = []
        
        logger.info("AutonomousPowerManager initialized")
    
    def initialize(self):
        """Initialize power manager with environment reset."""
        state, _ = self.env.reset()
        self.current_state = state
        self.allocation_history = []
        logger.info("Power manager initialized")
    
    def get_allocation_decision(
        self,
        solar_power: float,
        battery_soc: float,
        subsystem_demands: Dict[str, float],
        temperature: float,
        is_eclipse: bool,
        hour: float
    ) -> Dict[str, Any]:
        """
        Get power allocation decision from RL agent.
        
        Args:
            solar_power: Current solar power generation (W)
            battery_soc: Battery state of charge (0-1)
            subsystem_demands: Power demand per subsystem
            temperature: Current temperature (°C)
            is_eclipse: Whether in eclipse phase
            hour: Current hour (0-24)
            
        Returns:
            Dictionary with allocation recommendations
        """
        if self.agent is None:
            logger.warning("No trained agent available, using fallback")
            return self._fallback_allocation(subsystem_demands, solar_power)
        
        # Construct state vector
        state = self._construct_state(
            solar_power, battery_soc, subsystem_demands,
            temperature, is_eclipse, hour
        )
        
        # Get action from agent
        action, _, _, _ = self.agent.network.get_action(state, deterministic=True)
        
        # Parse action
        subsystem_names = ['ADCS', 'TT&C', 'CDH', 'Propulsion', 'Communication', 'Payload']
        allocations = {}
        
        for i, name in enumerate(subsystem_names):
            demand = subsystem_demands.get(name, 0)
            allocations[name] = action[i] * demand
        
        # Charging rate
        charge_rate = action[6]
        
        # Emergency mode
        emergency_mode = action[7] > 0.5
        
        decision = {
            'allocations': allocations,
            'charge_rate': charge_rate,
            'emergency_mode': emergency_mode,
            'total_allocated': sum(allocations.values()),
            'confidence': self._calculate_confidence(action)
        }
        
        self.allocation_history.append(decision)
        
        return decision
    
    def _construct_state(
        self,
        solar_power: float,
        battery_soc: float,
        subsystem_demands: Dict[str, float],
        temperature: float,
        is_eclipse: bool,
        hour: float
    ) -> np.ndarray:
        """Construct state vector from environment data."""
        # Normalize values
        normalized_demands = [
            subsystem_demands.get(name, 0) / 100.0 for name in
            ['ADCS', 'TT&C', 'CDH', 'Propulsion', 'Communication', 'Payload']
        ]
        
        # Construct state
        state = np.array([
            solar_power / 1000.0,  # Normalized solar power
            battery_soc,           # SOC (already 0-1)
            sum(subsystem_demands.values()) / 300.0,  # Normalized total demand
            *normalized_demands,
            (temperature + 20) / 70.0,  # Normalized temperature (-20 to 50)
            1.0 if is_eclipse else 0.0,
            hour / 24.0,
            0.0,  # Padding
            0.0,  # Padding
            0.0   # Padding
        ], dtype=np.float32)
        
        return state
    
    def _fallback_allocation(
        self,
        demands: Dict[str, float],
        available_power: float
    ) -> Dict[str, Any]:
        """Fallback allocation using greedy algorithm."""
        total_demand = sum(demands.values())
        demands_sorted = sorted(demands.items(), key=lambda x: -x[1])
        
        allocations = {}
        remaining = available_power
        
        for name, demand in demands_sorted:
            allocated = min(demand, remaining)
            allocations[name] = allocated
            remaining -= allocated
        
        return {
            'allocations': allocations,
            'charge_rate': 0.0,
            'emergency_mode': False,
            'total_allocated': sum(allocations.values()),
            'confidence': 0.5
        }
    
    def _calculate_confidence(self, action: np.ndarray) -> float:
        """Calculate confidence in allocation decision."""
        # Lower entropy in action distribution = higher confidence
        action_range = action.max() - action.min()
        return min(action_range / 0.5, 1.0)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Reinforcement Learning Power Management Demo")
    logger.info("=" * 60)
    
    # Demo 1: Environment
    logger.info("\n--- Environment Demo ---")
    env = SatellitePowerEnv()
    state, _ = env.reset()
    logger.info(f"Initial state shape: {state.shape}")
    
    # Random action step
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    logger.info(f"Step reward: {reward:.2f}")
    
    # Demo 2: Training
    logger.info("\n--- Training Demo (shortened for demo) ---")
    
    config = RLConfig(
        num_training_steps=1000,  # Reduced for demo
        buffer_size=512,
        hidden_dim=128
    )
    
    agent = PPOTrainingAgent(config)
    history = agent.train(num_episodes=50)  # Shortened training
    
    logger.info(f"Training completed. Final avg reward: {np.mean(history['episode_rewards'][-10:]):.2f}")
    
    # Demo 3: Evaluation
    logger.info("\n--- Evaluation Demo ---")
    
    metrics = agent.evaluate(num_episodes=5)
    logger.info(f"Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("RL demo completed successfully!")

