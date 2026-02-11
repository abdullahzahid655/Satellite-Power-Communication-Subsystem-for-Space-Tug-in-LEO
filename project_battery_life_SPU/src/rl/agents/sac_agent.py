"""
Soft Actor-Critic (SAC) Agent for H2Z Battery Life Optimization

Implementation of the SAC algorithm for continuous action space RL,
optimized for satellite battery management.

Features:
- Automatic entropy tuning
- Target entropy scheduling
- Double Q-networks
- Experience replay
- TensorBoard logging

Author: H2Z Development Team
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any
from pathlib import Path
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SACConfig:
    """Configuration for SAC agent."""
    # Environment
    state_dim: int = 20
    action_dim: int = 5
    action_range: Tuple[float, float] = (-1.0, 1.0)  # For tanh squashing
    
    # Network
    hidden_dim: int = 400
    hidden_layers: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    
    # Training
    buffer_size: int = 500_000
    batch_size: int = 256
    gamma: float = 0.995  # High discount for long-term planning
    tau: float = 0.005    # Soft target update
    initial_alpha: float = 0.2
    target_entropy: float = None  # Auto-set to -action_dim
    
    # Training settings
    update_actor_every: int = 1
    gradient_steps: int = 1
    train_freq: int = 1
    max_grad_norm: float = 0.5
    
    # Logging
    log_dir: str = "runs/sac_battery"
    save_dir: str = "models/sac_battery"
    save_freq: int = 10000
    eval_freq: int = 5000
    n_eval_episodes: int = 10
    seed: int = 42  # Random seed


class ReplayBuffer:
    """Experience replay buffer for SAC."""
    
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int, device: str = 'cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.device = device
        
        # Allocate buffers
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        self.position = 0
        self.size = 0
        
        logger.info(f"ReplayBuffer initialized: size={buffer_size}, device={device}")
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float
    ):
        """Store transition in buffer."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch of transitions."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'states': torch.FloatTensor(self.states[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(self.device),
            'dones': torch.FloatTensor(self.dones[indices]).to(self.device)
        }
    
    def clear(self):
        """Clear buffer."""
        self.position = 0
        self.size = 0
    
    def __len__(self):
        return self.size


class GaussianActor(nn.Module):
    """
    Gaussian policy network for SAC.
    
    Outputs mean and log_std for continuous actions.
    Uses reparameterization trick for backpropagation.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 400,
        hidden_layers: int = 2,
        action_range: Tuple[float, float] = (-1.0, 1.0)
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = action_range
        
        # Build network
        layers = []
        in_dim = state_dim
        
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        
        # Output layers
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # Log std bounds
        self.log_std_min = -20
        self.log_std_max = 2
        
        logger.info(f"GaussianActor initialized: state_dim={state_dim}, action_dim={action_dim}")
    
    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        with_log_prob: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor network.
        
        Args:
            state: State tensor
            deterministic: Whether to use deterministic action
            with_log_prob: Whether to return log probability
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action (if requested)
        """
        # Get hidden features
        features = self.network(state)
        
        # Get mean and log_std
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Clamp log_std
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        # Sample action using reparameterization trick
        if deterministic:
            action = torch.tanh(mean)
        else:
            noise = torch.randn_like(mean)
            action_raw = mean + std * noise
            action = torch.tanh(action_raw)
        
        # Calculate log probability
        if with_log_prob:
            # log_prob = log π(a|s) = log p(u) - log |det(da/du)|
            # where u = atanh(a) and a = tanh(u)
            log_prob = self._get_log_prob(action, mean, log_std, std, action_raw)
            return action, log_prob
        else:
            return action, None
    
    def _get_log_prob(
        self,
        action: torch.Tensor,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        std: torch.Tensor,
        action_raw: torch.Tensor
    ) -> torch.Tensor:
        """Calculate log probability of action."""
        # Gaussian log probability
        gaussian_log_prob = -0.5 * (((action_raw - mean) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        gaussian_log_prob = gaussian_log_prob.sum(dim=-1)
        
        # Jacobian of tanh transformation
        jacobian = 2 * np.log(2) - action_raw - F.softplus(-2 * action_raw)
        jacobian = jacobian.sum(dim=-1)
        
        # Final log probability
        log_prob = gaussian_log_prob - jacobian
        
        return log_prob
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action from state (numpy interface)."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, _ = self.forward(state_tensor, deterministic=deterministic)
            return action.squeeze(0).numpy()


class QNetwork(nn.Module):
    """Double Q-network for SAC critic."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 400,
        hidden_layers: int = 2
    ):
        super().__init__()
        
        # Q1 network
        self.q1_network = self._build_network(state_dim, action_dim, hidden_dim, hidden_layers)
        self.q1_head = nn.Linear(hidden_dim, 1)
        
        # Q2 network
        self.q2_network = self._build_network(state_dim, action_dim, hidden_dim, hidden_layers)
        self.q2_head = nn.Linear(hidden_dim, 1)
        
        logger.info(f"QNetwork initialized: state_dim={state_dim}, action_dim={action_dim}")
    
    def _build_network(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int
    ) -> nn.Sequential:
        """Build shared network layers."""
        layers = []
        in_dim = state_dim + action_dim
        
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through both Q-networks."""
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        
        # Q1
        q1_features = self.q1_network(x)
        q1 = self.q1_head(q1_features)
        
        # Q2
        q2_features = self.q2_network(x)
        q2 = self.q2_head(q2_features)
        
        return q1, q2


class SACAgent:
    """
    Soft Actor-Critic Agent for battery life optimization.
    
    Implements SAC with:
    - Automatic entropy tuning
    - Double Q-networks
    - Target networks
    - Experience replay
    
    Paper: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning"
    by Haarnoja et al. (2018)
    """
    
    def __init__(self, config: SACConfig = None, device: str = 'cpu'):
        self.config = config or SACConfig()
        self.device = device
        
        # Set random seeds
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        # Create networks
        self.actor = GaussianActor(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            hidden_layers=self.config.hidden_layers,
            action_range=self.config.action_range
        ).to(self.device)
        
        self.critic = QNetwork(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            hidden_layers=self.config.hidden_layers
        ).to(self.device)
        
        self.critic_target = QNetwork(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            hidden_layers=self.config.hidden_layers
        ).to(self.device)
        
        # Copy weights to target
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.config.actor_lr
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.config.critic_lr
        )
        
        # Entropy temperature (automatic tuning)
        if self.config.target_entropy is None:
            self.target_entropy = -self.config.action_dim
        else:
            self.target_entropy = self.config.target_entropy
        
        self.log_alpha = torch.tensor(np.log(self.config.initial_alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
        
        # Replay buffer
        self.buffer = ReplayBuffer(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            buffer_size=self.config.buffer_size,
            device=self.device
        )
        
        # Training metrics
        self.training_step = 0
        self.episode_rewards = []
        self.eval_rewards = []
        
        # Logging setup
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SACAgent initialized on device: {self.device}")
        logger.info(f"  Actor parameters: {sum(p.numel() for p in self.actor.parameters())}")
        logger.info(f"  Critic parameters: {sum(p.numel() for p in self.critic.parameters())}")
    
    @property
    def alpha(self) -> float:
        """Get current entropy temperature."""
        return self.log_alpha.exp().item()
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action from current policy."""
        action = self.actor.get_action(state, deterministic=deterministic)
        
        # Scale action to environment bounds
        action = self._scale_action(action)
        
        return action
    
    def _scale_action(self, action: torch.Tensor) -> np.ndarray:
        """Scale action from [-1, 1] to environment bounds."""
        low = np.array(self.config.action_range[0])
        high = np.array(self.config.action_range[1])
        
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        
        scaled = low + (action + 1) * (high - low) / 2
        return scaled
    
    def update(self) -> Dict[str, float]:
        """
        Perform one gradient update step.
        
        Returns:
            Dictionary of loss values
        """
        # Sample from buffer
        batch = self.buffer.sample(self.config.batch_size)
        
        # Update critic
        critic_loss = self._update_critic(batch)
        
        # Update actor and alpha
        actor_loss, alpha = self._update_actor(batch)
        
        # Update target networks
        self._update_target()
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha': alpha,
            'mean_Q': self._get_mean_Q(batch).item()
        }
    
    def _update_critic(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update critic networks."""
        with torch.no_grad():
            # Next actions and log probs from target actor
            next_action, next_log_prob = self.actor(
                batch['next_states'],
                with_log_prob=True
            )
            
            # Target Q values (using critic target)
            q1_target, q2_target = self.critic_target(batch['next_states'], next_action)
            q_target = torch.min(q1_target, q2_target).squeeze() - self.alpha * next_log_prob
            
            # TD target
            target = batch['rewards'] + (1 - batch['dones']) * self.config.gamma * q_target
        
        # Current Q values
        q1, q2 = self.critic(batch['states'], batch['actions'])
        q1 = q1.squeeze()
        q2 = q2.squeeze()
        
        # Critic loss (MSE)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """Update actor network and alpha."""
        # Get new actions and log probs
        action, log_prob = self.actor(batch['states'], with_log_prob=True)
        
        # Get Q values
        q1, q2 = self.critic(batch['states'], action)
        q = torch.min(q1, q2).squeeze()
        
        # Actor loss (maximize Q - alpha * log_prob)
        actor_loss = (self.alpha * log_prob - q).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()
        
        # Update alpha (temperature)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        return actor_loss.item(), self.alpha
    
    def _update_target(self):
        """Soft update target networks."""
        tau = self.config.tau
        
        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic.parameters()
        ):
            target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)
    
    def _get_mean_Q(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate mean Q value for logging."""
        with torch.no_grad():
            action, _ = self.actor(batch['states'], with_log_prob=False)
            q1, q2 = self.critic(batch['states'], action)
            return torch.min(q1, q2).mean()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float
    ):
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
    
    def train(
        self,
        env,
        total_timesteps: int,
        eval_env = None,
        callback = None
    ) -> Dict[str, List]:
        """
        Train the SAC agent.
        
        Args:
            env: Training environment
            total_timesteps: Total number of environment steps
            eval_env: Evaluation environment (optional)
            callback: Callback function (optional)
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting SAC training for {total_timesteps} timesteps...")
        
        # Initialize
        state, _ = env.reset()
        episode_reward = 0
        episode_count = 0
        
        # Training history
        history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'critic_losses': [],
            'actor_losses': [],
            'alphas': [],
            'mean_Qs': [],
            'eval_rewards': []
        }
        
        # Timestep loop
        for timestep in range(total_timesteps):
            # Select action
            if timestep < self.config.buffer_size:
                # Random exploration
                action = np.random.uniform(
                    low=self.config.action_range[0],
                    high=self.config.action_range[1],
                    size=self.config.action_dim
                )
                action = self._scale_action(action)
            else:
                # Policy action
                action = self.select_action(state, deterministic=False)
                # Add exploration noise
                action = action + np.random.normal(0, 0.1, size=self.config.action_dim)
                action = np.clip(action, 0, 20)  # Clip to valid range
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.store_transition(state, action, reward, next_state, float(done))
            
            # Update state
            state = next_state
            episode_reward += reward
            
            # Update if buffer is ready
            if timestep >= self.config.buffer_size:
                for _ in range(self.config.gradient_steps):
                    update_info = self.update()
                    
                    # Log losses
                    history['critic_losses'].append(update_info['critic_loss'])
                    history['actor_losses'].append(update_info['actor_loss'])
                    history['alphas'].append(update_info['alpha'])
                    history['mean_Qs'].append(update_info['mean_Q'])
            
            # Episode end
            if done:
                # Log episode
                self.episode_rewards.append(episode_reward)
                history['episode_rewards'].append(episode_reward)
                history['episode_lengths'].append(info.get('step', timestep))
                
                logger.info(
                    f"Episode {episode_count}: "
                    f"Reward={episode_reward:.2f}, "
                    f"SOH={info.get('battery_soh', 0)*100:.1f}%, "
                    f"Violations={info.get('safety_violations', 0)}"
                )
                
                # Reset
                state, _ = env.reset()
                episode_reward = 0
                episode_count += 1
            
            # Evaluation
            if eval_env is not None and timestep > 0 and timestep % self.config.eval_freq == 0:
                eval_results = self.evaluate(eval_env, n_episodes=self.config.n_eval_episodes)
                history['eval_rewards'].append(eval_results['mean_reward'])
                logger.info(f"Eval at timestep {timestep}: Mean Reward={eval_results['mean_reward']:.2f}")
            
            # Save checkpoint
            if timestep > 0 and timestep % self.config.save_freq == 0:
                self.save(timestep)
            
            # Callback
            if callback is not None:
                callback(locals(), globals())
        
        logger.info("Training completed!")
        return history
    
    def evaluate(self, env, n_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate the trained agent.
        
        Args:
            env: Environment for evaluation
            n_episodes: Number of episodes to run
            
        Returns:
            Dictionary with evaluation metrics
        """
        rewards = []
        sohs = []
        violations = []
        energy_processed = []
        
        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            while True:
                action = self.select_action(state, deterministic=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                state = next_state
                
                if terminated or truncated:
                    break
            
            rewards.append(episode_reward)
            sohs.append(info.get('battery_soh', 0))
            violations.append(info.get('safety_violations', 0))
            energy_processed.append(info.get('total_energy_wh', 0))
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'mean_final_soh': np.mean(sohs),
            'mean_violations': np.mean(violations),
            'total_energy_wh': np.sum(energy_processed),
            'n_episodes': n_episodes
        }
    
    def save(self, timestep: int = 0):
        """Save agent checkpoint."""
        checkpoint = {
            'timestep': timestep,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'config': self.config,
            'training_metrics': {
                'episode_rewards': self.episode_rewards
            }
        }
        
        filepath = self.save_dir / f"checkpoint_{timestep}.pt"
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load(self, filepath: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        
        self.training_step = checkpoint['timestep']
        logger.info(f"Agent loaded from {filepath}")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("SAC Agent Demo")
    logger.info("=" * 60)
    
    # Create agent
    config = SACConfig(
        state_dim=20,
        action_dim=5,
        buffer_size=1000,
        batch_size=64,
        hidden_dim=128
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = SACAgent(config, device=device)
    
    # Test action selection
    state = np.random.randn(20)
    action = agent.select_action(state, deterministic=True)
    
    logger.info(f"State shape: {state.shape}")
    logger.info(f"Action shape: {action.shape}")
    logger.info(f"Action values: {action}")
    
    # Test update
    logger.info("\nTesting buffer push and update...")
    for _ in range(100):
        state = np.random.randn(20)
        action = np.random.uniform(0, 20, 5)
        reward = np.random.randn()
        next_state = np.random.randn(20)
        done = np.random.rand() < 0.1
        
        agent.store_transition(state, action, reward, next_state, done)
    
    if len(agent.buffer) >= config.batch_size:
        update_info = agent.update()
        logger.info(f"Update successful: critic_loss={update_info['critic_loss']:.4f}, "
                   f"actor_loss={update_info['actor_loss']:.4f}, alpha={update_info['alpha']:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("SAC Agent demo completed!")

