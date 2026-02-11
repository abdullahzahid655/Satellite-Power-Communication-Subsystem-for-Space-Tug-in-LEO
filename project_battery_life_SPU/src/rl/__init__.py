"""
Reinforcement Learning Module for H2Z Battery Life Optimization

This package contains:
- environments/: Gymnasium environments for battery optimization
- agents/: RL agents (SAC, PPO, etc.)
- baselines/: Traditional charging strategies for comparison

Author: H2Z Development Team
"""

from src.rl.environments.h2z_battery_env import H2ZBatteryLifeEnv, EnvironmentConfig
from src.rl.agents.sac_agent import SACAgent, SACConfig
from src.rl.baselines.rule_based import (
    SimpleRuleBasedCharging,
    ConstantCurrentCharging,
    ConstantCurrentConstantVoltage,
    TemperatureAwareCharging,
    AdaptiveCharging
)

__all__ = [
    'H2ZBatteryLifeEnv',
    'EnvironmentConfig',
    'SACAgent',
    'SACConfig',
    'SimpleRuleBasedCharging',
    'ConstantCurrentCharging',
    'ConstantCurrentConstantVoltage',
    'TemperatureAwareCharging',
    'AdaptiveCharging'
]

