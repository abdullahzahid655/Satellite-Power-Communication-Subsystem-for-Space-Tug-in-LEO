"""
Baseline Charging Strategies for Battery Life Optimization

Implements traditional rule-based charging strategies for comparison:
1. Simple Rule-Based Charging (current baseline)
2. Constant Current (CC) Charging
3. Constant Current-Constant Voltage (CC-CV) Charging
4. Temperature-Aware Rule-Based Charging

These baselines are used to evaluate the improvement achieved by the RL agent.

Author: H2Z Development Team
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ChargingConfig:
    """Configuration for charging strategies."""
    # Battery specs
    capacity_wh: float = 163.22
    voltage_nominal: float = 28.0
    capacity_ah: float = 5.83
    C_rate_1C: float = 5.83
    
    # Limits
    soc_min: float = 0.15
    soc_max: float = 0.95
    temp_min_K: float = 173.15
    temp_max_K: float = 293.15
    voltage_min_V: float = 21.0
    voltage_max_V: float = 44.0
    
    # Charging limits
    charge_current_max_A: float = 20.0
    discharge_current_max_A: float = 20.0
    trickle_current_A: float = 0.5


class BaseChargingStrategy(ABC):
    """Base class for charging strategies."""
    
    def __init__(self, config: ChargingConfig = None):
        self.config = config or ChargingConfig()
        self.metrics = self._init_metrics()
    
    @abstractmethod
    def get_action(self, state: Dict[str, float]) -> np.ndarray:
        """Get charging action based on current state."""
        pass
    
    def _init_metrics(self) -> Dict[str, float]:
        """Initialize tracking metrics."""
        return {
            'charge_cycles': 0,
            'total_charge_energy_wh': 0.0,
            'total_discharge_energy_wh': 0.0,
            'fast_charge_events': 0,
            'deep_discharge_events': 0,
            'thermal_violations': 0,
            'safety_violations': 0,
            'avg_charge_efficiency': []
        }
    
    def reset_metrics(self):
        """Reset tracking metrics."""
        self.metrics = self._init_metrics()
    
    def _validate_state(self, state: Dict[str, float]) -> Tuple[bool, str]:
        """Validate battery is within safe limits."""
        soc = state.get('battery_soc', 0.8)
        temp = state.get('battery_temperature', 210.18)
        voltage = state.get('battery_voltage', 33.6)
        
        if soc < self.config.soc_min or soc > self.config.soc_max:
            return False, f"SOC {soc:.2f} out of range"
        if temp < self.config.temp_min_K or temp > self.config.temp_max_K:
            return False, f"Temperature {temp:.1f}K out of range"
        if voltage < self.config.voltage_min_V or voltage > self.config.voltage_max_V:
            return False, f"Voltage {voltage:.1f}V out of range"
        
        return True, "OK"
    
    def _record_metrics(
        self,
        state: Dict[str, float],
        action: np.ndarray,
        power_flow: float
    ):
        """Record metrics from charging cycle."""
        # Fast charging (>1.5C)
        c_rate = abs(action[0]) / self.config.C_rate_1C
        if c_rate > 1.5:
            self.metrics['fast_charge_events'] += 1
        
        # Deep discharge (>80% DoD)
        dod = 1.0 - state.get('battery_soc', 0.8)
        if dod > 0.8:
            self.metrics['deep_discharge_events'] += 1
        
        # Thermal violations
        temp = state.get('battery_temperature', 210.18)
        if temp > 283 or temp < 253:
            self.metrics['thermal_violations'] += 1
        
        # Safety violations
        is_valid, _ = self._validate_state(state)
        if not is_valid:
            self.metrics['safety_violations'] += 1
        
        # Energy tracking
        if power_flow > 0:
            self.metrics['total_charge_energy_wh'] += power_flow
            self.metrics['charge_cycles'] += power_flow / self.config.capacity_wh
        else:
            self.metrics['total_discharge_energy_wh'] += abs(power_flow)
        
        # Efficiency
        if power_flow > 0:
            efficiency = 0.93 - 0.01 * (temp - 210) / 100
            self.metrics['avg_charge_efficiency'].append(efficiency)


class SimpleRuleBasedCharging(BaseChargingStrategy):
    """
    Simple rule-based charging strategy.
    
    Rules:
    1. Charge at constant 0.3C in sunlight if SOC < 90%
    2. Discharge to meet demand in eclipse
    3. No thermal management
    
    This represents a typical baseline satellite charging system.
    """
    
    def __init__(self, config: ChargingConfig = None):
        super().__init__(config)
        self.charge_current = 1.75  # ~0.3C (1.75A)
        self.target_soc = 0.90
        logger.info("SimpleRuleBasedCharging initialized (baseline)")
    
    def get_action(self, state: Dict[str, float]) -> np.ndarray:
        """
        Get charging action.
        
        Action: [charge_current, discharge_limit, voltage_setpoint, heater_power, mppt_target]
        """
        soc = state.get('battery_soc', 0.8)
        solar_power = state.get('solar_power_available', 0)
        eclipse_remaining = state.get('eclipse_duration_remaining', 0)
        temperature = state.get('battery_temperature', 210.18)
        
        # Initialize action
        action = np.zeros(5)
        
        if eclipse_remaining > 0:
            # In eclipse - discharge to meet demand
            action[0] = 0.0  # No charging
            action[1] = min(self.config.discharge_current_max_A, 15.0)
            action[2] = 28.0  # Minimum voltage
            action[3] = 0.0  # No heating
            action[4] = 0.97  # MPPT target
        else:
            # In sunlight - charge if SOC below target
            if soc < self.target_soc:
                # Charge at fixed rate
                action[0] = self.charge_current
                action[1] = 5.0
                action[2] = 42.0  # Full charge voltage
                action[3] = 0.0  # No thermal management
                action[4] = 0.97
            else:
                # Trickle charge
                action[0] = self.config.trickle_current_A
                action[1] = 5.0
                action[2] = 33.6
                action[3] = 0.0
                action[4] = 0.97
        
        # Record metrics
        power_flow = action[0] * state.get('battery_voltage', 33.6)
        self._record_metrics(state, action, power_flow)
        
        return action


class ConstantCurrentCharging(BaseChargingStrategy):
    """
    Constant Current (CC) Charging Strategy.
    
    Charges at fixed current rate regardless of SOC or temperature.
    Simple but less efficient than adaptive methods.
    """
    
    def __init__(self, config: ChargingConfig = None, c_rate: float = 0.5):
        super().__init__(config)
        self.c_rate = c_rate
        self.charge_current = c_rate * self.config.C_rate_1C
        logger.info(f"ConstantCurrentCharging initialized (C-rate={c_rate})")
    
    def get_action(self, state: Dict[str, float]) -> np.ndarray:
        """Get charging action with constant current."""
        soc = state.get('battery_soc', 0.8)
        eclipse_remaining = state.get('eclipse_duration_remaining', 0)
        
        action = np.zeros(5)
        
        if eclipse_remaining > 0:
            # Discharge mode
            action[0] = 0.0
            action[1] = self.config.discharge_current_max_A
            action[2] = 28.0
            action[3] = 0.0
            action[4] = 0.97
        else:
            # Constant current charging
            if soc < 0.95:
                action[0] = self.charge_current
                action[1] = 5.0
                action[2] = 42.0
                action[3] = 0.0
                action[4] = 0.97
            else:
                # Float charge
                action[0] = self.config.trickle_current_A
                action[1] = 5.0
                action[2] = 33.6
                action[3] = 0.0
                action[4] = 0.97
        
        power_flow = action[0] * state.get('battery_voltage', 33.6)
        self._record_metrics(state, action, power_flow)
        
        return action


class ConstantCurrentConstantVoltage(BaseChargingStrategy):
    """
    Constant Current - Constant Voltage (CC-CV) Charging Strategy.
    
    Two-phase charging:
    1. CC Phase: Charge at constant current until voltage reaches target
    2. CV Phase: Maintain constant voltage, current tapers off
    
    This is a common smart charging method.
    """
    
    def __init__(self, config: ChargingConfig = None):
        super().__init__(config)
        self.cc_current = 2.92  # 0.5C (2.92A)
        self.cv_voltage = 42.0  # Full charge voltage for 7S Li-Ion
        
        # Phase tracking
        self.in_cv_phase = False
        logger.info("CC-CV Charging initialized")
    
    def get_action(self, state: Dict[str, float]) -> np.ndarray:
        """Get CC-CV charging action."""
        soc = state.get('battery_soc', 0.8)
        voltage = state.get('battery_voltage', 33.6)
        eclipse_remaining = state.get('eclipse_duration_remaining', 0)
        
        action = np.zeros(5)
        
        if eclipse_remaining > 0:
            # Discharge mode
            action[0] = 0.0
            action[1] = self.config.discharge_current_max_A
            action[2] = 28.0
            action[3] = 0.0
            action[4] = 0.97
            self.in_cv_phase = False
        else:
            # Determine charging phase
            if not self.in_cv_phase:
                # CC phase
                if voltage >= self.cv_voltage - 0.5:
                    self.in_cv_phase = True
                
                if soc < 0.95:
                    action[0] = self.cc_current
                else:
                    action[0] = self.config.trickle_current_A
            else:
                # CV phase
                action[0] = 0.5  # Tapered current
                action[2] = self.cv_voltage
            
            action[1] = 5.0
            action[2] = self.cv_voltage if self.in_cv_phase else 42.0
            action[3] = 0.0
            action[4] = 0.97
        
        power_flow = action[0] * state.get('battery_voltage', 33.6)
        self._record_metrics(state, action, power_flow)
        
        return action


class TemperatureAwareCharging(BaseChargingStrategy):
    """
    Temperature-Aware Rule-Based Charging Strategy.
    
    Adjusts charging based on battery temperature:
    - Reduces charge rate at low temperatures (prevents lithium plating)
    - Reduces charge rate at high temperatures (prevents degradation)
    - No thermal management, just rate adjustment
    """
    
    def __init__(self, config: ChargingConfig = None):
        super().__init__(config)
        
        # Temperature thresholds (K)
        self.temp_optimal_min = 253.0   # -20°C
        self.temp_optimal_max = 283.0    # +10°C
        self.temp_cold = 243.0           # -30°C
        self.temp_hot = 293.0            # +20°C
        
        logger.info("TemperatureAwareCharging initialized")
    
    def get_action(self, state: Dict[str, float]) -> np.ndarray:
        """Get temperature-aware charging action."""
        soc = state.get('battery_soc', 0.8)
        temp = state.get('battery_temperature', 210.18)
        eclipse_remaining = state.get('eclipse_duration_remaining', 0)
        solar_power = state.get('solar_power_available', 0)
        
        action = np.zeros(5)
        
        if eclipse_remaining > 0:
            # Discharge mode
            action[0] = 0.0
            action[1] = self.config.discharge_current_max_A
            action[2] = 28.0
            action[3] = 0.0
            action[4] = 0.97
        else:
            # Calculate charge rate based on temperature
            if temp < self.temp_cold:
                # Very cold - minimal charging to prevent plating
                charge_rate = 0.1
                action[3] = 5.0  # Enable heater
            elif temp < self.temp_optimal_min:
                # Cold - reduced charging
                charge_rate = 0.3
                action[3] = 3.0
            elif temp > self.temp_hot:
                # Hot - reduced charging
                charge_rate = 0.3
                action[3] = 0.0
            elif temp > self.temp_optimal_max:
                # Warm - moderate charging
                charge_rate = 0.5
                action[3] = 0.0
            else:
                # Optimal temperature - full charging
                charge_rate = 0.7
                action[3] = 0.0
            
            # Apply charge rate
            charge_current = charge_rate * self.config.C_rate_1C
            
            if soc < 0.90:
                action[0] = charge_current
            else:
                action[0] = self.config.trickle_current_A
            
            action[1] = 5.0
            action[2] = 42.0
            action[4] = 0.97
        
        power_flow = action[0] * state.get('battery_voltage', 33.6)
        self._record_metrics(state, action, power_flow)
        
        return action


class AdaptiveCharging(BaseChargingStrategy):
    """
    Advanced Adaptive Charging Strategy.
    
    Combines temperature awareness with SOC-based adjustments:
    - Temperature-adjusted charge rates
    - SOC-based charge limits
    - Heater control for thermal management
    """
    
    def __init__(self, config: ChargingConfig = None):
        super().__init__(config)
        
        # Temperature thresholds
        self.temp_critical_low = 193.0   # -80°C
        self.temp_low = 233.0            # -40°C
        self.temp_optimal = 263.0        # -10°C
        self.temp_high = 283.0            # +10°C
        
        # SOC thresholds
        self.soc_target = 0.85
        self.soc_float = 0.95
        
        logger.info("AdaptiveCharging initialized")
    
    def get_action(self, state: Dict[str, float]) -> np.ndarray:
        """Get adaptive charging action."""
        soc = state.get('battery_soc', 0.8)
        temp = state.get('battery_temperature', 210.18)
        eclipse_remaining = state.get('eclipse_duration_remaining', 0)
        soh = state.get('battery_soh', 1.0)
        
        action = np.zeros(5)
        
        if eclipse_remaining > 0:
            # Discharge - reduce demand based on SOH
            discharge_factor = max(0.5, soh)
            action[0] = 0.0
            action[1] = self.config.discharge_current_max_A * discharge_factor
            action[2] = 28.0
            action[3] = 0.0
            action[4] = 0.97
        else:
            # Calculate charge parameters based on temperature
            if temp < self.temp_critical_low:
                # Critical - minimal operation
                charge_rate = 0.05
                heater_power = 15.0
            elif temp < self.temp_low:
                # Cold - slow charge with heating
                charge_rate = 0.2
                heater_power = 15.0
            elif temp < self.temp_optimal:
                # Moderate - normal charge
                charge_rate = 0.5
                heater_power = 5.0
            elif temp < self.temp_high:
                # Optimal - fast charge
                charge_rate = 0.7
                heater_power = 0.0
            else:
                # Hot - slow charge
                charge_rate = 0.3
                heater_power = 0.0
            
            # Adjust for SOH (older batteries need gentler treatment)
            if soh < 0.9:
                charge_rate *= 0.8
            
            # Set charge current
            charge_current = charge_rate * self.config.C_rate_1C
            
            # Determine voltage setpoint
            if soc < self.soc_target:
                # Bulk charging
                voltage_setpoint = 42.0
            elif soc < self.soc_float:
                # Absorption
                voltage_setpoint = 41.0
            else:
                # Float
                voltage_setpoint = 33.6
                charge_current = self.config.trickle_current_A
            
            action[0] = charge_current
            action[1] = 5.0
            action[2] = voltage_setpoint
            action[3] = heater_power
            action[4] = 0.97
        
        power_flow = action[0] * state.get('battery_voltage', 33.6)
        self._record_metrics(state, action, power_flow)
        
        return action


def run_baseline_simulation(
    env,
    strategy: BaseChargingStrategy,
    num_episodes: int = 10
) -> Dict[str, Any]:
    """
    Run simulation with a baseline strategy.
    
    Args:
        env: Gymnasium environment
        strategy: Charging strategy
        num_episodes: Number of episodes to run
        
    Returns:
        Dictionary with simulation results
    """
    results = {
        'strategy': strategy.__class__.__name__,
        'episodes': [],
        'summary': {}
    }
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        state = info.get('state_dict', {})
        strategy.reset_metrics()
        
        episode_data = {
            'reward': 0.0,
            'steps': 0,
            'final_soh': 1.0,
            'final_soc': 0.8,
            'violations': 0,
            'energy_wh': 0.0
        }
        
        while True:
            action = strategy.get_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = info.get('state_dict', {})
            
            episode_data['reward'] += reward
            episode_data['steps'] += 1
            episode_data['final_soh'] = info.get('battery_soh', 1.0)
            episode_data['final_soc'] = info.get('battery_soc', 0.8)
            episode_data['violations'] += info.get('safety_violations', 0)
            episode_data['energy_wh'] += info.get('total_energy_wh', 0.0)
            
            if terminated or truncated:
                break
            
            obs = next_obs
            state = next_state
        
        # Add episode results
        episode_data.update(strategy.metrics)
        results['episodes'].append(episode_data)
    
    # Calculate summary
    rewards = [e['reward'] for e in results['episodes']]
    final_sohs = [e['final_soh'] for e in results['episodes']]
    violations = [e['violations'] for e in results['episodes']]
    
    results['summary'] = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_final_soh': np.mean(final_sohs),
        'mean_violations': np.mean(violations),
        'total_charge_events': sum(e['fast_charge_events'] for e in results['episodes']),
        'total_discharge_events': sum(e['deep_discharge_events'] for e in results['episodes']),
        'total_thermal_violations': sum(e['thermal_violations'] for e in results['episodes']),
        'total_safety_violations': sum(e['safety_violations'] for e in results['episodes'])
    }
    
    return results


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Baseline Charging Strategies Demo")
    logger.info("=" * 60)
    
    # Import environment
    import sys
    sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
    from src.rl.environments.h2z_battery_env import H2ZBatteryLifeEnv, EnvironmentConfig
    
    # Create environment
    config = EnvironmentConfig()
    config.simulation_minutes_per_step = 60  # 1 hour steps for faster simulation
    
    env = H2ZBatteryLifeEnv(config)
    
    # Test each strategy
    strategies = [
        SimpleRuleBasedCharging(),
        ConstantCurrentCharging(c_rate=0.3),
        ConstantCurrentCharging(c_rate=0.5),
        ConstantCurrentConstantVoltage(),
        TemperatureAwareCharging(),
        AdaptiveCharging()
    ]
    
    for strategy in strategies:
        logger.info(f"\n--- Testing {strategy.__class__.__name__} ---")
        results = run_baseline_simulation(env, strategy, num_episodes=3)
        
        logger.info(f"  Mean Reward: {results['summary']['mean_reward']:.2f}")
        logger.info(f"  Mean Final SOH: {results['summary']['mean_final_soh']*100:.2f}%")
        logger.info(f"  Mean Violations: {results['summary']['mean_violations']:.2f}")
        logger.info(f"  Fast Charge Events: {results['summary']['total_charge_events']}")
        logger.info(f"  Deep Discharge Events: {results['summary']['total_discharge_events']}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Baseline demo completed!")

