"""
H2Z Space Tug Battery Life Optimization Environment

Gymnasium-compatible environment for RL-based battery charge/discharge optimization.

This environment simulates the complete satellite power system with:
- 20-dimensional state space (battery, orbital, power, mission, degradation)
- 5-dimensional continuous action space (charge current, voltage, heater, MPPT)
- Multi-objective reward function balancing lifespan, mission, efficiency, thermal
- Physics-based battery degradation model

Based on the comprehensive AI prompt specifications.

Author: H2Z Development Team
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any
from enum import Enum
import logging
from pathlib import Path
import random

import gymnasium as gym
from gymnasium import spaces

# Import battery degradation model
from src.core.battery_degradation import (
    BatteryDegradationModel,
    BatteryDegradationConfig,
    calculate_battery_efficiency
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MissionMode(Enum):
    """Mission mode enumeration."""
    STANDBY = 0
    OBSERVATION = 1
    DEBRIS_CAPTURE = 2


@dataclass
class EnvironmentConfig:
    """Configuration for the H2Z Battery Life Environment."""
    
    # ===== Satellite Specifications =====
    orbit_altitude_km: float = 500.0
    orbit_period_minutes: float = 98.0
    eclipse_duration_minutes: float = 36.26
    sunlight_duration_minutes: float = 61.74
    mission_duration_years: float = 3.0
    
    # ===== Battery Specifications =====
    battery_capacity_wh: float = 163.22
    battery_voltage_nominal: float = 28.0
    battery_capacity_ah: float = 5.83  # 163.22 / 28
    dod_max: float = 0.80
    
    # ===== Power System =====
    solar_array_power_bol_w: float = 851.61
    solar_array_area_m2: float = 2.733
    mppt_efficiency_base: float = 0.97
    bcr_efficiency: float = 0.91
    bdr_efficiency: float = 0.89
    pdu_efficiency: float = 0.98
    
    # ===== Subsystem Power Demands =====
    power_adcs_w: float = 41.26
    power_ttc_w: float = 20.32
    power_cdh_w: float = 13.71
    power_propulsion_w: float = 96.60
    power_comm_w: float = 28.19
    power_payload_w: float = 13.00
    
    # ===== Thermal =====
    battery_heater_power_w: float = 15.0
    thermal_dissipation_base_w: float = 10.0
    
    # ===== RL Settings =====
    max_steps_per_episode: int = 500  # ~1 day of simulation
    simulation_minutes_per_step: float = 12.0  # Each step = 12 minutes
    
    # ===== Reward Weights =====
    w_lifespan: float = 5.0
    w_mission: float = 3.0
    w_efficiency: float = 1.0
    w_thermal: float = 2.0
    p_degradation: float = 1.0
    p_safety: float = 10.0


class H2ZBatteryLifeEnv(gym.Env):
    """
    Gymnasium environment for H2Z Space Tug battery life optimization.
    
    State Space (20 dimensions):
    =============================
    Battery State (6):
    - battery_soc: State of Charge (0.0-1.0)
    - battery_voltage: Voltage (25-42V)
    - battery_current: Current (-20A to +20A)
    - battery_temperature: Temperature (173.15-273.15 K)
    - battery_internal_resistance: R_int (increases with age)
    - battery_soh: State of Health (0.0-1.0)
    
    Orbital/Environmental (5):
    - orbital_position: Position in orbit (0.0-1.0)
    - time_to_eclipse: Minutes to next eclipse
    - eclipse_duration_remaining: Minutes left in eclipse
    - solar_irradiance: Solar flux (0-1367 W/m²)
    - beta_angle: Sun angle (-83.573° to +83.573°)
    
    Power System (4):
    - solar_power_available: Available solar power (0-851.61 W)
    - total_power_demand: Total subsystem demand (0-300 W)
    - mppt_efficiency: MPPT efficiency (0.85-0.98)
    - thermal_dissipation: Heat generated (0-50 W)
    
    Mission Context (3):
    - mission_mode: 0=standby, 1=observation, 2=debris_capture
    - days_since_launch: Mission days (0-1095)
    - battery_cycles_completed: Cycle count (0-17520)
    
    Degradation State (2):
    - capacity_fade_rate: Estimated fade velocity
    - sei_layer_thickness_estimate: SEI growth estimate
    
    Action Space (5 dimensions):
    =============================
    - charge_current_setpoint: 0-20A (C-rate: 0-2C)
    - discharge_current_limit: 0-20A maximum allowed
    - voltage_setpoint: 28-42V (for CC-CV charging)
    - heater_power: 0-15W (active battery heating)
    - mppt_efficiency_target: 0.90-0.98 (MPPT optimization)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, config: EnvironmentConfig = None, render_mode: str = None):
        self.config = config or EnvironmentConfig()
        self.render_mode = render_mode
        
        # Initialize degradation model
        self.degradation_model = BatteryDegradationModel()
        
        # Calculate derived parameters
        self._calculate_derived_parameters()
        
        # Define action space (5 continuous actions)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 28.0, 0.0, 0.90], dtype=np.float32),
            high=np.array([20.0, 20.0, 42.0, 15.0, 0.98], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define observation space (20 continuous observations)
        obs_low = np.array([
            0.0,    # battery_soc
            21.0,   # battery_voltage
            -20.0,  # battery_current
            173.15, # battery_temperature
            0.01,   # battery_internal_resistance
            0.5,    # battery_soh
            0.0,    # orbital_position
            0.0,    # time_to_eclipse
            0.0,    # eclipse_duration_remaining
            0.0,    # solar_irradiance
            -83.573,# beta_angle
            0.0,    # solar_power_available
            0.0,    # total_power_demand
            0.85,   # mppt_efficiency
            0.0,    # thermal_dissipation
            0.0,    # mission_mode
            0.0,    # days_since_launch
            0.0,    # battery_cycles_completed
            0.0,    # capacity_fade_rate
            0.0     # sei_layer_thickness_estimate
        ], dtype=np.float32)
        
        obs_high = np.array([
            1.0,    # battery_soc
            44.0,   # battery_voltage
            20.0,   # battery_current
            293.15, # battery_temperature
            0.5,    # battery_internal_resistance
            1.0,    # battery_soh
            1.0,    # orbital_position
            61.74,  # time_to_eclipse
            36.26,  # eclipse_duration_remaining
            1367.0, # solar_irradiance
            83.573, # beta_angle
            851.61, # solar_power_available
            350.0,  # total_power_demand
            0.98,   # mppt_efficiency
            50.0,   # thermal_dissipation
            2.0,    # mission_mode
            1095.0, # days_since_launch
            17520.0,# battery_cycles_completed
            0.01,   # capacity_fade_rate
            100.0   # sei_layer_thickness_estimate (nm)
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )
        
        # State variables
        self.state = None
        self.current_step = 0
        self.episode_reward = 0
        self.done = False
        
        # Tracking metrics
        self.metrics = {
            'lifespan_reward': 0.0,
            'mission_reward': 0.0,
            'efficiency_reward': 0.0,
            'thermal_reward': 0.0,
            'degradation_penalty': 0.0,
            'safety_penalty': 0.0,
            'safety_violations': 0,
            'thermal_violations': 0,
            'power_deficit_events': 0,
            'lithium_plating_events': 0,
            'total_energy_processed_wh': 0.0,
            'charge_efficiency': []
        }
        
        logger.info("H2ZBatteryLifeEnv initialized")
        logger.info(f"  State dim: {self.observation_space.shape[0]}")
        logger.info(f"  Action dim: {self.action_space.shape[0]}")
    
    def _calculate_derived_parameters(self):
        """Calculate derived simulation parameters."""
        # Orbital fractions
        self.orbit_period_minutes = self.config.orbit_period_minutes
        self.eclipse_fraction = self.config.eclipse_duration_minutes / self.orbit_period_minutes
        self.sunlight_fraction = self.config.sunlight_duration_minutes / self.orbit_period_minutes
        
        # Power demands by mode
        self.power_demands = {
            MissionMode.STANDBY: {
                'ADCS': self.config.power_adcs_w * 0.8,
                'TT&C': self.config.power_ttc_w * 0.5,
                'CDH': self.config.power_cdh_w * 0.8,
                'Propulsion': 0.0,
                'Communication': self.config.power_comm_w * 0.5,
                'Payload': 0.0
            },
            MissionMode.OBSERVATION: {
                'ADCS': self.config.power_adcs_w,
                'TT&C': self.config.power_ttc_w,
                'CDH': self.config.power_cdh_w,
                'Propulsion': 0.0,
                'Communication': self.config.power_comm_w,
                'Payload': self.config.power_payload_w * 0.5
            },
            MissionMode.DEBRIS_CAPTURE: {
                'ADCS': self.config.power_adcs_w * 1.2,
                'TT&C': self.config.power_ttc_w * 1.2,
                'CDH': self.config.power_cdh_w,
                'Propulsion': self.config.power_propulsion_w,
                'Communication': self.config.power_comm_w * 1.2,
                'Payload': self.config.power_payload_w
            }
        }
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        obs = np.array([
            # Battery State (6)
            self.state['battery_soc'],
            self.state['battery_voltage'],
            self.state['battery_current'],
            self.state['battery_temperature'],
            self.state['battery_internal_resistance'],
            self.state['battery_soh'],
            
            # Orbital/Environmental (5)
            self.state['orbital_position'],
            self.state['time_to_eclipse'],
            self.state['eclipse_duration_remaining'],
            self.state['solar_irradiance'],
            self.state['beta_angle'],
            
            # Power System (4)
            self.state['solar_power_available'],
            self.state['total_power_demand'],
            self.state['mppt_efficiency'],
            self.state['thermal_dissipation'],
            
            # Mission Context (3)
            float(self.state['mission_mode']),
            self.state['days_since_launch'],
            self.state['battery_cycles_completed'],
            
            # Degradation State (2)
            self.state['capacity_fade_rate'],
            self.state['sei_layer_thickness']
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_reward(
        self,
        action: np.ndarray,
        degradation_state: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate multi-objective reward.
        
        R_total = w1 * R_lifespan 
                + w2 * R_mission_success 
                + w3 * R_energy_efficiency 
                + w4 * R_thermal_stability
                - p1 * P_degradation
                - p2 * P_safety_violation
        """
        reward_components = {}
        penalties = {}
        
        # ===== POSITIVE REWARDS =====
        
        # R_lifespan: Reward for maintaining battery health
        soh_change = self.state['battery_soh'] - self.state.get('prev_soh', self.state['battery_soh'])
        reward_components['lifespan'] = soh_change * 1000 * self.config.w_lifespan
        
        # R_mission_success: Ensure all critical systems get power
        critical_power_needed = (
            self.state['demands']['ADCS'] +
            self.state['demands']['TT&C'] +
            self.state['demands']['CDH']
        )
        
        if self.state['power_available'] >= critical_power_needed:
            reward_components['mission'] = 100 * self.config.w_mission
        else:
            power_deficit = critical_power_needed - self.state['power_available']
            reward_components['mission'] = -500 * self.config.w_mission
            self.metrics['power_deficit_events'] += 1
            penalties['power_deficit'] = 500
        
        # R_energy_efficiency: Efficient use of available power
        if self.state['power_available'] > 0:
            efficiency = min(self.state['power_used'] / self.state['power_available'], 1.0)
            reward_components['efficiency'] = efficiency * 10 * self.config.w_efficiency
        else:
            reward_components['efficiency'] = 0
        
        # R_thermal_stability: Penalty for temperature deviation from optimal
        T_optimal = 273.15  # 0°C
        T_battery = self.state['battery_temperature']
        thermal_penalty = -((T_battery - T_optimal) ** 2) / 100
        reward_components['thermal'] = thermal_penalty * self.config.w_thermal
        
        # ===== PENALTIES =====
        
        # P_degradation: Penalize fast charging, deep discharge, unsafe conditions
        
        # Fast charging penalty
        c_rate = abs(self.state['battery_current']) / self.degradation_model.C_rate_1C
        if c_rate > 1.5:
            penalties['fast_charge'] = 50 * c_rate
            self.metrics['degradation_penalty'] += 50 * c_rate
        
        # Deep discharge penalty
        dod = 1.0 - self.state['battery_soc']
        if dod > 0.8:
            penalties['deep_discharge'] = 100 * dod
            self.metrics['degradation_penalty'] += 100 * dod
        
        # Lithium plating risk
        if (self.state['battery_temperature'] < 253 and 
            self.state['battery_current'] > 0 and
            c_rate > 0.5):
            penalties['lithium_plating'] = 200
            self.metrics['lithium_plating_events'] += 1
            self.metrics['degradation_penalty'] += 200
        
        # Overheating
        if self.state['battery_temperature'] > 283:
            penalties['overheating'] = 150
            self.metrics['degradation_penalty'] += 150
            self.metrics['thermal_violations'] += 1
        
        # P_safety_violation: Critical safety limits
        
        # SOC limits
        if self.state['battery_soc'] < 0.15:
            penalties['low_soc'] = 1000
            self.metrics['safety_violations'] += 1
        elif self.state['battery_soc'] > 0.95:
            penalties['high_soc'] = 1000
            self.metrics['safety_violations'] += 1
        
        # Voltage limits
        if self.state['battery_voltage'] < 21.0:
            penalties['low_voltage'] = 500
            self.metrics['safety_violations'] += 1
        elif self.state['battery_voltage'] > 44.0:
            penalties['high_voltage'] = 500
            self.metrics['safety_violations'] += 1
        
        # Temperature limits
        if self.state['battery_temperature'] < 173.15:
            penalties['cold_temp'] = 300
            self.metrics['safety_violations'] += 1
        elif self.state['battery_temperature'] > 293.15:
            penalties['hot_temp'] = 300
            self.metrics['safety_violations'] += 1
        
        # Calculate total reward
        total_reward = (
            reward_components.get('lifespan', 0) +
            reward_components.get('mission', 0) +
            reward_components.get('efficiency', 0) +
            reward_components.get('thermal', 0) -
            sum(penalties.values())
        )
        
        # Store components
        self.metrics['lifespan_reward'] = reward_components.get('lifespan', 0)
        self.metrics['mission_reward'] = reward_components.get('mission', 0)
        self.metrics['efficiency_reward'] = reward_components.get('efficiency', 0)
        self.metrics['thermal_reward'] = reward_components.get('thermal', 0)
        self.metrics['safety_penalty'] = sum(penalties.values())
        
        return total_reward, reward_components, penalties
    
    def _update_battery_dynamics(self, action: np.ndarray) -> Dict[str, float]:
        """
        Update battery state based on action and power flow.
        
        Args:
            action: Agent's action vector
            
        Returns:
            Dictionary with updated state metrics
        """
        # Parse action
        charge_current_setpoint = np.clip(action[0], 0, 20)
        discharge_current_limit = np.clip(action[1], 0, 20)
        voltage_setpoint = np.clip(action[2], 28, 42)
        heater_power = np.clip(action[3], 0, 15)
        mppt_target = np.clip(action[4], 0.90, 0.98)
        
        # Calculate power available
        solar_power_raw = self.state['solar_irradiance'] * self.config.solar_array_area_m2 * 0.318
        solar_power_available = solar_power_raw * min(self.state['mppt_efficiency'], mppt_target)
        
        # Account for heater power (draws from solar)
        heater_draw = heater_power / (self.config.pdu_efficiency * self.config.bdr_efficiency)
        
        # Available for charging
        solar_for_charging = max(0, solar_power_available - heater_draw)
        
        # Total demand
        total_demand = self.state['total_power_demand']
        
        # Determine battery current
        if self.state['eclipse_duration_remaining'] > 0:
            # In eclipse - discharge to meet demand
            power_from_battery = min(total_demand, self.state['battery_soc'] * self.degradation_model.current_capacity_wh * 10)
            battery_current = -power_from_battery / self.state['battery_voltage']
            battery_current = max(-discharge_current_limit, min(0, battery_current))
        else:
            # In sunlight - charge if solar available
            if solar_for_charging > 0 and self.state['battery_soc'] < 0.95:
                # Calculate charge current based on available power
                charge_power = solar_for_charging * self.config.bcr_efficiency
                battery_current = charge_power / self.state['battery_voltage']
                battery_current = min(charge_current_setpoint, battery_current)
            else:
                battery_current = 0
        
        # Update voltage (simplified OCV model)
        soc = self.state['battery_soc']
        ocv = 21.0 + 21.0 * soc + 0.5 * np.sin(2 * np.pi * soc)  # OCV between 21-42V
        voltage_drop = battery_current * self.state['battery_internal_resistance']
        new_voltage = ocv - voltage_drop
        
        # Update temperature (thermal model)
        # Heat from internal resistance + heater
        I_squared_R = (battery_current ** 2) * self.state['battery_internal_resistance']
        heating_power = I_squared_R + heater_power
        
        # Cooling (radiation to space)
        T_amb = self.state['battery_temperature'] - 50  # Space is cold
        cooling = (self.state['battery_temperature'] - T_amb) * 0.5  # Simplified
        
        # Temperature change
        temp_change = (heating_power - cooling) * 0.1  # Thermal time constant
        new_temperature = np.clip(
            self.state['battery_temperature'] + temp_change,
            173.15,
            293.15
        )
        
        # Update SOC (coulomb counting)
        dt_hours = self.config.simulation_minutes_per_step / 60
        soc_change = (battery_current * dt_hours) / self.degradation_model.current_capacity_ah
        new_soc = np.clip(
            self.state['battery_soc'] + soc_change,
            self.config.dod_max,
            1.0
        )
        
        # Update degradation
        degradation_state = self.degradation_model.update(
            soc=new_soc,
            temperature_K=new_temperature,
            current_A=battery_current,
            time_delta_hours=dt_hours
        )
        
        # Track energy
        if battery_current > 0:
            energy_in = battery_current * new_voltage * dt_hours * self.config.bcr_efficiency
            self.metrics['total_energy_processed_wh'] += energy_in
        
        return {
            'solar_power_available': solar_power_available,
            'power_available': solar_power_available - heater_draw,
            'power_used': total_demand,
            'battery_current': battery_current,
            'battery_voltage': new_voltage,
            'battery_temperature': new_temperature,
            'heater_power': heater_power,
            'mppt_efficiency': mppt_target,
            **degradation_state
        }
    
    def _update_orbital_state(self) -> Dict[str, float]:
        """Update orbital position and conditions."""
        # Update orbital position
        position_increment = self.config.simulation_minutes_per_step / self.orbit_period_minutes
        new_position = (self.state['orbital_position'] + position_increment) % 1.0
        
        # Update time to eclipse
        eclipse_threshold = 1.0 - self.eclipse_fraction
        if new_position < eclipse_threshold:
            # In sunlight
            time_to_eclipse = (eclipse_threshold - new_position) * self.orbit_period_minutes
            eclipse_remaining = 0.0
        else:
            # In eclipse
            time_to_eclipse = 0.0
            eclipse_remaining = (new_position - eclipse_threshold) * self.orbit_period_minutes
        
        # Solar irradiance (0 during eclipse)
        if eclipse_remaining > 0:
            solar_irradiance = 0.0
        else:
            # Sinusoidal variation with orbit
            solar_factor = np.cos(2 * np.pi * (new_position - 0.25))
            solar_irradiance = max(0, 1367.0 * solar_factor)
        
        # Beta angle (simplified - can vary based on season)
        beta_angle = 45.0 * np.sin(2 * np.pi * self.state['days_since_launch'] / 365)
        
        return {
            'orbital_position': new_position,
            'time_to_eclipse': time_to_eclipse,
            'eclipse_duration_remaining': eclipse_remaining,
            'solar_irradiance': solar_irradiance,
            'beta_angle': beta_angle
        }
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Reset degradation model
        self.degradation_model.reset()
        
        # Initial state
        self.state = {
            # Battery (6)
            'battery_soc': 0.8,
            'battery_voltage': 33.6,  # ~80% SOC
            'battery_current': 0.0,
            'battery_temperature': 210.18,  # -63°C nominal
            'battery_internal_resistance': 0.05,
            'battery_soh': 1.0,
            'prev_soh': 1.0,
            
            # Orbital (5)
            'orbital_position': 0.0,
            'time_to_eclipse': self.config.sunlight_duration_minutes,
            'eclipse_duration_remaining': 0.0,
            'solar_irradiance': 1367.0,
            'beta_angle': 45.0,
            
            # Power (4)
            'solar_power_available': self.config.solar_array_power_bol_w,
            'total_power_demand': sum(self.power_demands[MissionMode.STANDBY].values()),
            'mppt_efficiency': self.config.mppt_efficiency_base,
            'thermal_dissipation': self.config.thermal_dissipation_base_w,
            
            # Mission (3)
            'mission_mode': MissionMode.STANDBY.value,
            'days_since_launch': 0.0,
            'battery_cycles_completed': 0.0,
            
            # Degradation (2)
            'capacity_fade_rate': 0.0,
            'sei_layer_thickness': 1.0,  # nm
            
            # Demands (dict)
            'demands': self.power_demands[MissionMode.STANDBY].copy()
        }
        
        # Reset metrics
        self.metrics = {
            'lifespan_reward': 0.0,
            'mission_reward': 0.0,
            'efficiency_reward': 0.0,
            'thermal_reward': 0.0,
            'degradation_penalty': 0.0,
            'safety_penalty': 0.0,
            'safety_violations': 0,
            'thermal_violations': 0,
            'power_deficit_events': 0,
            'lithium_plating_events': 0,
            'total_energy_processed_wh': 0.0,
            'charge_efficiency': []
        }
        
        self.current_step = 0
        self.episode_reward = 0
        self.done = False
        
        logger.info("H2ZBatteryLifeEnv reset to initial state")
        
        return self._get_observation(), {"state_dict": self.state.copy()}
    
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
        
        # Store previous SOH for reward calculation
        self.state['prev_soh'] = self.state['battery_soh']
        
        # Update orbital state
        orbital_updates = self._update_orbital_state()
        self.state.update(orbital_updates)
        
        # Mission mode transitions (randomly)
        if np.random.random() < 0.01:
            mode = np.random.choice(list(MissionMode))
            self.state['mission_mode'] = mode.value
            self.state['demands'] = self.power_demands[mode].copy()
            self.state['total_power_demand'] = sum(self.state['demands'].values())
        
        # Update battery dynamics
        battery_updates = self._update_battery_dynamics(action)
        self.state.update(battery_updates)
        
        # Update mission context
        self.state['days_since_launch'] += self.config.simulation_minutes_per_step / (60 * 24)
        self.state['battery_cycles_completed'] = self.degradation_model.cycles_completed
        self.state['capacity_fade_rate'] = self.degradation_model.capacity_fade / max(1, self.state['days_since_launch'])
        self.state['sei_layer_thickness'] = self.degradation_model.sei_thickness * 1e9
        
        # Validate state
        is_valid, error_msg = self.degradation_model.validate_state(
            self.state['battery_soc'],
            self.state['battery_temperature'],
            self.state['battery_voltage']
        )
        
        if not is_valid:
            logger.warning(f"Safety violation: {error_msg}")
        
        # Calculate reward
        reward, reward_components, penalties = self._calculate_reward(action, self.degradation_model.history)
        self.episode_reward += reward
        
        # Check termination conditions
        terminated = False
        if self.state['battery_soh'] < 0.5:  # Battery failed
            terminated = True
            logger.warning("Episode terminated: Battery SOH below 50%")
        if self.metrics['safety_violations'] > 10:  # Too many safety violations
            terminated = True
            logger.warning("Episode terminated: Too many safety violations")
        
        # Check truncation
        max_steps = self.config.max_steps_per_episode
        truncated = self.current_step >= max_steps
        
        # Info
        info = {
            'state_dict': self.state.copy(),
            'battery_soc': self.state['battery_soc'],
            'battery_soh': self.state['battery_soh'],
            'battery_temperature': self.state['battery_temperature'],
            'battery_voltage': self.state['battery_voltage'],
            'battery_current': self.state['battery_current'],
            'solar_power_available': self.state['solar_power_available'],
            'total_power_demand': self.state['total_power_demand'],
            'mission_mode': self.state['mission_mode'],
            'days_since_launch': self.state['days_since_launch'],
            'cycles_completed': self.state['battery_cycles_completed'],
            'episode_reward': self.episode_reward,
            'step': self.current_step,
            'reward_components': reward_components,
            'penalties': penalties,
            'safety_violations': self.metrics['safety_violations'],
            'power_deficit_events': self.metrics['power_deficit_events'],
            'lithium_plating_events': self.metrics['lithium_plating_events'],
            'total_energy_wh': self.metrics['total_energy_processed_wh']
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def render(self, mode: str = 'human'):
        """Render environment state."""
        if mode == 'human':
            print(f"\n{'='*60}")
            print(f"H2Z Battery Life Environment - Step {self.current_step}")
            print(f"{'='*60}")
            print(f"Battery SOC: {self.state['battery_soc']*100:.1f}%")
            print(f"Battery SOH: {self.state['battery_soh']*100:.1f}%")
            print(f"Battery Temperature: {self.state['battery_temperature']-273.15:.1f}°C")
            print(f"Battery Voltage: {self.state['battery_voltage']:.1f}V")
            print(f"Battery Current: {self.state['battery_current']:.2f}A")
            print(f"Solar Power: {self.state['solar_power_available']:.1f}W")
            print(f"Power Demand: {self.state['total_power_demand']:.1f}W")
            print(f"Episode Reward: {self.episode_reward:.2f}")
            print(f"{'='*60}\n")
        
        return None


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("H2Z Battery Life Environment Demo")
    logger.info("=" * 60)
    
    # Create environment
    env = H2ZBatteryLifeEnv()
    
    # Test reset
    obs, info = env.reset(seed=42)
    logger.info(f"Initial observation shape: {obs.shape}")
    logger.info(f"Initial info: {info}")
    
    # Test random actions
    logger.info("\nRunning 10 random steps...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        logger.info(f"Step {i+1}: Reward={reward:.2f}, SOC={info['battery_soc']*100:.1f}%, "
                   f"SOH={info['battery_soh']*100:.1f}%, Violations={info['safety_violations']}")
        
        if terminated or truncated:
            logger.info("Episode ended!")
            break
    
    logger.info("\n" + "=" * 60)
    logger.info("Environment demo completed!")

