"""
H2Z Space Tug Battery Degradation Physics Model

Physics-based semi-empirical model for Li-Ion battery degradation
incorporating:
- Capacity fade (Arrhenius kinetics)
- Internal resistance growth
- SEI layer growth
- Lithium plating risk
- Calendar aging

Based on electrochemical models for space-grade Li-Ion batteries.

Author: H2Z Development Team
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DegradationMechanism(Enum):
    """Battery degradation mechanisms."""
    CAPACITY_FADE = "capacity_fade"
    RESISTANCE_GROWTH = "resistance_growth"
    SEI_GROWTH = "sei_growth"
    LITHIUM_PLATING = "lithium_plating"
    CALENDAR_AGING = "calendar_aging"


@dataclass
class BatteryDegradationConfig:
    """Configuration for battery degradation model."""
    # Nominal battery specifications
    nominal_capacity_wh: float = 163.22  # Wh (from requirements)
    nominal_voltage: float = 28.0       # V (7S Li-Ion pack)
    nominal_current_ah: float = 5.83      # Ah (163.22Wh / 28V)
    
    # Model parameters (calibrated to Li-Ion chemistry)
    A_pre_exponential: float = 1e-6      # Pre-exponential factor
    Ea_activation: float = 31500         # J/mol (activation energy)
    Ea_SEI: float = 35000               # J/mol (SEI growth activation)
    Ea_calendar: float = 25000           # J/mol (calendar aging)
    
    # Degradation coefficients
    k_cycle: float = 0.0001              # Cycle fade coefficient
    k_calendar: float = 0.00005          # Calendar fade coefficient
    k_resistance: float = 0.0005         # Resistance growth coefficient
    k_SEI: float = 1e-12                # SEI growth coefficient
    
    # Physical constraints
    sei_initial_thickness_m: float = 1e-9  # meters
    R_int_initial_ohms: float = 0.05       # Ohms
    plating_threshold_temp_K: float = 273.0  # K
    plating_threshold_c_rate: float = 0.5     # C-rate
    
    # Operating limits
    soc_min: float = 0.15
    soc_max: float = 0.95
    temp_min_K: float = 173.15    # -100°C
    temp_max_K: float = 293.15    # +20°C
    voltage_min_V: float = 21.0
    voltage_max_V: float = 44.0
    
    # Gas constant
    R_gas: float = 8.314  # J/(mol·K)


class BatteryDegradationModel:
    """
    Physics-based battery degradation simulator.
    
    Implements semi-empirical models for:
    1. Capacity Fade: Based on Arrhenius equation and cycle counting
    2. Internal Resistance Growth: Temperature and cycle dependent
    3. SEI Layer Growth: Time and temperature dependent
    4. Lithium Plating: Fast charging at low temperatures
    5. Calendar Aging: Time-based degradation
    
    References:
    - Doyle-Fuller-Newman electrochemical model
    - NASA EVBatt models for Li-Ion batteries
    - Arrhenius temperature dependence
    """
    
    def __init__(self, config: BatteryDegradationConfig = None):
        self.config = config or BatteryDegradationConfig()
        
        # State variables
        self.capacity_fade = 0.0          # Cumulative capacity loss (fraction)
        self.R_int = self.config.R_int_initial_ohms
        self.sei_thickness = self.config.sei_initial_thickness_m
        self.lithium_plated = 0.0        # Cumulative lithium plating loss
        
        # Tracking variables
        self.cycles_completed = 0
        self.calendar_days = 0.0
        self.last_soc = 0.8
        self.last_temp = 210.18  # Nominal temperature (K)
        self.last_update_time = 0.0
        
        # History for visualization
        self.history = {
            'time_days': [],
            'capacity_fade': [],
            'R_int': [],
            'sei_thickness': [],
            'SOH': [],
            'cycles': []
        }
        
        logger.info("BatteryDegradationModel initialized")
        logger.info(f"  Nominal Capacity: {self.config.nominal_capacity_wh:.2f} Wh")
        logger.info(f"  Nominal Voltage: {self.config.nominal_voltage:.1f} V")
        logger.info(f"  Initial R_int: {self.config.R_int_initial_ohms:.4f} Ohms")
    
    @property
    def nominal_capacity_ah(self) -> float:
        """Calculate nominal capacity in Ah."""
        return self.config.nominal_capacity_wh / self.config.nominal_voltage
    
    @property
    def C_rate_1C(self) -> float:
        """Calculate 1C current in Amperes."""
        return self.nominal_capacity_ah
    
    @property
    def current_capacity_wh(self) -> float:
        """Calculate current remaining capacity."""
        return self.config.nominal_capacity_wh * (1 - self.capacity_fade)
    
    @property
    def current_capacity_ah(self) -> float:
        """Calculate current remaining capacity in Ah."""
        return self.current_capacity_wh / self.config.nominal_voltage
    
    @property
    def SOH(self) -> float:
        """Calculate State of Health (0-1)."""
        return 1.0 - self.capacity_fade
    
    def update(
        self,
        soc: float,
        temperature_K: float,
        current_A: float,
        time_delta_hours: float,
        record_history: bool = True
    ) -> Dict[str, float]:
        """
        Update battery degradation state.
        
        Args:
            soc: State of charge (0-1)
            temperature: Battery temperature (K)
            current: Charge current (A, positive=charging, negative=discharging)
            time_delta_hours: Time step in hours
            record_history: Whether to record history
            
        Returns:
            Dictionary with updated degradation state
        """
        # Calculate C-rate
        c_rate = abs(current_A) / self.C_rate_1C
        
        # Calculate average SOC during this period
        soc_avg = (soc + self.last_soc) / 2
        
        # Calculate temperature factor (Arrhenius)
        temp_factor = np.exp(
            -self.config.Ea_activation / 
            (self.config.R_gas * temperature_K)
        )
        
        # ===== CAPACITY FADE =====
        
        # Cycle-based fade (Arrhenius + DoD dependence)
        cycle_fade_rate = (
            self.config.A_pre_exponential * 
            temp_factor * 
            (soc_avg ** 0.5) *           # SOC dependence
            (c_rate ** 0.5) +            # C-rate dependence
            1e-8                          # Base rate
        )
        
        # Depth of discharge factor
        dod = abs(soc - self.last_soc)
        dod_factor = np.sqrt(dod) if dod > 0 else 0
        
        # Add cycle fade
        capacity_fade_delta = (
            cycle_fade_rate * 
            dod_factor * 
            time_delta_hours * 
            24  # Convert to per-day equivalent
        )
        
        # Calendar aging (time-based)
        calendar_fade_rate = (
            self.config.A_pre_exponential * 
            np.exp(-self.config.Ea_calendar / (self.config.R_gas * temperature_K))
        )
        calendar_fade_delta = calendar_fade_rate * np.sqrt(time_delta_hours / 24)
        
        # Add both fade mechanisms
        self.capacity_fade += capacity_fade_delta + calendar_fade_delta
        
        # ===== INTERNAL RESISTANCE GROWTH =====
        
        # Resistance grows with cycle count and temperature
        R_int_growth = (
            self.config.k_resistance * 
            dod_factor * 
            np.exp(0.05 * (temperature_K - 298))  # Temperature acceleration
        )
        self.R_int *= (1 + R_int_growth * time_delta_hours * 24)
        
        # ===== SEI LAYER GROWTH =====
        
        # SEI grows with time, temperature, and high SOC
        sei_growth_rate = (
            self.config.k_SEI * 
            np.exp(-self.config.Ea_SEI / (self.config.R_gas * temperature_K)) *
            (1 + soc_avg)  # Faster at high SOC
        )
        self.sei_thickness += sei_growth_rate * time_delta_hours * 3600  # Convert to seconds
        
        # ===== LITHIUM PLATING RISK =====
        
        # Lithium plating occurs during fast charging at low temperatures
        plating_events = 0
        permanent_plating_loss = 0.0
        
        if (temperature_K < self.config.plating_threshold_temp_K and 
            current_A > 0 and  # Charging
            c_rate > self.config.plating_threshold_c_rate):
            
            # Calculate plating severity
            temp_below_threshold = self.config.plating_threshold_temp_K - temperature_K
            rate_above_threshold = c_rate - self.config.plating_threshold_c_rate
            
            # Permanent capacity loss from plating
            permanent_plating_loss = (
                0.001 *  # Base loss coefficient
                temp_below_threshold * 
                rate_above_threshold * 
                time_delta_hours * 24
            )
            
            self.lithium_plated += permanent_plating_loss
            plating_events = 1
            
            logger.warning(
                f"Lithium plating risk! T={temperature_K:.1f}K, "
                f"C-rate={c_rate:.2f}, Loss={permanent_plating_loss:.6f}"
            )
        
        # Add plating loss to total capacity fade
        self.capacity_fade += permanent_plating_loss
        
        # ===== UPDATE TRACKING =====
        
        # Count cycles (each full DoD = 1 cycle equivalent)
        self.cycles_completed += dod
        self.calendar_days += time_delta_hours / 24
        
        # Update state
        self.last_soc = soc
        self.last_temp = temperature_K
        self.last_update_time += time_delta_hours
        
        # Record history
        if record_history:
            self.history['time_days'].append(self.calendar_days)
            self.history['capacity_fade'].append(self.capacity_fade)
            self.history['R_int'].append(self.R_int)
            self.history['sei_thickness'].append(self.sei_thickness * 1e9)  # nm
            self.history['SOH'].append(self.SOH)
            self.history['cycles'].append(self.cycles_completed)
        
        # Return current state
        return {
            'SOH': self.SOH,
            'capacity_fade': self.capacity_fade,
            'R_int': self.R_int,
            'sei_thickness_nm': self.sei_thickness * 1e9,
            'lithium_plated': self.lithium_plated,
            'cycles_completed': self.cycles_completed,
            'calendar_days': self.calendar_days,
            'plating_events': plating_events,
            'capacity_wh': self.current_capacity_wh,
            'capacity_ah': self.current_capacity_ah
        }
    
    def reset(self):
        """Reset degradation model to initial state."""
        self.capacity_fade = 0.0
        self.R_int = self.config.R_int_initial_ohms
        self.sei_thickness = self.config.sei_initial_thickness_m
        self.lithium_plated = 0.0
        self.cycles_completed = 0
        self.calendar_days = 0.0
        self.last_soc = 0.8
        self.last_temp = 210.18
        self.last_update_time = 0.0
        
        # Clear history
        self.history = {
            'time_days': [],
            'capacity_fade': [],
            'R_int': [],
            'sei_thickness': [],
            'SOH': [],
            'cycles': []
        }
        
        logger.info("BatteryDegradationModel reset to initial state")
    
    def project_SOH(
        self,
        target_days: float = 1095,  # 3 years
        step_days: float = 1.0,
        soc_profile: np.ndarray = None,
        temp_profile_K: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        Project battery SOH over mission lifetime.
        
        Args:
            target_days: Mission duration in days
            step_days: Time step in days
            soc_profile: Optional SOC profile (otherwise uses nominal)
            temp_profile_K: Optional temperature profile (otherwise uses nominal)
            
        Returns:
            Dictionary with time series projections
        """
        # Reset model
        self.reset()
        
        # Create time array
        days = np.arange(0, target_days + step_days, step_days)
        
        # Default profiles
        if soc_profile is None:
            soc_profile = np.full(len(days), 0.7)
        if temp_profile_K is None:
            temp_profile_K = np.full(len(days), 210.18)  # Nominal
        
        # Projections
        soh_projection = np.zeros(len(days))
        r_int_projection = np.zeros(len(days))
        
        # Simulate day by day
        for i, day in enumerate(days):
            soc = soc_profile[min(i, len(soc_profile) - 1)]
            temp = temp_profile_K[min(i, len(temp_profile_K) - 1)]
            
            # Typical daily cycle (one eclipse period)
            # ~0.4 DoD per day for LEO satellite
            daily_dod = 0.4
            current = 0.5 * self.C_rate_1C  # 0.5C charge/discharge
            
            # Simulate eclipse (discharge)
            self.update(
                soc=soc - daily_dod/2,
                temperature_K=temp,
                current=-current,
                time_delta_hours=0.6,  # ~36 min eclipse
                record_history=False
            )
            
            # Simulate sunlight (charge)
            self.update(
                soc=soc + daily_dod/2,
                temperature_K=temp,
                current=current,
                time_delta_hours=1.0,  # ~61 min charging
                record_history=False
            )
            
            soh_projection[i] = self.SOH
            r_int_projection[i] = self.R_int
        
        # Reset after projection
        self.reset()
        
        return {
            'days': days,
            'SOH': soh_projection,
            'R_int': r_int_projection,
            'initial_SOH': 1.0,
            'final_SOH': soh_projection[-1],
            'SOH_at_3_years': soh_projection[min(1095, len(soh_projection) - 1)]
        }
    
    def get_degradation_rate(
        self,
        soc: float,
        temperature_K: float,
        c_rate: float
    ) -> Dict[str, float]:
        """
        Calculate instantaneous degradation rates.
        
        Useful for reward function and analysis.
        
        Args:
            soc: State of charge (0-1)
            temperature: Temperature (K)
            c_rate: C-rate
            
        Returns:
            Dictionary with degradation rates
        """
        temp_factor = np.exp(
            -self.config.Ea_activation / 
            (self.config.R_gas * temperature_K)
        )
        
        return {
            'capacity_fade_rate_per_day': (
                self.config.A_pre_exponential * 
                temp_factor * 
                (soc ** 0.5) * 
                (c_rate ** 0.5) +
                self.config.A_pre_exponential * 
                np.exp(-self.config.Ea_calendar / (self.config.R_gas * temperature_K))
            ),
            'resistance_growth_rate': (
                self.config.k_resistance * 
                np.exp(0.05 * (temperature_K - 298))
            ),
            'sei_growth_rate_nm_per_day': (
                self.config.k_SEI * 1e9 * 24 *
                np.exp(-self.config.Ea_SEI / (self.config.R_gas * temperature_K)) *
                (1 + soc)
            ),
            'lithium_plating_risk': float(
                temperature_K < self.config.plating_threshold_temp_K and 
                c_rate > self.config.plating_threshold_c_rate
            )
        }
    
    def validate_state(self, soc: float, temperature_K: float, voltage_V: float) -> Tuple[bool, str]:
        """
        Validate battery is within safe operating limits.
        
        Args:
            soc: State of charge (0-1)
            temperature: Temperature (K)
            voltage: Voltage (V)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if soc < self.config.soc_min:
            return False, f"SOC {soc:.2f} below minimum {self.config.soc_min}"
        
        if soc > self.config.soc_max:
            return False, f"SOC {soc:.2f} above maximum {self.config.soc_max}"
        
        if temperature_K < self.config.temp_min_K:
            return False, f"Temperature {temperature_K:.1f}K below minimum {self.config.temp_min_K}K"
        
        if temperature_K > self.config.temp_max_K:
            return False, f"Temperature {temperature_K:.1f}K above maximum {self.config.temp_max_K}K"
        
        if voltage_V < self.config.voltage_min_V:
            return False, f"Voltage {voltage_V:.1f}V below minimum {self.config.voltage_min_V}V"
        
        if voltage_V > self.config.voltage_max_V:
            return False, f"Voltage {voltage_V:.1f}V above maximum {self.config.voltage_max_V}V"
        
        return True, "OK"


def calculate_battery_efficiency(
    voltage_V: float,
    current_A: float,
    R_int_ohms: float,
    charge_efficiency: float = 0.93,
    discharge_efficiency: float = 0.95
) -> Dict[str, float]:
    """
    Calculate battery charge/discharge efficiency.
    
    Args:
        voltage: Battery voltage (V)
        current: Current (A, positive=charging)
        R_int: Internal resistance (Ohms)
        charge_efficiency: Base charge efficiency
        discharge_efficiency: Base discharge efficiency
        
    Returns:
        Dictionary with efficiency metrics
    """
    # Power loss due to internal resistance
    I_squared_R = current_A ** 2 * R_int_ohms
    
    # Useful power
    if current_A > 0:  # Charging
        input_power = voltage_V * current_A
        stored_power = input_power - I_squared_R
        efficiency = stored_power / input_power * charge_efficiency
    else:  # Discharging
        output_power = voltage_V * abs(current_A) - I_squared_R
        efficiency = output_power / (voltage_V * abs(current_A)) * discharge_efficiency
    
    return {
        'input_power_W': voltage_V * current_A if current_A > 0 else 0,
        'output_power_W': voltage_V * abs(current_A) if current_A < 0 else 0,
        'loss_power_W': I_squared_R,
        'efficiency': efficiency
    }


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Battery Degradation Model Demo")
    logger.info("=" * 60)
    
    # Create degradation model
    model = BatteryDegradationModel()
    
    # Simulate 3-year mission
    projection = model.project_SOH(target_days=1095)
    
    logger.info(f"\n3-Year Mission Projection:")
    logger.info(f"  Initial SOH: {projection['initial_SOH']*100:.2f}%")
    logger.info(f"  Final SOH: {projection['final_SOH']*100:.2f}%")
    logger.info(f"  SOH @ 3 years: {projection['SOH_at_3_years']*100:.2f}%")
    
    # Test degradation rate calculation
    rates = model.get_degradation_rate(
        soc=0.7,
        temperature_K=210.18,
        c_rate=0.5
    )
    
    logger.info(f"\nDegradation Rates at T=210.18K, SOC=70%, 0.5C:")
    logger.info(f"  Capacity fade rate: {rates['capacity_fade_rate_per_day']*100:.4f}%/day")
    logger.info(f"  SEI growth rate: {rates['sei_growth_rate_nm_per_day']:.4f} nm/day")
    logger.info(f"  Li plating risk: {rates['lithium_plating_risk']}")
    
    # State validation
    is_valid, msg = model.validate_state(soc=0.8, temperature_K=250.0, voltage_V=30.0)
    logger.info(f"\nState Validation: {msg}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Battery degradation demo completed!")

