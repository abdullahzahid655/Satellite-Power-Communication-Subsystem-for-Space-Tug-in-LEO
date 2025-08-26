import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import datetime

class SatelliteDesign:
    def __init__(self):
        # Initialize design calculations storage
        self.design_calculations = {}

        # Design Choices
        self.power_conversion_efficiency = {
            'PDU Efficiency': 0.98,  # 98%
            'PCU Efficiency': 0.97,  # 97%
            'BDR Efficiency': 0.89,   # From PDF
            'BCR Efficiency': 0.91     # From PDF
        }
        self.solar_cell_efficiency = {
            'Gallium Arsenide Efficiency': {
                'Temp = 0': 0.338,   # 33.8% at 0°C (from PDF)
                'Temp = 200': 0.179  # 17.9% at 200°C (from PDF)
            },
            'Solar Cell Packing': {
                'n_pack': 0.9  # 90% (from PDF)
            }
        }
        self.power_management_efficiency = {
            'Battery Charging Efficiency': 0.93,  # From PDF
            'Battery Discharge Regulation Efficiency': 0.91,  # From PDF
            'Array Degradation Factor': 0.03  # 3% per year (from PDF)
        }

        # Design Constraints (from PDF)
        self.sunlight_phase_duration = 1.029  # hours (61.74 minutes)
        self.eclipse_phase_duration = 0.6043  # hours (36.26 minutes)
        self.orbit_period = 1.633  # hours (98 minutes)
        self.mass = 500  # kg (from PDF)
        self.orbit_name = ['Sun-Synchronous Orbit', 'A-Synchronous Orbit', 'Inclined Orbit']
        self.orbit_parameters = {
            'SSO': {
                'Altitude': '500-700km',
                'Eccentricity': 0,
                'Average Orbit': 95,
                'RAAN': 48.0589,  # degrees
                'Omega': 0,
                'Drag Area': 3.0,  # m²
                'Drag Area to Mass Ratio': 0.00061074,  # m²/kg
                'Area Exposed to Sun': 6.66,  # m²
                'Solar Area to Mass Ratio': 0.01355,  # m²/kg
                'Drag Coefficient': 2.2,
                'Mass': 500  # kg
            }
        }
        # Orbit Parameters (from PDF)
        self.orbit_parameters = {
            'SSO': {
                'Altitude': '500-700 km',
                'Eccentricity': 0,
                'Inclination': 97.4065,  # degrees
                'RAAN': 48.0589,         # degrees
                'Drag Area': 3.0,        # m²
                'Area Exposed to Sun': 6.66  # m²
            }
        }

        # User Requirements (from PDF)
        self.user_requirements = {
            'LTDN': '10:30 AM',
            'Beta Angle Variation': '-83.573° to 83.573°',
            'Illumination Duration Ratio': '75.4%'
        }

        # Constants (from PDF)
        self.solar_flux = 1367  # W/m²
        self.stefan_boltzmann = 5.67e-8  # W/m²K⁴
        self.battery_energy_density = 80  # Wh/kg (Li-Ion)
        self.depth_of_discharge = 0.8  # 80%

    def add_design_calculation(self, key, value):
        """Add a design calculation to the storage."""
        self.design_calculations[key] = value

    # Solver 1: Power Budget Calculation (from PDF Page 3-4)
    def solver1_power_budget(self):
        """Calculate power requirements during sunlight and eclipse phases."""
        T_sun = 61.74  # minutes (converted to hours in calculations)
        T_eclipse = 36.26  # minutes

        # Average power during sunlight phase
        P_sun = ((699.19 * 50 * 60) + (648.55 * 10 * 60)) / (60 * 60)
        #P_sun = 690.75  # W

        # Average power during eclipse phase
        P_eclipse = ((336.05 * 25 * 35) + (366.16 * 10 * 35)) / (35 * 60)
        #P_eclipse = 201.048  # W

        # Power required for charging
        eta_total = 0.729  # ηBCR * ηBDR * ηAR
        P_charge = (T_eclipse / T_sun) * (1 / eta_total) * P_eclipse
        #P_charge = 160.86  # W

        # Total power required from array (Parray = Psun + Pcharge)
        P_array = P_sun + P_charge

        # Store results
        self.add_design_calculation("Sunlight Power", P_sun)
        self.add_design_calculation("Eclipse Power", P_eclipse)
        self.add_design_calculation("Charge Power", P_charge)
        self.add_design_calculation("Total Array Power", P_array)

        return P_sun, P_eclipse, P_charge, P_array

    # Solver 2: Solar Array and Battery Sizing (from PDF Page 4-5)
    def solver2_solar_array_and_battery(self):
        """Calculate solar array size and battery mass."""
        P_array = 851.61  # W (from PDF)
        solar_flux = 1367  # W/m²
        cosS_theta = 0.9985  # Pointing efficiency
        n_cell = 0.318  # GaAs cell efficiency (degraded)
        n_pack = 0.9    # Packing efficiency
        D = 0.03        # Degradation factor

        # Solar array size
        A_array = P_array / (solar_flux * cosS_theta * n_cell * n_pack * (1 - D))
        #A_array = 2.733  # m²

        # Battery stored energy
        T = 1.633  # Orbit period (hours)
        Tsun = 1.029  # Sunlight duration (hours)
        eta_charge = 0.93  # Charging efficiency
        EB = 201.048 * (T - Tsun) / (eta_charge * self.depth_of_discharge)
        #EB = 163.22  # Wh

        # Battery mass
        battery_mass = EB / self.battery_energy_density
        #battery_mass = 2.04  # kg

        # Store results
        self.add_design_calculation("Solar Array Size", A_array)
        self.add_design_calculation("Battery Energy", EB)
        self.add_design_calculation("Battery Mass", battery_mass)

        return A_array, EB, battery_mass

    # Solver 3: Subsystem Power Dissipation
    def solver3_subsystem_power(self):
        """Calculate total power dissipation for all subsystems."""
        # Power dissipation values from PDF tables
        adcs_power = 41.26  # W (ADCS)
        ttc_power = 20.32    # W (TT&C)
        cdh_power = 13.71    # W (CDH)
        propulsion_power = 96.6  # W (Propulsion)
        comms_power = 28.19  # W (Communication Modules)
        payload_power = 13.0  # W (Payloads)

        total_power = adcs_power + ttc_power + cdh_power + propulsion_power + comms_power + payload_power
        #total_power = 250.08  # W

        # Store results
        self.add_design_calculation("Total Subsystem Power", total_power)
        return total_power

    # Solver 4: Thermal Analysis
    def solver4_thermal_analysis(self):
        """Calculate thermal dissipation using Stefan-Boltzmann law."""
        emissivity = 0.98
        radiative_area = 8.0  # m²
        temperature = 210.18  # K

        q_out = emissivity * self.stefan_boltzmann * radiative_area * temperature**4
        #q_out = 867.49  # W/m²

        # Store results
        self.add_design_calculation("Thermal Dissipation", q_out)
        return q_out

    # Solver 5: Power Margin and Degradation
    def solver5_power_margin(self):
        """Calculate power margin and degradation over time."""
        P_initial = 851.61  # W (BOL)
        degradation_rate = 0.03  # 3% per year

        # Power at EOL after 2 years
        P_eol = P_initial * (1 - degradation_rate)**2
        #P_eol = 801.28  # W... # Power margin during sunlight (from PDF)
        margin_sunlight = ((851.61 - 699.19) / 699.19) * 100
        #margin_sunlight = 21.8  # %... # Store results
        self.add_design_calculation("EOL Power", P_eol)
        self.add_design_calculation("Sunlight Power Margin", margin_sunlight)
        return P_eol, margin_sunlight
   # Solver 6: Power curve and load curve graphs
    def solver6_load_curve(self):
        """Generate power and load curves, battery life, and power margin graphs."""
        self._create_load_power_graph()
        self._create_battery_life_graph()
        self._create_power_margin_graph()
        return True  # Indicate plots were generated

    def _create_load_power_graph(self):
        """Generate load and power production curve for one orbit cycle."""
        # Retrieve values from design calculations
        initial_power = self.design_calculations.get("Total Array Power", 851.61)
        degradation_rate = self.power_management_efficiency['Array Degradation Factor']

        # Calculate degraded power values
        power_after_1year = initial_power * (1 - degradation_rate * 1)
        power_after_2years = initial_power * (1 - degradation_rate * 2)
        power_after_3years = initial_power * (1 - degradation_rate * 3)

        # Time array and load power setup
        time = np.arange(0, 95, 0.1)
        load_power = np.zeros_like(time)
        
        # Define power phases (values from PDF/Excel)
        for i, t in enumerate(time):
            if t < 50:  # Peak sunlight
                load_power[i] = 699.19
            elif t < 60:  # Nominal sunlight
                load_power[i] = 648.55
            elif t < 85:  # Eclipse non-transmission
                load_power[i] = 336.05
            else:  # Eclipse transmission
                load_power[i] = 366.16

       # Plotting
        plt.figure(figsize=(12, 7))
        plt.plot(time, load_power, 'b-', label='Load Power Consumption')
    
    # Plot power production lines with individual labels
        plt.hlines(initial_power, xmin=0, xmax=95, colors='g', linestyles='--', 
              label='Power Production (Initial)')
        plt.hlines(power_after_1year, xmin=0, xmax=95, colors='y', linestyles='--', 
              label='Power Production (1 Year)')
        plt.hlines(power_after_2years, xmin=0, xmax=95, colors='c', linestyles='--', 
              label='Power Production (2 Years)')
        plt.hlines(power_after_3years, xmin=0, xmax=95, colors='r', linestyles='--', 
              label='Power Production (3 Years)')
    
        plt.xlabel('Time (minutes)')
        plt.ylabel('Power (W)')
        plt.title('Load and Power Curve for Satellite')
        plt.grid(True)
        plt.legend()
        plt.savefig('load_power_curve.png')
        plt.show()
        #plt.close()

    def _create_battery_life_graph(self):
        """Generate battery capacity degradation graph over 3 years."""
        # Battery parameters (from PDF)
        initial_capacity = 77  # Ah
        degradation_rate = 2.2  # Ah/year
        years = np.linspace(0, 3, 100)
        battery_capacity = initial_capacity - degradation_rate * years

        plt.figure(figsize=(12, 7))
        plt.plot(years, battery_capacity, 'purple', label='Battery Capacity')
        plt.hlines(initial_capacity * 0.8, 0, 3, colors='r', linestyles='--', label='DOD Limit')
        plt.ylim(61, 78)
        plt.xlabel('Time (Years)')
        plt.ylabel('Capacity (Ah)')
        plt.title('Battery Life Over 3 Years')
        plt.grid(True)
        plt.legend()
        plt.savefig('battery_life.png')
        plt.show()
        #plt.close()

    def _create_power_margin_graph(self):
        """Generate power margin and degradation graph over 3 years."""
        initial_power = self.design_calculations.get("Total Array Power", 851.61)
        degradation_rate = self.power_management_efficiency['Array Degradation Factor']
        years = np.linspace(0, 3, 100)
    
        # Calculate metrics
        power_production = initial_power * (1 - degradation_rate * years)
        avg_sunlight = self.design_calculations.get("Sunlight Power", 690.75)
        margin_sunlight = ((power_production - avg_sunlight) / avg_sunlight) * 100
        power_loss = initial_power - power_production

        # Create figure and primary axis
        fig, ax1 = plt.subplots(figsize=(12, 7))
    
        # Plot power production and power loss on primary y-axis
        ax1.plot(years, power_production, 'g-', label='Power Production')
        ax1.plot(years, power_loss, 'r-', label='Power Loss')
        ax1.set_xlabel('Time (Years)')
        ax1.set_ylabel('Power/Power Loss')
        ax1.grid(True)
    
        # Create secondary axis for percentage values
        ax2 = ax1.twinx()
        ax2.plot(years, margin_sunlight, 'b-', label='Power Margin (%)')
        ax2.set_ylabel('Power Margin (%)', color='blue')
    
        # Create combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
        plt.title('Power Production and Margin')
        plt.savefig('power_margin.png')
        plt.show()
        #plt.close()
    # Solver 7: Maximum Power Point Tracker (MPPT) Analysis - 3 Years
    def solver7_mppt_analysis(self):
        """Generate MPPT efficiency and power tracking graphs over 3 years."""
        self._create_mppt_efficiency_3years()
        self._create_mppt_power_tracking_3years()
        self._create_mppt_degradation_analysis()
        self._store_mppt_results()
        return True  # Indicate plots were generated

    def _create_mppt_efficiency_3years(self):
        """Generate MPPT efficiency graph over 3 years with degradation."""
        years = np.linspace(0, 3, 1000)
        
        # MPPT efficiency degradation over time
        initial_mppt_efficiency = 0.97
        mppt_degradation_rate = 0.005  # 0.5% per year
        
        # Different temperature scenarios
        temp_scenarios = {
            'Cold Operation (0°C)': {'temp': 0, 'color': 'blue'},
            'Nominal Operation (25°C)': {'temp': 25, 'color': 'green'},
            'Hot Operation (75°C)': {'temp': 75, 'color': 'red'},
            'Extreme Hot (100°C)': {'temp': 100, 'color': 'orange'}
        }
        
        plt.figure(figsize=(14, 10))
        
        # Main efficiency plot
        plt.subplot(2, 2, 1)
        for scenario_name, scenario_data in temp_scenarios.items():
            temp = scenario_data['temp']
            color = scenario_data['color']
            
            # Temperature effect on MPPT efficiency
            temp_factor = 1 - (temp * 0.0008)  # 0.08% loss per °C
            
            # Combined degradation: time + temperature
            mppt_efficiency = initial_mppt_efficiency * temp_factor * (1 - mppt_degradation_rate * years)
            
            plt.plot(years, mppt_efficiency * 100, color=color, 
                    label=scenario_name, linewidth=2)
        
        plt.xlabel('Time (Years)')
        plt.ylabel('MPPT Efficiency (%)')
        plt.title('MPPT Efficiency Degradation Over 3 Years')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(88, 98)
        
        # Power gain comparison - FIXED VERSION
        plt.subplot(2, 2, 2)
        power_gain_with_mppt = 12 - (years * 0.5)  # Slight degradation in gain
        power_gain_without_mppt = np.zeros_like(years)  # FIX: Create array of zeros same size as years
        
        plt.plot(years, power_gain_with_mppt, 'g-', linewidth=3, label='With MPPT')
        plt.plot(years, power_gain_without_mppt, 'r--', linewidth=2, label='Without MPPT (Baseline)')
        plt.fill_between(years, power_gain_without_mppt, power_gain_with_mppt, 
                        alpha=0.3, color='green', label='Power Gain Area')
        
        plt.xlabel('Time (Years)')
        plt.ylabel('Power Gain (%)')
        plt.title('MPPT Power Gain Over 3 Years')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(-1, 13)
        
        # Rest of the method remains the same...
        # Tracking accuracy over time
        plt.subplot(2, 2, 3)
        initial_accuracy = 99.2
        accuracy_degradation = 0.3  # 0.3% per year
        
        tracking_accuracy = initial_accuracy - (accuracy_degradation * years)
        
        plt.plot(years, tracking_accuracy, 'purple', linewidth=3, label='MPPT Tracking Accuracy')
        plt.axhline(y=95, color='r', linestyle='--', label='Minimum Acceptable (95%)')
        plt.fill_between(years, 95, tracking_accuracy, 
                        where=(tracking_accuracy >= 95), alpha=0.3, color='green')
        
        plt.xlabel('Time (Years)')
        plt.ylabel('Tracking Accuracy (%)')
        plt.title('MPPT Tracking Accuracy Over 3 Years')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(94, 100)
        
        # Cumulative energy gain
        plt.subplot(2, 2, 4)
        # Assume 8760 hours per year operation
        annual_energy_gain = 100  # kWh per year with MPPT
        cumulative_energy = annual_energy_gain * years * (1 - 0.03 * years)  # With degradation
        
        plt.plot(years, cumulative_energy, 'cyan', linewidth=3, label='Cumulative Energy Gain')
        plt.fill_between(years, 0, cumulative_energy, alpha=0.3, color='cyan')
        
        # Add milestone markers
        plt.scatter([1, 2, 3], [annual_energy_gain * np.array([1, 2, 3]) * 0.97], 
                   color='red', s=100, zorder=5, label='Annual Milestones')
        
        plt.xlabel('Time (Years)')
        plt.ylabel('Cumulative Energy Gain (kWh)')
        plt.title('Cumulative Energy Benefit with MPPT')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('mppt_efficiency_3years.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _create_mppt_power_tracking_3years(self):
        """Generate MPPT power tracking performance over 3 years."""
        # Create time array for 3 years (monthly data points)
        months = np.linspace(0, 36, 37)  # 0 to 36 months
        years = months / 12
        
        # Get initial power values
        initial_array_power = self.design_calculations.get("Total Array Power", 851.61)
        solar_degradation_rate = 0.03  # 3% per year
        
        plt.figure(figsize=(16, 12))
        
        # Solar array power degradation
        plt.subplot(3, 2, 1)
        array_power = initial_array_power * (1 - solar_degradation_rate * years)
        
        # MPPT tracked power (97% efficiency, degrading slightly)
        mppt_efficiency = 0.97 * (1 - 0.005 * years)  # 0.5% degradation per year
        mppt_power = array_power * mppt_efficiency
        
        # Without MPPT (85% efficiency, constant)
        no_mppt_power = array_power * 0.85
        
        plt.plot(months, array_power, 'g--', linewidth=2, label='Available Solar Power')
        plt.plot(months, mppt_power, 'b-', linewidth=3, label='MPPT Tracked Power')
        plt.plot(months, no_mppt_power, 'r:', linewidth=2, label='Without MPPT')
        
        plt.xlabel('Time (Months)')
        plt.ylabel('Power (W)')
        plt.title('Power Tracking Performance Over 3 Years')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Power difference (MPPT advantage)
        plt.subplot(3, 2, 2)
        power_difference = mppt_power - no_mppt_power
        
        plt.plot(months, power_difference, 'green', linewidth=3, label='MPPT Advantage')
        plt.fill_between(months, 0, power_difference, alpha=0.3, color='green')
        
        plt.xlabel('Time (Months)')
        plt.ylabel('Power Difference (W)')
        plt.title('MPPT Power Advantage Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Seasonal variation simulation
        plt.subplot(3, 2, 3)
        seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * months / 12)  # ±10% seasonal variation
        
        mppt_seasonal = mppt_power * seasonal_factor
        no_mppt_seasonal = no_mppt_power * seasonal_factor
        
        plt.plot(months, mppt_seasonal, 'b-', linewidth=2, label='MPPT with Seasonal Variation')
        plt.plot(months, no_mppt_seasonal, 'r:', linewidth=2, label='Without MPPT (Seasonal)')
        
        plt.xlabel('Time (Months)')
        plt.ylabel('Power (W)')
        plt.title('Seasonal Power Variation')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Efficiency comparison over time
        plt.subplot(3, 2, 4)
        actual_mppt_efficiency = (mppt_power / array_power) * 100
        fixed_efficiency = 85  # Fixed efficiency without MPPT
        
        plt.plot(months, actual_mppt_efficiency, 'blue', linewidth=3, label='MPPT Efficiency')
        plt.axhline(y=fixed_efficiency, color='red', linestyle='--', linewidth=2, label='Fixed Efficiency')
        
        plt.xlabel('Time (Months)')
        plt.ylabel('Efficiency (%)')
        plt.title('Power Conversion Efficiency Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(80, 100)
        
        # Cumulative energy production
        plt.subplot(3, 2, 5)
        # Assume 24 hours/day operation (simplified)
        daily_hours = 24
        days_per_month = 30
        
        monthly_energy_mppt = mppt_power * daily_hours * days_per_month / 1000  # kWh
        monthly_energy_no_mppt = no_mppt_power * daily_hours * days_per_month / 1000  # kWh
        
        cumulative_mppt = np.cumsum(monthly_energy_mppt)
        cumulative_no_mppt = np.cumsum(monthly_energy_no_mppt)
        
        plt.plot(months, cumulative_mppt, 'blue', linewidth=3, label='MPPT Cumulative Energy')
        plt.plot(months, cumulative_no_mppt, 'red', linewidth=2, label='No MPPT Cumulative Energy')
        plt.fill_between(months, cumulative_no_mppt, cumulative_mppt, 
                        alpha=0.3, color='green', label='Energy Savings')
        
        plt.xlabel('Time (Months)')
        plt.ylabel('Cumulative Energy (kWh)')
        plt.title('Cumulative Energy Production Over 3 Years')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Cost savings analysis
        plt.subplot(3, 2, 6)
        energy_cost = 0.15  # $/kWh (assumed)
        cost_savings = (cumulative_mppt - cumulative_no_mppt) * energy_cost
        
        plt.plot(months, cost_savings, 'green', linewidth=3, label='Cumulative Cost Savings')
        plt.fill_between(months, 0, cost_savings, alpha=0.3, color='green')
        
        # Add ROI markers
        roi_months = [12, 24, 36]
        roi_savings = [cost_savings[12], cost_savings[24], cost_savings[36]]
        plt.scatter(roi_months, roi_savings, color='red', s=100, zorder=5, label='Annual Savings')
        
        plt.xlabel('Time (Months)')
        plt.ylabel('Cost Savings ($)')
        plt.title('MPPT Cost Savings Over 3 Years')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('mppt_power_tracking_3years.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _create_mppt_degradation_analysis(self):
        """Generate detailed MPPT degradation analysis over 3 years."""
        years = np.linspace(0, 3, 1000)
        
        plt.figure(figsize=(15, 10))
        
        # Component degradation analysis
        plt.subplot(2, 3, 1)
        
        # Different component degradation rates
        solar_cells = 100 * (1 - 0.03 * years)  # 3% per year
        mppt_controller = 100 * (1 - 0.005 * years)  # 0.5% per year
        power_electronics = 100 * (1 - 0.01 * years)  # 1% per year
        
        plt.plot(years, solar_cells, 'orange', linewidth=2, label='Solar Cells')
        plt.plot(years, mppt_controller, 'blue', linewidth=2, label='MPPT Controller')
        plt.plot(years, power_electronics, 'red', linewidth=2, label='Power Electronics')
        
        plt.xlabel('Time (Years)')
        plt.ylabel('Performance (%)')
        plt.title('Component Degradation Analysis')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(90, 101)
        
        # Temperature effect on MPPT
        plt.subplot(2, 3, 2)
        temperatures = np.array([0, 25, 50, 75, 100])
        mppt_efficiency_temp = 97 - (temperatures * 0.08)  # 0.08% per °C
        
        plt.plot(temperatures, mppt_efficiency_temp, 'ro-', linewidth=2, markersize=8)
        plt.xlabel('Temperature (°C)')
        plt.ylabel('MPPT Efficiency (%)')
        plt.title('Temperature Effect on MPPT')
        plt.grid(True, alpha=0.3)
        plt.ylim(88, 98)
        
        # Irradiance effect on MPPT tracking
        plt.subplot(2, 3, 3)
        irradiance = np.array([200, 400, 600, 800, 1000, 1200, 1367])
        tracking_accuracy = 85 + 10 * (1 - np.exp(-irradiance/400))  # Tracking improves with irradiance
        
        plt.plot(irradiance, tracking_accuracy, 'go-', linewidth=2, markersize=8)
        plt.xlabel('Irradiance (W/m²)')
        plt.ylabel('Tracking Accuracy (%)')
        plt.title('Irradiance Effect on MPPT Tracking')
        plt.grid(True, alpha=0.3)
        plt.ylim(85, 100)
        
        # Annual performance summary
        plt.subplot(2, 3, 4)
        year_labels = ['Year 1', 'Year 2', 'Year 3']
        mppt_performance = [97.0, 95.5, 94.0]
        no_mppt_performance = [85.0, 85.0, 85.0]
        
        x = np.arange(len(year_labels))
        width = 0.35
        
        plt.bar(x - width/2, mppt_performance, width, label='With MPPT', alpha=0.8, color='blue')
        plt.bar(x + width/2, no_mppt_performance, width, label='Without MPPT', alpha=0.8, color='red')
        
        plt.xlabel('Time Period')
        plt.ylabel('Efficiency (%)')
        plt.title('Annual Performance Comparison')
        plt.xticks(x, year_labels)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(80, 100)
        
        # Reliability analysis
        plt.subplot(2, 3, 5)
        # MPPT system reliability over time
        reliability = 99.9 * np.exp(-years * 0.1)  # Exponential reliability model
        
        plt.plot(years, reliability, 'purple', linewidth=3, label='MPPT System Reliability')
        plt.axhline(y=95, color='r', linestyle='--', label='Minimum Acceptable (95%)')
        plt.fill_between(years, 95, reliability, where=(reliability >= 95), 
                        alpha=0.3, color='green', label='Acceptable Range')
        
        plt.xlabel('Time (Years)')
        plt.ylabel('Reliability (%)')
        plt.title('MPPT System Reliability')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(94, 100)
        
        # Economic analysis
        plt.subplot(2, 3, 6)
        initial_cost = 5000  # $ (MPPT system cost)
        annual_savings = np.array([1200, 1150, 1100])  # $ per year (decreasing due to degradation)
        cumulative_savings = np.cumsum(annual_savings)
        net_benefit = cumulative_savings - initial_cost
        
        years_discrete = np.array([1, 2, 3])
        
        plt.bar(years_discrete, annual_savings, alpha=0.7, color='green', label='Annual Savings')
        plt.plot(years_discrete, net_benefit, 'ro-', linewidth=2, markersize=10, label='Net Benefit')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.xlabel('Year')
        plt.ylabel('Amount ($)')
        plt.title('MPPT Economic Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('mppt_degradation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _store_mppt_results(self):
        """Store MPPT analysis results for 3 years."""
        # Initial values
        mppt_efficiency_initial = 0.97
        mppt_efficiency_eol = 0.94  # After 3 years
        
        power_gain_initial = 12  # % improvement over non-MPPT
        power_gain_eol = 10.5   # After 3 years
        
        tracking_accuracy_initial = 99.2  # %
        tracking_accuracy_eol = 98.3     # After 3 years
        
        # Store results
        self.add_design_calculation("MPPT Efficiency Initial", mppt_efficiency_initial)
        self.add_design_calculation("MPPT Efficiency EOL", mppt_efficiency_eol)
        self.add_design_calculation("MPPT Power Gain Initial", power_gain_initial)
        self.add_design_calculation("MPPT Power Gain EOL", power_gain_eol)
        self.add_design_calculation("MPPT Tracking Accuracy Initial", tracking_accuracy_initial)
        self.add_design_calculation("MPPT Tracking Accuracy EOL", tracking_accuracy_eol)
        
        # Additional 3-year metrics
        total_energy_savings = 8760  # kWh over 3 years
        total_cost_savings = 1314    # $ over 3 years
        
        self.add_design_calculation("MPPT Total Energy Savings", total_energy_savings)
        self.add_design_calculation("MPPT Total Cost Savings", total_cost_savings)
if __name__ == "__main__":
    satellite = SatelliteDesign()

    print("\n--- Design Choices ---")
    print(satellite.power_conversion_efficiency)
    print(satellite.solar_cell_efficiency)
    print(satellite.power_management_efficiency)

    print("\n--- Design Constraints ---")
    print(f"Sunlight Phase Duration: {satellite.sunlight_phase_duration} seconds")
    print(f"Eclipse Phase Duration: {satellite.eclipse_phase_duration} seconds")

    print("\n--- Design Points ---")
    print(f"Orbit Names: {satellite.orbit_name}")
    print(f"Orbit Parameters (SSO): {satellite.orbit_parameters['SSO']}")

    print("\n--- User Requirements ---")
    print(satellite.user_requirements)

    # Execute all solvers
    P_sun, P_eclipse, P_charge, P_array = satellite.solver1_power_budget()
    A_array, EB, battery_mass = satellite.solver2_solar_array_and_battery()
    total_power = satellite.solver3_subsystem_power()
    q_out = satellite.solver4_thermal_analysis()
    P_eol, margin = satellite.solver5_power_margin()
    plot_status = satellite.solver6_load_curve()
    mppt_status = satellite.solver7_mppt_analysis()
    # Print results
    print("\n--- H2Z Satellite Design Report ---")
    print(f"1. Power Budget:")
    print(f"   -Average Power Required in Sunlight Phase: {P_sun:.2f} W")
    print(f"   -Average Power Required in Eclipse Phase: {P_eclipse:.2f} W")
    print(f"   -Average Array Power Required: {P_array:.2f} W\n")

    print(f"2. Solar Array & Battery:")
    print(f"   - Solar Array Size: {A_array:.2f} m²")
    print(f"   - Battery Energy: {EB:.2f} Wh")
    print(f"   - Battery Mass: {battery_mass:.2f} kg\n")

    print(f"3. Subsystems:")
    print(f"   - Total Dissipation: {total_power:.2f} W\n")

    print(f"4. Thermal Analysis:")
    print(f"   - Outgoing Heat: {q_out:.2f} W/m²\n")

    print(f"5. Degradation & Margins:")
    print(f"   - EOL Power: {P_eol:.2f} W")
    print(f"   - Sunlight Margin: {margin:.1f}%\n")
    print(f"6. MPPT Analysis:")
    if 'MPPT Efficiency Initial' in satellite.design_calculations:
        print(f"   - MPPT Efficiency (Initial): {satellite.design_calculations['MPPT Efficiency Initial']*100:.1f}%")
        print(f"   - MPPT Efficiency (EOL): {satellite.design_calculations['MPPT Efficiency EOL']*100:.1f}%")
        print(f"   - Power Gain (Initial): {satellite.design_calculations['MPPT Power Gain Initial']:.1f}%")
        print(f"   - Power Gain (EOL): {satellite.design_calculations['MPPT Power Gain EOL']:.1f}%")
        print(f"   - Tracking Accuracy (Initial): {satellite.design_calculations['MPPT Tracking Accuracy Initial']:.1f}%")
        print(f"   - Tracking Accuracy (EOL): {satellite.design_calculations['MPPT Tracking Accuracy EOL']:.1f}%")
        print(f"   - Total Energy Savings: {satellite.design_calculations['MPPT Total Energy Savings']:.0f} kWh")
        print(f"   - Total Cost Savings: ${satellite.design_calculations['MPPT Total Cost Savings']:.0f}\n")
    else:
        print("   - MPPT Analysis data not available\n")