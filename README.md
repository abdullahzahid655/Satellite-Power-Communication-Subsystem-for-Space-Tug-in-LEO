# 🛰️ H2Z Satellite Power & Communication Subsystem
## AI-Enhanced Professional Portfolio Project

<div align="center">

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10+-green.svg)
![Status](https://img.shields.io/badge/Status-Active-yellow)

**A comprehensive satellite power and communication system simulation integrating aerospace engineering with cutting-edge AI/ML techniques**

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Core Aerospace Functions](#core-aerospace-functions)
- [AI/ML Capabilities](#aiml-capabilities)
- [Visualization Dashboard](#visualization-dashboard)
- [Requirements](#requirements)
- [Usage](#usage)
- [Output Examples](#output-examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project demonstrates the integration of **aerospace engineering domain expertise** with **modern AI/ML techniques** for satellite power system management. Originally a Final Year Project for a Space Tug mission in Low Earth Orbit (LEO), it has been enhanced with professional-grade software engineering practices and advanced machine learning capabilities.

### Mission Context
- **Mission**: Space Tug for Active Debris Removal (ADR) in LEO
- **Orbit**: Sun-Synchronous Orbit at 500-700 km altitude
- **Purpose**: Validate power budgets, solar array sizing, battery management, and SDR communication for autonomous space operations

---

## Key Features

### 🛠️ Core Aerospace Functions
- **Power Budget Analysis**: Comprehensive calculation of sunlight/eclipse power requirements
- **Solar Array Sizing**: Multi-junction GaAs cell optimization with degradation modeling
- **Battery Management**: Li-Ion sizing with depth-of-discharge considerations
- **Thermal Analysis**: Stefan-Boltzmann radiative heat dissipation modeling
- **MPPT Efficiency**: Maximum Power Point Tracking performance analysis

### 🤖 AI/ML Capabilities
- **Predictive Analytics**
  - LSTM Solar Irradiance Forecaster
  - Physics-Informed Battery Degradation Predictor
  - Autoencoder-based Anomaly Detection

- **Optimization Algorithms**
  - Genetic Algorithm for Power Allocation
  - Particle Swarm Optimization for MPPT
  - Bayesian Optimization for Hyperparameter Tuning

- **Autonomous Systems**
  - PPO Reinforcement Learning Agent for Power Management
  - Fuzzy Logic Controller for Thermal Regulation
  - Real-time Fault Detection and Response

### 📊 Visualization Dashboard
- Interactive Plotly-based monitoring
- Real-time power system telemetry
- 3D orbital visualization
- MPPT efficiency analysis
- Anomaly detection dashboard

---

## Project Structure

```
H2Z_Satellite/
├── src/
│   ├── core/
│   │   ├── power_budget.py          # Core aerospace calculations
│   │   ├── thermal_analysis.py       # Thermal modeling
│   │   └── mppt_analysis.py          # MPPT efficiency
│   ├── ml_models/
│   │   ├── predictive/
│   │   │   ├── solar_forecaster.py   # LSTM forecasting
│   │   │   ├── battery_predictor.py  # Degradation prediction
│   │   │   └── anomaly_detector.py   # Autoencoder
│   │   ├── optimization/
│   │   │   ├── genetic_optimizer.py # GA optimization
│   │   │   └── pso_optimizer.py     # PSO MPPT
│   │   └── autonomous/
│   │       └── rl_agent.py           # PPO RL agent
│   ├── visualization/
│   │   ├── dashboard.py              # Interactive dashboard
│   │   └── plots.py                  # Visualization utilities
│   ├── api/
│   │   ├── routes.py                 # FastAPI routes
│   │   └── schemas.py                # Pydantic schemas
│   ├── config/
│   │   ├── settings.py               # Configuration
│   │   └── hyperparameters.yaml      # ML hyperparameters
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── ml/
│   └── main.py                       # Entry point
├── data/
│   ├── raw/                          # Raw telemetry data
│   ├── processed/                     # Processed datasets
│   └── models/                        # Saved ML models
├── docs/                              # Documentation
├── notebooks/                        # Jupyter notebooks
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup
├── Dockerfile                         # Containerization
└── README.md                         # This file
```

---

## Quick Start

### Prerequisites
- Python 3.10 or higher
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/H2Z-Satellite-Power.git
cd H2Z-Satellite-Power

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the main application
python src/main.py
```

---

## Core Aerospace Functions

### Power Budget Calculation

```python
from src.core.power_budget import (
    SatelliteSystem,
    PowerBudgetCalculator,
    OrbitalParameters,
    SolarArraySpecifications,
    BatterySpecifications,
    SubsystemPower
)

# Initialize satellite system
satellite = SatelliteSystem("H2Z_LEO_Space_Tug")

# Run complete analysis
report = satellite.run_complete_analysis()

# Access power budget results
power = report['power_analysis']
print(f"Solar Array Power: {power['array_requirements']['required_power_watts']:.2f} W")
print(f"Battery Capacity: {power['battery_requirements']['capacity_wh']:.2f} Wh")
print(f"Power Margin: {power['array_requirements']['power_margin_percent']:.1f}%")
```

### MPPT Analysis

```python
from src.core.power_budget import MPPTAnalyzer

mppt = MPPTAnalyzer(solar_array)
results = mppt.power_gain_analysis(years=3)

print(f"Initial Efficiency: {results['mppt_efficiency'][0]*100:.1f}%")
print(f"Final Efficiency (3yr): {results['mppt_efficiency'][-1]*100:.1f}%")
```

### Thermal Analysis

```python
from src.core.power_budget import ThermalAnalyzer

thermal = ThermalAnalyzer(emissivity=0.98, radiative_area=8.0)
results = thermal.thermal_analysis(subsystem_dissipation)

print(f"Equilibrium Temperature: {results['equilibrium_temperature_c']:.1f}°C")
print(f"Total Dissipation: {results['subsystem_dissipation_watts']:.2f} W")
```

---

## AI/ML Capabilities

### Solar Irradiance Forecasting (LSTM)

```python
from src.ml_models.predictive.solar_forecaster import LSTMSolarForecaster, TrainingConfig

# Configure model
config = TrainingConfig(
    sequence_length=48,
    forecast_horizon=6,
    epochs=100,
    hidden_dim=64
)

forecaster = LSTMSolarForecaster(config)

# Train on historical data
forecaster.train(solar_data_dataframe)

# Generate forecast
forecast = forecaster.predict(historical_irradiance)
```

### Anomaly Detection (Autoencoder)

```python
from src.ml_models.predictive.anomaly_detector import AnomalyDetector

detector = AnomalyDetector(encoding_dim=8)

# Train on normal operation data
detector.train(normal_power_data)

# Detect anomalies
anomalies, errors = detector.detect(new_power_data)
```

### Genetic Algorithm Optimization

```python
from src.ml_models.optimization.genetic_optimizer import GeneticAlgorithmOptimizer

def objective_function(solution):
    # Your optimization objective
    return -solution[0]**2 - solution[1]**2

ga = GeneticAlgorithmOptimizer(
    population_size=100,
    generations=200
)

best_solution, best_fitness, history = ga.optimize(
    objective_function,
    bounds=[(-10, 10), (-10, 10)],
    maximize=True
)
```

### PPO Reinforcement Learning Agent

```python
from src.ml_models.autonomous.rl_agent import PPOAgent, RLConfig

config = RLConfig(
    learning_rate=3e-4,
    gamma=0.99,
    clip_epsilon=0.2,
    num_training_steps=50000
)

agent = PPOTrainingAgent(config)

# Train the agent
history = agent.train(num_episodes=1000)

# Evaluate
metrics = agent.evaluate(num_episodes=10)
print(f"Mean Reward: {metrics['mean_reward']:.2f}")
```

---

## Visualization Dashboard

### Run Interactive Dashboard

```bash
# Generate sample visualizations
python -c "
from src.visualization.dashboard import PowerSystemDashboard
dashboard = PowerSystemDashboard()
# ... populate with data ...
fig = dashboard.create_power_monitor()
fig.write_html('power_dashboard.html')
"
```

### Dashboard Components

| Component | Description |
|-----------|-------------|
| Solar Power Monitor | Real-time solar generation visualization |
| Battery SOC Gauge | State of charge with color-coded indicators |
| Power Demand Trends | Historical consumption patterns |
| Subsystem Distribution | Pie chart of power allocation |
| Thermal Profile | Temperature over time |
| Eclipse Phase Tracker | Day/night cycle visualization |

---

## Requirements

### Core Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
matplotlib>=3.7.0
plotly>=5.15.0
```

### AI/ML Dependencies

```
torch>=2.0.0
tensorflow>=2.13.0
scikit-learn>=1.3.0
xgboost>=1.7.0
gymnasium>=0.28.0
stable-baselines3>=2.0.0
optuna>=3.3.0
```

### Development Dependencies

```
pytest>=7.4.0
black>=23.0.0
mypy>=1.4.0
pre-commit>=3.3.0
sphinx>=7.0.0
```

See [requirements.txt](requirements.txt) for complete dependency list.

---

## Usage

### Basic Analysis

```python
# Run comprehensive analysis
python src/main.py

# Select option 1 for power budget analysis
# Select option 4 for complete demonstration
```

### Generate Reports

```python
from src.core.power_budget import SatelliteSystem

satellite = SatelliteSystem("My_Satellite")
satellite.export_report("my_analysis_report.json")
```

### Create Visualizations

```python
from src.visualization.dashboard import PowerSystemDashboard

dashboard = PowerSystemDashboard()
# ... populate data ...
fig = dashboard.create_power_monitor()
fig.write_html("dashboard.html")
```

---

## Output Examples

### Power Budget Summary

```
============================================================
POWER BUDGET SUMMARY
============================================================

☀️ Solar Array Requirements:
   • Required Power: 851.61 W
   • Required Area: 2.733 m²
   • Required Area (EOL): 2.92 m²
   • Power Margin: 21.8%

🔋 Battery Requirements:
   • Capacity: 163.22 Wh (5.83 Ah)
   • Mass: 2.04 kg
   • Max Discharge: 178.93 W

🌡️ Thermal Analysis:
   • Equilibrium Temperature: 63.0 °C
   • Total Dissipation: 191.89 W
   • Thermal Margin: 237.0 K
```

### MPPT Efficiency Analysis

```
MPPT EFFICIENCY ANALYSIS (3 Years)

Initial MPPT Efficiency: 97.0%
Final MPPT Efficiency (3 years): 94.0%

Initial Power Advantage: 12.0%
Final Power Advantage (3 years): 10.5%

Annual Energy Savings: ~2,920 kWh
Total 3-Year Savings: ~8,760 kWh
```

---

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- [API Documentation](docs/api.md) - Complete API reference
- [ML Models Guide](docs/ml_models.md) - AI/ML model documentation
- [User Guide](docs/user_guide.md) - Getting started guide
- [Examples](docs/examples/) - Usage examples

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Air University Islamabad** - Academic institution
- **Project Supervisors** - Prof. Dr. Ali Sarosh, Prof. Dr. Akram Rashid
- **Open Source Community** - For the amazing tools and libraries

---

<div align="center">

**Built with ❤️ for the future of space exploration**

*This project demonstrates professional software engineering practices combined with aerospace domain expertise and modern AI/ML techniques.*

</div>

