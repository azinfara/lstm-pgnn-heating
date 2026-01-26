# LSTM-PGNN: Physics-Guided Neural Network for Building Heating Prediction

A hybrid deep learning model that combines physics-based thermodynamic calculations with LSTM neural networks to predict hourly heating demand in buildings. This approach leverages domain knowledge from building physics while utilizing data-driven learning to capture complex patterns in heating behavior.

## Overview

Traditional building energy models rely either purely on physics-based simulations (computationally expensive, requires detailed inputs) or purely data-driven approaches (ignores physical laws, requires extensive data). This LSTM-PGNN model bridges both approaches by:

- Using physics-based calculations as a foundation
- Learning correction factors to adjust physics predictions
- Applying LSTM networks to capture temporal dynamics and residual patterns

The result is a model that generalizes better with less data, respects physical constraints, and provides interpretable predictions.

## Model Architecture

The LSTM-PGNN (version 2.2 baseline) combines three key components:

### 1. Static Encoder

Processes building characteristics through a 3-layer feedforward network to learn four correction factors:
- **k_env**: Environmental/envelope correction (walls, insulation, infiltration)
- **k_int**: Internal gains correction (occupancy, equipment)
- **k_sol**: Solar gains correction (windows, orientation, shading)
- **k_vent**: Ventilation correction (airflow, heat recovery)

Each correction factor is constrained to positive values using softplus activation with a minimum of 0.3.

### 2. Physics-Based Calculation Module

Computes baseline heating demand using thermodynamic principles:

```
Q_envelope = k_env × UA_envelope × ΔT
Q_infiltration = k_env × ṁ_infiltration × cp × ΔT
Q_ventilation = k_vent × ṁ_ventilation × cp × ΔT
Q_internal = k_int × internal_gains
Q_solar = k_sol × g_value × A_window × radiation

Q_physics = max(0, Q_envelope + Q_infiltration + Q_ventilation - Q_internal - Q_solar)
```

Key physics features:
- Dynamic ventilation based on occupancy schedules
- Natural infiltration calculated from building airtightness (Q50)
- Solar gains modulated by window properties and shading
- Temperature-dependent heating activation

### 3. LSTM Correction Layer

A single-layer LSTM (64 hidden units) captures temporal patterns and residual errors:

```
δ_LSTM = 5.0 × tanh(LSTM_output)
Q_pred = Q_physics × (1 + δ_LSTM)
```

The correction is capped at ±5× to prevent unrealistic adjustments while allowing meaningful pattern learning.

### Training Strategy

The model uses a **peak-weighted loss function** that prioritizes accurate predictions during high-demand periods:
- Base weight (1×) for loads < 100 kW
- Progressive weighting up to 4× for loads > 250 kW
- Regularization terms for LSTM corrections (λ=1e-3) and k-factors (λ=1e-3)

This ensures the model performs well across the full range of operating conditions while excelling at peak load prediction.

## Repository Structure

```
├── data/                                    # Processed data files
│   ├── mm_data_pinn_with_schedule/         # Memory-mapped data arrays
│   ├── processed_Heating_chunks/           # Processed heating data
│   ├── schedule/                           # Occupancy schedules
│   ├── weather/                            # Weather data (TRY files)
│   ├── metadata_pinn_with_schedule.pkl     # Dataset metadata
│   ├── ohe_encoder_pinn_with_schedule.pkl  # One-hot encoders
│   ├── scaler_static_pinn_with_schedule.pkl # Static feature scaler
│   ├── scaler_ts_pinn_with_schedule.pkl    # Time series scaler
│   ├── scaler_y_pinn_with_schedule.pkl     # Target scaler
│   ├── scalers_with_schedule.pkl           # Combined scalers
│   ├── test_runids_pinn_stratified_with_schedule.pkl
│   ├── train_runids_pinn_stratified_with_schedule.pkl
│   ├── training_package_with_schedule.pkl  # Complete training package
│   └── val_runids_pinn_stratified_with_schedule.pkl
├── model/                                   # Pre-trained model
│   ├── lstm_pgnn_v2_2_baseline.pt
│   └── model_config.json                   # Model hyperparameters
├── src/                                     # Source code
│   ├── data_utils.py                       # Data loading and preprocessing
│   └── model.py                            # Model architecture
├── requirements.txt                         # Dependencies
├── .gitignore
└── README.md
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.12+
- NumPy, Pandas, Scikit-learn, Joblib

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/lstm-pgnn-heating.git
cd lstm-pgnn-heating

# Install dependencies
pip install -r requirements.txt
```

## Dataset

### Sample Data
The `sample_data/` folder contains a curated subset for demonstration and testing:
- **10 buildings** across 6 building types
- **Sample parquet file** with preprocessed features
- **Preprocessed data** ready for model inference
- Includes all necessary scaler files and metadata

The sample data maintains the statistical properties of the full dataset while being suitable for public sharing and quick testing.

**Building types in sample**:
- Accommodation
- Apartment
- Commercial  
- Educational
- Hospital
- Office

### Full Dataset Structure
The complete `data/` folder includes preprocessed data files ready for model training:

**Memory-mapped arrays** (in `mm_data_pinn_with_schedule/`):
- Efficient storage for large-scale building simulation data
- Enables training on datasets too large for RAM
- Full dataset: 5,764 building simulation runs

**Weather data** (in `weather/`):
Four Finnish climate scenarios covering different time periods and climate projections:
- **Vantaa-TRY2020**: Test Reference Year 2020 (-25.0°C to 30.0°C), 1,465 runs
- **Vantaa2018_FMI_Measured**: Measured 2018 data (-23.9°C to 31.7°C), 1,390 runs
- **Vantaa_2050_ver2020_RCP45**: Future climate projection RCP4.5 (-20.6°C to 31.1°C), 1,390 runs
- **Vantaa_TRY2012**: Test Reference Year 2012 (-20.6°C to 28.8°C), 1,519 runs

**Occupancy schedules** (in `schedule/`):
Building type-specific occupancy patterns (25 unique schedules):
- Accommodation: 2 schedules (mean occupancy: 0.26-0.65)
- Apartment: 6 schedules (mean occupancy: 0.35-1.00)
- Commercial: 4 schedules (mean occupancy: 0.32-0.60)
- Educational: 2 schedules (mean occupancy: 0.25-0.38)
- Hospital: 3 schedules (mean occupancy: 0.48-0.74)
- Office: 2 schedules (mean occupancy: 0.28-0.39)

### Data Loading

```python
import joblib

# Load the complete training package
pkg = joblib.load("data/training_package_with_schedule.pkl")

# Access split information
train_runids = pkg["train_runids"]
val_runids = pkg["val_runids"]
test_runids = pkg["test_runids"]

# Load scalers for preprocessing
scalers = joblib.load("data/scalers_with_schedule.pkl")
ts_scaler = scalers["ts_scaler"]        # MinMaxScaler for time series
static_scaler = scalers["static_scaler"] # StandardScaler for static features
y_scaler = scalers["y_scaler"]          # StandardScaler for target
ohe_encoders = scalers["ohe_encoders"]  # OneHotEncoders for categories
```

### Input Features

#### Static Features (Building Characteristics)

| Feature | Description | Range | Typical Range (5%-95%) | Units |
|---------|-------------|-------|------------------------|-------|
| Total Area | Building floor area | 328.80 - 87,808.10 | 1,220.50 - 35,729.00 | m² |
| Total Volume | Building volume | 847.80 - 377,453.50 | 4,519.50 - 178,861.40 | m³ |
| HRUefficiency | Heat recovery unit efficiency | 0.55 - 0.85 | 0.55 - 0.85 | - |
| Total int gain | Internal heat gains | 3.93 - 48.94 | 9.07 - 33.11 | W/m² |
| ExtWin U | Window U-value | 0.94 - 1.39 | 0.94 - 1.39 | W/(m²·K) |
| ExtWin g | Window solar factor (g-value) | 0.00 - 0.51 | 0.00 - 0.51 | - |
| WWR | Window-to-wall ratio | 0.05 - 0.88 | 0.10 - 0.62 | - |
| Q50 | Air tightness at 50 Pa | 1.00 - 5.00 | 1.00 - 5.00 | 1/h |
| ShadingRate | External shading factor | 0.00 - 0.80 | 0.00 - 0.80 | - |
| Number_of_AHU | Number of air handling units | 1 - 72 | 2 - 28 | - |
| Max_airflow | Maximum airflow rate | 25 - 39,858 | 50 - 13,229 | dm³/h |
| ExtWall U | Wall U-value | 0.12 - 1.07 | 0.15 - 0.91 | W/(m²·K) |
| ExtWall Area | External wall area | 149 - 38,024 | 603 - 10,586 | m² |
| Type | Building type | 6 categories: Accommodation, Apartment, Commercial, Educational, Hospital, Office | - | - |
| Orientation | Building orientation | 4 categories: 0°, 90°, 180°, 270° | - | - |

#### Time Series Features (Hourly, 8760 timesteps)
- **Outdoor Temperature** (°C): Driving force for heating demand
- **Solar Radiation** (W/m²): Influences passive solar gains
- **Outdoor Humidity** (gr/kg): Affects ventilation load
- **Occupancy Schedule** (0-1): Modulates ventilation and internal gains
- **Temporal Features**: Hour of day, day of week, month, is_weekend

#### Target Variable
- **Hourly Heating Demand** (kW): Total building heating load

## Usage

### Loading the Pre-trained Model

```python
import torch
import json
from src.model import LSTMPINNHeatingV2_2_Enhanced

# Load model configuration
with open('model/model_config.json', 'r') as f:
    config = json.load(f)

# Initialize model architecture
model = LSTMPINNHeatingV2_2_Enhanced(
    ts_dim=config['ts_dim'],
    static_dim=config['static_dim'],
    hidden_lstm=config['hidden_lstm'],
    num_layers=config['num_layers'],
    bidirectional=config['bidirectional'],
    dropout=config['dropout'],
    setpoint=config['setpoint'],
    correction_cap=config['correction_cap'],
    exposure_factor=config['exposure_factor'],
    ua_multiplier=config['ua_multiplier'],
    schedule_weight=config['schedule_weight']
)

# Load trained weights
checkpoint = torch.load('model/lstm_pgnn_v2_2_baseline.pt', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.eval()
```

### Data Preparation

```python
import joblib
from src.data_preparation import prepare_input_data

# Load preprocessing scalers
scalers = joblib.load('data/scalers_with_schedule.pkl')

# Your building data should include:
# - Static features: (n_buildings, 27) - includes numeric + one-hot encoded categories
# - Time series: (n_buildings, 8760, 8) - hourly weather and occupancy

# Apply preprocessing
input_data = prepare_input_data(
    static_features=building_characteristics,
    time_series_features=hourly_data,
    scalers=scalers
)
```

### Understanding the Training Package

The `training_package_with_schedule.pkl` contains everything needed for training:

```python
import joblib

pkg = joblib.load("data/training_package_with_schedule.pkl")

# Dataset splits (RunIDs)
train_runids = pkg["train_runids"]  # Training set building runs
val_runids = pkg["val_runids"]      # Validation set
test_runids = pkg["test_runids"]    # Test set

# Feature dimensions
n_features = pkg["n_features"]      # Total features: 27 + 8 = 35
n_static = pkg["n_static"]          # Static features: 27 (13 numeric + 14 OHE)
n_ts = pkg["n_ts"]                  # Time series features: 8
n_num = pkg["n_num"]                # Numeric static features: 13

# Column names
time_series_cols = pkg["time_series_cols"]  # 8 TS feature names
static_num_cols = pkg["static_num_cols"]    # 13 numeric feature names
static_cat_cols = pkg["static_cat_cols"]    # 2 categorical features

# Data mappings
run_to_rows = pkg["run_to_rows"]           # Maps RunID to data row indices
building_of_run = pkg["building_of_run"]    # Maps RunID to building ID
building_type = pkg["building_type"]        # Maps building ID to type

# Memory-mapped data location
mm_dir = pkg["mm_dir"]              # Path to memory-mapped arrays
total_rows = pkg["total_rows"]      # Total data rows (runs × 8760)
```

### Running Predictions

```python
import torch

# Prepare batch dictionary
batch = {
    'X_ts': torch.tensor(time_series_scaled, dtype=torch.float32),  # (B, 8760, 8)
    'static': torch.tensor(static_scaled, dtype=torch.float32),     # (B, 27)
}

# Generate predictions
with torch.no_grad():
    outputs = model(batch)
    
    # Get hourly heating predictions in kW
    Q_pred_kW = outputs['Q_pred_kW']  # Shape: (B, 8760, 1)
    
    # Optional: Access intermediate outputs
    Q_physics_kW = outputs['Q_phys_kW']  # Physics-based baseline
    delta = outputs['delta']  # LSTM corrections (typically ±20%)
    
    # Get learned correction factors for each building
    k_env = outputs['k_env']    # Envelope correction factor
    k_int = outputs['k_int']    # Internal gains correction
    k_sol = outputs['k_sol']    # Solar gains correction
    k_vent = outputs['k_vent']  # Ventilation correction

# Convert to numpy for analysis
predictions = Q_pred_kW.cpu().numpy()  # Shape: (B, 8760, 1)
```

### Computing Annual Metrics

```python
import numpy as np

# Calculate annual heating energy (MWh)
annual_heating_MWh = predictions.sum(axis=1) / 1000  # kWh -> MWh

# Calculate heating intensity (kWh/m²)
heating_intensity = (predictions.sum(axis=1).squeeze() / building_area) * 1000

# Calculate peak load (kW)
peak_load = predictions.max(axis=1)

print(f"Annual Heating: {annual_heating_MWh[0]:.1f} MWh")
print(f"Heating Intensity: {heating_intensity[0]:.1f} kWh/m²")
print(f"Peak Load: {peak_load[0]:.1f} kW")
```

## Model Configuration

The `model_config.json` file contains all model hyperparameters:

```json
{
  "name": "Baseline",
  "ts_dim": 8,
  "static_dim": 27,
  "hidden_lstm": 64,
  "num_layers": 1,
  "bidirectional": false,
  "dropout": 0.0,
  "setpoint": 22.0,
  "correction_cap": 5.0,
  "exposure_factor": 20.0,
  "ua_multiplier": 1.2,
  "schedule_weight": 1.0
}
```

**Parameter Descriptions:**
- `ts_dim`: Number of time series input features (8)
- `static_dim`: Total static features including one-hot encoded categories (27)
- `hidden_lstm`: LSTM hidden layer size (64 for baseline)
- `num_layers`: Number of LSTM layers (1)
- `bidirectional`: Whether to use bidirectional LSTM (false)
- `dropout`: Dropout rate (0.0 for baseline)
- `setpoint`: Heating setpoint temperature in °C (22.0)
- `correction_cap`: Maximum LSTM correction factor (5.0)
- `exposure_factor`: Infiltration exposure factor (20.0)
- `ua_multiplier`: Envelope UA adjustment multiplier (1.2)
- `schedule_weight`: Occupancy schedule weight (1.0)

## Model Performance

### Baseline Configuration
- **Architecture**: LSTM-PGNN v2.2 Baseline
- **LSTM Hidden Size**: 64
- **Number of Layers**: 1
- **Bidirectional**: False
- **Total Parameters**: ~45,000
- **Training**: AdamW optimizer, lr=3e-4, peak-weighted loss

### Overall Test Set Performance

Performance on held-out test buildings (1,159 runs across 10 buildings):

- **RMSE**: 49.74 kW
- **MAE**: 28.83 kW
- **R²**: 0.82

The model demonstrates strong predictive performance across diverse building types and scales, with consistent accuracy for early-stage energy planning applications.

## Key Features

✅ **Physics-informed predictions** - Respects thermodynamic principles and physical constraints  
✅ **Temporal modeling** - Captures hourly heating dynamics with LSTM  
✅ **Interpretable corrections** - Four learned adjustment factors with physical meaning  
✅ **Generalizes across building types** - Trained on diverse building stock  
✅ **Peak-load optimized** - Weighted loss ensures accuracy during high-demand periods  
✅ **Fast inference** - Single forward pass for 8760-hour predictions (~0.1s per building)  
✅ **Robust to input variations** - Handles wide range of building parameters  
✅ **Lightweight architecture** - Only ~45k parameters, suitable for deployment  

## Limitations

- Trained on simulation data from Finnish climate (TRY weather files); validation recommended for other climates
- Performance varies by building type and quality of input data
- Requires complete hourly weather and occupancy schedule data (8760 timesteps)
- Best suited for buildings within the training parameter ranges (see typical ranges in dataset section)
- Assumes constant setpoint temperature (22°C); dynamic setpoints not modeled
- It should be used during the early stage design, not for actual sizing of the equipments
- Occupancy schedules must be provided as input (not predicted)

## Model Interpretability

### Understanding Correction Factors

The four k-factors provide physical insight into model predictions:

- **k_env > 1**: Building loses more heat than basic physics predicts (e.g., thermal bridges, air leakage)
- **k_env < 1**: Better insulation or less heat loss than expected
- **k_int > 1**: More internal gains than estimated (e.g., additional equipment)
- **k_int < 1**: Internal gains are overestimated or less effective
- **k_sol > 1**: More effective solar gains (e.g., optimal orientation)
- **k_sol < 1**: Solar gains blocked by factors not captured (e.g., shading, neighboring buildings)
- **k_vent > 1**: Ventilation load higher than expected (e.g., poor heat recovery, door openings)
- **k_vent < 1**: More efficient ventilation than assumed

### LSTM Corrections

The δ values show where data-driven learning adds value beyond physics:
- **Small corrections (|δ| < 0.1)**: Physics model is highly accurate
- **Moderate corrections (0.1 < |δ| < 0.3)**: Normal adjustment for unmodeled effects
- **Large corrections (|δ| > 0.5)**: May indicate missing physics, poor parameters, or unique building behavior

