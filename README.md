\# LSTM-PGNN: Physics-Guided Neural Network for Building Heating Prediction



A hybrid deep learning model that combines physics-based thermodynamic calculations with LSTM neural networks to predict hourly heating demand in buildings. This approach leverages domain knowledge from building physics while utilizing data-driven learning to capture complex patterns in heating behavior.



\## Overview



Traditional building energy models rely either purely on physics-based simulations (computationally expensive, requires detailed inputs) or purely data-driven approaches (ignores physical laws, requires extensive data). This LSTM-PGNN model bridges both approaches by:



\- Using physics-based calculations as a foundation

\- Learning correction factors to adjust physics predictions

\- Applying LSTM networks to capture temporal dynamics and residual patterns



The result is a model that generalizes better with less data, respects physical constraints, and provides interpretable predictions.



\## Model Architecture



The LSTM-PGNN (version 2.2 baseline) combines three key components:



\### 1. Static Encoder

Processes building characteristics through a 3-layer feedforward network to learn four correction factors:

\- \*\*k\_env\*\*: Environmental/envelope correction (walls, insulation, infiltration)

\- \*\*k\_int\*\*: Internal gains correction (occupancy, equipment)

\- \*\*k\_sol\*\*: Solar gains correction (windows, orientation, shading)

\- \*\*k\_vent\*\*: Ventilation correction (airflow, heat recovery)



Each correction factor is constrained to positive values using softplus activation with a minimum of 0.3.



\### 2. Physics-Based Calculation Module

Computes baseline heating demand using thermodynamic principles:



```

Q\_envelope = k\_env × UA\_envelope × ΔT

Q\_infiltration = k\_env × ṁ\_infiltration × cp × ΔT

Q\_ventilation = k\_vent × ṁ\_ventilation × cp × ΔT

Q\_internal = k\_int × internal\_gains

Q\_solar = k\_sol × g\_value × A\_window × radiation



Q\_physics = max(0, Q\_envelope + Q\_infiltration + Q\_ventilation - Q\_internal - Q\_solar)

```



Key physics features:

\- Dynamic ventilation based on occupancy schedules

\- Natural infiltration calculated from building airtightness (Q50)

\- Solar gains modulated by window properties and shading

\- Temperature-dependent heating activation



\### 3. LSTM Correction Layer

A single-layer LSTM (64 hidden units) captures temporal patterns and residual errors:



```

δ\_LSTM = 5.0 × tanh(LSTM\_output)

Q\_pred = Q\_physics × (1 + δ\_LSTM)

```



The correction is capped at ±5× to prevent unrealistic adjustments while allowing meaningful pattern learning.



\### Training Strategy

The model uses a \*\*peak-weighted loss function\*\* that prioritizes accurate predictions during high-demand periods:

\- Base weight (1×) for loads < 100 kW

\- Progressive weighting up to 4× for loads > 250 kW

\- Regularization terms for LSTM corrections (λ=1e-3) and k-factors (λ=1e-3)



This ensures the model performs well across the full range of operating conditions while excelling at peak load prediction.



\## Repository Structure



```

├── data/                                    # Processed data files

│   ├── mm\_data\_pinn\_with\_schedule/         # Memory-mapped data arrays

│   ├── processed\_Heating\_chunks/           # Processed heating data

│   ├── schedule/                           # Occupancy schedules

│   ├── weather/                            # Weather data

│   ├── metadata\_pinn\_with\_schedule.pkl     # Dataset metadata

│   ├── ohe\_encoder\_pinn\_with\_schedule.pkl  # One-hot encoders

│   ├── scaler\_static\_pinn\_with\_schedule.pkl # Static feature scaler

│   ├── scaler\_ts\_pinn\_with\_schedule.pkl    # Time series scaler

│   ├── scaler\_y\_pinn\_with\_schedule.pkl     # Target scaler

│   ├── scalers\_with\_schedule.pkl           # Combined scalers

│   ├── test\_runids\_pinn\_stratified\_with\_schedule.pkl

│   ├── train\_runids\_pinn\_stratified\_with\_schedule.pkl

│   ├── training\_package\_with\_schedule.pkl  # Complete training package

│   └── val\_runids\_pinn\_stratified\_with\_schedule.pkl

├── model/                                   # Pre-trained model

│   ├── lstm\_pgnn\_v2\_2\_baseline.pt

│   └── model\_config.json                   # Model hyperparameters

├── src/                                     # Source code

│   ├── data\_utils.py                       # Data loading and preprocessing

│   └── model.py                            # Model architecture

├── requirements.txt                         # Dependencies

├── .gitignore

└── README.md

```



\## Installation



\### Requirements

\- Python 3.8+

\- PyTorch 1.12+

\- NumPy, Pandas, Scikit-learn, Joblib



\### Setup



```bash

\# Clone the repository

git clone https://github.com/azinfara/lstm-pgnn-heating.git

cd lstm-pgnn-heating



\# Install dependencies

pip install -r requirements.txt

```



\## Dataset



\### Sample Data

The `sample\_data/` folder contains a curated subset for demonstration and testing:

\- \*\*10 buildings\*\* across 6 building types

\- \*\*Sample parquet file\*\* with preprocessed features

\- \*\*Preprocessed data\*\* ready for model inference

\- Includes all necessary scaler files and metadata



The sample data maintains the statistical properties of the full dataset while being suitable for public sharing and quick testing.



\*\*Building types in sample\*\*:

\- Accommodation

\- Apartment

\- Commercial  

\- Educational

\- Hospital

\- Office



\### Full Dataset Structure

The complete `data/` folder includes preprocessed data files ready for model training:



\*\*Memory-mapped arrays\*\* (in `mm\_data\_pinn\_with\_schedule/`):

\- Efficient storage for large-scale building simulation data

\- Enables training on datasets too large for RAM

\- Full dataset: 5,764 building simulation runs



\*\*Weather data\*\* (in `weather/`):

Four Finnish climate scenarios covering different time periods and climate projections:

\- \*\*Vantaa-TRY2020\*\*: Test Reference Year 2020 (-25.0°C to 30.0°C), 1,465 runs

\- \*\*Vantaa2018\_FMI\_Measured\*\*: Measured 2018 data (-23.9°C to 31.7°C), 1,390 runs

\- \*\*Vantaa\_2050\_ver2020\_RCP45\*\*: Future climate projection RCP4.5 (-20.6°C to 31.1°C), 1,390 runs

\- \*\*Vantaa\_TRY2012\*\*: Test Reference Year 2012 (-20.6°C to 28.8°C), 1,519 runs



\*\*Occupancy schedules\*\* (in `schedule/`):

Building type-specific occupancy patterns (25 unique schedules):

\- Accommodation: 2 schedules (mean occupancy: 0.26-0.65)

\- Apartment: 6 schedules (mean occupancy: 0.35-1.00)

\- Commercial: 4 schedules (mean occupancy: 0.32-0.60)

\- Educational: 2 schedules (mean occupancy: 0.25-0.38)

\- Hospital: 3 schedules (mean occupancy: 0.48-0.74)

\- Office: 2 schedules (mean occupancy: 0.28-0.39)



\### Data Loading



```python

import joblib



\# Load the complete training package

pkg = joblib.load("data/training\_package\_with\_schedule.pkl")



\# Access split information

train\_runids = pkg\["train\_runids"]

val\_runids = pkg\["val\_runids"]

test\_runids = pkg\["test\_runids"]



\# Load scalers for preprocessing

scalers = joblib.load("data/scalers\_with\_schedule.pkl")

ts\_scaler = scalers\["ts\_scaler"]        # MinMaxScaler for time series

static\_scaler = scalers\["static\_scaler"] # StandardScaler for static features

y\_scaler = scalers\["y\_scaler"]          # StandardScaler for target

ohe\_encoders = scalers\["ohe\_encoders"]  # OneHotEncoders for categories

```



\### Input Features



\#### Static Features (Building Characteristics)



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

| Number\_of\_AHU | Number of air handling units | 1 - 72 | 2 - 28 | - |

| Max\_airflow | Maximum airflow rate | 25 - 39,858 | 50 - 13,229 | dm³/h |

| ExtWall U | Wall U-value | 0.12 - 1.07 | 0.15 - 0.91 | W/(m²·K) |

| ExtWall Area | External wall area | 149 - 38,024 | 603 - 10,586 | m² |

| Type | Building type | 6 categories: Accommodation, Apartment, Commercial, Educational, Hospital, Office | - | - |

| Orientation | Building orientation | 4 categories: 0°, 90°, 180°, 270° | - | - |



\#### Time Series Features (Hourly, 8760 timesteps)

\- \*\*Outdoor Temperature\*\* (°C): Driving force for heating demand

\- \*\*Solar Radiation\*\* (W/m²): Influences passive solar gains

\- \*\*Outdoor Humidity\*\* (gr/kg): Affects ventilation load

\- \*\*Occupancy Schedule\*\* (0-1): Modulates ventilation and internal gains

\- \*\*Temporal Features\*\*: Hour of day, day of week, month, is\_weekend



\#### Target Variable

\- \*\*Hourly Heating Demand\*\* (kW): Total building heating load



\## Usage



\### Loading the Pre-trained Model



```python

import torch

import json

from src.model import LSTMPINNHeatingV2\_2\_Enhanced



\# Load model configuration

with open('model/model\_config.json', 'r') as f:

&nbsp;   config = json.load(f)



\# Initialize model architecture

model = LSTMPINNHeatingV2\_2\_Enhanced(

&nbsp;   ts\_dim=config\['ts\_dim'],

&nbsp;   static\_dim=config\['static\_dim'],

&nbsp;   hidden\_lstm=config\['hidden\_lstm'],

&nbsp;   num\_layers=config\['num\_layers'],

&nbsp;   bidirectional=config\['bidirectional'],

&nbsp;   dropout=config\['dropout'],

&nbsp;   setpoint=config\['setpoint'],

&nbsp;   correction\_cap=config\['correction\_cap'],

&nbsp;   exposure\_factor=config\['exposure\_factor'],

&nbsp;   ua\_multiplier=config\['ua\_multiplier'],

&nbsp;   schedule\_weight=config\['schedule\_weight']

)



\# Load trained weights

checkpoint = torch.load('model/lstm\_pgnn\_v2\_2\_baseline.pt', map\_location='cpu')

model.load\_state\_dict(checkpoint\['state\_dict'])

model.eval()

```



\### Data Preparation



```python

import joblib

from src.data\_preparation import prepare\_input\_data



\# Load preprocessing scalers

scalers = joblib.load('data/scalers\_with\_schedule.pkl')



\# Your building data should include:

\# - Static features: (n\_buildings, 27) - includes numeric + one-hot encoded categories

\# - Time series: (n\_buildings, 8760, 8) - hourly weather and occupancy



\# Apply preprocessing

input\_data = prepare\_input\_data(

&nbsp;   static\_features=building\_characteristics,

&nbsp;   time\_series\_features=hourly\_data,

&nbsp;   scalers=scalers

)

```



\### Understanding the Training Package



The `training\_package\_with\_schedule.pkl` contains everything needed for training:



```python

import joblib



pkg = joblib.load("data/training\_package\_with\_schedule.pkl")



\# Dataset splits (RunIDs)

train\_runids = pkg\["train\_runids"]  # Training set building runs

val\_runids = pkg\["val\_runids"]      # Validation set

test\_runids = pkg\["test\_runids"]    # Test set



\# Feature dimensions

n\_features = pkg\["n\_features"]      # Total features: 27 + 8 = 35

n\_static = pkg\["n\_static"]          # Static features: 27 (13 numeric + 14 OHE)

n\_ts = pkg\["n\_ts"]                  # Time series features: 8

n\_num = pkg\["n\_num"]                # Numeric static features: 13



\# Column names

time\_series\_cols = pkg\["time\_series\_cols"]  # 8 TS feature names

static\_num\_cols = pkg\["static\_num\_cols"]    # 13 numeric feature names

static\_cat\_cols = pkg\["static\_cat\_cols"]    # 2 categorical features



\# Data mappings

run\_to\_rows = pkg\["run\_to\_rows"]           # Maps RunID to data row indices

building\_of\_run = pkg\["building\_of\_run"]    # Maps RunID to building ID

building\_type = pkg\["building\_type"]        # Maps building ID to type



\# Memory-mapped data location

mm\_dir = pkg\["mm\_dir"]              # Path to memory-mapped arrays

total\_rows = pkg\["total\_rows"]      # Total data rows (runs × 8760)

```



\### Running Predictions



```python

import torch



\# Prepare batch dictionary

batch = {

&nbsp;   'X\_ts': torch.tensor(time\_series\_scaled, dtype=torch.float32),  # (B, 8760, 8)

&nbsp;   'static': torch.tensor(static\_scaled, dtype=torch.float32),     # (B, 27)

}



\# Generate predictions

with torch.no\_grad():

&nbsp;   outputs = model(batch)

&nbsp;   

&nbsp;   # Get hourly heating predictions in kW

&nbsp;   Q\_pred\_kW = outputs\['Q\_pred\_kW']  # Shape: (B, 8760, 1)

&nbsp;   

&nbsp;   # Optional: Access intermediate outputs

&nbsp;   Q\_physics\_kW = outputs\['Q\_phys\_kW']  # Physics-based baseline

&nbsp;   delta = outputs\['delta']  # LSTM corrections (typically ±20%)

&nbsp;   

&nbsp;   # Get learned correction factors for each building

&nbsp;   k\_env = outputs\['k\_env']    # Envelope correction factor

&nbsp;   k\_int = outputs\['k\_int']    # Internal gains correction

&nbsp;   k\_sol = outputs\['k\_sol']    # Solar gains correction

&nbsp;   k\_vent = outputs\['k\_vent']  # Ventilation correction



\# Convert to numpy for analysis

predictions = Q\_pred\_kW.cpu().numpy()  # Shape: (B, 8760, 1)

```



\### Computing Annual Metrics



```python

import numpy as np



\# Calculate annual heating energy (MWh)

annual\_heating\_MWh = predictions.sum(axis=1) / 1000  # kWh -> MWh



\# Calculate heating intensity (kWh/m²)

heating\_intensity = (predictions.sum(axis=1).squeeze() / building\_area) \* 1000



\# Calculate peak load (kW)

peak\_load = predictions.max(axis=1)



print(f"Annual Heating: {annual\_heating\_MWh\[0]:.1f} MWh")

print(f"Heating Intensity: {heating\_intensity\[0]:.1f} kWh/m²")

print(f"Peak Load: {peak\_load\[0]:.1f} kW")

```



\## Model Configuration



The `model\_config.json` file contains all model hyperparameters:



```json

{

&nbsp; "name": "Baseline",

&nbsp; "ts\_dim": 8,

&nbsp; "static\_dim": 27,

&nbsp; "hidden\_lstm": 64,

&nbsp; "num\_layers": 1,

&nbsp; "bidirectional": false,

&nbsp; "dropout": 0.0,

&nbsp; "setpoint": 22.0,

&nbsp; "correction\_cap": 5.0,

&nbsp; "exposure\_factor": 20.0,

&nbsp; "ua\_multiplier": 1.2,

&nbsp; "schedule\_weight": 1.0

}

```



\*\*Parameter Descriptions:\*\*

\- `ts\_dim`: Number of time series input features (8)

\- `static\_dim`: Total static features including one-hot encoded categories (27)

\- `hidden\_lstm`: LSTM hidden layer size (64 for baseline)

\- `num\_layers`: Number of LSTM layers (1)

\- `bidirectional`: Whether to use bidirectional LSTM (false)

\- `dropout`: Dropout rate (0.0 for baseline)

\- `setpoint`: Heating setpoint temperature in °C (22.0)

\- `correction\_cap`: Maximum LSTM correction factor (5.0)

\- `exposure\_factor`: Infiltration exposure factor (20.0)

\- `ua\_multiplier`: Envelope UA adjustment multiplier (1.2)

\- `schedule\_weight`: Occupancy schedule weight (1.0)



\## Model Performance



\### Baseline Configuration

\- \*\*Architecture\*\*: LSTM-PGNN v2.2 Baseline

\- \*\*LSTM Hidden Size\*\*: 64

\- \*\*Number of Layers\*\*: 1

\- \*\*Bidirectional\*\*: False

\- \*\*Total Parameters\*\*: ~45,000

\- \*\*Training\*\*: AdamW optimizer, lr=3e-4, peak-weighted loss



\### Overall Test Set Performance



Performance on held-out test buildings (1,159 runs across 10 buildings):



\- \*\*RMSE\*\*: 49.74 kW

\- \*\*MAE\*\*: 28.83 kW

\- \*\*R²\*\*: 0.82



The model demonstrates strong predictive performance across diverse building types and scales, with consistent accuracy for early-stage energy planning applications.





\## Limitations



\- Trained on simulation data from Finnish climate (TRY weather files); validation recommended for other climates

\- Performance varies by building type and quality of input data

\- Requires complete hourly weather and occupancy schedule data (8760 timesteps)

\- Best suited for buildings within the training parameter ranges (see typical ranges in dataset section)

\- Assumes constant setpoint temperature (22°C); dynamic setpoints not modeled

\- Does not explicitly model thermal mass dynamics or multi-zone effects

\- Occupancy schedules must be provided as input (not predicted)



\## Model Interpretability



\### Understanding Correction Factors



The four k-factors provide physical insight into model predictions:



\- \*\*k\_env > 1\*\*: Building loses more heat than basic physics predicts (e.g., thermal bridges, air leakage)

\- \*\*k\_env < 1\*\*: Better insulation or less heat loss than expected

\- \*\*k\_int > 1\*\*: More internal gains than estimated (e.g., additional equipment)

\- \*\*k\_int < 1\*\*: Internal gains are overestimated or less effective

\- \*\*k\_sol > 1\*\*: More effective solar gains (e.g., optimal orientation)

\- \*\*k\_sol < 1\*\*: Solar gains blocked by factors not captured (e.g., shading, neighboring buildings)

\- \*\*k\_vent > 1\*\*: Ventilation load higher than expected (e.g., poor heat recovery, door openings)

\- \*\*k\_vent < 1\*\*: More efficient ventilation than assumed



\### LSTM Corrections



The δ values show where data-driven learning adds value beyond physics:

\- \*\*Small corrections (|δ| < 0.1)\*\*: Physics model is highly accurate

\- \*\*Moderate corrections (0.1 < |δ| < 0.3)\*\*: Normal adjustment for unmodeled effects

\- \*\*Large corrections (|δ| > 0.5)\*\*: May indicate missing physics, poor parameters, or unique building behavior







