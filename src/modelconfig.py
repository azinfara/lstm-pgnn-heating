import json
import torch
from src.model import LSTMPINNHeatingV2_2_Enhanced

# Load config
with open('model/model_config.json', 'r') as f:
    config = json.load(f)

# Initialize model with correct architecture
model = LSTMPINNHeatingV2_2_Enhanced(
    ts_dim=config['architecture']['ts_dim'],
    static_dim=config['architecture']['static_dim'],
    hidden_lstm=config['architecture']['hidden_lstm'],
    num_layers=config['architecture']['num_layers'],
    bidirectional=config['architecture']['bidirectional'],
    dropout=config['architecture']['dropout'],
    setpoint=config['physics_parameters']['setpoint'],
    correction_cap=config['physics_parameters']['correction_cap'],
    exposure_factor=config['physics_parameters']['exposure_factor'],
    ua_multiplier=config['physics_parameters']['ua_multiplier'],
    schedule_weight=config['physics_parameters']['schedule_weight']
)

# Load trained weights
checkpoint = torch.load('model/lstm_pgnn_v2_2_baseline.pt')
model.load_state_dict(checkpoint['state_dict'])